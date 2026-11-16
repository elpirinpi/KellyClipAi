from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
import sqlite3
import shutil
import socket
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

import yt_dlp
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from . import database
from .editor import (
    editor_preview_path,
    editor_state,
    find_clip,
    render_editor_version,
    save_editor_draft,
)
from .integrations import (
    IntegrationError,
    authorization_url,
    complete_oauth,
    download_zoom_recording,
    integration_status,
    list_zoom_recordings,
    publish_clip,
    provider_missing_credentials,
)
from .llm import LLMConfigurationError, resolve_llm
from .media import MediaProcessingError
from .pipeline import process_project
from .scheduler import schedule_worker
from .schemas import (
    ClipArchiveRequest,
    ClipEditRequest,
    ClipResolutionRequest,
    CustomTemplateRequest,
    ProcessRequest,
    ProjectConfig,
    PublishClipsRequest,
    ZoomImportRequest,
)
from .settings import AppSettings, get_settings

_ALLOWED_DOWNLOAD_HOSTS = {
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "tiktok.com",
    "instagram.com",
    "twitch.tv",
    "x.com",
    "twitter.com",
    "facebook.com",
}
_VIDEO_SUFFIXES = {".3gp", ".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _parse_config(value: str) -> ProjectConfig:
    try:
        return ProjectConfig.model_validate_json(value)
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Configuración de proyecto no válida: {error}") from error


def _serialize_project(project: dict) -> dict:
    project_id = project["id"]
    project["thumbnail_url"] = (
        f"/api/projects/{project_id}/thumbnail" if project.get("metadata") else None
    )
    project["clips"] = [
        {
            **clip,
            "video_url": f"/api/projects/{project_id}/clips/{clip['id']}/file",
            "download_url": f"/api/projects/{project_id}/clips/{clip['id']}/download",
        }
        for clip in project.get("clips") or []
    ]
    return project


def _validate_download_url(value: str) -> str:
    try:
        parsed = urlparse(value.strip())
        port = parsed.port
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail="Usa un enlace HTTPS público y válido. Solo se admiten plataformas compatibles.",
        ) from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or port not in (None, 443)
    ):
        raise HTTPException(
            status_code=422,
            detail="Usa un enlace HTTPS público y válido. Solo se admiten plataformas compatibles.",
        )
    hostname = parsed.hostname.lower().rstrip(".")
    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in _ALLOWED_DOWNLOAD_HOSTS):
        raise HTTPException(status_code=422, detail="La plataforma de este enlace no está permitida.")
    try:
        addresses = {ipaddress.ip_address(result[4][0]) for result in socket.getaddrinfo(hostname, None)}
    except OSError as error:
        raise HTTPException(status_code=422, detail="No se pudo resolver el dominio del enlace.") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise HTTPException(status_code=422, detail="El enlace debe apuntar a una dirección pública.")
    return value.strip()


def _download_video(url: str, target_dir: Path) -> tuple[Path, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "max_filesize": 2 * 1024 * 1024 * 1024,
        "socket_timeout": 20,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            expected = Path(downloader.prepare_filename(info)).with_suffix(".mp4")
            candidates = [expected, *target_dir.glob("*")]
            video = next(
                path for path in candidates if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
            )
            title = str(info.get("title") or video.stem)[:200]
            return video, title
    except yt_dlp.utils.DownloadError as error:
        detail = str(error).replace(url, "[enlace]").strip()[-300:]
        raise HTTPException(
            status_code=422,
            detail=f"yt-dlp no pudo descargar ese video autorizado: {detail or 'revisa el enlace y los permisos.'}",
        ) from error
    except (OSError, StopIteration, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"yt-dlp no pudo guardar el video: {error}",
        ) from error


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_settings.data_dir.mkdir(parents=True, exist_ok=True)
        database.initialize(app_settings.database_path)
        database.fail_interrupted_schedule_jobs(app_settings.database_path)
        scheduler_task = asyncio.create_task(schedule_worker(app_settings))
        try:
            yield
        finally:
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass

    application = FastAPI(title="kellyClipAi API", version="1.0.0", lifespan=lifespan)
    application.state.settings = app_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/settings")
    def settings_status() -> dict:
        return {
            "provider": app_settings.llm_provider,
            "model": app_settings.llm_model,
            "base_url": app_settings.llm_base_url,
            "has_llm_key": bool(app_settings.llm_api_key.strip()),
            "whisper_model": app_settings.whisper_model,
            "integrations": integration_status(app_settings, app_settings.database_path),
        }

    @application.get("/api/integrations")
    def integrations_status() -> dict:
        return integration_status(app_settings, app_settings.database_path)

    @application.get("/api/integrations/{provider}/connect")
    def connect_integration(provider: str) -> dict[str, str]:
        state = secrets.token_urlsafe(32)
        try:
            target = authorization_url(app_settings, provider, state)
        except IntegrationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        database.save_oauth_state(
            app_settings.database_path,
            state,
            provider,
            (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        )
        return {"authorization_url": target}

    @application.get("/api/integrations/{provider}/callback")
    def integration_callback(
        provider: str,
        state: str = "",
        code: str = "",
        error: str = "",
    ) -> RedirectResponse:
        saved_provider = database.consume_oauth_state(
            app_settings.database_path,
            state,
            datetime.now(timezone.utc).isoformat(),
        ) if state else None
        result = "error"
        detail = "La autorización no se pudo validar. Inicia de nuevo la conexión."
        if saved_provider == provider and not error and code:
            try:
                account = complete_oauth(app_settings, provider, code)
                database.save_connected_account(
                    app_settings.database_path,
                    account["provider"],
                    account["account_id"],
                    account["display_name"],
                    account["access_token"],
                    account.get("refresh_token"),
                    account.get("expires_at"),
                    account.get("metadata") or {},
                )
                result = "connected"
                detail = ""
            except IntegrationError as integration_error:
                detail = str(integration_error)
        elif error:
            detail = f"El proveedor rechazó la autorización: {error[:160]}."
        elif saved_provider and saved_provider != provider:
            detail = "El proveedor no coincide con la autorización iniciada."
        query = urlencode({"integration": provider, "status": result, "message": detail})
        return RedirectResponse(
            f"{app_settings.frontend_base_url}/settings?{query}",
            status_code=303,
        )

    @application.post("/api/integrations/{provider}/disconnect")
    def disconnect_integration(provider: str) -> dict[str, str]:
        if provider not in {"youtube", "tiktok", "instagram", "zoom"}:
            raise HTTPException(status_code=404, detail="No se reconoce esa integración.")
        database.disconnect_account(app_settings.database_path, provider)
        return {"status": "disconnected"}

    @application.get("/api/integrations/zoom/recordings")
    def zoom_recordings() -> list[dict]:
        try:
            return list_zoom_recordings(app_settings)
        except IntegrationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/integrations/zoom/import", status_code=201)
    async def import_zoom_recording(request: ZoomImportRequest) -> dict:
        project_id = str(uuid.uuid4())
        project_dir = app_settings.data_dir / "projects" / project_id
        input_path = project_dir / "source"
        try:
            await asyncio.to_thread(
                download_zoom_recording,
                app_settings,
                request.meeting_id,
                request.file_id,
                input_path,
            )
            database.create_project(
                app_settings.database_path,
                project_id,
                "Grabación de Zoom",
                input_path,
                request.config.model_dump(mode="json"),
            )
        except IntegrationError as error:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"No se pudo guardar la grabación: {error}") from error
        return _serialize_project(database.get_project(app_settings.database_path, project_id))

    def serialize_schedule_job(job: dict) -> dict:
        return {
            key: value
            for key, value in job.items()
            if key != "output_path"
        } | {
            "download_url": (
                f"/api/scheduled-jobs/{job['id']}/download"
                if job.get("output_path")
                else None
            )
        }

    @application.get("/api/projects/{project_id}/scheduled")
    def project_schedule(project_id: str) -> list[dict]:
        if database.get_project(app_settings.database_path, project_id) is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        return [
            serialize_schedule_job(job)
            for job in database.list_schedule_jobs(app_settings.database_path, project_id)
        ]

    @application.get("/api/scheduled-jobs")
    def scheduled_jobs() -> list[dict]:
        return [
            serialize_schedule_job(job)
            for job in database.list_schedule_jobs(app_settings.database_path)
        ]

    @application.get("/api/scheduled-jobs/{job_id}/download")
    def download_scheduled_export(job_id: str) -> FileResponse:
        job = database.get_schedule_job(app_settings.database_path, job_id)
        if job is None or not job.get("output_path"):
            raise HTTPException(status_code=404, detail="No hay un MP4 exportado para esta tarea.")
        export_root = (app_settings.data_dir / "exports").resolve()
        export_path = Path(job["output_path"]).resolve()
        if not export_path.is_relative_to(export_root) or not export_path.is_file():
            raise HTTPException(status_code=404, detail="El MP4 exportado ya no está disponible.")
        return FileResponse(
            export_path,
            media_type="video/mp4",
            filename=f"{job['clip_id']}.mp4",
        )

    @application.get("/api/templates")
    def templates() -> list[dict[str, str]]:
        return database.list_custom_templates(app_settings.database_path)

    @application.post("/api/templates", status_code=201)
    def create_template(template: CustomTemplateRequest) -> dict[str, str]:
        try:
            return database.save_custom_template(
                app_settings.database_path,
                template.name,
                template.base_template,
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=409,
                detail="Ya existe una plantilla con ese nombre.",
            ) from error

    @application.get("/api/projects")
    def projects() -> list[dict]:
        return [_serialize_project(project) for project in database.list_projects(app_settings.database_path)]

    @application.post("/api/projects", status_code=201)
    def create_uploaded_project(
        file: UploadFile = File(...),
        config: str = Form(...),
    ) -> dict:
        project_config = _parse_config(config)
        filename = Path(file.filename or "video.mp4")
        if filename.suffix.lower() not in _VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Para crear clips con IA, sube un video MP4, MOV, AVI, MKV, WebM o 3GP.",
            )
        project_id = str(uuid.uuid4())
        project_dir = app_settings.data_dir / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=False)
        input_path = project_dir / "source"
        try:
            with input_path.open("wb") as target:
                shutil.copyfileobj(file.file, target, length=1024 * 1024)
            database.create_project(
                app_settings.database_path,
                project_id,
                filename.name,
                input_path,
                project_config.model_dump(mode="json"),
            )
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        return _serialize_project(database.get_project(app_settings.database_path, project_id))

    @application.post("/api/projects/from-url", status_code=201)
    def create_link_project(
        url: str = Form(...),
        config: str = Form(...),
    ) -> dict:
        validated_url = _validate_download_url(url)
        project_config = _parse_config(config)
        project_id = str(uuid.uuid4())
        project_dir = app_settings.data_dir / "projects" / project_id
        download_dir = project_dir / "download"
        try:
            downloaded, title = _download_video(validated_url, download_dir)
            input_path = project_dir / "source"
            shutil.move(str(downloaded), input_path)
            shutil.rmtree(download_dir, ignore_errors=True)
            database.create_project(
                app_settings.database_path,
                project_id,
                title,
                input_path,
                project_config.model_dump(mode="json"),
            )
        except HTTPException:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"No se pudo guardar el video: {error}") from error
        return _serialize_project(database.get_project(app_settings.database_path, project_id))

    @application.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        return _serialize_project(project)

    @application.get("/api/projects/{project_id}/thumbnail")
    def get_thumbnail(project_id: str) -> FileResponse:
        project_dir = app_settings.data_dir / "projects" / project_id
        if database.get_project(app_settings.database_path, project_id) is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        thumbnail = project_dir / "thumbnail.jpg"
        if not thumbnail.is_file():
            raise HTTPException(status_code=404, detail="La miniatura aún no está disponible.")
        return FileResponse(thumbnail, media_type="image/jpeg")

    @application.post("/api/projects/{project_id}/process", status_code=202)
    def process(
        project_id: str,
        request: ProcessRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(status_code=409, detail="Este proyecto ya se está procesando.")
        stored_config = request.config or ProjectConfig.model_validate(project["config"])
        try:
            resolve_llm(request.llm, app_settings)
        except LLMConfigurationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if any(
            job["status"] in {"scheduled", "processing"}
            for job in database.list_schedule_jobs(app_settings.database_path, project_id)
        ):
            raise HTTPException(
                status_code=409,
                detail="Este proyecto tiene publicaciones pendientes. Espera a que se resuelvan antes de reprocesarlo.",
            )
        if stored_config.schedule and stored_config.schedule.scheduled_at <= datetime.now(timezone.utc):
            raise HTTPException(
                status_code=422,
                detail="La fecha programada ya pasó. Elige una fecha futura antes de procesar.",
            )
        if stored_config.options.broll:
            provider = app_settings.broll_provider.strip().casefold()
            if provider not in {"pexels", "pixabay"}:
                raise HTTPException(
                    status_code=422,
                    detail="B-roll requiere configurar BROLL_PROVIDER como Pexels o Pixabay en .env.",
                )
            missing = provider_missing_credentials(provider, app_settings)
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Faltan credenciales en .env para B-roll: {', '.join(missing)}.",
                )
        if request.config is not None:
            project["config"] = request.config.model_dump(mode="json")
            database.update_project(
                app_settings.database_path,
                project_id,
                config=project["config"],
            )
        if not database.queue_project(
            app_settings.database_path,
            project_id,
            project["config"],
        ):
            raise HTTPException(status_code=409, detail="Este proyecto ya se está procesando.")
        background_tasks.add_task(process_project, project_id, request.llm, app_settings)
        return {"project_id": project_id, "status": "queued"}

    @application.get("/api/projects/{project_id}/events")
    async def project_events(project_id: str, request: Request) -> EventSourceResponse:
        if database.get_project(app_settings.database_path, project_id) is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        try:
            after_id = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            after_id = 0

        async def event_stream():
            nonlocal after_id
            while True:
                for event in database.events_after(
                    app_settings.database_path, project_id, after_id
                ):
                    after_id = event["id"]
                    yield {
                        "id": str(event["id"]),
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False),
                    }
                project = database.get_project(app_settings.database_path, project_id)
                if project is None or project["status"] in {"completed", "failed"}:
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.35)

        return EventSourceResponse(event_stream(), ping=15)

    def editable_project_clip(project_id: str, clip_id: str) -> tuple[dict, dict]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        clip = find_clip(project, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        if project["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail="El editor estará disponible cuando termine el procesamiento.",
            )
        return project, clip

    @application.get("/api/projects/{project_id}/clips/{clip_id}/edit")
    def get_clip_editor(project_id: str, clip_id: str) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        return editor_state(project, clip)

    @application.put("/api/projects/{project_id}/clips/{clip_id}/edit")
    def put_clip_editor(
        project_id: str,
        clip_id: str,
        edit: ClipEditRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        try:
            return save_editor_draft(app_settings, project, clip, edit)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/projects/{project_id}/clips/{clip_id}/edit/preview")
    def preview_clip_edit(
        project_id: str,
        clip_id: str,
        edit: ClipEditRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        try:
            return render_editor_version(app_settings, project, clip, edit, preview=True)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except MediaProcessingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/api/projects/{project_id}/clips/{clip_id}/edit/preview")
    def get_clip_edit_preview(project_id: str, clip_id: str) -> FileResponse:
        editable_project_clip(project_id, clip_id)
        try:
            preview = editor_preview_path(app_settings, project_id, clip_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(preview, media_type="video/mp4")

    @application.post("/api/projects/{project_id}/clips/{clip_id}/edit/export")
    def export_clip_edit(
        project_id: str,
        clip_id: str,
        edit: ClipEditRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        try:
            return render_editor_version(app_settings, project, clip, edit, preview=False)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except MediaProcessingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def clip_file(project_id: str, clip_id: str) -> Path:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        clip = next((item for item in project.get("clips") or [] if item["id"] == clip_id), None)
        if clip is None:
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        clip_path = app_settings.data_dir / "projects" / project_id / "clips" / clip["filename"]
        if not clip_path.is_file():
            raise HTTPException(status_code=404, detail="El archivo de clip ya no está disponible.")
        return clip_path

    @application.post("/api/projects/{project_id}/publish")
    def publish_project_clips(project_id: str, request: PublishClipsRequest) -> dict:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] != "completed":
            raise HTTPException(status_code=409, detail="El proyecto aún no tiene clips listos para publicar.")
        missing = provider_missing_credentials(request.platform, app_settings)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Faltan credenciales en .env: {', '.join(missing)}.",
            )
        if not database.get_connected_account(app_settings.database_path, request.platform):
            raise HTTPException(
                status_code=409,
                detail=f"Conecta una cuenta de {request.platform} desde Ajustes antes de publicar.",
            )
        clips_by_id = {clip["id"]: clip for clip in project.get("clips") or []}
        if any(clip_id not in clips_by_id for clip_id in request.clip_ids):
            raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")
        published: list[dict[str, str]] = []
        failed: dict[str, str] | None = None
        not_attempted: list[str] = []
        for index, clip_id in enumerate(request.clip_ids):
            clip = clips_by_id[clip_id]
            try:
                result = publish_clip(
                    app_settings,
                    request.platform,
                    clip,
                    clip_file(project_id, clip_id),
                    request.visibility,
                )
            except IntegrationError as error:
                failed = {"clip_id": clip_id, "error": str(error)}
                not_attempted = request.clip_ids[index + 1 :]
                break
            published.append({"clip_id": clip_id, **result})
        return {
            "published": published,
            "failed": failed,
            "not_attempted": not_attempted,
        }

    def stored_editor_request(project: dict, clip: dict) -> ClipEditRequest:
        try:
            return ClipEditRequest.model_validate(editor_state(project, clip))
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail="Los cambios guardados del editor no se pueden volver a renderizar.",
            ) from error

    @application.post("/api/projects/{project_id}/clips/{clip_id}/quality")
    def render_clip_quality(
        project_id: str,
        clip_id: str,
        request: ClipResolutionRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        edit = stored_editor_request(project, clip)
        try:
            return render_editor_version(
                app_settings,
                project,
                clip,
                edit,
                preview=False,
                output_resolution=request.resolution,
            )
        except (ValueError, MediaProcessingError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/projects/{project_id}/clips/{clip_id}/improve")
    def improve_clip_audio(project_id: str, clip_id: str) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        edit = stored_editor_request(project, clip)
        try:
            return render_editor_version(
                app_settings,
                project,
                clip,
                edit,
                preview=False,
                enhance_audio=True,
            )
        except (ValueError, MediaProcessingError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/projects/{project_id}/clips/archive")
    def archive_clips(project_id: str, request: ClipArchiveRequest) -> FileResponse:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] != "completed":
            raise HTTPException(status_code=409, detail="El proyecto aún no tiene clips listos para descargar.")

        clips_by_id = {
            clip["id"]: clip
            for clip in project.get("clips") or []
        }
        missing_ids = [clip_id for clip_id in request.clip_ids if clip_id not in clips_by_id]
        if missing_ids:
            raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")

        with tempfile.NamedTemporaryFile(prefix="kellyclipai-", suffix=".zip", delete=False) as archive_file:
            archive_path = Path(archive_file.name)
        try:
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
                for clip_id in request.clip_ids:
                    archive.write(clip_file(project_id, clip_id), arcname=f"{clip_id}.mp4")
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"kellyClipAi-clips-{project_id[:8]}.zip",
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    @application.post("/api/projects/{project_id}/clips/{clip_id}/share", status_code=201)
    def create_clip_share(project_id: str, clip_id: str) -> dict[str, str]:
        clip_file(project_id, clip_id)
        token = database.create_or_get_clip_share(
            app_settings.database_path,
            project_id,
            clip_id,
            secrets.token_urlsafe(24),
        )
        return {"share_url": f"/share/{token}"}

    def shared_clip(token: str) -> tuple[dict, dict]:
        share = database.get_clip_share(app_settings.database_path, token)
        if share is None:
            raise HTTPException(status_code=404, detail="El enlace compartido no existe.")
        project = database.get_project(app_settings.database_path, share["project_id"])
        if project is None:
            raise HTTPException(status_code=404, detail="El proyecto compartido ya no existe.")
        clip = next(
            (item for item in project.get("clips") or [] if item["id"] == share["clip_id"]),
            None,
        )
        if clip is None:
            raise HTTPException(status_code=404, detail="El clip compartido ya no existe.")
        return project, clip

    @application.get("/api/shares/{token}")
    def get_shared_clip(token: str) -> dict:
        project, clip = shared_clip(token)
        clip_file(project["id"], clip["id"])
        return {
            "project_title": project["title"],
            "clip_id": clip["id"],
            "title": clip["title"],
            "duration": clip["duration"],
            "width": clip["width"],
            "height": clip["height"],
            "video_url": f"/api/shares/{token}/file",
        }

    @application.get("/api/shares/{token}/file")
    def stream_shared_clip(token: str) -> FileResponse:
        project, clip = shared_clip(token)
        return FileResponse(clip_file(project["id"], clip["id"]), media_type="video/mp4")

    @application.get("/api/projects/{project_id}/clips/{clip_id}/file")
    def stream_clip(project_id: str, clip_id: str) -> FileResponse:
        return FileResponse(clip_file(project_id, clip_id), media_type="video/mp4")

    @application.get("/api/projects/{project_id}/clips/{clip_id}/download")
    def download_clip(project_id: str, clip_id: str) -> FileResponse:
        return FileResponse(
            clip_file(project_id, clip_id),
            media_type="video/mp4",
            filename=f"{clip_id}.mp4",
        )

    return application


app = create_app()
