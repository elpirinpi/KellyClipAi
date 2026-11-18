from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import sqlite3
import shutil
import socket
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlencode, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

import yt_dlp
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sse_starlette.sse import EventSourceResponse
from starlette.requests import ClientDisconnect
from starlette.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.types import Scope

from . import database
from .auth import (
    AuthenticationError,
    AuthenticationUnavailable,
    SupabaseAuthVerifier,
)
from .credential_store import (
    CredentialStoreError,
    decrypt_api_key,
    encrypt_api_key,
)
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
    LLMConfig,
    LLMSettingsRequest,
    ProcessRequest,
    ProjectConfig,
    PublishClipsRequest,
    StagedDownloadRequest,
    StagedProjectRequest,
    StagedUploadRequest,
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
_AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav"}
_STAGED_SUFFIXES = _VIDEO_SUFFIXES | _AUDIO_SUFFIXES
_STAGED_WRITE_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            request_path = str(scope.get("path", "")).lstrip("/")
            if (
                error.status_code != 404
                or request_path == "api"
                or request_path.startswith("api/")
            ):
                raise
            return await super().get_response("index.html", scope)


def _parse_config(value: str) -> ProjectConfig:
    try:
        return ProjectConfig.model_validate_json(value)
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Configuración de proyecto no válida: {error}") from error


def _serialize_project(project: dict) -> dict:
    project.pop("owner_id", None)
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


def _download_video(
    url: str,
    target_dir: Path,
    progress_hook: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, str]:
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
    if progress_hook is not None:
        options["progress_hooks"] = [progress_hook]
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


def _download_staged_media(settings: AppSettings, staged_id: str, url: str) -> None:
    staging_dir = settings.data_dir / "staging" / staged_id
    last_progress_at = 0.0
    last_progress_bytes = 0

    def report_progress(data: dict[str, Any]) -> None:
        nonlocal last_progress_at, last_progress_bytes
        if data.get("status") != "downloading":
            return
        downloaded = max(0, int(data.get("downloaded_bytes") or 0))
        total_value = data.get("total_bytes") or data.get("total_bytes_estimate")
        total = max(0, int(total_value)) if total_value else None
        now = time.monotonic()
        if (
            now - last_progress_at < 0.5
            and downloaded - last_progress_bytes < 4 * _STAGED_WRITE_SIZE
        ):
            return
        speed_value = data.get("speed")
        eta_value = data.get("eta")
        database.update_staged_media(
            settings.database_path,
            staged_id,
            bytes_received=downloaded,
            total_bytes=total,
            speed=max(0.0, float(speed_value)) if speed_value is not None else None,
            eta=max(0.0, float(eta_value)) if eta_value is not None else None,
        )
        last_progress_at = now
        last_progress_bytes = downloaded

    try:
        downloaded, title = _download_video(url, staging_dir, report_progress)
        staged_path = settings.data_dir / "staging" / f"{staged_id}.source"
        shutil.move(str(downloaded), staged_path)
        shutil.rmtree(staging_dir, ignore_errors=True)
        file_size = staged_path.stat().st_size
        database.update_staged_media(
            settings.database_path,
            staged_id,
            status="ready",
            filename=staged_path.name,
            title=title,
            file_size=file_size,
            bytes_received=file_size,
            total_bytes=file_size,
            speed=None,
            eta=None,
            error=None,
        )
    except Exception as error:
        logger.exception("No se pudo completar la descarga staged %s", staged_id)
        detail = error.detail if isinstance(error, HTTPException) else str(error)
        database.update_staged_media(
            settings.database_path,
            staged_id,
            status="failed",
            error=f"No se pudo descargar el video: {detail}",
            speed=None,
            eta=None,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)


def _serialize_staged_media(staged: dict[str, Any]) -> dict[str, Any]:
    return {
        key: staged[key]
        for key in (
            "id",
            "source_type",
            "status",
            "filename",
            "title",
            "file_size",
            "duration",
            "width",
            "height",
            "kind",
            "bytes_received",
            "total_bytes",
            "speed",
            "eta",
            "error",
            "project_id",
            "created_at",
            "updated_at",
        )
    }


def create_app(
    settings: AppSettings | None = None,
    web_dir: Path | None = None,
    *,
    auth_required: bool = True,
    auth_verifier: Any | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    verifier = auth_verifier
    if (
        verifier is None
        and app_settings.supabase_url
        and app_settings.supabase_anon_key
    ):
        verifier = SupabaseAuthVerifier(
            app_settings.supabase_url,
            app_settings.supabase_anon_key,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_settings.data_dir.mkdir(parents=True, exist_ok=True)
        database.initialize(app_settings.database_path)
        database.fail_interrupted_staged_media(app_settings.database_path)
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
            if isinstance(verifier, SupabaseAuthVerifier):
                await verifier.close()

    application = FastAPI(title="kellyClipAi API", version="1.0.3", lifespan=lifespan)
    application.state.settings = app_settings
    application.state.auth_required = auth_required
    application.state.auth_verifier = verifier
    application.state.claimed_legacy_users = set()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["DELETE", "GET", "POST", "PUT"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
    )

    @application.middleware("http")
    async def authenticate_api_request(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/") or request.method == "OPTIONS":
            return await call_next(request)
        if (
            path in {"/api/health", "/api/auth/config"}
            or path.startswith("/api/shares/")
            or (
                path.startswith("/api/integrations/")
                and path.endswith("/callback")
            )
        ):
            return await call_next(request)

        user_id: str
        if not application.state.auth_required:
            user_id = "test-user"
        else:
            authorization = request.headers.get("authorization", "")
            token = (
                authorization[7:].strip()
                if authorization.lower().startswith("bearer ")
                else ""
            )
            if not token and _is_media_request(request):
                token = request.cookies.get("kellyclip_media", "")
            if not token:
                return JSONResponse(
                    {"detail": "Inicia sesión para continuar."},
                    status_code=401,
                )
            current_verifier = application.state.auth_verifier
            if current_verifier is None:
                return JSONResponse(
                    {
                        "detail": (
                            "Supabase no está configurado. Añade SUPABASE_URL y "
                            "SUPABASE_ANON_KEY al archivo .env."
                        )
                    },
                    status_code=503,
                )
            try:
                authenticated = await current_verifier.verify(token)
            except AuthenticationUnavailable as error:
                return JSONResponse({"detail": str(error)}, status_code=503)
            except AuthenticationError as error:
                return JSONResponse({"detail": str(error)}, status_code=401)
            user_id = authenticated.id

        request.state.user_id = user_id
        if user_id not in application.state.claimed_legacy_users:
            database.claim_legacy_data(app_settings.database_path, user_id)
            application.state.claimed_legacy_users.add(user_id)

        path_parts = path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[1] == "projects":
            project_id = path_parts[2]
            if project_id != "from-url" and database.get_project(
                app_settings.database_path, project_id, user_id
            ) is None:
                return JSONResponse(
                    {"detail": "No se encontró el proyecto."},
                    status_code=404,
                )
        elif (
            len(path_parts) >= 3
            and path_parts[1] == "staged-media"
            and path_parts[2] not in {"latest", "uploads", "downloads"}
            and database.get_staged_media(
                app_settings.database_path, path_parts[2], user_id
            )
            is None
        ):
            return JSONResponse(
                {"detail": "No se encontró la importación."},
                status_code=404,
            )
        elif len(path_parts) >= 3 and path_parts[1] == "scheduled-jobs":
            if database.get_schedule_job(
                app_settings.database_path, path_parts[2], user_id
            ) is None:
                return JSONResponse(
                    {"detail": "No se encontró la tarea programada."},
                    status_code=404,
                )
        return await call_next(request)

    def request_user_id(request: Request) -> str:
        user_id = getattr(request.state, "user_id", None)
        if not isinstance(user_id, str):
            raise HTTPException(status_code=401, detail="Inicia sesión para continuar.")
        return user_id

    def _is_media_request(request: Request) -> bool:
        if request.method != "GET":
            return False
        path = request.url.path
        return (
            path.endswith(("/file", "/download", "/thumbnail", "/edit/preview"))
            or (
                path.startswith("/api/scheduled-jobs/")
                and path.endswith("/download")
            )
        )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/auth/config")
    def auth_config() -> dict[str, str | bool]:
        configured = bool(app_settings.supabase_url and app_settings.supabase_anon_key)
        return {
            "configured": configured,
            "supabase_url": app_settings.supabase_url if configured else "",
            "supabase_anon_key": app_settings.supabase_anon_key if configured else "",
        }

    @application.post("/api/auth/media-cookie", status_code=204)
    def set_media_cookie(request: Request, response: Response) -> Response:
        authorization = request.headers.get("authorization", "")
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not token:
            raise HTTPException(status_code=401, detail="La sesión no está disponible.")
        response.set_cookie(
            "kellyclip_media",
            token,
            max_age=3600,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            path="/api",
        )
        response.status_code = 204
        return response

    @application.delete("/api/auth/media-cookie", status_code=204)
    def clear_media_cookie() -> Response:
        response = Response(status_code=204)
        response.delete_cookie("kellyclip_media", path="/api", samesite="lax")
        return response

    @application.get("/api/settings")
    def settings_status(request: Request) -> dict:
        user_id = request_user_id(request)
        credentials = database.get_llm_credentials(app_settings.database_path, user_id)
        return {
            "provider": credentials["provider"] if credentials else app_settings.llm_provider,
            "model": credentials["model"] if credentials else app_settings.llm_model,
            "base_url": credentials["base_url"] if credentials else app_settings.llm_base_url,
            "has_llm_key": credentials is not None or bool(app_settings.llm_api_key),
            "has_saved_llm_key": credentials is not None,
            "has_env_llm_key": bool(app_settings.llm_api_key),
            "whisper_model": app_settings.whisper_model,
            "integrations": integration_status(
                app_settings,
                app_settings.database_path,
                user_id,
            ),
        }

    @application.put("/api/settings/llm-key")
    def save_llm_settings(request: Request, payload: LLMSettingsRequest) -> dict[str, Any]:
        user_id = request_user_id(request)
        current = database.get_llm_credentials(app_settings.database_path, user_id)
        api_key = payload.api_key.strip()
        keep_current_key = bool(
            not api_key and current is not None and current["provider"] == payload.provider
        )
        copy_environment_key = bool(
            not api_key
            and current is None
            and app_settings.llm_api_key
            and app_settings.llm_provider == payload.provider
        )
        if not api_key and not keep_current_key and not copy_environment_key:
            raise HTTPException(
                status_code=422,
                detail="Introduce tu clave API para guardarla de forma segura.",
            )
        try:
            if api_key:
                encrypted = encrypt_api_key(api_key)
            elif keep_current_key and current is not None:
                encrypted = bytes(current["encrypted_api_key"])
            else:
                encrypted = encrypt_api_key(app_settings.llm_api_key)
            database.save_llm_credentials(
                app_settings.database_path,
                user_id,
                payload.provider,
                encrypted,
                payload.model.strip(),
                payload.base_url.strip(),
            )
        except CredentialStoreError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return {"saved": True, "has_api_key": True}

    @application.delete("/api/settings/llm-key", status_code=204)
    def delete_llm_settings(request: Request) -> Response:
        database.delete_llm_credentials(
            app_settings.database_path,
            request_user_id(request),
        )
        return Response(status_code=204)

    @application.get("/api/integrations")
    def integrations_status(request: Request) -> dict:
        return integration_status(
            app_settings,
            app_settings.database_path,
            request_user_id(request),
        )

    @application.get("/api/integrations/{provider}/connect")
    def connect_integration(provider: str, request: Request) -> dict[str, str]:
        owner_id = request_user_id(request)
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
            owner_id,
        )
        return {"authorization_url": target}

    @application.get("/api/integrations/{provider}/callback")
    def integration_callback(
        provider: str,
        state: str = "",
        code: str = "",
        error: str = "",
    ) -> RedirectResponse:
        saved_state = database.consume_oauth_state_record(
            app_settings.database_path,
            state,
            datetime.now(timezone.utc).isoformat(),
        ) if state else None
        result = "error"
        detail = "La autorización no se pudo validar. Inicia de nuevo la conexión."
        if saved_state and saved_state["provider"] == provider and not error and code:
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
                    saved_state["owner_id"],
                )
                result = "connected"
                detail = ""
            except IntegrationError as integration_error:
                detail = str(integration_error)
        elif error:
            detail = f"El proveedor rechazó la autorización: {error[:160]}."
        elif saved_state and saved_state["provider"] != provider:
            detail = "El proveedor no coincide con la autorización iniciada."
        query = urlencode({"integration": provider, "status": result, "message": detail})
        return RedirectResponse(
            f"{app_settings.frontend_base_url}/settings?{query}",
            status_code=303,
        )

    @application.post("/api/integrations/{provider}/disconnect")
    def disconnect_integration(provider: str, request: Request) -> dict[str, str]:
        if provider not in {"youtube", "tiktok", "instagram", "zoom"}:
            raise HTTPException(status_code=404, detail="No se reconoce esa integración.")
        database.disconnect_account(
            app_settings.database_path,
            provider,
            request_user_id(request),
        )
        return {"status": "disconnected"}

    @application.get("/api/integrations/zoom/recordings")
    def zoom_recordings(request: Request) -> list[dict]:
        try:
            return list_zoom_recordings(
                app_settings,
                request_user_id(request),
            )
        except IntegrationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/integrations/zoom/import", status_code=201)
    async def import_zoom_recording(payload: ZoomImportRequest, request: Request) -> dict:
        owner_id = request_user_id(request)
        project_id = str(uuid.uuid4())
        project_dir = app_settings.data_dir / "projects" / project_id
        input_path = project_dir / "source"
        try:
            await asyncio.to_thread(
                download_zoom_recording,
                app_settings,
                payload.meeting_id,
                payload.file_id,
                input_path,
                owner_id,
            )
            database.create_project(
                app_settings.database_path,
                project_id,
                "Grabación de Zoom",
                input_path,
                payload.config.model_dump(mode="json"),
                owner_id,
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
            if key not in {"output_path", "owner_id"}
        } | {
            "download_url": (
                f"/api/scheduled-jobs/{job['id']}/download"
                if job.get("output_path")
                else None
            )
        }

    @application.get("/api/projects/{project_id}/scheduled")
    def project_schedule(project_id: str, request: Request) -> list[dict]:
        if database.get_project(app_settings.database_path, project_id) is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        return [
            serialize_schedule_job(job)
            for job in database.list_schedule_jobs(
                app_settings.database_path,
                project_id,
                request_user_id(request),
            )
        ]

    @application.get("/api/scheduled-jobs")
    def scheduled_jobs(request: Request) -> list[dict]:
        return [
            serialize_schedule_job(job)
            for job in database.list_schedule_jobs(
                app_settings.database_path,
                owner_id=request_user_id(request),
            )
        ]

    @application.get("/api/scheduled-jobs/{job_id}/download")
    def download_scheduled_export(job_id: str, request: Request) -> FileResponse:
        job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
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
    def templates(request: Request) -> list[dict[str, str]]:
        return database.list_custom_templates(
            app_settings.database_path,
            request_user_id(request),
        )

    @application.post("/api/templates", status_code=201)
    def create_template(template: CustomTemplateRequest, request: Request) -> dict[str, str]:
        try:
            return database.save_custom_template(
                app_settings.database_path,
                template.name,
                template.base_template,
                request_user_id(request),
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=409,
                detail="Ya existe una plantilla con ese nombre.",
            ) from error

    @application.get("/api/projects")
    def projects(request: Request) -> list[dict]:
        return [
            _serialize_project(project)
            for project in database.list_projects(
                app_settings.database_path,
                request_user_id(request),
            )
        ]

    @application.get("/api/staged-media/latest")
    def latest_staged_media(
        source: Literal["upload", "url"],
        request: Request,
    ) -> dict | None:
        staged = database.latest_staged_media(
            app_settings.database_path,
            source,
            request_user_id(request),
        )
        return _serialize_staged_media(staged) if staged else None

    @application.post("/api/staged-media/uploads", status_code=201)
    def create_staged_upload(payload: StagedUploadRequest, request: Request) -> dict:
        owner_id = request_user_id(request)
        filename = Path(payload.filename.replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in _STAGED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Formato no compatible. Usa un formato de video o audio admitido.",
            )
        staged_id = str(uuid.uuid4())
        (app_settings.data_dir / "staging").mkdir(parents=True, exist_ok=True)
        database.create_staged_media(
            app_settings.database_path,
            staged_id,
            "upload",
            "receiving",
            filename,
            file_size=payload.file_size,
            duration=payload.duration,
            width=payload.width,
            height=payload.height,
            kind=payload.kind,
            owner_id=owner_id,
        )
        return _serialize_staged_media(
            database.get_staged_media(app_settings.database_path, staged_id)
        )

    @application.post("/api/staged-media/{staged_id}/content")
    async def receive_staged_upload(staged_id: str, request: Request) -> dict:
        staged = database.get_staged_media(app_settings.database_path, staged_id)
        if staged is None or staged["source_type"] != "upload":
            raise HTTPException(status_code=404, detail="No se encontró la carga local.")
        if staged["status"] != "receiving":
            raise HTTPException(status_code=409, detail="La carga local ya no está disponible.")

        staging_root = app_settings.data_dir / "staging"
        partial_path = staging_root / f"{staged_id}.part"
        staged_path = staging_root / f"{staged_id}.source"
        received = 0
        last_saved = 0
        try:
            with partial_path.open("wb") as target:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > staged["file_size"]:
                        raise HTTPException(
                            status_code=413,
                            detail="La carga supera el tamaño del archivo seleccionado.",
                        )
                    for offset in range(0, len(chunk), _STAGED_WRITE_SIZE):
                        block = memoryview(chunk)[offset : offset + _STAGED_WRITE_SIZE]
                        await asyncio.to_thread(target.write, block)
                    if received - last_saved >= 4 * _STAGED_WRITE_SIZE:
                        database.update_staged_media(
                            app_settings.database_path,
                            staged_id,
                            bytes_received=received,
                        )
                        last_saved = received
                await asyncio.to_thread(target.flush)
            if received != staged["file_size"]:
                raise HTTPException(
                    status_code=400,
                    detail="La carga se interrumpió antes de recibir el archivo completo.",
                )
            partial_path.replace(staged_path)
        except HTTPException as error:
            partial_path.unlink(missing_ok=True)
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="failed",
                error=str(error.detail),
            )
            raise
        except ClientDisconnect as error:
            partial_path.unlink(missing_ok=True)
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="failed",
                error="La conexión se interrumpió durante la carga. Vuelve a seleccionarlo.",
            )
            raise HTTPException(
                status_code=400,
                detail="La conexión se interrumpió durante la carga. Vuelve a seleccionarlo.",
            ) from error
        except OSError as error:
            partial_path.unlink(missing_ok=True)
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="failed",
                error=f"No se pudo guardar el archivo en disco: {error}",
            )
            raise HTTPException(
                status_code=507,
                detail=f"No se pudo guardar el archivo en disco. Revisa el espacio disponible: {error}",
            ) from error

        database.update_staged_media(
            app_settings.database_path,
            staged_id,
            status="ready",
            bytes_received=received,
            total_bytes=received,
            speed=None,
            eta=None,
            error=None,
        )
        return _serialize_staged_media(
            database.get_staged_media(app_settings.database_path, staged_id)
        )

    @application.post("/api/staged-media/downloads", status_code=202)
    def create_staged_download(
        payload: StagedDownloadRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> dict:
        owner_id = request_user_id(request)
        validated_url = _validate_download_url(payload.url)
        staged_id = str(uuid.uuid4())
        (app_settings.data_dir / "staging").mkdir(parents=True, exist_ok=True)
        database.create_staged_media(
            app_settings.database_path,
            staged_id,
            "url",
            "downloading",
            "Descarga de enlace",
            config=payload.config.model_dump(mode="json"),
            owner_id=owner_id,
        )
        background_tasks.add_task(
            _download_staged_media,
            app_settings,
            staged_id,
            validated_url,
        )
        return _serialize_staged_media(
            database.get_staged_media(app_settings.database_path, staged_id)
        )

    @application.get("/api/staged-media/{staged_id}")
    def get_staged_media(staged_id: str) -> dict:
        staged = database.get_staged_media(app_settings.database_path, staged_id)
        if staged is None:
            raise HTTPException(status_code=404, detail="No se encontró la importación.")
        return _serialize_staged_media(staged)

    @application.delete("/api/staged-media/{staged_id}", status_code=204)
    def delete_staged_media(staged_id: str) -> Response:
        staged = database.get_staged_media(app_settings.database_path, staged_id)
        if staged is None:
            raise HTTPException(status_code=404, detail="No se encontró la importación.")
        if not database.claim_staged_media_for_deletion(
            app_settings.database_path, staged_id
        ):
            raise HTTPException(status_code=409, detail="No se puede cancelar esta importación ahora.")
        try:
            if staged["source_type"] == "upload":
                for suffix in (".part", ".source"):
                    (app_settings.data_dir / "staging" / f"{staged_id}{suffix}").unlink(missing_ok=True)
            else:
                shutil.rmtree(app_settings.data_dir / "staging" / staged_id, ignore_errors=True)
                (app_settings.data_dir / "staging" / f"{staged_id}.source").unlink(missing_ok=True)
        except OSError as error:
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="failed",
                error=f"No se pudo limpiar el archivo temporal: {error}",
            )
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo limpiar el archivo temporal: {error}",
            ) from error
        if not database.delete_staged_media(app_settings.database_path, staged_id):
            raise HTTPException(status_code=409, detail="La importación cambió mientras se cancelaba.")
        return Response(status_code=204)

    @application.post("/api/staged-media/{staged_id}/project", status_code=201)
    def create_project_from_staged_media(
        staged_id: str,
        request: StagedProjectRequest,
    ) -> dict:
        staged = database.get_staged_media(app_settings.database_path, staged_id)
        if staged is None:
            raise HTTPException(status_code=404, detail="No se encontró la importación.")
        if staged["status"] == "completed" and staged["project_id"]:
            project = database.get_project(app_settings.database_path, staged["project_id"])
            if project:
                return _serialize_project(project)
        if staged["status"] != "ready":
            raise HTTPException(status_code=409, detail="El archivo aún no está listo para crear el proyecto.")
        config = request.config.model_dump(mode="json") if request.config else staged["config"]
        if config is None:
            raise HTTPException(status_code=422, detail="Falta la configuración del proyecto.")
        if staged["source_type"] == "upload" and Path(staged["filename"]).suffix.lower() not in _VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Para crear clips con IA, selecciona un archivo de video compatible.",
            )
        staged_path = app_settings.data_dir / "staging" / f"{staged_id}.source"
        if not staged_path.is_file():
            raise HTTPException(status_code=404, detail="El archivo guardado ya no está disponible.")
        if not database.claim_staged_media(app_settings.database_path, staged_id):
            raise HTTPException(status_code=409, detail="La importación ya está creando un proyecto.")

        project_id = str(uuid.uuid4())
        project_dir = app_settings.data_dir / "projects" / project_id
        input_path = project_dir / "source"
        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            shutil.move(str(staged_path), input_path)
            database.create_project(
                app_settings.database_path,
                project_id,
                staged["title"] or staged["filename"],
                input_path,
                config,
                staged["owner_id"],
            )
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="completed",
                project_id=project_id,
                error=None,
            )
        except Exception:
            if input_path.is_file():
                shutil.move(str(input_path), staged_path)
            shutil.rmtree(project_dir, ignore_errors=True)
            database.update_staged_media(
                app_settings.database_path,
                staged_id,
                status="ready",
            )
            raise
        return _serialize_project(database.get_project(app_settings.database_path, project_id))

    @application.post("/api/projects", status_code=201)
    def create_uploaded_project(
        request: Request,
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
                request_user_id(request),
            )
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        return _serialize_project(database.get_project(app_settings.database_path, project_id))

    @application.post("/api/projects/from-url", status_code=201)
    def create_link_project(
        request: Request,
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
                request_user_id(request),
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
        credentials = database.get_llm_credentials(
            app_settings.database_path,
            project["owner_id"],
        )
        if credentials is None and not app_settings.llm_api_key:
            raise HTTPException(
                status_code=422,
                detail="Falta la clave API. Guárdala en Ajustes antes de procesar el video.",
            )
        try:
            llm_config = (
                LLMConfig(
                    provider=credentials["provider"],
                    api_key=decrypt_api_key(bytes(credentials["encrypted_api_key"])),
                    model=credentials["model"],
                    base_url=credentials["base_url"],
                )
                if credentials is not None
                else LLMConfig(
                    provider=app_settings.llm_provider,
                    api_key=app_settings.llm_api_key,
                    model=app_settings.llm_model,
                    base_url=app_settings.llm_base_url,
                )
            )
            resolve_llm(llm_config, app_settings)
        except CredentialStoreError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except LLMConfigurationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if any(
            job["status"] in {"scheduled", "processing"}
            for job in database.list_schedule_jobs(
                app_settings.database_path,
                project_id,
                project["owner_id"],
            )
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
        background_tasks.add_task(process_project, project_id, llm_config, app_settings)
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
        if not database.get_connected_account(
            app_settings.database_path,
            request.platform,
            project["owner_id"],
        ):
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
                    project["owner_id"],
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

    if web_dir is not None:
        application.mount("/", SPAStaticFiles(directory=web_dir, html=True), name="frontend")

    return application


app = create_app()
