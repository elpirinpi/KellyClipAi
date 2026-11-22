from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlencode, urlparse
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import yt_dlp
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from . import (
    brand,
    captions,
    clip_versions,
    database,
    moments,
    project_clone,
    publish_guide,
    translate,
    watch_folder,
    waveform,
)
from .auth import (
    AuthenticationError,
    AuthenticationUnavailable,
    SupabaseAuthVerifier,
)
from .autostart import (
    AutostartError,
    autostart_supported,
    is_autostart_enabled,
    set_autostart,
)
from .credential_store import (
    CredentialStoreError,
    decrypt_api_key,
    encrypt_api_key,
)
from .desktop_prefs import load_desktop_prefs, save_desktop_prefs, tray_available
from .editor import (
    EditorRenderJobManager,
    editor_preview_path,
    editor_state,
    find_clip,
    render_editor_version,
    save_editor_draft,
)
from .face_tracking import (
    STUDIO_FACE_TRACK_SAMPLE_RATE,
    STUDIO_FACE_TRACK_SIZES,
    FaceCropKeyframe,
    analyze_source_face_track,
    forget_face_track_job,
    peek_cached_source_face_track,
    start_face_track_job,
)
from .first_run import (
    WHISPER_MODELS,
    disk_space,
    downloaded_models,
    is_setup_complete,
    mark_setup_complete,
    needs_setup,
    whisper_model_from_setup,
)
from .integrations import (
    IntegrationError,
    authorization_url,
    complete_oauth,
    download_zoom_recording,
    integration_status,
    list_account_summaries,
    list_zoom_recordings,
    provider_missing_credentials,
    search_broll_videos,
)
from .llm import LLMConfigurationError, LLMResponseError, resolve_llm
from .media import (
    MediaProcessingError,
    create_frame_thumbnail,
    format_subtitles,
    get_encoder_info,
    probe_media,
)
from .pipeline import _resolve_llm_for_project, process_project
from .process_queue import ProjectProcessingQueue, request_project_cancellation
from .scheduler import schedule_worker
from .schemas import (
    DEFAULT_PROJECTS_PAGE_SIZE,
    MAX_PROJECTS_PAGE_SIZE,
    MAX_UPLOAD_BYTES,
    ApplySavedMomentRequest,
    AutostartRequest,
    BatchEnqueueRequest,
    BrandKitRequest,
    BrollAssignRequest,
    BrollImportRequest,
    BrollSearchRequest,
    BrollUpdateRequest,
    CaptionApplyRequest,
    CaptionGenerateRequest,
    ClipArchiveRequest,
    ClipEditRequest,
    ClipProposal,
    ClipResolutionRequest,
    CrossPublishRequest,
    CustomTemplateRequest,
    DesktopPreferencesRequest,
    DuplicateProjectRequest,
    EditorRenderJobResponse,
    FaceTrackKeyframe,
    FaceTrackPendingResponse,
    FaceTrackResponse,
    LLMConfig,
    LLMSettingsRequest,
    NightWindowRequest,
    OverdueScheduleResolution,
    ProcessRequest,
    ProjectConfig,
    PublishClipsRequest,
    RescheduleRequest,
    SavedMomentRequest,
    SavedMomentUpdateRequest,
    SaveProjectFilesRequest,
    SetupCompleteRequest,
    StagedDownloadRequest,
    StagedProjectRequest,
    StagedUploadRequest,
    StudioEditRequest,
    TranscriptSegment,
    TranslateSubtitlesRequest,
    UpdateChannelRequest,
    WatchFolderRequest,
    WatchFolderUpdateRequest,
    ZoomImportRequest,
)
from .settings import AppSettings, get_settings
from .studio import (
    clip_transcript_words,
    local_transcript_words,
    project_duration,
    project_transcript_words,
    words_to_cues,
)
from .updater import UpdateError, UpdateManager
from .version import APP_VERSION

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
# Extensión del archivo de un clip manual creado desde el editor visual,
# según el tipo de contenido que envía el navegador (MediaRecorder suele
# entregar video/webm). Los tipos video/* sin entrada en la tabla usan
# .mp4; el reemplazo de un clip existente conserva su nombre de archivo.
_STUDIO_EXPORT_SUFFIXES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/avi": ".avi",
    "video/3gpp": ".3gp",
}
_STAGED_WRITE_SIZE = 1024 * 1024
_STAGING_MAX_AGE = timedelta(hours=24)
logger = logging.getLogger(__name__)


class _StagedDownloadCancelled(RuntimeError):
    """El usuario canceló una descarga staged mientras estaba en curso.

    Se lanza desde el hook de progreso de yt-dlp (que propaga las
    excepciones sin atraparlas) para abortar la descarga; la tarea la
    captura y limpia los archivos parciales sin marcar el staged como
    fallido. No hereda de OSError/ValueError para que _download_video
    no la convierta en HTTPException 422.
    """


class _ChunkTooLargeError(RuntimeError):
    """El cuerpo supera el rango declarado o el tamaño del archivo."""


# --- Locks por staged_id para la subida reanudable (Fase 2, B4/B-F11) -------
# Dos POST /content concurrentes al mismo staged_id se truncaban
# mutuamente (ambos abrían el .part con "wb"). Cada staged tiene su lock
# asyncio: las subidas de distintos staged siguen en paralelo, pero las
# del mismo staged (y su borrado) se serializan. Los locks se conservan
# (son pequeños y su número está acotado por el número de staged creados):
# borrarlos al soltarlos abriría una carrera entre quien suelta y quien
# está a punto de adquirir.
_STAGED_UPLOAD_LOCKS: dict[str, asyncio.Lock] = {}
_STAGED_UPLOAD_LOCKS_GUARD = threading.Lock()


def _staged_upload_lock(staged_id: str) -> asyncio.Lock:
    with _STAGED_UPLOAD_LOCKS_GUARD:
        lock = _STAGED_UPLOAD_LOCKS.get(staged_id)
        if lock is None:
            lock = asyncio.Lock()
            _STAGED_UPLOAD_LOCKS[staged_id] = lock
        return lock


_CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    """Parsea `Content-Range: bytes <inicio>-<fin>/<total>`.

    Devuelve (inicio, fin, total) o None si no hay cabecera. Lanza
    ValueError si la cabecera está presente pero mal formada o el rango
    es incoherente (fin < inicio, fin >= total, total <= 0).
    """
    if value is None or not value.strip():
        return None
    match = _CONTENT_RANGE_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(
            "La cabecera Content-Range debe tener el formato "
            "'bytes <inicio>-<fin>/<total>'."
        )
    start, end, total = (int(part) for part in match.groups())
    if end < start or total <= 0 or end >= total:
        raise ValueError("El rango de la cabecera Content-Range no es válido.")
    return start, end, total


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


def _serialize_project(
    project: dict,
    *,
    include_transcript: bool = False,
    include_clip_transcripts: bool = True,
) -> dict:
    project.pop("owner_id", None)
    if not include_transcript:
        project.pop("transcript", None)
    project_id = project["id"]
    project["thumbnail_url"] = (
        f"/api/projects/{project_id}/thumbnail" if project.get("metadata") else None
    )
    serialized_clips = []
    for clip in project.get("clips") or []:
        serialized = {
            **clip,
            "video_url": (
                f"/api/projects/{project_id}/clips/{clip['id']}/file"
                if clip.get("status") != "failed"
                else None
            ),
            "download_url": (
                f"/api/projects/{project_id}/clips/{clip['id']}/download"
                if clip.get("status") != "failed"
                else None
            ),
        }
        if not include_clip_transcripts:
            # B-R10: en el listado, cada clip traía su transcripción
            # palabra por palabra (kilobytes por clip). El detalle de
            # proyecto y el editor sí la necesitan; el listado, no.
            serialized.pop("transcript", None)
        serialized_clips.append(serialized)
    project["clips"] = serialized_clips
    return project


def _serialize_clip(project_id: str, clip: dict) -> dict:
    """Un clip con la misma forma que en el detalle de proyecto."""
    return {
        **clip,
        "video_url": (
            f"/api/projects/{project_id}/clips/{clip['id']}/file"
            if clip.get("status") != "failed"
            else None
        ),
        "download_url": (
            f"/api/projects/{project_id}/clips/{clip['id']}/download"
            if clip.get("status") != "failed"
            else None
        ),
    }


_CLIENT_ERROR_MAX_CHARS = 4000


def _client_error_text(value: object) -> str | None:
    """Un campo del reporte de error de la interfaz como texto corto.

    Tolerante: cualquier valor no nulo se convierte a texto y se
    recorta a 4000 caracteres, para que un reporte malformado nunca
    tumbe el endpoint ni inunde el log. Devuelve None si está vacío.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return None
    if len(text) > _CLIENT_ERROR_MAX_CHARS:
        text = text[:_CLIENT_ERROR_MAX_CHARS] + "… (truncado)"
    return text


def _probe_clip_dimensions(path: Path) -> tuple[int, int] | None:
    """Dimensiones del video exportado, o None si no se pueden leer.

    El export del editor no debe fallar porque la sonda no pueda leer
    el archivo: en ese caso el clip conserva sus dimensiones anteriores
    (reemplazo) o queda en 0x0 (clip manual nuevo).
    """
    try:
        metadata = probe_media(path, shutil.which("ffprobe") or "ffprobe")
        return int(metadata["width"]), int(metadata["height"])
    except (KeyError, MediaProcessingError, OSError, TypeError, ValueError):
        return None


def _probe_source_duration(path: Path) -> float | None:
    """Duración del video fuente según el archivo, o None si no se lee.

    La sonda exige pista de audio (como el pipeline); un video sin
    audio o ilegible devuelve None y el llamador usa la duración
    guardada en el proyecto.
    """
    try:
        metadata = probe_media(path, shutil.which("ffprobe") or "ffprobe")
        duration = float(metadata["duration"])
    except (KeyError, MediaProcessingError, OSError, TypeError, ValueError):
        return None
    return duration if duration > 0 else None


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


def _staged_download_cancelled(settings: AppSettings, staged_id: str) -> bool:
    """El staged fue cancelado/borrado mientras se descargaba.

    El borrado marca el staged como 'deleting' y luego elimina la fila;
    en ambos casos la descarga debe abortar y limpiar en vez de
    completar el staged.
    """
    staged = database.get_staged_media(settings.database_path, staged_id)
    return staged is None or staged["status"] == "deleting"


def _cleanup_staged_download(settings: AppSettings, staged_id: str) -> None:
    staging_dir = settings.data_dir / "staging" / staged_id
    shutil.rmtree(staging_dir, ignore_errors=True)
    (settings.data_dir / "staging" / f"{staged_id}.source").unlink(missing_ok=True)


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
        # Aborto temprano: si el usuario canceló la descarga, yt-dlp
        # propaga esta excepción y la tarea limpia los parciales. Si
        # algún día yt-dlp la tragara, el chequeo posterior a la descarga
        # seguiría impidiendo que el staged se completara.
        if _staged_download_cancelled(settings, staged_id):
            raise _StagedDownloadCancelled(
                f"Descarga staged {staged_id} cancelada por el usuario."
            )
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
        if _staged_download_cancelled(settings, staged_id):
            # El usuario canceló mientras descargaba (o el hook no llegó
            # a abortar a tiempo): limpiar sin completar el staged.
            logger.info(
                "Descarga staged %s cancelada; se descartan los archivos.",
                staged_id,
            )
            _cleanup_staged_download(settings, staged_id)
            return
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
    except _StagedDownloadCancelled:
        logger.info("Descarga staged %s cancelada por el usuario.", staged_id)
        _cleanup_staged_download(settings, staged_id)
    except Exception as error:
        if _staged_download_cancelled(settings, staged_id):
            # La descarga falló a la vez que el usuario la cancelaba:
            # limpiar sin marcarla como fallida (la fila ya no existe o
            # está en 'deleting' y el borrado la eliminará).
            _cleanup_staged_download(settings, staged_id)
            return
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


def _remove_staging_artifacts(settings: AppSettings, staged_ids: list[str]) -> None:
    staging_root = (settings.data_dir / "staging").resolve()
    for staged_id in staged_ids:
        for candidate in (
            staging_root / staged_id,
            staging_root / f"{staged_id}.source",
        ):
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink():
                candidate.unlink()
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(staging_root):
                logger.warning("Se omitió una ruta de staging fuera de la carpeta local.")
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()


def _cleanup_expired_staging(settings: AppSettings) -> None:
    cutoff = datetime.now(UTC) - _STAGING_MAX_AGE
    expired_ids = database.cleanup_expired_staged_media(
        settings.database_path, cutoff.isoformat()
    )
    staging_root = settings.data_dir / "staging"
    try:
        _remove_staging_artifacts(settings, expired_ids)
        if not staging_root.is_dir():
            return
        for entry in staging_root.iterdir():
            try:
                if entry.is_symlink():
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff.timestamp():
                        entry.unlink()
                    continue
                if entry.is_dir():
                    file_times = [
                        path.stat().st_mtime
                        for path in entry.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    ]
                    newest = max(file_times, default=entry.stat().st_mtime)
                    if newest < cutoff.timestamp():
                        shutil.rmtree(entry)
                elif entry.is_file() and entry.stat().st_mtime < cutoff.timestamp():
                    entry.unlink()
            except OSError:
                logger.exception("No se pudo limpiar el archivo de staging %s", entry)
    except OSError:
        logger.exception("No se pudo limpiar staging caducado")


def _local_storage_bytes(settings: AppSettings) -> int:
    total = 0
    for root in (
        settings.data_dir / "projects",
        settings.data_dir / "staging",
        settings.data_dir / "exports",
    ):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError as error:
                logger.exception("No se pudo consultar el tamaño de %s", path)
                raise HTTPException(
                    status_code=500,
                    detail="No se pudo calcular el almacenamiento local.",
                ) from error
    return total


def create_app(
    settings: AppSettings | None = None,
    web_dir: Path | None = None,
    *,
    auth_required: bool = True,
    auth_verifier: Any | None = None,
    updater: UpdateManager | None = None,
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
    async def lifespan(application: FastAPI):
        app_settings.data_dir.mkdir(parents=True, exist_ok=True)
        database.initialize(app_settings.database_path)
        database.fail_interrupted_projects(app_settings.database_path)
        database.fail_interrupted_staged_media(app_settings.database_path)
        _cleanup_expired_staging(app_settings)
        database.fail_interrupted_schedule_jobs(app_settings.database_path)
        database.hold_overdue_schedule_jobs(
            app_settings.database_path,
            database.utc_now(),
        )
        processing_queue = ProjectProcessingQueue(
            app_settings.processing_concurrency,
            lambda project_id, llm_config, retry_failed_only: asyncio.to_thread(
                process_project,
                project_id,
                llm_config,
                app_settings,
                retry_failed_only=retry_failed_only,
            ),
            # Cola persistente (Fase 2, B-S10): con database_path, los
            # trabajos encolados sobreviven a un reinicio y start()
            # recupera los pendientes. fail_interrupted_projects (arriba)
            # ya no falla los proyectos con trabajo recuperable.
            database_path=app_settings.database_path,
            # Franja nocturna (Fase 3, ítem 11): fuera de la franja, los
            # workers esperan en vez de procesar. Apagada por defecto.
            gate=lambda: watch_folder.night_window_allows(
                app_settings.database_path
            ),
        )
        await processing_queue.start()
        application.state.processing_queue = processing_queue

        async def _submit_background_project(project_id: str) -> None:
            """Encola un proyecto ya creado (carpeta observada, lote)."""
            project = database.get_project(app_settings.database_path, project_id)
            if project is None:
                raise RuntimeError(f"No se encontró el proyecto {project_id}.")
            if not database.queue_project(
                app_settings.database_path,
                project_id,
                project["config"],
            ):
                raise RuntimeError(
                    f"El proyecto {project_id} ya se está procesando."
                )
            await processing_queue.submit(project_id, None)

        application.state.submit_background_project = _submit_background_project
        scheduler_task = asyncio.create_task(schedule_worker(app_settings))
        # Carpeta observada + barrido de lotes (Fase 3, ítem 11): sin
        # carpetas ni lotes pendientes, el worker solo duerme.
        watch_task = asyncio.create_task(
            watch_folder.watch_folder_worker(
                app_settings,
                _submit_background_project,
            )
        )
        try:
            yield
        finally:
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
            await processing_queue.stop()
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass
            if isinstance(verifier, SupabaseAuthVerifier):
                await verifier.close()

    application = FastAPI(title="kellyClipAi API", version=APP_VERSION, lifespan=lifespan)
    application.state.settings = app_settings
    application.state.auth_required = auth_required
    application.state.auth_verifier = verifier
    application.state.claimed_legacy_users = set()
    application.state.updater = updater
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
            path in {"/api/health", "/api/auth/config", "/api/client-errors"}
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
            if project_id != "from-url" and not database.project_exists(
                app_settings.database_path, project_id, user_id
            ):
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
        elif (
            len(path_parts) >= 3
            and path_parts[1] == "scheduled-jobs"
            and database.get_schedule_job(
                app_settings.database_path, path_parts[2], user_id
            )
            is None
        ):
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
        return path.endswith(("/file", "/download", "/thumbnail", "/edit/preview", "/source", "/brand-kit/logo")) or (
            path.startswith("/api/scheduled-jobs/") and path.endswith("/download")
        )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # --- Codificador de video (v1.4.2) ------------------------------------
    # El frontend muestra el codificador activo (NVENC / QSV / AMF /
    # CPU) en Ajustes. La detección vive en backend/media.py y se
    # cachea por proceso, así que este endpoint no re-ejecuta ffmpeg.
    # La autenticación la aplica el middleware de /api/* como en los
    # demás endpoints de estado (solo /api/health es público).

    @application.get("/api/encoder")
    def get_encoder_status() -> dict[str, str]:
        return get_encoder_info(shutil.which("ffmpeg"))

    # --- Errores de la interfaz (v1.4.3) ----------------------------------
    # El ErrorBoundary del frontend envía aquí los errores no controlados
    # para que queden en desktop.log: en la app empaquetada la consola
    # del navegador no es visible (WebView2 con debug=False) y sin esto
    # un crash de la interfaz era indiagnosticable (eso pasó con el
    # crash del clip manual del 22-11-2026). Endpoint público (como
    # /api/health): el fallo puede ocurrir antes de iniciar sesión, y
    # el cuerpo solo alimenta el log — sin acceso a datos y con cada
    # campo limitado. Responde 204 con cualquier JSON válido; solo un
    # cuerpo que no sea JSON devuelve 400.

    @application.post("/api/client-errors", status_code=204)
    async def report_client_error(request: Request) -> Response:
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(
                status_code=400, detail="El cuerpo debe ser JSON válido."
            ) from None
        if not isinstance(payload, dict):
            payload = {}
        message = _client_error_text(payload.get("message"))
        stack = _client_error_text(payload.get("stack"))
        component_stack = _client_error_text(
            payload.get("component_stack", payload.get("componentStack"))
        )
        url = _client_error_text(payload.get("url"))
        logger.error(
            "Error de la interfaz: %s | url=%s | stack=%s | componentStack=%s",
            message or "(sin mensaje)",
            url or "-",
            stack or "-",
            component_stack or "-",
        )
        return Response(status_code=204)

    def update_status() -> dict[str, Any]:
        if application.state.updater is None:
            return {
                "enabled": False,
                "current_version": APP_VERSION,
                "latest_version": None,
                "update_available": False,
                "state": "unavailable",
                "progress": 0,
                "error": None,
                "channel": "stable",
                "release_notes": None,
                "release_url": None,
                "is_prerelease": False,
                "rollback": False,
                "previous_version": None,
                "rollback_available": False,
            }
        return application.state.updater.status()

    @application.get("/api/updates")
    def get_update_status() -> dict[str, Any]:
        return update_status()

    @application.post("/api/updates/check")
    def check_for_updates() -> dict[str, Any]:
        if application.state.updater is None:
            return update_status()
        return application.state.updater.start_check()

    @application.post("/api/updates/download")
    def download_update() -> dict[str, Any]:
        if application.state.updater is None:
            raise HTTPException(
                status_code=409,
                detail="Las actualizaciones están disponibles solo en la app instalada.",
            )
        try:
            return application.state.updater.start_download()
        except UpdateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.post("/api/updates/install")
    def install_update() -> dict[str, str]:
        if application.state.updater is None:
            raise HTTPException(
                status_code=409,
                detail="Las actualizaciones están disponibles solo en la app instalada.",
            )
        try:
            application.state.updater.open_installer()
        except UpdateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "installer_opened"}

    # --- Canal beta y rollback (Fase 3, ítems 17 y 20, C.6.37, C.6.40) ---

    @application.post("/api/updates/channel")
    def set_update_channel(payload: UpdateChannelRequest) -> dict[str, Any]:
        if application.state.updater is None:
            raise HTTPException(
                status_code=409,
                detail="Las actualizaciones están disponibles solo en la app instalada.",
            )
        try:
            return application.state.updater.set_channel(payload.channel)
        except UpdateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @application.get("/api/updates/rollback")
    def get_rollback_status() -> dict[str, Any]:
        if application.state.updater is None:
            return {
                "available": False,
                "previous_version": None,
                "installer_ready": False,
            }
        return application.state.updater.rollback_info()

    @application.post("/api/updates/rollback")
    def start_rollback() -> dict[str, Any]:
        if application.state.updater is None:
            raise HTTPException(
                status_code=409,
                detail="Las actualizaciones están disponibles solo en la app instalada.",
            )
        try:
            return application.state.updater.start_rollback()
        except UpdateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    # --- Escritorio: auto-inicio y bandeja (Fase 3, ítem 18) -------------

    def _application_executable() -> Path | None:
        updater = application.state.updater
        return updater.application_path if updater is not None else None

    @application.get("/api/desktop/autostart")
    def get_autostart_status() -> dict[str, bool]:
        executable = _application_executable()
        supported = autostart_supported() and executable is not None
        return {
            "supported": supported,
            "enabled": is_autostart_enabled(executable) if supported else False,
        }

    @application.post("/api/desktop/autostart")
    def set_autostart_status(payload: AutostartRequest) -> dict[str, bool]:
        executable = _application_executable()
        if executable is None or not autostart_supported():
            raise HTTPException(
                status_code=409,
                detail="El auto-inicio solo está disponible en la app instalada en Windows.",
            )
        try:
            set_autostart(payload.enabled, executable)
        except AutostartError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return {"enabled": payload.enabled}

    @application.get("/api/desktop/preferences")
    def get_desktop_preferences() -> dict[str, Any]:
        available, reason = tray_available()
        return {
            **load_desktop_prefs(app_settings.data_dir),
            "tray_available": available,
            "tray_reason": reason,
            "portable": os.getenv("KELLYCLIP_PORTABLE") == "1",
        }

    @application.post("/api/desktop/preferences")
    def set_desktop_preferences(
        payload: DesktopPreferencesRequest,
    ) -> dict[str, Any]:
        updates = payload.model_dump(exclude_unset=True, exclude_none=True)
        available, reason = tray_available()
        return {
            **save_desktop_prefs(app_settings.data_dir, updates),
            "tray_available": available,
            "tray_reason": reason,
            "portable": os.getenv("KELLYCLIP_PORTABLE") == "1",
        }

    # --- Asistente de primer arranque (Fase 3, ítem 19, C.6.42) ----------

    @application.get("/api/setup")
    def get_setup_status(request: Request) -> dict[str, Any]:
        user_id = request_user_id(request)
        credentials = database.get_llm_credentials(app_settings.database_path, user_id)
        completed = is_setup_complete(app_settings.data_dir)
        has_projects = database.count_projects(app_settings.database_path) > 0
        has_llm_key = credentials is not None or bool(app_settings.llm_api_key)
        saved_whisper_model = credentials.get("whisper_model") if credentials else None
        space = disk_space(app_settings.data_dir)
        downloaded = downloaded_models()
        return {
            "needs_setup": needs_setup(
                completed=completed,
                has_projects=has_projects,
                has_llm_key=has_llm_key,
            ),
            "completed": completed,
            "has_llm_key": has_llm_key,
            "has_projects": has_projects,
            "whisper_model": (
                saved_whisper_model
                or whisper_model_from_setup(app_settings.data_dir)
                or app_settings.whisper_model
            ),
            "whisper_models": [
                {
                    "name": model.name,
                    "size_bytes": model.size_bytes,
                    "label": model.label,
                    "description": model.description,
                    "downloaded": model.name in downloaded,
                }
                for model in WHISPER_MODELS
            ],
            "disk_free_bytes": space["free_bytes"],
            "disk_total_bytes": space["total_bytes"],
            "portable": os.getenv("KELLYCLIP_PORTABLE") == "1",
        }

    @application.post("/api/setup/complete")
    def complete_setup(payload: SetupCompleteRequest) -> dict[str, bool]:
        mark_setup_complete(
            app_settings.data_dir,
            whisper_model=payload.whisper_model,
            skipped=payload.skipped,
        )
        return {"completed": True}

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
        saved_whisper_model = credentials.get("whisper_model") if credentials else None
        return {
            "provider": credentials["provider"] if credentials else app_settings.llm_provider,
            "model": credentials["model"] if credentials else app_settings.llm_model,
            "base_url": credentials["base_url"] if credentials else app_settings.llm_base_url,
            "has_llm_key": credentials is not None or bool(app_settings.llm_api_key),
            "has_saved_llm_key": credentials is not None,
            "has_env_llm_key": bool(app_settings.llm_api_key),
            # El modelo Whisper guardado en Ajustes manda; si nunca se
            # guardó, vale el elegido en el asistente de primer arranque
            # (Fase 3, ítem 19) y, en último lugar, el de entorno
            # (WHISPER_MODEL).
            "whisper_model": (
                saved_whisper_model
                or whisper_model_from_setup(app_settings.data_dir)
                or app_settings.whisper_model
            ),
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
                # Guardar en Ajustes es una elección explícita del usuario:
                # las migraciones no reescribirán este modelo (B21).
                model_customized=True,
                # whisper_model None = sin cambio (el cliente no lo envió).
                whisper_model=payload.whisper_model,
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
            (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
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
            datetime.now(UTC).isoformat(),
        ) if state else None
        result = "error"
        detail = "La autorización no se pudo validar. Inicia de nuevo la conexión."
        if saved_state and saved_state["provider"] == provider and not error and code:
            try:
                account = complete_oauth(app_settings, provider, code)
                # save_account_document (Fase 3, ítem 10): una segunda
                # cuenta del mismo proveedor se guarda como extra, sin
                # desplazar a la predeterminada.
                database.save_account_document(
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
        if not database.project_exists(app_settings.database_path, project_id):
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

    @application.post("/api/scheduled-jobs/{job_id}/resolve-overdue")
    def resolve_overdue_schedule_job(
        job_id: str,
        resolution: OverdueScheduleResolution,
        request: Request,
    ) -> dict:
        job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        if not job.get("confirmation_required") or not database.resolve_overdue_schedule_job(
            app_settings.database_path,
            job_id,
            resolution.action,
        ):
            raise HTTPException(
                status_code=409,
                detail="La publicación ya no está pendiente de confirmación.",
            )
        updated_job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
        if updated_job is None:
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        return serialize_schedule_job(updated_job)

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

    # --- Plantilla de marca (Fase 3, ítem 5) -------------------------------
    # Un kit por usuario: colores, fuente y logo. El logo vive en disco
    # (<data_dir>/brand/<owner>/logo.<ext>); la BD guarda el nombre.

    def _brand_kit_payload(owner_id: str) -> dict[str, Any] | None:
        kit = database.get_brand_kit(app_settings.database_path, owner_id)
        if kit is None:
            return None
        payload = dict(kit)
        logo_path = brand.brand_logo_path(
            app_settings.data_dir, owner_id, kit.get("logo_filename")
        )
        payload["logo_url"] = (
            "/api/brand-kit/logo" if logo_path is not None else None
        )
        return payload

    @application.get("/api/brand-kit")
    def get_brand_kit(request: Request) -> dict[str, Any]:
        return {"brand_kit": _brand_kit_payload(request_user_id(request))}

    @application.put("/api/brand-kit")
    def put_brand_kit(kit: BrandKitRequest, request: Request) -> dict[str, Any]:
        owner_id = request_user_id(request)
        database.save_brand_kit(
            app_settings.database_path,
            kit.model_dump(mode="json"),
            owner_id,
        )
        return {"brand_kit": _brand_kit_payload(owner_id)}

    @application.delete("/api/brand-kit", status_code=204)
    def delete_brand_kit(request: Request) -> None:
        owner_id = request_user_id(request)
        database.delete_brand_kit(app_settings.database_path, owner_id)
        shutil.rmtree(
            brand.brand_dir(app_settings.data_dir, owner_id),
            ignore_errors=True,
        )

    @application.post("/api/brand-kit/logo")
    async def upload_brand_logo(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        content_type = (file.content_type or "").lower().split(";")[0].strip()
        suffix = Path(file.filename or "").suffix.lower()
        if (
            content_type not in brand.LOGO_MEDIA_TYPES
            and suffix not in brand.LOGO_EXTENSIONS
        ):
            raise HTTPException(
                status_code=415,
                detail="El logo debe ser una imagen PNG, JPG o WebP.",
            )
        if suffix not in brand.LOGO_EXTENSIONS:
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
            }[content_type]
        data = await file.read(brand.MAX_LOGO_BYTES + 1)
        if len(data) > brand.MAX_LOGO_BYTES:
            raise HTTPException(
                status_code=413,
                detail="El logo supera el límite de 2 MB.",
            )
        if not data:
            raise HTTPException(
                status_code=400, detail="El archivo del logo está vacío."
            )
        directory = brand.brand_dir(app_settings.data_dir, owner_id)
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.iterdir():
            if old.is_file() and old.suffix.lower() in brand.LOGO_EXTENSIONS:
                old.unlink(missing_ok=True)
        filename = f"logo{suffix}"
        (directory / filename).write_bytes(data)
        if database.get_brand_kit(app_settings.database_path, owner_id) is None:
            database.save_brand_kit(app_settings.database_path, {}, owner_id)
        database.set_brand_kit_logo(
            app_settings.database_path, filename, owner_id
        )
        return {"brand_kit": _brand_kit_payload(owner_id)}

    @application.get("/api/brand-kit/logo")
    def get_brand_logo(request: Request) -> FileResponse:
        owner_id = request_user_id(request)
        kit = database.get_brand_kit(app_settings.database_path, owner_id)
        logo_path = (
            brand.brand_logo_path(
                app_settings.data_dir, owner_id, kit.get("logo_filename")
            )
            if kit is not None
            else None
        )
        if logo_path is None:
            raise HTTPException(status_code=404, detail="No hay logo guardado.")
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(logo_path.suffix.lower(), "image/png")
        return FileResponse(logo_path, media_type=media_type)

    @application.get("/api/projects")
    def projects(
        request: Request,
        limit: int = Query(
            DEFAULT_PROJECTS_PAGE_SIZE, ge=1, le=MAX_PROJECTS_PAGE_SIZE
        ),
        offset: int = Query(0, ge=0),
    ) -> dict:
        """Listado paginado de proyectos (Fase 2, B-R10).

        Respuesta: {"items": [...], "total": N, "limit": L, "offset": O}.
        El cambio de forma (antes devolvía una lista plana) es
        deliberado: el frontend actual no consume este endpoint (solo
        pide proyectos por id), y la vista "Mis proyectos" necesita el
        total para paginar. Cada clip del listado viene SIN su
        transcripción palabra por palabra; el detalle de proyecto
        (GET /api/projects/{id}) la sigue incluyendo.
        """
        owner_id = request_user_id(request)
        total = database.count_projects(app_settings.database_path, owner_id)
        items = [
            _serialize_project(project, include_clip_transcripts=False)
            for project in database.list_projects(
                app_settings.database_path,
                owner_id,
                limit=limit,
                offset=offset,
            )
        ]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @application.get("/api/storage")
    def storage_usage() -> dict[str, int]:
        return {"used_bytes": _local_storage_bytes(app_settings)}

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
        """Recibe el contenido de una carga local, con reanudación (B-F11).

        Protocolo:
        - Sin cabecera ni parámetro: subida de un solo disparo (como
          antes): el cuerpo debe ser el archivo completo.
        - Con `Content-Range: bytes <inicio>-<fin>/<total>` (o el
          parámetro `?offset=<inicio>`): subida por fragmentos. El total
          debe coincidir con el file_size declarado al crear el staged.
          El inicio debe ser <= a los bytes ya guardados en disco (el
          .part es la fuente de verdad; si el cliente reenvía desde
          antes, se sobrescribe desde ahí). Un inicio mayor se rechaza
          con 409 indicando desde dónde reanudar.
        - Cada fragmento aceptado devuelve 200 con el staged actualizado
          (status "receiving", bytes_received al día); solo al completar
          el último byte el staged pasa a "ready" y el .part se mueve
          a .source.
        - Ante una interrupción (desconexión, fragmento incompleto o
          error de disco) NO se borra el .part ni se marca failed: el
          staged queda en "receiving" con bytes_received actualizado
          (visible en el GET) y el cliente reanuda desde ahí.
        - La subida legacy sin rango que llega corta sigue siendo 400 +
          failed, como antes: sin rango no hay forma de saber si el
          cliente reintentará.
        El lock por staged_id serializa fragmentos concurrentes y el
        borrado del staged (carrera B4).
        """
        async with _staged_upload_lock(staged_id):
            staged = database.get_staged_media(app_settings.database_path, staged_id)
            if staged is None or staged["source_type"] != "upload":
                raise HTTPException(status_code=404, detail="No se encontró la carga local.")
            if staged["status"] != "receiving":
                raise HTTPException(status_code=409, detail="La carga local ya no está disponible.")
            file_size = int(staged["file_size"])
            try:
                content_range = _parse_content_range(
                    request.headers.get("content-range")
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            offset_param = request.query_params.get("offset")
            resumable = content_range is not None or offset_param is not None
            chunk_start = 0
            chunk_length: int | None = None
            if content_range is not None:
                chunk_start, chunk_end, range_total = content_range
                if range_total != file_size:
                    raise HTTPException(
                        status_code=400,
                        detail="El total de Content-Range no coincide con el tamaño declarado del archivo.",
                    )
                chunk_length = chunk_end - chunk_start + 1
            elif offset_param is not None:
                try:
                    chunk_start = int(offset_param)
                except ValueError as error:
                    raise HTTPException(
                        status_code=400,
                        detail="El parámetro offset debe ser un número entero.",
                    ) from error
                if chunk_start < 0:
                    raise HTTPException(
                        status_code=400,
                        detail="El parámetro offset no puede ser negativo.",
                    )

            staging_root = app_settings.data_dir / "staging"
            partial_path = staging_root / f"{staged_id}.part"
            staged_path = staging_root / f"{staged_id}.source"
            part_size = partial_path.stat().st_size if partial_path.is_file() else 0
            if chunk_start > part_size:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"El offset {chunk_start} supera los {part_size} bytes ya "
                        f"recibidos. Reanuda desde {part_size}."
                    ),
                )
            received_in_request = 0
            last_saved = 0
            try:
                mode = "r+b" if partial_path.is_file() else "wb"
                with partial_path.open(mode) as target:
                    if chunk_start < part_size:
                        # El cliente reenvía desde antes de lo guardado
                        # (su vista de bytes_received iba por detrás): se
                        # trunca y se sobrescribe desde ahí.
                        target.truncate(chunk_start)
                    target.seek(chunk_start)
                    async for chunk in request.stream():
                        if not chunk:
                            continue
                        if (
                            chunk_length is not None
                            and received_in_request + len(chunk) > chunk_length
                        ):
                            raise _ChunkTooLargeError(
                                "El cuerpo supera el rango declarado en Content-Range."
                            )
                        if chunk_start + received_in_request + len(chunk) > file_size:
                            raise _ChunkTooLargeError(
                                "La carga supera el tamaño del archivo seleccionado."
                            )
                        received_in_request += len(chunk)
                        for offset in range(0, len(chunk), _STAGED_WRITE_SIZE):
                            block = memoryview(chunk)[offset : offset + _STAGED_WRITE_SIZE]
                            await asyncio.to_thread(target.write, block)
                        if received_in_request - last_saved >= 4 * _STAGED_WRITE_SIZE:
                            database.update_staged_media(
                                app_settings.database_path,
                                staged_id,
                                bytes_received=chunk_start + received_in_request,
                            )
                            last_saved = received_in_request
                    await asyncio.to_thread(target.flush)
            except _ChunkTooLargeError as error:
                if resumable:
                    # Se revierte solo este fragmento: el .part vuelve a
                    # su tamaño previo y el cliente puede reintentarlo.
                    try:
                        with partial_path.open("r+b") as target:
                            target.truncate(chunk_start)
                    except OSError:
                        pass
                    database.update_staged_media(
                        app_settings.database_path,
                        staged_id,
                        bytes_received=chunk_start,
                    )
                else:
                    partial_path.unlink(missing_ok=True)
                    database.update_staged_media(
                        app_settings.database_path,
                        staged_id,
                        status="failed",
                        error=str(error),
                    )
                raise HTTPException(status_code=413, detail=str(error)) from error
            except ClientDisconnect as error:
                # Interrupción: NO se borra el .part ni se marca failed.
                # El cliente reanuda desde los bytes guardados en disco.
                bytes_on_disk = (
                    partial_path.stat().st_size if partial_path.is_file() else chunk_start
                )
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    bytes_received=bytes_on_disk,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "La conexión se interrumpió durante la carga. "
                        f"Reanuda desde el byte {bytes_on_disk}."
                    ),
                ) from error
            except OSError as error:
                # Error de disco (p. ej. sin espacio): se conserva lo
                # recibido para reanudar cuando haya espacio.
                bytes_on_disk = (
                    partial_path.stat().st_size if partial_path.is_file() else chunk_start
                )
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    bytes_received=bytes_on_disk,
                    error=f"No se pudo guardar el archivo en disco: {error}",
                )
                raise HTTPException(
                    status_code=507,
                    detail=f"No se pudo guardar el archivo en disco. Revisa el espacio disponible: {error}",
                ) from error

            total_received = chunk_start + received_in_request
            if chunk_length is not None and received_in_request < chunk_length:
                # El fragmento quedó incompleto (el cliente cerró antes
                # de enviar el rango declarado): se conservan los bytes
                # recibidos y el estado receiving para reanudar.
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    bytes_received=total_received,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"El fragmento quedó incompleto: se recibieron "
                        f"{received_in_request} de {chunk_length} bytes. "
                        f"Reanuda desde {total_received}."
                    ),
                )
            if total_received == file_size:
                partial_path.replace(staged_path)
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    status="ready",
                    bytes_received=total_received,
                    total_bytes=total_received,
                    speed=None,
                    eta=None,
                    error=None,
                )
            elif resumable:
                # Fragmento aceptado; quedan más por venir.
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    bytes_received=total_received,
                )
            else:
                # Subida legacy sin rango que llega corta: como antes,
                # 400 y failed (sin rango no se puede reanudar).
                partial_path.unlink(missing_ok=True)
                database.update_staged_media(
                    app_settings.database_path,
                    staged_id,
                    status="failed",
                    error="La carga se interrumpió antes de recibir el archivo completo.",
                )
                raise HTTPException(
                    status_code=400,
                    detail="La carga se interrumpió antes de recibir el archivo completo.",
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
    async def delete_staged_media(staged_id: str) -> Response:
        # El mismo lock de la subida (B4): si hay un fragmento en curso,
        # el borrado espera a que termine en vez de borrar el .part
        # mientras se escribe.
        async with _staged_upload_lock(staged_id):
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
        file: Annotated[UploadFile, File(...)],
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
            # Copia por bloques contando bytes: la subida no declara su
            # tamaño por adelantado y un video mayor que MAX_UPLOAD_BYTES
            # se rechaza con 413 en cuanto supera el límite, sin esperar
            # a copiarlo entero ni llenar el disco (A2).
            bytes_written = 0
            with input_path.open("wb") as target:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "El video supera el tamaño máximo de "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024 * 1024)} GB por proyecto."
                            ),
                        )
                    target.write(chunk)
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
        return _serialize_project(project, include_transcript=True)

    @application.delete("/api/projects/{project_id}")
    def delete_project_route(project_id: str, request: Request) -> dict[str, bool]:
        owner_id = request_user_id(request)
        project = database.get_project(
            app_settings.database_path, project_id, owner_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(
                status_code=409,
                detail="No se puede eliminar un proyecto mientras se procesa.",
            )
        jobs = database.list_schedule_jobs(
            app_settings.database_path, project_id, owner_id
        )
        if any(job["status"] == "processing" for job in jobs):
            raise HTTPException(
                status_code=409,
                detail="Espera a que termine la publicación antes de eliminar el proyecto.",
            )
        project_root = (app_settings.data_dir / "projects").resolve()
        project_path = project_root / project_id
        if not project_path.resolve().is_relative_to(project_root):
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        try:
            staged_ids = database.delete_project(
                app_settings.database_path, project_id, owner_id
            )
        except database.ProjectBusyError as error:
            # El proyecto empezó a procesarse entre la comprobación de
            # estado y el borrado: la guarda atómica de delete_project
            # lo impidió. Para borrarlo hay que cancelarlo primero.
            raise HTTPException(status_code=409, detail=str(error)) from error
        if staged_ids is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        try:
            if project_path.is_symlink():
                project_path.unlink()
            elif project_path.is_dir():
                shutil.rmtree(project_path)
            export_root = (app_settings.data_dir / "exports").resolve()
            for job in jobs:
                if not job.get("output_path"):
                    continue
                export_path = Path(job["output_path"]).resolve()
                if (
                    export_path.is_relative_to(export_root)
                    and export_path.is_file()
                ):
                    export_path.unlink()
            _remove_staging_artifacts(app_settings, staged_ids)
        except OSError as error:
            logger.exception(
                "No se pudieron eliminar todos los archivos del proyecto %s",
                project_id,
            )
            raise HTTPException(
                status_code=500,
                detail="El proyecto se eliminó de la base de datos, pero no se pudieron limpiar todos sus archivos.",
            ) from error
        return {"deleted": True}

    @application.get("/api/projects/{project_id}/thumbnail")
    def get_thumbnail(project_id: str) -> FileResponse:
        project_dir = app_settings.data_dir / "projects" / project_id
        if not database.project_exists(app_settings.database_path, project_id):
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        thumbnail = project_dir / "thumbnail.jpg"
        if not thumbnail.is_file():
            raise HTTPException(status_code=404, detail="La miniatura aún no está disponible.")
        return FileResponse(thumbnail, media_type="image/jpeg")

    @application.get("/api/projects/{project_id}/proposals/{proposal_id}/thumbnail")
    def get_proposal_thumbnail(project_id: str, proposal_id: str) -> FileResponse:
        """Miniatura de una propuesta sin editar (hotfix v1.4.1).

        Las propuestas (edición automática apagada, v1.3.2) no tienen
        clip renderizado, así que la pantalla de resultados no tenía
        ninguna vista previa que mostrar para ellas. El fotograma se
        extrae del video fuente en el punto medio del momento y se
        cachea en disco; las siguientes peticiones sirven el archivo
        guardado. La ruta termina en /thumbnail, así que el middleware
        la trata como petición de medio y acepta la cookie
        kellyclip_media (un <img> no puede enviar Authorization).
        """
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        metadata = project.get("metadata")
        records = metadata.get("clip_proposals") if isinstance(metadata, dict) else None
        record = next(
            (
                item
                for item in records or []
                if isinstance(item, dict) and item.get("id") == proposal_id
            ),
            None,
        )
        proposal = record.get("proposal") if isinstance(record, dict) else None
        if not isinstance(proposal, dict):
            raise HTTPException(status_code=404, detail="No se encontró la propuesta.")
        try:
            start = float(proposal["inicio"])
            end = float(proposal["fin"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=404, detail="No se encontró la propuesta.") from None
        if not end > start >= 0:
            raise HTTPException(status_code=404, detail="No se encontró la propuesta.")
        project_dir = app_settings.data_dir / "projects" / project_id
        thumbnail_dir = project_dir / "proposal-thumbnails"
        thumbnail = thumbnail_dir / f"{proposal_id}.jpg"
        if thumbnail.is_file():
            # La caché manda: la vista previa sigue disponible aunque
            # el video fuente ya se haya borrado.
            return FileResponse(thumbnail, media_type="image/jpeg")
        source = project_dir / "source"
        if not source.is_file():
            raise HTTPException(status_code=404, detail="El video original ya no está disponible.")
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise HTTPException(status_code=503, detail="FFmpeg no está disponible.")
        duration = project_duration(project)
        timestamp = (start + end) / 2
        if duration is not None:
            timestamp = min(timestamp, max(0.0, duration - 0.1))
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        try:
            create_frame_thumbnail(source, thumbnail, timestamp, ffmpeg_path)
        except MediaProcessingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not thumbnail.is_file():
            raise HTTPException(
                status_code=422,
                detail="No se pudo extraer el fotograma de la propuesta.",
            )
        return FileResponse(thumbnail, media_type="image/jpeg")

    @application.post("/api/projects/{project_id}/process", status_code=202)
    async def process(
        project_id: str,
        request: ProcessRequest,
    ) -> dict[str, str]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(status_code=409, detail="Este proyecto ya se está procesando.")
        stored_config = request.config or ProjectConfig.model_validate(project["config"])
        # El motor de señales (por defecto) no necesita clave de IA: el
        # gate de credenciales solo aplica al modo "llm". En "signals+ai"
        # el pulido es best-effort (sin clave, títulos por plantilla).
        llm_config: LLMConfig | None = None
        if stored_config.moment_engine == "llm":
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
            except ValidationError as error:
                # Las credenciales guardadas o el entorno pueden traer una
                # base_url que el validador actual rechaza (p. ej. un
                # servidor local en LAN http://192.168.x.x): es un error de
                # configuración, no un 500. Se devuelve 422 con el motivo.
                details = "; ".join(
                    f"{'.'.join(str(part) for part in item.get('loc', ()))}: {item.get('msg', '')}"
                    for item in error.errors()[:3]
                )
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "La configuración del proveedor de IA no es válida"
                        f"{f': {details}' if details else ''}. Revísala en Ajustes."
                    ),
                ) from error
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
        if stored_config.schedule and stored_config.schedule.scheduled_at <= datetime.now(UTC):
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
        processing_queue: ProjectProcessingQueue | None = getattr(
            application.state, "processing_queue", None
        )
        if processing_queue is None:
            raise HTTPException(
                status_code=503,
                detail="La cola de procesamiento no está disponible. Reinicia la aplicación.",
            )
        try:
            await processing_queue.submit(project_id, llm_config)
        except RuntimeError as error:
            database.update_project(
                app_settings.database_path,
                project_id,
                status="failed",
                step="failed",
                error="La cola de procesamiento se cerró antes de aceptar el trabajo.",
            )
            raise HTTPException(
                status_code=503,
                detail="La cola de procesamiento no está disponible. Inténtalo de nuevo.",
            ) from error
        return {"project_id": project_id, "status": "queued"}

    @application.post("/api/projects/{project_id}/cancel")
    async def cancel_project(project_id: str, request: Request) -> dict:
        """Cancela un procesamiento en curso (B14/B-P7).

        Proyecto en cola: se saca de la cola y se marca cancelled. Proyecto
        procesándose: cancelación cooperativa — el pipeline consulta el flag
        y el estado persistido entre fases, se detiene pronto y conserva los
        clips ya renderizados. Devuelve el proyecto actualizado.
        """
        owner_id = request_user_id(request)
        project = database.get_project(
            app_settings.database_path, project_id, owner_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] not in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail="Este proyecto no se está procesando.",
            )
        # Primero el flag en memoria (el pipeline lo consulta entre fases)
        # y luego el estado persistido, que es la fuente de verdad.
        request_project_cancellation(project_id)
        processing_queue: ProjectProcessingQueue | None = getattr(
            application.state, "processing_queue", None
        )
        if processing_queue is not None:
            processing_queue.discard_queued(project_id)
        database.update_project(
            app_settings.database_path,
            project_id,
            status="cancelled",
            step="cancelled",
            progress=project.get("progress"),
            error=None,
        )
        database.publish_event(
            app_settings.database_path,
            project_id,
            "progress",
            {
                "status": "cancelled",
                "step": "cancelled",
                "progress": project.get("progress"),
                "message": "Procesamiento cancelado.",
            },
        )
        updated = database.get_project(
            app_settings.database_path, project_id, owner_id
        )
        assert updated is not None
        return _serialize_project(updated)

    @application.post(
        "/api/projects/{project_id}/retry-failed-clips", status_code=202
    )
    async def retry_failed_clips(project_id: str) -> dict[str, str]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(
                status_code=409, detail="Este proyecto ya se está procesando."
            )
        failed_ids = {
            str(clip["id"])
            for clip in project.get("clips") or []
            if clip.get("status") == "failed"
        }
        proposals = (project.get("metadata") or {}).get("clip_proposals") or []
        proposal_ids = {
            str(proposal["id"])
            for proposal in proposals
            if isinstance(proposal, dict) and proposal.get("id")
        }
        if not failed_ids.intersection(proposal_ids):
            raise HTTPException(
                status_code=409,
                detail="No hay clips fallidos con una propuesta guardada que se pueda reintentar.",
            )
        if not database.queue_project(
            app_settings.database_path,
            project_id,
            project["config"],
            preserve_clips=True,
        ):
            raise HTTPException(status_code=409, detail="Este proyecto ya se está procesando.")
        processing_queue: ProjectProcessingQueue | None = getattr(
            application.state, "processing_queue", None
        )
        if processing_queue is None:
            raise HTTPException(
                status_code=503,
                detail="La cola de procesamiento no está disponible. Reinicia la aplicación.",
            )
        try:
            await processing_queue.submit(
                project_id,
                None,
                retry_failed_only=True,
            )
        except RuntimeError as error:
            database.update_project(
                app_settings.database_path,
                project_id,
                status="failed",
                step="failed",
                error="La cola de procesamiento se cerró antes de aceptar el reintento.",
            )
            raise HTTPException(
                status_code=503,
                detail="La cola de procesamiento no está disponible. Inténtalo de nuevo.",
            ) from error
        return {"project_id": project_id, "status": "queued"}

    @application.post("/api/projects/{project_id}/improve-titles")
    def improve_titles(project_id: str) -> dict[str, Any]:
        """Mejora los títulos de los clips de un proyecto con UNA llamada a la IA (Fase D).

        Reutiliza moments.polish_titles, el mismo pulido del modo
        "signals+ai". Actualiza las propuestas guardadas y los títulos de
        los clips renderizados que comparten id con ellas, en una sola
        escritura atómica. Sin clave de IA configurada responde 409: a
        diferencia del pulido durante el procesado, aquí el usuario lo
        pidió de forma explícita.
        """
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(
                status_code=409,
                detail="Espera a que termine el procesamiento antes de mejorar los títulos.",
            )
        metadata = dict(project.get("metadata") or {})
        records = metadata.get("clip_proposals") or []
        proposals: list[ClipProposal] = []
        record_ids: list[str] = []
        for record in records:
            if not isinstance(record, dict) or not record.get("id"):
                continue
            try:
                proposals.append(ClipProposal.model_validate(record.get("proposal")))
            except (ValidationError, TypeError):
                # Propuesta corrupta: se conserva tal cual, sin pulir.
                continue
            record_ids.append(str(record["id"]))
        if not proposals:
            raise HTTPException(
                status_code=409,
                detail="Este proyecto no tiene clips para mejorar.",
            )
        try:
            llm = _resolve_llm_for_project(project, None, app_settings)
        except (LLMConfigurationError, CredentialStoreError, ValidationError) as error:
            raise HTTPException(
                status_code=409,
                detail="Configura una clave de IA en Ajustes para mejorar títulos.",
            ) from error
        # La transcripción queda persistida en la columna transcript_json del
        # proyecto (el pipeline la guarda al transcribir). Si un proyecto
        # viejo no la trae, se construyen segmentos sintéticos desde el
        # título y la descripción de cada clip para que polish_titles tenga
        # contexto: cada segmento cubre la ventana de su clip.
        transcript: list[TranscriptSegment] = []
        for segment in project.get("transcript") or []:
            try:
                transcript.append(TranscriptSegment.model_validate(segment))
            except (ValidationError, TypeError):
                continue
        if not transcript:
            transcript = [
                TranscriptSegment(
                    start=proposal.inicio,
                    end=proposal.fin,
                    text=f"{proposal.titulo} {proposal.descripcion}".strip(),
                )
                for proposal in proposals
            ]
        polished = moments.polish_titles(proposals, transcript, llm)
        if polished is None:
            serialized = _serialize_project(project, include_transcript=False)
            return {
                "improved": 0,
                "message": "Los títulos ya están bien, no hubo cambios.",
                "clips": serialized.get("clips") or [],
            }
        improved = sum(
            1
            for original, updated in zip(proposals, polished, strict=True)
            if updated.titulo != original.titulo
        )
        polished_by_id = {
            record_id: proposal.model_dump(mode="json")
            for record_id, proposal in zip(record_ids, polished, strict=True)
        }
        # Se reconstruyen los registros conservando los ids originales (los
        # clips renderizados comparten id con su propuesta) y los registros
        # que no se pudieron validar, tal cual estaban.
        new_records: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict) or not record.get("id"):
                new_records.append(record)
                continue
            updated = polished_by_id.get(str(record["id"]))
            new_records.append(
                {"id": record["id"], "proposal": updated} if updated is not None else record
            )
        metadata["clip_proposals"] = new_records
        new_clips: list[dict[str, Any]] = []
        for clip in project.get("clips") or []:
            if not isinstance(clip, dict):
                new_clips.append(clip)
                continue
            updated = polished_by_id.get(str(clip.get("id")))
            if updated is None:
                new_clips.append(clip)
                continue
            new_clips.append(
                {
                    **clip,
                    "title": updated["titulo"],
                    "description": updated.get("descripcion", clip.get("description", "")),
                    "keywords": updated.get("palabras_clave", clip.get("keywords", [])),
                }
            )
        try:
            # Una sola sentencia UPDATE: o se guardan propuestas y clips
            # juntos, o no se guarda nada.
            database.update_project(
                app_settings.database_path,
                project_id,
                metadata=metadata,
                clips=new_clips,
            )
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudieron guardar los títulos mejorados: {error}",
            ) from error
        refreshed = database.get_project(app_settings.database_path, project_id)
        if refreshed is None:  # pragma: no cover - el proyecto se borró entre medias
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        serialized = _serialize_project(refreshed, include_transcript=False)
        return {
            "improved": improved,
            "message": (
                f"Se mejoraron {improved} título(s)."
                if improved
                else "Los títulos ya están bien, no hubo cambios."
            ),
            "clips": serialized.get("clips") or [],
        }

    @application.get("/api/projects/{project_id}/events")
    async def project_events(project_id: str, request: Request) -> EventSourceResponse:
        if not database.project_exists(app_settings.database_path, project_id):
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
                status = database.get_project_status(
                    app_settings.database_path, project_id
                )
                if status is None or status in {"completed", "failed", "cancelled"}:
                    # BUG 2 (v1.3.2): el estado terminal y su evento se
                    # confirman en una sola transacción, pero esta
                    # iteración pudo drenar los eventos justo antes de
                    # esa confirmación y leer el estado terminal
                    # después. Drenaje final antes de cerrar: si el
                    # evento terminal llegó entre ambas lecturas, se
                    # entrega en vez de perderse (el frontend se
                    # quedaba en 98 %).
                    for event in database.events_after(
                        app_settings.database_path, project_id, after_id
                    ):
                        after_id = event["id"]
                        yield {
                            "id": str(event["id"]),
                            "event": event["event"],
                            "data": json.dumps(event["data"], ensure_ascii=False),
                        }
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

    # --- Renders del editor como trabajos asíncronos (Fase 2, B-P8) --------
    # Los endpoints síncronos de arriba se MANTIENEN por compatibilidad
    # (el frontend actual los usa y los tests de contrato los cubren).
    # Los endpoints /jobs son la vía nueva: encolan el render en vez de
    # bloquear la petición HTTP minutos, exponen el progreso por SSE y
    # permiten cancelar. El resultado del trabajo, al completarse, tiene
    # la misma forma que la respuesta del endpoint síncrono equivalente.
    def editor_job_manager() -> EditorRenderJobManager:
        manager = getattr(application.state, "editor_jobs", None)
        if manager is None:
            manager = EditorRenderJobManager(app_settings)
            application.state.editor_jobs = manager
        return manager

    def editor_job_or_404(job_id: str) -> dict:
        job = editor_job_manager().get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="No se encontró el trabajo de render."
            )
        return job.to_dict()

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/edit/preview/jobs",
        status_code=202,
        response_model=EditorRenderJobResponse,
    )
    async def create_preview_job(
        project_id: str,
        clip_id: str,
        edit: ClipEditRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        try:
            job = editor_job_manager().submit(project, clip, edit, "preview")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return job.to_dict()

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/edit/export/jobs",
        status_code=202,
        response_model=EditorRenderJobResponse,
    )
    async def create_export_job(
        project_id: str,
        clip_id: str,
        edit: ClipEditRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        try:
            job = editor_job_manager().submit(project, clip, edit, "export")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return job.to_dict()

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/quality/jobs",
        status_code=202,
        response_model=EditorRenderJobResponse,
    )
    async def create_quality_job(
        project_id: str,
        clip_id: str,
        request: ClipResolutionRequest,
    ) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        edit = stored_editor_request(project, clip)
        try:
            job = editor_job_manager().submit(
                project, clip, edit, "quality", output_resolution=request.resolution
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return job.to_dict()

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/improve/jobs",
        status_code=202,
        response_model=EditorRenderJobResponse,
    )
    async def create_improve_job(project_id: str, clip_id: str) -> dict:
        project, clip = editable_project_clip(project_id, clip_id)
        edit = stored_editor_request(project, clip)
        try:
            job = editor_job_manager().submit(
                project, clip, edit, "improve", enhance_audio=True
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return job.to_dict()

    @application.get(
        "/api/editor-jobs/{job_id}",
        response_model=EditorRenderJobResponse,
    )
    def get_editor_job(job_id: str) -> dict:
        return editor_job_or_404(job_id)

    @application.get("/api/editor-jobs/{job_id}/events")
    async def editor_job_events(job_id: str, request: Request) -> EventSourceResponse:
        if editor_job_manager().get(job_id) is None:
            raise HTTPException(
                status_code=404, detail="No se encontró el trabajo de render."
            )

        async def event_stream():
            last_state: tuple[str, float] | None = None
            while True:
                job = editor_job_manager().get(job_id)
                if job is None:
                    break
                state = (job.status, round(job.progress, 1))
                if state != last_state:
                    last_state = state
                    yield {
                        "event": "progress",
                        "data": json.dumps(job.to_dict(), ensure_ascii=False),
                    }
                if job.status in {"completed", "failed", "cancelled"}:
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.35)

        return EventSourceResponse(event_stream(), ping=15)

    @application.post(
        "/api/editor-jobs/{job_id}/cancel",
        response_model=EditorRenderJobResponse,
    )
    def cancel_editor_job(job_id: str) -> dict:
        job = editor_job_manager().get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="No se encontró el trabajo de render."
            )
        if not editor_job_manager().cancel(job_id):
            raise HTTPException(
                status_code=409, detail="Este trabajo ya terminó y no se puede cancelar."
            )
        return job.to_dict()

    @application.get(
        "/api/projects/{project_id}/clips/{clip_id}/subtitles/{format_name}"
    )
    def download_clip_subtitles(
        project_id: str,
        clip_id: str,
        format_name: Literal["srt", "vtt"],
    ) -> Response:
        _, clip = editable_project_clip(project_id, clip_id)
        if clip.get("status") == "failed":
            raise HTTPException(
                status_code=409,
                detail="No se pueden exportar subtítulos de un clip fallido.",
            )
        transcript = clip.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise HTTPException(
                status_code=409,
                detail="El clip no tiene una transcripción con marcas de tiempo.",
            )
        try:
            content = format_subtitles(transcript, format_name)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail="La transcripción guardada no tiene tiempos válidos para exportar.",
            ) from error
        media_type = (
            "text/vtt; charset=utf-8"
            if format_name == "vtt"
            else "application/x-subrip; charset=utf-8"
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{clip_id}.{format_name}"'
            },
        )

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/subtitles/translate"
    )
    def translate_clip_subtitles(
        project_id: str,
        clip_id: str,
        body: TranslateSubtitlesRequest,
    ) -> Response:
        """Traduce los subtítulos del clip con el LLM y los descarga.

        Sin clave de IA configurada responde 409 con un mensaje claro:
        la traducción es explícita del usuario, no un pulido opcional.
        """
        _, clip = editable_project_clip(project_id, clip_id)
        if clip.get("status") == "failed":
            raise HTTPException(
                status_code=409,
                detail="No se pueden traducir subtítulos de un clip fallido.",
            )
        transcript = clip.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise HTTPException(
                status_code=409,
                detail="El clip no tiene una transcripción con marcas de tiempo.",
            )
        project = database.get_project(app_settings.database_path, project_id)
        assert project is not None
        try:
            llm = _resolve_llm_for_project(project, None, app_settings)
        except (LLMConfigurationError, CredentialStoreError, ValidationError) as error:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Configura una clave de IA en Ajustes para traducir "
                    "subtítulos (la traducción usa tu proveedor de IA)."
                ),
            ) from error
        cues = translate.clip_words_to_cues(transcript)
        if not cues:
            raise HTTPException(
                status_code=409,
                detail="El clip no tiene subtítulos para traducir.",
            )
        try:
            translated = translate.translate_cues(
                cues, body.target_language, llm
            )
        except (LLMConfigurationError, LLMResponseError) as error:
            raise HTTPException(
                status_code=502,
                detail=f"La traducción falló: {error}",
            ) from error
        try:
            content = format_subtitles(
                translate.translated_words(translated), body.format_name
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail="La traducción no produjo subtítulos válidos.",
            ) from error
        media_type = (
            "text/vtt; charset=utf-8"
            if body.format_name == "vtt"
            else "application/x-subrip; charset=utf-8"
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{clip_id}-{body.target_language}'
                    f'.{body.format_name}"'
                )
            },
        )

    # --- Versiones de clip (Fase 3, ítem 6) --------------------------------
    # Cada render del editor y cada exportación del Studio archivan el
    # archivo vigente antes de sustituirlo; aquí se listan y restauran.

    def _clip_version_paths(
        project_id: str, clip_id: str
    ) -> tuple[dict, dict, Path, Path]:
        project, clip = editable_project_clip(project_id, clip_id)
        project_dir = app_settings.data_dir / "projects" / project_id
        clips_dir = project_dir / "clips"
        return project, clip, project_dir, clips_dir

    @application.get("/api/projects/{project_id}/clips/{clip_id}/versions")
    def list_clip_versions(project_id: str, clip_id: str) -> dict:
        _, _, project_dir, _ = _clip_version_paths(project_id, clip_id)
        return {
            "versions": clip_versions.list_versions(project_dir, clip_id),
        }

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/versions/{version}/restore"
    )
    def restore_clip_version(
        project_id: str, clip_id: str, version: int
    ) -> dict:
        project, clip, project_dir, clips_dir = _clip_version_paths(
            project_id, clip_id
        )
        filename = Path(str(clip.get("filename", ""))).name
        if not filename or not filename.lower().endswith(".mp4"):
            raise HTTPException(
                status_code=409,
                detail="El archivo guardado del clip no es válido.",
            )
        try:
            clip_versions.restore_version(
                project_dir, clips_dir, clip_id, filename, version
            )
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="No existe esa versión del clip."
            ) from error
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo restaurar la versión: {error}",
            ) from error
        # La versión restaurada puede tener otra duración o resolución:
        # se vuelve a sondear el archivo vigente (best-effort).
        target = clips_dir / filename
        try:
            ffprobe_path = shutil.which("ffprobe")
            if ffprobe_path is not None:
                metadata = probe_media(target, ffprobe_path)
                clips = list(project.get("clips") or [])
                for index, current in enumerate(clips):
                    if current.get("id") == clip_id:
                        clips[index] = {
                            **current,
                            "duration": metadata["duration"],
                            "width": metadata["width"],
                            "height": metadata["height"],
                        }
                        break
                database.update_project(
                    app_settings.database_path, project_id, clips=clips
                )
        except (MediaProcessingError, OSError, KeyError) as error:
            logger.warning(
                "No se pudo sondear el clip restaurado (%s).", error
            )
        return {
            "clip_id": clip_id,
            "restored_version": version,
            "versions": clip_versions.list_versions(project_dir, clip_id),
        }

    @application.get(
        "/api/projects/{project_id}/clips/{clip_id}/versions/{version}/file"
    )
    def download_clip_version(
        project_id: str, clip_id: str, version: int
    ) -> FileResponse:
        _, _, project_dir, _ = _clip_version_paths(project_id, clip_id)
        path = clip_versions.versions_dir(project_dir, clip_id) / f"v{version}.mp4"
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail="No existe esa versión del clip."
            )
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{clip_id}-v{version}.mp4",
        )

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
        account = (
            database.get_account_by_id(
                app_settings.database_path,
                request.platform,
                request.account_id,
                project["owner_id"],
            )
            if request.account_id
            else database.get_connected_account(
                app_settings.database_path,
                request.platform,
                project["owner_id"],
            )
        )
        if account is None:
            raise HTTPException(
                status_code=409,
                detail=f"Conecta una cuenta de {request.platform} desde Ajustes antes de publicar.",
            )
        clips_by_id = {clip["id"]: clip for clip in project.get("clips") or []}
        if any(clip_id not in clips_by_id for clip_id in request.clip_ids):
            raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")
        if any(clips_by_id[clip_id].get("status") == "failed" for clip_id in request.clip_ids):
            raise HTTPException(
                status_code=422,
                detail="No se pueden publicar clips cuyo renderizado falló.",
            )
        for clip_id in request.clip_ids:
            clip_file(project_id, clip_id)
        scheduled_at = database.utc_now()
        jobs = database.enqueue_schedule_jobs(
            app_settings.database_path,
            project_id,
            request.clip_ids,
            request.platform,
            scheduled_at,
            request.visibility,
            account_id=request.account_id,
        )
        return {
            "published": [],
            "failed": None,
            "not_attempted": [],
            "jobs": [
                serialize_schedule_job(job)
                for job in jobs
                if job["clip_id"] in request.clip_ids
            ],
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

    @application.get("/api/projects/{project_id}/original/download")
    def download_original(project_id: str) -> FileResponse:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        source = app_settings.data_dir / "projects" / project_id / "source"
        if not source.is_file():
            raise HTTPException(status_code=404, detail="El video original ya no está disponible.")
        title = Path(project["title"]).name
        filename = title if Path(title).suffix else f"{title}.mp4"
        filename = "".join(
            "_" if character in '<>:"/\\|?*' or ord(character) < 32 else character
            for character in filename
        ).strip(" .")
        return FileResponse(
            source,
            media_type="application/octet-stream",
            filename=filename or f"{project_id}.mp4",
        )

    @application.post("/api/projects/{project_id}/downloads/save")
    def save_project_files(
        project_id: str,
        request: SaveProjectFilesRequest,
    ) -> dict[str, list[str]]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        destination = Path(request.destination).expanduser()
        try:
            destination = destination.resolve(strict=True)
        except OSError as error:
            raise HTTPException(
                status_code=422,
                detail="La carpeta elegida no existe o no está disponible.",
            ) from error
        if not destination.is_dir():
            raise HTTPException(status_code=422, detail="El destino debe ser una carpeta.")

        sources: list[tuple[Path, str]] = []
        if request.include_original:
            source = app_settings.data_dir / "projects" / project_id / "source"
            if not source.is_file():
                raise HTTPException(status_code=404, detail="El video original ya no está disponible.")
            title = Path(project["title"]).name
            original_name = title if Path(title).suffix else f"{title}.mp4"
            sources.append((source, original_name))

        clips_by_id = {clip["id"]: clip for clip in project.get("clips") or []}
        missing_ids = [clip_id for clip_id in request.clip_ids if clip_id not in clips_by_id]
        if missing_ids:
            raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")
        for clip_id in request.clip_ids:
            sources.append((clip_file(project_id, clip_id), f"{clip_id}.mp4"))

        saved: list[str] = []
        temporary_files: list[Path] = []
        try:
            for source, requested_name in sources:
                if not source.is_file():
                    raise HTTPException(status_code=404, detail="Uno de los archivos ya no está disponible.")
                safe_name = "".join(
                    "_" if character in '<>:"/\\|?*' or ord(character) < 32 else character
                    for character in Path(requested_name).name
                ).strip(" .") or f"{project_id}.mp4"
                stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
                output = destination / safe_name
                index = 2
                while output.exists():
                    output = destination / f"{stem} ({index}){suffix}"
                    index += 1
                with tempfile.NamedTemporaryFile(
                    prefix=".kellyclipai-", suffix=".tmp", dir=destination, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                temporary_files.append(temporary_path)
                shutil.copyfile(source, temporary_path)
                temporary_path.replace(output)
                temporary_files.remove(temporary_path)
                saved.append(output.name)
        except HTTPException:
            for temporary_path in temporary_files:
                temporary_path.unlink(missing_ok=True)
            for saved_name in saved:
                (destination / saved_name).unlink(missing_ok=True)
            raise
        except OSError as error:
            for temporary_path in temporary_files:
                temporary_path.unlink(missing_ok=True)
            for saved_name in saved:
                (destination / saved_name).unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"No se pudieron guardar los archivos en la carpeta elegida: {error}",
            ) from error
        return {"saved": saved}

    @application.post(
        "/api/projects/{project_id}/clips/{clip_id}/share", status_code=201
    )
    def create_clip_share(project_id: str, clip_id: str) -> dict[str, str]:
        clip_file(project_id, clip_id)
        token = database.create_or_get_clip_share(
            app_settings.database_path,
            project_id,
            clip_id,
            secrets.token_urlsafe(24),
        )
        share = database.get_clip_share(app_settings.database_path, token)
        if share is None:
            raise HTTPException(
                status_code=500, detail="No se pudo crear el enlace compartido."
            )
        return {
            "share_url": f"/share/{token}",
            "expires_at": share["expires_at"],
        }

    @application.delete("/api/projects/{project_id}/clips/{clip_id}/share")
    def revoke_clip_share_route(
        project_id: str,
        clip_id: str,
        request: Request,
    ) -> dict[str, bool]:
        project = database.get_project(
            app_settings.database_path,
            project_id,
            request_user_id(request),
        )
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if not any(
            str(clip.get("id")) == clip_id for clip in project.get("clips") or []
        ):
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        if not database.revoke_clip_share(
            app_settings.database_path, project_id, clip_id
        ):
            raise HTTPException(
                status_code=404,
                detail="No hay un enlace compartido activo para este clip.",
            )
        return {"revoked": True}

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

    # --- Editor visual ShortStudio (v1.3.0) ---------------------------------
    # El editor reproduce siempre el video fuente (GET /source) y edita
    # sobre él en el navegador (los cues van en tiempo absoluto del
    # fuente); el backend solo sirve el video, los cues de subtítulos y
    # guarda la especificación y el video exportado. La
    # autenticación y la propiedad del proyecto las verifica el middleware
    # (404 si el proyecto no existe o es de otro usuario), igual que en
    # el resto de rutas /api/projects/{project_id}/...; el clip se
    # resuelve aquí con 404 propio.

    def studio_project_clip(project_id: str, clip_id: str) -> tuple[dict, dict]:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        clip = find_clip(project, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        return project, clip

    @application.get("/api/projects/{project_id}/source")
    def stream_project_source(project_id: str) -> FileResponse:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        source_path = app_settings.data_dir / "projects" / project_id / "source"
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="El video original ya no está disponible.")
        # FileResponse de Starlette resuelve las peticiones Range (206)
        # que el elemento <video> usa para buscar. El archivo "source"
        # no tiene extensión, así que se sirve como video/mp4 (la gran
        # mayoría de las fuentes lo son: yt-dlp fusiona a mp4) y el
        # navegador detecta el contenedor real.
        return FileResponse(source_path, media_type="video/mp4")

    @application.get("/api/projects/{project_id}/clips/{clip_id}/cues")
    def clip_cues(project_id: str, clip_id: str) -> dict:
        _, clip = studio_project_clip(project_id, clip_id)
        return {"cues": words_to_cues(clip_transcript_words(clip))}

    @application.get("/api/projects/{project_id}/cues")
    def project_cues(
        project_id: str,
        start: float = Query(...),
        end: float = Query(...),
    ) -> dict:
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if start < 0 or end <= start:
            raise HTTPException(
                status_code=400,
                detail="El rango no es válido: el inicio debe ser mayor o igual que cero y el fin posterior al inicio.",
            )
        duration = project_duration(project)
        if duration is not None and end > duration:
            raise HTTPException(
                status_code=400,
                detail="El fin del rango supera la duración del video.",
            )
        words = project_transcript_words(project.get("transcript"), start, end)
        return {"cues": words_to_cues(words)}

    @application.get("/api/projects/{project_id}/waveform")
    def project_waveform(
        project_id: str,
        start: float = Query(0.0, ge=0),
        end: float | None = Query(None),
        buckets: int = Query(1200, ge=10, le=4000),
    ) -> dict:
        """Picos de la forma de onda del video fuente (Fase 3, ítem 4b).

        El navegador dibuja las barras en la línea de tiempo del Studio
        sin decodificar el audio de un video largo. Los picos están
        normalizados a 0–1 por el máximo del rango.
        """
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        source = app_settings.data_dir / "projects" / project_id / "source"
        if not source.is_file():
            raise HTTPException(
                status_code=404, detail="No se encontró el video original."
            )
        duration = project_duration(project)
        resolved_end = end if end is not None else duration
        if resolved_end is None or resolved_end <= start:
            raise HTTPException(
                status_code=400,
                detail="El rango no es válido: el fin debe ser posterior al inicio.",
            )
        if duration is not None and resolved_end > duration:
            raise HTTPException(
                status_code=400,
                detail="El fin del rango supera la duración del video.",
            )
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise HTTPException(
                status_code=503, detail="FFmpeg no está disponible."
            )
        peaks = waveform.compute_waveform_peaks(
            source, start, resolved_end - start, ffmpeg_path, buckets=buckets
        )
        return {
            "start": start,
            "end": resolved_end,
            "buckets": buckets,
            "peaks": peaks,
        }

    @application.get(
        "/api/projects/{project_id}/face-track",
        response_model=FaceTrackResponse,
        responses={202: {"model": FaceTrackPendingResponse}},
    )
    def project_face_track(
        project_id: str,
        ratio: str = Query(default="9:16"),
    ) -> FaceTrackResponse | JSONResponse:
        """Pista de seguimiento facial del video fuente (Studio, v1.3.1).

        Cubre el video fuente completo a 2 fps (el Studio reproduce el
        fuente). Los keyframes llevan tiempos absolutos y x/y como
        fracción del desplazamiento máximo de recorte en cada eje. Sin
        rostro en todo el video devuelve has_face=false y keyframes=[]
        (no es un error).

        Asíncrono (v1.3.2): el análisis recorre el video entero y
        tarda minutos en videos largos, así que la primera llamada
        lanza un trabajo en segundo plano y responde 202 con el
        progreso; el frontend sondea el mismo endpoint hasta recibir
        200 (o 503 si el análisis falla, con reintento en el siguiente
        GET). Varias peticiones del mismo proyecto y ratio comparten
        un solo trabajo (sin análisis duplicados). Con la caché
        caliente (mismo proyecto y ratio) la respuesta es 200
        inmediata. La caché en memoria se pierde al reiniciar el
        backend.
        """
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if ratio not in STUDIO_FACE_TRACK_SIZES:
            raise HTTPException(
                status_code=400,
                detail="El ratio no es válido: usa 9:16, 1:1 o 16:9.",
            )
        source_path = app_settings.data_dir / "projects" / project_id / "source"
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="El video original ya no está disponible.")
        duration = _probe_source_duration(source_path) or project_duration(project)
        if duration is None or duration <= 0:
            raise HTTPException(
                status_code=503,
                detail="No se pudo determinar la duración del video original.",
            )
        video_duration = duration
        output_width, output_height = STUDIO_FACE_TRACK_SIZES[ratio]
        # Intervalos de habla de la transcripción del proyecto (espejo
        # de render_clip, media.py): activan la selección del orador
        # por apertura de labios. Sin transcripción, None.
        words = project_transcript_words(project.get("transcript"), 0.0, video_duration)
        speech_intervals = (
            [(float(word["start"]), float(word["end"])) for word in words] or None
        )

        def _response(
            has_face: bool, keyframes: list[FaceCropKeyframe]
        ) -> FaceTrackResponse:
            return FaceTrackResponse(
                project_id=project_id,
                ratio=ratio,  # type: ignore[arg-type]  # validado arriba
                duration=video_duration,
                sample_fps=STUDIO_FACE_TRACK_SAMPLE_RATE,
                has_face=has_face,
                keyframes=[
                    FaceTrackKeyframe(time=point.time, x=point.x, y=point.y)
                    for point in keyframes
                ],
            )

        # Caché caliente: respuesta inmediata sin lanzar ningún trabajo.
        cached = peek_cached_source_face_track(
            source_path,
            video_duration,
            output_width,
            output_height,
            speech_intervals=speech_intervals,
        )
        if cached is not None:
            has_face, keyframes = cached
            return _response(has_face, keyframes)
        try:
            stat = source_path.stat()
        except OSError:
            raise HTTPException(
                status_code=404, detail="El video original ya no está disponible."
            ) from None
        # La clave incluye la identidad del archivo fuente: reemplazar
        # el video invalida el trabajo anterior.
        job_key = (project_id, ratio, stat.st_size, stat.st_mtime_ns)

        def _run(
            on_progress: Callable[[float], None],
        ) -> tuple[bool, list[FaceCropKeyframe]]:
            return analyze_source_face_track(
                source_path,
                video_duration,
                output_width,
                output_height,
                speech_intervals=speech_intervals,
                on_progress=on_progress,
            )

        job = start_face_track_job(job_key, _run)
        if job.status == "done" and job.result is not None:
            has_face, keyframes = job.result
            return _response(has_face, keyframes)
        if job.status == "failed":
            # Se olvida el trabajo para que el siguiente GET reintente.
            forget_face_track_job(job_key)
            raise HTTPException(
                status_code=503,
                detail=job.error or "No se pudo calcular el seguimiento facial.",
            )
        return JSONResponse(
            status_code=202,
            content=FaceTrackPendingResponse(progress=job.progress).model_dump(
                mode="json"
            ),
        )

    @application.put("/api/projects/{project_id}/clips/{clip_id}/studio")
    def save_clip_studio_edit(
        project_id: str,
        clip_id: str,
        payload: StudioEditRequest,
    ) -> dict:
        project, clip = studio_project_clip(project_id, clip_id)
        clip["studio_edit"] = payload.studio_edit
        database.update_project(
            app_settings.database_path,
            project_id,
            clips=project.get("clips") or [],
        )
        return _serialize_clip(project_id, clip)

    @application.post("/api/projects/{project_id}/studio/export")
    async def export_studio_clip(
        project_id: str,
        response: Response,
        file: Annotated[UploadFile, File(...)],
        clip_id: str | None = Form(default=None),
        start: float = Form(...),
        end: float = Form(...),
        title: str | None = Form(default=None),
        studio_edit: str | None = Form(default=None),
    ) -> dict:
        """Guarda el video exportado por el editor visual.

        Con ``clip_id`` reemplaza el archivo de ese clip (mismo nombre,
        sustitución atómica, sin huérfanos) y actualiza sus datos; sin
        él crea un clip manual nuevo. El transcript del clip resultante
        se reconstruye desde la transcripción del proyecto en el rango
        [start, end) para que los cues sigan siendo válidos (tiempos
        absolutos + locales, misma forma que ``render_clip``); si el
        proyecto no tiene transcripción en ese rango, queda vacío.
        """
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if start < 0 or end <= start:
            raise HTTPException(
                status_code=400,
                detail="El rango no es válido: el inicio debe ser mayor o igual que cero y el fin posterior al inicio.",
            )
        parsed_studio_edit: dict[str, Any] | None = None
        if studio_edit is not None and studio_edit.strip():
            try:
                parsed = json.loads(studio_edit)
            except ValueError as error:
                raise HTTPException(
                    status_code=422, detail="studio_edit debe ser un objeto JSON válido."
                ) from error
            if not isinstance(parsed, dict):
                raise HTTPException(status_code=422, detail="studio_edit debe ser un objeto JSON.")
            parsed_studio_edit = parsed
        existing_clip: dict[str, Any] | None = None
        if clip_id:
            existing_clip = find_clip(project, clip_id)
            if existing_clip is None:
                raise HTTPException(status_code=404, detail="No se encontró el clip.")
        content_type = (file.content_type or "").lower().split(";")[0].strip()
        if content_type.startswith("video/"):
            suffix = _STUDIO_EXPORT_SUFFIXES.get(content_type, ".mp4")
        else:
            # Sin tipo de contenido video/* (algunos clientes no lo
            # envían), se acepta por la extensión del nombre subido.
            fallback = Path(file.filename or "").suffix.lower()
            suffix = fallback if fallback in _VIDEO_SUFFIXES else None
        if suffix is None:
            raise HTTPException(status_code=415, detail="Solo se admiten archivos de video.")

        clips_dir = app_settings.data_dir / "projects" / project_id / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clips_root = clips_dir.resolve()
        if existing_clip is not None:
            filename = str(existing_clip.get("filename") or f"{clip_id}.mp4")
            target = (clips_dir / Path(filename).name).resolve()
            if not target.is_relative_to(clips_root):
                raise HTTPException(
                    status_code=422,
                    detail="El nombre de archivo guardado del clip no es válido.",
                )
            created = False
        else:
            new_clip_id = str(uuid.uuid4())
            filename = f"{new_clip_id}{suffix}"
            target = clips_dir / filename
            created = True

        # Escritura en streaming a un temporal y sustitución atómica: el
        # archivo viejo solo se reemplaza cuando el nuevo está completo
        # y dentro del tope (el mismo MAX_UPLOAD_BYTES de las subidas).
        temp_path = clips_dir / f".studio-{uuid.uuid4().hex}.tmp"
        written = 0
        try:
            with temp_path.open("wb") as target_file:
                while True:
                    chunk = await file.read(_STAGED_WRITE_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "El video exportado supera el límite de "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024 * 1024)} GB por proyecto."
                            ),
                        )
                    target_file.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail="El archivo de video está vacío.")
            if existing_clip is not None:
                # Versiones (Fase 3, ítem 6): el archivo vigente se
                # archiva antes de sustituirlo. Si el archivado falla,
                # no se sobrescribe nada (mejor fallar que perderlo).
                try:
                    clip_versions.archive_version(
                        app_settings.data_dir / "projects" / project_id,
                        clips_dir,
                        str(clip_id),
                        filename,
                    )
                except OSError as error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "No se pudo archivar la versión anterior del "
                            f"clip: {error}"
                        ),
                    ) from error
            temp_path.replace(target)
        finally:
            temp_path.unlink(missing_ok=True)

        dimensions = _probe_clip_dimensions(target)
        transcript = local_transcript_words(
            project_transcript_words(project.get("transcript"), start, end),
            start,
            end,
        )
        clean_title = title.strip()[:200] if title and title.strip() else None
        clips = project.get("clips") or []
        if existing_clip is not None:
            existing_clip.update(
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "status": "ready",
                    "transcript": transcript,
                }
            )
            existing_clip.pop("error", None)
            if clean_title is not None:
                existing_clip["title"] = clean_title
            if parsed_studio_edit is not None:
                existing_clip["studio_edit"] = parsed_studio_edit
            if dimensions is not None:
                existing_clip["width"], existing_clip["height"] = dimensions
            clip = existing_clip
        else:
            clip = {
                "id": new_clip_id,
                "title": clean_title or f"Clip manual {start:.1f}s - {end:.1f}s",
                "start": start,
                "end": end,
                "duration": end - start,
                "score": None,
                "viral_reason": "",
                "description": "",
                "hashtags": [],
                "keywords": [],
                "emojis": [],
                "resolution": 720,
                "filename": filename,
                "transcript": transcript,
                "width": dimensions[0] if dimensions else 0,
                "height": dimensions[1] if dimensions else 0,
                "status": "ready",
                "studio_edit": parsed_studio_edit,
                "manual": True,
            }
            clips.append(clip)
        database.update_project(app_settings.database_path, project_id, clips=clips)
        response.status_code = 201 if created else 200
        return _serialize_clip(project_id, clip)

    # ------------------------------------------------------------------
    # Publicación y flujo (Fase 3, ítems 9–15)
    # ------------------------------------------------------------------

    async def _enqueue_created_project(project_id: str) -> None:
        """Encola un proyecto ya creado (lote, duplicado, carpeta).

        Usa el mismo camino que el worker de fondo cuando el lifespan ya
        arrancó; si la app se construyó sin lifespan (tests de endpoints
        sin contexto), usa la cola directamente si existe.
        """
        submit_background = getattr(
            application.state, "submit_background_project", None
        )
        if submit_background is not None:
            await submit_background(project_id)
            return
        processing_queue: ProjectProcessingQueue | None = getattr(
            application.state, "processing_queue", None
        )
        if processing_queue is None:
            raise RuntimeError("La cola de procesamiento no está disponible.")
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise RuntimeError(f"No se encontró el proyecto {project_id}.")
        if not database.queue_project(
            app_settings.database_path,
            project_id,
            project["config"],
        ):
            raise RuntimeError(f"El proyecto {project_id} ya se está procesando.")
        await processing_queue.submit(project_id, None)

    @application.get("/api/publish/best-times")
    def publish_best_times(platform: str | None = None) -> dict[str, Any]:
        """Guía de mejor hora por plataforma (orientativa, no datos de
        audiencia) + próximas franjas sugeridas."""
        if platform is not None and platform not in publish_guide.BEST_TIMES_GUIDE:
            raise HTTPException(status_code=404, detail="No se reconoce esa plataforma.")
        guide = publish_guide.best_time_suggestions(platform) if platform else {
            "platforms": publish_guide.BEST_TIMES_GUIDE,
            "note": publish_guide.BEST_TIMES_NOTE,
        }
        slots = (
            {platform: publish_guide.next_best_slots(platform)}
            if platform
            else {
                name: publish_guide.next_best_slots(name)
                for name in publish_guide.BEST_TIMES_GUIDE
            }
        )
        return {**guide, "next_slots": slots}

    @application.get("/api/integrations/accounts")
    def list_publish_accounts(request: Request) -> dict[str, list[dict[str, Any]]]:
        return list_account_summaries(
            app_settings.database_path,
            request_user_id(request),
        )

    @application.post("/api/integrations/{provider}/accounts/{account_id}/set-default")
    def set_default_publish_account(
        provider: str,
        account_id: str,
        request: Request,
    ) -> dict[str, list[dict[str, Any]]]:
        if provider not in {"youtube", "tiktok", "instagram"}:
            raise HTTPException(status_code=404, detail="No se reconoce esa integración.")
        if not database.set_default_account(
            app_settings.database_path,
            provider,
            account_id,
            request_user_id(request),
        ):
            raise HTTPException(
                status_code=404,
                detail="No se encontró esa cuenta o ya es la predeterminada.",
            )
        return list_publish_accounts(request)

    @application.delete(
        "/api/integrations/{provider}/accounts/{account_id}",
        status_code=204,
    )
    def delete_publish_account(
        provider: str,
        account_id: str,
        request: Request,
    ) -> Response:
        if provider not in {"youtube", "tiktok", "instagram", "zoom"}:
            raise HTTPException(status_code=404, detail="No se reconoce esa integración.")
        if not database.delete_account_by_id(
            app_settings.database_path,
            provider,
            account_id,
            request_user_id(request),
        ):
            raise HTTPException(status_code=404, detail="No se encontró esa cuenta.")
        return Response(status_code=204)

    @application.post("/api/projects/{project_id}/cross-publish", status_code=201)
    def cross_publish_project_clips(
        project_id: str,
        request: CrossPublishRequest,
        http_request: Request,
    ) -> dict[str, Any]:
        """Publicación cruzada (ítem 10): los mismos clips a varias
        plataformas y cuentas a la vez. Crea un job por destino, todos
        con el mismo group_id. Sin scheduled_at, el destino se publica
        cuanto antes."""
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] != "completed":
            raise HTTPException(status_code=409, detail="El proyecto aún no tiene clips listos para publicar.")
        clips_by_id = {clip["id"]: clip for clip in project.get("clips") or []}
        if any(clip_id not in clips_by_id for clip_id in request.clip_ids):
            raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")
        if any(clips_by_id[clip_id].get("status") == "failed" for clip_id in request.clip_ids):
            raise HTTPException(
                status_code=422,
                detail="No se pueden publicar clips cuyo renderizado falló.",
            )
        for clip_id in request.clip_ids:
            clip_file(project_id, clip_id)
        owner_id = str(project["owner_id"])
        for target in request.targets:
            missing = provider_missing_credentials(target.platform, app_settings)
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Faltan credenciales en .env para {target.platform}: {', '.join(missing)}.",
                )
            account = (
                database.get_account_by_id(
                    app_settings.database_path,
                    target.platform,
                    target.account_id,
                    owner_id,
                )
                if target.account_id
                else database.get_connected_account(
                    app_settings.database_path,
                    target.platform,
                    owner_id,
                )
            )
            if account is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Conecta una cuenta de {target.platform} desde Ajustes antes de publicar.",
                )
            if (
                target.scheduled_at is not None
                and target.scheduled_at < datetime.now(UTC) - timedelta(minutes=1)
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"La fecha programada para {target.platform} ya pasó.",
                )
        group_id = str(uuid.uuid4())
        jobs: list[dict[str, Any]] = []
        # La unicidad (proyecto, clip, plataforma, hora) impide dos jobs
        # idénticos: dos cuentas de la misma plataforma a la misma hora
        # se escalonan 1 s (mismo minuto, distinto segundo).
        used_slots: set[tuple[str, str]] = set()
        for target in request.targets:
            scheduled_at = (
                target.scheduled_at.isoformat()
                if target.scheduled_at is not None
                else database.utc_now()
            )
            base = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
            while (target.platform, scheduled_at) in used_slots:
                base += timedelta(seconds=1)
                scheduled_at = base.isoformat()
            used_slots.add((target.platform, scheduled_at))
            created = database.enqueue_schedule_jobs(
                app_settings.database_path,
                project_id,
                request.clip_ids,
                target.platform,
                scheduled_at,
                target.visibility,
                account_id=target.account_id,
                group_id=group_id,
            )
            jobs.extend(
                serialize_schedule_job(job)
                for job in created
                if job["clip_id"] in request.clip_ids
            )
        return {"group_id": group_id, "jobs": jobs}

    # Ruta fuera de /api/scheduled-jobs/*: el middleware de autenticación
    # trata el tercer segmento de esa ruta como un job_id y devolvería
    # 404 para "groups".
    @application.get("/api/schedule-groups/{group_id}")
    def scheduled_jobs_group(group_id: str, request: Request) -> list[dict]:
        return [
            serialize_schedule_job(job)
            for job in database.list_schedule_jobs_by_group(
                app_settings.database_path,
                group_id,
                request_user_id(request),
            )
        ]

    @application.post("/api/scheduled-jobs/{job_id}/reschedule")
    def reschedule_scheduled_job(
        job_id: str,
        payload: RescheduleRequest,
        request: Request,
    ) -> dict:
        job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        if job.get("confirmation_required"):
            raise HTTPException(
                status_code=409,
                detail="Esta publicación vencida está pendiente de confirmación; resuélvela primero.",
            )
        if payload.scheduled_at < datetime.now(UTC) - timedelta(minutes=1):
            raise HTTPException(status_code=422, detail="La fecha programada ya pasó.")
        if not database.reschedule_schedule_job(
            app_settings.database_path,
            job_id,
            payload.scheduled_at.isoformat(),
        ):
            raise HTTPException(
                status_code=409,
                detail="Solo se puede reprogramar una publicación pendiente.",
            )
        updated = database.get_schedule_job(app_settings.database_path, job_id)
        if updated is None:  # pragma: no cover - se borró entre medias
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        return serialize_schedule_job(updated)

    @application.post("/api/scheduled-jobs/{job_id}/retry")
    def retry_scheduled_job(job_id: str, request: Request) -> dict:
        job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        if not database.retry_schedule_job(app_settings.database_path, job_id):
            raise HTTPException(
                status_code=409,
                detail="Solo se puede reintentar una publicación fallida.",
            )
        updated = database.get_schedule_job(app_settings.database_path, job_id)
        if updated is None:  # pragma: no cover - se borró entre medias
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        return serialize_schedule_job(updated)

    @application.delete("/api/scheduled-jobs/{job_id}", status_code=204)
    def cancel_scheduled_job(job_id: str, request: Request) -> Response:
        job = database.get_schedule_job(
            app_settings.database_path,
            job_id,
            request_user_id(request),
        )
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró la publicación.")
        if not database.cancel_schedule_job(app_settings.database_path, job_id):
            raise HTTPException(
                status_code=409,
                detail="Solo se puede cancelar una publicación pendiente.",
            )
        return Response(status_code=204)

    @application.get("/api/queue")
    def global_processing_queue(request: Request) -> list[dict[str, Any]]:
        """Vista global de la cola de procesamiento (ítem 11)."""
        return database.list_processing_jobs(
            app_settings.database_path,
            request_user_id(request),
        )

    @application.post("/api/queue/batch", status_code=201)
    async def enqueue_batch(
        payload: BatchEnqueueRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> dict[str, Any]:
        """Encola varios videos a la vez (ítem 11): staged listos se
        convierten en proyecto y se encolan ahora; enlaces se descargan
        en segundo plano y el worker los convierte al quedar listos."""
        owner_id = request_user_id(request)
        config = (
            payload.config.model_dump(mode="json")
            if payload.config is not None
            else ProjectConfig().model_dump(mode="json")
        )
        batch_id = str(uuid.uuid4())
        items: list[dict[str, Any]] = []
        for staged_id in payload.staged_ids:
            staged = database.get_staged_media(app_settings.database_path, staged_id)
            if staged is None:
                items.append({"staged_id": staged_id, "status": "failed", "error": "No se encontró la importación."})
                continue
            if staged.get("owner_id") not in (None, owner_id):
                items.append({"staged_id": staged_id, "status": "failed", "error": "Esta importación pertenece a otra cuenta."})
                continue
            if staged["status"] == "completed" and staged.get("project_id"):
                items.append({"staged_id": staged_id, "status": "done", "project_id": str(staged["project_id"])})
                continue
            if staged["status"] != "ready":
                items.append({"staged_id": staged_id, "status": "failed", "error": "El archivo aún no está listo para crear el proyecto."})
                continue
            try:
                project_id = await asyncio.to_thread(
                    watch_folder.create_project_from_ready_staged,
                    app_settings,
                    staged_id,
                )
            except Exception as error:
                items.append({"staged_id": staged_id, "status": "failed", "error": str(error)})
                continue
            if project_id is None:
                items.append({"staged_id": staged_id, "status": "failed", "error": "La importación cambió mientras se procesaba el lote."})
                continue
            if payload.config is not None:
                database.update_project(
                    app_settings.database_path,
                    project_id,
                    config=config,
                )
            try:
                await _enqueue_created_project(project_id)
            except RuntimeError as error:
                items.append({"staged_id": staged_id, "status": "failed", "project_id": project_id, "error": str(error)})
                continue
            items.append({"staged_id": staged_id, "status": "done", "project_id": project_id})
        (app_settings.data_dir / "staging").mkdir(parents=True, exist_ok=True)
        for url in payload.urls:
            try:
                validated_url = _validate_download_url(url)
            except HTTPException as error:
                items.append({"url": url, "status": "failed", "error": str(error.detail)})
                continue
            staged_id = str(uuid.uuid4())
            database.create_staged_media(
                app_settings.database_path,
                staged_id,
                "url",
                "downloading",
                "Descarga de enlace",
                config=config,
                owner_id=owner_id,
            )
            database.create_batch_item(
                app_settings.database_path,
                batch_id,
                owner_id,
                staged_id=staged_id,
                url=url,
                status="downloading",
            )
            background_tasks.add_task(
                _download_staged_media,
                app_settings,
                staged_id,
                validated_url,
            )
            items.append({"url": url, "staged_id": staged_id, "status": "downloading"})
        return {"batch_id": batch_id, "items": items}

    @application.get("/api/queue/batch")
    def list_batch_items(request: Request) -> list[dict[str, Any]]:
        return database.list_batch_items(
            app_settings.database_path,
            request_user_id(request),
        )

    @application.get("/api/watch-folders")
    def list_watch_folders(request: Request) -> list[dict[str, Any]]:
        return database.list_watch_folders(
            app_settings.database_path,
            request_user_id(request),
        )

    @application.post("/api/watch-folders", status_code=201)
    def create_watch_folder(
        payload: WatchFolderRequest,
        request: Request,
    ) -> dict[str, Any]:
        folder_path = Path(payload.path).expanduser()
        if not folder_path.is_dir():
            raise HTTPException(
                status_code=422,
                detail="La carpeta observada no existe o no es una carpeta.",
            )
        config = (
            payload.config.model_dump(mode="json")
            if payload.config is not None
            else ProjectConfig().model_dump(mode="json")
        )
        try:
            return database.create_watch_folder(
                app_settings.database_path,
                request_user_id(request),
                str(folder_path),
                enabled=payload.enabled,
                config=config,
                poll_seconds=payload.poll_seconds,
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=409,
                detail="Esa carpeta ya está observada.",
            ) from error

    @application.put("/api/watch-folders/{folder_id}")
    def update_watch_folder(
        folder_id: str,
        payload: WatchFolderUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        folder = database.get_watch_folder(
            app_settings.database_path,
            folder_id,
            owner_id,
        )
        if folder is None:
            raise HTTPException(status_code=404, detail="No se encontró la carpeta observada.")
        updates: dict[str, Any] = {}
        if payload.enabled is not None:
            updates["enabled"] = payload.enabled
        if payload.config is not None:
            updates["config"] = payload.config.model_dump(mode="json")
        if payload.poll_seconds is not None:
            updates["poll_seconds"] = payload.poll_seconds
        if updates:
            database.update_watch_folder(
                app_settings.database_path,
                folder_id,
                owner_id,
                **updates,
            )
        updated = database.get_watch_folder(
            app_settings.database_path,
            folder_id,
            owner_id,
        )
        assert updated is not None
        return updated

    @application.delete("/api/watch-folders/{folder_id}", status_code=204)
    def delete_watch_folder(folder_id: str, request: Request) -> Response:
        if not database.delete_watch_folder(
            app_settings.database_path,
            folder_id,
            request_user_id(request),
        ):
            raise HTTPException(status_code=404, detail="No se encontró la carpeta observada.")
        return Response(status_code=204)

    @application.post("/api/watch-folders/{folder_id}/scan")
    async def scan_watch_folder_now(folder_id: str, request: Request) -> dict[str, Any]:
        owner_id = request_user_id(request)
        folder = database.get_watch_folder(
            app_settings.database_path,
            folder_id,
            owner_id,
        )
        if folder is None:
            raise HTTPException(status_code=404, detail="No se encontró la carpeta observada.")
        created = await watch_folder.scan_folder_once(
            app_settings,
            folder,
            _enqueue_created_project,
        )
        return {"scanned": True, "created": created}

    @application.get("/api/preferences/night-window")
    def get_night_window_preference() -> dict[str, Any]:
        return watch_folder.get_night_window(app_settings.database_path)

    @application.put("/api/preferences/night-window")
    def set_night_window_preference(payload: NightWindowRequest) -> dict[str, Any]:
        try:
            publish_guide.parse_hhmm(payload.start)
            publish_guide.parse_hhmm(payload.end)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return watch_folder.set_night_window(
            app_settings.database_path,
            enabled=payload.enabled,
            start=payload.start,
            end=payload.end,
        )

    @application.post("/api/projects/{project_id}/generate-captions")
    def generate_captions(
        project_id: str,
        payload: CaptionGenerateRequest,
    ) -> dict[str, Any]:
        """Genera 3 variantes de título, descripción y hashtags por clip
        (ítem 12). Requiere LLM: sin clave configurada responde 409."""
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(
                status_code=409,
                detail="Espera a que termine el procesamiento antes de generar títulos.",
            )
        clips = [
            clip
            for clip in project.get("clips") or []
            if isinstance(clip, dict) and clip.get("status") != "failed"
        ]
        if payload.clip_ids is not None:
            wanted = set(payload.clip_ids)
            clips = [clip for clip in clips if str(clip.get("id")) in wanted]
            if len(clips) != len(wanted):
                raise HTTPException(status_code=404, detail="Uno o más clips seleccionados no existen.")
        if not clips:
            raise HTTPException(
                status_code=409,
                detail="Este proyecto no tiene clips listos para generar títulos.",
            )
        try:
            llm = _resolve_llm_for_project(project, None, app_settings)
        except (LLMConfigurationError, CredentialStoreError, ValidationError) as error:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Generar títulos necesita IA: configura una clave en "
                    "Ajustes (Groq es gratis) para usarlo."
                ),
            ) from error
        try:
            variants = captions.generate_caption_variants(llm, clips)
        except LLMResponseError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        metadata = dict(project.get("metadata") or {})
        stored = dict(metadata.get("caption_variants") or {})
        stored.update(variants)
        metadata["caption_variants"] = stored
        database.update_project(
            app_settings.database_path,
            project_id,
            metadata=metadata,
        )
        return {"variants": variants}

    @application.put("/api/projects/{project_id}/clips/{clip_id}/caption")
    def apply_clip_caption(
        project_id: str,
        clip_id: str,
        payload: CaptionApplyRequest,
    ) -> dict[str, Any]:
        """Guarda la edición del usuario (título/descripción/hashtags
        elegidos) en el clip antes de publicar."""
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        clips = list(project.get("clips") or [])
        index = next(
            (
                position
                for position, clip in enumerate(clips)
                if isinstance(clip, dict) and str(clip.get("id")) == clip_id
            ),
            None,
        )
        if index is None:
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        clips[index] = captions.apply_caption_to_clip(
            clips[index],
            title=payload.title,
            description=payload.description,
            hashtags=payload.hashtags,
        )
        database.update_project(
            app_settings.database_path,
            project_id,
            clips=clips,
        )
        return _serialize_clip(project_id, clips[index])

    @application.get("/api/broll")
    def list_broll_library(
        request: Request,
        q: str | None = None,
        favorites: bool = False,
    ) -> list[dict[str, Any]]:
        return database.list_broll_items(
            app_settings.database_path,
            request_user_id(request),
            query=q,
            favorites_only=favorites,
        )

    @application.post("/api/broll/search")
    def search_broll_provider(payload: BrollSearchRequest) -> dict[str, Any]:
        try:
            results = search_broll_videos(
                app_settings,
                payload.provider,
                payload.query,
                per_page=payload.per_page,
            )
        except IntegrationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"provider": payload.provider, "results": results}

    @application.post("/api/broll/import", status_code=201)
    def import_broll_item(
        payload: BrollImportRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Descarga un resultado del proveedor a la biblioteca local."""
        parsed = urlparse(payload.download_url)
        allowed_hosts = {
            "pexels": ("pexels.com",),
            "pixabay": ("pixabay.com",),
        }[payload.provider]
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not any(
                parsed.hostname == host or parsed.hostname.endswith(f".{host}")
                for host in allowed_hosts
            )
        ):
            raise HTTPException(
                status_code=422,
                detail="La URL de descarga no pertenece al proveedor indicado.",
            )
        owner_id = request_user_id(request)
        broll_dir = app_settings.data_dir / "broll"
        broll_dir.mkdir(parents=True, exist_ok=True)
        item_id = str(uuid.uuid4())
        target_path = broll_dir / f"{item_id}.mp4"
        written = 0
        try:
            with (
                httpx.Client(
                    timeout=httpx.Timeout(30, read=300), follow_redirects=True
                ) as client,
                client.stream("GET", payload.download_url) as response,
            ):
                if response.is_error:
                    raise HTTPException(
                        status_code=502,
                        detail=f"El proveedor no pudo entregar el video (HTTP {response.status_code}).",
                    )
                with target_path.open("wb") as target:
                    for chunk in response.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > 250 * 1024 * 1024:
                            raise HTTPException(
                                status_code=422,
                                detail="El B-roll supera el límite local de 250 MB.",
                            )
                        target.write(chunk)
        except HTTPException:
            target_path.unlink(missing_ok=True)
            raise
        except httpx.RequestError as error:
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo descargar el B-roll desde {payload.provider.title()}.",
            ) from error
        if written == 0:
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=502,
                detail="El proveedor devolvió un archivo vacío.",
            )
        item = database.create_broll_item(
            app_settings.database_path,
            owner_id,
            title=payload.title,
            query=payload.query,
            provider=payload.provider.title(),
            external_id=payload.external_id or None,
            creator=payload.creator,
            attribution_url=payload.attribution_url,
            preview_url=payload.preview_url,
            local_path=str(target_path),
            duration=payload.duration,
            width=payload.width,
            height=payload.height,
        )
        return item

    @application.put("/api/broll/{item_id}")
    def update_broll_library_item(
        item_id: str,
        payload: BrollUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        item = database.get_broll_item(
            app_settings.database_path,
            item_id,
            owner_id,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="No se encontró ese B-roll.")
        updates: dict[str, Any] = {}
        if payload.title is not None:
            updates["title"] = payload.title
        if payload.favorite is not None:
            updates["favorite"] = payload.favorite
        if updates:
            database.update_broll_item(
                app_settings.database_path,
                item_id,
                owner_id,
                **updates,
            )
        updated = database.get_broll_item(
            app_settings.database_path,
            item_id,
            owner_id,
        )
        assert updated is not None
        return updated

    @application.delete("/api/broll/{item_id}", status_code=204)
    def delete_broll_library_item(item_id: str, request: Request) -> Response:
        owner_id = request_user_id(request)
        item = database.get_broll_item(
            app_settings.database_path,
            item_id,
            owner_id,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="No se encontró ese B-roll.")
        database.delete_broll_item(app_settings.database_path, item_id, owner_id)
        local_path = item.get("local_path")
        if local_path:
            try:
                Path(str(local_path)).unlink(missing_ok=True)
            except OSError:
                pass
        return Response(status_code=204)

    @application.get("/api/broll/{item_id}/file")
    def stream_broll_file(item_id: str, request: Request) -> FileResponse:
        item = database.get_broll_item(
            app_settings.database_path,
            item_id,
            request_user_id(request),
        )
        if item is None or not item.get("local_path"):
            raise HTTPException(status_code=404, detail="No hay un archivo local para este B-roll.")
        broll_root = (app_settings.data_dir / "broll").resolve()
        file_path = Path(str(item["local_path"])).resolve()
        if not file_path.is_relative_to(broll_root) or not file_path.is_file():
            raise HTTPException(status_code=404, detail="El archivo de B-roll ya no está disponible.")
        return FileResponse(file_path, media_type="video/mp4", filename=f"{item_id}.mp4")

    @application.get("/api/projects/{project_id}/broll")
    def list_project_broll(
        project_id: str,
        request: Request,
        clip_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not database.project_exists(app_settings.database_path, project_id):
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        return database.list_broll_segments(
            app_settings.database_path,
            project_id,
            clip_id,
            request_user_id(request),
        )

    @application.post("/api/broll/assign", status_code=201)
    def assign_broll_to_clip(
        payload: BrollAssignRequest,
        request: Request,
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        item = database.get_broll_item(
            app_settings.database_path,
            payload.broll_id,
            owner_id,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="No se encontró ese B-roll.")
        project = database.get_project(app_settings.database_path, payload.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if not any(
            isinstance(clip, dict) and str(clip.get("id")) == payload.clip_id
            for clip in project.get("clips") or []
        ):
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        return database.assign_broll_segment(
            app_settings.database_path,
            owner_id,
            broll_id=payload.broll_id,
            project_id=payload.project_id,
            clip_id=payload.clip_id,
            start=payload.start,
            end=payload.end,
            label=payload.label,
        )

    @application.delete("/api/broll/segments/{segment_id}", status_code=204)
    def delete_broll_assignment(segment_id: str, request: Request) -> Response:
        if not database.delete_broll_segment(
            app_settings.database_path,
            segment_id,
            request_user_id(request),
        ):
            raise HTTPException(status_code=404, detail="No se encontró esa asignación de B-roll.")
        return Response(status_code=204)

    @application.get("/api/saved-moments")
    def list_saved_moments_endpoint(
        request: Request,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        return database.list_saved_moments(
            app_settings.database_path,
            request_user_id(request),
            query=q,
        )

    @application.post("/api/saved-moments", status_code=201)
    def create_saved_moment_endpoint(
        payload: SavedMomentRequest,
        request: Request,
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        if payload.project_id is not None:
            project = database.get_project(app_settings.database_path, payload.project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
            if payload.clip_id is not None and not any(
                isinstance(clip, dict) and str(clip.get("id")) == payload.clip_id
                for clip in project.get("clips") or []
            ):
                raise HTTPException(status_code=404, detail="No se encontró el clip.")
        return database.create_saved_moment(
            app_settings.database_path,
            owner_id,
            title=payload.title,
            description=payload.description,
            project_id=payload.project_id,
            clip_id=payload.clip_id,
            start=payload.start,
            end=payload.end,
            excerpt=payload.excerpt,
            tags=payload.tags,
        )

    @application.put("/api/saved-moments/{moment_id}")
    def update_saved_moment_endpoint(
        moment_id: str,
        payload: SavedMomentUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        owner_id = request_user_id(request)
        moment = database.get_saved_moment(
            app_settings.database_path,
            moment_id,
            owner_id,
        )
        if moment is None:
            raise HTTPException(status_code=404, detail="No se encontró ese momento guardado.")
        updates: dict[str, Any] = {}
        if payload.title is not None:
            updates["title"] = payload.title
        if payload.description is not None:
            updates["description"] = payload.description
        if payload.excerpt is not None:
            updates["excerpt"] = payload.excerpt
        if payload.tags is not None:
            updates["tags"] = payload.tags
        if updates:
            database.update_saved_moment(
                app_settings.database_path,
                moment_id,
                owner_id,
                **updates,
            )
        updated = database.get_saved_moment(
            app_settings.database_path,
            moment_id,
            owner_id,
        )
        assert updated is not None
        return updated

    @application.delete("/api/saved-moments/{moment_id}", status_code=204)
    def delete_saved_moment_endpoint(moment_id: str, request: Request) -> Response:
        if not database.delete_saved_moment(
            app_settings.database_path,
            moment_id,
            request_user_id(request),
        ):
            raise HTTPException(status_code=404, detail="No se encontró ese momento guardado.")
        return Response(status_code=204)

    @application.post("/api/saved-moments/{moment_id}/apply")
    def apply_saved_moment_endpoint(
        moment_id: str,
        payload: ApplySavedMomentRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Aplica un momento guardado a otro proyecto: su tema pasa a ser
        el "momento solicitado" de la detección en ese proyecto."""
        owner_id = request_user_id(request)
        moment = database.get_saved_moment(
            app_settings.database_path,
            moment_id,
            owner_id,
        )
        if moment is None:
            raise HTTPException(status_code=404, detail="No se encontró ese momento guardado.")
        project = database.get_project(app_settings.database_path, payload.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        if project["status"] in {"processing", "queued"}:
            raise HTTPException(
                status_code=409,
                detail="Espera a que termine el procesamiento antes de aplicar el momento.",
            )
        config = dict(project.get("config") or {})
        topic = moment["title"]
        if moment.get("description"):
            topic = f"{topic}: {moment['description']}"
        config["moment"] = topic[:500]
        try:
            validated = ProjectConfig.model_validate(config)
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=f"La configuración del proyecto no admite este momento: {error}",
            ) from error
        database.update_project(
            app_settings.database_path,
            payload.project_id,
            config=validated.model_dump(mode="json"),
        )
        refreshed = database.get_project(app_settings.database_path, payload.project_id)
        assert refreshed is not None
        return _serialize_project(refreshed)

    @application.post("/api/projects/{project_id}/duplicate", status_code=201)
    async def duplicate_project_endpoint(
        project_id: str,
        payload: DuplicateProjectRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Duplica un proyecto sin volver a subir el video (ítem 15) y,
        opcionalmente, lo reprocesa con otra configuración."""
        project = database.get_project(app_settings.database_path, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="No se encontró el proyecto.")
        try:
            new_id = project_clone.duplicate_project(
                app_settings,
                project,
                owner_id=request_user_id(request),
                title=payload.title,
                ratio=payload.ratio,
                duration=payload.duration,
                subtitle_template=payload.subtitle_template,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if payload.process:
            try:
                await _enqueue_created_project(new_id)
            except RuntimeError as error:
                database.update_project(
                    app_settings.database_path,
                    new_id,
                    status="failed",
                    step="failed",
                    error="La cola de procesamiento se cerró antes de aceptar el trabajo.",
                )
                raise HTTPException(
                    status_code=503,
                    detail="El duplicado se creó pero la cola no está disponible. Inténtalo de nuevo.",
                ) from error
        duplicated = database.get_project(app_settings.database_path, new_id)
        if duplicated is None:  # pragma: no cover - se borró entre medias
            raise HTTPException(status_code=404, detail="No se encontró el duplicado.")
        return _serialize_project(duplicated)

    if web_dir is not None:
        application.mount("/", SPAStaticFiles(directory=web_dir, html=True), name="frontend")

    return application


app = create_app()
