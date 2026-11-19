from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from . import database
from .integrations import IntegrationError, provider_missing_credentials, publish_clip
from .settings import AppSettings

_logger = logging.getLogger(__name__)

# Supervisor del programador (Fase 2, B-S7).
_SCHEDULE_JOB_TIMEOUT_SECONDS = 1800.0  # 30 min por publicación: una subida
# de un clip grande a una conexión lenta tarda minutos; 30 min deja margen
# de sobra sin permitir que un job colgado bloquee la cola para siempre.
_SUPERVISOR_INITIAL_BACKOFF_SECONDS = 2.0
_SUPERVISOR_MAX_BACKOFF_SECONDS = 60.0
_POLL_INTERVAL_SECONDS = 2.0


def _export_clip(settings: AppSettings, job: dict[str, Any], clip: dict[str, Any]) -> Path:
    source = (
        settings.data_dir
        / "projects"
        / str(job["project_id"])
        / "clips"
        / str(clip["filename"])
    )
    if not source.is_file():
        raise FileNotFoundError("El MP4 del clip programado ya no está disponible.")

    export_dir = settings.data_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / f"{job['id']}.mp4"
    temporary = export_dir / f"{job['id']}.mp4.tmp"
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size == 0:
            raise OSError("El archivo exportado está vacío.")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def process_schedule_job(settings: AppSettings, job: dict[str, Any]) -> None:
    project = database.get_project(settings.database_path, str(job["project_id"]))
    if project is None:
        raise FileNotFoundError("No se encontró el proyecto de esta publicación programada.")
    clip = next(
        (item for item in project.get("clips") or [] if item["id"] == job["clip_id"]),
        None,
    )
    if clip is None:
        raise FileNotFoundError("No se encontró el clip de esta publicación programada.")

    missing = provider_missing_credentials(str(job["platform"]), settings)
    owner_id = str(project["owner_id"])
    account = database.get_connected_account(
        settings.database_path,
        str(job["platform"]),
        owner_id,
    )
    if missing or account is None:
        exported = _export_clip(settings, job, clip)
        reason = (
            f"Faltan credenciales: {', '.join(missing)}."
            if missing
            else f"No hay una cuenta de {job['platform']} conectada."
        )
        database.finish_schedule_job(
            settings.database_path,
            str(job["id"]),
            "exported",
            output_path=str(exported),
            error=f"{reason} El MP4 quedó exportado localmente; no se publicó.",
        )
        return

    try:
        result = publish_clip(
            settings,
            str(job["platform"]),
            clip,
            settings.data_dir
            / "projects"
            / str(job["project_id"])
            / "clips"
            / str(clip["filename"]),
            str(job["visibility"]),
            owner_id,
        )
    except IntegrationError as error:
        exported = _export_clip(settings, job, clip)
        database.finish_schedule_job(
            settings.database_path,
            str(job["id"]),
            "failed",
            output_path=str(exported),
            error=f"{error} El MP4 quedó exportado localmente; la publicación no se confirmó.",
        )
        return

    database.finish_schedule_job(
        settings.database_path,
        str(job["id"]),
        "published",
        external_id=result["external_id"],
        external_url=result.get("url") or None,
    )


async def schedule_worker(settings: AppSettings) -> None:
    """Supervisor del programador de publicaciones (Fase 2, B-S7).

    Antes, un error de BD en due_schedule_jobs o en el finish del handler
    mataba el worker en silencio y las publicaciones programadas dejaban
    de ejecutarse hasta reiniciar (B22). Ahora el ciclo completo está
    dentro de un try: ante un error se registra, se espera con backoff
    exponencial (2 s → 60 s) y se reintenta. El backoff se resetea en
    cuanto un ciclo termina bien. La cancelación (apagado) se propaga.
    """
    backoff = _SUPERVISOR_INITIAL_BACKOFF_SECONDS
    while True:
        try:
            await _process_due_jobs(settings)
            backoff = _SUPERVISOR_INITIAL_BACKOFF_SECONDS
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception(
                "El supervisor del programador falló; reintentando en %.0f s.",
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _SUPERVISOR_MAX_BACKOFF_SECONDS)


def _finish_job_failed(settings: AppSettings, job: dict[str, Any], error: str) -> None:
    """Marca un job como fallido sin lanzar nunca (B22).

    Si la BD no responde, el error se registra y el job queda en
    'processing': fail_interrupted_schedule_jobs lo marcará como fallido
    en el próximo arranque.
    """
    try:
        database.finish_schedule_job(
            settings.database_path,
            str(job["id"]),
            "failed",
            error=error,
        )
    except Exception:
        _logger.exception(
            "No se pudo marcar la tarea programada %s como fallida.", job["id"]
        )


async def _process_due_jobs(settings: AppSettings) -> None:
    """Un ciclo del supervisor: procesa los jobs vencidos, uno a uno.

    Cada job corre con timeout (B-S7): un job colgado (p. ej. una subida
    atascada) no debe bloquear los demás. Al agotar el timeout el job se
    marca como fallido; el hilo de to_thread puede seguir en segundo
    plano, pero su finish posterior es un no-op porque finish_schedule_job
    solo actualiza jobs en estado 'processing'.
    """
    jobs = database.due_schedule_jobs(settings.database_path, database.utc_now())
    for job in jobs:
        try:
            if not database.claim_schedule_job(settings.database_path, str(job["id"])):
                continue
        except Exception:
            _logger.exception(
                "No se pudo reclamar la tarea programada %s; se reintentará "
                "en el próximo ciclo.",
                job["id"],
            )
            continue
        try:
            await asyncio.wait_for(
                asyncio.to_thread(process_schedule_job, settings, job),
                timeout=_SCHEDULE_JOB_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _logger.error(
                "La tarea programada %s superó el tiempo máximo de %.0f s; "
                "se marca como fallida.",
                job["id"],
                _SCHEDULE_JOB_TIMEOUT_SECONDS,
            )
            _finish_job_failed(
                settings,
                job,
                "La publicación superó el tiempo máximo de "
                f"{int(_SCHEDULE_JOB_TIMEOUT_SECONDS // 60)} minutos y se canceló.",
            )
        except Exception as error:
            _logger.exception("No se pudo completar la tarea programada %s", job["id"])
            _finish_job_failed(
                settings,
                job,
                f"No se pudo completar la tarea programada: {error}",
            )
