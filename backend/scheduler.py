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
    account = database.get_connected_account(settings.database_path, str(job["platform"]))
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
    while True:
        for job in database.due_schedule_jobs(settings.database_path, database.utc_now()):
            if not database.claim_schedule_job(settings.database_path, str(job["id"])):
                continue
            try:
                await asyncio.to_thread(process_schedule_job, settings, job)
            except Exception as error:
                _logger.exception("No se pudo completar la tarea programada %s", job["id"])
                database.finish_schedule_job(
                    settings.database_path,
                    str(job["id"]),
                    "failed",
                    error=f"No se pudo completar la tarea programada: {error}",
                )
        await asyncio.sleep(2)
