"""Carpeta observada + barrido de lotes + franja nocturna (Fase 3, ítem 11).

Todo opcional y apagado por defecto: sin carpetas configuradas el worker
no hace nada, y sin franja nocturna activada la cola procesa siempre.

* Carpeta observada: sondeo (polling) con stdlib — sin dependencias
  nuevas como watchdog, para que funcione en cualquier PC sin instalar
  nada. Cada cierto tiempo (configurable por carpeta, 5 min por defecto)
  se listan los videos nuevos y se crea un proyecto por cada uno, que se
  encola para procesar. Un video se detecta una sola vez: la clave es
  nombre + tamaño + fecha de modificación.
* Barrido de lotes: los enlaces de un lote (POST /api/queue/batch) se
  descargan como staged media en segundo plano; cuando una descarga queda
  lista, este mismo worker crea su proyecto y lo encola. Sin este
  barrido, un enlace encolado se quedaría en "downloading" para siempre.
* Franja nocturna: el gate de la cola (process_queue) consulta
  night_window_allows() antes de tomar cada trabajo.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import database
from .publish_guide import night_processing_allowed
from .settings import AppSettings

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"}
_DEFAULT_POLL_SECONDS = 300
_MIN_POLL_SECONDS = 60
_BATCH_SWEEP_SECONDS = 30.0

# Dueño de los ajustes a nivel de dispositivo (la franja nocturna protege
# el PC compartido, no a un usuario concreto).
_DEVICE_OWNER_ID = "__device__"
_NIGHT_ENABLED_KEY = "night_window.enabled"
_NIGHT_START_KEY = "night_window.start"
_NIGHT_END_KEY = "night_window.end"


def get_night_window(database_path: Path) -> dict[str, Any]:
    """Franja nocturna guardada (por defecto: apagada, 22:00–06:00)."""
    preferences = database.get_preferences(database_path, _DEVICE_OWNER_ID)
    return {
        "enabled": bool(preferences.get(_NIGHT_ENABLED_KEY, False)),
        "start": str(preferences.get(_NIGHT_START_KEY, "22:00")),
        "end": str(preferences.get(_NIGHT_END_KEY, "06:00")),
    }


def set_night_window(
    database_path: Path,
    *,
    enabled: bool,
    start: str,
    end: str,
) -> dict[str, Any]:
    database.set_preference(database_path, _DEVICE_OWNER_ID, _NIGHT_ENABLED_KEY, enabled)
    database.set_preference(database_path, _DEVICE_OWNER_ID, _NIGHT_START_KEY, start)
    database.set_preference(database_path, _DEVICE_OWNER_ID, _NIGHT_END_KEY, end)
    return get_night_window(database_path)


def night_window_allows(database_path: Path, *, now: datetime | None = None) -> bool:
    """Gate de la cola: True si se puede procesar ahora."""
    window = get_night_window(database_path)
    return night_processing_allowed(
        enabled=bool(window["enabled"]),
        start=str(window["start"]),
        end=str(window["end"]),
        now=now,
    )


def file_key(path: Path) -> str:
    """Identidad estable de un archivo: nombre + tamaño + modificación."""
    stat = path.stat()
    return f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"


def detect_new_videos(
    folder_path: Path,
    seen_keys: set[str],
) -> list[Path]:
    """Videos nuevos de una carpeta (no recursivo, sin archivos ocultos).

    Devuelve los videos cuya clave no está en `seen_keys`, ordenados por
    nombre para un orden determinista. Un archivo que crece entre dos
    sondeos (copia en curso) cambia de clave y se detecta cuando deja de
    crecer: el tamaño forma parte de la clave.
    """
    if not folder_path.is_dir():
        return []
    found: list[Path] = []
    try:
        entries = sorted(folder_path.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_file():
            continue
        if entry.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            key = file_key(entry)
        except OSError:
            continue
        if key not in seen_keys:
            found.append(entry)
    return found


def create_project_from_file(
    settings: AppSettings,
    source: Path,
    owner_id: str,
    config: dict[str, Any],
) -> str:
    """Crea un proyecto copiando el video a su directorio (sin mover el
    original: la carpeta observada no es nuestra para vaciarla)."""
    project_id = str(uuid.uuid4())
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    project_dir.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source, input_path)
        database.create_project(
            settings.database_path,
            project_id,
            source.stem[:200],
            input_path,
            config,
            owner_id,
        )
    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        raise
    return project_id


def create_project_from_ready_staged(
    settings: AppSettings,
    staged_id: str,
) -> str | None:
    """Convierte un staged media listo en proyecto (barrido de lotes).

    Replica la lógica de POST /api/staged-media/{id}/project: reclama el
    staged, mueve el archivo al proyecto y marca el staged como
    completado. Devuelve el id del proyecto o None si el staged ya no
    está listo (otro flujo lo tomó antes).
    """
    staged = database.get_staged_media(settings.database_path, staged_id)
    if staged is None:
        return None
    if staged["status"] == "completed" and staged.get("project_id"):
        return str(staged["project_id"])
    if staged["status"] != "ready":
        return None
    config = staged.get("config")
    if config is None:
        return None
    staged_path = settings.data_dir / "staging" / f"{staged_id}.source"
    if not staged_path.is_file():
        return None
    if not database.claim_staged_media(settings.database_path, staged_id):
        return None
    project_id = str(uuid.uuid4())
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    try:
        project_dir.mkdir(parents=True, exist_ok=False)
        shutil.move(str(staged_path), input_path)
        database.create_project(
            settings.database_path,
            project_id,
            (staged.get("title") or staged.get("filename") or "Video")[:200],
            input_path,
            config,
            staged.get("owner_id"),
        )
        database.update_staged_media(
            settings.database_path,
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
            settings.database_path,
            staged_id,
            status="ready",
        )
        raise
    return project_id


async def sweep_batch_items(
    settings: AppSettings,
    submit: Callable[[str], Awaitable[None]],
) -> int:
    """Convierte los lotes con descarga lista en proyectos encolados.

    Devuelve cuántos proyectos se crearon y encolaron en este barrido.
    """
    created = 0
    try:
        items = database.list_pending_batch_items(settings.database_path)
    except Exception:
        logger.exception("No se pudieron leer los lotes pendientes.")
        return 0
    for item in items:
        staged_id = item.get("staged_id")
        if not staged_id:
            continue
        staged = database.get_staged_media(settings.database_path, str(staged_id))
        if staged is None:
            database.update_batch_item(
                settings.database_path,
                str(item["id"]),
                status="failed",
                error="La descarga desapareció antes de crear el proyecto.",
            )
            continue
        if staged["status"] == "failed":
            database.update_batch_item(
                settings.database_path,
                str(item["id"]),
                status="failed",
                error=str(staged.get("error") or "La descarga falló."),
            )
            continue
        if staged["status"] != "ready":
            continue
        try:
            project_id = await asyncio.to_thread(
                create_project_from_ready_staged, settings, str(staged_id)
            )
        except Exception as error:
            logger.exception(
                "No se pudo crear el proyecto del lote %s", item["id"]
            )
            database.update_batch_item(
                settings.database_path,
                str(item["id"]),
                status="failed",
                error=str(error),
            )
            continue
        if project_id is None:
            continue
        try:
            await submit(project_id)
        except Exception as error:
            logger.exception(
                "No se pudo encolar el proyecto %s del lote %s",
                project_id,
                item["id"],
            )
            database.update_batch_item(
                settings.database_path,
                str(item["id"]),
                status="failed",
                error=f"El proyecto se creó pero no se pudo encolar: {error}",
                project_id=project_id,
            )
            continue
        database.update_batch_item(
            settings.database_path,
            str(item["id"]),
            status="done",
            project_id=project_id,
        )
        created += 1
    return created


async def scan_folder_once(
    settings: AppSettings,
    folder: dict[str, Any],
    submit: Callable[[str], Awaitable[None]],
) -> int:
    """Un sondeo de una carpeta observada. Devuelve videos nuevos encolados.

    Estabilidad en dos sondeos: un video que aparece por primera vez solo
    se registra como candidato; se crea su proyecto en el siguiente
    sondeo, si su clave (nombre + tamaño + modificación) no cambió. Así
    una copia en curso — que crece entre sondeos y cambia de clave — no
    se encola a medias.
    """
    folder_id = str(folder["id"])
    folder_path = Path(str(folder["path"]))
    try:
        seen = database.list_seen_files(settings.database_path, folder_id)
        completed = {key for key, project_id in seen.items() if project_id}
        candidates = await asyncio.to_thread(
            detect_new_videos, folder_path, completed
        )
    except Exception:
        logger.exception("No se pudo sondear la carpeta observada %s", folder_path)
        return 0
    config = folder.get("config") or {}
    created = 0
    for candidate in candidates:
        try:
            key = file_key(candidate)
        except OSError:
            continue
        if key not in seen:
            # Primera vista: registrar como candidato y esperar al
            # siguiente sondeo para confirmar que dejó de crecer.
            database.mark_file_seen(settings.database_path, folder_id, key)
            continue
        # Visto antes sin proyecto: estable (misma clave) o reintento tras
        # un fallo de creación anterior. Marcar antes de crear: si la
        # creación falla, el archivo no se reintenta en bucle en cada
        # sondeo (un cambio posterior de tamaño o fecha lo detecta de nuevo).
        try:
            project_id = await asyncio.to_thread(
                create_project_from_file,
                settings,
                candidate,
                str(folder["owner_id"]),
                dict(config),
            )
        except Exception:
            logger.exception(
                "No se pudo crear el proyecto del video %s", candidate
            )
            continue
        database.mark_file_seen(
            settings.database_path, folder_id, key, project_id=project_id
        )
        try:
            await submit(project_id)
        except Exception:
            logger.exception(
                "No se pudo encolar el proyecto %s de la carpeta %s",
                project_id,
                folder_path,
            )
            continue
        created += 1
    database.touch_watch_folder_scan(settings.database_path, folder_id)
    return created


async def watch_folder_worker(
    settings: AppSettings,
    submit: Callable[[str], Awaitable[None]],
) -> None:
    """Bucle del worker: sondea carpetas y barre lotes hasta cancelarse.

    Sin carpetas y sin lotes pendientes, duerme y vuelve a comprobar: el
    coste es una consulta ligera cada 30 s. La cancelación (apagado) se
    propaga; cualquier otro error se registra y el bucle continúa.
    """
    last_scan: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    while True:
        try:
            await sweep_batch_items(settings, submit)
            try:
                folders = database.list_watch_folders(settings.database_path)
            except Exception:
                logger.exception("No se pudieron leer las carpetas observadas.")
                folders = []
            now = loop.time()
            for folder in folders:
                if not folder.get("enabled"):
                    continue
                poll = max(
                    int(folder.get("poll_seconds") or _DEFAULT_POLL_SECONDS),
                    _MIN_POLL_SECONDS,
                )
                if now - last_scan.get(str(folder["id"]), 0.0) < poll:
                    continue
                last_scan[str(folder["id"])] = now
                await scan_folder_once(settings, folder, submit)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("El worker de carpetas observadas falló; reintentando.")
        await asyncio.sleep(_BATCH_SWEEP_SECONDS)
