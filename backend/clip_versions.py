"""Versiones de clip e historial (Fase 3, ítem 6).

Hoy el render del editor y la exportación del Studio SOBRESCRIBEN el
archivo del clip. Este módulo archiva el archivo vigente antes de
cada sustitución como v1, v2… y permite volver atrás.

Ubicación: ``<proyecto>/versions/<clip_id>/vN.mp4`` (con su ``.ass``
si lo hay). A PROPÓSITO fuera de ``clips/``: el reproceso completo
borra el directorio viejo de clips (``pipeline._swap_staged_clips``)
y las versiones deben sobrevivir a un reproceso.

El disco es la fuente de verdad: ``list_versions`` escanea el
directorio, así que no hay estado que pueda desincronizarse con la
base de datos. Se conservan como máximo ``MAX_VERSIONS`` versiones;
al archivar la 11ª se borra la más antigua.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_VERSIONS = 10
_VERSION_FILE = re.compile(r"^v(\d+)\.mp4$")


def versions_dir(project_dir: Path, clip_id: str) -> Path:
    return Path(project_dir) / "versions" / clip_id


def _version_number(filename: str) -> int | None:
    match = _VERSION_FILE.match(filename)
    return int(match.group(1)) if match else None


def list_versions(project_dir: Path, clip_id: str) -> list[dict[str, Any]]:
    """Versiones archivadas, de la más reciente a la más antigua."""
    directory = versions_dir(project_dir, clip_id)
    if not directory.is_dir():
        return []
    versions: list[dict[str, Any]] = []
    for entry in directory.iterdir():
        number = _version_number(entry.name)
        if number is None or not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        versions.append(
            {
                "version": number,
                "filename": entry.name,
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(),
                "has_subtitles": entry.with_suffix(".ass").is_file(),
            }
        )
    versions.sort(key=lambda item: item["version"], reverse=True)
    return versions


def _prune_old_versions(directory: Path) -> None:
    entries = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and _version_number(entry.name) is not None
    ]
    entries.sort(key=lambda entry: _version_number(entry.name) or 0)
    while len(entries) > MAX_VERSIONS:
        oldest = entries.pop(0)
        oldest.unlink(missing_ok=True)
        oldest.with_suffix(".ass").unlink(missing_ok=True)


def archive_version(
    project_dir: Path,
    clips_dir: Path,
    clip_id: str,
    current_filename: str,
) -> dict[str, Any] | None:
    """Archiva el archivo vigente del clip como la siguiente versión.

    Mueve el mp4 (y su .ass si existe) a ``versions/<clip_id>/``.
    Devuelve el registro de la versión archivada, o None si no había
    archivo vigente que archivar (primer render).
    """
    current = Path(clips_dir) / Path(current_filename).name
    if not current.is_file():
        return None
    directory = versions_dir(project_dir, clip_id)
    directory.mkdir(parents=True, exist_ok=True)
    existing = [
        _version_number(entry.name)
        for entry in directory.iterdir()
        if entry.is_file()
    ]
    number = max((item for item in existing if item is not None), default=0) + 1
    target = directory / f"v{number}.mp4"
    os.replace(current, target)
    subtitle = current.with_suffix(".ass")
    if subtitle.is_file():
        os.replace(subtitle, target.with_suffix(".ass"))
    _prune_old_versions(directory)
    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    return {
        "version": number,
        "filename": target.name,
        "size_bytes": size,
        "created_at": datetime.now(UTC).isoformat(),
        "has_subtitles": target.with_suffix(".ass").is_file(),
    }


def restore_version(
    project_dir: Path,
    clips_dir: Path,
    clip_id: str,
    current_filename: str,
    version: int,
) -> dict[str, Any]:
    """Restaura una versión archivada como archivo vigente del clip.

    El archivo vigente actual se archiva primero como una versión
    nueva (volver atrás nunca destruye nada). Devuelve el registro
    de la versión archivada en ese paso (o None si no había archivo
    vigente). Lanza ``FileNotFoundError`` si la versión no existe.
    """
    directory = versions_dir(project_dir, clip_id)
    source = directory / f"v{version}.mp4"
    if not source.is_file():
        raise FileNotFoundError(f"No existe la versión v{version} del clip.")
    archived = archive_version(project_dir, clips_dir, clip_id, current_filename)
    target = Path(clips_dir) / Path(current_filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    subtitle_source = source.with_suffix(".ass")
    if subtitle_source.is_file():
        shutil.copyfile(subtitle_source, target.with_suffix(".ass"))
    else:
        target.with_suffix(".ass").unlink(missing_ok=True)
    return archived or {
        "version": None,
        "filename": None,
        "size_bytes": 0,
        "created_at": datetime.now(UTC).isoformat(),
        "has_subtitles": False,
    }
