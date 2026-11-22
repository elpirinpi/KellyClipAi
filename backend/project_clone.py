"""Duplicar proyecto / reprocesar con otra configuración (Fase 3, ítem 15).

Clona un proyecto sin volver a subir el video: el archivo fuente se copia
en local al directorio del proyecto nuevo (nunca se comparte el archivo
del original, para que borrar el original no rompa la copia), la
transcripción se reutiliza tal cual y la configuración admite cambios de
ratio, duración o plantilla de subtítulos. Los clips y las propuestas no
se copian: pertenecen a la configuración anterior y se regeneran al
reprocesar.
"""

from __future__ import annotations

import shutil
import uuid
from typing import Any

from pydantic import ValidationError

from . import database
from .schemas import ProjectConfig
from .settings import AppSettings


def duplicate_project(
    settings: AppSettings,
    project: dict[str, Any],
    *,
    owner_id: str,
    title: str | None = None,
    ratio: str | None = None,
    duration: str | None = None,
    subtitle_template: str | None = None,
) -> str:
    """Crea una copia del proyecto y devuelve el id nuevo.

    Lanza FileNotFoundError si el video fuente del original ya no existe,
    y ValueError si la configuración resultante no es válida.
    """
    # La fuente vive siempre en data_dir/projects/{id}/source (get_project
    # no devuelve input_path: project_from_row la retira).
    source_path = settings.data_dir / "projects" / str(project["id"]) / "source"
    if not source_path.is_file():
        raise FileNotFoundError(
            "El video fuente del proyecto original ya no está disponible."
        )
    try:
        config = ProjectConfig.model_validate(project.get("config") or {})
    except ValidationError as error:
        raise ValueError(f"La configuración del proyecto original no es válida: {error}") from error
    overrides: dict[str, Any] = {}
    if ratio is not None:
        overrides["ratio"] = ratio
    if duration is not None:
        overrides["duration"] = duration
    if subtitle_template is not None:
        overrides["subtitle_template"] = subtitle_template
    if overrides:
        try:
            config = config.model_copy(update=overrides)
            # model_copy no revalida: validar el volcado para rechazar
            # valores fuera del enum (p. ej. un ratio inexistente).
            config = ProjectConfig.model_validate(config.model_dump(mode="json"))
        except ValidationError as error:
            raise ValueError(f"La configuración pedida no es válida: {error}") from error

    new_id = str(uuid.uuid4())
    project_dir = settings.data_dir / "projects" / new_id
    input_path = project_dir / "source"
    project_dir.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source_path, input_path)
        database.create_project(
            settings.database_path,
            new_id,
            (title.strip() if title and title.strip() else f"{project['title']} (copia)")[:200],
            input_path,
            config.model_dump(mode="json"),
            owner_id,
        )
        # La transcripción se reutiliza: el video es el mismo y el
        # reproceso no necesita transcribir de nuevo (el pipeline la lee
        # de transcript_json cuando existe; además la caché por hash del
        # archivo la cubriría igualmente).
        transcript = project.get("transcript")
        if transcript:
            database.update_project(
                settings.database_path,
                new_id,
                transcript=transcript,
            )
    except Exception:
        shutil.rmtree(project_dir, ignore_errors=True)
        try:
            database.delete_project(settings.database_path, new_id)
        except Exception:
            pass
        raise
    return new_id
