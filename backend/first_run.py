"""Asistente de primer arranque (Fase 3, ítem 19, C.6.42).

Estado del asistente en ``<data_dir>/setup.json``: si el usuario ya
completó el asistente (o ya usaba la app: tiene proyectos o una clave
LLM guardada), no se vuelve a mostrar. La elección del modelo Whisper
sin clave LLM también vive aquí: la tabla ``llm_credentials`` exige
una clave cifrada, así que el asistente no puede guardarla ahí cuando
el usuario decide funcionar sin clave (motor por señales).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETUP_FILE_NAME = "setup.json"

# Espacio libre que se reserva además del modelo elegido: el video
# original, la transcripción y los clips renderizados necesitan sitio.
_FREE_SPACE_HEADROOM_BYTES = 1024 * 1024 * 1024

_MIB = 1024 * 1024


@dataclass(frozen=True)
class WhisperModelInfo:
    name: str
    size_bytes: int
    label: str
    description: str


# Tamaños aproximados de los modelos faster-whisper (Systran). El
# "small" es el modelo por defecto de la app (~500 MB, C.6.42).
WHISPER_MODELS: tuple[WhisperModelInfo, ...] = (
    WhisperModelInfo(
        "tiny",
        75 * _MIB,
        "tiny · rápido",
        "El más ligero (~75 MB). Transcribe rápido en cualquier PC, con algo menos de precisión.",
    ),
    WhisperModelInfo(
        "small",
        466 * _MIB,
        "small · equilibrado",
        "El recomendado (~500 MB). Buen equilibrio entre precisión y "
        "velocidad en la mayoría de los equipos.",
    ),
    WhisperModelInfo(
        "medium",
        1500 * _MIB,
        "medium · preciso",
        "Más preciso (~1,5 GB). Recomendado con 8 GB de RAM o más; transcribe más despacio.",
    ),
    WhisperModelInfo(
        "large-v3",
        3100 * _MIB,
        "large-v3 · máxima precisión",
        "La máxima precisión (~3,1 GB). Solo para equipos potentes con 16 GB de RAM o más.",
    ),
)

WHISPER_MODEL_NAMES = frozenset(model.name for model in WHISPER_MODELS)


def setup_file_path(data_dir: Path) -> Path:
    return data_dir / SETUP_FILE_NAME


def read_setup(data_dir: Path) -> dict[str, Any]:
    """Lee ``setup.json``; un archivo ausente o corrupto equivale a no
    haber completado el asistente (nunca rompe el arranque)."""
    path = setup_file_path(data_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("No se pudo leer %s; se trata como primer arranque.", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_setup(data_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Escritura atómica (temporal + reemplazo) para no dejar un
    ``setup.json`` a medias si la app se cierra en ese momento."""
    path = setup_file_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


def mark_setup_complete(
    data_dir: Path,
    *,
    whisper_model: str | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    """Marca el asistente como completado (o saltado) y guarda la
    elección del modelo Whisper, si la hubo."""
    payload = read_setup(data_dir)
    payload["completed"] = True
    payload["skipped"] = skipped
    if whisper_model is not None:
        payload["whisper_model"] = whisper_model
    return _write_setup(data_dir, payload)


def is_setup_complete(data_dir: Path) -> bool:
    return read_setup(data_dir).get("completed") is True


def whisper_model_from_setup(data_dir: Path) -> str | None:
    """Modelo Whisper elegido en el asistente (sin clave LLM).

    ``None`` si nunca se guardó o si el valor no es un modelo válido:
    el llamador cae al modelo de entorno en ese caso.
    """
    value = read_setup(data_dir).get("whisper_model")
    return value if isinstance(value, str) and value in WHISPER_MODEL_NAMES else None


def needs_setup(
    *,
    completed: bool,
    has_projects: bool,
    has_llm_key: bool,
) -> bool:
    """¿Hay que mostrar el asistente?

    Los usuarios que actualizan desde una versión anterior no lo ven:
    si ya tienen proyectos o una clave guardada, la app ya estaba en
    uso y el asistente no aporta nada.
    """
    return not completed and not has_projects and not has_llm_key


def disk_space(data_dir: Path) -> dict[str, int | None]:
    """Espacio del disco donde vive ``data_dir`` (None si no se puede
    medir: el asistente muestra el aviso genérico en ese caso)."""
    try:
        usage = shutil.disk_usage(data_dir)
    except OSError:
        logger.warning("No se pudo medir el espacio libre en %s.", data_dir)
        return {"free_bytes": None, "total_bytes": None}
    return {"free_bytes": usage.free, "total_bytes": usage.total}


def model_fits(free_bytes: int | None, model_name: str) -> bool:
    """¿Cabe el modelo en el disco libre, con el margen de seguridad?

    Sin medida de espacio (free_bytes None) se responde que sí: el
    asistente no debe bloquear al usuario por un fallo de medición.
    """
    if free_bytes is None:
        return True
    model = next(
        (item for item in WHISPER_MODELS if item.name == model_name),
        None,
    )
    if model is None:
        return False
    return free_bytes >= model.size_bytes + _FREE_SPACE_HEADROOM_BYTES


def downloaded_models() -> set[str]:
    """Modelos Whisper ya descargados en la caché de Hugging Face.

    faster-whisper descarga de ``Systran/faster-whisper-<modelo>``; la
    caché guarda cada modelo bajo ``models--Systran--faster-whisper-<modelo>``
    con al menos un snapshot. Best-effort: cualquier fallo devuelve un
    conjunto vacío (el asistente simplemente no marca nada como
    descargado).
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return set()
    try:
        cache_dir = Path(HF_HUB_CACHE)
        downloaded = set()
        for model in WHISPER_MODELS:
            snapshots = cache_dir / f"models--Systran--faster-whisper-{model.name}" / "snapshots"
            if snapshots.is_dir() and any(snapshots.iterdir()):
                downloaded.add(model.name)
        return downloaded
    except OSError:
        return set()
