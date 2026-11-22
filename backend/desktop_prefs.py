"""Preferencias de escritorio (Fase 3, ítem 18, C.6.38).

Viven en ``<data_dir>/desktop.json`` para que las lean tanto el
proceso de escritorio (``desktop.py``, al arrancar) como el backend
(los endpoints de Ajustes). Son preferencias del dispositivo, no de
la cuenta: no van en la base de datos.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PREFS_FILE_NAME = "desktop.json"

# Minimizar a la bandeja al cerrar: activado por defecto, porque es el
# comportamiento que evita matar un procesamiento largo al cerrar la
# ventana por accidente. Iniciar minimizado: apagado por defecto.
DEFAULT_PREFS: dict[str, bool] = {
    "minimize_to_tray": True,
    "start_minimized": False,
}


def prefs_file_path(data_dir: Path) -> Path:
    return data_dir / PREFS_FILE_NAME


def load_desktop_prefs(data_dir: Path) -> dict[str, bool]:
    """Preferencias guardadas, con los valores por defecto para lo que
    falte. Un archivo ausente o corrupto equivale a los valores por
    defecto (nunca rompe el arranque ni los Ajustes)."""
    prefs = dict(DEFAULT_PREFS)
    path = prefs_file_path(data_dir)
    if not path.is_file():
        return prefs
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("No se pudo leer %s; se usan las preferencias por defecto.", path)
        return prefs
    if not isinstance(payload, dict):
        return prefs
    for key in DEFAULT_PREFS:
        value = payload.get(key)
        if isinstance(value, bool):
            prefs[key] = value
    return prefs


def save_desktop_prefs(data_dir: Path, updates: dict[str, Any]) -> dict[str, bool]:
    """Aplica cambios parciales (solo claves conocidas y booleanas) y
    guarda el resultado con escritura atómica."""
    prefs = load_desktop_prefs(data_dir)
    for key, value in updates.items():
        if key in DEFAULT_PREFS and isinstance(value, bool):
            prefs[key] = value
    path = prefs_file_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    temporary.replace(path)
    return prefs


def tray_available() -> tuple[bool, str | None]:
    """¿Se puede mostrar la bandeja del sistema aquí?

    Devuelve (True, None) o (False, motivo en español). pystray y
    Pillow son dependencias opcionales: si faltan, la app funciona
    igual sin bandeja (el cierre de la ventana cierra la app).
    """
    if os.name != "nt":
        return False, "La bandeja del sistema solo está disponible en la app de Windows."
    try:
        import pystray  # noqa: F401
    except ImportError:
        return False, "La bandeja no está disponible: falta el componente pystray."
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False, "La bandeja no está disponible: falta el componente Pillow."
    return True, None
