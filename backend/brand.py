"""Rutas y constantes de la plantilla de marca (Fase 3, ítem 5).

El logo vive en disco (``<data_dir>/brand/<owner>/logo.<ext>``) y la
base de datos solo guarda su nombre de archivo. El directorio del
propietario se sanea: un id de usuario con caracteres raros no puede
escapar del directorio de marca.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Formatos de logo admitidos (raster: ffmpeg lo superpone con overlay;
# SVG no lo admite el filtro de imagen de ffmpeg).
LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
LOGO_MEDIA_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_LOGO_BYTES = 2 * 1024 * 1024
_SAFE_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def owner_dir_name(owner_id: str) -> str:
    """Nombre de directorio seguro para un id de propietario."""
    if _SAFE_OWNER.fullmatch(owner_id or ""):
        return owner_id
    return hashlib.sha256((owner_id or "legacy").encode("utf-8")).hexdigest()[:16]


def brand_dir(data_dir: Path, owner_id: str) -> Path:
    return Path(data_dir) / "brand" / owner_dir_name(owner_id)


def brand_logo_path(
    data_dir: Path, owner_id: str, logo_filename: str | None
) -> Path | None:
    """Ruta del logo guardado, o None si no hay logo válido."""
    if not logo_filename:
        return None
    filename = Path(str(logo_filename)).name
    if Path(filename).suffix.lower() not in LOGO_EXTENSIONS:
        return None
    path = brand_dir(data_dir, owner_id) / filename
    return path if path.is_file() else None
