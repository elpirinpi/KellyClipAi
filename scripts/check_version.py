#!/usr/bin/env python3
"""Comprueba que todas las fuentes de versión del repo coincidan con VERSION.

Fuente de verdad: el archivo ``VERSION`` de la raíz del repo (una sola línea,
por ejemplo ``1.0.11``).

Fuentes comprobadas:
- ``backend/version.py``: ``APP_VERSION = "..."``
- ``app/package.json``: ``"version": "..."`` de nivel superior
- ``app/package-lock.json``: ``"version": "..."`` de nivel superior y
  ``packages[""].version`` (ambas deben coincidir)
- ``installer/KellyClipAi.iss``: ``#define AppVersion "..."``

Uso::

    python scripts/check_version.py [--root <ruta-del-repo>]

Sale con código 0 si todas las fuentes coinciden con ``VERSION``; en caso
contrario imprime qué archivo difiere (y con qué versión) y sale con código 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
APP_VERSION_PATTERN = re.compile(r"""APP_VERSION\s*=\s*["']([^"']+)["']""")
ISS_VERSION_PATTERN = re.compile(r"""#define\s+AppVersion\s+"([^"]+)\"""")

# (ruta relativa, descripción de dónde se busca la versión)
VERSION_SOURCES = (
    ("backend/version.py", "APP_VERSION"),
    ("app/package.json", '"version"'),
    ("app/package-lock.json", '"version"'),
    ("installer/KellyClipAi.iss", "#define AppVersion"),
)


def _read_expected_version(root: Path) -> str:
    version_file = root / "VERSION"
    if not version_file.is_file():
        raise SystemExit(f"ERROR: no se encontró el archivo de versión: {version_file}")
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise SystemExit("ERROR: VERSION está vacío.")
    if not VERSION_PATTERN.match(version):
        raise SystemExit(
            f"ERROR: VERSION contiene un valor inválido: {version!r} "
            "(se espera X.Y.Z, con sufijo opcional, p. ej. 1.0.11)."
        )
    return version


def _version_from_python_module(path: Path) -> str:
    match = APP_VERSION_PATTERN.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError('no se encontró una asignación APP_VERSION = "..."')
    return match.group(1)


def _versions_from_package_json(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    top_level = data.get("version")
    if top_level is None:
        raise ValueError('no se encontró "version" de nivel superior')
    found.append(('"version"', str(top_level)))
    packages = data.get("packages")
    if isinstance(packages, dict):
        root_package = packages.get("")
        if isinstance(root_package, dict) and "version" in root_package:
            found.append(('packages[""].version', str(root_package["version"])))
    return found


def _version_from_iss(path: Path) -> str:
    match = ISS_VERSION_PATTERN.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError('no se encontró #define AppVersion "..."')
    return match.group(1)


def _collect_versions(root: Path) -> dict[str, list[tuple[str, str]]]:
    """Devuelve {ruta relativa: [(ubicación, versión), ...]} por fuente."""
    collected: dict[str, list[tuple[str, str]]] = {}
    for relative_path, _description in VERSION_SOURCES:
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"{relative_path}: archivo no encontrado")
        if relative_path.endswith(".py"):
            collected[relative_path] = [("APP_VERSION", _version_from_python_module(path))]
        elif relative_path.endswith(".json"):
            collected[relative_path] = _versions_from_package_json(path)
        elif relative_path.endswith(".iss"):
            collected[relative_path] = [("#define AppVersion", _version_from_iss(path))]
    return collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba que las fuentes de versión coincidan con VERSION."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Raíz del repo (por defecto, la carpeta que contiene scripts/).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    expected = _read_expected_version(root)
    print(f"Versión esperada (VERSION): {expected}")

    mismatches: list[str] = []
    try:
        collected = _collect_versions(root)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    except json.JSONDecodeError as error:
        print(f"ERROR: JSON inválido: {error}")
        return 1

    for relative_path, versions in collected.items():
        for location, found in versions:
            status = "OK" if found == expected else "DIFIERE"
            print(f"  {relative_path} [{location}]: {found} ({status})")
            if found != expected:
                mismatches.append(f"{relative_path} [{location}] declara {found!r}")

    if mismatches:
        print()
        print(f"ERROR: la versión no coincide con VERSION ({expected}):")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        print("Actualiza cada archivo a la versión de VERSION antes de publicar.")
        return 1

    print(f"Todas las fuentes de versión coinciden con VERSION ({expected}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
