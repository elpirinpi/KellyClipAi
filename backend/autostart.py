"""Auto-inicio opcional con Windows (Fase 3, ítem 18, C.6.39).

Registra la app en ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run``
(el registro por usuario: no pide permisos de administrador). El
``winreg`` se importa dentro de cada función para que este módulo se
pueda importar en Linux/macOS (donde el auto-inicio simplemente no
está disponible) y en los tests.
"""

from __future__ import annotations

import os
from pathlib import Path

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "kellyClipAi"


class AutostartError(RuntimeError):
    pass


def autostart_supported() -> bool:
    """¿Se puede gestionar el auto-inicio en esta plataforma?"""
    if os.name != "nt":
        return False
    try:
        import winreg  # noqa: F401
    except ImportError:
        return False
    return True


def _open_run_key(write: bool):  # type: ignore[no-untyped-def]
    import winreg

    access = winreg.KEY_READ | (winreg.KEY_SET_VALUE if write else 0)
    return winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, access)


def _stored_matches(executable_path: Path, stored: object) -> bool:
    """¿El valor guardado apunta a este ejecutable?

    La comparación ignora las comillas que rodean la ruta y las
    diferencias de mayúsculas/minúsculas (Windows no distingue).
    """
    if not isinstance(stored, str):
        return False
    stored_path = stored.strip().strip('"')
    return os.path.normcase(stored_path) == os.path.normcase(str(executable_path))


def is_autostart_enabled(executable_path: Path | None) -> bool:
    """¿Está registrado el auto-inicio para este ejecutable?

    Sin ejecutable (modo desarrollo/servidor) o sin soporte de
    plataforma, la respuesta es False: no hay nada que registrar.
    """
    if executable_path is None or not autostart_supported():
        return False
    import winreg

    try:
        with _open_run_key(write=False) as key:
            value, _kind = winreg.QueryValueEx(key, _VALUE_NAME)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AutostartError("No se pudo leer el registro de inicio de Windows.") from error
    return _stored_matches(executable_path, value)


def set_autostart(enabled: bool, executable_path: Path | None) -> None:
    """Activa o desactiva el auto-inicio para este ejecutable."""
    if executable_path is None:
        raise AutostartError("El auto-inicio solo está disponible en la app instalada.")
    if not autostart_supported():
        raise AutostartError("El auto-inicio con Windows solo está disponible en Windows.")
    import winreg

    try:
        with _open_run_key(write=True) as key:
            if enabled:
                winreg.SetValueEx(
                    key,
                    _VALUE_NAME,
                    0,
                    winreg.REG_SZ,
                    f'"{executable_path}"',
                )
            else:
                try:
                    winreg.DeleteValue(key, _VALUE_NAME)
                except FileNotFoundError:
                    pass
    except OSError as error:
        raise AutostartError("No se pudo escribir en el registro de inicio de Windows.") from error
