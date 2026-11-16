from __future__ import annotations

import ctypes
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import subprocess
from pathlib import Path


def _resource_path(*parts: str) -> Path:
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_dir.joinpath(*parts)


def _show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(
        None,
        message,
        "kellyClipAi",
        0x10,
    )


def _prepare_user_directory() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("Windows no proporcionó la carpeta LOCALAPPDATA.")
    user_dir = Path(local_app_data) / "kellyClipAi"
    user_dir.mkdir(parents=True, exist_ok=True)
    env_file = user_dir / ".env"
    if not env_file.exists():
        template = _resource_path("config", ".env.example")
        if not template.is_file() and not getattr(sys, "frozen", False):
            template = _resource_path(".env.example")
        if not template.is_file():
            raise FileNotFoundError("No se encontró la plantilla de configuración .env.example.")
        env_file.write_bytes(template.read_bytes())
    os.environ["KELLYCLIP_ENV_FILE"] = str(env_file)
    os.environ["KELLYCLIP_DATA_DIR"] = str(user_dir / "data")
    os.environ["KELLYCLIP_OAUTH_REDIRECT_BASE_URL"] = "http://127.0.0.1:8000"
    os.environ["KELLYCLIP_FRONTEND_BASE_URL"] = "http://127.0.0.1:8000"
    ffmpeg_dir = _resource_path("bin")
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    return user_dir


def _ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((host, port))
        except OSError as error:
            raise RuntimeError(
                f"El puerto local {port} ya está ocupado. Cierra la otra app o servicio que lo usa e inténtalo de nuevo."
            ) from error


def _wait_for_backend(thread: threading.Thread) -> None:
    deadline = time.monotonic() + 30
    health_url = "http://127.0.0.1:8000/api/health"
    while time.monotonic() < deadline and thread.is_alive():
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError(
        "El backend local no inició correctamente. Consulta el registro de la aplicación para más detalles."
    )


def main(smoke_test: bool = False) -> None:
    user_dir = _prepare_user_directory()
    logging.basicConfig(
        filename=user_dir / "desktop.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )
    web_dir = _resource_path("web")
    if not (web_dir / "index.html").is_file():
        raise FileNotFoundError("No se encontraron los archivos de la interfaz incluidos en el instalador.")
    for binary in ("ffmpeg.exe", "ffprobe.exe"):
        if not (_resource_path("bin") / binary).is_file():
            raise FileNotFoundError(f"No se encontró {binary} incluido en el instalador.")

    _ensure_port_available("127.0.0.1", 8000)

    from backend.main import create_app
    from backend.settings import get_settings
    import uvicorn
    import webview

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(get_settings(), web_dir=web_dir),
            host="127.0.0.1",
            port=8000,
            log_level="warning",
            access_log=False,
            log_config=None,
        )
    )
    server_thread = threading.Thread(target=server.run, name="kellyclipai-api", daemon=True)
    server_thread.start()
    try:
        _wait_for_backend(server_thread)
        if smoke_test:
            for path in ("/", "/settings", "/api/health"):
                with urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(f"La comprobación local {path} respondió HTTP {response.status}.")
            for binary in ("ffmpeg.exe", "ffprobe.exe"):
                subprocess.run(
                    [str(_resource_path("bin") / binary), "-version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            return
        webview.create_window(
            "kellyClipAi",
            "http://127.0.0.1:8000/",
            min_size=(360, 600),
            width=1440,
            height=940,
        )
        webview.start(gui="edgechromium", debug=False)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


if __name__ == "__main__":
    try:
        main(smoke_test=os.getenv("KELLYCLIP_SMOKE_TEST") == "1")
    except Exception as error:
        logging.exception("No se pudo iniciar kellyClipAi")
        try:
            _show_error(
                f"No se pudo iniciar kellyClipAi.\n\n{error}\n\n"
                "Si falta WebView2, conéctate a internet y vuelve a ejecutar el instalador. "
                "Registro: %LOCALAPPDATA%\\kellyClipAi\\desktop.log"
            )
        except Exception:
            print(f"No se pudo iniciar kellyClipAi: {error}", file=sys.stderr)
        raise
