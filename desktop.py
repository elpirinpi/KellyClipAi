from __future__ import annotations

import ctypes
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_WINDOW_TITLE = "kellyClipAi"
_SINGLE_INSTANCE_MUTEX = "Local\\kellyClipAi-desktop"
# Puerto preferido del backend local y último puerto de respaldo: si el
# 8000 está ocupado se prueba 8001-8010 en orden (B34).
_DEFAULT_PORT = 8000
_MAX_FALLBACK_PORT = 8010
_logger = logging.getLogger(__name__)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    """Backstop para excepciones no capturadas en hilos secundarios: sin
    esto van a stderr, que en el build --windowed es devnull y el fallo
    de arranque queda invisible (B35)."""
    thread_name = args.thread.name if args.thread is not None else "desconocido"
    _logger.error(
        "Excepción no capturada en el hilo %s",
        thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


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


def _create_log_handler(log_path: Path) -> RotatingFileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return RotatingFileHandler(
        log_path,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )


class _SingleInstance:
    def __init__(self, mutex_name: str = _SINGLE_INSTANCE_MUTEX) -> None:
        self._mutex_name = mutex_name
        self._handle: int | None = None
        self._kernel32: ctypes.WinDLL | None = None

    def acquire(self) -> bool:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        create_mutex.restype = ctypes.c_void_p
        ctypes.set_last_error(0)
        handle = create_mutex(None, True, self._mutex_name)
        error = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(error)
        if error == 183:
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle(handle)
            return False
        kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._handle = handle
        self._kernel32 = kernel32
        return True

    def close(self) -> None:
        if self._handle is None or self._kernel32 is None:
            return
        if not self._kernel32.ReleaseMutex(self._handle):
            _logger.error(
                "No se pudo liberar el mutex de instancia única: %s",
                ctypes.WinError(ctypes.get_last_error()),
            )
        if not self._kernel32.CloseHandle(self._handle):
            _logger.error(
                "No se pudo cerrar el mutex de instancia única: %s",
                ctypes.WinError(ctypes.get_last_error()),
            )
        self._handle = None
        self._kernel32 = None


def _find_main_window(user32: ctypes.WinDLL) -> int | None:
    matching_windows: list[int] = []
    enum_windows_proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_ssize_t)
    user32.IsWindowVisible.argtypes = (ctypes.c_void_p,)
    user32.IsWindowVisible.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = (ctypes.c_void_p,)
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    user32.GetWindowTextW.restype = ctypes.c_int

    @enum_windows_proc
    def visit_window(hwnd: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title_length = user32.GetWindowTextLengthW(hwnd)
        if title_length <= 0:
            return True
        title = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title, len(title))
        if title.value.casefold().startswith(_WINDOW_TITLE.casefold()):
            matching_windows.append(hwnd)
            return False
        return True

    user32.EnumWindows.argtypes = (enum_windows_proc, ctypes.c_ssize_t)
    user32.EnumWindows.restype = ctypes.c_int
    user32.EnumWindows(visit_window, 0)
    return matching_windows[0] if matching_windows else None


def _focus_existing_window(timeout: float = 10) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.ShowWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
    user32.BringWindowToTop.argtypes = (ctypes.c_void_p,)
    user32.SetForegroundWindow.argtypes = (ctypes.c_void_p,)
    user32.SetForegroundWindow.restype = ctypes.c_int
    user32.FlashWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hwnd = _find_main_window(user32)
        if hwnd is not None:
            user32.ShowWindow(hwnd, 9)
            user32.BringWindowToTop(hwnd)
            if user32.SetForegroundWindow(hwnd):
                return True
            user32.FlashWindow(hwnd, True)
            return False
        time.sleep(0.2)
    return False


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
    # Valores por defecto para el puerto preferido (8000); _run_app los
    # sobrescribe con el puerto finalmente elegido antes de leer la
    # configuración.
    os.environ["KELLYCLIP_OAUTH_REDIRECT_BASE_URL"] = "http://127.0.0.1:8000"
    os.environ["KELLYCLIP_FRONTEND_BASE_URL"] = "http://127.0.0.1:8000"
    ffmpeg_dir = _resource_path("bin")
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    return user_dir


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((host, port))
        except OSError:
            return False
    return True


def _find_available_port(
    host: str,
    start: int = _DEFAULT_PORT,
    end: int = _MAX_FALLBACK_PORT,
) -> int:
    """Primer puerto libre entre `start` y `end` (ambos incluidos).

    El puerto preferido es el 8000, pero cualquier otro servicio local
    puede ocuparlo; probar los siguientes evita que la app quede
    inutilizable por un puerto ocupado.
    """
    for port in range(start, end + 1):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(
        f"No hay puertos locales libres entre {start} y {end}. Cierra la "
        "otra app o servicio que los usa e inténtalo de nuevo."
    )


def _wait_for_backend(
    thread: threading.Thread,
    port: int,
    startup_error: dict[str, BaseException | None] | None = None,
) -> None:
    deadline = time.monotonic() + 30
    health_url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline and thread.is_alive():
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    if startup_error is not None and startup_error.get("error") is not None:
        raise RuntimeError(
            "El backend local no inició correctamente: "
            f"{startup_error['error']}. Consulta el registro de la "
            "aplicación para más detalles."
        )
    raise RuntimeError(
        "El backend local no inició correctamente. Consulta el registro de la aplicación para más detalles."
    )


def _configure_uvicorn_logging(log_handler: logging.Handler) -> None:
    """Dirige los loggers de uvicorn al mismo archivo que desktop.log.

    Se comparte el handler del log principal (no se crea otro sobre el
    mismo archivo, para no romper la rotación) y se desactiva la
    propagación para no duplicar cada registro en el logger raíz.
    Con log_config=None uvicorn no instala handlers propios, así que
    esta configuración sobrevive a su configure_logging().
    """
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = [log_handler]
        logger.propagate = False


def _run_server(
    server: object,
    startup_error: dict[str, BaseException | None],
) -> None:
    """Arranque del servidor en su hilo: cualquier excepción queda
    registrada en desktop.log y guardada para el mensaje de error del
    hilo principal, en vez de perderse en stderr (devnull en el build
    --windowed)."""
    try:
        server.run()  # type: ignore[attr-defined]
    except Exception as error:
        startup_error["error"] = error
        _logger.exception("El servidor local terminó con un error")


class DesktopApi:
    def select_download_folder(self) -> str | None:
        import webview

        selected = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        return selected[0] if selected else None


def _run_app(
    user_dir: Path,
    smoke_test: bool,
    log_handler: logging.Handler | None = None,
) -> None:
    web_dir = _resource_path("web")
    if not (web_dir / "index.html").is_file():
        raise FileNotFoundError("No se encontraron los archivos de la interfaz incluidos en el instalador.")
    for binary in ("ffmpeg.exe", "ffprobe.exe"):
        if not (_resource_path("bin") / binary).is_file():
            raise FileNotFoundError(f"No se encontró {binary} incluido en el instalador.")

    port = _find_available_port("127.0.0.1")
    if port != _DEFAULT_PORT:
        _logger.warning(
            "El puerto %d está ocupado; el backend usará el puerto %d. "
            "Si algún proveedor OAuth exige la URL de redirección exacta, "
            "actualízala en su consola para este puerto.",
            _DEFAULT_PORT,
            port,
        )
    _logger.info("Backend local en http://127.0.0.1:%d", port)
    # Las URLs base deben apuntar al puerto elegido y fijarse antes de
    # leer la configuración, que las toma de estas variables.
    os.environ["KELLYCLIP_OAUTH_REDIRECT_BASE_URL"] = f"http://127.0.0.1:{port}"
    os.environ["KELLYCLIP_FRONTEND_BASE_URL"] = f"http://127.0.0.1:{port}"

    import uvicorn
    import webview

    from backend.main import create_app
    from backend.settings import get_settings
    from backend.updater import UpdateManager
    from backend.version import APP_VERSION

    if log_handler is not None:
        _configure_uvicorn_logging(log_handler)
    updater = UpdateManager(
        APP_VERSION,
        user_dir,
        Path(sys.executable) if getattr(sys, "frozen", False) else None,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(get_settings(), web_dir=web_dir, updater=updater),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            log_config=None,
        )
    )
    startup_error: dict[str, BaseException | None] = {"error": None}
    server_thread = threading.Thread(
        target=_run_server,
        args=(server, startup_error),
        name="kellyclipai-api",
        daemon=True,
    )
    server_thread.start()
    try:
        _wait_for_backend(server_thread, port, startup_error)
        if smoke_test:
            for path in ("/", "/settings", "/api/health"):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
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
            _WINDOW_TITLE,
            f"http://127.0.0.1:{port}/",
            min_size=(360, 600),
            width=1440,
            height=940,
            js_api=DesktopApi(),
        )
        webview.start(gui="edgechromium", debug=False)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


def main(smoke_test: bool = False) -> None:
    user_dir = _prepare_user_directory()
    log_handler = _create_log_handler(user_dir / "desktop.log")
    logging.basicConfig(
        handlers=[log_handler],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    threading.excepthook = _log_thread_exception
    instance = _SingleInstance()
    if not instance.acquire():
        if not _focus_existing_window():
            _logger.warning("No se pudo traer a primer plano la ventana de kellyClipAi.")
            _show_error(
                "kellyClipAi ya se está ejecutando. No se pudo traer su ventana al frente; "
                "busca la aplicación abierta en la barra de tareas."
            )
        return
    try:
        _run_app(user_dir, smoke_test, log_handler)
    finally:
        instance.close()


if __name__ == "__main__":
    try:
        main(smoke_test=os.getenv("KELLYCLIP_SMOKE_TEST") == "1")
    except Exception as error:
        _logger.exception("No se pudo iniciar kellyClipAi")
        try:
            _show_error(
                f"No se pudo iniciar kellyClipAi.\n\n{error}\n\n"
                "Si falta WebView2, conéctate a internet y vuelve a ejecutar el instalador. "
                "Registro: %LOCALAPPDATA%\\kellyClipAi\\desktop.log"
            )
        except (OSError, AttributeError, ctypes.ArgumentError):
            print(f"No se pudo iniciar kellyClipAi: {error}", file=sys.stderr)
        raise
