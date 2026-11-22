"""Bandeja del sistema (system tray) para la app de escritorio
(Fase 3, ítem 18, C.6.38).

La app procesa videos largos; hoy cerrar la ventana mata el proceso.
Con la bandeja, cerrar la ventana la oculta y el procesamiento sigue;
el icono permite mostrarla u ocultarla, abrir la carpeta de clips y
salir de la app.

pystray es una dependencia OPCIONAL: si no está instalada (o falta
Pillow para dibujar el icono), ``start()`` devuelve False y la app
funciona como antes, sin bandeja. El icono se genera en memoria con
Pillow (un triángulo de reproducción blanco sobre el azul de la app),
sin necesidad de ningún archivo de imagen en el bundle.

Hilos: pystray corre su bucle de mensajes de Windows en un hilo
daemon propio; los métodos de pywebview (show/hide/destroy) se
pueden llamar desde ese hilo porque WinForms los redirige al hilo
de la interfaz con ``Control.Invoke``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRAY_ICON_SIZE = 64
# Azul de la app (#0F4C9C) y blanco del triángulo.
_TRAY_BLUE = (15, 76, 156, 255)
_TRAY_WHITE = (255, 255, 255, 255)


def build_tray_image(size: int = _TRAY_ICON_SIZE):  # type: ignore[no-untyped-def]
    """Genera el icono de la bandeja con Pillow.

    Un cuadrado azul redondeado con un triángulo de reproducción
    blanco. Sin fuentes ni archivos externos: solo formas.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [0, 0, size - 1, size - 1],
        radius=size // 5,
        fill=_TRAY_BLUE,
    )
    # Triángulo de reproducción centrado, apuntando a la derecha.
    left = size * 0.36
    right = size * 0.72
    top = size * 0.28
    bottom = size * 0.72
    middle = size * 0.5
    draw.polygon(
        [(left, top), (right, middle), (left, bottom)],
        fill=_TRAY_WHITE,
    )
    return image


class TrayController:
    """Icono de bandeja con Mostrar/Ocultar, Abrir carpeta de clips
    y Salir, más el "cerrar minimiza a la bandeja".

    ``minimize_to_tray`` es una función (no un booleano) para que el
    interruptor de Ajustes se aplique sin reiniciar la app: se
    consulta en cada intento de cierre.
    """

    def __init__(
        self,
        *,
        window: Any,
        clips_dir: Path,
        on_quit: Callable[[], None],
        minimize_to_tray: Callable[[], bool],
        visible: bool = True,
        icon_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._window = window
        self._clips_dir = clips_dir
        self._on_quit = on_quit
        self._minimize_to_tray = minimize_to_tray
        self._visible = visible
        self._quitting = False
        self._icon_factory = icon_factory
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def running(self) -> bool:
        return self._icon is not None

    def _menu_label(self, _item: Any) -> str:
        return "Ocultar kellyClipAi" if self._visible else "Mostrar kellyClipAi"

    def _toggle_window(self, _icon: Any = None, _item: Any = None) -> None:
        if self._visible:
            self.hide_window()
        else:
            self.show_window()

    def show_window(self) -> None:
        try:
            self._window.show()
            restore = getattr(self._window, "restore", None)
            if callable(restore):
                restore()
        except Exception:
            logger.exception("No se pudo mostrar la ventana desde la bandeja")
            return
        self._visible = True

    def hide_window(self) -> None:
        try:
            self._window.hide()
        except Exception:
            logger.exception("No se pudo ocultar la ventana desde la bandeja")
            return
        self._visible = False

    def open_clips_folder(self, _icon: Any = None, _item: Any = None) -> None:
        try:
            self._clips_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(self._clips_dir))  # type: ignore[attr-defined]
        except OSError:
            logger.exception("No se pudo abrir la carpeta de clips")

    def quit_app(self, _icon: Any = None, _item: Any = None) -> None:
        """Salir desde la bandeja: detiene el icono y pide cerrar la
        ventana (el cierre normal de desktop.py apaga el backend)."""
        self._quitting = True
        self.stop()
        try:
            self._on_quit()
        except Exception:
            logger.exception("No se pudo cerrar la ventana desde la bandeja")

    def _on_close_requested(self, window: Any) -> bool:
        """Manejador del evento `closing` de pywebview.

        Devolver False cancela el cierre (pywebview 5.4, winforms):
        la ventana se oculta y la app sigue en la bandeja. Cualquier
        otro valor deja que el cierre siga su curso.
        """
        if self._quitting:
            return True
        try:
            minimize = self._minimize_to_tray()
        except Exception:
            logger.exception("No se pudo leer la preferencia de bandeja")
            minimize = True
        if not minimize:
            return True
        self.hide_window()
        return False

    def _attach_close_handler(self) -> None:
        """Engancha el "cerrar minimiza" al evento `closing` de la
        ventana, si esta versión de pywebview lo expone."""
        events = getattr(self._window, "events", None)
        closing = getattr(events, "closing", None) if events is not None else None
        if closing is None:
            logger.warning(
                "Esta versión de pywebview no expone el evento de cierre; "
                "cerrar la ventana cerrará la app aunque la bandeja esté activa."
            )
            return
        try:
            closing.__iadd__(self._on_close_requested)
        except Exception:
            logger.exception("No se pudo enganchar el cierre a la bandeja")

    def _default_icon_factory(self, name: str, image: Any, title: str, menu: Any) -> Any:
        import pystray

        return pystray.Icon(name, image, title, menu)

    def start(self) -> bool:
        """Crea el icono y arranca su bucle en un hilo daemon.

        Devuelve False si pystray o Pillow no están disponibles (la
        app sigue sin bandeja) o si el icono no se pudo crear. Con
        una factoría inyectada (tests), el menú es una lista simple
        y no hace falta importar pystray.
        """
        try:
            image = build_tray_image()
        except ImportError:
            logger.warning("Pillow no está disponible; la app arranca sin bandeja.")
            return False
        try:
            if self._icon_factory is not None:
                menu: Any = [
                    (self._menu_label, self._toggle_window),
                    ("Abrir carpeta de clips", self.open_clips_folder),
                    None,
                    ("Salir", self.quit_app),
                ]
                factory = self._icon_factory
            else:
                import pystray

                menu = pystray.Menu(
                    pystray.MenuItem(self._menu_label, self._toggle_window, default=True),
                    pystray.MenuItem("Abrir carpeta de clips", self.open_clips_folder),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Salir", self.quit_app),
                )
                factory = self._default_icon_factory
            self._icon = factory("kellyClipAi", image, "kellyClipAi", menu)
        except ImportError:
            logger.warning("pystray no está disponible; la app arranca sin bandeja.")
            self._icon = None
            return False
        except Exception:
            logger.exception("No se pudo crear el icono de la bandeja")
            self._icon = None
            return False
        self._attach_close_handler()
        self._thread = threading.Thread(
            target=self._icon.run,
            name="kellyclipai-tray",
            daemon=True,
        )
        self._thread.start()
        logger.info("Bandeja del sistema activa.")
        return True

    def stop(self) -> None:
        """Detiene el icono de la bandeja (idempotente)."""
        icon, self._icon = self._icon, None
        if icon is None:
            return
        try:
            icon.stop()
        except Exception:
            logger.exception("No se pudo detener el icono de la bandeja")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
