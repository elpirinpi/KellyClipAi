"""Aceleración GPU del host de escritorio (v1.4.2).

La ventana es pywebview 5.4 con gui="edgechromium" (WebView2), que no
expone flags de Chromium en webview.start(): los argumentos viajan por
la variable de entorno WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS, que
WebView2 lee al crear su entorno. Estos tests comprueban que la lista
solo trae flags seguros de aceleración (nada que fuerce render por
software ni que ignore la lista de bloqueo de GPUs) y que un valor
definido por el usuario se respeta.
"""

from __future__ import annotations

import os

import pytest

import desktop as desktop_module


def test_browser_arguments_enable_gpu_rasterization() -> None:
    args = desktop_module._webview2_browser_arguments().split()
    assert "--enable-gpu-rasterization" in args
    assert "--enable-zero-copy" in args


def test_browser_arguments_never_force_software_rendering() -> None:
    args = desktop_module._webview2_browser_arguments().split()
    for forbidden in desktop_module._SOFTWARE_RENDER_ARGUMENTS:
        assert forbidden not in args


def test_browser_arguments_do_not_ignore_gpu_blocklist() -> None:
    # --ignore-gpu-blocklist forzaría la GPU en drivers en lista de
    # bloqueo y podría colgar la ventana: el fallback a software de
    # WebView2 es el comportamiento seguro en esos equipos.
    args = desktop_module._webview2_browser_arguments().split()
    assert "--ignore-gpu-blocklist" not in args
    assert "--disable-gpu-driver-bug-workarounds" not in args


def test_browser_arguments_are_all_flags() -> None:
    # Cada elemento es un flag Chromium (--algo), sin valores pegados
    # ni espacios que WebView2 interpretaría mal.
    for arg in desktop_module._webview2_browser_arguments().split():
        assert arg.startswith("--")
        assert "=" not in arg or arg.split("=", 1)[0].startswith("--")


def test_configure_sets_env_var_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(desktop_module._WEBVIEW2_ARGS_ENV_VAR, raising=False)
    desktop_module._configure_webview2_arguments()
    assert (
        os.environ[desktop_module._WEBVIEW2_ARGS_ENV_VAR]
        == desktop_module._webview2_browser_arguments()
    )


def test_configure_respects_user_defined_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Un usuario (o su sistema) puede necesitar forzar software en un
    # equipo con drivers rotos: su valor no se pisa.
    monkeypatch.setenv(desktop_module._WEBVIEW2_ARGS_ENV_VAR, "--disable-gpu")
    desktop_module._configure_webview2_arguments()
    assert os.environ[desktop_module._WEBVIEW2_ARGS_ENV_VAR] == "--disable-gpu"
