"""Fase 3 — Escritorio y distribución (ítems 17–20, C.6.36–C.6.42).

La bandeja, el registro de Windows y el instalador no se pueden
ejercer en esta VM Linux: esos caminos se prueban con dobles (ventana
e icono falsos) y con los guardas de plataforma, y quedan marcados
como pendientes de verificación en Windows en el informe.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.updater as updater_module
import desktop as desktop_module
import desktop_tray as tray_module
from backend.autostart import (
    AutostartError,
    autostart_supported,
    is_autostart_enabled,
    set_autostart,
)
from backend.desktop_prefs import (
    load_desktop_prefs,
    save_desktop_prefs,
    tray_available,
)
from backend.first_run import (
    WHISPER_MODELS,
    disk_space,
    downloaded_models,
    is_setup_complete,
    mark_setup_complete,
    model_fits,
    needs_setup,
    whisper_model_from_setup,
)
from backend.main import create_app
from backend.settings import AppSettings
from backend.updater import UpdateError, UpdateManager, _version_key
from desktop_tray import TrayController, build_tray_image

_ENV_KEYS = (
    "KELLYCLIP_PORTABLE",
    "KELLYCLIP_ENV_FILE",
    "KELLYCLIP_DATA_DIR",
    "KELLYCLIP_OAUTH_REDIRECT_BASE_URL",
    "KELLYCLIP_FRONTEND_BASE_URL",
    "PATH",
    "LOCALAPPDATA",
)


@contextmanager
def _saved_env() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )


def _release(
    version: str,
    *,
    prerelease: bool = False,
    draft: bool = False,
    with_asset: bool = True,
    notes: str | None = None,
) -> dict[str, Any]:
    asset_name = f"KellyClipAi-Setup-{version}.exe"
    release: dict[str, Any] = {
        "tag_name": f"v{version}",
        "draft": draft,
        "prerelease": prerelease,
        "html_url": f"https://github.com/elpirinpi/KellyClipAi/releases/tag/v{version}",
        "body": notes,
        "assets": [],
    }
    if with_asset:
        release["assets"] = [
            {
                "name": asset_name,
                "browser_download_url": (
                    "https://github.com/elpirinpi/KellyClipAi/releases/download/"
                    f"v{version}/{asset_name}"
                ),
                "digest": f"sha256:{'a' * 64}",
                "size": 123,
            }
        ]
    return release


# ---------------------------------------------------------------------------
# Ítem 17 — Modo portable (C.6.36)
# ---------------------------------------------------------------------------


def test_portable_mode_detects_marker_file_and_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KELLYCLIP_PORTABLE", raising=False)
    assert desktop_module._is_portable_mode(tmp_path) is False

    monkeypatch.setenv("KELLYCLIP_PORTABLE", "1")
    assert desktop_module._is_portable_mode(tmp_path) is True

    monkeypatch.delenv("KELLYCLIP_PORTABLE", raising=False)
    (tmp_path / "portable.txt").write_text("modo portable", encoding="utf-8")
    assert desktop_module._is_portable_mode(tmp_path) is True


def test_prepare_user_directory_uses_app_folder_in_portable_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "portable.txt").write_text("modo portable", encoding="utf-8")
    monkeypatch.delenv("KELLYCLIP_PORTABLE", raising=False)
    with _saved_env():
        user_dir = desktop_module._prepare_user_directory(tmp_path)

        assert user_dir == tmp_path
        assert (tmp_path / ".env").is_file()
        assert os.environ["KELLYCLIP_DATA_DIR"] == str(tmp_path / "data")
        assert os.environ["KELLYCLIP_PORTABLE"] == "1"


def test_prepare_user_directory_uses_localappdata_when_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KELLYCLIP_PORTABLE", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    with _saved_env():
        user_dir = desktop_module._prepare_user_directory(tmp_path)

    assert user_dir == tmp_path / "kellyClipAi"
    assert (user_dir / ".env").is_file()
    assert os.environ.get("KELLYCLIP_PORTABLE") != "1"


def test_portable_mutex_differs_per_folder(tmp_path: Path) -> None:
    installed = desktop_module._instance_mutex_name(False, tmp_path)
    portable_a = desktop_module._instance_mutex_name(True, tmp_path / "a")
    portable_b = desktop_module._instance_mutex_name(True, tmp_path / "b")

    assert installed == "Local\\kellyClipAi-desktop"
    assert portable_a != installed
    assert portable_a != portable_b
    assert portable_a == desktop_module._instance_mutex_name(True, tmp_path / "a")


# ---------------------------------------------------------------------------
# Ítem 18 — Auto-inicio y preferencias de escritorio (C.6.38, C.6.39)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="En Windows el auto-inicio sí está disponible.")
def test_autostart_is_unavailable_outside_windows(tmp_path: Path) -> None:
    assert autostart_supported() is False
    assert is_autostart_enabled(tmp_path / "KellyClipAi.exe") is False
    with pytest.raises(AutostartError):
        set_autostart(True, tmp_path / "KellyClipAi.exe")


def test_desktop_prefs_default_roundtrip_and_invalid_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    assert load_desktop_prefs(data_dir) == {
        "minimize_to_tray": True,
        "start_minimized": False,
    }

    saved = save_desktop_prefs(data_dir, {"start_minimized": True})
    assert saved == {"minimize_to_tray": True, "start_minimized": True}
    assert load_desktop_prefs(data_dir) == saved

    # Claves desconocidas y valores no booleanos se ignoran.
    saved = save_desktop_prefs(
        data_dir,
        {"start_minimized": "yes", "unknown": True},  # type: ignore[dict-item]
    )
    assert saved["start_minimized"] is True

    (data_dir / "desktop.json").write_text("{no json", encoding="utf-8")
    assert load_desktop_prefs(data_dir) == {
        "minimize_to_tray": True,
        "start_minimized": False,
    }


@pytest.mark.skipif(os.name == "nt", reason="En Windows la bandeja puede estar disponible.")
def test_tray_is_reported_unavailable_outside_windows() -> None:
    available, reason = tray_available()
    assert available is False
    assert reason is not None and "Windows" in reason


# ---------------------------------------------------------------------------
# Ítem 18 — Bandeja del sistema con dobles (C.6.38)
# ---------------------------------------------------------------------------


class _FakeClosingEvent:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeClosingEvent:
        self.handlers.append(handler)
        return self


class _FakeWindow:
    def __init__(self) -> None:
        self.events = type("Events", (), {"closing": _FakeClosingEvent()})()
        self.calls: list[str] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")

    def restore(self) -> None:
        self.calls.append("restore")

    def destroy(self) -> None:
        self.calls.append("destroy")


class _FakeIcon:
    def __init__(self, *args: Any) -> None:
        self.args = args
        self._stopped = threading.Event()
        self.runs = 0

    def run(self) -> None:
        self.runs += 1
        self._stopped.wait(timeout=10)

    def stop(self) -> None:
        self._stopped.set()


def _make_controller(
    tmp_path: Path, *, minimize: bool = True
) -> tuple[TrayController, _FakeWindow, list[str]]:
    window = _FakeWindow()
    quit_calls: list[str] = []
    created: list[_FakeIcon] = []

    def factory(*args: Any) -> _FakeIcon:
        icon = _FakeIcon(*args)
        created.append(icon)
        return icon

    controller = TrayController(
        window=window,
        clips_dir=tmp_path / "projects",
        on_quit=lambda: quit_calls.append("quit"),
        minimize_to_tray=lambda: minimize,
        icon_factory=factory,
    )
    controller._created_icons = created  # type: ignore[attr-defined]
    return controller, window, quit_calls


def test_tray_image_is_a_blue_square_with_a_white_play_triangle() -> None:
    pytest.importorskip("PIL")
    image = build_tray_image()

    assert image.size == (64, 64)
    assert image.mode == "RGBA"
    # Esquina transparente (fuera del cuadrado redondeado).
    assert image.getpixel((0, 0))[3] == 0
    # Centro: dentro del triángulo blanco.
    assert image.getpixel((32, 32)) == (255, 255, 255, 255)
    # Borde superior central: azul de la app.
    assert image.getpixel((32, 4)) == (15, 76, 156, 255)


def test_tray_close_hides_window_instead_of_quitting(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    controller, window, _quit = _make_controller(tmp_path, minimize=True)
    try:
        assert controller.start() is True
        assert controller.running is True
        handlers = window.events.closing.handlers
        assert len(handlers) == 1

        assert handlers[0](window) is False
        assert window.calls == ["hide"]
        assert controller.visible is False
    finally:
        controller.stop()


def test_tray_close_quits_when_preference_is_off(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    controller, window, _quit = _make_controller(tmp_path, minimize=False)
    try:
        assert controller.start() is True
        handlers = window.events.closing.handlers

        assert handlers[0](window) is True
        assert window.calls == []
    finally:
        controller.stop()


def test_tray_toggle_show_hide_and_quit(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    controller, window, quit_calls = _make_controller(tmp_path)
    try:
        assert controller.start() is True

        controller._toggle_window()
        assert window.calls == ["hide"]
        controller._toggle_window()
        assert window.calls == ["hide", "show", "restore"]

        controller.quit_app()
        assert quit_calls == ["quit"]
        assert controller.running is False
        # Tras salir, el cierre ya no se cancela.
        assert window.events.closing.handlers[0](window) is True
    finally:
        controller.stop()


def test_tray_open_clips_folder_creates_and_opens_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("PIL")
    opened: list[str] = []
    monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)
    controller, _window, _quit = _make_controller(tmp_path)
    try:
        assert controller.start() is True
        controller.open_clips_folder()
        assert opened == [str(tmp_path / "projects")]
        assert (tmp_path / "projects").is_dir()
    finally:
        controller.stop()


def test_tray_start_fails_gracefully_without_pillow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _missing_pillow(*args: Any, **kwargs: Any) -> None:
        raise ImportError("No module named 'PIL'")

    monkeypatch.setattr(tray_module, "build_tray_image", _missing_pillow)
    controller, _window, _quit = _make_controller(tmp_path)
    assert controller.start() is False
    assert controller.running is False


# ---------------------------------------------------------------------------
# Ítem 19 — Asistente de primer arranque (C.6.42)
# ---------------------------------------------------------------------------


def test_needs_setup_skips_existing_users() -> None:
    assert needs_setup(completed=False, has_projects=False, has_llm_key=False) is True
    assert needs_setup(completed=True, has_projects=False, has_llm_key=False) is False
    assert needs_setup(completed=False, has_projects=True, has_llm_key=False) is False
    assert needs_setup(completed=False, has_projects=False, has_llm_key=True) is False


def test_setup_completion_and_whisper_choice(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    assert is_setup_complete(data_dir) is False
    assert whisper_model_from_setup(data_dir) is None

    mark_setup_complete(data_dir, whisper_model="tiny")
    assert is_setup_complete(data_dir) is True
    assert whisper_model_from_setup(data_dir) == "tiny"

    (data_dir / "setup.json").write_text(
        json.dumps({"completed": True, "whisper_model": "base"}), encoding="utf-8"
    )
    assert whisper_model_from_setup(data_dir) is None


def test_whisper_model_sizes_and_disk_fit() -> None:
    models = {model.name: model for model in WHISPER_MODELS}
    assert set(models) == {"tiny", "small", "medium", "large-v3"}
    assert models["small"].size_bytes == 466 * 1024 * 1024

    one_gib = 1024 * 1024 * 1024
    assert model_fits(one_gib, "tiny") is False  # tiny + 1 GiB de margen
    assert model_fits(2 * one_gib, "tiny") is True
    assert model_fits(2 * one_gib, "large-v3") is False
    assert model_fits(5 * one_gib, "large-v3") is True
    assert model_fits(None, "large-v3") is True  # Sin medida no se bloquea.
    assert model_fits(5 * one_gib, "base") is False


def test_disk_space_reports_tmp_directory(tmp_path: Path) -> None:
    space = disk_space(tmp_path)
    assert isinstance(space["free_bytes"], int) and space["free_bytes"] > 0
    assert isinstance(space["total_bytes"], int) and space["total_bytes"] > 0


def test_downloaded_models_returns_a_set() -> None:
    assert isinstance(downloaded_models(), set)


# ---------------------------------------------------------------------------
# Ítem 20 — Canal beta + notas de versión (C.6.40)
# ---------------------------------------------------------------------------


def test_prerelease_version_ordering() -> None:
    assert _version_key("1.4.0-beta.1") < _version_key("1.4.0-beta.2")
    assert _version_key("1.4.0-beta.2") < _version_key("1.4.0-rc.1")
    assert _version_key("1.4.0-rc.1") < _version_key("1.4.0")
    assert _version_key("1.4.0-beta.1") > _version_key("1.3.2")
    assert _version_key("1.3.2") == _version_key("v1.3.2")
    with pytest.raises(UpdateError):
        _version_key("main")


def test_beta_channel_accepts_prereleases_with_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        updater_module,
        "_request_json_list",
        lambda _url: [
            _release("1.4.0-beta.1", prerelease=True, notes="## Novedades\n- Algo nuevo"),
        ],
    )
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.set_channel("beta")

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)
    status = manager.status()

    assert status["state"] == "available"
    assert status["latest_version"] == "1.4.0-beta.1"
    assert status["is_prerelease"] is True
    assert status["channel"] == "beta"
    assert status["release_notes"] == "## Novedades\n- Algo nuevo"
    assert status["release_url"] is not None and "1.4.0-beta.1" in status["release_url"]


def test_beta_channel_skips_drafts_and_releases_without_installer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        updater_module,
        "_request_json_list",
        lambda _url: [
            _release("1.4.0-beta.2", prerelease=True, draft=True),
            _release("1.4.0-beta.1", prerelease=True, with_asset=False),
            _release("1.3.9", notes="Estable anterior"),
        ],
    )
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.set_channel("beta")

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["latest_version"] == "1.3.9"
    assert manager.status()["is_prerelease"] is False


def test_stable_channel_still_rejects_prereleases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        updater_module,
        "_request_json",
        lambda _url: _release("1.4.0-beta.1", prerelease=True),
    )
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["state"] == "error"
    assert "no es una versión estable" in manager.status()["error"]


def test_stable_update_includes_release_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        updater_module,
        "_request_json",
        lambda _url: _release("1.3.3", notes="Corrección del atasco."),
    )
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["release_notes"] == "Corrección del atasco."
    assert manager.status()["is_prerelease"] is False


def test_channel_persists_and_resets_pending_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: _release("1.3.3"))
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)
    assert manager.status()["state"] == "available"

    status = manager.set_channel("beta")
    assert status["channel"] == "beta"
    assert status["state"] == "idle"
    assert status["update_available"] is False
    assert status["latest_version"] is None

    fresh = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    assert fresh.channel == "beta"

    with pytest.raises(UpdateError):
        manager.set_channel("nightly")


# ---------------------------------------------------------------------------
# Ítem 17 — Rollback de versiones (C.6.37)
# ---------------------------------------------------------------------------


def test_version_history_records_previous_version(tmp_path: Path) -> None:
    first = UpdateManager("1.3.1", tmp_path, tmp_path / "KellyClipAi.exe")
    assert first.rollback_info()["available"] is False

    second = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    assert second.rollback_info() == {
        "available": True,
        "previous_version": "1.3.1",
        "installer_ready": False,
    }
    assert second.status()["previous_version"] == "1.3.1"
    assert second.status()["rollback_available"] is True

    # Sin cambio de versión, el historial se conserva.
    third = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    assert third.rollback_info()["previous_version"] == "1.3.1"


def test_rollback_downloads_previous_installer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"previous installer bytes"
    release = _release("1.3.1")
    release["assets"][0]["size"] = len(payload)
    release["assets"][0]["digest"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: release)

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        updater_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )
    UpdateManager("1.3.1", tmp_path, tmp_path / "KellyClipAi.exe")
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    initial = manager.start_rollback()
    assert initial["state"] == "checking"
    assert initial["rollback"] is True
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    status = manager.status()
    assert status["state"] == "downloaded"
    assert status["rollback"] is True
    assert status["latest_version"] == "1.3.1"
    installer = tmp_path / "updates" / "KellyClipAi-Setup-1.3.1.exe"
    assert installer.read_bytes() == payload


def test_rollback_reuses_kept_installer_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"kept previous installer"
    release = _release("1.3.1")
    release["assets"][0]["size"] = len(payload)
    release["assets"][0]["digest"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: release)
    monkeypatch.setattr(
        updater_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no debe descargar")),
    )
    UpdateManager("1.3.1", tmp_path, tmp_path / "KellyClipAi.exe")
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    installer = tmp_path / "updates" / "KellyClipAi-Setup-1.3.1.exe"
    installer.parent.mkdir(parents=True, exist_ok=True)
    installer.write_bytes(payload)

    manager.start_rollback()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["state"] == "downloaded"
    assert manager.status()["rollback"] is True


def test_rollback_without_previous_version_raises(tmp_path: Path) -> None:
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    with pytest.raises(UpdateError, match="versión anterior"):
        manager.start_rollback()


def test_cleanup_keeps_current_version_installer(tmp_path: Path) -> None:
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    update_dir = tmp_path / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)
    current = update_dir / "KellyClipAi-Setup-1.3.2.exe"
    old = update_dir / "KellyClipAi-Setup-1.3.0.exe"
    new = update_dir / "KellyClipAi-Setup-1.3.3.exe"
    unrelated = update_dir / "notas.txt"
    for path in (current, old, new, unrelated):
        path.write_bytes(b"x")

    manager._cleanup_old_installers(keep=new)

    assert current.is_file()
    assert new.is_file()
    assert unrelated.is_file()
    assert not old.exists()


# ---------------------------------------------------------------------------
# Endpoints (Fase 3, ítems 17–20)
# ---------------------------------------------------------------------------


def test_setup_endpoints_complete_the_wizard(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, auth_required=False)) as client:
        initial = client.get("/api/setup")
        assert initial.status_code == 200
        body = initial.json()
        assert body["needs_setup"] is True
        assert body["completed"] is False
        assert len(body["whisper_models"]) == 4
        assert body["whisper_model"] == "small"
        assert isinstance(body["disk_free_bytes"], int)

        done = client.post("/api/setup/complete", json={"whisper_model": "tiny"})
        assert done.status_code == 200
        assert done.json() == {"completed": True}

        after = client.get("/api/setup")
        assert after.json()["needs_setup"] is False
        assert after.json()["whisper_model"] == "tiny"

        # El modelo del asistente alimenta también a Ajustes.
        settings_status = client.get("/api/settings")
        assert settings_status.json()["whisper_model"] == "tiny"

        invalid = client.post("/api/setup/complete", json={"whisper_model": "base"})
        assert invalid.status_code == 422


def test_update_channel_and_rollback_endpoints(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    manager = UpdateManager("1.3.2", tmp_path, tmp_path / "KellyClipAi.exe")
    with TestClient(create_app(settings, auth_required=False, updater=manager)) as client:
        channel = client.post("/api/updates/channel", json={"channel": "beta"})
        assert channel.status_code == 200
        assert channel.json()["channel"] == "beta"

        status = client.get("/api/updates")
        assert status.json()["channel"] == "beta"
        assert status.json()["release_notes"] is None
        assert status.json()["rollback_available"] is False

        rollback = client.get("/api/updates/rollback")
        assert rollback.json()["available"] is False

        started = client.post("/api/updates/rollback")
        assert started.status_code == 409

        invalid = client.post("/api/updates/channel", json={"channel": "nightly"})
        assert invalid.status_code == 422


def test_update_endpoints_without_updater_report_unavailable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, auth_required=False)) as client:
        status = client.get("/api/updates")
        body = status.json()
        assert body["enabled"] is False
        assert body["channel"] == "stable"
        assert body["rollback_available"] is False

        assert client.post("/api/updates/channel", json={"channel": "beta"}).status_code == 409
        assert client.post("/api/updates/rollback").status_code == 409
        assert client.get("/api/updates/rollback").json()["available"] is False


@pytest.mark.skipif(os.name == "nt", reason="En Windows el auto-inicio sí está disponible.")
def test_desktop_endpoints_on_linux(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings, auth_required=False)) as client:
        autostart = client.get("/api/desktop/autostart")
        assert autostart.json() == {"supported": False, "enabled": False}
        assert client.post("/api/desktop/autostart", json={"enabled": True}).status_code == 409

        prefs = client.get("/api/desktop/preferences")
        body = prefs.json()
        assert body["minimize_to_tray"] is True
        assert body["start_minimized"] is False
        assert body["tray_available"] is False
        assert body["tray_reason"] is not None
        assert body["portable"] is False

        updated = client.post("/api/desktop/preferences", json={"start_minimized": True})
        assert updated.status_code == 200
        assert updated.json()["start_minimized"] is True
        assert updated.json()["minimize_to_tray"] is True

        again = client.get("/api/desktop/preferences")
        assert again.json()["start_minimized"] is True

        invalid = client.post("/api/desktop/preferences", json={"unknown_pref": True})
        assert invalid.status_code == 422
