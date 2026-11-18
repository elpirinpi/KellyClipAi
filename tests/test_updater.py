from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.updater as updater_module
from backend.updater import UpdateError, UpdateManager, _version_tuple
from backend.main import create_app
from backend.settings import AppSettings


def _release(version: str) -> dict:
    asset_name = f"KellyClipAi-Setup-{version}.exe"
    return {
        "tag_name": f"v{version}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": asset_name,
                "browser_download_url": (
                    f"https://github.com/elpirinpi/KellyClipAi/releases/download/"
                    f"v{version}/{asset_name}"
                ),
                "digest": f"sha256:{'a' * 64}",
                "size": 123,
            }
        ],
    }


def test_update_manager_detects_a_new_stable_windows_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: _release("1.0.4"))
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    initial = manager.start_check()
    assert initial["state"] == "checking"
    assert manager._thread is not None
    manager._thread.join(timeout=2)
    status = manager.status()

    assert status["state"] == "available"
    assert status["update_available"] is True
    assert status["latest_version"] == "1.0.4"


def test_update_manager_reports_current_when_no_newer_release_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: _release("1.0.3"))
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["state"] == "current"
    assert manager.status()["update_available"] is False


def test_update_manager_rejects_untrusted_installer_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _release("1.0.4")
    release["assets"][0]["browser_download_url"] = "https://example.com/installer.exe"
    monkeypatch.setattr(updater_module, "_request_json", lambda _url: release)
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True

    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["state"] == "error"
    assert "no es confiable" in manager.status()["error"]


def test_update_download_verifies_github_sha256_before_marking_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"verified installer bytes"
    release = _release("1.0.4")
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
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    installer = tmp_path / "updates" / "KellyClipAi-Setup-1.0.4.exe"
    assert manager.status()["state"] == "downloaded"
    assert installer.read_bytes() == payload


def test_version_parser_rejects_non_release_versions() -> None:
    with pytest.raises(UpdateError, match="versión no válida"):
        _version_tuple("main")


def test_updates_are_reported_unavailable_outside_the_installed_app(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    manager = UpdateManager("1.0.4", tmp_path, None)

    with TestClient(create_app(settings, auth_required=False, updater=manager)) as client:
        status = client.get("/api/updates")
        unavailable = client.post("/api/updates/download")

    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert unavailable.status_code == 409
