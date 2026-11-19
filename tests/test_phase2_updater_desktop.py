from __future__ import annotations

import hashlib
import http.server
import io
import logging
import os
import socket
import sys
import threading
import types
import urllib.error
from pathlib import Path

import httpx
import pytest

import backend.updater as updater_module
import desktop as desktop_module
from backend import integrations
from backend.integrations import IntegrationError
from backend.updater import UpdateManager

# ---------------------------------------------------------------------------
# Ayudas compartidas
# ---------------------------------------------------------------------------


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    client_type = httpx.Client
    monkeypatch.setattr(
        integrations.httpx,
        "Client",
        lambda *args, **kwargs: client_type(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )


def _no_upload_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fragmentos pequeños y sin esperas para que las subidas de prueba
    sean rápidas y deterministas."""
    monkeypatch.setattr(integrations, "_UPLOAD_CHUNK_SIZE", 10)
    monkeypatch.setattr(
        integrations, "_upload_retry_delay", lambda *_args, **_kwargs: 0.0
    )


_YOUTUBE_LOCATION = "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=test"


def _youtube_clip() -> dict:
    return {
        "title": "Un consejo",
        "description": "Un consejo práctico para hoy.",
        "hashtags": {"youtube": ["Consejos"]},
        "keywords": ["consejo"],
    }


def _instagram_clip() -> dict:
    return {
        "title": "Un consejo",
        "description": "Una idea útil",
        "hashtags": {"instagram": ["Ideas"]},
    }


# ---------------------------------------------------------------------------
# Ítem 11 — Descarga reanudable del actualizador
# ---------------------------------------------------------------------------


def _release_for(payload: bytes, version: str = "1.0.4") -> dict:
    asset_name = f"KellyClipAi-Setup-{version}.exe"
    return {
        "tag_name": f"v{version}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": asset_name,
                "browser_download_url": (
                    "https://github.com/elpirinpi/KellyClipAi/releases/download/"
                    f"v{version}/{asset_name}"
                ),
                "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                "size": len(payload),
            }
        ],
    }


def _manager_with_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes,
) -> UpdateManager:
    monkeypatch.setattr(
        updater_module, "_request_json", lambda _url: _release_for(payload)
    )
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=5)
    assert manager.status()["state"] == "available"
    return manager


def _part_path(tmp_path: Path, version: str = "1.0.4") -> Path:
    return tmp_path / "updates" / f"KellyClipAi-Setup-{version}.exe.part"


def _target_path(tmp_path: Path, version: str = "1.0.4") -> Path:
    return tmp_path / "updates" / f"KellyClipAi-Setup-{version}.exe"


class _DownloadResponse(io.BytesIO):
    """Respuesta fake de urllib con status y cabeceras."""

    def __init__(
        self,
        payload: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_update_download_resumes_existing_part_with_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = bytes(range(256)) * 4
    manager = _manager_with_release(monkeypatch, tmp_path, payload)
    part = _part_path(tmp_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(payload[:512])
    seen_ranges: list[str | None] = []

    def fake_urlopen(request, **_kwargs):
        seen_ranges.append(request.get_header("Range"))
        return _DownloadResponse(
            payload[512:],
            status=206,
            headers={
                "Content-Length": "512",
                "Content-Range": "bytes 512-1023/1024",
            },
        )

    monkeypatch.setattr(updater_module.urllib.request, "urlopen", fake_urlopen)

    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert seen_ranges == ["bytes=512-"]
    assert manager.status()["state"] == "downloaded"
    assert _target_path(tmp_path).read_bytes() == payload
    assert not part.exists()


def test_update_download_restarts_from_zero_when_server_ignores_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = bytes(range(256)) * 4
    manager = _manager_with_release(monkeypatch, tmp_path, payload)
    part = _part_path(tmp_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"BASURA" * 10)

    def fake_urlopen(request, **_kwargs):
        assert request.get_header("Range") == "bytes=60-"
        return _DownloadResponse(payload, status=200)

    monkeypatch.setattr(updater_module.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.WARNING, logger="backend.updater"):
        manager.start_download()
        assert manager._thread is not None
        manager._thread.join(timeout=5)

    assert manager.status()["state"] == "downloaded"
    assert _target_path(tmp_path).read_bytes() == payload
    assert any(
        "no admite reanudación" in record.getMessage() for record in caplog.records
    )


def test_update_download_keeps_part_after_network_error_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = bytes(range(100))
    manager = _manager_with_release(monkeypatch, tmp_path, payload)

    class _FlakyResponse(_DownloadResponse):
        def __init__(self) -> None:
            super().__init__(payload, status=200)
            self._reads = 0

        def read(self, size: int = -1) -> bytes:
            self._reads += 1
            if self._reads == 1:
                return super().read(40)
            raise urllib.error.URLError("conexión cortada")

    monkeypatch.setattr(
        updater_module.urllib.request, "urlopen", lambda *_a, **_k: _FlakyResponse()
    )
    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert manager.status()["state"] == "error"
    assert manager.status()["progress"] == 40
    part = _part_path(tmp_path)
    assert part.read_bytes() == payload[:40]

    seen_ranges: list[str | None] = []

    def resume_urlopen(request, **_kwargs):
        seen_ranges.append(request.get_header("Range"))
        return _DownloadResponse(
            payload[40:],
            status=206,
            headers={
                "Content-Length": "60",
                "Content-Range": "bytes 40-99/100",
            },
        )

    monkeypatch.setattr(updater_module.urllib.request, "urlopen", resume_urlopen)
    resumed = manager.start_download()
    assert resumed["progress"] == 40
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert seen_ranges == ["bytes=40-"]
    assert manager.status()["state"] == "downloaded"
    assert _target_path(tmp_path).read_bytes() == payload


def test_update_download_promotes_complete_part_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"instalador completo"
    manager = _manager_with_release(monkeypatch, tmp_path, payload)
    part = _part_path(tmp_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(payload)

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("No debe descargar nada con el .part completo")

    monkeypatch.setattr(updater_module.urllib.request, "urlopen", fail_urlopen)

    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert manager.status()["state"] == "downloaded"
    assert _target_path(tmp_path).read_bytes() == payload
    assert not part.exists()


def test_update_download_discards_corrupt_complete_part(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"instalador correcto"
    manager = _manager_with_release(monkeypatch, tmp_path, payload)
    part = _part_path(tmp_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"X" * len(payload))

    monkeypatch.setattr(
        updater_module.urllib.request,
        "urlopen",
        lambda *_a, **_k: _DownloadResponse(payload, status=200),
    )

    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert manager.status()["state"] == "downloaded"
    assert _target_path(tmp_path).read_bytes() == payload


def test_update_download_rejects_mismatched_content_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = bytes(range(256)) * 4
    manager = _manager_with_release(monkeypatch, tmp_path, payload)
    part = _part_path(tmp_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(payload[:512])

    monkeypatch.setattr(
        updater_module.urllib.request,
        "urlopen",
        lambda *_a, **_k: _DownloadResponse(
            payload[512:],
            status=206,
            headers={
                "Content-Length": "512",
                "Content-Range": "bytes 0-511/1024",
            },
        ),
    )

    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=5)

    assert manager.status()["state"] == "error"
    assert "no reanudó" in manager.status()["error"]
    # El .part se conserva para reintentar la reanudación.
    assert part.read_bytes() == payload[:512]
    assert manager.status()["progress"] == 50


def test_parse_content_range() -> None:
    assert updater_module._parse_content_range("bytes 10-19/100") == (10, 19, 100)
    assert updater_module._parse_content_range("bytes 0-9/*") == (0, 9, None)
    assert updater_module._parse_content_range("bytes 512-1023/1024") == (
        512,
        1023,
        1024,
    )
    assert updater_module._parse_content_range("bogus") is None
    assert updater_module._parse_content_range(None) is None


# ---------------------------------------------------------------------------
# Ítem 9 — Subida YouTube por chunks con reanudación
# ---------------------------------------------------------------------------


def test_upload_chunk_size_is_a_multiple_of_256kib() -> None:
    assert integrations._UPLOAD_CHUNK_SIZE % (256 * 1024) == 0


def test_upload_retry_delay_honors_retry_after() -> None:
    response = httpx.Response(429, headers={"Retry-After": "7"})
    assert integrations._upload_retry_delay(0, response) == 7.0
    capped = httpx.Response(503, headers={"Retry-After": "600"})
    assert integrations._upload_retry_delay(0, capped) == 60.0
    delay = integrations._upload_retry_delay(0)
    assert 1.0 <= delay <= 1.5


def test_youtube_uploads_in_chunks_with_content_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = bytes(range(25))
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(data)
    ranges: list[str] = []
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": _YOUTUBE_LOCATION})
        content_range = request.headers["Content-Range"]
        ranges.append(content_range)
        bodies.append(bytes(request.content))
        end = int(content_range.split("-")[1].split("/")[0])
        if end == 24:
            return httpx.Response(200, json={"id": "yt-1"})
        return httpx.Response(308, headers={"Range": f"bytes=0-{end}"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_youtube(
        "access-token", _youtube_clip(), file_path, "public"
    )

    assert result == {
        "external_id": "yt-1",
        "url": "https://www.youtube.com/watch?v=yt-1",
    }
    assert ranges == ["bytes 0-9/25", "bytes 10-19/25", "bytes 20-24/25"]
    assert bodies == [data[0:10], data[10:20], data[20:25]]


def test_youtube_upload_resumes_after_network_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = bytes(range(25))
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(data)
    ranges: list[str] = []
    queries: list[str] = []
    chunk2_calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal chunk2_calls
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": _YOUTUBE_LOCATION})
        content_range = request.headers["Content-Range"]
        if content_range == "bytes */25":
            queries.append(content_range)
            return httpx.Response(308, headers={"Range": "bytes=0-9"})
        ranges.append(content_range)
        if content_range == "bytes 10-19/25":
            chunk2_calls += 1
            if chunk2_calls == 1:
                raise httpx.ConnectError("corte de red", request=request)
            return httpx.Response(308, headers={"Range": "bytes=0-19"})
        if content_range == "bytes 0-9/25":
            return httpx.Response(308, headers={"Range": "bytes=0-9"})
        return httpx.Response(200, json={"id": "yt-2"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_youtube(
        "access-token", _youtube_clip(), file_path, "public"
    )

    assert result["external_id"] == "yt-2"
    # El fragmento 1 no se reenvía; el 2 se reenvía tras consultar al servidor.
    assert ranges == [
        "bytes 0-9/25",
        "bytes 10-19/25",
        "bytes 10-19/25",
        "bytes 20-24/25",
    ]
    assert queries == ["bytes */25"]


def test_youtube_upload_retries_on_500_and_skips_confirmed_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = bytes(range(25))
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(data)
    ranges: list[str] = []
    chunk2_seen = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal chunk2_seen
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": _YOUTUBE_LOCATION})
        content_range = request.headers["Content-Range"]
        if content_range == "bytes */25":
            # El servidor sí recibió el fragmento 2 aunque respondió 500.
            return httpx.Response(308, headers={"Range": "bytes=0-19"})
        ranges.append(content_range)
        if content_range == "bytes 10-19/25" and not chunk2_seen:
            chunk2_seen = True
            return httpx.Response(500, json={"error": {"message": "boom"}})
        if content_range == "bytes 0-9/25":
            return httpx.Response(308, headers={"Range": "bytes=0-9"})
        return httpx.Response(200, json={"id": "yt-3"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_youtube(
        "access-token", _youtube_clip(), file_path, "public"
    )

    assert result["external_id"] == "yt-3"
    # El fragmento 2 se envió una sola vez: la consulta confirmó que llegó.
    assert ranges == ["bytes 0-9/25", "bytes 10-19/25", "bytes 20-24/25"]


def test_youtube_upload_fails_when_session_expires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(25)))
    ranges: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": _YOUTUBE_LOCATION})
        content_range = request.headers["Content-Range"]
        if content_range == "bytes */25":
            return httpx.Response(404, json={"error": {"message": "not found"}})
        ranges.append(content_range)
        if content_range == "bytes 0-9/25":
            return httpx.Response(308, headers={"Range": "bytes=0-9"})
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    with pytest.raises(IntegrationError, match="expiró"):
        integrations._publish_youtube(
            "access-token", _youtube_clip(), file_path, "public"
        )

    assert ranges == ["bytes 0-9/25", "bytes 10-19/25"]


def test_youtube_upload_completes_when_status_query_reports_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(20)))

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, headers={"Location": _YOUTUBE_LOCATION})
        content_range = request.headers["Content-Range"]
        if content_range == "bytes */20":
            # El último fragmento llegó aunque la respuesta se perdió.
            return httpx.Response(200, json={"id": "yt-5"})
        if content_range == "bytes 0-9/20":
            return httpx.Response(308, headers={"Range": "bytes=0-9"})
        raise httpx.ConnectError("corte de red", request=request)

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_youtube(
        "access-token", _youtube_clip(), file_path, "public"
    )

    assert result["external_id"] == "yt-5"


def test_youtube_upload_rejects_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"")

    with pytest.raises(IntegrationError, match="vacío"):
        integrations._publish_youtube(
            "access-token", _youtube_clip(), file_path, "public"
        )


# ---------------------------------------------------------------------------
# Ítem 9 — Subida Instagram por chunks con reanudación
# ---------------------------------------------------------------------------


def test_instagram_uploads_in_chunks_with_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(25)))
    offsets: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "rupload.facebook.com":
            offsets.append(request.headers["offset"])
            assert request.headers["file_size"] == "25"
            return httpx.Response(200, json={"success": True})
        if request.url.path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "ig-1"})
        if request.method == "GET":
            return httpx.Response(200, json={"status_code": "FINISHED"})
        return httpx.Response(200, json={"id": "container-1"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_instagram(
        "access-token",
        {"account_id": "instagram-account"},
        _instagram_clip(),
        file_path,
    )

    assert result == {"external_id": "ig-1", "url": ""}
    assert offsets == ["0", "10", "20"]


def test_instagram_upload_resumes_from_bytes_transfered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(25)))
    offsets: list[str] = []
    chunk2_calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal chunk2_calls
        if request.url.host == "rupload.facebook.com":
            offsets.append(request.headers["offset"])
            if request.headers["offset"] == "10" and chunk2_calls == 0:
                chunk2_calls += 1
                raise httpx.ConnectError("corte de red", request=request)
            return httpx.Response(200, json={"success": True})
        if request.url.path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "ig-2"})
        if request.method == "GET":
            if "video_status" in request.url.params.get("fields", ""):
                return httpx.Response(
                    200,
                    json={
                        "status_code": "IN_PROGRESS",
                        "video_status": {"uploading_phase": {"bytes_transfered": 10}},
                    },
                )
            return httpx.Response(200, json={"status_code": "FINISHED"})
        return httpx.Response(200, json={"id": "container-1"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_instagram(
        "access-token",
        {"account_id": "instagram-account"},
        _instagram_clip(),
        file_path,
    )

    assert result == {"external_id": "ig-2", "url": ""}
    assert offsets == ["0", "10", "10", "20"]


def test_instagram_upload_retries_on_429_and_falls_back_to_last_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(25)))
    offsets: list[str] = []
    chunk2_calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal chunk2_calls
        if request.url.host == "rupload.facebook.com":
            offsets.append(request.headers["offset"])
            if request.headers["offset"] == "10" and chunk2_calls == 0:
                chunk2_calls += 1
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json={"success": True})
        if request.url.path.endswith("/media_publish"):
            return httpx.Response(200, json={"id": "ig-3"})
        if request.method == "GET":
            # Sin video_status: se reenvía desde el último offset confirmado.
            return httpx.Response(200, json={"status_code": "FINISHED"})
        return httpx.Response(200, json={"id": "container-1"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    result = integrations._publish_instagram(
        "access-token",
        {"account_id": "instagram-account"},
        _instagram_clip(),
        file_path,
    )

    assert result == {"external_id": "ig-3", "url": ""}
    assert offsets == ["0", "10", "10", "20"]


def test_instagram_upload_fails_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(bytes(range(25)))
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.host == "rupload.facebook.com":
            attempts += 1
            return httpx.Response(500, json={"error": {"message": "boom"}})
        if request.method == "GET":
            return httpx.Response(200, json={"status_code": "IN_PROGRESS"})
        return httpx.Response(200, json={"id": "container-1"})

    _no_upload_delays(monkeypatch)
    _install_transport(monkeypatch, handle)

    with pytest.raises(IntegrationError, match="tras varios intentos"):
        integrations._publish_instagram(
            "access-token",
            {"account_id": "instagram-account"},
            _instagram_clip(),
            file_path,
        )

    assert attempts == integrations._UPLOAD_MAX_ATTEMPTS


def test_instagram_upload_rejects_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"")

    with pytest.raises(IntegrationError, match="vacío"):
        integrations._publish_instagram(
            "access-token",
            {"account_id": "instagram-account"},
            _instagram_clip(),
            file_path,
        )


# ---------------------------------------------------------------------------
# Ítem 12a — Puerto de respaldo automático
# ---------------------------------------------------------------------------


def _occupy_ports(first: int, last: int) -> list[socket.socket]:
    """Ocupa los puertos indicados con sockets en escucha. Si un puerto ya
    está ocupado por otro proceso, igual cuenta como ocupado."""
    blockers: list[socket.socket] = []
    for port in range(first, last + 1):
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", port))
            blocker.listen(1)
        except OSError:
            blocker.close()
            continue
        blockers.append(blocker)
    return blockers


def _close_sockets(blockers: list[socket.socket]) -> None:
    for blocker in blockers:
        blocker.close()


def test_find_available_port_skips_occupied_ports() -> None:
    blockers = _occupy_ports(8000, 8001)
    try:
        assert desktop_module._find_available_port("127.0.0.1") == 8002
    finally:
        _close_sockets(blockers)


def test_is_port_free_reports_occupied_and_free_ports() -> None:
    blockers = _occupy_ports(8000, 8000)
    try:
        assert desktop_module._is_port_free("127.0.0.1", 8000) is False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        assert desktop_module._is_port_free("127.0.0.1", free_port) is True
    finally:
        _close_sockets(blockers)


def test_find_available_port_raises_when_range_exhausted() -> None:
    blockers = _occupy_ports(8000, 8010)
    try:
        with pytest.raises(RuntimeError, match="No hay puertos locales libres"):
            desktop_module._find_available_port("127.0.0.1")
    finally:
        _close_sockets(blockers)


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_response(200)
        else:
            self.send_response(404)
        self.end_headers()

    def log_message(self, *_args) -> None:
        pass


def test_wait_for_backend_uses_given_port() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    stop = threading.Event()
    backend_thread = threading.Thread(target=stop.wait, args=(10,), daemon=True)
    backend_thread.start()
    try:
        desktop_module._wait_for_backend(backend_thread, port)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


def test_wait_for_backend_reports_dead_thread_fast() -> None:
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()

    with pytest.raises(RuntimeError, match="no inició correctamente"):
        desktop_module._wait_for_backend(dead, 59999)


def test_wait_for_backend_includes_startup_error() -> None:
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    startup_error: dict[str, BaseException | None] = {
        "error": RuntimeError("puerto ocupado (bind)")
    }

    with pytest.raises(RuntimeError, match="puerto ocupado"):
        desktop_module._wait_for_backend(dead, 59999, startup_error)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _fake_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ffmpeg.exe").write_bytes(b"fake")
    (tmp_path / "bin" / "ffprobe.exe").write_bytes(b"fake")
    monkeypatch.setattr(
        desktop_module, "_resource_path", lambda *parts: tmp_path.joinpath(*parts)
    )


def test_run_app_propagates_fallback_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_bundle(monkeypatch, tmp_path)
    blockers = _occupy_ports(8000, 8000)

    class _FakeConfig:
        def __init__(self, app, host=None, port=None, **kwargs) -> None:
            self.app = app
            self.host = host
            self.port = port
            self.kwargs = kwargs

    class _FakeServer:
        def __init__(self, config: _FakeConfig) -> None:
            self.config = config
            self.should_exit = False

        def run(self) -> None:
            return None

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.Server = _FakeServer
    fake_uvicorn.Config = _FakeConfig
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    created_windows: list[tuple[str, str]] = []
    started: list[dict] = []

    fake_webview = types.ModuleType("webview")
    fake_webview.create_window = lambda title, url, **kwargs: created_windows.append(
        (title, url)
    )
    fake_webview.start = lambda **kwargs: started.append(kwargs)
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    import backend.main as main_module
    import backend.settings as settings_module

    monkeypatch.setattr(main_module, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(settings_module, "get_settings", lambda: object())

    waited_ports: list[int] = []
    monkeypatch.setattr(
        desktop_module,
        "_wait_for_backend",
        lambda thread, port, error=None: waited_ports.append(port),
    )

    handler = _ListHandler()
    desktop_logger = logging.getLogger("desktop")
    desktop_logger.addHandler(handler)
    desktop_logger.setLevel(logging.INFO)
    saved_uvicorn = {}
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        saved_uvicorn[name] = (list(logger.handlers), logger.propagate)
    saved_env = {
        key: os.environ.get(key)
        for key in (
            "KELLYCLIP_OAUTH_REDIRECT_BASE_URL",
            "KELLYCLIP_FRONTEND_BASE_URL",
        )
    }
    try:
        desktop_module._run_app(tmp_path, False, handler)

        assert len(waited_ports) == 1
        port = waited_ports[0]
        assert port != 8000 and 8001 <= port <= 8010
        assert created_windows == [("kellyClipAi", f"http://127.0.0.1:{port}/")]
        assert started
        assert (
            os.environ["KELLYCLIP_OAUTH_REDIRECT_BASE_URL"]
            == f"http://127.0.0.1:{port}"
        )
        assert os.environ["KELLYCLIP_FRONTEND_BASE_URL"] == f"http://127.0.0.1:{port}"
        assert any(
            "8000" in message and str(port) in message for message in handler.messages
        )
        assert any(
            f"http://127.0.0.1:{port}" in message for message in handler.messages
        )
        # El logging de uvicorn apunta al mismo handler del log principal.
        assert logging.getLogger("uvicorn.error").handlers == [handler]
    finally:
        _close_sockets(blockers)
        desktop_logger.removeHandler(handler)
        for name, (handlers, propagate) in saved_uvicorn.items():
            logger = logging.getLogger(name)
            logger.handlers = handlers
            logger.propagate = propagate
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_run_app_raises_when_no_port_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_bundle(monkeypatch, tmp_path)
    blockers = _occupy_ports(8000, 8010)
    try:
        with pytest.raises(RuntimeError, match="No hay puertos locales libres"):
            desktop_module._run_app(tmp_path, False)
    finally:
        _close_sockets(blockers)


# ---------------------------------------------------------------------------
# Ítem 12b — Errores de arranque en el log
# ---------------------------------------------------------------------------


def test_configure_uvicorn_logging_routes_to_desktop_log(tmp_path: Path) -> None:
    log_path = tmp_path / "desktop.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    saved = {}
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        saved[name] = (list(logger.handlers), logger.propagate)
    try:
        desktop_module._configure_uvicorn_logging(handler)
        logging.getLogger("uvicorn.error").error("fallo de arranque simulado")
        handler.flush()

        assert "fallo de arranque simulado" in log_path.read_text(encoding="utf-8")
        assert logging.getLogger("uvicorn.error").handlers == [handler]
        assert logging.getLogger("uvicorn.error").propagate is False
    finally:
        for name, (handlers, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers = handlers
            logger.propagate = propagate
        handler.close()


def test_run_server_logs_exception_and_records_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Boom:
        def run(self) -> None:
            raise RuntimeError("bind fallido")

    holder: dict[str, BaseException | None] = {"error": None}
    with caplog.at_level(logging.ERROR, logger="desktop"):
        desktop_module._run_server(_Boom(), holder)

    assert isinstance(holder["error"], RuntimeError)
    assert "bind fallido" in str(holder["error"])
    assert any(
        "El servidor local terminó con un error" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        record.exc_text and "bind fallido" in record.exc_text
        for record in caplog.records
    )


def test_run_server_success_leaves_no_error() -> None:
    class _Ok:
        def run(self) -> None:
            return None

    holder: dict[str, BaseException | None] = {"error": None}
    desktop_module._run_server(_Ok(), holder)
    assert holder["error"] is None


def test_log_thread_exception_logs_thread_name_and_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    try:
        raise ValueError("fallo en hilo")
    except ValueError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
    # ExceptHookArgs es un structseq y no se construye con 4 argumentos;
    # un namespace con los mismos atributos basta para la función.
    args = types.SimpleNamespace(
        exc_type=exc_type,
        exc_value=exc_value,
        exc_traceback=exc_traceback,
        thread=threading.current_thread(),
    )

    with caplog.at_level(logging.ERROR, logger="desktop"):
        desktop_module._log_thread_exception(args)

    assert any("MainThread" in record.getMessage() for record in caplog.records)
    assert any(
        record.exc_text and "fallo en hilo" in record.exc_text
        for record in caplog.records
    )
