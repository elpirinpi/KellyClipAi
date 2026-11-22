"""Tests de Fase 1 (v1.0.11): reintento robusto del LLM, validación de
base_url, guarda anti-bucle en la subida TikTok, censura sin mutar la
transcripción y re-verificación SHA-256 del instalador."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from backend import integrations, llm as llm_module, updater as updater_module
from backend.llm import (
    LLMConfigurationError,
    LLMResponseError,
    ResolvedLLM,
    propose_clips,
)
from backend.media import _write_ass
from backend.schemas import (
    MAX_UPLOAD_BYTES,
    MAX_VIDEO_DURATION_SECONDS,
    ClipProposal,
    LLMConfig,
    LLMSettingsRequest,
    ProjectConfig,
    StagedUploadRequest,
)
from backend.updater import UpdateError, UpdateManager


# ---------------------------------------------------------------------------
# Ítem 5: límites de entrada (schemas)
# ---------------------------------------------------------------------------


def test_input_limit_constants_have_the_contracted_values() -> None:
    assert MAX_UPLOAD_BYTES == 4294967296
    assert MAX_VIDEO_DURATION_SECONDS == 14400


@pytest.mark.parametrize("size", [1, MAX_UPLOAD_BYTES])
def test_staged_upload_accepts_sizes_within_the_limit(size: int) -> None:
    request = StagedUploadRequest(filename="video.mp4", file_size=size)

    assert request.file_size == size


@pytest.mark.parametrize("size", [0, MAX_UPLOAD_BYTES + 1])
def test_staged_upload_rejects_sizes_outside_the_limit(size: int) -> None:
    with pytest.raises(ValidationError):
        StagedUploadRequest(filename="video.mp4", file_size=size)


# ---------------------------------------------------------------------------
# Ítem 9: validación de base_url del LLM
# ---------------------------------------------------------------------------

_VALID_BASE_URLS = [
    "https://api.groq.com/openai/v1",
    "https://api.openai.com/v1",
    "https://llm.example.com:8443/v1",
    "https://8.8.8.8/v1",
    "http://localhost:1234/v1",
    "http://localhost/v1",
    "http://127.0.0.1:1234/v1",
    "http://[::1]:1234/v1",
    "https://localhost:1234/v1",
]

_INVALID_BASE_URLS = [
    "http://api.example.com/v1",  # http público: la clave viajaría en claro
    "http://8.8.8.8/v1",
    "https://192.168.1.10/v1",  # IP privada
    "https://10.0.0.5/v1",
    "https://172.16.0.1/v1",
    "http://169.254.169.254/latest/meta-data/",  # link-local / metadatos
    "https://169.254.169.254/",
    "https://metadata.google.internal/computeMetadata/v1/",
    "https://0.0.0.0/v1",  # no especificada
    "https://224.0.0.1/v1",  # multicast
    "https://240.0.0.1/v1",  # reservada
    "api.groq.com/openai/v1",  # sin esquema
    "ftp://api.example.com/v1",
    "https://",  # sin host
]


@pytest.mark.parametrize("base_url", _VALID_BASE_URLS)
def test_llm_base_url_accepts_safe_urls(base_url: str) -> None:
    assert LLMConfig(base_url=base_url).base_url == base_url
    assert (
        LLMSettingsRequest(
            provider="Groq",
            model="qwen/qwen3.8-27b",
            base_url=base_url,
        ).base_url
        == base_url
    )


@pytest.mark.parametrize("base_url", _INVALID_BASE_URLS)
def test_llm_base_url_rejects_unsafe_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        LLMConfig(base_url=base_url)
    with pytest.raises(ValidationError):
        LLMSettingsRequest(
            provider="Groq",
            model="qwen/qwen3.8-27b",
            base_url=base_url,
        )


def test_llm_base_url_allows_empty_and_none_as_provider_default() -> None:
    assert LLMConfig(base_url=None).base_url is None
    assert LLMConfig(base_url="").base_url == ""
    assert LLMConfig(base_url="   ").base_url == "   "
    assert (
        LLMSettingsRequest(
            provider="Groq", model="qwen/qwen3.8-27b", base_url=""
        ).base_url
        == ""
    )


def test_llm_base_url_rejects_metadata_host_case_insensitively() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(base_url="https://METADATA.GOOGLE.INTERNAL/")


# ---------------------------------------------------------------------------
# Ítem 17: soporte backend para whisper_model en Ajustes
# ---------------------------------------------------------------------------


def test_llm_settings_request_accepts_whisper_model() -> None:
    request = LLMSettingsRequest(
        provider="Groq",
        model="qwen/qwen3.8-27b",
        whisper_model="medium",
    )

    assert request.whisper_model == "medium"


def test_llm_settings_request_defaults_whisper_model_to_none() -> None:
    request = LLMSettingsRequest(provider="Groq", model="qwen/qwen3.8-27b")

    assert request.whisper_model is None


def test_llm_settings_request_rejects_unknown_whisper_model() -> None:
    with pytest.raises(ValidationError):
        LLMSettingsRequest(
            provider="Groq",
            model="qwen/qwen3.8-27b",
            whisper_model="huge",
        )


# ---------------------------------------------------------------------------
# Ítem 6: reintento robusto del LLM
# ---------------------------------------------------------------------------


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    client_type = httpx.Client
    monkeypatch.setattr(
        llm_module.httpx,
        "Client",
        lambda *args, **kwargs: client_type(
            *args,
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )


def _llm() -> ResolvedLLM:
    return ResolvedLLM(
        "OpenAI", "test-token", "gpt-4o-mini", "https://llm.example"
    )


def _choices_response(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ]
        },
    )


def test_llm_retries_request_errors_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("conexión reseteada", request=request)
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    proposals = propose_clips([], ProjectConfig(), 60, _llm())

    assert proposals == []
    assert attempts == 3
    assert len(delays) == 2
    # Backoff exponencial 0.5 * 2**intento + jitter en [0, 0.5].
    assert 0.5 <= delays[0] <= 1.0
    assert 1.0 <= delays[1] <= 1.5


def test_llm_gives_up_after_repeated_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("timeout", request=request)

    _install_transport(monkeypatch, handle)

    with pytest.raises(
        LLMConfigurationError, match="No se pudo conectar con el proveedor"
    ):
        propose_clips([], ProjectConfig(), 60, _llm())

    assert attempts == 3


def test_llm_honors_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _llm()) == []
    assert delays == [2.0]


def test_llm_caps_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "600"})
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _llm()) == []
    assert delays == [60.0]


def test_llm_retries_invalid_json_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        requests.append(request)
        attempts += 1
        if attempts == 1:
            return _choices_response("esto no es JSON")
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _llm()) == []
    assert attempts == 2
    retry_payload = json.loads(requests[1].content)
    assert "no era un objeto JSON válido" in retry_payload["messages"][1]["content"]


def test_llm_raises_response_error_after_repeated_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _choices_response("todavía no es JSON")

    _install_transport(monkeypatch, handle)

    with pytest.raises(LLMResponseError, match="no devolvió un JSON válido"):
        propose_clips([], ProjectConfig(), 60, _llm())

    assert attempts == 3


def test_llm_retries_empty_response_as_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": []})

    _install_transport(monkeypatch, handle)

    with pytest.raises(LLMResponseError, match="no devolvió un JSON válido"):
        propose_clips([], ProjectConfig(), 60, _llm())

    assert attempts == 3


def test_llm_retries_truncated_response_asking_for_fewer_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        requests.append(request)
        attempts += 1
        if attempts == 1:
            return _choices_response(
                '{"clips": [{"inicio": 1, "fin": 4, "titulo": "truncado',
                finish_reason="length",
            )
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _llm()) == []
    assert attempts == 2
    retry_payload = json.loads(requests[1].content)
    assert "como máximo 3 clips" in retry_payload["messages"][1]["content"]


def test_llm_raises_response_error_when_response_keeps_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _choices_response(
            '{"clips": [{"inicio": 1',
            finish_reason="length",
        )

    _install_transport(monkeypatch, handle)

    with pytest.raises(LLMResponseError, match="truncó la respuesta"):
        propose_clips([], ProjectConfig(), 60, _llm())

    assert attempts == 3


def test_llm_response_error_is_exported_at_module_level() -> None:
    assert llm_module.LLMResponseError is LLMResponseError
    assert issubclass(LLMResponseError, RuntimeError)


# ---------------------------------------------------------------------------
# Ítem 6 (actualización): HTTP 400 json_validate_failed de Groq con Qwen3
# ---------------------------------------------------------------------------


def _groq_llm() -> ResolvedLLM:
    return ResolvedLLM(
        "Groq", "gsk_test", "qwen/qwen3.8-27b", "https://api.groq.com/openai/v1"
    )


def _error_response(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


def test_llm_retries_json_validate_failed_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _error_response(
                400,
                "json_validate_failed",
                "Failed to generate JSON. Please adjust your prompt.",
            )
        return _choices_response('{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _groq_llm()) == []
    assert attempts == 2
    assert delays == [0.5]


def test_llm_does_not_retry_other_400_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _error_response(400, "invalid_api_key", "Invalid API key.")

    _install_transport(monkeypatch, handle)

    with pytest.raises(LLMConfigurationError, match="HTTP 400"):
        propose_clips([], ProjectConfig(), 60, _groq_llm())

    assert attempts == 1


def test_llm_falls_back_without_response_format_after_json_mode_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "response_format" in payload:
            return _error_response(
                400, "json_validate_failed", "Failed to generate JSON."
            )
        return _choices_response('<think>pensando en los clips</think>\n{"clips": []}')

    _install_transport(monkeypatch, handle)

    assert propose_clips([], ProjectConfig(), 60, _groq_llm()) == []
    # 3 intentos en modo JSON + 1 sin response_format.
    assert len(requests) == 4
    assert "response_format" not in json.loads(requests[3].content)


def test_llm_raises_response_error_when_json_mode_and_fallback_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _error_response(
            400, "json_validate_failed", "Failed to generate JSON."
        )

    _install_transport(monkeypatch, handle)

    # LLMJsonModeError es subclase de LLMResponseError: el pipeline la
    # captura por ventana en vez de abortar el proyecto entero.
    with pytest.raises(LLMResponseError, match="json_validate_failed"):
        propose_clips([], ProjectConfig(), 60, _groq_llm())

    # 3 intentos en modo JSON + 3 del fallback sin response_format.
    assert attempts == 6


def test_llm_json_mode_error_is_a_response_error() -> None:
    assert issubclass(llm_module.LLMJsonModeError, LLMResponseError)


def test_extract_json_ignores_think_blocks() -> None:
    content = (
        "<think>Debo elegir los mejores momentos del video.</think>\n"
        '{"clips": [{"inicio": 1, "fin": 5, "titulo": "t"}]}'
    )

    assert llm_module._extract_json(content) == [
        {"inicio": 1, "fin": 5, "titulo": "t"}
    ]


def test_extract_json_ignores_thinking_blocks_with_brackets() -> None:
    # Sin quitar el bloque, el primer "[" estaría dentro del razonamiento
    # y el parseo fallaría.
    content = (
        "<thinking>los segmentos [1, 2] son buenos</thinking>\n"
        '{"clips": []}'
    )

    assert llm_module._extract_json(content) == []


# ---------------------------------------------------------------------------
# Ítem 14: _write_ass no muta la transcripción guardada
# ---------------------------------------------------------------------------


def test_write_ass_does_not_mutate_saved_transcript(tmp_path: Path) -> None:
    proposal = ClipProposal(
        inicio=0,
        fin=4,
        titulo="Clip con censura",
        puntuacion=8,
        razon_viral="Prueba de censura.",
    )
    config = ProjectConfig()
    config.options.censor = True
    words = [
        {
            "start": 0.1,
            "end": 0.8,
            "preview_start": 0.1,
            "preview_end": 0.8,
            "text": "esto",
        },
        {
            "start": 0.9,
            "end": 1.6,
            "preview_start": 0.9,
            "preview_end": 1.6,
            "text": "es",
        },
        {
            "start": 1.7,
            "end": 2.4,
            "preview_start": 1.7,
            "preview_end": 2.4,
            "text": "mierda",
        },
    ]
    subtitle_path = tmp_path / "clip.ass"

    censor_ranges = _write_ass(
        subtitle_path, words, proposal, config, 4, [(0, 4)]
    )

    # La transcripción guardada conserva las palabras originales.
    assert [word["text"] for word in words] == ["esto", "es", "mierda"]
    # Solo el .ass renderizado sale censurado.
    subtitle = subtitle_path.read_text(encoding="utf-8")
    assert "****" in subtitle
    assert "mierda" not in subtitle
    assert censor_ranges == [(1.7, 2.4)]


# ---------------------------------------------------------------------------
# Ítem 11: guarda anti-bucle en la subida TikTok
# ---------------------------------------------------------------------------


def _install_tiktok_transport(
    monkeypatch: pytest.MonkeyPatch, requests: list[httpx.Request]
) -> None:
    client_type = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/creator_info/query/"):
            return httpx.Response(
                200,
                json={"data": {"privacy_level_options": ["SELF_ONLY"]}},
            )
        if path.endswith("/video/init/"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "upload_url": "https://open-upload.tiktokapis.com/upload",
                        "publish_id": "publish-1",
                    }
                },
            )
        return httpx.Response(200, json={"data": {}})

    monkeypatch.setattr(
        integrations.httpx,
        "Client",
        lambda *args, **kwargs: client_type(
            *args,
            transport=httpx.MockTransport(handle),
            **kwargs,
        ),
    )


class _FakeUploadFile:
    """Archivo cuyo read() simula un EOF prematuro o lecturas cortas."""

    def __init__(self, read_results: list[bytes]) -> None:
        self._read_results = list(read_results)

    def __enter__(self) -> "_FakeUploadFile":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _size: int) -> bytes:
        if self._read_results:
            return self._read_results.pop(0)
        return b""


def _patch_file_open(
    monkeypatch: pytest.MonkeyPatch, target: Path, fake: _FakeUploadFile
) -> None:
    original_open = Path.open

    def fake_open(self: Path, *args: Any, **kwargs: Any):
        if self == target:
            return fake
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)


def test_tiktok_upload_aborts_on_premature_empty_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests: list[httpx.Request] = []
    _install_tiktok_transport(monkeypatch, requests)
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"x" * 100)
    _patch_file_open(
        monkeypatch, clip_file, _FakeUploadFile([b""])
    )

    with pytest.raises(
        integrations.IntegrationError,
        match="terminó antes de enviar todos los bytes",
    ):
        integrations._publish_tiktok(
            "token", {"title": "Clip"}, clip_file, "private"
        )

    # No se envió ningún fragmento a la URL de carga.
    assert not [
        request for request in requests if request.method == "PUT"
    ]


def test_tiktok_upload_aborts_when_iterations_exceed_expected_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests: list[httpx.Request] = []
    _install_tiktok_transport(monkeypatch, requests)
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"x" * 100)
    # Lecturas de 1 byte: con 100 bytes el total esperado es 1 fragmento
    # (+2 de margen); la guarda debe abortar en vez de iterar 100 veces.
    _patch_file_open(
        monkeypatch, clip_file, _FakeUploadFile([b"x"] * 100)
    )

    with pytest.raises(
        integrations.IntegrationError,
        match="número máximo de fragmentos",
    ):
        integrations._publish_tiktok(
            "token", {"title": "Clip"}, clip_file, "private"
        )

    assert len([r for r in requests if r.method == "PUT"]) <= 3


# ---------------------------------------------------------------------------
# Ítem 20: re-verificación SHA-256 + limpieza de instaladores viejos
# ---------------------------------------------------------------------------


def _downloaded_manager(
    tmp_path: Path, installer_bytes: bytes, expected_sha256: str
) -> UpdateManager:
    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager._latest = {
        "version": "1.0.4",
        "download_url": "https://github.com/elpirinpi/KellyClipAi/releases/download/v1.0.4/KellyClipAi-Setup-1.0.4.exe",
        "sha256": expected_sha256,
        "size": len(installer_bytes),
        "asset_name": "KellyClipAi-Setup-1.0.4.exe",
    }
    manager._status["state"] = "downloaded"
    installer = tmp_path / "updates" / "KellyClipAi-Setup-1.0.4.exe"
    installer.parent.mkdir(parents=True, exist_ok=True)
    installer.write_bytes(installer_bytes)
    return manager


def test_open_installer_rejects_a_tampered_installer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        updater_module.os, "startfile", opened.append, raising=False
    )
    manager = _downloaded_manager(
        tmp_path,
        b"contenido modificado tras la descarga",
        hashlib.sha256(b"contenido original verificado").hexdigest(),
    )

    with pytest.raises(UpdateError, match="SHA-256"):
        manager.open_installer()

    assert opened == []
    assert manager.status()["state"] == "downloaded"


def test_open_installer_runs_when_hash_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        updater_module.os, "startfile", opened.append, raising=False
    )
    payload = b"instalador verificado"
    manager = _downloaded_manager(
        tmp_path, payload, hashlib.sha256(payload).hexdigest()
    )

    manager.open_installer()

    assert opened == [str(tmp_path / "updates" / "KellyClipAi-Setup-1.0.4.exe")]
    assert manager.status()["state"] == "installer_opened"


def test_cleanup_removes_old_installers_and_keeps_current(
    tmp_path: Path,
) -> None:
    manager = UpdateManager("1.0.4", tmp_path, tmp_path / "KellyClipAi.exe")
    update_dir = tmp_path / "updates"
    # El constructor ya crea el directorio (historial de versiones,
    # Fase 3, C.6.37): exist_ok para no depender de ese detalle.
    update_dir.mkdir(parents=True, exist_ok=True)
    current = update_dir / "KellyClipAi-Setup-1.0.4.exe"
    current.write_bytes(b"actual")
    old_exe = update_dir / "KellyClipAi-Setup-1.0.3.exe"
    old_exe.write_bytes(b"viejo")
    old_part = update_dir / "KellyClipAi-Setup-1.0.2.exe.part"
    old_part.write_bytes(b"parcial")
    unrelated = update_dir / "notas.txt"
    unrelated.write_text("no tocar")

    manager._cleanup_old_installers(keep=current)

    assert current.is_file()
    assert unrelated.is_file()
    assert not old_exe.exists()
    assert not old_part.exists()


def test_download_cleans_up_previous_installers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"nuevo instalador"
    release = {
        "tag_name": "v1.0.4",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "KellyClipAi-Setup-1.0.4.exe",
                "browser_download_url": (
                    "https://github.com/elpirinpi/KellyClipAi/releases/download/"
                    "v1.0.4/KellyClipAi-Setup-1.0.4.exe"
                ),
                "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                "size": len(payload),
            }
        ],
    }
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
    update_dir = tmp_path / "updates"
    update_dir.mkdir(parents=True)
    old_installer = update_dir / "KellyClipAi-Setup-1.0.3.exe"
    old_installer.write_bytes(b"viejo")
    ancient_installer = update_dir / "KellyClipAi-Setup-1.0.2.exe"
    ancient_installer.write_bytes(b"muy viejo")

    manager = UpdateManager("1.0.3", tmp_path, tmp_path / "KellyClipAi.exe")
    manager.enabled = True
    manager.start_check()
    assert manager._thread is not None
    manager._thread.join(timeout=2)
    manager.start_download()
    assert manager._thread is not None
    manager._thread.join(timeout=2)

    assert manager.status()["state"] == "downloaded"
    assert (update_dir / "KellyClipAi-Setup-1.0.4.exe").read_bytes() == payload
    # Fase 3 (C.6.37): el instalador de la versión instalada se
    # conserva como candidato de "Restaurar versión anterior"; solo
    # los instaladores más antiguos se borran.
    assert old_installer.exists()
    assert not ancient_installer.exists()
