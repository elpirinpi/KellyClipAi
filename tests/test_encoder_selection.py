"""Selección del codificador de video (v1.4.2).

NVENC > QSV > AMF > libx264, con la misma comprobación en dos pasos
para cada candidato (listado en `ffmpeg -encoders` + encode de prueba
corto). ffmpeg está mockeado en todos los tests: ningún test ejecuta
un ffmpeg real ni necesita GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import database
from backend import media as media_module
from backend.main import create_app as create_application
from backend.settings import AppSettings

_NVENC_LINE = " V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n"
_QSV_LINE = " V..... h264_qsv  Intel Quick Sync Video H.264 encoder\n"
_AMF_LINE = " V..... h264_amf  AMD AMF H.264 encoder\n"
_LIBX264_LINE = " V..... libx264  libx264 H.264 encoder\n"

_LIBX264_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
_NVENC_ARGS = [
    "-c:v",
    "h264_nvenc",
    "-preset",
    "p4",
    "-cq",
    "21",
    "-pix_fmt",
    "yuv420p",
]
_QSV_ARGS = [
    "-c:v",
    "h264_qsv",
    "-preset",
    "veryfast",
    "-global_quality",
    "21",
    "-pix_fmt",
    "yuv420p",
]
_AMF_ARGS = [
    "-c:v",
    "h264_amf",
    "-quality",
    "balanced",
    "-rc",
    "cqp",
    "-qp_i",
    "21",
    "-qp_p",
    "21",
    "-pix_fmt",
    "yuv420p",
]


@pytest.fixture(autouse=True)
def _reset_encoder_cache(monkeypatch: pytest.MonkeyPatch):
    # La caché es por proceso y las variables de entorno se leen en
    # cada selección: cada test parte de un estado limpio.
    monkeypatch.delenv("KELLYCLIP_NVENC", raising=False)
    monkeypatch.delenv("KELLYCLIP_GPU", raising=False)
    media_module.reset_encoder_cache()
    yield
    media_module.reset_encoder_cache()


def _fake_run_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    encoders_stdout: str,
    probe_returncodes: dict[str, int] | None = None,
) -> list[list[str]]:
    """Mock de subprocess.run: lista de codificadores + probe por encoder.

    `probe_returncodes` mapea nombre de codificador a código de salida
    del encode de prueba (0 = éxito). Sin entrada, el probe pasa.
    """
    calls: list[list[str]] = []
    returncodes = probe_returncodes or {}

    def fake_run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(args))
        if "-encoders" in args:
            return SimpleNamespace(stdout=encoders_stdout, returncode=0)
        for encoder, returncode in returncodes.items():
            if encoder in args:
                return SimpleNamespace(stdout="", returncode=returncode)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(media_module.subprocess, "run", fake_run)
    return calls


def _settings(data_dir: Path) -> AppSettings:
    return AppSettings(
        data_dir=data_dir,
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="test-model",
        llm_base_url="",
        whisper_model="small",
    )


# ---------------------------------------------------------------------------
# Selección por orden de preferencia
# ---------------------------------------------------------------------------


def test_nvenc_wins_when_all_hardware_encoders_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch,
        encoders_stdout=_NVENC_LINE + _QSV_LINE + _AMF_LINE + _LIBX264_LINE,
    )
    assert media_module._video_encoder_args("ffmpeg") == _NVENC_ARGS
    assert media_module.is_nvenc_available("ffmpeg") is True
    assert media_module.get_encoder_info("ffmpeg") == {
        "encoder": "h264_nvenc",
        "kind": "nvenc",
        "label": "NVENC (GPU NVIDIA)",
        "reason": "GPU NVIDIA, probe OK",
    }
    # Un -encoders y un probe por candidato listado (la sonda es
    # completa: la caché no depende de las variables de entorno).
    assert len(calls) == 4
    assert "-encoders" in calls[0]
    assert "h264_nvenc" in calls[1]
    assert "h264_qsv" in calls[2]
    assert "h264_amf" in calls[3]


def test_qsv_wins_when_nvenc_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_QSV_LINE + _AMF_LINE + _LIBX264_LINE)
    assert media_module._video_encoder_args("ffmpeg") == _QSV_ARGS
    assert media_module.is_nvenc_available("ffmpeg") is False
    assert media_module.get_encoder_info("ffmpeg") == {
        "encoder": "h264_qsv",
        "kind": "qsv",
        "label": "Quick Sync (Intel)",
        "reason": "Quick Sync (Intel), probe OK",
    }


def test_amf_wins_when_nvenc_and_qsv_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_AMF_LINE + _LIBX264_LINE)
    assert media_module._video_encoder_args("ffmpeg") == _AMF_ARGS
    assert media_module.get_encoder_info("ffmpeg") == {
        "encoder": "h264_amf",
        "kind": "amf",
        "label": "AMF (AMD)",
        "reason": "GPU AMD, probe OK",
    }


def test_libx264_when_no_hardware_encoder_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(monkeypatch, encoders_stdout=_LIBX264_LINE)
    # Fallback CPU: EXACTAMENTE los mismos argumentos de siempre.
    assert media_module._video_encoder_args("ffmpeg") == _LIBX264_ARGS
    info = media_module.get_encoder_info("ffmpeg")
    assert info["encoder"] == "libx264"
    assert info["kind"] == "cpu"
    assert info["label"] == "CPU (libx264)"
    assert info["reason"] == (
        "CPU; sin codificador HW disponible: nvenc no listado, qsv no listado, amf no listado"
    )
    # Sin candidatos listados, ni siquiera se intenta un encode de prueba.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Fallo del probe
# ---------------------------------------------------------------------------


def test_probe_failure_falls_through_to_next_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # El caso clave: ffmpeg con NVENC compilado en un equipo sin GPU
    # NVIDIA lista el codificador pero el encode falla → se pasa a QSV.
    _fake_run_factory(
        monkeypatch,
        encoders_stdout=_NVENC_LINE + _QSV_LINE + _LIBX264_LINE,
        probe_returncodes={"h264_nvenc": 1},
    )
    assert media_module._video_encoder_args("ffmpeg") == _QSV_ARGS
    assert media_module.is_nvenc_available("ffmpeg") is False


def test_probe_failure_of_all_falls_back_to_libx264(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_run_factory(
        monkeypatch,
        encoders_stdout=_NVENC_LINE + _QSV_LINE + _AMF_LINE + _LIBX264_LINE,
        probe_returncodes={"h264_nvenc": 1, "h264_qsv": 1, "h264_amf": 1},
    )
    assert media_module._video_encoder_args("ffmpeg") == _LIBX264_ARGS
    assert media_module.get_encoder_info("ffmpeg")["reason"] == (
        "CPU; sin codificador HW disponible: "
        "nvenc falló el probe, qsv falló el probe, amf falló el probe"
    )


# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------


def test_nvenc_env_var_skips_only_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_NVENC_LINE + _QSV_LINE + _LIBX264_LINE)
    monkeypatch.setenv("KELLYCLIP_NVENC", "0")
    assert media_module.is_nvenc_available("ffmpeg") is False
    assert media_module._video_encoder_args("ffmpeg") == _QSV_ARGS


def test_nvenc_env_var_with_no_other_encoder_falls_back_to_libx264(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_NVENC_LINE + _LIBX264_LINE)
    monkeypatch.setenv("KELLYCLIP_NVENC", "0")
    assert media_module._video_encoder_args("ffmpeg") == _LIBX264_ARGS
    assert media_module.get_encoder_info("ffmpeg")["reason"] == (
        "CPU; sin codificador HW disponible: "
        "nvenc desactivado por KELLYCLIP_NVENC=0, qsv no listado, amf no listado"
    )


def test_gpu_env_var_forces_libx264_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch,
        encoders_stdout=_NVENC_LINE + _QSV_LINE + _AMF_LINE + _LIBX264_LINE,
    )
    monkeypatch.setenv("KELLYCLIP_GPU", "0")
    assert media_module._video_encoder_args("ffmpeg") == _LIBX264_ARGS
    assert media_module.is_nvenc_available("ffmpeg") is False
    assert media_module.get_encoder_info("ffmpeg")["reason"] == (
        "CPU; KELLYCLIP_GPU=0 (codificadores por hardware desactivados)"
    )
    # KELLYCLIP_GPU=0 ataja antes de la sonda: ni un ffmpeg se ejecuta.
    assert calls == []


def test_gpu_env_var_applies_to_cached_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(monkeypatch, encoders_stdout=_NVENC_LINE + _LIBX264_LINE)
    assert media_module._video_encoder_args("ffmpeg") == _NVENC_ARGS
    probe_calls = len(calls)
    # La variable tiene efecto inmediato aunque la sonda esté cacheada:
    # la selección se reevalúa en cada llamada.
    monkeypatch.setenv("KELLYCLIP_GPU", "0")
    assert media_module._video_encoder_args("ffmpeg") == _LIBX264_ARGS
    assert len(calls) == probe_calls


# ---------------------------------------------------------------------------
# Caché y log
# ---------------------------------------------------------------------------


def test_detection_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run_factory(monkeypatch, encoders_stdout=_NVENC_LINE + _QSV_LINE + _LIBX264_LINE)
    assert media_module._video_encoder_args("ffmpeg") == _NVENC_ARGS
    assert media_module._video_encoder_args("ffmpeg") == _NVENC_ARGS
    assert media_module.get_encoder_info("ffmpeg")["encoder"] == "h264_nvenc"
    assert media_module.is_nvenc_available("ffmpeg") is True
    # La sonda corrió una sola vez (1 -encoders + 2 probes).
    assert len(calls) == 3


def test_render_logs_selected_encoder(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_NVENC_LINE + _LIBX264_LINE)
    with caplog.at_level(logging.INFO, logger="backend.media"):
        media_module._video_encoder_args("ffmpeg")
    assert "Render encoder: h264_nvenc (GPU NVIDIA, probe OK)" in caplog.text


def test_render_logs_cpu_fallback_with_diagnostic_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_run_factory(
        monkeypatch,
        encoders_stdout=_NVENC_LINE + _QSV_LINE + _LIBX264_LINE,
        probe_returncodes={"h264_nvenc": 1, "h264_qsv": 1},
    )
    with caplog.at_level(logging.INFO, logger="backend.media"):
        media_module._video_encoder_args("ffmpeg")
    assert (
        "Render encoder: libx264 (CPU; sin codificador HW disponible: "
        "nvenc falló el probe, qsv falló el probe, amf no listado)" in caplog.text
    )


def test_get_encoder_info_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_module.shutil, "which", lambda _name: None)
    assert media_module.get_encoder_info() == {
        "encoder": "libx264",
        "kind": "cpu",
        "label": "CPU (libx264)",
        "reason": "CPU; ffmpeg no encontrado en el PATH",
    }


# ---------------------------------------------------------------------------
# GET /api/encoder
# ---------------------------------------------------------------------------


def test_encoder_endpoint_returns_active_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_run_factory(monkeypatch, encoders_stdout=_QSV_LINE + _AMF_LINE + _LIBX264_LINE)
    monkeypatch.setattr(media_module.shutil, "which", lambda _name: "ffmpeg")
    settings = _settings(tmp_path)
    database.initialize(settings.database_path)
    client = TestClient(create_application(settings, auth_required=False))
    response = client.get("/api/encoder")
    assert response.status_code == 200
    assert response.json() == {
        "encoder": "h264_qsv",
        "kind": "qsv",
        "label": "Quick Sync (Intel)",
        "reason": "Quick Sync (Intel), probe OK",
    }
