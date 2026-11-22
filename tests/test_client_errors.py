"""Reportes de errores de la interfaz (v1.4.3).

POST /api/client-errors recibe los errores no controlados que captura
el ErrorBoundary del frontend y los registra con el logger del
backend para que queden en desktop.log. Endpoint público (el fallo
puede ocurrir antes de iniciar sesión) y tolerante: cualquier JSON
válido devuelve 204.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app as create_application
from backend.settings import AppSettings


def _settings(data_dir: Path) -> AppSettings:
    return AppSettings(
        data_dir=data_dir,
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="",
        whisper_model="small",
    )


def _client(settings: AppSettings, *, auth_required: bool = False) -> TestClient:
    return TestClient(create_application(settings, auth_required=auth_required))


def test_client_error_report_is_accepted_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path / "data")
    payload = {
        "message": "Cannot read properties of null (reading 'toFixed')",
        "stack": (
            "TypeError: Cannot read properties of null\n"
            "    at ResultsPage (ResultsPage.tsx:805:1)"
        ),
        "component_stack": "\n    at ResultsPage\n    at App",
        "url": "http://127.0.0.1:8000/results",
        "user_agent": "TestAgent/1.0",
    }
    with _client(settings) as client:
        with caplog.at_level(logging.ERROR, logger="backend.main"):
            response = client.post("/api/client-errors", json=payload)
    assert response.status_code == 204
    assert response.content == b""
    assert "Error de la interfaz" in caplog.text
    assert "Cannot read properties of null" in caplog.text
    assert "ResultsPage.tsx:805" in caplog.text
    assert "http://127.0.0.1:8000/results" in caplog.text


def test_client_error_report_accepts_empty_and_partial_payloads(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    with _client(settings) as client:
        assert client.post("/api/client-errors", json={}).status_code == 204
        assert (
            client.post("/api/client-errors", json={"message": "solo mensaje"}).status_code
            == 204
        )
        # Un JSON válido que no es un objeto también se acepta.
        assert (
            client.post("/api/client-errors", json=["no", "un", "objeto"]).status_code
            == 204
        )
        assert client.post("/api/client-errors", json="texto").status_code == 204
        # Campos con tipos equivocados no tumban el endpoint.
        assert (
            client.post(
                "/api/client-errors", json={"message": 42, "stack": ["a", "b"]}
            ).status_code
            == 204
        )


def test_client_error_report_accepts_camel_case_component_stack(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path / "data")
    with _client(settings) as client:
        with caplog.at_level(logging.ERROR, logger="backend.main"):
            response = client.post(
                "/api/client-errors",
                json={"message": "m", "componentStack": "\n    at Foo"},
            )
    assert response.status_code == 204
    assert "at Foo" in caplog.text


def test_client_error_report_truncates_long_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path / "data")
    long_message = "x" * 5000 + "COLA"
    with _client(settings) as client:
        with caplog.at_level(logging.ERROR, logger="backend.main"):
            response = client.post("/api/client-errors", json={"message": long_message})
    assert response.status_code == 204
    assert "… (truncado)" in caplog.text
    assert "COLA" not in caplog.text


def test_client_error_report_rejects_invalid_json(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with _client(settings) as client:
        response = client.post(
            "/api/client-errors",
            content=b"{no es json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_client_error_report_is_public_without_auth(tmp_path: Path) -> None:
    # El fallo puede ocurrir antes de iniciar sesión: el endpoint no
    # exige token aunque la app tenga la autenticación activada.
    settings = _settings(tmp_path / "data")
    with _client(settings, auth_required=True) as client:
        response = client.post("/api/client-errors", json={"message": "sin sesión"})
    assert response.status_code == 204
