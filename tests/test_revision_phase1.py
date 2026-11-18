from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import database
from backend import llm as llm_module
from backend import pipeline
from backend.main import create_app
from backend.schemas import ProjectConfig
from backend.llm import ResolvedLLM
from backend.settings import AppSettings


def _settings(data_dir: Path) -> AppSettings:
    return AppSettings(
        data_dir=data_dir,
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )


def _create_project(
    database_path: Path,
    project_id: str,
    status: str,
) -> None:
    config = {"ratio": "9:16"}
    database.create_project(
        database_path,
        project_id,
        project_id,
        Path(f"{project_id}.mp4"),
        config,
        "test-user",
    )
    if status == "queued":
        assert database.queue_project(database_path, project_id, config)
    elif status != "ready":
        database.update_project(
            database_path,
            project_id,
            status=status,
            step="finding",
            progress=35,
        )


def test_startup_fails_projects_interrupted_by_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    for project_id, status in (
        ("queued-project", "queued"),
        ("processing-project", "processing"),
        ("ready-project", "ready"),
    ):
        _create_project(settings.database_path, project_id, status)

    with TestClient(create_app(settings)):
        pass

    for project_id in ("queued-project", "processing-project"):
        project = database.get_project(settings.database_path, project_id)
        assert project is not None
        assert project["status"] == "failed"
        assert "reinici" in project["error"].casefold()
        assert database.queue_project(
            settings.database_path, project_id, project["config"]
        )

    ready_project = database.get_project(settings.database_path, "ready-project")
    assert ready_project is not None
    assert ready_project["status"] == "ready"


def test_llm_keeps_individual_candidates_for_pipeline_validation(
    monkeypatch,
) -> None:
    valid = {
        "inicio": 10,
        "fin": 30,
        "titulo": "Momento válido",
        "puntuacion": 8,
        "razon_viral": "Gancho claro.",
        "palabras_clave": [],
        "emojis": [],
    }
    invalid = {**valid, "dato_inesperado": "se debe descartar en el pipeline"}
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"clips": [valid, invalid]})}}
                ]
            },
        )
    )
    client = httpx.Client(transport=transport)
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **_kwargs: client)

    proposals = llm_module.propose_clips(
        [],
        ProjectConfig(),
        60,
        ResolvedLLM("OpenAI", "test-token", "gpt-4o-mini", "https://llm.example"),
    )

    assert proposals == [valid, invalid]


def test_invalid_clip_proposals_are_logged_and_retried_once(monkeypatch, caplog) -> None:
    valid = {
        "inicio": 10,
        "fin": 30,
        "titulo": "Primer clip",
        "puntuacion": 8,
        "razon_viral": "Gancho claro.",
        "palabras_clave": [],
        "emojis": [],
    }
    overlapping = {**valid, "inicio": 20, "fin": 40, "titulo": "Solapado"}
    out_of_range = {**valid, "inicio": 75, "fin": 90, "titulo": "Fuera de rango"}
    retried = {**valid, "inicio": 32, "fin": 52, "titulo": "Clip corregido"}
    responses = [[valid, overlapping, out_of_range], [retried]]
    feedback: list[list[str] | None] = []

    def propose_clips(*_args, validation_feedback=None):
        feedback.append(validation_feedback)
        return responses.pop(0)

    monkeypatch.setattr(pipeline, "propose_clips", propose_clips)
    proposals = pipeline._propose_valid_clips(
        [],
        ProjectConfig(),
        60,
        ResolvedLLM("OpenAI", "test-token", "gpt-4o-mini", "https://llm.example"),
    )

    assert [proposal.titulo for proposal in proposals] == [
        "Primer clip",
        "Clip corregido",
    ]
    assert len(feedback) == 2
    assert feedback[0] is None
    assert feedback[1]
    assert any("solapa" in error for error in feedback[1])
    assert any("duración" in error for error in feedback[1])
    assert "Solapado" in caplog.text
    assert "Fuera de rango" in caplog.text


def test_project_list_omits_transcript_but_project_detail_retains_it(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_project(settings.database_path, "project-with-transcript", "ready")
    transcript = [
        {"start": 0.0, "end": 1.0, "text": "Transcripción privada", "words": []}
    ]
    database.update_project(
        settings.database_path,
        "project-with-transcript",
        transcript=transcript,
    )

    with TestClient(create_app(settings, auth_required=False)) as client:
        listed = client.get("/api/projects")
        detailed = client.get("/api/projects/project-with-transcript")

    assert listed.status_code == 200
    assert "transcript" not in listed.json()[0]
    assert detailed.status_code == 200
    assert detailed.json()["transcript"] == transcript


def test_project_middleware_and_sse_use_lightweight_status_queries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_project(settings.database_path, "completed-project", "ready")
    database.update_project(
        settings.database_path, "completed-project", status="completed"
    )
    database.publish_event(
        settings.database_path,
        "completed-project",
        "progress",
        {"status": "completed", "message": "Listo"},
    )
    monkeypatch.setattr(
        database,
        "get_project",
        lambda *_args, **_kwargs: pytest.fail(
            "The middleware and SSE polling must not load the full project."
        ),
    )

    with TestClient(create_app(settings, auth_required=False)) as client:
        response = client.get("/api/projects/completed-project/events")

    assert response.status_code == 200
    assert "Listo" in response.text
