from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import database, moments
from backend.main import create_app as create_application
from backend.schemas import ClipProposal, ProjectConfig, TranscriptSegment
from backend.settings import AppSettings


def create_app(settings: AppSettings):
    return create_application(settings, auth_required=False)


def _settings(data_dir: Path, **overrides: Any) -> AppSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "cors_origins": (),
        "llm_provider": "Groq",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_base_url": "",
        "whisper_model": "small",
    }
    values.update(overrides)
    return AppSettings(**values)


def _proposal(
    *,
    inicio: float = 0.0,
    fin: float = 10.0,
    titulo: str = "Título original",
    descripcion: str = "Descripción original",
) -> ClipProposal:
    return ClipProposal(
        inicio=inicio,
        fin=fin,
        titulo=titulo,
        puntuacion=7.5,
        razon_viral="Momento interesante.",
        descripcion=descripcion,
    )


def _create_project(
    settings: AppSettings,
    project_id: str,
    *,
    proposals: list[ClipProposal] | None = None,
    clips: list[dict[str, Any]] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    status: str | None = None,
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    # El lifespan de la app inicializa la BD; aquí se hace explícito para
    # poder crear el proyecto antes de abrir el TestClient.
    database.initialize(settings.database_path)
    database.create_project(
        settings.database_path,
        project_id,
        f"Proyecto {project_id}",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    values: dict[str, Any] = {}
    if proposals is not None:
        values["metadata"] = {
            "duration": 60.0,
            "clip_proposals": [
                {"id": f"clip-{index + 1}", "proposal": proposal.model_dump(mode="json")}
                for index, proposal in enumerate(proposals)
            ],
        }
    if clips is not None:
        values["clips"] = clips
    if transcript is not None:
        values["transcript"] = transcript
    if status is not None:
        values["status"] = status
    if values:
        database.update_project(settings.database_path, project_id, **values)


def _clip(
    clip_id: str, title: str, *, status: str = "ready", transcript: list[dict] | None = None
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "title": title,
        "start": 0.0,
        "end": 10.0,
        "duration": 10.0,
        "score": 7.5,
        "status": status,
        "transcript": transcript if transcript is not None else [],
    }


def _transcript() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 5.0, "text": "Hola a todos, bienvenidos al video.", "words": []},
        {"start": 5.0, "end": 10.0, "text": "Hoy hablamos de un tema increíble.", "words": []},
        {"start": 10.0, "end": 20.0, "text": "Y al final, la sorpresa del día.", "words": []},
    ]


# ---------------------------------------------------------------------------
# Fase D (ítem 6): POST /api/projects/{project_id}/improve-titles
# ---------------------------------------------------------------------------


def test_improve_titles_success_updates_proposals_and_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    calls: list[tuple[list[ClipProposal], list[TranscriptSegment]]] = []

    def fake_polish(proposals, transcript, llm, **_kwargs):
        calls.append((list(proposals), list(transcript)))
        return [
            proposals[0].model_copy(
                update={
                    "titulo": "Título mejorado uno",
                    "descripcion": "Nueva descripción uno",
                    "palabras_clave": ["tema"],
                }
            ),
            proposals[1].model_copy(update={"titulo": "Título mejorado dos"}),
        ]

    monkeypatch.setattr(moments, "polish_titles", fake_polish)
    _create_project(
        settings,
        "project-1",
        proposals=[
            _proposal(titulo="Título original uno", descripcion="Descripción uno"),
            _proposal(inicio=10.0, fin=20.0, titulo="Título original dos"),
        ],
        clips=[
            _clip(
                "clip-1",
                "Título original uno",
                transcript=[{"start": 0.0, "end": 1.0, "text": "hola"}],
            ),
            _clip("clip-2", "Título original dos"),
            _clip("clip-sin-propuesta", "Sin propuesta"),
        ],
        transcript=_transcript(),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 200
    body = response.json()
    assert body["improved"] == 2
    assert "Se mejoraron 2 título(s)." in body["message"]
    titles = {clip["id"]: clip["title"] for clip in body["clips"]}
    assert titles["clip-1"] == "Título mejorado uno"
    assert titles["clip-2"] == "Título mejorado dos"
    assert titles["clip-sin-propuesta"] == "Sin propuesta"
    # UNA sola llamada al pulido, con la transcripción real guardada.
    assert len(calls) == 1
    assert [segment.text for segment in calls[0][1]] == [
        segment["text"] for segment in _transcript()
    ]
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    records = stored["metadata"]["clip_proposals"]
    assert records[0]["id"] == "clip-1"
    assert records[0]["proposal"]["titulo"] == "Título mejorado uno"
    assert records[0]["proposal"]["descripcion"] == "Nueva descripción uno"
    assert records[0]["proposal"]["palabras_clave"] == ["tema"]
    assert records[1]["proposal"]["titulo"] == "Título mejorado dos"
    stored_clips = {clip["id"]: clip for clip in stored["clips"]}
    assert stored_clips["clip-1"]["title"] == "Título mejorado uno"
    assert stored_clips["clip-1"]["description"] == "Nueva descripción uno"
    assert stored_clips["clip-1"]["keywords"] == ["tema"]
    # Los demás campos del clip se conservan intactos.
    assert stored_clips["clip-1"]["status"] == "ready"
    assert stored_clips["clip-1"]["transcript"] == [{"start": 0.0, "end": 1.0, "text": "hola"}]
    assert stored_clips["clip-sin-propuesta"]["title"] == "Sin propuesta"


def test_improve_titles_without_llm_returns_409_and_keeps_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sin clave en .env ni credenciales guardadas: no hay proveedor.
    settings = _settings(tmp_path / "data", llm_api_key="")
    calls: list[None] = []

    def fake_polish(*_args, **_kwargs):
        calls.append(None)
        raise AssertionError("No se debe llamar al LLM sin proveedor configurado.")

    monkeypatch.setattr(moments, "polish_titles", fake_polish)
    _create_project(
        settings,
        "project-1",
        proposals=[_proposal()],
        clips=[_clip("clip-1", "Título original")],
        transcript=_transcript(),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 409
    assert "Ajustes" in response.json()["detail"]
    assert calls == []
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    assert stored["metadata"]["clip_proposals"][0]["proposal"]["titulo"] == "Título original"
    assert stored["clips"][0]["title"] == "Título original"


def test_improve_titles_polish_none_returns_zero_and_keeps_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    monkeypatch.setattr(moments, "polish_titles", lambda *args, **kwargs: None)
    _create_project(
        settings,
        "project-1",
        proposals=[_proposal()],
        clips=[_clip("clip-1", "Título original")],
        transcript=_transcript(),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 200
    assert response.json()["improved"] == 0
    assert "no hubo cambios" in response.json()["message"]
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    assert stored["metadata"]["clip_proposals"][0]["proposal"]["titulo"] == "Título original"
    assert stored["clips"][0]["title"] == "Título original"


def test_improve_titles_missing_project_returns_404(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/no-existe/improve-titles")
    assert response.status_code == 404


def test_improve_titles_without_proposals_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    calls: list[None] = []
    monkeypatch.setattr(moments, "polish_titles", lambda *args, **kwargs: calls.append(None))
    _create_project(settings, "project-1", clips=[_clip("clip-1", "Clip")])
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 409
    assert "no tiene clips" in response.json()["detail"]
    assert calls == []


def test_improve_titles_while_processing_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    calls: list[None] = []
    monkeypatch.setattr(moments, "polish_titles", lambda *args, **kwargs: calls.append(None))
    _create_project(settings, "project-1", proposals=[_proposal()], status="processing")
    with TestClient(create_app(settings)) as client:
        # El lifespan marca como fallidos los proyectos en "processing"
        # (fail_interrupted_projects): el estado se pone tras arrancar.
        database.update_project(settings.database_path, "project-1", status="processing")
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 409
    assert calls == []


def test_improve_titles_without_saved_transcript_uses_proposal_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sin transcripción persistida: el contexto sale del título y la
    # descripción de cada clip, en un segmento por ventana de clip.
    settings = _settings(tmp_path / "data")
    seen: list[list[TranscriptSegment]] = []

    def fake_polish(proposals, transcript, llm, **_kwargs):
        seen.append(list(transcript))
        return [proposals[0].model_copy(update={"titulo": "Título mejorado"})]

    monkeypatch.setattr(moments, "polish_titles", fake_polish)
    _create_project(
        settings,
        "project-1",
        proposals=[
            _proposal(
                inicio=10.0,
                fin=25.0,
                titulo="Título original",
                descripcion="Sobre la sorpresa final",
            )
        ],
        clips=[_clip("clip-1", "Título original")],
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/projects/project-1/improve-titles")
    assert response.status_code == 200
    assert response.json()["improved"] == 1
    assert len(seen) == 1
    assert len(seen[0]) == 1
    assert seen[0][0].start == 10.0
    assert seen[0][0].end == 25.0
    assert "Título original" in seen[0][0].text
    assert "Sobre la sorpresa final" in seen[0][0].text
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    assert stored["clips"][0]["title"] == "Título mejorado"
