"""Tests del editor visual ShortStudio (v1.3.0, backend).

Cubre: GET /api/projects/{id}/source, conversión de transcripción a
cues (backend/studio.py), GET .../clips/{id}/cues, GET .../cues,
PUT .../clips/{id}/studio y POST .../studio/export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend import database
from backend.auth import AuthenticatedUser, AuthenticationError
from backend.main import create_app as create_application
from backend.schemas import ProjectConfig
from backend.settings import AppSettings
from backend.studio import (
    clip_transcript_words,
    project_transcript_words,
    words_to_cues,
)

USER_A = "10000000-0000-4000-8000-000000000001"
USER_B = "20000000-0000-4000-8000-000000000002"


class TokenVerifier:
    async def verify(self, token: str) -> AuthenticatedUser:
        users = {"token-a": USER_A, "token-b": USER_B}
        if token not in users:
            raise AuthenticationError("Sesión no válida.")
        return AuthenticatedUser(users[token], None)


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


def _word(start: float, end: float, text: str) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text}


def _clip_word(
    start: float, end: float, text: str, *, preview_start: float, preview_end: float
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "preview_start": preview_start,
        "preview_end": preview_end,
        "text": text,
    }


def _clip(
    clip_id: str,
    title: str = "Clip",
    *,
    start: float = 10.0,
    end: float = 20.0,
    transcript: list[dict[str, Any]] | None = None,
    status: str = "ready",
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "title": title,
        "start": start,
        "end": end,
        "duration": end - start,
        "score": 7.5,
        "status": status,
        "filename": f"{clip_id}.mp4",
        "transcript": transcript if transcript is not None else [],
        "width": 720,
        "height": 1280,
    }


def _transcript() -> list[dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hola a todos, bienvenidos al video.",
            "words": [
                _word(0.0, 0.4, "Hola"),
                _word(0.5, 0.8, "a"),
                _word(0.9, 1.4, "todos,"),
                _word(1.5, 2.2, "bienvenidos"),
                _word(2.3, 2.6, "al"),
                _word(2.7, 3.2, "video."),
            ],
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "Hoy hablamos de un tema increíble.",
            "words": [
                _word(5.0, 5.4, "Hoy"),
                _word(5.5, 6.2, "hablamos"),
                _word(6.3, 6.6, "de"),
                _word(6.7, 7.0, "un"),
                _word(7.1, 7.5, "tema"),
                _word(7.6, 8.4, "increíble."),
            ],
        },
    ]


def _create_project(
    settings: AppSettings,
    project_id: str,
    *,
    owner_id: str = "test-user",
    clips: list[dict[str, Any]] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "completed",
    source_bytes: bytes = b"fake-source-video-bytes",
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    if source_bytes is not None:
        (project_dir / "source").write_bytes(source_bytes)
    database.initialize(settings.database_path)
    database.create_project(
        settings.database_path,
        project_id,
        f"Proyecto {project_id}",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        owner_id,
    )
    values: dict[str, Any] = {"status": status}
    if clips is not None:
        values["clips"] = clips
    if transcript is not None:
        values["transcript"] = transcript
    if metadata is not None:
        values["metadata"] = metadata
    database.update_project(settings.database_path, project_id, **values)


def _client(settings: AppSettings, *, auth: bool = False) -> TestClient:
    if auth:
        return TestClient(create_application(settings, auth_verifier=TokenVerifier()))
    return TestClient(create_application(settings, auth_required=False))


# ---------------------------------------------------------------------------
# words_to_cues (puro)
# ---------------------------------------------------------------------------


def test_words_to_cues_groups_short_words_into_one_cue() -> None:
    cues = words_to_cues([_word(0.0, 0.4, "Hola"), _word(0.5, 0.9, "a"), _word(1.0, 1.6, "todos")])
    assert cues == [
        {"id": "cue-1", "start": 0.0, "end": 1.6, "text": "Hola a todos"},
    ]


def test_words_to_cues_splits_on_max_chars_without_splitting_words() -> None:
    # Palabras cada 0.3 s: el corte lo provoca el tope de 42 caracteres,
    # no la duración máxima de 6 s.
    words = [
        _word(0.3 * index, 0.3 * index + 0.25, word)
        for index, word in enumerate(
            "esta es una frase bastante larga que debe partirse en dos cues".split()
        )
    ]
    cues = words_to_cues(words)
    assert len(cues) == 2
    assert all(len(cue["text"]) <= 42 for cue in cues)
    assert cues[0]["text"] == "esta es una frase bastante larga que debe"
    assert cues[1]["text"] == "partirse en dos cues"
    assert [cue["id"] for cue in cues] == ["cue-1", "cue-2"]


def test_words_to_cues_breaks_at_sentence_end_once_cue_is_long_enough() -> None:
    words = [
        _word(0.0, 0.5, "Primera"),
        _word(0.6, 1.0, "frase"),
        _word(1.1, 1.6, "bastante"),
        _word(1.7, 2.1, "larga."),
        _word(2.2, 2.6, "Segunda"),
        _word(2.7, 3.1, "frase."),
    ]
    cues = words_to_cues(words)
    assert [cue["text"] for cue in cues] == [
        "Primera frase bastante larga.",
        "Segunda frase.",
    ]


def test_words_to_cues_does_not_break_short_cue_at_sentence_end() -> None:
    words = [
        _word(0.0, 0.5, "Hola."),
        _word(0.6, 1.0, "Sigo"),
        _word(1.1, 1.6, "hablando"),
        _word(1.7, 2.1, "sin"),
        _word(2.2, 2.6, "parar."),
    ]
    cues = words_to_cues(words)
    assert len(cues) == 1
    assert cues[0]["text"] == "Hola. Sigo hablando sin parar."


def test_words_to_cues_keeps_a_word_longer_than_max_chars_whole() -> None:
    long_word = "a" * 50
    cues = words_to_cues(
        [_word(0.0, 0.5, "Antes"), _word(0.6, 1.2, long_word), _word(1.3, 1.8, "después")]
    )
    assert [cue["text"] for cue in cues] == ["Antes", long_word, "después"]


def test_words_to_cues_splits_on_max_duration() -> None:
    words = [_word(0.0, 0.5, "Uno"), _word(5.0, 5.5, "dos"), _word(6.5, 7.0, "tres")]
    cues = words_to_cues(words)
    assert [cue["text"] for cue in cues] == ["Uno dos", "tres"]


def test_words_to_cues_ignores_words_without_valid_times() -> None:
    cues = words_to_cues(
        [
            _word(0.0, 0.5, "válida"),
            {"text": "sin tiempos"},
            {"start": 1.0, "end": 0.5, "text": "fin antes que inicio"},
            {"start": -1.0, "end": 0.5, "text": "inicio negativo"},
            {"start": "no-numérico", "end": 2.0, "text": "no numérica"},
            {"start": 2.0, "end": 2.5, "text": "   "},
            {"start": 2.0, "end": 2.5, "text": None},
            _word(3.0, 3.5, "también válida"),
        ]
    )
    assert cues == [
        {"id": "cue-1", "start": 0.0, "end": 3.5, "text": "válida también válida"},
    ]


def test_words_to_cues_empty_input_returns_empty_list() -> None:
    assert words_to_cues([]) == []
    assert words_to_cues(None) == []


def test_clip_transcript_words_uses_absolute_times_ignoring_preview() -> None:
    # El editor reproduce el video fuente: los cues van en tiempo
    # absoluto aunque el clip guarde también tiempos locales.
    clip = _clip(
        "clip-1",
        start=10.0,
        end=20.0,
        transcript=[
            _clip_word(10.0, 10.5, "Hola", preview_start=0.0, preview_end=0.5),
            _clip_word(10.6, 11.0, "mundo", preview_start=0.6, preview_end=1.0),
        ],
    )
    assert clip_transcript_words(clip) == [
        {"start": 10.0, "end": 10.5, "text": "Hola"},
        {"start": 10.6, "end": 11.0, "text": "mundo"},
    ]


def test_clip_transcript_words_uses_absolute_times_without_preview() -> None:
    clip = _clip(
        "clip-1",
        start=10.0,
        end=20.0,
        transcript=[_word(12.0, 12.5, "Hola"), _word(12.6, 13.0, "mundo")],
    )
    assert clip_transcript_words(clip) == [
        {"start": 12.0, "end": 12.5, "text": "Hola"},
        {"start": 12.6, "end": 13.0, "text": "mundo"},
    ]


def test_project_transcript_words_filters_to_range() -> None:
    words = project_transcript_words(_transcript(), 5.0, 10.0)
    assert [word["text"] for word in words] == [
        "Hoy",
        "hablamos",
        "de",
        "un",
        "tema",
        "increíble.",
    ]
    # Una palabra que empieza antes del rango pero termina dentro se incluye.
    words = project_transcript_words(_transcript(), 3.0, 6.0)
    assert [word["text"] for word in words] == ["video.", "Hoy", "hablamos"]


# ---------------------------------------------------------------------------
# Autenticación y propiedad (los cinco endpoints nuevos)
# ---------------------------------------------------------------------------


def _studio_requests(project_id: str, clip_id: str) -> list[tuple[str, str, dict]]:
    return [
        ("GET", f"/api/projects/{project_id}/source", {}),
        ("GET", f"/api/projects/{project_id}/clips/{clip_id}/cues", {}),
        ("GET", f"/api/projects/{project_id}/cues?start=0&end=5", {}),
        (
            "PUT",
            f"/api/projects/{project_id}/clips/{clip_id}/studio",
            {"json": {"studio_edit": {"ratio": "9:16"}}},
        ),
        (
            "POST",
            f"/api/projects/{project_id}/studio/export",
            {
                "files": {"file": ("clip.mp4", b"video", "video/mp4")},
                "data": {"start": "0", "end": "5"},
            },
        ),
    ]


def test_studio_endpoints_require_authentication(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        owner_id=USER_A,
        clips=[_clip("clip-1")],
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    with _client(settings, auth=True) as client:
        for method, url, kwargs in _studio_requests("project-1", "clip-1"):
            response = client.request(method, url, **kwargs)
            assert response.status_code == 401, (method, url, response.text)


def test_studio_endpoints_reject_other_owners(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        owner_id=USER_A,
        clips=[_clip("clip-1")],
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    headers = {"Authorization": "Bearer token-b"}
    with _client(settings, auth=True) as client:
        for method, url, kwargs in _studio_requests("project-1", "clip-1"):
            response = client.request(method, url, headers=headers, **kwargs)
            assert response.status_code == 404, (method, url, response.text)


def test_studio_endpoints_allow_the_owner(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        owner_id=USER_A,
        clips=[_clip("clip-1")],
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    headers = {"Authorization": "Bearer token-a"}
    with _client(settings, auth=True) as client:
        assert client.get("/api/projects/project-1/source", headers=headers).status_code == 200
        assert (
            client.get("/api/projects/project-1/clips/clip-1/cues", headers=headers).status_code
            == 200
        )


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/source
# ---------------------------------------------------------------------------


def test_get_source_serves_the_original_video(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", source_bytes=b"0123456789abcdef")
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/source")
    assert response.status_code == 200
    assert response.content == b"0123456789abcdef"
    assert response.headers["content-type"].startswith("video/mp4")


def test_get_source_supports_range_requests(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", source_bytes=b"0123456789abcdef")
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/source", headers={"Range": "bytes=4-9"})
    assert response.status_code == 206
    assert response.content == b"456789"
    assert response.headers["content-range"] == "bytes 4-9/16"


def test_get_source_404_without_project_or_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1")
    (settings.data_dir / "projects" / "project-1" / "source").unlink()
    with _client(settings) as client:
        assert client.get("/api/projects/missing/source").status_code == 404
        assert client.get("/api/projects/project-1/source").status_code == 404


# ---------------------------------------------------------------------------
# GET .../clips/{id}/cues y GET .../cues
# ---------------------------------------------------------------------------


def test_get_clip_cues_uses_absolute_source_times(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        clips=[
            _clip(
                "clip-1",
                start=10.0,
                end=20.0,
                transcript=[
                    _clip_word(10.0, 10.5, "Hola", preview_start=0.0, preview_end=0.5),
                    _clip_word(10.6, 11.0, "a", preview_start=0.6, preview_end=1.0),
                    _clip_word(11.1, 11.6, "todos", preview_start=1.1, preview_end=1.6),
                ],
            )
        ],
    )
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/clips/clip-1/cues")
    assert response.status_code == 200
    assert response.json() == {
        "cues": [{"id": "cue-1", "start": 10.0, "end": 11.6, "text": "Hola a todos"}]
    }


def test_get_clip_cues_empty_when_clip_has_no_timed_words(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        clips=[_clip("clip-1", transcript=[{"text": "sin tiempos"}])],
    )
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/clips/clip-1/cues")
    assert response.status_code == 200
    assert response.json() == {"cues": []}


def test_get_clip_cues_404_for_missing_clip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    with _client(settings) as client:
        assert client.get("/api/projects/project-1/clips/nope/cues").status_code == 404
        assert client.get("/api/projects/missing/clips/clip-1/cues").status_code == 404


def test_get_project_cues_filters_to_range(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/cues?start=5&end=10")
    assert response.status_code == 200
    assert response.json() == {
        "cues": [
            {
                "id": "cue-1",
                "start": 5.0,
                "end": 8.4,
                "text": "Hoy hablamos de un tema increíble.",
            }
        ]
    }


def test_get_project_cues_validates_range(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    with _client(settings) as client:
        assert client.get("/api/projects/project-1/cues?start=5&end=5").status_code == 400
        assert client.get("/api/projects/project-1/cues?start=8&end=5").status_code == 400
        assert client.get("/api/projects/project-1/cues?start=-1&end=5").status_code == 400
        assert client.get("/api/projects/project-1/cues?start=0&end=61").status_code == 400
        assert client.get("/api/projects/project-1/cues?start=0&end=60").status_code == 200


def test_get_project_cues_404_for_missing_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    with _client(settings) as client:
        assert client.get("/api/projects/missing/cues?start=0&end=5").status_code == 404


# ---------------------------------------------------------------------------
# PUT .../clips/{id}/studio
# ---------------------------------------------------------------------------


def test_put_studio_saves_edit_and_survives_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    # Spec con la forma real de StudioEditSpec (v1.3.1): el ratio vive
    # en settings.ratio; el backend la trata como un dict opaco y debe
    # devolverla tal cual, sin validar ni tocar el ratio.
    edit = {
        "version": 1,
        "settings": {"ratio": "1:1", "fitMode": "cover", "zoom": 100},
        "cues": [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hola"}],
    }
    with _client(settings) as client:
        response = client.put(
            "/api/projects/project-1/clips/clip-1/studio",
            json={"studio_edit": edit},
        )
        assert response.status_code == 200
        assert response.json()["studio_edit"] == edit
        assert response.json()["studio_edit"]["settings"]["ratio"] == "1:1"
        assert response.json()["video_url"] == "/api/projects/project-1/clips/clip-1/file"
        detail = client.get("/api/projects/project-1")
    assert detail.status_code == 200
    saved = next(clip for clip in detail.json()["clips"] if clip["id"] == "clip-1")
    assert saved["studio_edit"] == edit
    assert saved["studio_edit"]["settings"]["ratio"] == "1:1"
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    assert stored["clips"][0]["studio_edit"] == edit
    assert stored["clips"][0]["studio_edit"]["settings"]["ratio"] == "1:1"


def test_put_studio_404_for_missing_clip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    with _client(settings) as client:
        response = client.put(
            "/api/projects/project-1/clips/nope/studio",
            json={"studio_edit": {}},
        )
    assert response.status_code == 404


def test_put_studio_rejects_invalid_bodies(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    with _client(settings) as client:
        assert (
            client.put(
                "/api/projects/project-1/clips/clip-1/studio",
                json={"studio_edit": ["no", "un", "objeto"]},
            ).status_code
            == 422
        )
        assert (
            client.put(
                "/api/projects/project-1/clips/clip-1/studio",
                json={"studio_edit": "texto"},
            ).status_code
            == 422
        )
        assert client.put("/api/projects/project-1/clips/clip-1/studio", json={}).status_code == 422


# ---------------------------------------------------------------------------
# POST .../studio/export
# ---------------------------------------------------------------------------


def _export_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "files": {"file": ("clip.mp4", b"nuevo-video", "video/mp4")},
        "data": {"start": "5", "end": "15"},
    }
    payload.update(overrides)
    return payload


def test_export_replaces_existing_clip_file_and_data(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    clips_dir = settings.data_dir / "projects" / "project-1" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "clip-1.mp4").write_bytes(b"video-viejo")
    _create_project(
        settings,
        "project-1",
        clips=[_clip("clip-1", title="Viejo")],
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    edit = {"ratio": "9:16", "trim": [5.0, 15.0]}
    with _client(settings) as client:
        response = client.post(
            "/api/projects/project-1/studio/export",
            files={"file": ("clip.mp4", b"video-nuevo", "video/mp4")},
            data={
                "clip_id": "clip-1",
                "start": "5",
                "end": "15",
                "title": "Nuevo título",
                "studio_edit": json.dumps(edit),
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "clip-1"
    assert body["filename"] == "clip-1.mp4"
    assert body["start"] == 5.0
    assert body["end"] == 15.0
    assert body["duration"] == 10.0
    assert body["title"] == "Nuevo título"
    assert body["status"] == "ready"
    assert body["studio_edit"] == edit
    assert body["video_url"] == "/api/projects/project-1/clips/clip-1/file"
    # El archivo viejo se sustituye: mismo nombre, contenido nuevo y
    # ningún archivo huérfano (ni temporal) en la carpeta de clips.
    assert (clips_dir / "clip-1.mp4").read_bytes() == b"video-nuevo"
    assert [path.name for path in clips_dir.iterdir()] == ["clip-1.mp4"]
    # El transcript se reconstruye desde el proyecto en el rango nuevo.
    assert [word["text"] for word in body["transcript"]] == [
        "Hoy",
        "hablamos",
        "de",
        "un",
        "tema",
        "increíble.",
    ]
    assert body["transcript"][0]["preview_start"] == 0.0
    assert body["transcript"][0]["start"] == 5.0


def test_export_creates_manual_clip_without_clip_id(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        clips=[],
        transcript=_transcript(),
        metadata={"duration": 60.0},
    )
    with _client(settings) as client:
        response = client.post(
            "/api/projects/project-1/studio/export",
            files={"file": ("manual.webm", b"video-manual", "video/webm")},
            data={
                "start": "0",
                "end": "5",
                "title": "Manual uno",
                "studio_edit": json.dumps({"ratio": "9:16"}),
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Manual uno"
    assert body["start"] == 0.0
    assert body["end"] == 5.0
    assert body["duration"] == 5.0
    assert body["status"] == "ready"
    assert body["manual"] is True
    assert body["studio_edit"] == {"ratio": "9:16"}
    assert body["filename"] == f"{body['id']}.webm"
    assert body["video_url"] == f"/api/projects/project-1/clips/{body['id']}/file"
    clips_dir = settings.data_dir / "projects" / "project-1" / "clips"
    assert (clips_dir / body["filename"]).read_bytes() == b"video-manual"
    assert [word["text"] for word in body["transcript"]] == [
        "Hola",
        "a",
        "todos,",
        "bienvenidos",
        "al",
        "video.",
    ]
    stored = database.get_project(settings.database_path, "project-1")
    assert stored is not None
    assert len(stored["clips"]) == 1
    assert stored["clips"][0]["id"] == body["id"]


def test_export_manual_clip_generates_title_when_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[], metadata={"duration": 60.0})
    with _client(settings) as client:
        response = client.post(
            "/api/projects/project-1/studio/export",
            files={"file": ("clip.mp4", b"video", "video/mp4")},
            data={"clip_id": "", "start": "2", "end": "7"},
        )
    assert response.status_code == 201
    assert response.json()["title"] == "Clip manual 2.0s - 7.0s"
    assert response.json()["transcript"] == []
    assert response.json()["studio_edit"] is None


def test_export_rejects_non_video_content(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[])
    with _client(settings) as client:
        response = client.post(
            "/api/projects/project-1/studio/export",
            files={"file": ("notas.txt", b"texto plano", "text/plain")},
            data={"start": "0", "end": "5"},
        )
    assert response.status_code == 415


def test_export_rejects_oversized_upload_and_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    clips_dir = settings.data_dir / "projects" / "project-1" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "clip-1.mp4").write_bytes(b"video-viejo")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
    with _client(settings) as client:
        response = client.post(
            "/api/projects/project-1/studio/export",
            files={"file": ("clip.mp4", b"este video supera el tope", "video/mp4")},
            data={"clip_id": "clip-1", "start": "0", "end": "5"},
        )
    assert response.status_code == 413
    # El archivo viejo queda intacto y no queda ningún temporal.
    assert (clips_dir / "clip-1.mp4").read_bytes() == b"video-viejo"
    assert [path.name for path in clips_dir.iterdir()] == ["clip-1.mp4"]


def test_export_validates_range_and_clip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[_clip("clip-1")])
    with _client(settings) as client:
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                **_export_payload(data={"start": "5", "end": "5"}),
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                **_export_payload(data={"start": "9", "end": "5"}),
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                **_export_payload(data={"clip_id": "nope", "start": "0", "end": "5"}),
            ).status_code
            == 404
        )


def test_export_rejects_invalid_studio_edit_and_empty_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", clips=[])
    with _client(settings) as client:
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                files={"file": ("clip.mp4", b"video", "video/mp4")},
                data={"start": "0", "end": "5", "studio_edit": "{no-json"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                files={"file": ("clip.mp4", b"video", "video/mp4")},
                data={"start": "0", "end": "5", "studio_edit": "[1, 2]"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/projects/project-1/studio/export",
                files={"file": ("clip.mp4", b"", "video/mp4")},
                data={"start": "0", "end": "5"},
            ).status_code
            == 400
        )
