from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import database
from backend import integrations
from backend import llm as llm_module
from backend import media as media_module
from backend import pipeline
from backend.main import create_app
from backend.llm import _prompt
from backend.media import MediaProcessingError, render_clip
from backend.schemas import ClipProposal, ProjectConfig
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


def test_clip_proposal_includes_description_and_platform_hashtags() -> None:
    proposal = ClipProposal.model_validate(
        {
            "inicio": 1,
            "fin": 20,
            "titulo": "Un consejo",
            "puntuacion": 8,
            "razon_viral": "Explica una idea útil.",
            "descripcion": "Un consejo práctico para hoy.",
            "hashtags": {
                "youtube": ["#Consejos"],
                "tiktok": ["#Aprender"],
                "instagram": ["#Ideas"],
            },
        }
    )
    system, _user = _prompt([], ProjectConfig(), 30)

    assert proposal.descripcion == "Un consejo práctico para hoy."
    assert proposal.hashtags["youtube"] == ["#Consejos"]
    assert proposal.hashtags["tiktok"] == ["#Aprender"]
    assert proposal.hashtags["instagram"] == ["#Ideas"]
    assert "descripcion" in system
    assert "hashtags" in system


def test_youtube_publishes_description_and_hashtags_not_viral_reason(
    monkeypatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"mp4-data")
    upload_requests: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            upload_requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={
                    "Location": "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=test"
                },
            )
        return httpx.Response(200, json={"id": "youtube-video"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(integrations.httpx, "Client", lambda **_kwargs: client)

    result = integrations._publish_youtube(
        "access-token",
        {
            "title": "Un consejo",
            "description": "Un consejo práctico para hoy.",
            "hashtags": {"youtube": ["Consejos", "#YouTubeShorts"]},
            "viral_reason": "No publicar esta explicación interna.",
            "keywords": ["consejo"],
        },
        file_path,
        "public",
    )

    payload = upload_requests[0]
    assert result["external_id"] == "youtube-video"
    assert payload["snippet"]["description"] == (
        "Un consejo práctico para hoy.\n\n#Consejos #YouTubeShorts #Shorts"
    )
    assert "No publicar" not in payload["snippet"]["description"]
    assert payload["status"]["selfDeclaredMadeForKids"] is False


def test_clip_keeps_generated_metadata_and_publishes_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = "metadata-project"
    project_dir = settings.data_dir / "projects" / project_id
    input_path = project_dir / "source"
    database.create_project(
        settings.database_path,
        project_id,
        "Metadata",
        input_path,
        {"ratio": "9:16"},
        "test-user",
    )
    monkeypatch.setattr(
        pipeline.database, "resolve_subtitle_template", lambda *_args: "Default"
    )
    progress_events: list[tuple[str, str, int | None, str]] = []

    def render_clip_with_progress(
        _input_path,
        _output_path,
        _subtitle_path,
        _proposal,
        _config,
        _transcript,
        _ffmpeg_path,
        on_progress,
        **kwargs,
    ):
        kwargs["on_analysis_progress"](50)
        on_progress(50)
        return 10.0, []

    monkeypatch.setattr(pipeline, "render_clip", render_clip_with_progress)
    monkeypatch.setattr(
        pipeline,
        "_publish",
        lambda _settings, _project_id, status, step, progress, message: progress_events.append(
            (status, step, progress, message)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda *_args, **_kwargs: {"width": 720, "height": 1280},
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffprobe")
    proposal = ClipProposal.model_validate(
        {
            "inicio": 1,
            "fin": 11,
            "titulo": "Un consejo",
            "puntuacion": 8,
            "razon_viral": "Explica una idea útil.",
            "descripcion": "Un consejo práctico.",
            "hashtags": {"youtube": ["Consejos"]},
        }
    )

    clips = pipeline._render_projects(
        settings,
        project_id,
        input_path,
        ProjectConfig(),
        [proposal],
        [],
        "ffmpeg",
        "test-user",
    )

    assert clips[0]["description"] == "Un consejo práctico."
    assert clips[0]["hashtags"] == {"youtube": ["Consejos"]}
    assert (
        "processing",
        "editing",
        72,
        "Preparando clip 1 de 1…",
    ) in progress_events
    assert (
        "processing",
        "editing",
        72,
        "Analizando reencuadre del clip 1 de 1… 50%",
    ) in progress_events
    assert (
        "processing",
        "editing",
        85,
        "Editando clip 1 de 1…",
    ) in progress_events


def test_social_captions_select_hashtags_for_the_target_platform() -> None:
    clip = {
        "title": "Consejo",
        "description": "Un consejo útil.",
        "hashtags": {
            "youtube": ["Youtube"],
            "tiktok": ["TikTok"],
            "instagram": ["Instagram"],
        },
    }

    assert integrations._publication_caption(
        clip, "tiktok", max_length=2200
    ) == "Un consejo útil.\n\n#TikTok"
    assert integrations._publication_caption(
        clip, "instagram", max_length=2200
    ) == "Un consejo útil.\n\n#Instagram"


def test_ffmpeg_timeout_kills_and_reaps_the_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stopped = threading.Event()

    class BlockingStdout:
        def __iter__(self):
            return self

        def __next__(self):
            stopped.wait(timeout=2)
            if stopped.is_set():
                raise StopIteration
            raise AssertionError("The render timeout did not stop reading FFmpeg output.")

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stdout = BlockingStdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True
            stopped.set()

        def wait(self, timeout=None):
            self.wait_calls += 1
            return -9 if self.killed else 0

    process = FakeProcess()
    launch_options: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        launch_options.update(kwargs)
        return process

    monkeypatch.setattr(media_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(media_module, "_FFMPEG_RENDER_TIMEOUT_SECONDS", 0.01, raising=False)
    proposal = ClipProposal(
        inicio=0,
        fin=1,
        titulo="Clip de prueba",
        puntuacion=7,
        razon_viral="Una frase.",
    )

    with pytest.raises(MediaProcessingError, match="tiempo máximo"):
        render_clip(
            tmp_path / "source.mp4",
            tmp_path / "output" / "clip.mp4",
            tmp_path / "output" / "clip.ass",
            proposal,
            ProjectConfig(options={"face_tracking": False}),
            [],
            "ffmpeg",
            lambda _percent: None,
        )

    assert process.killed
    assert process.wait_calls >= 1
    if media_module.os.name == "nt":
        assert launch_options["creationflags"] == media_module.subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in launch_options


def test_ffprobe_subprocess_hides_console_on_windows(monkeypatch) -> None:
    launch_options: dict[str, object] = {}

    def fake_run(arguments, **kwargs):
        launch_options.update(kwargs)
        return media_module.subprocess.CompletedProcess(
            arguments,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(media_module.subprocess, "run", fake_run)

    media_module._run(["ffprobe", "-version"], "leer los metadatos")

    if media_module.os.name == "nt":
        assert launch_options["creationflags"] == media_module.subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in launch_options


def test_each_database_connection_enforces_foreign_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)

    with database.connection(settings.database_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO project_events(project_id, event_type, payload_json, created_at)
                VALUES ('missing-project', 'progress', '{}', 'now')
                """
            )


def test_deferred_render_callbacks_keep_their_clip_index(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = "callback-project"
    input_path = settings.data_dir / "projects" / project_id / "source"
    database.create_project(
        settings.database_path,
        project_id,
        "Callbacks",
        input_path,
        {"ratio": "9:16"},
        "test-user",
    )
    callbacks = []
    monkeypatch.setattr(
        pipeline.database, "resolve_subtitle_template", lambda *_args: "Default"
    )

    def capture_callback(*args, **_kwargs):
        callbacks.append(args[7])
        return (10.0, [])

    monkeypatch.setattr(pipeline, "render_clip", capture_callback)
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda *_args, **_kwargs: {"width": 720, "height": 1280},
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffprobe")
    proposals = [
        ClipProposal(
            inicio=0,
            fin=10,
            titulo=f"Clip {index + 1}",
            puntuacion=7,
            razon_viral="Una frase.",
        )
        for index in range(2)
    ]

    pipeline._render_projects(
        settings,
        project_id,
        input_path,
        ProjectConfig(),
        proposals,
        [],
        "ffmpeg",
        "test-user",
    )
    for callback in callbacks:
        callback(50)

    events = database.events_after(settings.database_path, project_id, 1)
    messages = [event["data"]["message"] for event in events]
    assert [message for message in messages if message.startswith("Preparando")] == [
        "Preparando clip 1 de 2…",
        "Preparando clip 2 de 2…",
    ]
    assert [message for message in messages if message.startswith("Editando")] == [
        "Editando clip 1 de 2…",
        "Editando clip 2 de 2…",
    ]


def test_youtube_oauth_requests_read_scope_for_channel_lookup(tmp_path: Path) -> None:
    settings = replace(
        _settings(tmp_path / "data"),
        youtube_client_id="client-id",
        youtube_client_secret="client-secret",
    )
    authorization = integrations.authorization_url(settings, "youtube", "state")
    scopes = set(parse_qs(urlparse(authorization).query)["scope"][0].split())

    assert "https://www.googleapis.com/auth/youtube.upload" in scopes
    assert "https://www.googleapis.com/auth/youtube.readonly" in scopes
