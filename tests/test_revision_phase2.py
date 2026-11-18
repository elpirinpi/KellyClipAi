from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import database
from backend import llm as llm_module
from backend import main as main_module
from backend import pipeline as pipeline_module
from backend.llm import LLMConfigurationError, ResolvedLLM
from backend.process_queue import ProjectProcessingQueue
from backend.schemas import ClipProposal, ProjectConfig, TranscriptSegment
from backend.settings import AppSettings


@pytest.mark.skipif(os.name != "nt", reason="DPAPI solo está disponible en Windows")
def test_oauth_tokens_are_encrypted_and_legacy_tokens_migrate(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)

    with database.connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, expires_at, metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-owner",
                "instagram",
                "account-id",
                "Kelly",
                "legacy-access-token",
                "legacy-refresh-token",
                None,
                json.dumps({"user_access_token": "legacy-user-token"}),
                database.utc_now(),
            ),
        )

    database.initialize(database_path)
    with database.connection(database_path) as connection:
        stored = connection.execute(
            """
            SELECT access_token, refresh_token, metadata_json
            FROM connected_accounts WHERE owner_id = ? AND provider = ?
            """,
            ("test-owner", "instagram"),
        ).fetchone()
    account = database.get_connected_account(database_path, "instagram", "test-owner")

    assert stored["access_token"] != "legacy-access-token"
    assert stored["refresh_token"] != "legacy-refresh-token"
    assert "legacy-user-token" not in stored["metadata_json"]
    assert account is not None
    assert account["access_token"] == "legacy-access-token"
    assert account["refresh_token"] == "legacy-refresh-token"
    assert account["metadata"]["user_access_token"] == "legacy-user-token"

    database.save_connected_account(
        database_path,
        "instagram",
        "new-account",
        "Kelly",
        "new-access-token",
        "new-refresh-token",
        None,
        {"user_access_token": "new-user-token"},
        "new-owner",
    )
    with database.connection(database_path) as connection:
        newly_saved = connection.execute(
            """
            SELECT access_token, refresh_token, metadata_json
            FROM connected_accounts WHERE owner_id = ? AND provider = ?
            """,
            ("new-owner", "instagram"),
        ).fetchone()
    assert "new-access-token" not in newly_saved["access_token"]
    assert "new-refresh-token" not in newly_saved["refresh_token"]
    assert "new-user-token" not in newly_saved["metadata_json"]
    saved_account = database.get_connected_account(
        database_path, "instagram", "new-owner"
    )
    assert saved_account is not None
    assert saved_account["access_token"] == "new-access-token"
    assert saved_account["refresh_token"] == "new-refresh-token"
    assert saved_account["metadata"]["user_access_token"] == "new-user-token"

    database.initialize(database_path)
    assert database.get_connected_account(database_path, "instagram", "test-owner") == account


@pytest.mark.parametrize(("max_concurrency", "expected_parallel"), [(1, 1), (2, 2)])
def test_project_queue_enforces_configured_concurrency(
    max_concurrency: int,
    expected_parallel: int,
) -> None:
    async def verify() -> None:
        active = 0
        max_active = 0
        lock = asyncio.Lock()
        parallel_started = asyncio.Event()
        release_workers = asyncio.Event()

        async def process_project(
            _project_id: str,
            _llm_config: object,
            _retry_failed_only: bool,
        ) -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
                if active == expected_parallel:
                    parallel_started.set()
            await release_workers.wait()
            async with lock:
                active -= 1

        queue = ProjectProcessingQueue(max_concurrency, process_project)
        await queue.start()
        for project_id in ("project-1", "project-2"):
            await queue.submit(project_id, None)

        await asyncio.wait_for(parallel_started.wait(), timeout=1)
        assert max_active == expected_parallel
        release_workers.set()
        await queue.join()
        await queue.stop()

    asyncio.run(verify())


@pytest.mark.parametrize(
    ("status_code", "message", "expected_attempts"),
    [
        (401, "La clave API", 1),
        (402, "crédito o la facturación", 1),
        (429, "límite de solicitudes", 3),
    ],
)
def test_llm_provider_errors_are_translated_without_raw_body(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    message: str,
    expected_attempts: int,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, text="private provider error body")

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr("time.sleep", delays.append)

    with pytest.raises(LLMConfigurationError, match=message) as error:
        llm_module.propose_clips(
            [],
            ProjectConfig(),
            60,
            ResolvedLLM("OpenAI", "test-token", "test-model", "https://llm.example"),
        )

    assert "private provider error body" not in str(error.value)
    assert attempts == expected_attempts
    assert delays == ([0.5, 1.0] if status_code == 429 else [])


def test_llm_retries_server_errors_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="private provider error body")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"clips": []}'}}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr("time.sleep", delays.append)

    proposals = llm_module.propose_clips(
        [],
        ProjectConfig(),
        60,
        ResolvedLLM("OpenAI", "test-token", "test-model", "https://llm.example"),
    )

    assert proposals == []
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_failed_clip_can_retry_without_regenerating_proposals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_base_url="",
        whisper_model="small",
    )
    database.initialize(settings.database_path)
    project_id = "partial-render-project"
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True)
    input_path = project_dir / "source"
    input_path.write_bytes(b"video source")
    config = ProjectConfig()
    database.create_project(
        settings.database_path,
        project_id,
        "Partial render",
        input_path,
        config.model_dump(mode="json"),
        "test-owner",
    )
    transcript = [
        TranscriptSegment(
            start=0,
            end=1,
            text="Palabras reales.",
            words=[{"start": 0.1, "end": 0.5, "text": "reales."}],
        )
    ]
    database.update_project(
        settings.database_path,
        project_id,
        metadata={"duration": 100, "width": 1920, "height": 1080},
        transcript=[segment.model_dump(mode="json") for segment in transcript],
    )
    proposals = [
        ClipProposal(
            inicio=0,
            fin=20,
            titulo="Clip uno",
            puntuacion=9,
            razon_viral="Gancho claro.",
        ),
        ClipProposal(
            inicio=30,
            fin=50,
            titulo="Clip dos",
            puntuacion=8,
            razon_viral="Otra idea completa.",
        ),
    ]
    proposal_calls = 0
    render_attempts: dict[str, int] = {}

    def propose(*_args: object, **_kwargs: object) -> list[ClipProposal]:
        nonlocal proposal_calls
        proposal_calls += 1
        return proposals

    def render(
        _input_path: Path,
        output_path: Path,
        _subtitle_path: Path,
        proposal: ClipProposal,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, list[dict]]:
        render_attempts[proposal.titulo] = render_attempts.get(proposal.titulo, 0) + 1
        if proposal.titulo == "Clip dos" and render_attempts[proposal.titulo] == 1:
            raise pipeline_module.MediaProcessingError("Fallo real de render de prueba.")
        output_path.write_bytes(b"rendered clip")
        return proposal.fin - proposal.inicio, []

    monkeypatch.setattr(pipeline_module.shutil, "which", lambda _binary: "ffmpeg")
    monkeypatch.setattr(
        pipeline_module,
        "resolve_llm",
        lambda *_args: ResolvedLLM(
            "Groq", "test-key", "test-model", "https://llm.example"
        ),
    )
    monkeypatch.setattr(pipeline_module, "_propose_valid_clips", propose)
    monkeypatch.setattr(pipeline_module, "render_clip", render)
    monkeypatch.setattr(
        pipeline_module,
        "probe_media",
        lambda *_args: {"duration": 20, "width": 720, "height": 1280},
    )

    assert database.queue_project(
        settings.database_path, project_id, config.model_dump(mode="json")
    )
    pipeline_module.process_project(project_id, None, settings)
    partial = database.get_project(settings.database_path, project_id)
    assert partial is not None
    assert [clip["status"] for clip in partial["clips"]] == ["ready", "failed"]
    first_clip_id = partial["clips"][0]["id"]

    assert database.queue_project(
        settings.database_path,
        project_id,
        config.model_dump(mode="json"),
        preserve_clips=True,
    )
    pipeline_module.process_project(
        project_id, None, settings, retry_failed_only=True
    )

    retried = database.get_project(settings.database_path, project_id)
    assert retried is not None
    assert proposal_calls == 1
    assert [clip["status"] for clip in retried["clips"]] == ["ready", "ready"]
    assert retried["clips"][0]["id"] == first_clip_id
    assert render_attempts == {"Clip uno": 1, "Clip dos": 2}


def test_retry_failed_clips_api_uses_queue_without_an_llm_key(
    monkeypatch: pytest.MonkeyPatch,
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
    database.initialize(settings.database_path)
    project_id = "failed-clip-api-project"
    config = ProjectConfig()
    database.create_project(
        settings.database_path,
        project_id,
        "Failed clip",
        tmp_path / "source.mp4",
        config.model_dump(mode="json"),
        "test-user",
    )
    failed_proposal = ClipProposal(
        inicio=0,
        fin=20,
        titulo="Clip pendiente",
        puntuacion=7,
        razon_viral="Una razón real.",
    )
    database.update_project(
        settings.database_path,
        project_id,
        status="completed",
        metadata={
            "duration": 30,
            "clip_proposals": [
                {
                    "id": "failed-clip-id",
                    "proposal": failed_proposal.model_dump(mode="json"),
                }
            ],
        },
        clips=[
            {
                "id": "failed-clip-id",
                "title": failed_proposal.titulo,
                "status": "failed",
                "error": "Fallo de render.",
            }
        ],
    )
    queued: list[tuple[str, bool]] = []
    retry_started = threading.Event()

    def process_project(
        queued_project_id: str,
        _llm_config: object,
        _settings: AppSettings,
        *,
        retry_failed_only: bool,
    ) -> None:
        queued.append((queued_project_id, retry_failed_only))
        retry_started.set()

    monkeypatch.setattr(main_module, "process_project", process_project)
    with TestClient(main_module.create_app(settings, auth_required=False)) as client:
        response = client.post(f"/api/projects/{project_id}/retry-failed-clips")
        assert response.status_code == 202, response.text
        assert retry_started.wait(timeout=2)
        project_response = client.get(f"/api/projects/{project_id}")

    assert response.status_code == 202
    assert response.json() == {"project_id": project_id, "status": "queued"}
    assert queued == [(project_id, True)]
    failed_clip = project_response.json()["clips"][0]
    assert failed_clip["status"] == "failed"
    assert failed_clip["video_url"] is None
    assert failed_clip["download_url"] is None
