"""Tests de la Fase 2 (v1.1.0), backend: cola, programador y editor.

Cubre: cola de procesamiento persistente con recuperación y reintentos
(ítem 10), supervisor del programador con timeout por job (ítem 7) y
renders del editor como trabajos asíncronos con progreso SSE (ítem 6).
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.editor as editor_module
import backend.process_queue as process_queue_module
import backend.scheduler as scheduler_module
from backend import database
from backend.main import create_app as create_application
from backend.process_queue import ProjectProcessingQueue
from backend.schemas import ProjectConfig
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


def _create_project(
    settings: AppSettings, project_id: str, *, status: str = "processing"
) -> Path:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    database.create_project(
        settings.database_path,
        project_id,
        f"Proyecto {project_id}",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    if status != "processing":
        database.update_project(settings.database_path, project_id, status=status)
    return project_dir


def _make_video(path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg no está disponible.")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-f",
            "mp4",  # el destino ("source") no tiene extensión
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


# ---------------------------------------------------------------------------
# Ítem 10: cola persistente, recuperación, reintentos y apagado elegante
# ---------------------------------------------------------------------------


def test_queue_submit_persists_job_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)

    async def run() -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        async def processor(_pid: str, _llm: Any, _retry: bool) -> None:
            started.set()
            await release.wait()

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        await queue.submit(project_id, None)
        await asyncio.wait_for(started.wait(), timeout=5)
        with sqlite3.connect(settings.database_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status FROM processing_jobs WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        assert row is not None
        assert row["status"] == "running"
        # Segunda subida del mismo proyecto: duplicado activo rechazado.
        with pytest.raises(RuntimeError, match="ya tiene un trabajo en la cola"):
            await queue.submit(project_id, None)
        release.set()
        await queue.join()
        await queue.stop()

    asyncio.run(run())


def test_queue_recovers_persisted_job_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    job_id, created = database.enqueue_processing_job(
        settings.database_path, project_id, False
    )
    assert created is True

    async def run() -> None:
        processed = asyncio.Event()

        async def processor(pid: str, _llm: Any, _retry: bool) -> None:
            assert pid == project_id
            database.update_project(
                settings.database_path, pid, status="completed"
            )
            processed.set()

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        await asyncio.wait_for(processed.wait(), timeout=5)
        await queue.join()
        await queue.stop()

    asyncio.run(run())
    row = database.get_processing_job(settings.database_path, job_id)
    assert row is not None and row["status"] == "done"


def test_queue_retries_transient_errors_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_queue_module, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)

    async def run() -> None:
        calls: list[int] = []
        done = asyncio.Event()

        async def processor(pid: str, _llm: Any, _retry: bool) -> None:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("Connection timeout con el proveedor")
            database.update_project(
                settings.database_path, pid, status="completed"
            )
            done.set()

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        await queue.submit(project_id, None)
        await asyncio.wait_for(done.wait(), timeout=10)
        await queue.join()
        await queue.stop()
        assert len(calls) == 3

    asyncio.run(run())
    rows = database.list_recoverable_processing_jobs(settings.database_path)
    assert rows == []


def test_queue_gives_up_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_queue_module, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    job_id, _ = database.enqueue_processing_job(
        settings.database_path, project_id, False
    )

    async def run() -> None:
        calls: list[int] = []
        gave_up = asyncio.Event()

        async def processor(_pid: str, _llm: Any, _retry: bool) -> None:
            calls.append(1)
            if len(calls) >= 3:
                gave_up.set()
            raise RuntimeError("Connection timeout con el proveedor")

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        # El trabajo ya está persistido: start() lo recupera y lo ejecuta.
        await asyncio.wait_for(gave_up.wait(), timeout=10)
        await asyncio.sleep(0.2)
        await queue.stop()
        assert len(calls) == 3

    asyncio.run(run())
    row = database.get_processing_job(settings.database_path, job_id)
    assert row is not None and row["status"] == "failed"


def test_queue_does_not_retry_non_transient_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_queue_module, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    job_id, _ = database.enqueue_processing_job(
        settings.database_path, project_id, False
    )

    async def run() -> None:
        calls: list[int] = []
        failed = asyncio.Event()

        async def processor(_pid: str, _llm: Any, _retry: bool) -> None:
            calls.append(1)
            failed.set()
            raise ValueError("bug de programa, no transitorio")

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path
        )
        await queue.start()
        await asyncio.wait_for(failed.wait(), timeout=5)
        await asyncio.sleep(0.2)
        await queue.stop()
        assert len(calls) == 1

    asyncio.run(run())
    row = database.get_processing_job(settings.database_path, job_id)
    assert row is not None and row["status"] == "failed"


def test_queue_stop_drains_in_flight_work(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)

    async def run() -> None:
        started = asyncio.Event()
        finished = asyncio.Event()

        async def processor(pid: str, _llm: Any, _retry: bool) -> None:
            started.set()
            await asyncio.sleep(0.3)
            database.update_project(
                settings.database_path, pid, status="completed"
            )
            finished.set()

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path, drain_timeout=5.0
        )
        await queue.start()
        await queue.submit(project_id, None)
        # Esperar a que el trabajo esté EN CURSO: stop() drena solo el
        # trabajo en curso; uno aún pendiente se queda persistido.
        await asyncio.wait_for(started.wait(), timeout=5)
        await queue.stop()
        assert finished.is_set()

    asyncio.run(run())


def test_queue_stop_times_out_and_returns_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_queue_module, "_POST_KILL_GRACE_SECONDS", 0.05)
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    job_id, _ = database.enqueue_processing_job(
        settings.database_path, project_id, False
    )

    async def run() -> None:
        started = asyncio.Event()

        async def processor(_pid: str, _llm: Any, _retry: bool) -> None:
            started.set()
            await asyncio.sleep(30)

        queue = ProjectProcessingQueue(
            1, processor, database_path=settings.database_path, drain_timeout=0.1
        )
        await queue.start()
        await asyncio.wait_for(started.wait(), timeout=5)
        start = time.monotonic()
        await queue.stop()
        assert time.monotonic() - start < 3.0

    asyncio.run(run())
    # El trabajo interrumpido queda 'running' y se recupera al arrancar.
    row = database.get_processing_job(settings.database_path, job_id)
    assert row is not None and row["status"] == "running"


# ---------------------------------------------------------------------------
# Ítem 7: supervisor del programador
# ---------------------------------------------------------------------------


def test_scheduler_supervisor_survives_database_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    calls: list[int] = []

    def flaky_due_jobs(_database_path: Path, _now: str) -> list:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("la BD no responde")
        return []

    monkeypatch.setattr(database, "due_schedule_jobs", flaky_due_jobs)
    monkeypatch.setattr(scheduler_module, "_SUPERVISOR_INITIAL_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(scheduler_module, "_POLL_INTERVAL_SECONDS", 0.01)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                scheduler_module.schedule_worker(settings), timeout=0.5
            )

    asyncio.run(run())
    # Sobrevivió al error y siguió sondeando.
    assert len(calls) >= 2


def test_scheduler_job_timeout_marks_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_project(settings, project_id)
    job_id = str(uuid.uuid4())
    now = database.utc_now()
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs
                (id, project_id, clip_id, platform, scheduled_at, visibility,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, 'youtube', ?, 'private', 'scheduled', ?, ?)
            """,
            (job_id, project_id, "clip-1", "2000-01-01T00:00:00Z", now, now),
        )
        conn.commit()

    def slow_job(_settings: AppSettings, _job: dict[str, Any]) -> None:
        time.sleep(5)

    monkeypatch.setattr(scheduler_module, "process_schedule_job", slow_job)
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_JOB_TIMEOUT_SECONDS", 0.05)

    async def run() -> None:
        await scheduler_module._process_due_jobs(settings)

    asyncio.run(run())
    job = database.get_schedule_job(settings.database_path, job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert "tiempo máximo" in (job["error"] or "")


# ---------------------------------------------------------------------------
# Ítem 6: renders del editor como trabajos asíncronos con progreso SSE
# ---------------------------------------------------------------------------


def _create_completed_project_with_clip(
    settings: AppSettings,
    project_id: str,
    *,
    with_transcript: bool = False,
) -> str:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    _make_video(project_dir / "source")
    clip = {
        "id": "clip-1",
        "title": "Clip de prueba",
        "start": 0.0,
        "end": 2.0,
        "duration": 2.0,
        "score": 8,
        "viral_reason": "Una idea completa.",
        "description": "Descripción.",
        "hashtags": {"youtube": [], "tiktok": [], "instagram": []},
        "keywords": [],
        "emojis": [],
        "resolution": 720,
        "filename": "clip-1.mp4",
        "transcript": (
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "palabra",
                    "preview_start": 0.0,
                    "preview_end": 1.0,
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "otra",
                    "preview_start": 1.0,
                    "preview_end": 2.0,
                },
            ]
            if with_transcript
            else []
        ),
        "width": 720,
        "height": 1280,
        "status": "ready",
    }
    config = ProjectConfig(framing="center")
    database.create_project(
        settings.database_path,
        project_id,
        "Proyecto editor",
        project_dir / "source",
        config.model_dump(mode="json"),
        "test-user",
    )
    database.update_project(
        settings.database_path,
        project_id,
        status="completed",
        clips=[clip],
        metadata={"duration": 2.0, "width": 320, "height": 240, "has_audio": True},
    )
    return "clip-1"


def _edit_payload() -> dict[str, Any]:
    return {
        "ratio": "9:16",
        "subtitle_template": "Default",
        "phrases": [],
    }


def _poll_job(
    client: TestClient, job_id: str, *, timeout: float = 30.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/editor-jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"El trabajo {job_id} no terminó en {timeout} s: {job}")


def test_editor_preview_job_completes_with_real_render(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        clip_id = _create_completed_project_with_clip(settings, project_id)
        created = client.post(
            f"/api/projects/{project_id}/clips/{clip_id}/edit/preview/jobs",
            json=_edit_payload(),
        )
        assert created.status_code == 202
        job = created.json()
        assert job["status"] in {"queued", "running"}
        assert job["kind"] == "preview"
        finished = _poll_job(client, job["id"])
        assert finished["status"] == "completed"
        assert finished["result"] is not None


def test_editor_job_cancelled_while_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_render(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        time.sleep(0.5)
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress(50.0)
        return {"output_path": "clip.mp4"}

    monkeypatch.setattr(editor_module, "render_editor_version", slow_render)
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        clip_id = _create_completed_project_with_clip(settings, project_id)
        created = client.post(
            f"/api/projects/{project_id}/clips/{clip_id}/edit/preview/jobs",
            json=_edit_payload(),
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        cancelled = client.post(f"/api/editor-jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        finished = _poll_job(client, job_id, timeout=10.0)
        assert finished["status"] == "cancelled"
        # Un trabajo terminado ya no se puede cancelar.
        again = client.post(f"/api/editor-jobs/{job_id}/cancel")
        assert again.status_code == 409


def test_editor_job_events_stream_ends_on_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        editor_module,
        "render_editor_version",
        lambda *_a, **_k: {"output_path": "clip.mp4"},
    )
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        clip_id = _create_completed_project_with_clip(settings, project_id)
        created = client.post(
            f"/api/projects/{project_id}/clips/{clip_id}/edit/preview/jobs",
            json=_edit_payload(),
        )
        job_id = created.json()["id"]
        _poll_job(client, job_id, timeout=10.0)
        with client.stream("GET", f"/api/editor-jobs/{job_id}/events") as response:
            assert response.status_code == 200
            text = response.read().decode("utf-8")
        assert "completed" in text
        assert "event: progress" in text


def test_editor_job_invalid_edit_returns_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        project_id = str(uuid.uuid4())
        clip_id = _create_completed_project_with_clip(
            settings, project_id, with_transcript=True
        )
        # El clip tiene frases base; una edición sin ellas es inválida.
        response = client.post(
            f"/api/projects/{project_id}/clips/{clip_id}/edit/preview/jobs",
            json=_edit_payload(),
        )
        assert response.status_code == 422


def test_editor_job_unknown_id_returns_404(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/editor-jobs/no-existe").status_code == 404
        assert client.post("/api/editor-jobs/no-existe/cancel").status_code == 404
        assert client.get("/api/editor-jobs/no-existe/events").status_code == 404
