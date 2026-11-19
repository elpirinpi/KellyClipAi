"""Tests de la Fase 2 (v1.1.0), backend: rendimiento y robustez.

Cubre: NVENC con fallback a libx264 (ítem 1), throttle de progreso y no
reescribir clips_json por clip (ítem 2), cachés de análisis y de
transcripción (ítem 3), paginación de proyectos (ítem 4), backup de BD
(ítem 8), PyAV vs ffprobe (ítem 25), manejo paciente del 429 de Groq
(backoff, Retry-After, TPM/TPD, pacing entre ventanas, max_tokens 2000)
y subida reanudable (ítem 22).

La cola persistente, el supervisor del programador y los trabajos
asíncronos del editor viven en tests/test_phase2_queue_scheduler.py.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.llm as llm_module
import backend.media as media_module
from backend import database, face_tracking, pipeline
from backend.llm import (
    LLMConfigurationError,
    ResolvedLLM,
    propose_clips,
)
from backend.main import create_app as create_application
from backend.media import MediaProcessingError
from backend.pipeline import ProjectCancelled, _TokenPacer
from backend.schemas import (
    ClipProposal,
    ProjectConfig,
    TranscriptSegment,
    Word,
)
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


def _make_video(
    path: Path, *, width: int = 640, height: int = 360, with_audio: bool = True
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg no está disponible.")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration=2:size={width}x{height}:rate=30",
    ]
    if with_audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-shortest",
        ]
    command += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if with_audio:
        command += ["-c:a", "aac"]
    command += ["-y", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return path


def _install_llm_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> None:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **_kwargs: client)


def _choices_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )


def _groq_llm() -> ResolvedLLM:
    return ResolvedLLM(
        "Groq", "test-key", "qwen/qwen3.8-27b", "https://api.groq.com/openai/v1"
    )


# ---------------------------------------------------------------------------
# Ítem 1: NVENC con fallback a libx264
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_nvenc_cache():
    media_module.reset_nvenc_cache()
    yield
    media_module.reset_nvenc_cache()


def _fake_run_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    encoders_stdout: str,
    probe_returncode: int = 0,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(list(args))
        if "-encoders" in args:
            return SimpleNamespace(stdout=encoders_stdout, returncode=0)
        return SimpleNamespace(stdout="", returncode=probe_returncode)

    monkeypatch.setattr(media_module.subprocess, "run", fake_run)
    return calls


def test_nvenc_available_when_encoder_listed_and_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch,
        encoders_stdout=" V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n",
    )
    assert media_module.is_nvenc_available("ffmpeg") is True
    assert media_module._video_encoder_args("ffmpeg") == [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "21",
        "-pix_fmt",
        "yuv420p",
    ]
    # Una llamada a -encoders y una al encode de prueba.
    assert len(calls) == 2
    assert "-encoders" in calls[0]
    assert "h264_nvenc" in calls[1]


def test_nvenc_unavailable_when_encoder_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch, encoders_stdout=" V..... libx264  libx264 H.264 encoder\n"
    )
    assert media_module.is_nvenc_available("ffmpeg") is False
    # Sin codificador, ni siquiera se intenta el encode de prueba.
    assert len(calls) == 1
    assert media_module._video_encoder_args("ffmpeg") == [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
    ]


def test_nvenc_unavailable_when_listed_but_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # El caso clave: ffmpeg con NVENC compilado en un equipo sin GPU
    # NVIDIA lista el codificador pero el encode falla.
    _fake_run_factory(
        monkeypatch,
        encoders_stdout=" V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n",
        probe_returncode=1,
    )
    assert media_module.is_nvenc_available("ffmpeg") is False


def test_nvenc_detection_is_cached_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch,
        encoders_stdout=" V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n",
    )
    assert media_module.is_nvenc_available("ffmpeg") is True
    assert media_module.is_nvenc_available("ffmpeg") is True
    assert media_module.is_nvenc_available("ffmpeg") is True
    assert len(calls) == 2


def test_nvenc_disabled_by_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run_factory(
        monkeypatch,
        encoders_stdout=" V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n",
    )
    monkeypatch.setenv("KELLYCLIP_NVENC", "0")
    assert media_module.is_nvenc_available("ffmpeg") is False
    assert calls == []


# ---------------------------------------------------------------------------
# Ítem 3: caché de transcripción por hash de archivo
# ---------------------------------------------------------------------------


def test_transcript_cache_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    transcript = [
        TranscriptSegment(
            start=0, end=3, text="Una idea útil.", words=[Word(start=1, end=2, text="idea")]
        ).model_dump(mode="json")
    ]
    database.save_cached_transcript(
        settings.database_path, "abc123", "small", "auto", "es", transcript, 2.0
    )
    cached = database.get_cached_transcript(
        settings.database_path, "abc123", "small", "auto"
    )
    assert cached is not None
    assert cached["language"] == "es"
    assert cached["transcript"] == transcript
    assert cached["duration"] == 2.0


def test_transcript_cache_invalidates_on_model_or_language_change(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    database.save_cached_transcript(
        settings.database_path, "abc123", "small", "auto", "es", [], 2.0
    )
    assert (
        database.get_cached_transcript(
            settings.database_path, "abc123", "base", "auto"
        )
        is None
    )
    assert (
        database.get_cached_transcript(
            settings.database_path, "abc123", "small", "es"
        )
        is None
    )
    assert (
        database.get_cached_transcript(
            settings.database_path, "abc123", "small", "auto"
        )
        is not None
    )


def test_transcript_cache_prunes_to_ten_entries(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    for index in range(12):
        database.save_cached_transcript(
            settings.database_path,
            f"hash-{index:02d}",
            "small",
            "auto",
            "es",
            [],
            2.0,
        )
    with database.connection(settings.database_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM transcript_cache").fetchone()[0]
    assert count == 10
    # Las dos más antiguas se podaron; la más reciente sigue.
    assert (
        database.get_cached_transcript(
            settings.database_path, "hash-00", "small", "auto"
        )
        is None
    )
    assert (
        database.get_cached_transcript(
            settings.database_path, "hash-11", "small", "auto"
        )
        is not None
    )


def test_pipeline_reuses_cached_transcript_across_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    source_bytes = b"same-video-bytes"
    segment = TranscriptSegment(
        start=0,
        end=3,
        text="Una idea útil.",
        words=[Word(start=1, end=2, text="idea")],
    )
    transcribe_calls: list[Path] = []

    def fake_transcribe(path: Path, *_args: Any, **_kwargs: Any):
        transcribe_calls.append(path)
        return [segment], "es"

    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda _path, _ffprobe: {
            "duration": 2.0,
            "width": 320,
            "height": 240,
            "has_audio": True,
        },
    )
    monkeypatch.setattr(pipeline, "create_thumbnail", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "_propose_valid_clips", lambda *a, **k: [])

    project_ids = []
    for index in range(2):
        project_id = str(uuid.uuid4())
        project_ids.append(project_id)
        project_dir = settings.data_dir / "projects" / project_id
        project_dir.mkdir(parents=True)
        (project_dir / "source").write_bytes(source_bytes)
        database.create_project(
            settings.database_path,
            project_id,
            f"Proyecto {index}",
            project_dir / "source",
            ProjectConfig().model_dump(mode="json"),
            "test-user",
        )
        pipeline.process_project(project_id, None, settings)

    # Un solo Whisper para dos proyectos con el mismo video.
    assert len(transcribe_calls) == 1
    second = database.get_project(settings.database_path, project_ids[1])
    assert second is not None
    assert second["status"] == "completed"
    assert second["transcript"][0]["text"] == "Una idea útil."


# ---------------------------------------------------------------------------
# Ítem 4: paginación en GET /api/projects
# ---------------------------------------------------------------------------


def _create_listed_project(
    settings: AppSettings, project_id: str, *, with_transcript: bool = False
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    clips = []
    if with_transcript:
        clips = [
            {
                "id": "clip-1",
                "title": "Clip",
                "start": 0.0,
                "end": 10.0,
                "transcript": [
                    {"start": 0.0, "end": 1.0, "text": "palabra"},
                    {"start": 1.0, "end": 2.0, "text": "por"},
                    {"start": 2.0, "end": 3.0, "text": "palabra"},
                ],
            }
        ]
    database.create_project(
        settings.database_path,
        project_id,
        f"Proyecto {project_id}",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    if clips:
        database.update_project(settings.database_path, project_id, clips=clips)


def test_projects_pagination_envelope_and_pages(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        for index in range(3):
            _create_listed_project(settings, f"project-{index}")
        first = client.get("/api/projects", params={"limit": 2, "offset": 0})
        assert first.status_code == 200
        body = first.json()
        assert body["total"] == 3
        assert body["limit"] == 2
        assert body["offset"] == 0
        assert len(body["items"]) == 2
        second = client.get("/api/projects", params={"limit": 2, "offset": 2})
        assert second.status_code == 200
        assert second.json()["total"] == 3
        assert len(second.json()["items"]) == 1
        # Sin parámetros: la página por defecto trae los 3.
        default = client.get("/api/projects")
        assert default.status_code == 200
        assert default.json()["total"] == 3
        assert default.json()["limit"] == 20
        assert len(default.json()["items"]) == 3


def test_projects_pagination_validates_bounds(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/projects", params={"limit": 101}).status_code == 422
        assert client.get("/api/projects", params={"limit": 0}).status_code == 422
        assert client.get("/api/projects", params={"offset": -1}).status_code == 422
        assert client.get("/api/projects", params={"limit": 100}).status_code == 200


def test_projects_list_excludes_clip_transcripts_but_detail_keeps_them(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        _create_listed_project(settings, "project-t", with_transcript=True)
        listed = client.get("/api/projects")
        assert listed.status_code == 200
        assert "transcript" not in listed.json()["items"][0]["clips"][0]
        detail = client.get("/api/projects/project-t")
        assert detail.status_code == 200
        assert len(detail.json()["clips"][0]["transcript"]) == 3


# ---------------------------------------------------------------------------
# Ítem 8: backup automático de la BD
# ---------------------------------------------------------------------------


def test_backup_database_creates_openable_copy(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_listed_project(settings, "project-b")
    backup = database.backup_database(settings.database_path, reason="manual")
    assert backup is not None
    assert backup.parent == settings.data_dir / "backups"
    assert backup.name.startswith("kellyclipai-")
    assert backup.suffix == ".sqlite3"
    with sqlite3.connect(backup) as conn:
        row = conn.execute(
            "SELECT title FROM projects WHERE id = 'project-b'"
        ).fetchone()
    assert row is not None and "project-b" in row[0]


def test_backup_database_returns_none_for_missing_or_empty_db(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.sqlite3"
    assert database.backup_database(missing) is None
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"")
    assert database.backup_database(empty) is None


def test_backup_retention_keeps_five_most_recent(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    backup_dir = settings.data_dir / "backups"
    # Limpiar la copia periódica del arranque para contar solo las de la prueba.
    for path in backup_dir.glob("*.sqlite3"):
        path.unlink()
    created: list[Path] = []
    # mtimes crecientes en el pasado: la más antigua es la primera, y la
    # copia recién creada (mtime real, posterior) nunca es la podada.
    base = time.time() - 100
    for index in range(7):
        path = database.backup_database(settings.database_path, reason="manual")
        assert path is not None
        # mtimes crecientes y deterministas: la más antigua es la primera.
        os.utime(path, (base + index, base + index))
        created.append(path)
    remaining = sorted(backup_dir.glob("*.sqlite3"))
    assert len(remaining) == 5
    assert set(remaining) == set(created[2:])


def test_pre_migration_backup_on_version_downgrade(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_listed_project(settings, "project-m")
    with sqlite3.connect(settings.database_path) as conn:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    database.initialize(settings.database_path)
    backups = list((settings.data_dir / "backups").glob("*pre-migration-v2*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(settings.database_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_periodic_backup_skipped_when_fresh_and_created_when_stale(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    # El arranque ya creó una copia periódica: no hace falta otra.
    assert database.ensure_periodic_backup(settings.database_path) is None
    backup_dir = settings.data_dir / "backups"
    for path in backup_dir.glob("*.sqlite3"):
        old = time.time() - (25 * 3600)
        os.utime(path, (old, old))
    created = database.ensure_periodic_backup(settings.database_path)
    assert created is not None
    assert "periodic" in created.name


# ---------------------------------------------------------------------------
# Ítem 2: throttle de progreso + no reescribir clips_json por clip
# ---------------------------------------------------------------------------


def _counting_update_project(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    original = database.update_project

    def wrapper(database_path: Path, project_id: str, **kwargs: Any) -> None:
        calls.append(dict(kwargs))
        return original(database_path, project_id, **kwargs)

    monkeypatch.setattr(database, "update_project", wrapper)
    return calls


def test_publish_throttles_repeated_same_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_listed_project(settings, "project-th")
    calls = _counting_update_project(monkeypatch)
    pipeline._reset_progress_throttle()
    pipeline._publish(settings, "project-th", "processing", "editing", 50, "A")
    pipeline._publish(settings, "project-th", "processing", "editing", 50, "B")
    pipeline._publish(settings, "project-th", "processing", "editing", 50, "C")
    assert len(calls) == 1
    # Un % distinto sí se publica.
    pipeline._publish(settings, "project-th", "processing", "editing", 51, "D")
    assert len(calls) == 2
    # Un paso distinto con el mismo % también.
    pipeline._publish(settings, "project-th", "processing", "finding", 51, "E")
    assert len(calls) == 3


def test_publish_terminal_always_writes_and_clears_throttle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    _create_listed_project(settings, "project-th2")
    calls = _counting_update_project(monkeypatch)
    pipeline._reset_progress_throttle()
    pipeline._publish(settings, "project-th2", "processing", "editing", 50, "A")
    pipeline._publish(settings, "project-th2", "completed", "done", 100, "Listo")
    assert len(calls) == 2
    # Tras el estado terminal, el mismo progreso vuelve a publicarse
    # (un reproceso no hereda el throttle del anterior).
    pipeline._publish(settings, "project-th2", "processing", "editing", 50, "A")
    assert len(calls) == 3


def _render_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AppSettings, str, Path, list[int]]:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True)
    input_path = project_dir / "source"
    input_path.write_bytes(b"video-bytes")
    database.create_project(
        settings.database_path,
        project_id,
        "Proyecto render",
        input_path,
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    rendered_calls: list[int] = []

    def fake_render_clip(*_args: Any, **_kwargs: Any) -> tuple[float, list]:
        rendered_calls.append(1)
        return 10.0, []

    monkeypatch.setattr(pipeline, "render_clip", fake_render_clip)
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda _path, _ffprobe: {
            "duration": 10.0,
            "width": 720,
            "height": 1280,
            "has_audio": True,
        },
    )
    return settings, project_id, input_path, rendered_calls


def _proposal_record(clip_id: str, start: float) -> dict[str, Any]:
    return {
        "id": clip_id,
        "proposal": ClipProposal(
            inicio=start,
            fin=start + 10,
            titulo=f"Clip {clip_id}",
            puntuacion=8,
            razon_viral="Una idea completa.",
        ).model_dump(mode="json"),
    }


def test_render_projects_does_not_rewrite_clips_json_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, project_id, input_path, rendered_calls = _render_setup(
        tmp_path, monkeypatch
    )
    calls = _counting_update_project(monkeypatch)
    pipeline._reset_progress_throttle()
    records = [_proposal_record("clip-1", 0.0), _proposal_record("clip-2", 20.0)]
    rendered = pipeline._render_projects(
        settings,
        project_id,
        input_path,
        ProjectConfig(),
        records,
        [],
        "ffmpeg",
        "test-user",
    )
    assert len(rendered_calls) == 2
    assert [clip["id"] for clip in rendered] == ["clip-1", "clip-2"]
    assert all(clip["status"] == "ready" for clip in rendered)
    # Progreso sí se publica, pero clips_json nunca se escribe por clip.
    assert calls
    assert not any("clips" in call for call in calls)


def test_cancel_during_render_keeps_rendered_and_pending_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, project_id, input_path, rendered_calls = _render_setup(
        tmp_path, monkeypatch
    )
    calls = _counting_update_project(monkeypatch)
    pipeline._reset_progress_throttle()
    kept_clip = {"id": "kept", "title": "Clip que queda", "start": 0.0, "status": "ready"}
    failed_original = {
        "id": "clip-2",
        "title": "Viejo fallido",
        "start": 20.0,
        "status": "failed",
        "error": "Falló antes.",
    }
    records = [_proposal_record("clip-1", 10.0), _proposal_record("clip-2", 20.0)]
    # Cancelar en cuanto termine el primer render.
    should_cancel = lambda: len(rendered_calls) >= 1  # noqa: E731
    with pytest.raises(ProjectCancelled) as error:
        pipeline._render_projects(
            settings,
            project_id,
            input_path,
            ProjectConfig(),
            records,
            [],
            "ffmpeg",
            "test-user",
            existing_clips=[kept_clip, failed_original],
            should_cancel=should_cancel,
        )
    cancelled = error.value.clips
    assert [clip["id"] for clip in cancelled] == ["kept", "clip-1", "clip-2"]
    assert cancelled[1]["status"] == "ready"
    # El pendiente conserva su entrada ORIGINAL (fallida), no desaparece.
    assert cancelled[2]["title"] == "Viejo fallido"
    assert cancelled[2]["status"] == "failed"
    assert not any("clips" in call for call in calls)


# ---------------------------------------------------------------------------
# Ítem 3: cachés de silencios y de análisis facial
# ---------------------------------------------------------------------------


def test_silence_cache_reuses_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    calls: list[tuple[float, float]] = []
    monkeypatch.setattr(
        media_module,
        "_detect_silence_intervals",
        lambda _p, start, duration, _f: calls.append((start, duration))
        or [(1.0, 2.0)],
    )
    media_module.clear_analysis_caches()
    first = media_module._silence_intervals(video, 0.0, 10.0, "ffmpeg")
    second = media_module._silence_intervals(video, 0.0, 10.0, "ffmpeg")
    assert first == second == [(1.0, 2.0)]
    assert len(calls) == 1
    # Otro rango sí vuelve a detectar.
    media_module._silence_intervals(video, 5.0, 10.0, "ffmpeg")
    assert len(calls) == 2


def test_face_analysis_cache_reuses_result_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    calls: list[int] = []
    sentinel = [[None]]
    monkeypatch.setattr(
        face_tracking,
        "_analyze_face_crop_paths_uncached",
        lambda *a, **k: calls.append(1) or sentinel,
    )
    face_tracking.clear_face_analysis_cache()
    progress: list[float] = []
    first = face_tracking.analyze_face_crop_paths(
        video, 0.0, [(0.0, 10.0)], 720, 1280
    )
    second = face_tracking.analyze_face_crop_paths(
        video,
        0.0,
        [(0.0, 10.0)],
        720,
        1280,
        on_progress=progress.append,
    )
    assert first is second is sentinel
    assert len(calls) == 1
    # En el acierto de caché se sigue notificando el 100 %.
    assert progress == [100.0]


# ---------------------------------------------------------------------------
# Ítem 25: PyAV vs ffprobe
# ---------------------------------------------------------------------------


def test_probe_media_uses_pyav(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "source.mp4", width=640, height=360)
    info = media_module.probe_media(video, "ffprobe")
    assert info["duration"] == pytest.approx(2.0, abs=0.1)
    assert info["width"] == 640
    assert info["height"] == 360
    assert info["has_audio"] is True


def test_probe_media_rejects_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"esto no es un video" * 10)
    with pytest.raises(MediaProcessingError):
        media_module.probe_media(corrupt, "ffprobe")


def test_probe_media_rejects_video_without_audio(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "silent.mp4", with_audio=False)
    with pytest.raises(MediaProcessingError):
        media_module.probe_media(video, "ffprobe")


def test_probe_media_falls_back_to_ffprobe_without_pyav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video(tmp_path / "source.mp4", width=640, height=360)
    monkeypatch.setitem(sys.modules, "av", None)
    info = media_module.probe_media(video, shutil.which("ffprobe") or "ffprobe")
    assert info["duration"] == pytest.approx(2.0, abs=0.1)
    assert info["width"] == 640
    assert info["height"] == 360
    assert info["has_audio"] is True


# ---------------------------------------------------------------------------
# 429 de Groq: backoff paciente, Retry-After, TPM vs TPD, max_tokens, pacing
# ---------------------------------------------------------------------------


def test_llm_429_then_success_returns_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="rate limited")
        return _choices_response('{"clips": []}')

    _install_llm_transport(monkeypatch, handle)
    assert propose_clips([], ProjectConfig(), 60, _groq_llm()) == []
    assert attempts == 2
    assert delays == [1.0]


def test_llm_persistent_429_gives_up_after_five_attempts_with_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    _install_llm_transport(
        monkeypatch,
        lambda _request: httpx.Response(429, text="rate limited"),
    )
    with pytest.raises(LLMConfigurationError) as error:
        propose_clips([], ProjectConfig(), 60, _groq_llm())
    assert "console.groq.com" in str(error.value)
    assert "espera" in str(error.value).lower()
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_llm_429_attempt_count_is_five(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, text="rate limited")

    _install_llm_transport(monkeypatch, handle)
    with pytest.raises(LLMConfigurationError):
        propose_clips([], ProjectConfig(), 60, _groq_llm())
    assert attempts == 5


def test_llm_429_honors_retry_after_with_120s_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "200"})
        return _choices_response('{"clips": []}')

    _install_llm_transport(monkeypatch, handle)
    assert propose_clips([], ProjectConfig(), 60, _groq_llm()) == []
    # Retry-After de 200 s, con tope de 120 s solo para 429.
    assert delays == [120.0]


def test_llm_429_retry_after_below_cap_is_used_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    attempts = 0

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "45"})
        return _choices_response('{"clips": []}')

    _install_llm_transport(monkeypatch, handle)
    assert propose_clips([], ProjectConfig(), 60, _groq_llm()) == []
    assert delays == [45.0]


def test_llm_on_rate_limit_callback_is_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    attempts = 0
    notifications: list[tuple[float, int]] = []

    def handle(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return httpx.Response(429, text="rate limited")
        return _choices_response('{"clips": []}')

    _install_llm_transport(monkeypatch, handle)
    propose_clips(
        [],
        ProjectConfig(),
        60,
        _groq_llm(),
        on_rate_limit=lambda wait, attempt: notifications.append((wait, attempt)),
    )
    assert notifications == [(1.0, 1), (2.0, 2)]


def _groq_429_response(message: str) -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "message": message,
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )


def test_llm_429_daily_quota_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    _install_llm_transport(
        monkeypatch,
        lambda _request: _groq_429_response(
            "Rate limit reached for model `qwen/qwen3.8-27b` on tokens per day "
            "(TPD): Limit 200000, Used 199500, Requested 1500. "
            "Please try again tomorrow."
        ),
    )
    with pytest.raises(LLMConfigurationError) as error:
        propose_clips([], ProjectConfig(), 60, _groq_llm())
    message = str(error.value)
    assert "cuota diaria" in message
    assert "se resetea mañana" in message
    assert "cambiar de modelo" in message
    assert "console.groq.com" in message


def test_llm_429_per_minute_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    _install_llm_transport(
        monkeypatch,
        lambda _request: _groq_429_response(
            "Rate limit reached for model `qwen/qwen3.8-27b` on tokens per minute "
            "(TPM): Limit 8000, Used 7900, Requested 1500. "
            "Please try again in 12s."
        ),
    )
    with pytest.raises(LLMConfigurationError) as error:
        propose_clips([], ProjectConfig(), 60, _groq_llm())
    message = str(error.value)
    assert "por minuto" in message
    assert "unos minutos" in message
    assert "cuota diaria" not in message


def test_llm_request_uses_max_tokens_2000(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode("utf-8")))
        return _choices_response('{"clips": []}')

    _install_llm_transport(monkeypatch, handle)
    propose_clips([], ProjectConfig(), 60, _groq_llm())
    assert len(payloads) == 1
    assert payloads[0]["max_tokens"] == 2000


def test_token_pacer_does_not_wait_with_margin() -> None:
    now = [1000.0]
    pacer = _TokenPacer(
        budget_tokens=7000,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        monotonic=lambda: now[0],
    )
    assert pacer.wait_before_call(3000) == 0.0
    assert pacer.wait_before_call(3000) == 0.0


def test_token_pacer_waits_when_budget_is_consumed() -> None:
    now = [1000.0]
    waits: list[float] = []
    pacer = _TokenPacer(
        budget_tokens=7000,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        monotonic=lambda: now[0],
        on_wait=waits.append,
    )
    assert pacer.wait_before_call(4000) == 0.0
    # 4000 usados + 4000 no caben en 7000: hay que esperar a que caduque
    # la primera entrada (60 s después de registrarse).
    assert pacer.wait_before_call(4000) == 60.0
    assert waits == [60.0]
    # Tras la espera, la ventana se vació y la tercera llamada pasa.
    assert pacer.wait_before_call(3000) == 0.0


def test_token_pacer_waits_only_until_enough_budget_frees() -> None:
    now = [1000.0]
    pacer = _TokenPacer(
        budget_tokens=7000,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        monotonic=lambda: now[0],
    )
    assert pacer.wait_before_call(3000) == 0.0  # t=1000
    now[0] = 1010.0
    assert pacer.wait_before_call(3000) == 0.0  # t=1010, usados 6000
    now[0] = 1020.0
    # 6000 + 3000 no caben; basta con que caduque la primera (t=1060).
    assert pacer.wait_before_call(3000) == 40.0


def test_token_pacer_oversized_single_request_proceeds() -> None:
    now = [1000.0]
    pacer = _TokenPacer(
        budget_tokens=7000,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        monotonic=lambda: now[0],
    )
    # Una sola petición mayor que el presupuesto no se puede dividir:
    # procede sin esperar (el backoff de 429 cubre ese caso).
    assert pacer.wait_before_call(9000) == 0.0


def test_propose_valid_clips_paces_between_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    waits: list[float] = []
    pacer = _TokenPacer(
        budget_tokens=3000,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        monotonic=lambda: now[0],
        on_wait=waits.append,
    )
    calls: list[int] = []

    def propose(*_args: Any, **_kwargs: Any) -> list[ClipProposal]:
        calls.append(1)
        return [
            ClipProposal(
                inicio=10,
                fin=30,
                titulo=f"Ventana {len(calls)}",
                puntuacion=8,
                razon_viral="Una idea completa.",
            )
        ]

    monkeypatch.setattr(pipeline, "propose_clips", propose)
    transcript = [
        TranscriptSegment(
            start=0,
            end=3,
            text="Una idea útil.",
            words=[Word(start=1, end=2, text="idea")],
        )
    ]
    # 400 s = 2 ventanas (0-300, 240-400). Cada ventana estima ~2500
    # tokens; con presupuesto 3000 la segunda debe esperar.
    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(),
        400,
        ResolvedLLM("Groq", "k", "m", "https://llm.example"),
        token_pacer=pacer,
    )
    assert len(calls) == 2
    assert len(proposals) == 2
    assert len(waits) == 1
    assert waits[0] == pytest.approx(60.0, abs=0.1)


# ---------------------------------------------------------------------------
# Ítem 22: subida reanudable
# ---------------------------------------------------------------------------


def _create_staged_upload(client: TestClient, *, file_size: int) -> str:
    created = client.post(
        "/api/staged-media/uploads",
        json={"filename": "source.mp4", "file_size": file_size},
    )
    assert created.status_code == 201
    return created.json()["id"]


def test_upload_in_chunks_then_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        first = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-4/10",
            },
        )
        assert first.status_code == 200
        assert first.json()["status"] == "receiving"
        assert first.json()["bytes_received"] == 5
        second = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"56789",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 5-9/10",
            },
        )
        assert second.status_code == 200
        assert second.json()["status"] == "ready"
        assert second.json()["bytes_received"] == 10
        source = settings.data_dir / "staging" / f"{staged_id}.source"
        assert source.read_bytes() == b"0123456789"


def test_upload_resumes_after_interrupted_chunk(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        # Declara el rango completo pero envía solo 5 bytes (interrupción).
        interrupted = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-9/10",
            },
        )
        assert interrupted.status_code == 400
        staged = client.get(f"/api/staged-media/{staged_id}").json()
        # NO se marca failed ni se borra el .part: se puede reanudar.
        assert staged["status"] == "receiving"
        assert staged["bytes_received"] == 5
        part = settings.data_dir / "staging" / f"{staged_id}.part"
        assert part.read_bytes() == b"01234"
        resumed = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"56789",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 5-9/10",
            },
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "ready"


def test_upload_rejects_wrong_offset(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-4/10",
            },
        )
        wrong = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"789",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 7-9/10",
            },
        )
        assert wrong.status_code == 409
        assert "5" in wrong.json()["detail"]
        staged = client.get(f"/api/staged-media/{staged_id}").json()
        assert staged["status"] == "receiving"
        assert staged["bytes_received"] == 5


def test_upload_rejects_mismatched_total_and_bad_range(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        wrong_total = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-4/20",
            },
        )
        assert wrong_total.status_code == 400
        bad_range = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes foo",
            },
        )
        assert bad_range.status_code == 400
        staged = client.get(f"/api/staged-media/{staged_id}").json()
        assert staged["status"] == "receiving"
        assert staged["bytes_received"] == 0


def test_upload_chunks_via_offset_parameter(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        first = client.post(
            f"/api/staged-media/{staged_id}/content?offset=0",
            content=b"01234",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert first.status_code == 200
        assert first.json()["status"] == "receiving"
        assert first.json()["bytes_received"] == 5
        second = client.post(
            f"/api/staged-media/{staged_id}/content?offset=5",
            content=b"56789",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert second.status_code == 200
        assert second.json()["status"] == "ready"


def test_upload_chunk_exceeding_declared_range_rolls_back(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        staged_id = _create_staged_upload(client, file_size=10)
        overflow = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234567",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-4/10",
            },
        )
        assert overflow.status_code == 413
        staged = client.get(f"/api/staged-media/{staged_id}").json()
        assert staged["status"] == "receiving"
        assert staged["bytes_received"] == 0
        part = settings.data_dir / "staging" / f"{staged_id}.part"
        assert part.stat().st_size == 0
        # El fragmento se puede reintentar correctamente.
        retry = client.post(
            f"/api/staged-media/{staged_id}/content",
            content=b"01234",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": "bytes 0-4/10",
            },
        )
        assert retry.status_code == 200
        assert retry.json()["bytes_received"] == 5
