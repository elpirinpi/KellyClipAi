"""Tests de la Fase 1 (v1.0.11), backend núcleo.

Cubre: no destruir clips al reprocesar (A1), límites de entrada (A2),
reintento LLM por ventana (A3), invalidación de caché de transcripción
(A6), programación desacoplada (A7), migración de tokens tolerante (A10),
cancelación de descargas staged (ítem 12), claim_legacy_data con
llm_credentials (B20), no reescribir el modelo del usuario (B21),
cancelación de procesamiento (B14/B-P7), persistencia de whisper_model
(B2) y compatibilidad con el validador de base_url.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend import database, pipeline
from backend.credential_store import CredentialStoreError
from backend.llm import LLMResponseError, ResolvedLLM
from backend.main import create_app as create_application
from backend.media import MediaProcessingError
from backend.pipeline import ProjectCancelled
from backend.process_queue import (
    ProjectProcessingQueue,
    clear_project_cancellation,
    is_project_cancelled,
    request_project_cancellation,
)
from backend.schemas import (
    ClipProposal,
    ProjectConfig,
    ScheduledPublishing,
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


def _transcript_data() -> list[dict[str, Any]]:
    return [
        TranscriptSegment(
            start=0,
            end=3,
            text="Una idea útil.",
            words=[
                Word(start=1, end=2, text="idea"),
                Word(start=2, end=3, text="útil."),
            ],
        ).model_dump(mode="json")
    ]


def _create_ready_project(
    settings: AppSettings,
    project_id: str,
    *,
    clips: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    config: ProjectConfig | None = None,
) -> Path:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    database.create_project(
        settings.database_path,
        project_id,
        "Proyecto de prueba",
        project_dir / "source",
        (config or ProjectConfig()).model_dump(mode="json"),
        "test-user",
    )
    updates: dict[str, Any] = {}
    if metadata is not None:
        updates["metadata"] = metadata
    if transcript is not None:
        updates["transcript"] = transcript
    if clips is not None:
        updates["clips"] = clips
    if updates:
        database.update_project(settings.database_path, project_id, **updates)
    return project_dir


def _old_clip(project_dir: Path, clip_id: str = "old-clip") -> tuple[dict[str, Any], Path]:
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clips_dir / f"{clip_id}.mp4"
    clip_path.write_bytes(b"old-clip-bytes")
    clip = {
        "id": clip_id,
        "title": "Clip viejo",
        "start": 0.0,
        "end": 2.0,
        "duration": 2.0,
        "score": 8.0,
        "viral_reason": "Viejo pero bueno.",
        "keywords": [],
        "emojis": [],
        "width": 720,
        "height": 1280,
        "resolution": 720,
        "filename": clip_path.name,
        "transcript": [],
        "status": "ready",
    }
    return clip, clip_path


def _mock_ffmpeg_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        pipeline,
        "resolve_llm",
        lambda *_args: ResolvedLLM("Groq", "k", "m", "https://llm.example"),
    )


def _mock_render(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_titles: set[str] | None = None,
    on_render: Any | None = None,
) -> list[str]:
    rendered_titles: list[str] = []

    def fake_render(
        _input_path: Path,
        output_path: Path,
        _subtitle_path: Path,
        proposal: ClipProposal,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, list[dict]]:
        rendered_titles.append(proposal.titulo)
        if fail_titles and proposal.titulo in fail_titles:
            raise MediaProcessingError("Fallo de render de prueba.")
        output_path.write_bytes(b"rendered-clip")
        if on_render is not None:
            on_render(proposal)
        return proposal.fin - proposal.inicio, []

    monkeypatch.setattr(pipeline, "render_clip", fake_render)
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda *_args, **_kwargs: {"width": 720, "height": 1280},
    )
    monkeypatch.setattr(
        pipeline.database, "resolve_subtitle_template", lambda *_args: "Default"
    )
    return rendered_titles


def _proposal(title: str, inicio: float = 0, fin: float = 20) -> ClipProposal:
    return ClipProposal(
        inicio=inicio,
        fin=fin,
        titulo=title,
        puntuacion=9,
        razon_viral="Una idea completa.",
    )


# ---------------------------------------------------------------------------
# Ítem 1: no destruir clips al reprocesar (A1/B-P1)
# ---------------------------------------------------------------------------


def test_queue_project_preserves_existing_clips(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = _create_ready_project(settings, project_id)
    old_clip, _ = _old_clip(project_dir)
    database.update_project(settings.database_path, project_id, clips=[old_clip])
    config = ProjectConfig().model_dump(mode="json")

    assert database.queue_project(settings.database_path, project_id, config)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["clips"] == [old_clip]
    assert stored["status"] == "queued"


def test_failed_reprocess_keeps_previous_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    old_clip, old_path = _old_clip(project_dir)
    database.update_project(settings.database_path, project_id, clips=[old_clip])
    _mock_ffmpeg_and_llm(monkeypatch)

    def failing_propose(*_args: object, **_kwargs: object) -> list[ClipProposal]:
        raise MediaProcessingError("El proveedor LLM no responde.")

    monkeypatch.setattr(pipeline, "_propose_valid_clips", failing_propose)

    config = ProjectConfig().model_dump(mode="json")
    assert database.queue_project(settings.database_path, project_id, config)
    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["clips"] == [old_clip]
    assert old_path.is_file()
    assert old_path.read_bytes() == b"old-clip-bytes"
    assert not (project_dir / "clips.new").exists()
    assert not (project_dir / "clips.old").exists()


def test_all_clips_failing_keeps_previous_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    old_clip, old_path = _old_clip(project_dir)
    database.update_project(settings.database_path, project_id, clips=[old_clip])
    _mock_ffmpeg_and_llm(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_propose_valid_clips", lambda *_a, **_k: [_proposal("Nuevo")]
    )
    _mock_render(monkeypatch, fail_titles={"Nuevo"})

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert "Se conservaron los clips anteriores" in (stored["error"] or "")
    assert stored["clips"] == [old_clip]
    assert old_path.is_file()
    assert not (project_dir / "clips.new").exists()


def test_successful_reprocess_swaps_old_clips_for_new_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    old_clip, old_path = _old_clip(project_dir)
    database.update_project(settings.database_path, project_id, clips=[old_clip])
    _mock_ffmpeg_and_llm(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_propose_valid_clips", lambda *_a, **_k: [_proposal("Nuevo")]
    )
    _mock_render(monkeypatch)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert len(stored["clips"]) == 1
    assert stored["clips"][0]["title"] == "Nuevo"
    assert stored["clips"][0]["status"] == "ready"
    assert not old_path.exists()
    new_path = project_dir / "clips" / stored["clips"][0]["filename"]
    assert new_path.is_file()
    assert new_path.read_bytes() == b"rendered-clip"
    assert not (project_dir / "clips.new").exists()
    assert not (project_dir / "clips.old").exists()


# ---------------------------------------------------------------------------
# Ítem 5: límites de entrada (A2/B-P2)
# ---------------------------------------------------------------------------


def test_upload_rejects_files_over_max_size_with_413(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    # El límite real es 4 GiB; se baja solo para este test.
    monkeypatch.setattr(backend_main, "MAX_UPLOAD_BYTES", 10)
    config = ProjectConfig().model_dump(mode="json")

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/projects",
            files={"file": ("source.mp4", b"x" * 11, "video/mp4")},
            data={"config": json.dumps(config)},
        )
        projects = database.list_projects(settings.database_path)

    assert response.status_code == 413
    assert "máximo" in response.json()["detail"]
    assert projects == []


def test_process_rejects_video_over_max_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 7200},
        transcript=_transcript_data(),
    )
    _mock_ffmpeg_and_llm(monkeypatch)
    monkeypatch.setattr(pipeline, "MAX_VIDEO_DURATION_SECONDS", 3600)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert "supera el máximo" in (stored["error"] or "")


def test_process_fails_fast_when_disk_is_almost_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    # Disco grande (500 GB) con solo 100 bytes libres: el mínimo es 1 GB.
    usage = namedtuple("usage", "total used free")(500 * 10**9, 500 * 10**9 - 100, 100)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: usage)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert "menos de 1 GB" in (stored["error"] or "")


def test_disk_check_scales_down_on_small_filesystems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # tmpfs de 512 MB con 368 MB libres: el mínimo baja al 10 % del total
    # (51 MB) y el procesamiento no se bloquea.
    usage = namedtuple("usage", "total used free")(
        512 * 1024 * 1024, 144 * 1024 * 1024, 368 * 1024 * 1024
    )
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: usage)
    pipeline._check_free_disk_space(tmp_path)

    # El mismo sistema casi lleno sí falla, con el umbral efectivo.
    full_usage = namedtuple("usage", "total used free")(
        512 * 1024 * 1024, 500 * 1024 * 1024, 12 * 1024 * 1024
    )
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda _path: full_usage)
    with pytest.raises(MediaProcessingError, match="menos de 51 MB"):
        pipeline._check_free_disk_space(tmp_path)


# ---------------------------------------------------------------------------
# Ítem 6: reintento LLM por ventana (A3)
# ---------------------------------------------------------------------------


def test_llm_parse_error_in_one_window_does_not_abort_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = [
        TranscriptSegment(
            start=0,
            end=700,
            text="Transcripción larga.",
            words=[Word(start=10, end=12, text="Momento")],
        )
    ]
    calls: list[int] = []

    def propose(
        _window_transcript: list[TranscriptSegment],
        _project: ProjectConfig,
        _window_duration: float,
        _llm: ResolvedLLM,
        validation_feedback: list[str] | None = None,
    ) -> list[ClipProposal]:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise LLMResponseError("El modelo devolvió un JSON truncado.")
        return [_proposal(f"Ventana {len(calls)}", inicio=10, fin=30)]

    monkeypatch.setattr(pipeline, "propose_clips", propose)

    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(),
        700,
        ResolvedLLM("Groq", "k", "m", "https://llm.example"),
    )

    # 3 ventanas (0-300, 240-540, 480-700); la primera falla y las otras
    # dos aportan su clip.
    assert len(calls) == 3
    assert [(p.inicio, p.fin) for p in proposals] == [(250, 270), (490, 510)]


def test_all_windows_failing_still_fails_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = [
        TranscriptSegment(
            start=0,
            end=700,
            text="Transcripción larga.",
            words=[Word(start=10, end=12, text="Momento")],
        )
    ]

    def propose(*_args: object, **_kwargs: object) -> list[ClipProposal]:
        raise LLMResponseError("El modelo devolvió un JSON truncado.")

    monkeypatch.setattr(pipeline, "propose_clips", propose)

    with pytest.raises(
        MediaProcessingError, match="No se encontraron propuestas válidas"
    ):
        pipeline._propose_valid_clips(
            transcript,
            ProjectConfig(),
            700,
            ResolvedLLM("Groq", "k", "m", "https://llm.example"),
        )


# ---------------------------------------------------------------------------
# Ítem 7: invalidar caché de transcripción (A6/B-P4)
# ---------------------------------------------------------------------------


def test_transcript_cache_is_reused_with_same_whisper_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={
            "duration": 100,
            "whisper_model": "small",
            "whisper_language": "auto",
        },
        transcript=_transcript_data(),
    )
    _mock_ffmpeg_and_llm(monkeypatch)

    def transcribe_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("No debería volver a transcribir.")

    monkeypatch.setattr(pipeline, "transcribe", transcribe_must_not_run)
    monkeypatch.setattr(pipeline, "_propose_valid_clips", lambda *_a, **_k: [])

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"


def test_transcript_cache_is_invalidated_when_whisper_model_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={
            "duration": 100,
            "whisper_model": "small",
            "whisper_language": "auto",
        },
        transcript=_transcript_data(),
        config=ProjectConfig(whisper_model="medium"),
    )
    _mock_ffmpeg_and_llm(monkeypatch)
    transcribe_calls: list[str] = []

    def fake_transcribe(
        _input_path: Path, model: str, *_args: object, **_kwargs: object
    ) -> tuple[list[TranscriptSegment], str]:
        transcribe_calls.append(model)
        return (
            [
                TranscriptSegment(
                    start=0,
                    end=3,
                    text="Nueva transcripción.",
                    words=[Word(start=1, end=2, text="Nueva")],
                )
            ],
            "es",
        )

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda *_a, **_k: {"duration": 100, "width": 1920, "height": 1080},
    )
    monkeypatch.setattr(pipeline, "create_thumbnail", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_propose_valid_clips", lambda *_a, **_k: [])

    pipeline.process_project(project_id, None, settings)

    assert transcribe_calls == ["medium"]
    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["metadata"]["whisper_model"] == "medium"
    assert stored["metadata"]["whisper_language"] == "auto"


def test_transcript_cache_is_invalidated_when_whisper_language_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={
            "duration": 100,
            "whisper_model": "small",
            "whisper_language": "es",
        },
        transcript=_transcript_data(),
        config=ProjectConfig(whisper_language="en"),
    )
    _mock_ffmpeg_and_llm(monkeypatch)
    transcribe_calls: list[str] = []

    def fake_transcribe(
        _input_path: Path, model: str, *_args: object, **_kwargs: object
    ) -> tuple[list[TranscriptSegment], str]:
        transcribe_calls.append(model)
        return (
            [
                TranscriptSegment(
                    start=0,
                    end=3,
                    text="New transcript.",
                    words=[Word(start=1, end=2, text="New")],
                )
            ],
            "en",
        )

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        pipeline,
        "probe_media",
        lambda *_a, **_k: {"duration": 100, "width": 1920, "height": 1080},
    )
    monkeypatch.setattr(pipeline, "create_thumbnail", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_propose_valid_clips", lambda *_a, **_k: [])

    pipeline.process_project(project_id, None, settings)

    assert transcribe_calls == ["small"]
    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["metadata"]["whisper_language"] == "en"


# ---------------------------------------------------------------------------
# Ítem 8: desacoplar la programación (A7/B-P5)
# ---------------------------------------------------------------------------


def test_schedule_failure_completes_project_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    config = ProjectConfig(
        schedule=ScheduledPublishing(
            scheduled_at=datetime.now(UTC) + timedelta(hours=2),
            platform="youtube",
            visibility="private",
        )
    )
    _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
        config=config,
    )
    _mock_ffmpeg_and_llm(monkeypatch)
    monkeypatch.setattr(
        pipeline, "_propose_valid_clips", lambda *_a, **_k: [_proposal("Nuevo")]
    )
    _mock_render(monkeypatch)

    def failing_enqueue(*_args: object, **_kwargs: object) -> list[dict]:
        raise RuntimeError("La base de datos de tareas no responde.")

    monkeypatch.setattr(
        pipeline.database, "enqueue_schedule_jobs", failing_enqueue
    )

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["clips"][0]["status"] == "ready"
    assert "No se pudieron programar" in stored["metadata"]["schedule_warning"]
    events = database.events_after(settings.database_path, project_id, 0)
    assert any(
        "Advertencia" in str(event["data"].get("message", "")) for event in events
    )


# ---------------------------------------------------------------------------
# Ítem 10: migración de tokens tolerante a fallos (A10/B-S2)
# ---------------------------------------------------------------------------


def test_oauth_migration_survives_corrupt_json_and_dpapi_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)

    def failing_encrypt(_token: str) -> str:
        raise CredentialStoreError("DPAPI simulado: perfil dañado.")

    monkeypatch.setattr(database, "encrypt_oauth_token", failing_encrypt)
    with database.connection(database_path) as conn:
        conn.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, metadata_json, updated_at
            ) VALUES ('legacy', 'youtube', 'a1', 'Canal', 'plain-token', NULL, '{}', ?)
            """,
            (database.utc_now(),),
        )
        conn.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, metadata_json, updated_at
            ) VALUES ('legacy', 'tiktok', 'a2', 'Cuenta', 'dpapi:v1:AAA', NULL, 'no-json{{{', ?)
            """,
            (database.utc_now(),),
        )
        conn.execute(
            """
            INSERT INTO connected_accounts (
                owner_id, provider, account_id, display_name, access_token,
                refresh_token, metadata_json, updated_at
            ) VALUES ('legacy', 'instagram', 'a3', 'Cuenta', 'dpapi:v1:BBB', NULL, '{}', ?)
            """,
            (database.utc_now(),),
        )
        conn.execute("PRAGMA user_version = 0")

    # No debe tumbar initialize() ni el arranque de la app.
    database.initialize(database_path)

    with database.connection(database_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        rows = {
            row["provider"]: dict(row)
            for row in conn.execute("SELECT * FROM connected_accounts").fetchall()
        }

    assert version == 3
    youtube = rows["youtube"]
    assert youtube["access_token"] == ""
    assert youtube["refresh_token"] is None
    youtube_metadata = json.loads(youtube["metadata_json"])
    assert youtube_metadata["needs_reconnect"] is True
    assert "DPAPI simulado" in youtube_metadata["migration_error"]
    # JSON corrupto con token ya cifrado: sobrevive con el token intacto.
    assert rows["tiktok"]["access_token"] == "dpapi:v1:AAA"
    assert json.loads(rows["tiktok"]["metadata_json"]) == {}
    # Cuenta sana: intacta.
    assert rows["instagram"]["access_token"] == "dpapi:v1:BBB"


# ---------------------------------------------------------------------------
# Ítem 12 (backend): cancelar una descarga staged en curso
# ---------------------------------------------------------------------------


def test_deleting_a_downloading_staged_media_returns_204(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    staged_id = str(uuid.uuid4())

    with TestClient(create_app(settings)) as client:
        # Se crea dentro del contexto: el arranque marca como fallidas las
        # descargas interrumpidas de arranques anteriores.
        database.create_staged_media(
            settings.database_path,
            staged_id,
            "url",
            "downloading",
            "Video de enlace",
            owner_id="test-user",
        )
        response = client.delete(f"/api/staged-media/{staged_id}")

    assert response.status_code == 204
    assert database.get_staged_media(settings.database_path, staged_id) is None


def test_staged_download_task_cleans_up_when_row_is_deleted_mid_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    staged_id = str(uuid.uuid4())
    database.create_staged_media(
        settings.database_path,
        staged_id,
        "url",
        "downloading",
        "Video de enlace",
    )

    def fake_download(url: str, target_dir: Path, progress_hook=None):
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "video.mp4"
        target.write_bytes(b"video-bytes")
        # El usuario cancela (borra la fila) mientras descargaba.
        with database.connection(settings.database_path) as conn:
            conn.execute(
                "DELETE FROM staged_media WHERE id = ?", (staged_id,)
            )
        return target, "Video de enlace"

    monkeypatch.setattr(backend_main, "_download_video", fake_download)

    backend_main._download_staged_media(
        settings, staged_id, "https://youtube.com/watch?v=x"
    )

    staging_root = settings.data_dir / "staging"
    assert not (staging_root / staged_id).exists()
    assert not (staging_root / f"{staged_id}.source").exists()


def test_staged_download_progress_hook_aborts_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    staged_id = str(uuid.uuid4())
    database.create_staged_media(
        settings.database_path,
        staged_id,
        "url",
        "downloading",
        "Video de enlace",
    )

    def fake_download(url: str, target_dir: Path, progress_hook=None):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "video.mp4.part").write_bytes(b"partial-bytes")
        # El usuario cancela antes del siguiente reporte de progreso.
        with database.connection(settings.database_path) as conn:
            conn.execute(
                "DELETE FROM staged_media WHERE id = ?", (staged_id,)
            )
        assert progress_hook is not None
        progress_hook(
            {
                "status": "downloading",
                "downloaded_bytes": 5 * 1024 * 1024,
                "total_bytes": 10 * 1024 * 1024,
                "speed": 1024,
                "eta": 5,
            }
        )
        raise AssertionError("El hook debería haber abortado la descarga.")

    monkeypatch.setattr(backend_main, "_download_video", fake_download)

    backend_main._download_staged_media(
        settings, staged_id, "https://youtube.com/watch?v=x"
    )

    staging_root = settings.data_dir / "staging"
    assert not (staging_root / staged_id).exists()
    assert not (staging_root / f"{staged_id}.source").exists()


# ---------------------------------------------------------------------------
# Ítem 15: claim_legacy_data incluye llm_credentials (B20/B-S4)
# ---------------------------------------------------------------------------


def test_claim_legacy_data_migrates_llm_credentials(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path,
        "legacy",
        "Groq",
        b"encrypted-key",
        "qwen/qwen3.8-27b",
        "",
    )

    database.claim_legacy_data(database_path, "user-1")

    claimed = database.get_llm_credentials(database_path, "user-1")
    assert claimed is not None
    assert claimed["encrypted_api_key"] == b"encrypted-key"
    assert claimed["model"] == "qwen/qwen3.8-27b"
    assert database.get_llm_credentials(database_path, "legacy") is None


def test_claim_legacy_data_keeps_existing_credentials_and_drops_legacy(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path, "user-1", "Groq", b"user-key", "model-a", ""
    )
    database.save_llm_credentials(
        database_path, "legacy", "OpenAI", b"legacy-key", "model-b", ""
    )

    database.claim_legacy_data(database_path, "user-1")

    kept = database.get_llm_credentials(database_path, "user-1")
    assert kept is not None
    assert kept["encrypted_api_key"] == b"user-key"
    assert kept["model"] == "model-a"
    assert database.get_llm_credentials(database_path, "legacy") is None


# ---------------------------------------------------------------------------
# Ítem 16: no reescribir el modelo del usuario en migraciones (B21/B-S5)
# ---------------------------------------------------------------------------


def test_migrations_respect_user_customized_model(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path,
        "groq-owner",
        "Groq",
        b"encrypted-key",
        "openai/gpt-oss-120b",
        "",
        model_customized=True,
    )

    with database.connection(database_path) as conn:
        conn.execute("PRAGMA user_version = 2")

    database.initialize(database_path)

    stored = database.get_llm_credentials(database_path, "groq-owner")
    assert stored is not None
    assert stored["model"] == "openai/gpt-oss-120b"
    assert stored["model_customized"] is True


def test_migrations_still_rewrite_non_customized_model(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    database.save_llm_credentials(
        database_path,
        "groq-owner",
        "Groq",
        b"encrypted-key",
        "openai/gpt-oss-120b",
        "",
    )

    with database.connection(database_path) as conn:
        conn.execute("PRAGMA user_version = 2")

    database.initialize(database_path)

    stored = database.get_llm_credentials(database_path, "groq-owner")
    assert stored is not None
    assert stored["model"] == "qwen/qwen3.8-27b"
    assert stored["model_customized"] is False


# ---------------------------------------------------------------------------
# Ítem 17: persistencia de whisper_model (B2)
# ---------------------------------------------------------------------------


def test_llm_settings_persist_whisper_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    monkeypatch.setattr(
        backend_main, "encrypt_api_key", lambda key: b"fake:" + key.encode()
    )

    with TestClient(create_app(settings)) as client:
        before = client.get("/api/settings")
        assert before.json()["whisper_model"] == "small"

        saved = client.put(
            "/api/settings/llm-key",
            json={
                "provider": "Groq",
                "api_key": "test-key",
                "model": "qwen/qwen3.8-27b",
                "base_url": "",
                "whisper_model": "medium",
            },
        )
        assert saved.status_code == 200
        after = client.get("/api/settings")
        assert after.json()["whisper_model"] == "medium"

        # Un PUT sin whisper_model no cambia nada.
        saved_again = client.put(
            "/api/settings/llm-key",
            json={
                "provider": "Groq",
                "api_key": "",
                "model": "qwen/qwen3.8-27b",
                "base_url": "",
            },
        )
        assert saved_again.status_code == 200
        assert client.get("/api/settings").json()["whisper_model"] == "medium"

    stored = database.get_llm_credentials(settings.database_path, "test-user")
    assert stored is not None
    assert stored["whisper_model"] == "medium"
    assert stored["model_customized"] is True


# ---------------------------------------------------------------------------
# Compatibilidad: ValidationError de base_url en /process -> 422, no 500
# ---------------------------------------------------------------------------


def test_process_with_invalid_env_base_url_returns_422(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path / "data", llm_base_url="http://192.168.1.10"
    )
    config = ProjectConfig().model_dump(mode="json")

    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/projects",
            files={"file": ("source.mp4", b"video-bytes", "video/mp4")},
            data={"config": json.dumps(config)},
        )
        assert created.status_code == 201
        project_id = created.json()["id"]
        response = client.post(f"/api/projects/{project_id}/process", json={})
        project = database.get_project(settings.database_path, project_id)

    assert response.status_code == 422
    assert "no es válida" in response.json()["detail"]
    assert project is not None
    assert project["status"] == "ready"


# ---------------------------------------------------------------------------
# Ítem 22: cancelación de procesamiento (B14/B-P7)
# ---------------------------------------------------------------------------


def test_cancel_queued_project_returns_updated_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(settings, project_id)

    with TestClient(create_app(settings)) as client:
        database.update_project(
            settings.database_path, project_id, status="queued"
        )
        response = client.post(f"/api/projects/{project_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["id"] == project_id
    assert database.get_project(settings.database_path, project_id)["status"] == "cancelled"


def test_cancel_processing_project_returns_updated_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(settings, project_id)

    with TestClient(create_app(settings)) as client:
        database.update_project(
            settings.database_path, project_id, status="processing"
        )
        response = client.post(f"/api/projects/{project_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_cancel_returns_409_when_project_is_not_processing(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(settings, project_id)

    with TestClient(create_app(settings)) as client:
        ready_response = client.post(f"/api/projects/{project_id}/cancel")
        database.update_project(
            settings.database_path, project_id, status="completed"
        )
        completed_response = client.post(f"/api/projects/{project_id}/cancel")

    assert ready_response.status_code == 409
    assert completed_response.status_code == 409


def test_cancel_returns_404_for_unknown_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")

    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/projects/{uuid.uuid4()}/cancel")

    assert response.status_code == 404


def test_cancelled_project_can_be_requeued(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(settings, project_id)
    config = ProjectConfig().model_dump(mode="json")
    database.update_project(settings.database_path, project_id, status="cancelled")

    assert database.queue_project(settings.database_path, project_id, config)
    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "queued"


def test_pipeline_stops_before_heavy_work_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    _mock_ffmpeg_and_llm(monkeypatch)

    def must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("El trabajo pesado no debería empezar.")

    monkeypatch.setattr(pipeline, "_propose_valid_clips", must_not_run)
    monkeypatch.setattr(pipeline, "transcribe", must_not_run)
    request_project_cancellation(project_id)
    try:
        pipeline.process_project(project_id, None, settings)
    finally:
        clear_project_cancellation(project_id)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "cancelled"


def test_pipeline_cancels_between_renders_and_keeps_rendered_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    project_dir = _create_ready_project(
        settings,
        project_id,
        metadata={"duration": 100},
        transcript=_transcript_data(),
    )
    old_clip, old_path = _old_clip(project_dir)
    database.update_project(settings.database_path, project_id, clips=[old_clip])
    _mock_ffmpeg_and_llm(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_propose_valid_clips",
        lambda *_a, **_k: [_proposal("Uno", 0, 20), _proposal("Dos", 30, 50)],
    )

    def cancel_after_first_render(_proposal: ClipProposal) -> None:
        request_project_cancellation(project_id)

    rendered = _mock_render(monkeypatch, on_render=cancel_after_first_render)
    try:
        pipeline.process_project(project_id, None, settings)
    finally:
        clear_project_cancellation(project_id)

    assert rendered == ["Uno"]
    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "cancelled"
    assert len(stored["clips"]) == 1
    assert stored["clips"][0]["title"] == "Uno"
    assert stored["clips"][0]["status"] == "ready"
    assert not old_path.exists()
    new_path = project_dir / "clips" / stored["clips"][0]["filename"]
    assert new_path.is_file()
    assert not (project_dir / "clips.new").exists()


def test_queue_discards_and_skips_cancelled_jobs() -> None:
    async def run() -> list[str]:
        calls: list[str] = []
        release = asyncio.Event()
        started = asyncio.Event()

        async def processor(
            project_id: str,
            _llm_config: object,
            _retry_failed_only: bool,
        ) -> None:
            calls.append(project_id)
            if project_id == "blocker":
                started.set()
                await release.wait()

        queue = ProjectProcessingQueue(1, processor)
        await queue.start()
        # Un trabajo bloqueante mantiene ocupado al único worker para que
        # los siguientes queden pendientes en la cola de forma determinista.
        await queue.submit("blocker", None)
        await asyncio.wait_for(started.wait(), timeout=5)
        await queue.submit("job-a", None)
        await queue.submit("job-b", None)
        assert queue.discard_queued("job-a") == 1
        assert queue.discard_queued("missing") == 0
        request_project_cancellation("job-b")
        release.set()
        await asyncio.wait_for(queue.join(), timeout=5)
        await queue.stop()
        return calls

    assert asyncio.run(run()) == ["blocker"]
    assert not is_project_cancelled("job-b")


def test_submit_clears_stale_cancellation_flag() -> None:
    async def run() -> None:
        async def processor(*_args: object) -> None:
            return None

        queue = ProjectProcessingQueue(1, processor)
        await queue.start()
        request_project_cancellation("job-c")
        assert is_project_cancelled("job-c")
        await queue.submit("job-c", None)
        assert not is_project_cancelled("job-c")
        await asyncio.wait_for(queue.join(), timeout=5)
        await queue.stop()

    asyncio.run(run())


def test_delete_project_refuses_active_processing_atomically(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "data" / "app.sqlite3"
    database.initialize(database_path)
    project_id = str(uuid.uuid4())
    database.create_project(
        database_path,
        project_id,
        "En curso",
        tmp_path / "source.mp4",
        ProjectConfig().model_dump(mode="json"),
        "test-user",
    )
    database.update_project(database_path, project_id, status="processing")

    with pytest.raises(database.ProjectBusyError):
        database.delete_project(database_path, project_id, "test-user")
    assert database.get_project(database_path, project_id) is not None

    database.update_project(database_path, project_id, status="cancelled")
    assert database.delete_project(database_path, project_id, "test-user") == []
    assert database.get_project(database_path, project_id) is None


def test_project_cancelled_exception_carries_rendered_clips() -> None:
    clips = [{"id": "clip-1", "status": "ready"}]
    error = ProjectCancelled(clips)

    assert error.clips == clips
    assert isinstance(error, RuntimeError)


def test_delete_project_still_rejects_active_processing_via_api(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    project_id = str(uuid.uuid4())
    _create_ready_project(settings, project_id)

    with TestClient(create_app(settings)) as client:
        database.update_project(
            settings.database_path, project_id, status="processing"
        )
        response = client.delete(f"/api/projects/{project_id}")

    assert response.status_code == 409
    assert database.get_project(settings.database_path, project_id) is not None
