from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import database
from backend.llm import LLMConfigurationError, resolve_llm
from backend.main import _validate_download_url, create_app
from backend.media import _subtract_intervals, create_thumbnail, probe_media, render_clip
from backend.scheduler import process_schedule_job
from backend.schemas import ClipProposal, LLMConfig, ProjectConfig
from backend.settings import AppSettings


def test_faster_whisper_decodes_audio_with_installed_pyav(
    video_with_audio: Path,
) -> None:
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(video_with_audio))

    assert audio.size > 0


def test_subtract_intervals_cuts_removed_phrases_from_video_ranges() -> None:
    assert _subtract_intervals([(0.0, 3.0), (4.0, 8.0)], [(1.0, 2.0), (5.0, 6.0)]) == [
        (0.0, 1.0),
        (2.0, 3.0),
        (4.0, 5.0),
        (6.0, 8.0),
    ]


@pytest.fixture(scope="session")
def video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg no está disponible.")
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=25:d=2",
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
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def test_ffprobe_reads_local_video_and_creates_thumbnail(
    video_with_audio: Path, tmp_path: Path
) -> None:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    assert ffprobe and ffmpeg

    metadata = probe_media(video_with_audio, ffprobe)
    thumbnail = tmp_path / "thumbnail.jpg"
    create_thumbnail(video_with_audio, thumbnail, float(metadata["duration"]), ffmpeg)

    assert 1.8 <= float(metadata["duration"]) <= 2.2
    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert metadata["has_audio"] is True
    assert thumbnail.is_file()
    assert thumbnail.stat().st_size > 0


def _create_completed_project_with_clip(
    client: TestClient,
    settings: AppSettings,
    source: Path,
) -> tuple[str, str, Path]:
    config = {
        "ratio": "16:9",
        "duration": "Cualquiera",
        "subtitle_template": "Default",
        "options": {
            "emojis": True,
            "keywords": True,
            "broll": False,
            "silence": False,
            "censor": False,
        },
        "moment": "",
        "whisper_model": "small",
    }
    response = client.post(
        "/api/projects",
        files={"file": ("source.mp4", source.read_bytes(), "video/mp4")},
        data={"config": json.dumps(config)},
    )
    assert response.status_code == 201
    project_id = response.json()["id"]
    clip_id = str(uuid.uuid4())
    clip_path = settings.data_dir / "projects" / project_id / "clips" / f"{clip_id}.mp4"
    clip_path.parent.mkdir(parents=True)
    shutil.copyfile(source, clip_path)
    clip = {
        "id": clip_id,
        "title": "Clip real de prueba",
        "start": 0.0,
        "end": 2.0,
        "duration": 2.0,
        "score": 7.5,
        "viral_reason": "Prueba de render local.",
        "keywords": [],
        "emojis": [],
        "width": 320,
        "height": 240,
        "resolution": 720,
        "audio_enhanced": False,
        "filename": clip_path.name,
        "transcript": [],
    }
    database.update_project(
        settings.database_path,
        project_id,
        status="completed",
        step="finalize",
        progress=100,
        config=config,
        clips=[clip],
    )
    return project_id, clip_id, clip_path


def test_archive_and_public_share_serve_selected_real_clip(
    video_with_audio: Path, tmp_path: Path
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings)) as client:
        project_id, clip_id, clip_path = _create_completed_project_with_clip(
            client, settings, video_with_audio
        )
        stored = database.get_project(settings.database_path, project_id)
        assert stored is not None
        second_id = str(uuid.uuid4())
        second_path = clip_path.with_name(f"{second_id}.mp4")
        shutil.copyfile(clip_path, second_path)
        second_clip = {**stored["clips"][0], "id": second_id, "filename": second_path.name}
        database.update_project(
            settings.database_path,
            project_id,
            clips=[*stored["clips"], second_clip],
        )
        archive = client.post(
            f"/api/projects/{project_id}/clips/archive",
            json={"clip_ids": [clip_id, second_id]},
        )
        assert archive.status_code == 200
        with ZipFile(BytesIO(archive.content)) as zipped:
            assert zipped.namelist() == [f"{clip_id}.mp4", f"{second_id}.mp4"]
            assert zipped.read(f"{clip_id}.mp4") == clip_path.read_bytes()
            assert zipped.read(f"{second_id}.mp4") == second_path.read_bytes()

        created = client.post(f"/api/projects/{project_id}/clips/{clip_id}/share")
        repeated = client.post(f"/api/projects/{project_id}/clips/{clip_id}/share")
        assert created.status_code == 201
        assert repeated.json() == created.json()
        token = created.json()["share_url"].rsplit("/", 1)[-1]
        details = client.get(f"/api/shares/{token}")
        file_response = client.get(f"/api/shares/{token}/file")
        assert details.status_code == 200
        assert details.json()["title"] == "Clip real de prueba"
        assert file_response.status_code == 200
        assert file_response.content == clip_path.read_bytes()


def test_clip_can_be_rendered_at_1080p_and_enhanced(
    video_with_audio: Path, tmp_path: Path
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings)) as client:
        project_id, clip_id, clip_path = _create_completed_project_with_clip(
            client, settings, video_with_audio
        )
        quality = client.post(
            f"/api/projects/{project_id}/clips/{clip_id}/quality",
            json={"resolution": 1080},
        )
        assert quality.status_code == 200, quality.text
        assert quality.json()["resolution"] == 1080
        metadata = probe_media(clip_path, shutil.which("ffprobe") or "ffprobe")
        assert metadata["width"] == 1920
        assert metadata["height"] == 1080
        assert metadata["has_audio"] is True

        improved = client.post(f"/api/projects/{project_id}/clips/{clip_id}/improve")
        assert improved.status_code == 200, improved.text
        assert improved.json()["resolution"] == 1080
        assert improved.json()["audio_enhanced"] is True
        stored = database.get_project(settings.database_path, project_id)
        assert stored is not None
        assert stored["clips"][0]["audio_enhanced"] is True
        assert probe_media(clip_path, shutil.which("ffprobe") or "ffprobe")["width"] == 1920


def test_custom_templates_are_persistent_and_resolve_to_saved_style(
    video_with_audio: Path, tmp_path: Path
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/templates",
            json={"name": "Mi estilo cálido", "base_template": "Modern"},
        )
        assert created.status_code == 201
        assert client.get("/api/templates").json() == [created.json()]
        assert database.resolve_subtitle_template(
            settings.database_path, "Mi estilo cálido"
        ) == "Modern"
        assert client.post(
            "/api/templates",
            json={"name": "Mi estilo cálido", "base_template": "Business"},
        ).status_code == 409
        assert client.post(
            "/api/templates",
            json={"name": "Default", "base_template": "Modern"},
        ).status_code == 422


def test_upload_persists_config_and_blocks_processing_without_llm_key(
    video_with_audio: Path, tmp_path: Path
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    config = {
        "ratio": "9:16",
        "duration": "Cualquiera",
        "subtitle_template": "Default",
        "options": {
            "emojis": True,
            "keywords": True,
            "broll": False,
            "silence": False,
            "censor": False,
        },
        "moment": "un momento concreto",
        "whisper_model": "small",
        "schedule": None,
    }

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/projects",
            files={"file": ("source.mp4", video_with_audio.read_bytes(), "video/mp4")},
            data={"config": json.dumps(config)},
        )
        assert response.status_code == 201
        project = response.json()
        project_id = project["id"]
        assert project["status"] == "ready"
        assert project["config"] == config
        assert (settings.data_dir / "projects" / project_id / "source").is_file()

        blocked = client.post(
            f"/api/projects/{project_id}/process",
            json={"llm": {"provider": "Groq", "api_key": ""}, "config": config},
        )
        assert blocked.status_code == 422
        assert "Falta la clave" in blocked.json()["detail"]

        restored = client.get(f"/api/projects/{project_id}")
        assert restored.status_code == 200
        assert restored.json()["status"] == "ready"
        assert restored.json()["transcript"] is None
        assert restored.json()["clips"] == []

        assert database.queue_project(settings.database_path, project_id, config)
        database.update_project(settings.database_path, project_id, status="failed")
        database.publish_event(
            settings.database_path,
            project_id,
            "failed",
            {"status": "failed", "message": "Error de prueba"},
        )
        assert database.queue_project(settings.database_path, project_id, config)
        current_events = database.events_after(settings.database_path, project_id, 0)
        assert all(event["data"]["status"] != "failed" for event in current_events)

        database.update_project(
            settings.database_path,
            project_id,
            status="completed",
            step="finalize",
            progress=100,
        )
        database.publish_event(
            settings.database_path,
            project_id,
            "progress",
            {"status": "completed", "step": "finalize", "progress": 100, "message": "Listo."},
        )
        stream = client.get(f"/api/projects/{project_id}/events")
        assert stream.status_code == 200
        assert "event: progress" in stream.text
        assert '"progress": 100' in stream.text


def test_url_import_rejects_non_https_and_unapproved_hosts() -> None:
    with pytest.raises(HTTPException, match="HTTPS"):
        _validate_download_url("http://youtube.com/watch?v=owned")
    with pytest.raises(HTTPException, match="no está permitida"):
        _validate_download_url("https://example.com/video")
    with pytest.raises(HTTPException, match="HTTPS"):
        _validate_download_url("https://youtube.com:invalid/watch?v=owned")


def test_environment_key_is_not_reused_with_another_provider() -> None:
    settings = AppSettings(
        data_dir=Path("data"),
        cors_origins=(),
        llm_provider="OpenAI",
        llm_api_key="configured-for-openai",
        llm_model="gpt-4o-mini",
        llm_base_url="",
        whisper_model="small",
    )
    assert resolve_llm(None, settings).provider == "OpenAI"
    with pytest.raises(LLMConfigurationError, match="Falta la clave"):
        resolve_llm(LLMConfig(provider="Anthropic"), settings)


def test_health_endpoint_uses_real_app(tmp_path: Path) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=("http://localhost:5173",),
        llm_provider="Groq",
        llm_api_key="",
        llm_model="",
        llm_base_url="",
        whisper_model="small",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
        settings_response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert settings_response.status_code == 200
    assert settings_response.json()["has_llm_key"] is False
    assert "api_key" not in settings_response.json()


def test_integrations_report_missing_credentials_without_exposing_secrets(
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
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/integrations")
        status = response.json()

    assert response.status_code == 200
    assert status["youtube"]["configured"] is False
    assert status["youtube"]["missing"] == ["YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"]
    assert status["zoom"]["connected"] is False
    assert "access_token" not in response.text


def test_oauth_state_is_consumed_once(tmp_path: Path) -> None:
    path = tmp_path / "data" / "kellyclipai.sqlite3"
    database.initialize(path)
    state = "one-time-state"
    database.save_oauth_state(
        path,
        state,
        "youtube",
        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )

    assert database.consume_oauth_state(path, state, database.utc_now()) == "youtube"
    assert database.consume_oauth_state(path, state, database.utc_now()) is None


def test_scheduled_config_with_timezone_is_persisted(
    video_with_audio: Path, tmp_path: Path
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
    scheduled_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    config = {
        "ratio": "9:16",
        "duration": "Cualquiera",
        "subtitle_template": "Default",
        "options": {
            "emojis": True,
            "keywords": True,
            "broll": False,
            "silence": False,
            "censor": False,
        },
        "moment": "",
        "whisper_model": "small",
        "schedule": {
            "scheduled_at": scheduled_at,
            "platform": "youtube",
            "visibility": "private",
        },
    }
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/projects",
            files={"file": ("source.mp4", video_with_audio.read_bytes(), "video/mp4")},
            data={"config": json.dumps(config)},
        )

    assert response.status_code == 201
    persisted = response.json()["config"]["schedule"]["scheduled_at"]
    assert datetime.fromisoformat(persisted.replace("Z", "+00:00")) == datetime.fromisoformat(
        scheduled_at
    )


def test_scheduled_job_exports_real_clip_if_account_is_not_connected(
    video_with_audio: Path, tmp_path: Path
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
    with TestClient(create_app(settings)) as client:
        project_id, clip_id, clip_path = _create_completed_project_with_clip(
            client, settings, video_with_audio
        )
        jobs = database.enqueue_schedule_jobs(
            settings.database_path,
            project_id,
            [clip_id],
            "youtube",
            (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "private",
        )
        assert len(jobs) == 1
        assert database.claim_schedule_job(settings.database_path, jobs[0]["id"])

        process_schedule_job(settings, jobs[0])

        job = database.get_schedule_job(settings.database_path, jobs[0]["id"])
        assert job is not None
        assert job["status"] == "exported"
        assert "no se publicó" in job["error"]
        response = client.get(f"/api/scheduled-jobs/{job['id']}/download")
        assert response.status_code == 200
        assert response.content == clip_path.read_bytes()


def test_broll_overlay_renders_with_a_real_local_video(
    video_with_audio: Path, tmp_path: Path
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    config = ProjectConfig(ratio="16:9")
    proposal = ClipProposal(
        inicio=0,
        fin=2,
        titulo="Clip de prueba",
        puntuacion=7,
        razon_viral="Prueba de composición de video.",
    )
    output = tmp_path / "clip-broll.mp4"
    render_clip(
        video_with_audio,
        output,
        tmp_path / "subtitles.ass",
        proposal,
        config,
        [],
        ffmpeg,
        lambda _: None,
        broll_path=video_with_audio,
    )
    metadata = probe_media(output, ffprobe)

    assert metadata["width"] == 1280
    assert metadata["height"] == 720
    assert metadata["has_audio"] is True


def test_broll_processing_requires_the_selected_provider_key(
    video_with_audio: Path, tmp_path: Path
) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        cors_origins=(),
        llm_provider="Groq",
        llm_api_key="configured-llm-key",
        llm_model="test-model",
        llm_base_url="",
        whisper_model="small",
        broll_provider="Pexels",
    )
    config = {
        "ratio": "9:16",
        "duration": "Cualquiera",
        "subtitle_template": "Default",
        "options": {
            "emojis": True,
            "keywords": True,
            "broll": True,
            "silence": False,
            "censor": False,
        },
        "moment": "",
        "whisper_model": "small",
    }
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/projects",
            files={"file": ("source.mp4", video_with_audio.read_bytes(), "video/mp4")},
            data={"config": json.dumps(config)},
        )
        project_id = created.json()["id"]
        response = client.post(
            f"/api/projects/{project_id}/process",
            json={"config": config},
        )

    assert response.status_code == 422
    assert "PEXELS_API_KEY" in response.json()["detail"]
    assert database.get_project(settings.database_path, project_id)["status"] == "ready"


def test_packaged_frontend_uses_spa_fallback_and_preserves_api_404(
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
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html>kellyClipAi</html>", encoding="utf-8")
    (web_dir / "app.js").write_text("window.kellyClipAi = true;", encoding="utf-8")

    with TestClient(create_app(settings, web_dir=web_dir)) as client:
        root = client.get("/")
        route = client.get("/p/project-id/configure")
        asset = client.get("/app.js")
        missing_api = client.get("/api/not-found")

    assert root.status_code == 200
    assert route.text == "<html>kellyClipAi</html>"
    assert asset.text == "window.kellyClipAi = true;"
    assert missing_api.status_code == 404


def test_desktop_launcher_creates_user_config_outside_install_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import desktop

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("PATH", "original-path")

    user_dir = desktop._prepare_user_directory()

    assert (user_dir / ".env").read_text(encoding="utf-8").startswith("# Carpeta local")
    assert Path(os.environ["KELLYCLIP_DATA_DIR"]) == user_dir / "data"
    assert os.environ["KELLYCLIP_FRONTEND_BASE_URL"] == "http://127.0.0.1:8000"
    assert Path(os.environ["PATH"].split(os.pathsep)[0]) == desktop._resource_path("bin")
