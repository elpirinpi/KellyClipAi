from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend import database
from backend.main import create_app
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
