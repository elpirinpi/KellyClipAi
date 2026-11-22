"""Tests del hotfix v1.4.1: bugs 1 y 4 reportados por Kelly (21-11-2026).

Bug 1 — "Me sale ese aviso siempre": el aviso "La aplicación se cerró
antes de completar la importación. Inténtalo de nuevo." aparecía en cada
arranque. Causa raíz: `latest_staged_media` filtraba por estado ANTES de
elegir la fila más reciente, así que una importación fallida antigua
seguía siendo "la última" aunque después hubiera importaciones exitosas
(sus filas pasan a 'completed' y el filtro las saltaba). El frontend
muestra ese error en cada montaje de la página principal.

Bug 4 — "Para poder borrar los proyectos tengo que entrar en ellos, no
hay un botón para eliminarlo desde fuera": el endpoint DELETE
/api/projects/{id} ya existía (lo usaba ResultsPage dentro del proyecto);
falta el botón en la lista "Mis proyectos" (ProjectsPage.tsx). Estos
tests cubren el endpoint desde el punto de vista de la lista: borrar un
proyecto terminado lo quita de GET /api/projects y de disco.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend import database
from backend.main import create_app as create_application
from backend.settings import AppSettings

INTERRUPTED_MESSAGE = (
    "La aplicación se cerró antes de completar la importación. Inténtalo de nuevo."
)


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


def _complete_staged_upload(client: TestClient, *, filename: str = "video.mp4") -> str:
    """Sube un archivo pequeño por la vía staged y crea su proyecto."""
    created = client.post(
        "/api/staged-media/uploads",
        json={"filename": filename, "file_size": 10},
    )
    assert created.status_code == 201
    staged_id = created.json()["id"]
    content = client.post(
        f"/api/staged-media/{staged_id}/content",
        content=b"0123456789",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Range": "bytes 0-9/10",
        },
    )
    assert content.status_code == 200
    assert content.json()["status"] == "ready"
    project = client.post(
        f"/api/staged-media/{staged_id}/project",
        json={"config": {}},
    )
    assert project.status_code == 201, project.text
    return staged_id


# --- Bug 1: latest_staged_media -------------------------------------------


def test_latest_ignores_failed_row_superseded_by_completed_import(
    tmp_path: Path,
) -> None:
    """Regresión del bug 1: una importación fallida antigua no debe seguir
    siendo "la última" después de una importación exitosa."""
    database_path = tmp_path / "kca.sqlite3"
    database.initialize(database_path)
    # Importación interrumpida (p. ej. Kelly cerró la app a mitad de
    # subida): la fila queda en 'receiving' y el arranque la marca failed.
    database.create_staged_media(
        database_path, "vieja", "upload", "receiving", "video1.mp4", file_size=100
    )
    database.fail_interrupted_staged_media(database_path)
    failed = database.get_staged_media(database_path, "vieja")
    assert failed is not None and failed["status"] == "failed"
    assert failed["error"] == INTERRUPTED_MESSAGE
    # Después, una importación exitosa: ready -> finalizing -> completed.
    database.create_staged_media(
        database_path, "nueva", "upload", "receiving", "video2.mp4", file_size=100
    )
    database.update_staged_media(database_path, "nueva", status="ready")
    assert database.claim_staged_media(database_path, "nueva")
    database.update_staged_media(
        database_path, "nueva", status="completed", project_id="proyecto-1"
    )
    # La más reciente está completada: no hay nada que reanudar ni avisar.
    assert database.latest_staged_media(database_path, "upload") is None


def test_latest_still_reports_newest_failed_import(tmp_path: Path) -> None:
    """El aviso NO se esconde: si la última importación falló, latest la
    sigue devolviendo con el mensaje exacto."""
    database_path = tmp_path / "kca.sqlite3"
    database.initialize(database_path)
    database.create_staged_media(
        database_path, "vieja", "upload", "receiving", "video1.mp4", file_size=100
    )
    database.update_staged_media(database_path, "vieja", status="ready")
    assert database.claim_staged_media(database_path, "vieja")
    database.update_staged_media(
        database_path, "vieja", status="completed", project_id="proyecto-1"
    )
    database.create_staged_media(
        database_path, "nueva", "upload", "receiving", "video2.mp4", file_size=100
    )
    database.fail_interrupted_staged_media(database_path)
    latest = database.latest_staged_media(database_path, "upload")
    assert latest is not None
    assert latest["id"] == "nueva"
    assert latest["status"] == "failed"
    assert latest["error"] == INTERRUPTED_MESSAGE


def test_latest_returns_receiving_and_ready_for_resume(tmp_path: Path) -> None:
    """La reanudación no se rompe: receiving y ready siguen devolviéndose."""
    database_path = tmp_path / "kca.sqlite3"
    database.initialize(database_path)
    database.create_staged_media(
        database_path, "a", "upload", "receiving", "video1.mp4", file_size=100
    )
    latest = database.latest_staged_media(database_path, "upload")
    assert latest is not None and latest["id"] == "a"
    assert latest["status"] == "receiving"
    database.update_staged_media(database_path, "a", status="ready")
    latest = database.latest_staged_media(database_path, "upload")
    assert latest is not None and latest["status"] == "ready"


def test_failed_upload_warning_disappears_after_successful_import(
    tmp_path: Path,
) -> None:
    """Extremo a extremo del bug 1 por HTTP, con la entrada exacta de
    Kelly: una subida interrumpida, reinicio de la app (el lifespan marca
    la fila como fallida), aviso visible en /latest, importación exitosa
    después, y el aviso ya no aparece."""
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/staged-media/uploads",
            json={"filename": "video1.mp4", "file_size": 10},
        )
        assert created.status_code == 201
        # La app se cierra a mitad de subida: la fila queda 'receiving'.
    # Reinicio: el lifespan ejecuta fail_interrupted_staged_media.
    with TestClient(create_app(settings)) as client:
        latest = client.get("/api/staged-media/latest?source=upload")
        assert latest.status_code == 200
        body = latest.json()
        assert body is not None and body["status"] == "failed"
        assert body["error"] == INTERRUPTED_MESSAGE
        # Kelly importa otro video y lo completa con éxito.
        _complete_staged_upload(client, filename="video2.mp4")
        latest = client.get("/api/staged-media/latest?source=upload")
        assert latest.status_code == 200
        assert latest.json() is None


def test_failed_staged_row_can_be_dismissed_via_delete(tmp_path: Path) -> None:
    """El botón "Descartar" del aviso borra la fila staged fallida: el
    aviso se mostró una vez y no reaparece en el siguiente arranque."""
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/staged-media/uploads",
            json={"filename": "video1.mp4", "file_size": 10},
        )
        assert created.status_code == 201
        staged_id = created.json()["id"]
    with TestClient(create_app(settings)) as client:
        latest = client.get("/api/staged-media/latest?source=upload")
        assert latest.json()["status"] == "failed"
        dismissed = client.delete(f"/api/staged-media/{staged_id}")
        assert dismissed.status_code == 204
        latest = client.get("/api/staged-media/latest?source=upload")
        assert latest.json() is None


def test_failed_staged_download_cleans_partial_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Patrón KellyAiClip ("sin parciales atrás"), ya presente en
    _download_staged_media: una descarga por enlace que falla no deja
    parciales (no se reanuda: la fila queda 'failed'). La fila conserva
    su mensaje de error; este test fija ese comportamiento."""
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    staged_id = str(uuid.uuid4())
    database.create_staged_media(
        settings.database_path,
        staged_id,
        "url",
        "downloading",
        "Descarga de enlace",
    )

    def failing_download(url: str, target_dir: Path, progress_hook=None):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "video.mp4.part").write_bytes(b"partial-bytes")
        raise RuntimeError("yt-dlp falló a mitad de la descarga")

    monkeypatch.setattr(backend_main, "_download_video", failing_download)
    backend_main._download_staged_media(settings, staged_id, "https://youtube.com/watch?v=x")

    staged = database.get_staged_media(settings.database_path, staged_id)
    assert staged is not None and staged["status"] == "failed"
    assert staged["error"] is not None and "No se pudo descargar" in staged["error"]
    staging_root = settings.data_dir / "staging"
    assert not (staging_root / staged_id).exists()
    assert not (staging_root / f"{staged_id}.source").exists()


# --- Bug 4: eliminar proyecto desde la lista ------------------------------


def test_delete_project_from_list_removes_it_from_projects_and_disk(
    tmp_path: Path,
) -> None:
    """El botón de eliminar de "Mis proyectos" usa DELETE
    /api/projects/{id}: el proyecto desaparece de la lista y de disco."""
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        uploaded = client.post(
            "/api/projects",
            files={"file": ("source.mp4", b"x" * 11, "video/mp4")},
            data={"config": json.dumps({})},
        )
        assert uploaded.status_code == 201, uploaded.text
        project_id = uploaded.json()["id"]
        project_dir = settings.data_dir / "projects" / project_id
        assert project_dir.is_dir()
        listed = client.get("/api/projects")
        assert listed.status_code == 200
        assert any(item["id"] == project_id for item in listed.json()["items"])
        deleted = client.delete(f"/api/projects/{project_id}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"deleted": True}
        assert not project_dir.exists()
        listed = client.get("/api/projects")
        assert listed.status_code == 200
        assert all(item["id"] != project_id for item in listed.json()["items"])


def test_delete_project_twice_returns_404(tmp_path: Path) -> None:
    """Borrar dos veces el mismo proyecto: la segunda es 404, no 500."""
    settings = _settings(tmp_path / "data")
    with TestClient(create_app(settings)) as client:
        uploaded = client.post(
            "/api/projects",
            files={"file": ("source.mp4", b"x" * 11, "video/mp4")},
            data={"config": json.dumps({})},
        )
        assert uploaded.status_code == 201
        project_id = uploaded.json()["id"]
        assert client.delete(f"/api/projects/{project_id}").status_code == 200
        second = client.delete(f"/api/projects/{project_id}")
        assert second.status_code == 404
