"""Tests del hotfix v1.4.1 (bugs 2 y 3 de la pantalla de clips).

Bug 3: las propuestas sin editar (edición automática apagada, v1.3.2)
no tenían ninguna vista previa — ni <video> ni <img> ni endpoint que
las sirviera. Cubre el endpoint nuevo
GET /api/projects/{id}/proposals/{proposalId}/thumbnail, el helper
create_frame_thumbnail y el camino de vista previa de los clips
renderizados (GET .../clips/{id}/file). El bug 2 es CSS (la sección
de propuestas usaba hex fijo en vez de las variables del tema): no
tiene tests de backend; se verifica por inspección y con el build.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend import database
from backend.auth import AuthenticatedUser, AuthenticationError
from backend.main import create_app as create_application
from backend.media import create_frame_thumbnail
from backend.schemas import ProjectConfig
from backend.settings import AppSettings

USER_A = "10000000-0000-4000-8000-000000000001"
PROJECT_ID = "project-1"
PROPOSAL_ID = "proposal-1"


class TokenVerifier:
    async def verify(self, token: str) -> AuthenticatedUser:
        if token != "token-a":
            raise AuthenticationError("Sesión no válida.")
        return AuthenticatedUser(USER_A, None)


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


def _make_video(path: Path, *, duration: int = 2) -> Path:
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
            f"testsrc=duration={duration}:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _proposal(
    proposal_id: str = PROPOSAL_ID,
    *,
    inicio: float = 0.5,
    fin: float = 1.5,
) -> dict[str, Any]:
    return {
        "id": proposal_id,
        "proposal": {
            "inicio": inicio,
            "fin": fin,
            "titulo": "Momento de prueba",
            "puntuacion": 8.0,
            "razon_viral": "Energía alta.",
        },
    }


def _create_project(
    settings: AppSettings,
    *,
    owner_id: str = "test-user",
    metadata: dict[str, Any] | None = None,
    clips: list[dict[str, Any]] | None = None,
    source: Path | None = None,
) -> Path:
    project_dir = settings.data_dir / "projects" / PROJECT_ID
    project_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        (project_dir / "source").write_bytes(source.read_bytes())
    database.initialize(settings.database_path)
    database.create_project(
        settings.database_path,
        PROJECT_ID,
        f"Proyecto {PROJECT_ID}",
        project_dir / "source",
        ProjectConfig().model_dump(mode="json"),
        owner_id,
    )
    values: dict[str, Any] = {"status": "completed"}
    if metadata is not None:
        values["metadata"] = metadata
    if clips is not None:
        values["clips"] = clips
    database.update_project(settings.database_path, PROJECT_ID, **values)
    return project_dir


def _metadata_with_proposal() -> dict[str, Any]:
    return {"duration": 2.0, "clip_proposals": [_proposal()]}


def _client(settings: AppSettings, *, auth: bool = False) -> TestClient:
    if auth:
        return TestClient(create_application(settings, auth_verifier=TokenVerifier()))
    return TestClient(create_application(settings, auth_required=False))


def _thumbnail_url(project_id: str = PROJECT_ID, proposal_id: str = PROPOSAL_ID) -> str:
    return f"/api/projects/{project_id}/proposals/{proposal_id}/thumbnail"


# ---------------------------------------------------------------------------
# create_frame_thumbnail (backend/media.py)
# ---------------------------------------------------------------------------


def test_create_frame_thumbnail_writes_a_jpeg(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "source.mp4")
    output = tmp_path / "frame.jpg"
    create_frame_thumbnail(video, output, 1.0, shutil.which("ffmpeg") or "ffmpeg")
    assert output.is_file()
    assert output.read_bytes()[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/proposals/{proposalId}/thumbnail
# ---------------------------------------------------------------------------


def test_proposal_thumbnail_generates_and_caches_the_frame(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    project_dir = _create_project(
        settings, metadata=_metadata_with_proposal(), source=video
    )
    client = _client(settings)
    response = client.get(_thumbnail_url())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
    cached = project_dir / "proposal-thumbnails" / f"{PROPOSAL_ID}.jpg"
    assert cached.is_file()
    assert cached.read_bytes() == response.content


def test_proposal_thumbnail_serves_cache_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    project_dir = _create_project(settings, metadata=_metadata_with_proposal())
    cached = project_dir / "proposal-thumbnails" / f"{PROPOSAL_ID}.jpg"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"\xff\xd8\xff\xe0cached")
    monkeypatch.setattr(
        main_module,
        "create_frame_thumbnail",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("No debe regenerar una miniatura cacheada.")
        ),
    )
    client = _client(settings)
    response = client.get(_thumbnail_url())
    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff\xe0cached"


def test_proposal_thumbnail_404_for_unknown_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database.initialize(settings.database_path)
    client = _client(settings)
    response = client.get(_thumbnail_url(project_id="no-existe"))
    assert response.status_code == 404


def test_proposal_thumbnail_404_for_unknown_proposal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    _create_project(settings, metadata=_metadata_with_proposal(), source=video)
    client = _client(settings)
    response = client.get(_thumbnail_url(proposal_id="otra-propuesta"))
    assert response.status_code == 404


def test_proposal_thumbnail_404_for_malformed_proposal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    metadata = {
        "duration": 2.0,
        "clip_proposals": [
            {"id": PROPOSAL_ID, "proposal": {"inicio": 0.5, "titulo": "Sin fin"}}
        ],
    }
    _create_project(settings, metadata=metadata, source=video)
    client = _client(settings)
    response = client.get(_thumbnail_url())
    assert response.status_code == 404


def test_proposal_thumbnail_404_without_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _create_project(settings, metadata=_metadata_with_proposal(), source=None)
    client = _client(settings)
    response = client.get(_thumbnail_url())
    assert response.status_code == 404


def test_proposal_thumbnail_503_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    _create_project(settings, metadata=_metadata_with_proposal(), source=video)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    client = _client(settings)
    response = client.get(_thumbnail_url())
    assert response.status_code == 503


def test_proposal_thumbnail_requires_auth_but_accepts_media_cookie(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    _create_project(
        settings,
        owner_id=USER_A,
        metadata=_metadata_with_proposal(),
        source=video,
    )
    client = _client(settings, auth=True)
    # Sin sesión: el <img> no puede enviar Authorization, así que sin
    # cookie de medios la petición se rechaza.
    assert client.get(_thumbnail_url()).status_code == 401
    # El flujo real: la app sincroniza la cookie tras el login.
    cookie_response = client.post(
        "/api/auth/media-cookie", headers={"Authorization": "Bearer token-a"}
    )
    assert cookie_response.status_code == 204
    response = client.get(_thumbnail_url())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# Vista previa de los clips renderizados (camino existente, bug 3)
# ---------------------------------------------------------------------------


def test_rendered_clip_file_serves_the_clip_bytes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    video = _make_video(tmp_path / "video.mp4")
    clips = [
        {
            "id": "clip-1",
            "title": "Clip",
            "start": 0.5,
            "end": 1.5,
            "duration": 1.0,
            "score": 7.5,
            "status": "ready",
            "filename": "clip-1.mp4",
            "transcript": [],
            "width": 720,
            "height": 1280,
        }
    ]
    project_dir = _create_project(
        settings, metadata={"duration": 2.0}, clips=clips, source=video
    )
    clips_dir = project_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "clip-1.mp4").write_bytes(video.read_bytes())
    client = _client(settings)
    response = client.get(f"/api/projects/{PROJECT_ID}/clips/clip-1/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == video.read_bytes()
