"""Tests de GET /api/projects/{id}/face-track (Studio, v1.3.1; asíncrono v1.3.2).

Cubre: 404 (proyecto inexistente, de otro usuario o sin fuente), 401
sin autenticación, 400 con ratio inválido, 200 con has_face=false en
un video sin rostros (video real, YuNet), forma de la respuesta y
fracciones en [0,1] (análisis mockeado: no hay fixture con rostro en
el repo y MediaPipe no está instalado en el venv de tests), la
selección de dimensiones por ratio, la caché en la segunda llamada,
503 cuando el seguimiento facial no está disponible o se desconoce la
duración, y la conversión píxeles → fracciones de
``analyze_source_face_track``. Además, el contrato asíncrono (v1.3.2):
202 con progreso mientras el trabajo corre, un solo vuelo por
proyecto+ratio, reintento tras un trabajo fallido, y la degradación
elegante sin MediaPipe (seguimiento con YuNet solo) más el salto del
landmarker con un solo rostro.
"""

from __future__ import annotations

import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_speaker_framing import _video_with_audio

import backend.main as main_module
from backend import database
from backend import face_tracking as face_tracking_module
from backend.auth import AuthenticatedUser, AuthenticationError
from backend.face_tracking import (
    STUDIO_FACE_TRACK_SAMPLE_RATE,
    FaceCropKeyframe,
    FaceTrackingError,
    _crop_geometry,
    _merge_speech_intervals,
    _speech_active_at,
    analyze_face_crop_paths,
    analyze_source_face_track,
    clear_face_analysis_cache,
    clear_face_track_jobs,
    peek_cached_source_face_track,
)
from backend.main import create_app as create_application
from backend.schemas import ProjectConfig
from backend.settings import AppSettings

# La fixture importada solo se usa como parámetro de test (pytest la
# resuelve por nombre); se reexporta para que ruff no la marque como
# import sin usar.
__all__ = ["_video_with_audio"]

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
            ],
        },
    ]


def _create_project(
    settings: AppSettings,
    project_id: str,
    *,
    owner_id: str = "test-user",
    transcript: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    source_bytes: bytes | None = b"fake-source-video-bytes",
    source_path: Path | None = None,
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    if source_path is not None:
        shutil.copyfile(source_path, project_dir / "source")
    elif source_bytes is not None:
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
    values: dict[str, Any] = {"status": "completed"}
    if transcript is not None:
        values["transcript"] = transcript
    if metadata is not None:
        values["metadata"] = metadata
    database.update_project(settings.database_path, project_id, **values)


def _client(settings: AppSettings, *, auth: bool = False) -> TestClient:
    if auth:
        return TestClient(create_application(settings, auth_verifier=TokenVerifier()))
    return TestClient(create_application(settings, auth_required=False))


def _fake_analysis(
    points: list[FaceCropKeyframe] | None,
    calls: list[dict[str, Any]],
):
    def fake(
        input_path: Path,
        duration: float,
        output_width: int,
        output_height: int,
        speech_intervals: list[tuple[float, float]] | None = None,
        sample_rate: float = STUDIO_FACE_TRACK_SAMPLE_RATE,
        on_progress: Any = None,
    ) -> tuple[bool, list[FaceCropKeyframe]]:
        calls.append(
            {
                "input_path": input_path,
                "duration": duration,
                "output_width": output_width,
                "output_height": output_height,
                "speech_intervals": speech_intervals,
                "sample_rate": sample_rate,
            }
        )
        if on_progress is not None:
            on_progress(100.0)
        if points is None:
            return False, []
        return True, points

    return fake


@pytest.fixture(autouse=True)
def _clear_face_caches():
    # La caché de análisis y el registro de trabajos son globales del
    # módulo: limpiarlos entre tests evita que un trabajo o un
    # resultado de otro test contamine el siguiente.
    clear_face_analysis_cache()
    clear_face_track_jobs()
    yield
    clear_face_analysis_cache()
    clear_face_track_jobs()


def _poll_face_track(
    client: TestClient, url: str, *, timeout: float = 15.0
):
    """Sondea GET face-track hasta que no sea 202 (200 o error)."""
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(url)
        if response.status_code != 202:
            return response
        if time.monotonic() > deadline:
            raise AssertionError(
                f"face-track siguió en 202 tras {timeout}s: {response.text}"
            )
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# 404 / 401 / 400
# ---------------------------------------------------------------------------


def test_face_track_404_for_missing_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    database.initialize(settings.database_path)
    with _client(settings) as client:
        response = client.get("/api/projects/missing/face-track")
    assert response.status_code == 404


def test_face_track_404_without_source_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1")
    (settings.data_dir / "projects" / "project-1" / "source").unlink()
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/face-track")
    assert response.status_code == 404


def test_face_track_requires_authentication(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", owner_id=USER_A)
    with _client(settings, auth=True) as client:
        response = client.get("/api/projects/project-1/face-track")
    assert response.status_code == 401


def test_face_track_rejects_other_owners_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", owner_id=USER_A)
    headers = {"Authorization": "Bearer token-b"}
    with _client(settings, auth=True) as client:
        response = client.get("/api/projects/project-1/face-track", headers=headers)
    assert response.status_code == 404


@pytest.mark.parametrize("ratio", ["4:5", "21:9", "foo", ""])
def test_face_track_400_for_invalid_ratio(ratio: str, tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})
    with _client(settings) as client:
        response = client.get(f"/api/projects/project-1/face-track?ratio={ratio}")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 200: video real sin rostros
# ---------------------------------------------------------------------------


def test_face_track_empty_for_video_without_faces(
    _video_with_audio: Path,
    tmp_path: Path,
) -> None:
    # Sin metadata ni transcripción: la duración viene de la sonda del
    # archivo (PyAV/ffprobe) y el análisis es real (YuNet, sin rostros
    # en un video negro). MediaPipe no interviene: sin rostros
    # detectados no se inicializa el landmarker.
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", source_path=_video_with_audio)
    with _client(settings) as client:
        response = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project_id"] == "project-1"
    assert body["ratio"] == "9:16"
    assert body["duration"] == pytest.approx(2.0, abs=0.05)
    assert body["sample_fps"] == 2.0
    assert body["has_face"] is False
    assert body["keyframes"] == []


# ---------------------------------------------------------------------------
# 200: forma de la respuesta (análisis mockeado)
# ---------------------------------------------------------------------------


def test_face_track_response_shape_and_fractions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(
        settings,
        "project-1",
        transcript=_transcript(),
        metadata={"duration": 2.0},
    )
    points = [
        FaceCropKeyframe(0.0, 0.0, 0.5),
        FaceCropKeyframe(0.75, 1.0, 0.0),
        FaceCropKeyframe(2.0, 0.25, 0.75),
    ]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        main_module, "analyze_source_face_track", _fake_analysis(points, calls)
    )
    with _client(settings) as client:
        response = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "project_id": "project-1",
        "ratio": "9:16",
        "duration": 2.0,
        "sample_fps": 2.0,
        "has_face": True,
        "keyframes": [
            {"time": 0.0, "x": 0.0, "y": 0.5},
            {"time": 0.75, "x": 1.0, "y": 0.0},
            {"time": 2.0, "x": 0.25, "y": 0.75},
        ],
    }
    times = [keyframe["time"] for keyframe in body["keyframes"]]
    assert times == sorted(times)
    for keyframe in body["keyframes"]:
        assert 0.0 <= keyframe["x"] <= 1.0
        assert 0.0 <= keyframe["y"] <= 1.0
    # El endpoint deriva los intervalos de habla de la transcripción
    # (espejo de render_clip) y usa las dimensiones 9:16 a 2 fps.
    assert len(calls) == 1
    assert calls[0]["output_width"] == 1080
    assert calls[0]["output_height"] == 1920
    assert calls[0]["sample_rate"] == 2.0
    assert calls[0]["speech_intervals"] == [
        (0.0, 0.4),
        (0.5, 0.8),
        (0.9, 1.4),
        (1.5, 2.2),
    ]


@pytest.mark.parametrize(
    ("ratio", "width", "height"),
    [("9:16", 1080, 1920), ("1:1", 1080, 1080), ("16:9", 1920, 1080)],
)
def test_face_track_ratio_selects_output_dimensions(
    ratio: str,
    width: int,
    height: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        main_module, "analyze_source_face_track", _fake_analysis(None, calls)
    )
    with _client(settings) as client:
        response = _poll_face_track(
            client, f"/api/projects/project-1/face-track?ratio={ratio}"
        )
    assert response.status_code == 200, response.text
    assert response.json()["ratio"] == ratio
    assert response.json()["has_face"] is False
    assert len(calls) == 1
    assert calls[0]["output_width"] == width
    assert calls[0]["output_height"] == height
    # Sin transcripción, sin intervalos de habla.
    assert calls[0]["speech_intervals"] is None


# ---------------------------------------------------------------------------
# Caché y errores
# ---------------------------------------------------------------------------


def test_face_track_second_call_uses_cache(
    _video_with_audio: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", source_path=_video_with_audio)
    original = face_tracking_module._analyze_face_crop_paths_uncached
    calls: list[int] = []

    def spy(*args: Any, **kwargs: Any):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        face_tracking_module, "_analyze_face_crop_paths_uncached", spy
    )
    with _client(settings) as client:
        first = _poll_face_track(client, "/api/projects/project-1/face-track")
        # Segunda llamada: 200 inmediata por el peek de caché, sin
        # pasar por 202 ni lanzar otro trabajo.
        second = client.get("/api/projects/project-1/face-track")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json() == second.json()
    assert len(calls) == 1


def test_face_track_503_when_face_tracking_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})

    def failing(*args: Any, **kwargs: Any):
        raise FaceTrackingError(
            "Falta OpenCV para realizar el seguimiento facial. "
            "Reinstala la aplicación."
        )

    monkeypatch.setattr(main_module, "analyze_source_face_track", failing)
    with _client(settings) as client:
        response = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert response.status_code == 503
    assert "OpenCV" in response.json()["detail"]


def test_face_track_503_when_duration_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fuente ilegible (bytes falsos: la sonda falla) y sin duración en
    # metadata ni transcripción: no hay forma de saber la duración.
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        main_module, "analyze_source_face_track", _fake_analysis(None, calls)
    )
    with _client(settings) as client:
        response = client.get("/api/projects/project-1/face-track")
    assert response.status_code == 503
    assert calls == []


# ---------------------------------------------------------------------------
# Contrato asíncrono (v1.3.2): 202 + sondeo, un solo vuelo, reintento
# ---------------------------------------------------------------------------


def test_face_track_202_while_pending_then_200(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})
    gate = threading.Event()
    calls: list[int] = []

    def slow(*args: Any, on_progress: Any = None, **kwargs: Any):
        calls.append(1)
        gate.wait(timeout=10)
        return True, [FaceCropKeyframe(0.0, 0.5, 0.5)]

    monkeypatch.setattr(main_module, "analyze_source_face_track", slow)
    with _client(settings) as client:
        try:
            first = client.get("/api/projects/project-1/face-track")
            assert first.status_code == 202, first.text
            body = first.json()
            assert body["status"] == "pending"
            assert 0.0 <= body["progress"] <= 100.0
            # Segunda petición mientras el trabajo corre: también 202,
            # sin lanzar otro análisis (un solo vuelo).
            second = client.get("/api/projects/project-1/face-track")
            assert second.status_code == 202, second.text
        finally:
            gate.set()
        done = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert done.status_code == 200, done.text
    assert done.json()["has_face"] is True
    assert len(calls) == 1


def test_face_track_reports_progress_while_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})
    gate = threading.Event()
    reported = threading.Event()

    def slow(*args: Any, on_progress: Any = None, **kwargs: Any):
        if on_progress is not None:
            on_progress(42.0)
        reported.set()
        gate.wait(timeout=10)
        return False, []

    monkeypatch.setattr(main_module, "analyze_source_face_track", slow)
    with _client(settings) as client:
        try:
            first = client.get("/api/projects/project-1/face-track")
            assert first.status_code == 202, first.text
            assert reported.wait(timeout=5), "El trabajo no reportó progreso."
            pending = client.get("/api/projects/project-1/face-track")
            assert pending.status_code == 202, pending.text
            assert pending.json() == {"status": "pending", "progress": 42.0}
        finally:
            gate.set()
        done = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert done.status_code == 200, done.text
    assert done.json()["has_face"] is False


def test_face_track_failed_job_retries_on_next_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    _create_project(settings, "project-1", metadata={"duration": 2.0})
    state = {"fail": True}
    calls: list[int] = []

    def flaky(*args: Any, **kwargs: Any):
        calls.append(1)
        if state["fail"]:
            raise FaceTrackingError("El detector facial falló.")
        return True, [FaceCropKeyframe(0.0, 0.5, 0.5)]

    monkeypatch.setattr(main_module, "analyze_source_face_track", flaky)
    with _client(settings) as client:
        failed = _poll_face_track(client, "/api/projects/project-1/face-track")
        assert failed.status_code == 503
        assert "detector" in failed.json()["detail"]
        state["fail"] = False
        done = _poll_face_track(client, "/api/projects/project-1/face-track")
    assert done.status_code == 200, done.text
    assert done.json()["has_face"] is True
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# analyze_source_face_track: píxeles → fracciones
# ---------------------------------------------------------------------------


def test_analyze_source_face_track_converts_pixels_to_fractions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    _, max_x, max_y = _crop_geometry(1920, 1080, 1080, 1920)
    assert max_y == 0  # 16:9 → 9:16: solo hay desplazamiento horizontal.
    pixel_points = [
        FaceCropKeyframe(0.0, 0.0, 0.0),
        FaceCropKeyframe(1.0, max_x / 2, 0.0),
        FaceCropKeyframe(2.0, max_x, 0.0),
    ]
    captured: dict[str, Any] = {}

    def fake_analyze(
        input_path: Path,
        clip_start: float,
        intervals: list[tuple[float, float]],
        output_width: int,
        output_height: int,
        speech_intervals: list[tuple[float, float]] | None = None,
        on_progress: Any = None,
        sample_rate: float = 4.0,
    ):
        captured.update(
            {
                "clip_start": clip_start,
                "intervals": intervals,
                "output_width": output_width,
                "output_height": output_height,
                "speech_intervals": speech_intervals,
                "sample_rate": sample_rate,
            }
        )
        return [pixel_points]

    monkeypatch.setattr(face_tracking_module, "analyze_face_crop_paths", fake_analyze)
    monkeypatch.setattr(
        face_tracking_module, "_probe_video_dimensions", lambda _path: (1920, 1080)
    )
    has_face, keyframes = analyze_source_face_track(
        video, 2.0, 1080, 1920, speech_intervals=[(0.0, 1.0)]
    )
    assert has_face is True
    assert keyframes == [
        FaceCropKeyframe(0.0, 0.0, 0.5),
        FaceCropKeyframe(1.0, 0.5, 0.5),
        FaceCropKeyframe(2.0, 1.0, 0.5),
    ]
    # Un solo intervalo (0, duration), clip_start=0, 2 fps.
    assert captured == {
        "clip_start": 0.0,
        "intervals": [(0.0, 2.0)],
        "output_width": 1080,
        "output_height": 1920,
        "speech_intervals": [(0.0, 1.0)],
        "sample_rate": 2.0,
    }


def test_analyze_source_face_track_empty_when_no_face(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    monkeypatch.setattr(
        face_tracking_module,
        "analyze_face_crop_paths",
        lambda *args, **kwargs: [None],
    )
    monkeypatch.setattr(
        face_tracking_module,
        "_probe_video_dimensions",
        lambda _path: pytest.fail("Sin rostro, no hace falta sondear dimensiones."),
    )
    assert analyze_source_face_track(video, 2.0, 1080, 1920) == (False, [])


def test_analyze_source_face_track_clamps_fractions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    _, max_x, _max_y = _crop_geometry(1920, 1080, 1080, 1920)
    monkeypatch.setattr(
        face_tracking_module,
        "analyze_face_crop_paths",
        lambda *args, **kwargs: [
            [
                FaceCropKeyframe(0.0, -50.0, 0.0),
                FaceCropKeyframe(1.0, max_x + 500.0, 0.0),
            ]
        ],
    )
    monkeypatch.setattr(
        face_tracking_module, "_probe_video_dimensions", lambda _path: (1920, 1080)
    )
    has_face, keyframes = analyze_source_face_track(video, 1.0, 1080, 1920)
    assert has_face is True
    assert keyframes == [
        FaceCropKeyframe(0.0, 0.0, 0.5),
        FaceCropKeyframe(1.0, 1.0, 0.5),
    ]


def test_analyze_source_face_track_rejects_non_positive_duration(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    with pytest.raises(ValueError, match="positiva"):
        analyze_source_face_track(video, 0.0, 1080, 1920)


# ---------------------------------------------------------------------------
# La frecuencia de muestreo forma parte de la clave de caché
# ---------------------------------------------------------------------------


def test_face_analysis_cache_distinguishes_sample_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    calls: list[float] = []
    sentinel = [[None]]
    monkeypatch.setattr(
        face_tracking_module,
        "_analyze_face_crop_paths_uncached",
        lambda *a, **k: calls.append(k.get("sample_rate", 4.0)) or sentinel,
    )
    clear_face_analysis_cache()
    first = face_tracking_module.analyze_face_crop_paths(
        video, 0.0, [(0.0, 10.0)], 720, 1280
    )
    second = face_tracking_module.analyze_face_crop_paths(
        video, 0.0, [(0.0, 10.0)], 720, 1280, sample_rate=2.0
    )
    third = face_tracking_module.analyze_face_crop_paths(
        video, 0.0, [(0.0, 10.0)], 720, 1280, sample_rate=2.0
    )
    assert first is second is third is sentinel
    # 4 fps y 2 fps son claves distintas; repetir 2 fps acierta en caché.
    assert calls == [4.0, 2.0]


def test_analyze_face_crop_paths_rejects_non_positive_sample_rate(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    with pytest.raises(ValueError, match="positiva"):
        face_tracking_module.analyze_face_crop_paths(
            video, 0.0, [(0.0, 1.0)], 720, 1280, sample_rate=0.0
        )


# ---------------------------------------------------------------------------
# Búsqueda de habla por bisect, progreso y peek de caché (v1.3.2)
# ---------------------------------------------------------------------------


def test_speech_lookup_matches_linear_scan() -> None:
    rng = random.Random(42)
    intervals: list[tuple[float, float]] = []
    for _ in range(200):
        start = rng.uniform(0, 3600)
        intervals.append((start, start + rng.uniform(0.05, 2.0)))
    starts, ends = _merge_speech_intervals(intervals)
    for _ in range(2000):
        moment = rng.uniform(-1, 3602)
        expected = any(start <= moment <= end for start, end in intervals)
        assert _speech_active_at(starts, ends, moment) == expected
    assert _merge_speech_intervals(None) == ([], [])
    assert _merge_speech_intervals([]) == ([], [])


def test_analyze_source_face_track_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    reported: list[float] = []

    def fake_analyze(*args: Any, on_progress: Any = None, **kwargs: Any):
        if on_progress is not None:
            on_progress(25.0)
            on_progress(100.0)
        return [[FaceCropKeyframe(0.0, 0.0, 0.0)]]

    monkeypatch.setattr(face_tracking_module, "analyze_face_crop_paths", fake_analyze)
    monkeypatch.setattr(
        face_tracking_module, "_probe_video_dimensions", lambda _path: (1920, 1080)
    )
    has_face, _keyframes = analyze_source_face_track(
        video, 2.0, 1080, 1920, on_progress=reported.append
    )
    assert has_face is True
    assert reported == [25.0, 100.0]


def test_peek_cached_source_face_track(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video-bytes")
    calls: list[int] = []
    _, max_x, _max_y = _crop_geometry(1920, 1080, 1080, 1920)
    points = [
        FaceCropKeyframe(0.0, 0.0, 0.0),
        FaceCropKeyframe(2.0, max_x, 0.0),
    ]

    def spy(*args: Any, **kwargs: Any):
        calls.append(1)
        return [points]

    monkeypatch.setattr(
        face_tracking_module, "_analyze_face_crop_paths_uncached", spy
    )
    monkeypatch.setattr(
        face_tracking_module, "_probe_video_dimensions", lambda _path: (1920, 1080)
    )
    assert peek_cached_source_face_track(video, 2.0, 1080, 1920) is None
    has_face, keyframes = analyze_source_face_track(video, 2.0, 1080, 1920)
    assert has_face is True
    assert len(calls) == 1
    peeked = peek_cached_source_face_track(video, 2.0, 1080, 1920)
    assert peeked == (True, keyframes)
    assert len(calls) == 1  # el peek no re-analiza


# ---------------------------------------------------------------------------
# Degradación sin MediaPipe y salto del landmarker con un solo rostro
# ---------------------------------------------------------------------------


def _fake_face_detector_yn(face_count: int):
    """Sustituto de cv2.FaceDetectorYN que devuelve rostros fijos."""
    import numpy as np

    class FakeDetector:
        def setInputSize(self, _size: tuple[int, int]) -> None:
            return None

        def detect(self, frame):
            height, width = frame.shape[:2]
            detections = np.zeros((face_count, 15), dtype=np.float32)
            for index in range(face_count):
                detections[index, 0] = width * (0.15 + 0.5 * index)
                detections[index, 1] = height * 0.3
                detections[index, 2] = width * 0.2
                detections[index, 3] = height * 0.3
                detections[index, -1] = 0.9
            return face_count, detections

    class FakeFaceDetectorYN:
        @staticmethod
        def create(*_args: Any, **_kwargs: Any) -> FakeDetector:
            return FakeDetector()

    return FakeFaceDetectorYN


def test_face_tracking_skips_lip_analysis_with_single_face(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    # Con un solo rostro la selección del orador no cambia nada, así
    # que el landmarker no debe inicializarse siquiera (en un video de
    # una sola persona hablando ahorra una inferencia por muestra).
    monkeypatch.setattr(
        face_tracking_module,
        "_get_lip_landmarker",
        lambda: pytest.fail("Con un solo rostro no debe inicializarse el landmarker."),
    )
    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    tracks = analyze_face_crop_paths(
        _video_with_audio,
        0,
        [(0.0, 1.0)],
        1080,
        1920,
        speech_intervals=[(0.0, 1.0)],
    )
    assert tracks[0] is not None
    assert len(tracks[0]) > 0


def test_face_tracking_degrades_without_lip_model(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    attempts: list[int] = []

    def failing_landmarker():
        attempts.append(1)
        raise FaceTrackingError(
            "Falta MediaPipe para estimar al orador activo. Reinstala la aplicación."
        )

    monkeypatch.setattr(
        face_tracking_module, "_get_lip_landmarker", failing_landmarker
    )
    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(2))
    tracks = analyze_face_crop_paths(
        _video_with_audio,
        0,
        [(0.0, 1.0)],
        1080,
        1920,
        speech_intervals=[(0.0, 1.0)],
    )
    # Sin landmarker el seguimiento sigue con YuNet solo: hay
    # keyframes en vez de un error, y el landmarker solo se intenta
    # una vez (tras fallar, se desactiva para el resto del análisis).
    assert tracks[0] is not None
    assert len(tracks[0]) > 0
    assert len(attempts) == 1
