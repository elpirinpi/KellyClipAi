"""Tests del hotfix v1.4.1 (bug 5): seguimiento facial fiable y rápido.

Regresiones que cubre:
- Un fotograma que no se lee (o una detección que falla) ya no tumba
  el análisis entero: la muestra se omite y el trabajo termina con
  resultado parcial (antes: FaceTrackingError en el primer fallo y
  503 para todo el video).
- Un video que no entrega ningún fotograma (p. ej. AV1 con cv2, que
  no lo decodifica) termina en un error claro, y la pista del Studio
  lo evita usando ffmpeg primero ("auto": keyframes → denso → cv2).
- Una excepción del landmarker de labios a mitad del análisis ya no
  tumba el trabajo: se sigue con YuNet solo.
- El análisis tiene plazo: con un presupuesto agotado termina en
  error claro en vez de quedarse colgado; y el perro guardián del
  trabajo falla un trabajo que deja de avanzar (antes quedaba en
  "pending" para siempre y el frontend sondeaba sin fin).
- La vía de keyframes por ffmpeg da (casi) la misma pista que la vía
  cv2 en un video con rostros (detector simulado): el muestreo más
  rápido no cambia el resultado.
- La altura de análisis que ffmpeg dará a ``scale=640:-2`` coincide
  con la sonda de un fotograma en varios tamaños (un alto erróneo
  desalineanría todo el pipe rawvideo).
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from test_face_track import _video_with_audio

from backend import face_tracking as face_tracking_module
from backend import media as media_module
from backend.face_tracking import (
    FaceCropKeyframe,
    FaceTrackingError,
    _FaceTrackCancelled,
    _ffmpeg_analysis_height,
    _ffmpeg_probe_analysis_height,
    _simplify_keyframes,
    analyze_face_crop_paths,
    analyze_source_face_track,
    clear_face_analysis_cache,
    clear_face_track_jobs,
    get_face_track_job,
    start_face_track_job,
)
from backend.media import render_clip
from backend.schemas import ClipProposal, EditOptions, ProjectConfig

# La fixture importada solo se usa como parámetro de test (pytest la
# resuelve por nombre); se reexporta para que ruff no la marque como
# import sin usar.
__all__ = ["_video_with_audio"]


@pytest.fixture(autouse=True)
def _clear_face_caches():
    clear_face_analysis_cache()
    clear_face_track_jobs()
    yield
    clear_face_analysis_cache()
    clear_face_track_jobs()


@pytest.fixture(scope="module")
def _video_with_keyframes(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Video negro de 2 s con un keyframe cada ~0,5 s (GOP 12 a 25 fps).

    La vía de keyframes necesita al menos dos keyframes: con uno
    solo, el filtro fps de ffmpeg no emite ningún fotograma (no
    puede calcular la duración del único fotograma) y la fuente
    queda vacía — en producción la cadena "auto" cae a la vía densa,
    pero para probar la vía de keyframes explícita hace falta este
    fixture.
    """
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg no está instalado")
    path = tmp_path_factory.mktemp("hotfix141-keyframes") / "source.mp4"
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
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "12",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _fake_face_detector_yn(face_count: int, fail_after: int | None = None):
    """Sustituto de cv2.FaceDetectorYN que devuelve rostros fijos.

    ``fail_after``: si se indica, detect() lanza a partir de esa
    llamada (para probar la tolerancia a fallos de detección).
    """
    import numpy as np

    calls = {"n": 0}

    class FakeDetector:
        def setInputSize(self, _size: tuple[int, int]) -> None:
            return None

        def detect(self, frame):
            calls["n"] += 1
            if fail_after is not None and calls["n"] > fail_after:
                raise RuntimeError("detector roto a mitad")
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


def _failing_capture_class(
    original_capture: Any,
    *,
    fail_reads_after: int | None = None,
    fail_grabs_after: int | None = None,
):
    """VideoCapture que deja de leer/grabar tras N operaciones."""

    class FailingVideoCapture:
        def __init__(self, path: str) -> None:
            self.capture = original_capture(path)
            self.reads = 0
            self.grabs = 0

        def __getattr__(self, name: str):
            return getattr(self.capture, name)

        def read(self):
            self.reads += 1
            if fail_reads_after is not None and self.reads > fail_reads_after:
                return False, None
            return self.capture.read()

        def grab(self):
            self.grabs += 1
            if fail_grabs_after is not None and self.grabs > fail_grabs_after:
                return False
            return self.capture.grab()

    return FailingVideoCapture


# ---------------------------------------------------------------------------
# Tolerancia a fallos por muestra (vía cv2)
# ---------------------------------------------------------------------------


def test_read_failure_mid_video_does_not_abort(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un video que deja de leerse a mitad da resultado parcial, no error.

    Antes: el primer read() fallido lanzaba FaceTrackingError y todo
    el análisis (y el trabajo del Studio) terminaba en 503. Ahora la
    muestra se omite y el intervalo se cierra con lo analizado hasta
    ahí. El fixture es un video negro sin rostros: el resultado es
    [None] (sin rostro), pero lo importante es que NO lanza.
    """
    import cv2

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        _failing_capture_class(cv2.VideoCapture, fail_reads_after=3),
    )
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920
    )
    assert tracks == [None]


def test_read_failure_mid_video_keeps_partial_track(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con rostros, un fallo de lectura a mitad conserva la pista parcial."""
    import cv2

    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        _failing_capture_class(cv2.VideoCapture, fail_reads_after=4),
    )
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920
    )
    assert tracks[0] is not None
    assert len(tracks[0]) > 0
    # La pista cubre hasta el fallo (último punto en 2.0 por el cierre
    # del intervalo, con la última posición conocida).
    assert tracks[0][-1].time == pytest.approx(2.0)


def test_grab_failure_mid_video_does_not_abort(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un grab() fallido a mitad (deriva de fps, fin de video) no tumba."""
    import cv2

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        _failing_capture_class(cv2.VideoCapture, fail_grabs_after=2),
    )
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920
    )
    assert tracks == [None]


def test_unreadable_video_raises_clear_error(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si NO se lee ningún fotograma, el error es claro (no un 503 opaco).

    Es el caso AV1 con cv2: la captura se abre (metadatos bien) pero
    ningún read() devuelve fotograma. El mensaje debe decir que no se
    pudo leer ningún fotograma, no solo "segundo 0.00".
    """
    import cv2

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        _failing_capture_class(cv2.VideoCapture, fail_reads_after=0),
    )
    with pytest.raises(FaceTrackingError, match="ningún fotograma"):
        analyze_face_crop_paths(_video_with_audio, 0, [(0.0, 2.0)], 1080, 1920)


def test_detect_failure_on_some_samples_does_not_abort(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El detector puede fallar en algunas muestras sin tumbar el análisis."""
    import cv2

    monkeypatch.setattr(
        cv2, "FaceDetectorYN", _fake_face_detector_yn(1, fail_after=2)
    )
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920
    )
    assert tracks[0] is not None
    assert len(tracks[0]) > 0


def test_detect_failure_on_all_samples_raises(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la detección falla en TODAS las muestras, es un error (no has_face=false).

    Un detector sistemáticamente roto no debe disfrazarse de "video
    sin rostros": el llamador debe enterarse.
    """
    import cv2

    monkeypatch.setattr(
        cv2, "FaceDetectorYN", _fake_face_detector_yn(1, fail_after=0)
    )
    with pytest.raises(FaceTrackingError, match="detector facial"):
        analyze_face_crop_paths(
            _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920
        )


def test_lip_landmarker_exception_mid_analysis_does_not_abort(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Una excepción del landmarker a mitad no tumba el trabajo.

    Antes solo se capturaba el fallo al CREAR el landmarker; un
    detect() que lanzara a mitad (fotograma raro, MediaPipe
    inestable) propagaba y mataba el análisis entero. Ahora se
    desactiva la detección de orador y se sigue con YuNet solo.
    """
    import cv2

    class ExplodingLandmarker:
        def detect(self, _image):
            raise RuntimeError("MediaPipe falló en este fotograma")

    monkeypatch.setattr(
        cv2, "FaceDetectorYN", _fake_face_detector_yn(2)
    )
    monkeypatch.setattr(
        face_tracking_module,
        "_get_lip_landmarker",
        lambda: (type("MP", (), {"Image": lambda **kwargs: None, "ImageFormat": type("F", (), {"SRGB": 0})})(), ExplodingLandmarker()),
    )
    # El landmarker necesita cvtColor + Image: el Image falso basta
    # porque detect() lanza antes de usarlo.
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


# ---------------------------------------------------------------------------
# Plazo y cancelación
# ---------------------------------------------------------------------------


def test_deadline_aborts_long_analysis(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con el presupuesto agotado, el análisis termina en error claro."""
    import cv2

    class SlowDetector:
        def setInputSize(self, _size: tuple[int, int]) -> None:
            return None

        def detect(self, frame):
            time.sleep(0.05)
            return 0, None

    class SlowFaceDetectorYN:
        @staticmethod
        def create(*_args: Any, **_kwargs: Any) -> SlowDetector:
            return SlowDetector()

    monkeypatch.setattr(cv2, "FaceDetectorYN", SlowFaceDetectorYN)
    with pytest.raises(FaceTrackingError, match="tiempo máximo"):
        analyze_face_crop_paths(
            _video_with_audio,
            0,
            [(0.0, 2.0)],
            1080,
            1920,
            deadline_seconds=0.05,
        )


def test_should_cancel_aborts_analysis(
    _video_with_audio: Path,
) -> None:
    """Una cancelación externa aborta el análisis con _FaceTrackCancelled."""
    calls = {"n": 0}

    def cancel_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(_FaceTrackCancelled):
        analyze_face_crop_paths(
            _video_with_audio,
            0,
            [(0.0, 2.0)],
            1080,
            1920,
            should_cancel=cancel_after_two,
        )


def test_invalid_frame_source_rejected(
    _video_with_audio: Path,
) -> None:
    with pytest.raises(ValueError, match="Fuente de fotogramas"):
        analyze_face_crop_paths(
            _video_with_audio, 0, [(0.0, 1.0)], 1080, 1920,
            frame_source="telepatía",
        )


# ---------------------------------------------------------------------------
# Perro guardián del trabajo (el job nunca queda colgado sin terminal)
# ---------------------------------------------------------------------------


def test_stalled_job_fails_with_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un trabajo que deja de avanzar termina en failed con mensaje claro.

    Antes: si el análisis se colgaba dentro de una llamada C, el
    trabajo quedaba en "pending" para siempre y el frontend sondeaba
    sin fin. Ahora el guardián lo marca como fallido y el siguiente
    GET puede reintentar.
    """
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_STALL_SECONDS", 0.3
    )
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_WATCHDOG_POLL_SECONDS", 0.05
    )
    gate = threading.Event()

    def stalled_runner(on_progress):
        # Nunca notifica progreso y tarda mucho: el guardián debe
        # fallar el trabajo sin esperar a que termine.
        gate.wait(timeout=10)
        return False, []

    job = start_face_track_job(("stall-test",), stalled_runner)
    assert job.status == "pending"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = get_face_track_job(("stall-test",))
        assert current is not None
        if current.status == "failed":
            break
        time.sleep(0.05)
    try:
        assert current.status == "failed"
        assert current.error is not None
        assert "sin avanzar" in current.error
    finally:
        gate.set()


def test_cancelled_worker_stops_at_next_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tras fallar por stall, el trabajador aborta en su siguiente progreso.

    El guardián marca el trabajo como cancelado; el análisis, al
    notificar progreso, recibe la cancelación como excepción y
    termina en vez de seguir consumiendo CPU en segundo plano.
    El mensaje que queda es el del guardián (la primera transición
    terminal gana).
    """
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_STALL_SECONDS", 0.3
    )
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_WATCHDOG_POLL_SECONDS", 0.05
    )
    stopped = threading.Event()

    def slow_runner(on_progress):
        on_progress(1.0)
        # Sin progreso durante más del umbral: el guardián falla el
        # trabajo mientras se duerme aquí.
        time.sleep(0.8)
        try:
            on_progress(2.0)
        except _FaceTrackCancelled:
            stopped.set()
            raise
        return False, []

    start_face_track_job(("cancel-test",), slow_runner)
    assert stopped.wait(timeout=5), "El trabajador no abortó al cancelarse."
    job = get_face_track_job(("cancel-test",))
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None and "sin avanzar" in job.error


def test_failed_job_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tras un trabajo fallido por stall, el siguiente intento funciona."""
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_STALL_SECONDS", 0.3
    )
    monkeypatch.setattr(
        face_tracking_module, "_FACE_TRACK_WATCHDOG_POLL_SECONDS", 0.05
    )
    gate = threading.Event()

    def stalled_runner(on_progress):
        gate.wait(timeout=10)
        return False, []

    start_face_track_job(("retry-test",), stalled_runner)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = get_face_track_job(("retry-test",))
        assert current is not None
        if current.status == "failed":
            break
        time.sleep(0.05)
    assert current.status == "failed"
    gate.set()
    # El endpoint olvida el trabajo fallido al entregar el 503; el
    # siguiente GET lanza un trabajo nuevo con la misma clave.
    face_tracking_module.forget_face_track_job(("retry-test",))

    def ok_runner(on_progress):
        on_progress(100.0)
        return True, [FaceCropKeyframe(0.0, 0.5, 0.5)]

    done = start_face_track_job(("retry-test",), ok_runner)
    deadline = time.monotonic() + 5
    while done.status == "pending" and time.monotonic() < deadline:
        time.sleep(0.05)
        done = get_face_track_job(("retry-test",))
        assert done is not None
    assert done.status == "done"
    assert done.result is not None and done.result[0] is True


# ---------------------------------------------------------------------------
# Vía ffmpeg: equivalencia con cv2 y reintentos
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"
)
def test_ffmpeg_keyframes_match_cv2_track(
    _video_with_keyframes: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La vía rápida (keyframes) da (casi) la misma pista que cv2.

    Detector simulado con un rostro fijo: ambas vías deben seguirlo
    igual. La rejilla de muestreo difiere (el filtro fps emite en
    tiempos de rejilla; cv2 muestrea en la rejilla del intervalo),
    así que se comparan las posiciones con tolerancia, no los
    tiempos exactos.
    """
    import cv2

    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    cv2_tracks = analyze_face_crop_paths(
        _video_with_keyframes, 0, [(0.0, 2.0)], 1080, 1920, frame_source="cv2"
    )
    clear_face_analysis_cache()
    ff_tracks = analyze_face_crop_paths(
        _video_with_keyframes,
        0,
        [(0.0, 2.0)],
        1080,
        1920,
        frame_source="ffmpeg-keyframes",
    )
    assert cv2_tracks[0] is not None
    assert ff_tracks[0] is not None
    # Mismo número de keyframes (el rostro no se mueve: la
    # simplificación deja los extremos en ambas vías).
    assert len(ff_tracks[0]) == len(cv2_tracks[0])
    for ff_point, cv2_point in zip(ff_tracks[0], cv2_tracks[0], strict=True):
        assert ff_point.x == pytest.approx(cv2_point.x, abs=2.0)
        assert ff_point.y == pytest.approx(cv2_point.y, abs=2.0)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"
)
def test_ffmpeg_dense_matches_cv2_track(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La vía densa por ffmpeg también coincide con cv2."""
    import cv2

    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    cv2_tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920, frame_source="cv2"
    )
    clear_face_analysis_cache()
    ff_tracks = analyze_face_crop_paths(
        _video_with_audio,
        0,
        [(0.0, 2.0)],
        1080,
        1920,
        frame_source="ffmpeg-dense",
    )
    assert cv2_tracks[0] is not None
    assert ff_tracks[0] is not None
    assert len(ff_tracks[0]) == len(cv2_tracks[0])
    for ff_point, cv2_point in zip(ff_tracks[0], cv2_tracks[0], strict=True):
        assert ff_point.x == pytest.approx(cv2_point.x, abs=2.0)
        assert ff_point.y == pytest.approx(cv2_point.y, abs=2.0)


def test_auto_falls_back_to_cv2_without_ffmpeg(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin ffmpeg instalado, "auto" cae a cv2 y el análisis funciona."""
    import cv2

    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920, frame_source="auto"
    )
    assert tracks[0] is not None
    assert len(tracks[0]) > 0


def test_auto_falls_back_when_ffmpeg_yields_nothing(
    _video_with_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si ffmpeg no entrega fotogramas, "auto" reintenta con cv2."""
    import cv2

    monkeypatch.setattr(cv2, "FaceDetectorYN", _fake_face_detector_yn(1))
    original_iter = face_tracking_module._FFmpegFrameSource.iter_interval

    def empty_iter(self, *args: Any, **kwargs: Any):
        # Como si ffmpeg terminara sin emitir ningún fotograma.
        if False:
            yield
        return

    monkeypatch.setattr(
        face_tracking_module._FFmpegFrameSource,
        "iter_interval",
        empty_iter,
    )
    tracks = analyze_face_crop_paths(
        _video_with_audio, 0, [(0.0, 2.0)], 1080, 1920, frame_source="auto"
    )
    assert tracks[0] is not None
    assert len(tracks[0]) > 0
    monkeypatch.setattr(
        face_tracking_module._FFmpegFrameSource,
        "iter_interval",
        original_iter,
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"
)
def test_source_face_track_uses_ffmpeg_on_real_video(
    _video_with_audio: Path,
) -> None:
    """La pista del Studio funciona en un video real sin rostros (vía auto)."""
    has_face, keyframes = analyze_source_face_track(
        _video_with_audio, 2.0, 1080, 1920
    )
    assert has_face is False
    assert keyframes == []


# ---------------------------------------------------------------------------
# Tope de keyframes de la simplificación (no borrar tramos enteros)
# ---------------------------------------------------------------------------


def test_simplify_keyframes_cap_keeps_long_excursion() -> None:
    """El tope de 48 keyframes no borra un tramo largo de la pista.

    Regresión (video de Kelly, 21-11-2026): la vía densa dejaba 200 s
    con la cara a la izquierda sin un solo keyframe, porque el tope
    elegía por índice uniforme y los puntos se concentraban en los
    tramos con movimiento. Ahora el tope se queda con los puntos de
    mayor error, así que la excursión larga se conserva.
    """
    points: list[FaceCropKeyframe] = []
    # 0-30 s: la cara se mueve mucho (muchos puntos de RDP).
    for i in range(61):
        points.append(FaceCropKeyframe(i * 0.5, 1000.0 + 300.0 * (i % 2), 0.0))
    # 30-260 s: la cara quieta a la izquierda (x=0).
    for i in range(61, 521):
        points.append(FaceCropKeyframe(i * 0.5, 0.0, 0.0))
    # 260-320 s: la cara se mueve mucho otra vez.
    for i in range(521, 641):
        points.append(FaceCropKeyframe(i * 0.5, 1000.0 + 300.0 * (i % 2), 0.0))
    simplified = _simplify_keyframes(points)
    assert len(simplified) <= 48
    assert simplified[0].time == 0.0
    assert simplified[-1].time == pytest.approx(320.0)
    # El tramo largo a la izquierda se conserva: interpolando la
    # pista simplificada en medio del tramo, la posición sigue
    # siendo la izquierda (x=0). Con el tope anterior (índice
    # uniforme) esos puntos se borraban y el frontend interpolaba
    # el centro (x≈1000) en todo ese tramo. Un tramo plano se
    # representa por sus extremos, así que se comprueba la
    # interpolación, no puntos interiores.
    def _interp_at(target: float) -> float:
        for first, second in zip(simplified, simplified[1:], strict=True):
            if first.time <= target <= second.time:
                if second.time == first.time:
                    return second.x
                frac = (target - first.time) / (second.time - first.time)
                return first.x + (second.x - first.x) * frac
        raise AssertionError(f"Sin cobertura en t={target}")

    assert _interp_at(100.0) == pytest.approx(0.0)
    assert _interp_at(200.0) == pytest.approx(0.0)


def test_simplify_keyframes_keeps_endpoints_under_cap() -> None:
    """Bajo el tope, la simplificación conserva extremos y orden."""
    points = [FaceCropKeyframe(i * 0.5, 500.0 + 100.0 * (i % 3), 0.0)
              for i in range(200)]
    simplified = _simplify_keyframes(points)
    assert len(simplified) <= 48
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
    times = [p.time for p in simplified]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Altura de análisis de ffmpeg
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg no está instalado"
)
@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1920, 1080),
        (1280, 720),
        (1000, 563),
        (720, 1280),
        (1920, 800),
        (854, 480),
        (640, 480),
    ],
)
def test_ffmpeg_probe_height_matches_formula(
    width: int, height: int, tmp_path: Path
) -> None:
    """La sonda de ffmpeg y la fórmula coinciden en tamaños habituales.

    Genera un video sintético del tamaño dado y compara el alto que
    ffmpeg realmente produce con scale=640:-2 contra la fórmula de
    respaldo. (En empates exactos impares la fórmula puede diferir
    en 2 px: por eso la sonda manda en producción.)
    """
    import subprocess

    video = tmp_path / f"test_{width}x{height}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi",
            "-i", f"testsrc2=size={width}x{height}:rate=1:duration=0.2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video),
        ],
        check=True,
        timeout=60,
    )
    probed = _ffmpeg_probe_analysis_height(
        video, "ffmpeg", width, height
    )
    assert probed == _ffmpeg_analysis_height(width, height)
    assert probed % 2 == 0


def test_render_uses_ffmpeg_dense_for_face_tracking(
    _video_with_audio: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regresión (hotfix v1.4.1, bug 5): el render pedía el análisis
    facial con la fuente por defecto (cv2), que no decodifica AV1 y
    tumbaba el render con MediaProcessingError. Ahora usa
    "ffmpeg-dense": mismo muestreo denso y labios que cv2, con un
    decodificador que sí maneja AV1."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    captured: dict[str, object] = {}

    def _fake_analyze(*args: Any, **kwargs: Any) -> list[None]:
        captured.update(kwargs)
        return [None]

    monkeypatch.setattr(media_module, "analyze_face_crop_paths", _fake_analyze)
    proposal = ClipProposal(
        inicio=0,
        fin=2,
        titulo="Render",
        puntuacion=7,
        razon_viral="Verifica la fuente del render.",
    )
    render_clip(
        _video_with_audio,
        tmp_path / "render.mp4",
        tmp_path / "render.ass",
        proposal,
        ProjectConfig(
            framing="speaker",
            options=EditOptions(face_tracking=True),
        ),
        [],
        ffmpeg,
        lambda _progress: None,
    )
    assert captured.get("frame_source") == "ffmpeg-dense"
