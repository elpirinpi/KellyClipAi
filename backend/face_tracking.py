from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FaceTrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceCropKeyframe:
    time: float
    x: float
    y: float


@dataclass(frozen=True)
class _Face:
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class _LipTrack:
    face: _Face
    aperture: float
    score: float
    timestamp: float


class _SpeakerActivityTracker:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.tracks: list[_LipTrack] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        measurements: list[tuple[_Face, float]],
        timestamp: float,
        speech_active: bool,
    ) -> _Face | None:
        current: list[_LipTrack] = []
        for face, aperture in measurements:
            previous = min(
                (
                    track
                    for track in self.tracks
                    if timestamp - track.timestamp <= 1.0
                ),
                key=lambda track: math.hypot(
                    (track.face.center[0] - face.center[0]) / self.width,
                    (track.face.center[1] - face.center[1]) / self.height,
                ),
                default=None,
            )
            if previous is not None:
                distance = math.hypot(
                    (previous.face.center[0] - face.center[0]) / self.width,
                    (previous.face.center[1] - face.center[1]) / self.height,
                )
            else:
                distance = float("inf")
            elapsed = timestamp - previous.timestamp if previous else 0.0
            score = (
                previous.score * math.exp(-max(elapsed, 0.0) / 0.75)
                if previous
                else 0.0
            )
            if previous is not None and distance <= 0.2 and speech_active:
                score += abs(aperture - previous.aperture)
            current.append(_LipTrack(face, aperture, score, timestamp))
        self.tracks = current
        speaker = max(current, key=lambda track: track.score, default=None)
        return speaker.face if speaker is not None and speaker.score >= 0.02 else None


def _get_lip_landmarker() -> tuple[Any, Any]:
    global _LIP_LANDMARKER
    if _LIP_LANDMARKER is not None:
        return _LIP_LANDMARKER
    model_path = Path(__file__).with_name("models") / _LIP_MODEL_FILENAME
    if not model_path.is_file():
        raise FaceTrackingError(
            "Falta el modelo de lectura labial incluido con la aplicación."
        )
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise FaceTrackingError(
            "Falta MediaPipe para estimar al orador activo. Reinstala la aplicación."
        ) from error
    with _LIP_LANDMARKER_LOCK:
        if _LIP_LANDMARKER is None:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=5,
            )
            _LIP_LANDMARKER = (
                mp,
                vision.FaceLandmarker.create_from_options(options),
            )
        return _LIP_LANDMARKER


def _lip_measurements(
    landmarks_by_face: list[list[Any]],
    faces: list[_Face],
    width: int,
    height: int,
) -> list[tuple[_Face, float]]:
    measurements: list[tuple[_Face, float]] = []
    for landmarks in landmarks_by_face:
        if len(landmarks) <= 291:
            continue
        xs = [float(landmark.x) * width for landmark in landmarks]
        ys = [float(landmark.y) * height for landmark in landmarks]
        landmark_face = _Face(
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
            1.0,
        )
        matched_face = max(
            faces,
            key=lambda face: _intersection_over_union(face, landmark_face),
            default=None,
        )
        if matched_face is None or _intersection_over_union(
            matched_face, landmark_face
        ) == 0:
            continue
        mouth_width = abs(xs[291] - xs[61])
        aperture = abs(ys[14] - ys[13]) / max(mouth_width, 1.0)
        measurements.append((matched_face, aperture))
    return measurements


_SAMPLE_RATE = 4.0
# Frecuencia de muestreo del seguimiento facial para el editor visual
# (v1.3.1, GET /api/projects/{id}/face-track): el render de clips
# muestrea a 4 fps, pero la pista del Studio cubre el video fuente
# completo y 2 fps bastan para interpolar el recorte en el navegador;
# a 4 fps un video de 1 h analizaría 14.400 fotogramas.
STUDIO_FACE_TRACK_SAMPLE_RATE = 2.0
# Dimensiones de salida por ratio para la pista del Studio. La
# geometría de recorte depende del ratio, así que cada ratio tiene su
# propia entrada en la caché (la clave incluye las dimensiones).
STUDIO_FACE_TRACK_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}
_MAX_DETECTOR_DIMENSION = 1280
_DROPOUT_HOLD_SECONDS = 1.5
_MAX_KEYFRAMES = 48
_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
_LIP_MODEL_FILENAME = "face_landmarker.task"
_LIP_LANDMARKER_LOCK = threading.Lock()
_LIP_LANDMARKER: tuple[Any, Any] | None = None

# --- Caché de análisis facial (Fase 2, B-R4) ---------------------------------
# El seguimiento facial muestrea el video a 4 fps con YuNet (+ MediaPipe
# si hay voz): es el paso más caro del render después del propio encode,
# y el editor lo repetía entero en cada re-render del mismo clip. Clave:
# identidad del archivo (ruta + tamaño + mtime) + inicio del clip +
# rangos retenidos + dimensiones de salida + intervalos de voz +
# frecuencia de muestreo (el render usa 4 fps y la pista del Studio
# 2 fps, sin colisionar). Si cambia cualquiera de ellos, la clave
# cambia y la caché se invalida sola. Los resultados son listas de
# FaceCropKeyframe (frozen): seguros de compartir entre renders.
# Acotada (LRU simple por orden de inserción).
_FACE_ANALYSIS_CACHE: dict[tuple[Any, ...], list[list[FaceCropKeyframe] | None]] = {}
_FACE_ANALYSIS_CACHE_LOCK = threading.Lock()
_FACE_ANALYSIS_CACHE_MAX_ENTRIES = 16


def _face_cache_key(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None,
    sample_rate: float = _SAMPLE_RATE,
) -> tuple[Any, ...] | None:
    try:
        stat = input_path.stat()
    except OSError:
        return None
    identity = (str(input_path.resolve()), stat.st_size, stat.st_mtime_ns)
    return (
        identity,
        round(clip_start, 3),
        tuple((round(start, 3), round(end, 3)) for start, end in intervals),
        output_width,
        output_height,
        tuple(
            (round(start, 3), round(end, 3)) for start, end in (speech_intervals or [])
        ),
        round(sample_rate, 3),
    )


def clear_face_analysis_cache() -> None:
    """Vacía la caché de análisis facial (tests)."""
    with _FACE_ANALYSIS_CACHE_LOCK:
        _FACE_ANALYSIS_CACHE.clear()


def _crop_geometry(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[float, float, float]:
    scale = max(output_width / source_width, output_height / source_height)
    max_x = max(0.0, source_width * scale - output_width)
    max_y = max(0.0, source_height * scale - output_height)
    return scale, max_x, max_y


def _crop_position(
    face_center: tuple[float, float],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    headroom: float = 0.42,
) -> tuple[float, float]:
    scale, max_x, max_y = _crop_geometry(
        source_width, source_height, output_width, output_height
    )
    x = min(max_x, max(0.0, face_center[0] * scale - output_width / 2))
    y = min(max_y, max(0.0, face_center[1] * scale - output_height * headroom))
    return x, y


def _crop_fraction(value: float, maximum: float) -> float:
    """Posición de recorte como fracción del desplazamiento máximo.

    Sin desplazamiento posible en ese eje (la fuente y la salida
    coinciden en esa dimensión), la ventana está centrada: 0.5. Se
    limita a [0, 1] por redondeo de coma flotante.
    """
    if maximum <= 0:
        return 0.5
    return min(1.0, max(0.0, value / maximum))


def _intersection_over_union(left: _Face, right: _Face) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


class _SubjectTracker:
    def __init__(self, width: int, height: int, crop_width: float | None = None) -> None:
        self.width = width
        self.height = height
        self.crop_width = crop_width or width * 0.5
        self.target: tuple[float, float] | None = None
        self.face: _Face | None = None
        self.last_seen: float | None = None
        self.last_update: float | None = None

    def reset(self) -> None:
        self.target = None
        self.face = None
        self.last_seen = None
        self.last_update = None

    def _select(self, faces: list[_Face], timestamp: float) -> _Face | None:
        recent = (
            self.face is not None
            and self.last_seen is not None
            and timestamp - self.last_seen <= _DROPOUT_HOLD_SECONDS
        )
        if not recent:
            return max(faces, key=lambda face: face.area * face.confidence, default=None)

        assert self.face is not None
        previous_center = self.face.center

        def distance(face: _Face) -> float:
            center = face.center
            return math.hypot(
                (center[0] - previous_center[0]) / self.width,
                (center[1] - previous_center[1]) / self.height,
            )

        candidate = min(
            faces,
            key=lambda face: distance(face)
            - 0.08 * _intersection_over_union(self.face, face)
            - 0.015 * face.confidence,
        )
        return candidate if distance(candidate) <= 0.28 else None

    def _move_toward(self, desired: tuple[float, float], timestamp: float) -> None:
        if self.target is None:
            self.target = desired
            self.last_update = timestamp
            return

        previous_update = (
            self.last_update if self.last_update is not None else timestamp
        )
        elapsed = max(0.0, timestamp - previous_update)
        if elapsed == 0:
            return
        delta_x = desired[0] - self.target[0]
        delta_y = desired[1] - self.target[1]
        if math.hypot(delta_x, delta_y) < self.width * 0.015:
            self.last_update = timestamp
            return

        response = 1.0 - math.exp(-elapsed / 0.35)
        move_x = delta_x * response
        move_y = delta_y * response
        distance = math.hypot(move_x, move_y)
        max_distance = max(self.crop_width * 0.8, 120.0) * elapsed
        if distance > max_distance:
            move_x *= max_distance / distance
            move_y *= max_distance / distance
        self.target = (self.target[0] + move_x, self.target[1] + move_y)
        self.last_update = timestamp

    def update(
        self,
        faces: list[_Face],
        timestamp: float,
        scene_changed: bool,
        preferred_face: _Face | None = None,
    ) -> tuple[float, float] | None:
        if scene_changed:
            self.reset()

        selected = preferred_face or (self._select(faces, timestamp) if faces else None)
        if selected is not None:
            self.face = selected
            self.last_seen = timestamp
            self._move_toward(selected.center, timestamp)
        elif self.last_seen is not None and timestamp - self.last_seen <= _DROPOUT_HOLD_SECONDS:
            self.last_update = timestamp
        else:
            self._move_toward((self.width / 2, self.height / 2), timestamp)
        return self.target


def _simplify_keyframes(
    points: list[FaceCropKeyframe],
    tolerance: float = 10.0,
) -> list[FaceCropKeyframe]:
    if len(points) <= 2:
        return points

    kept = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        first, last = points[start], points[end]
        duration = last.time - first.time
        if duration <= 0:
            continue
        largest_error = tolerance
        split_at: int | None = None
        for index in range(start + 1, end):
            point = points[index]
            fraction = (point.time - first.time) / duration
            expected_x = first.x + (last.x - first.x) * fraction
            expected_y = first.y + (last.y - first.y) * fraction
            error = max(abs(point.x - expected_x), abs(point.y - expected_y))
            if error > largest_error:
                largest_error = error
                split_at = index
        if split_at is not None:
            kept.add(split_at)
            pending.extend(((start, split_at), (split_at, end)))

    simplified = [points[index] for index in sorted(kept)]
    if len(simplified) <= _MAX_KEYFRAMES:
        return simplified
    return [
        simplified[round(index * (len(simplified) - 1) / (_MAX_KEYFRAMES - 1))]
        for index in range(_MAX_KEYFRAMES)
    ]


def analyze_face_crop_paths(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    sample_rate: float = _SAMPLE_RATE,
) -> list[list[FaceCropKeyframe] | None]:
    """Rutas de recorte facial, con caché por clip (B-R4).

    En un acierto de caché se devuelve el resultado guardado sin abrir
    el video ni cargar OpenCV/MediaPipe; se sigue notificando el 100 %
    de progreso para que el llamador no se quede esperando. La
    frecuencia de muestreo forma parte de la clave de caché: el render
    de clips usa 4 fps y la pista del Studio 2 fps sin colisionar.
    """
    if sample_rate <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    cache_key = _face_cache_key(
        input_path,
        clip_start,
        intervals,
        output_width,
        output_height,
        speech_intervals,
        sample_rate,
    )
    if cache_key is not None:
        with _FACE_ANALYSIS_CACHE_LOCK:
            cached = _FACE_ANALYSIS_CACHE.get(cache_key)
        if cached is not None:
            if on_progress is not None:
                on_progress(100.0)
            return cached
    result = _analyze_face_crop_paths_uncached(
        input_path,
        clip_start,
        intervals,
        output_width,
        output_height,
        speech_intervals=speech_intervals,
        on_progress=on_progress,
        sample_rate=sample_rate,
    )
    if cache_key is not None:
        with _FACE_ANALYSIS_CACHE_LOCK:
            _FACE_ANALYSIS_CACHE[cache_key] = result
            while len(_FACE_ANALYSIS_CACHE) > _FACE_ANALYSIS_CACHE_MAX_ENTRIES:
                _FACE_ANALYSIS_CACHE.pop(next(iter(_FACE_ANALYSIS_CACHE)))
    return result


def _probe_video_dimensions(input_path: Path) -> tuple[int, int]:
    """Dimensiones del video sin decodificar fotogramas (solo metadatos)."""
    try:
        import cv2
    except ImportError as error:
        raise FaceTrackingError(
            "Falta OpenCV para realizar el seguimiento facial. Reinstala la aplicación."
        ) from error
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise FaceTrackingError("No se pudo abrir el video para analizar los rostros.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width < 1 or height < 1:
        raise FaceTrackingError("No se pudieron leer las dimensiones del video.")
    return width, height


def analyze_source_face_track(
    input_path: Path,
    duration: float,
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    sample_rate: float = STUDIO_FACE_TRACK_SAMPLE_RATE,
) -> tuple[bool, list[FaceCropKeyframe]]:
    """Pista de seguimiento facial del video fuente completo (Studio, v1.3.1).

    Reutiliza ``analyze_face_crop_paths`` con un solo intervalo
    (0, duration) y clip_start=0, y convierte las posiciones de
    recorte en píxeles a FRACCIONES del desplazamiento máximo en cada
    eje (x = cropX / maxX; 0.5 si no hay desplazamiento posible en ese
    eje), para que el frontend las convierta a píxeles según su zoom.
    Devuelve (has_face, keyframes): tiempos absolutos del fuente,
    ascendentes, como mucho ``_MAX_KEYFRAMES`` (48) tras la
    simplificación. Sin rostro en todo el video — o si la fuente y la
    salida comparten aspecto y la ventana de recorte no puede moverse —
    devuelve (False, []).

    Caché: la misma caché en memoria del análisis facial (clave:
    identidad del archivo + intervalo + dimensiones de salida +
    intervalos de voz + frecuencia de muestreo), así que cada ratio
    tiene su propia entrada y la segunda llamada es instantánea. La
    caché se pierde al reiniciar el backend, igual que en el render.

    Latencia (estimación a partir del muestreo, sin mediciones en este
    entorno): a 2 fps, un video de 1 h son 7.200 muestras (una
    detección YuNet por muestra, más FaceLandmarker en las muestras
    con voz y rostro), y el muestreo secuencial decodifica
    prácticamente todos los fotogramas del video entre muestras
    (~108.000 a 30 fps). En una CPU modesta, del orden de 10–20 min
    para 1 h a 1080p, escalando linealmente con la duración (un video
    de 10 min, unos 2–4 min). La primera llamada es lenta; las
    siguientes, con la caché caliente, son instantáneas.
    """
    if duration <= 0:
        raise ValueError("La duración del video debe ser positiva.")
    paths = analyze_face_crop_paths(
        input_path,
        0.0,
        [(0.0, duration)],
        output_width,
        output_height,
        speech_intervals=speech_intervals,
        sample_rate=sample_rate,
    )
    points = paths[0] if paths else None
    if not points:
        return False, []
    source_width, source_height = _probe_video_dimensions(input_path)
    _, max_x, max_y = _crop_geometry(
        source_width, source_height, output_width, output_height
    )
    keyframes = [
        FaceCropKeyframe(
            point.time,
            _crop_fraction(point.x, max_x),
            _crop_fraction(point.y, max_y),
        )
        for point in points
    ]
    return True, keyframes


def _analyze_face_crop_paths_uncached(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    sample_rate: float = _SAMPLE_RATE,
) -> list[list[FaceCropKeyframe] | None]:
    model_path = Path(__file__).with_name("models") / _MODEL_FILENAME
    if not model_path.is_file():
        raise FaceTrackingError(
            "Falta el modelo de seguimiento facial incluido con la aplicación."
        )
    try:
        import cv2
    except ImportError as error:
        raise FaceTrackingError(
            "Falta OpenCV para realizar el seguimiento facial. Reinstala la aplicación."
        ) from error

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise FaceTrackingError("No se pudo abrir el video para analizar los rostros.")
    try:
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width < 1 or source_height < 1:
            raise FaceTrackingError("No se pudieron leer las dimensiones del video.")
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        sequential_sampling = math.isfinite(frame_rate) and frame_rate > 0
        scale, max_x, max_y = _crop_geometry(
            source_width, source_height, output_width, output_height
        )
        if max_x < 1 and max_y < 1:
            if on_progress is not None:
                on_progress(100.0)
            return [None for _ in intervals]

        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            0.55,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        media_pipe = None
        landmarker = None
        tracker = _SubjectTracker(
            source_width,
            source_height,
            output_width / scale,
        )
        speaker_tracker = _SpeakerActivityTracker(source_width, source_height)
        previous_thumbnail = None
        any_face_detected = False
        all_paths: list[list[FaceCropKeyframe] | None] = []
        total_samples = sum(
            max(1, math.ceil((interval_end - interval_start) * sample_rate))
            for interval_start, interval_end in intervals
        )
        if total_samples == 0:
            if on_progress is not None:
                on_progress(100.0)
            return []
        completed_samples = 0
        if on_progress is not None:
            on_progress(0.0)

        for interval_start, interval_end in intervals:
            interval_duration = interval_end - interval_start
            sample_count = max(1, math.ceil(interval_duration * sample_rate))
            if sequential_sampling:
                sequential_sampling = capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    (clip_start + interval_start) * 1000,
                )
            points: list[FaceCropKeyframe] = []
            previous_local_time: float | None = None
            for sample_index in range(sample_count):
                local_time = interval_start + interval_duration * sample_index / sample_count
                absolute_time = clip_start + local_time
                if sequential_sampling:
                    if previous_local_time is not None:
                        frame_step = max(
                            1,
                            round((local_time - previous_local_time) * frame_rate),
                        )
                        for _ in range(frame_step - 1):
                            if not capture.grab():
                                raise FaceTrackingError(
                                    f"No se pudo leer el video en el segundo {absolute_time:.2f}."
                                )
                else:
                    capture.set(cv2.CAP_PROP_POS_MSEC, absolute_time * 1000)
                ok, frame = capture.read()
                previous_local_time = local_time
                if not ok or frame is None:
                    raise FaceTrackingError(
                        f"No se pudo leer el video en el segundo {absolute_time:.2f}."
                    )

                frame_height, frame_width = frame.shape[:2]
                if frame_width != source_width or frame_height != source_height:
                    raise FaceTrackingError(
                        "Las dimensiones del video cambiaron durante el seguimiento facial."
                    )
                detector_scale = min(
                    1.0,
                    _MAX_DETECTOR_DIMENSION / max(frame_width, frame_height),
                )
                if detector_scale < 1:
                    detector_frame = cv2.resize(
                        frame,
                        (
                            round(frame_width * detector_scale),
                            round(frame_height * detector_scale),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    detector.setInputSize(
                        (detector_frame.shape[1], detector_frame.shape[0])
                    )
                else:
                    detector_frame = frame
                detector.setInputSize(
                    (detector_frame.shape[1], detector_frame.shape[0])
                )
                thumbnail = cv2.cvtColor(
                    cv2.resize(frame, (48, 27)), cv2.COLOR_BGR2GRAY
                )
                scene_changed = (
                    previous_thumbnail is not None
                    and float(cv2.absdiff(previous_thumbnail, thumbnail).mean()) >= 52.0
                )
                previous_thumbnail = thumbnail
                _, detections = detector.detect(detector_frame)
                faces = (
                    [
                        _Face(
                            float(detection[0]) / detector_scale,
                            float(detection[1]) / detector_scale,
                            float(detection[2]) / detector_scale,
                            float(detection[3]) / detector_scale,
                            float(detection[-1]),
                        )
                        for detection in detections
                    ]
                    if detections is not None
                    else []
                )
                any_face_detected |= bool(faces)
                preferred_face = None
                speech_active = any(
                    start <= absolute_time <= end
                    for start, end in speech_intervals or []
                )
                if scene_changed:
                    speaker_tracker.reset()
                measurements: list[tuple[_Face, float]] = []
                if speech_active and faces and speech_intervals:
                    if landmarker is None:
                        media_pipe, landmarker = _get_lip_landmarker()
                    if media_pipe is None:
                        raise FaceTrackingError(
                            "No se pudo inicializar el análisis del orador."
                        )
                    rgb_frame = cv2.cvtColor(detector_frame, cv2.COLOR_BGR2RGB)
                    image = media_pipe.Image(
                        image_format=media_pipe.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    with _LIP_LANDMARKER_LOCK:
                        result = landmarker.detect(image)
                    landmarks_by_face = result.face_landmarks or []
                    measurements = _lip_measurements(
                        landmarks_by_face,
                        faces,
                        frame_width,
                        frame_height,
                    )
                preferred_face = speaker_tracker.update(
                    measurements,
                    absolute_time,
                    speech_active,
                )
                target = tracker.update(
                    faces,
                    absolute_time,
                    scene_changed,
                    preferred_face=preferred_face,
                )
                if target is not None:
                    x, y = _crop_position(
                        target, source_width, source_height, output_width, output_height
                    )
                    point_time = local_time - interval_start
                    if scene_changed and points:
                        previous = points[-1]
                        hold_time = max(previous.time, point_time - 0.02)
                        if hold_time > previous.time:
                            points.append(
                                FaceCropKeyframe(hold_time, previous.x, previous.y)
                            )
                    points.append(
                        FaceCropKeyframe(point_time, x, y)
                    )
                completed_samples += 1
                if on_progress is not None:
                    on_progress(100 * completed_samples / total_samples)

            if points:
                if points[0].time > 0:
                    points.insert(
                        0,
                        FaceCropKeyframe(0.0, max_x / 2, max_y / 2),
                    )
                if points[-1].time < interval_duration:
                    last = points[-1]
                    points.append(
                        FaceCropKeyframe(interval_duration, last.x, last.y)
                    )
                all_paths.append(_simplify_keyframes(points))
            else:
                all_paths.append(None)

        return all_paths if any_face_detected else [None for _ in intervals]
    except cv2.error as error:
        raise FaceTrackingError(f"El detector facial no pudo analizar el video: {error}") from error
    finally:
        capture.release()
