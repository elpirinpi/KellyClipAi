from __future__ import annotations

import bisect
import logging
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FaceTrackingError(RuntimeError):
    pass


class _FrameSourceError(FaceTrackingError):
    """La fuente de fotogramas no entregó ningún fotograma.

    Es reintentable con otra fuente (p. ej. ffmpeg no está instalado
    o no pudo decodificar el video: se reintenta con cv2). Solo se
    lanza cuando la fuente no produjo NADA; un fallo a mitad de un
    análisis con muestras ya leídas se trata como fin de flujo, no
    como error de fuente.
    """


class _FaceTrackCancelled(FaceTrackingError):
    """El análisis se canceló (el trabajo se marcó como cancelado)."""


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

# --- Fuentes de fotogramas por ffmpeg (hotfix v1.4.1, bug 5) -----------------
# El camino cv2.VideoCapture NO decodifica AV1 con el ffmpeg que trae
# opencv-contrib-python (medido 21-11-2026 en el video de Kelly, AV1
# 1080p60 320 s: 0 fotogramas leídos, "Missing Sequence Header", y el
# análisis entero fallaba en el segundo 0.00 con 503). KellyAiClip
# aprendió lo mismo en su v1.1.2 ("nunca usar cv2.VideoCapture, que se
# congela con algunos MP4 en Windows") y extrae los fotogramas con
# ffmpeg; este módulo hace lo mismo para la pista del Studio
# (frame_source="auto": keyframes → denso → cv2). El render sigue con
# cv2 por defecto (sin cambios de comportamiento).
#
# - "ffmpeg-keyframes": `-skip_frame nokey` + filtro `fps=` a la
#   frecuencia de muestreo: ffmpeg solo decodifica los keyframes y el
#   filtro duplica el último para rellenar la rejilla (2 fps). En el
#   video de Kelly: 626 fotogramas (51 únicos) en 4,7 s, frente a
#   53,3 s de la vía densa y al fallo total de cv2. Los duplicados se
#   detectan por miniatura idéntica (diff 0.0 exacto) y reutilizan
#   los rostros sin volver a detectar, así YuNet solo corre sobre los
#   fotogramas únicos (~51 × 32 ms ≈ 1,6 s en esta VM). La rejilla
#   sigue siendo 2 fps: el tracker, el hold de dropout y la respuesta
#   del frontend no cambian; la selección de orador por labios SÍ se
#   desactiva en esta fuente (muestras únicas separadas por segundos:
#   la ventana de 1,0 s del tracker de labios nunca encadena y la
#   inferencia sería gasto inútil; se sigue la cara de mayor área,
#   como KellyAiClip).
# - "ffmpeg-dense": `-skip_frame noref -lowres 1` (la vía rápida ya
#   probada en visual_signals.py, 3,3x en H.264): decodifica solo
#   fotogramas de referencia. Respaldo cuando el video tiene muy
#   pocos keyframes (menos de 1 por cada 15 s: el recorte quedaría
#   casi estático) o cuando la vía de keyframes no entrega nada.
# - "cv2": el camino original (VideoCapture secuencial). Último
#   recurso: sin ffmpeg instalado, o si ambas vías ffmpeg fallan.
_FFMPEG_ANALYSIS_WIDTH = 640  # Ancho de análisis (KellyAiClip usa 640).
# Mínimo de muestras únicas para aceptar la vía de keyframes: 1 por
# cada 15 s de video (y al menos 1 en videos cortos). Por debajo, el
# recorte quedaría casi estático y se reintenta con muestreo denso.
_KEYFRAME_MIN_SECONDS_PER_SAMPLE = 15.0
# Fallos de lectura consecutivos tolerados antes de dar por terminado
# un intervalo (un fotograma corrupto no tumba el análisis; el fin
# del video también llega así cuando la duración de metadatos supera
# a los fotogramas reales).
_MAX_CONSECUTIVE_READ_FAILURES = 5

# --- Plazos del seguimiento facial (hotfix v1.4.1, bug 5) --------------------
# Antes no había ningún plazo: si la decodificación o el detector se
# quedaban colgados dentro de una llamada C, el trabajo quedaba en
# "pending" para siempre y el frontend sondeaba sin fin. Ahora hay
# dos redes: (1) un presupuesto total proporcional a la duración del
# video, comprobado en cada muestra (y con un temporizador que mata
# el ffmpeg si se cuelga en la lectura del pipe); (2) un perro
# guardián por trabajo que lo marca como fallido si pasa demasiado
# tiempo sin avanzar el progreso (ver _run_face_track_job). Ambos
# terminan en 503 con mensaje claro y permiten reintento en el
# siguiente GET (el endpoint olvida los trabajos fallidos).
_FACE_TRACK_DEADLINE_MIN_SECONDS = 180.0
_FACE_TRACK_DEADLINE_SECONDS_PER_MINUTE = 60.0
_FACE_TRACK_STALL_SECONDS = 120.0
# Intervalo del perro guardián entre comprobaciones.
_FACE_TRACK_WATCHDOG_POLL_SECONDS = 5.0
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


def _merge_speech_intervals(
    intervals: list[tuple[float, float]] | None,
) -> tuple[list[float], list[float]]:
    """Intervalos de habla fusionados y ordenados, para búsqueda binaria.

    ``speech_active`` se evalúa en cada muestra; con la transcripción
    de un video largo (miles de palabras) el escaneo lineal por muestra
    es O(palabras × muestras). Fusionar intervalos solapados o
    adyacentes no cambia el resultado booleano y deja la consulta en
    O(log n) con ``bisect`` (ver ``_speech_active_at``).
    """
    if not intervals:
        return [], []
    starts: list[float] = []
    ends: list[float] = []
    for start, end in sorted(intervals):
        if ends and start <= ends[-1]:
            ends[-1] = max(ends[-1], end)
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def _speech_active_at(starts: list[float], ends: list[float], moment: float) -> bool:
    """True si `moment` cae dentro de algún intervalo fusionado.

    Los intervalos fusionados son disjuntos y están ordenados, así que
    basta con el último que empieza en o antes de `moment`: si
    `moment` supera su fin, tampoco está dentro de ningún anterior.
    """
    index = bisect.bisect_right(starts, moment) - 1
    return index >= 0 and moment <= ends[index]


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
    """Simplificación Ramer-Douglas-Peucker, con tope de keyframes.

    El tope (``_MAX_KEYFRAMES``) se aplica quedándose con los puntos
    de MAYOR error de simplificación (los extremos siempre se
    conservan): quitar un punto de poco error apenas cambia la
    pista, así que quedarse con los de mayor error minimiza la
    desviación que introduce el tope. El tope anterior elegía por
    índice uniforme y podía borrar tramos enteros de la pista
    (medido el 21-11-2026 en el video de Kelly: la vía densa dejaba
    200 s con la cara a la izquierda sin un solo keyframe, y el
    frontend interpolaba el centro en todo ese tramo).
    """
    if len(points) <= 2:
        return points

    # Índice -> error con el que se añadió como punto de corte
    # (infinito en los extremos: siempre se conservan).
    kept: dict[int, float] = {0: math.inf, len(points) - 1: math.inf}
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
            kept[split_at] = largest_error
            pending.extend(((start, split_at), (split_at, end)))

    simplified = [points[index] for index in sorted(kept)]
    if len(simplified) <= _MAX_KEYFRAMES:
        return simplified
    ranked = sorted(kept, key=lambda index: -kept[index])
    chosen = sorted(ranked[:_MAX_KEYFRAMES])
    return [points[index] for index in chosen]


def analyze_face_crop_paths(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    sample_rate: float = _SAMPLE_RATE,
    frame_source: str = "cv2",
    deadline_seconds: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[list[FaceCropKeyframe] | None]:
    """Rutas de recorte facial, con caché por clip (B-R4).

    En un acierto de caché se devuelve el resultado guardado sin abrir
    el video ni cargar OpenCV/MediaPipe; se sigue notificando el 100 %
    de progreso para que el llamador no se quede esperando. La
    frecuencia de muestreo forma parte de la clave de caché: el render
    de clips usa 4 fps y la pista del Studio 2 fps sin colisionar.
    La fuente de fotogramas NO forma parte de la clave: las fuentes
    son equivalentes por construcción (mismo detector, mismo
    tracker), así que un resultado calculado por una vía vale para
    la otra.

    ``frame_source``: "cv2" (VideoCapture, por defecto: el render no
    cambia de comportamiento), "ffmpeg-keyframes" (solo keyframes,
    vía rápida), "ffmpeg-dense" (fotogramas de referencia) o "auto"
    (keyframes → denso → cv2, con reintento al fallar una vía; la
    usa la pista del Studio). ``deadline_seconds`` limita el tiempo
    total del análisis (por defecto, un presupuesto proporcional a
    la duración: 60 s por minuto de video, mínimo 3 min); superado,
    lanza FaceTrackingError con un mensaje claro en vez de quedarse
    colgado. ``should_cancel`` se consulta en cada muestra.
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
        frame_source=frame_source,
        deadline_seconds=deadline_seconds,
        should_cancel=should_cancel,
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


def _points_to_keyframes(
    points: list[FaceCropKeyframe],
    input_path: Path,
    output_width: int,
    output_height: int,
) -> list[FaceCropKeyframe]:
    """Posiciones de recorte en píxeles → fracciones del desplazamiento máximo."""
    source_width, source_height = _probe_video_dimensions(input_path)
    _, max_x, max_y = _crop_geometry(
        source_width, source_height, output_width, output_height
    )
    return [
        FaceCropKeyframe(
            point.time,
            _crop_fraction(point.x, max_x),
            _crop_fraction(point.y, max_y),
        )
        for point in points
    ]


def analyze_source_face_track(
    input_path: Path,
    duration: float,
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    sample_rate: float = STUDIO_FACE_TRACK_SAMPLE_RATE,
    on_progress: Callable[[float], None] | None = None,
    frame_source: str = "auto",
    deadline_seconds: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
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

    Fuente de fotogramas (hotfix v1.4.1, bug 5): "auto" por defecto —
    keyframes por ffmpeg, denso por ffmpeg, cv2 como último recurso
    (ver las constantes de fuentes arriba). El camino cv2 NO
    decodifica AV1 con el ffmpeg que trae opencv-contrib-python: en
    el video de Kelly (AV1 1080p60, 320 s) el análisis entero fallaba
    en el segundo 0.00 con 503; con la vía de keyframes tarda ~7 s
    en esta VM (extracción 4,7 s + YuNet sobre 51 fotogramas únicos).
    La rejilla sigue siendo 2 fps; en la vía de keyframes la
    selección de orador por labios se desactiva (muestras únicas
    separadas por segundos) y se sigue la cara de mayor área.

    Latencia (medida en este entorno, VM de 1,5 CPU, video de Kelly
    AV1 1080p60 320 s): vía de keyframes ~7 s en total; vía densa
    ~75 s (decodificación 53 s + YuNet 20 s); vía cv2: FALLO (no
    decodifica AV1). En H.264 la vía cv2 tardaba ~17 s/min. El
    endpoint no ejecuta el análisis en la petición: lo lanza en un
    trabajo en segundo plano con progreso (ver
    ``start_face_track_job``) y el frontend sondea hasta que está
    lista. ``on_progress`` recibe el porcentaje (0–100) y alimenta
    ese progreso; en un acierto de caché se notifica el 100 %.
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
        on_progress=on_progress,
        sample_rate=sample_rate,
        frame_source=frame_source,
        deadline_seconds=deadline_seconds,
        should_cancel=should_cancel,
    )
    points = paths[0] if paths else None
    if not points:
        return False, []
    return True, _points_to_keyframes(points, input_path, output_width, output_height)


def peek_cached_source_face_track(
    input_path: Path,
    duration: float,
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    sample_rate: float = STUDIO_FACE_TRACK_SAMPLE_RATE,
) -> tuple[bool, list[FaceCropKeyframe]] | None:
    """Resultado cacheado de la pista del Studio, sin analizar.

    Devuelve None si no está en caché: el llamador debe iniciar el
    análisis en segundo plano. Falla abierto: cualquier error al
    convertir el resultado cacheado devuelve None en vez de
    propagarse, para que el análisis real informe del fallo con su
    propio error (503 en el endpoint).
    """
    if duration <= 0:
        return None
    cache_key = _face_cache_key(
        input_path,
        0.0,
        [(0.0, duration)],
        output_width,
        output_height,
        speech_intervals,
        sample_rate,
    )
    if cache_key is None:
        return None
    with _FACE_ANALYSIS_CACHE_LOCK:
        cached = _FACE_ANALYSIS_CACHE.get(cache_key)
    if cached is None:
        return None
    points = cached[0] if cached else None
    if not points:
        return False, []
    try:
        return True, _points_to_keyframes(points, input_path, output_width, output_height)
    except Exception:  # noqa: BLE001 - el peek nunca debe tumbar la petición
        logger.debug("No se pudo leer la pista facial cacheada.", exc_info=True)
        return None


# --- Trabajos de seguimiento facial del Studio (v1.3.2) ---------------------
# La pista cubre el video fuente entero (minutos en videos largos), así
# que el endpoint la calcula en un hilo en segundo plano y el frontend
# sondea con GET hasta que está lista. Un solo vuelo por clave: varias
# peticiones del mismo proyecto+ratio comparten el mismo trabajo en vez
# de lanzar análisis duplicados que se pelean por la CPU. La clave la
# construye el endpoint e incluye la identidad del archivo fuente, así
# que reemplazar el video invalida el trabajo anterior. Los trabajos
# terminados se conservan (acotado) para que un GET posterior devuelva
# el resultado sin re-analizar; los fallidos se olvidan al entregarse
# el error, para que el siguiente GET reintente.
FaceTrackJobKey = tuple[Any, ...]


@dataclass
class FaceTrackJob:
    """Estado de un trabajo de seguimiento facial (hilo en segundo plano)."""

    status: str = "pending"  # "pending" | "done" | "failed"
    progress: float = 0.0
    result: tuple[bool, list[FaceCropKeyframe]] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        # Perro guardián (hotfix v1.4.1, bug 5): si el análisis se
        # cuelga dentro de una llamada C (decodificación, detector),
        # ningún callback llega y el trabajo quedaría en "pending"
        # para siempre. El guardián marca el trabajo como cancelado y
        # fallido cuando pasa demasiado tiempo sin avanzar el
        # progreso; el hilo trabajador, al llegar a su siguiente
        # notificación de progreso, ve la cancelación y termina.
        self.cancelled = threading.Event()
        self.started_at = time.monotonic()
        self._last_progress_at = self.started_at

    def snapshot(self) -> FaceTrackJob:
        """Copia del estado actual (lectura segura desde otro hilo)."""
        with self._lock:
            return FaceTrackJob(
                status=self.status,
                progress=self.progress,
                result=self.result,
                error=self.error,
            )

    def update_progress(self, value: float) -> None:
        with self._lock:
            self.progress = min(100.0, max(0.0, float(value)))
            self._last_progress_at = time.monotonic()

    def seconds_without_progress(self) -> float:
        """Segundos desde la última notificación de progreso (o el inicio)."""
        with self._lock:
            return time.monotonic() - self._last_progress_at

    def is_pending(self) -> bool:
        with self._lock:
            return self.status == "pending"

    def complete(self, result: tuple[bool, list[FaceCropKeyframe]]) -> None:
        # Solo la primera transición terminal cuenta: si el guardián
        # ya marcó el trabajo como fallido, un resultado tardío no lo
        # resucita (el frontend ya recibió el 503 y puede reintentar).
        with self._lock:
            if self.status != "pending":
                return
            self.result = result
            self.progress = 100.0
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            if self.status != "pending":
                return
            self.error = error
            self.status = "failed"


_FACE_TRACK_JOBS: dict[FaceTrackJobKey, FaceTrackJob] = {}
_FACE_TRACK_JOBS_LOCK = threading.Lock()
_FACE_TRACK_JOBS_MAX_ENTRIES = 32


def _prune_face_track_jobs_locked() -> None:
    """Acota el registro (solo se expulsan trabajos terminados)."""
    while len(_FACE_TRACK_JOBS) > _FACE_TRACK_JOBS_MAX_ENTRIES:
        for key, job in _FACE_TRACK_JOBS.items():
            if job.snapshot().status != "pending":
                del _FACE_TRACK_JOBS[key]
                break
        else:
            break


def _watch_face_track_job(job: FaceTrackJob) -> None:
    """Perro guardián: falla el trabajo si se queda sin avanzar.

    Corre en un hilo demonio por trabajo y termina solo cuando el
    trabajo deja de estar pendiente. Si el hilo trabajador está
    atascado dentro de una llamada C, no se le puede interrumpir:
    el trabajo se marca como fallido (el frontend recibe el 503 y
    puede reintentar) y el hilo atascado queda como demonio hasta
    que su llamada regrese o el proceso termine; en su siguiente
    notificación de progreso vería la cancelación y abortaría.
    """
    while job.is_pending():
        time.sleep(_FACE_TRACK_WATCHDOG_POLL_SECONDS)
        if not job.is_pending():
            return
        if job.seconds_without_progress() > _FACE_TRACK_STALL_SECONDS:
            job.cancelled.set()
            job.fail(
                "El seguimiento facial se detuvo sin avanzar durante "
                f"{int(_FACE_TRACK_STALL_SECONDS // 60)} minutos. "
                "Vuelve a intentarlo; si el problema continúa, el video "
                "podría estar dañado o en un formato difícil de analizar."
            )
            return


def _run_face_track_job(
    job: FaceTrackJob,
    runner: Callable[
        [Callable[[float], None]], tuple[bool, list[FaceCropKeyframe]]
    ],
) -> None:
    threading.Thread(
        target=_watch_face_track_job,
        args=(job,),
        name="face-track-watchdog",
        daemon=True,
    ).start()

    def _on_progress(value: float) -> None:
        # La cancelación se propaga como excepción desde el propio
        # callback: el análisis la recibe en su siguiente muestra y
        # aborta sin dejar el trabajo colgado. Si el guardián ya
        # falló el trabajo, fail() guarda ese mensaje (el primero).
        if job.cancelled.is_set():
            raise _FaceTrackCancelled(
                "El seguimiento facial se canceló porque se detuvo sin avanzar."
            )
        job.update_progress(value)

    try:
        result = runner(_on_progress)
    except FaceTrackingError as error:
        job.fail(str(error))
    except Exception:  # noqa: BLE001 - un fallo inesperado no debe dejar el trabajo colgado
        logger.exception("El seguimiento facial terminó con un error inesperado.")
        job.fail("No se pudo calcular el seguimiento facial.")
    else:
        job.complete(result)


def get_face_track_job(key: FaceTrackJobKey) -> FaceTrackJob | None:
    """Trabajo existente para la clave, o None (sin crearlo)."""
    with _FACE_TRACK_JOBS_LOCK:
        job = _FACE_TRACK_JOBS.get(key)
    return job.snapshot() if job is not None else None


def start_face_track_job(
    key: FaceTrackJobKey,
    runner: Callable[
        [Callable[[float], None]], tuple[bool, list[FaceCropKeyframe]]
    ],
) -> FaceTrackJob:
    """Inicia (o reutiliza) el trabajo de seguimiento para `key`.

    Un solo vuelo: si ya hay un trabajo para la clave (pendiente,
    terminado o fallido), se devuelve su estado en vez de lanzar otro
    análisis. El trabajo nuevo corre en un hilo demonio; el llamador
    sondea con ``get_face_track_job`` (o repitiendo esta llamada).
    """
    with _FACE_TRACK_JOBS_LOCK:
        existing = _FACE_TRACK_JOBS.get(key)
        if existing is not None:
            return existing.snapshot()
        job = FaceTrackJob()
        _FACE_TRACK_JOBS[key] = job
        _prune_face_track_jobs_locked()
    threading.Thread(
        target=_run_face_track_job,
        args=(job, runner),
        name="face-track",
        daemon=True,
    ).start()
    return job.snapshot()


def forget_face_track_job(key: FaceTrackJobKey) -> None:
    """Olvida un trabajo (tras entregar su error, para que el siguiente GET reintente)."""
    with _FACE_TRACK_JOBS_LOCK:
        _FACE_TRACK_JOBS.pop(key, None)


def clear_face_track_jobs() -> None:
    """Vacía el registro de trabajos (tests)."""
    with _FACE_TRACK_JOBS_LOCK:
        _FACE_TRACK_JOBS.clear()


def _check_deadline_or_cancel(
    deadline: float,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Aborta el análisis si se canceló o superó el presupuesto de tiempo."""
    if should_cancel is not None and should_cancel():
        raise _FaceTrackCancelled("El seguimiento facial se canceló.")
    if time.monotonic() > deadline:
        raise FaceTrackingError(
            "El seguimiento facial superó el tiempo máximo permitido para "
            "este video. Vuelve a intentarlo."
        )


@dataclass
class _Sample:
    """Un fotograma muestreado: tiempos y fotograma a resolución de análisis.

    ``frame`` es None cuando el fotograma no se pudo leer (muestra
    fallida: se cuenta para el progreso y se sigue con la siguiente,
    en vez de tumbar el análisis entero como antes). ``local_time``
    es el tiempo dentro del intervalo (0..duración).
    """

    absolute_time: float
    local_time: float
    frame: Any | None


class _Cv2FrameSource:
    """Muestreo con cv2.VideoCapture (camino original, render).

    Una sola captura para todos los intervalos, con avance secuencial
    (grab) entre muestras y reposicionamiento por intervalo. Los
    fotogramas se entregan ya reducidos a la resolución de análisis
    (como mucho ``_MAX_DETECTOR_DIMENSION`` en el lado mayor) y
    ``coord_scale`` convierte las coordenadas del detector a
    coordenadas de la fuente.
    """

    kind = "cv2"
    lip_tracking = True

    def __init__(self, input_path: Path, cv2_module: Any) -> None:
        self._cv2 = cv2_module
        self.capture = cv2_module.VideoCapture(str(input_path))
        if not self.capture.isOpened():
            self.capture.release()
            raise FaceTrackingError(
                "No se pudo abrir el video para analizar los rostros."
            )
        self.source_width = int(
            self.capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)
        )
        self.source_height = int(
            self.capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)
        )
        if self.source_width < 1 or self.source_height < 1:
            self.capture.release()
            raise FaceTrackingError(
                "No se pudieron leer las dimensiones del video."
            )
        frame_rate = float(self.capture.get(cv2_module.CAP_PROP_FPS))
        self._sequential = math.isfinite(frame_rate) and frame_rate > 0
        self._frame_rate = frame_rate
        detector_scale = min(
            1.0,
            _MAX_DETECTOR_DIMENSION / max(self.source_width, self.source_height),
        )
        self._detector_scale = detector_scale
        if detector_scale < 1:
            self.analysis_width = round(self.source_width * detector_scale)
            self.analysis_height = round(self.source_height * detector_scale)
        else:
            self.analysis_width = self.source_width
            self.analysis_height = self.source_height
        self.coord_scale = 1.0 / detector_scale
        self.last_error_detail = ""

    def close(self) -> None:
        self.capture.release()

    def iter_interval(
        self,
        clip_start: float,
        interval_start: float,
        interval_end: float,
        sample_rate: float,
    ) -> Iterator[_Sample]:
        interval_duration = interval_end - interval_start
        sample_count = max(1, math.ceil(interval_duration * sample_rate))
        if self._sequential:
            self._sequential = self.capture.set(
                self._cv2.CAP_PROP_POS_MSEC,
                (clip_start + interval_start) * 1000,
            )
        previous_local_time: float | None = None
        for sample_index in range(sample_count):
            local_time = (
                interval_start
                + interval_duration * sample_index / sample_count
            )
            absolute_time = clip_start + local_time
            frame = None
            if self._sequential:
                ok = True
                if previous_local_time is not None:
                    frame_step = max(
                        1,
                        round((local_time - previous_local_time) * self._frame_rate),
                    )
                    for _ in range(frame_step - 1):
                        if not self.capture.grab():
                            ok = False
                            break
                if ok:
                    ok, raw = self.capture.read()
                    frame = raw if ok else None
            else:
                self.capture.set(
                    self._cv2.CAP_PROP_POS_MSEC, absolute_time * 1000
                )
                ok, raw = self.capture.read()
                frame = raw if ok else None
            previous_local_time = local_time
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
                if (frame_width, frame_height) != (
                    self.source_width,
                    self.source_height,
                ):
                    # Dimensiones inesperadas a mitad del video: la
                    # muestra se descarta en vez de tumbar el análisis.
                    frame = None
                elif self._detector_scale < 1:
                    try:
                        frame = self._cv2.resize(
                            frame,
                            (self.analysis_width, self.analysis_height),
                            interpolation=self._cv2.INTER_AREA,
                        )
                    except self._cv2.error:
                        frame = None
            yield _Sample(absolute_time, local_time - interval_start, frame)


def _ffmpeg_analysis_height(source_width: int, source_height: int) -> int:
    """Alto de análisis para ``scale=640:-2`` (respaldo de la sonda).

    ffmpeg redondea al múltiplo de 2 más cercano; en empates exactos
    (alto escalado impar exacto) su redondeo no se puede reproducir
    con una fórmula — medido el 21-11-2026: el mismo valor exacto
    (363,0) dio 364 en un video de 1280x726 y 362 en uno de 640x363
    — así que este valor es solo el respaldo cuando la sonda de un
    fotograma (``_ffmpeg_probe_analysis_height``) no puede confirmar
    el alto real. Un alto erróneo desalineanría todo el pipe rawvideo.
    """
    scaled = source_height * _FFMPEG_ANALYSIS_WIDTH / source_width
    return max(2, round(scaled / 2) * 2)


_OUTPUT_VIDEO_DIMS = re.compile(r"bgr24.*?, (\d+)x(\d+)")


def _ffmpeg_probe_analysis_height(
    input_path: Path,
    ffmpeg_path: str,
    source_width: int,
    source_height: int,
) -> int:
    """Alto real que ffmpeg dará a ``scale=640:-2`` (sonda de 1 fotograma).

    Decodifica un solo fotograma (~0,3 s) y lee las dimensiones de
    salida del stderr. La línea de salida está después de "Output #0";
    la de entrada (dimensiones originales) está antes y no debe
    usarse. Si algo falla, devuelve la fórmula de respaldo.
    """
    fallback = _ffmpeg_analysis_height(source_width, source_height)
    command = [
        ffmpeg_path,
        "-v",
        "info",
        "-nostats",
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={_FFMPEG_ANALYSIS_WIDTH}:-2,format=bgr24",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.debug("La sonda de altura ffmpeg falló (%s); se usa la fórmula.", error)
        return fallback
    text = result.stderr.decode("utf-8", "replace")
    _, _, after_output = text.partition("Output #0")
    match = _OUTPUT_VIDEO_DIMS.search(after_output or text)
    if match is None:
        return fallback
    width, height = int(match.group(1)), int(match.group(2))
    if width != _FFMPEG_ANALYSIS_WIDTH or height < 2:
        return fallback
    return height


class _FFmpegFrameSource:
    """Muestreo por pipe ffmpeg (vías rápidas de la pista del Studio).

    Un proceso por intervalo: ``-ss inicio -t duración -i video -vf
    fps=...,scale=640:H,format=bgr24 -f rawvideo -``. Con
    "ffmpeg-keyframes" se añade ``-skip_frame nokey`` (solo se
    decodifican keyframes y el filtro fps duplica el último para
    rellenar la rejilla); con "ffmpeg-dense" se usa ``-skip_frame
    noref -lowres 1`` (la vía rápida de visual_signals.py). El alto
    de análisis se confirma con una sonda de un fotograma (el
    redondeo de ``scale=640:-2`` no es reproducible con una fórmula
    en empates exactos). stdin DEVNULL (ffmpeg lee stdin y se
    bloquea si hereda un pipe abierto) y un hilo drena el stderr
    para que no se llene el buffer del pipe (lecciones de
    visual_signals.py). Un temporizador mata el proceso al vencer
    el plazo, para que una lectura del pipe colgada no deje el
    análisis atascado.
    """

    lip_tracking = False  # Se fija por modo en __init__ (ver abajo).

    def __init__(
        self,
        input_path: Path,
        ffmpeg_path: str,
        source_width: int,
        source_height: int,
        mode: str,
        deadline: float,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self.kind = mode
        self._input_path = input_path
        self._ffmpeg_path = ffmpeg_path
        self.source_width = source_width
        self.source_height = source_height
        self.analysis_width = _FFMPEG_ANALYSIS_WIDTH
        self.analysis_height = _ffmpeg_probe_analysis_height(
            input_path, ffmpeg_path, source_width, source_height
        )
        self.coord_scale = source_width / self.analysis_width
        # Con keyframes, las muestras únicas están separadas por
        # segundos: la ventana de 1,0 s del tracker de labios nunca
        # encadena y la inferencia de MediaPipe sería gasto inútil,
        # así que se desactiva y se sigue la cara de mayor área (como
        # KellyAiClip, que no tiene detección de orador). Con
        # muestreo denso (2 fps) sí aplica.
        self.lip_tracking = mode == "ffmpeg-dense"
        self._deadline = deadline
        self._should_cancel = should_cancel
        self.last_error_detail = ""

    def close(self) -> None:
        # Sin proceso persistente (uno por intervalo, cerrado en su
        # propio finally); existe por simetría con _Cv2FrameSource.
        return None

    def iter_interval(
        self,
        clip_start: float,
        interval_start: float,
        interval_end: float,
        sample_rate: float,
    ) -> Iterator[_Sample]:
        import numpy as np

        interval_duration = interval_end - interval_start
        start = clip_start + interval_start
        if self.kind == "ffmpeg-keyframes":
            skip_args = ["-skip_frame", "nokey"]
        else:
            skip_args = ["-skip_frame", "noref", "-lowres", "1"]
        command = [
            self._ffmpeg_path,
            "-v",
            "error",
            "-nostats",
            *skip_args,
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{interval_duration:.3f}",
            "-i",
            str(self._input_path),
            "-an",
            "-vf",
            f"fps={sample_rate},"
            f"scale={self.analysis_width}:{self.analysis_height},"
            "format=bgr24",
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError as error:
            raise _FrameSourceError(
                f"No se pudo iniciar ffmpeg para extraer fotogramas: {error}"
            ) from error
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_chunks: list[bytes] = []

        def _drain_stderr() -> None:
            try:
                while True:
                    chunk = process.stderr.read(65536)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)
            except (OSError, ValueError):
                pass

        drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
        drain_thread.start()
        killed_by_deadline = False

        def _kill_on_deadline() -> None:
            nonlocal killed_by_deadline
            killed_by_deadline = True
            try:
                process.kill()
            except OSError:
                pass

        timer: threading.Timer | None = None
        delay = self._deadline - time.monotonic()
        if delay > 0:
            timer = threading.Timer(delay, _kill_on_deadline)
            timer.daemon = True
            timer.start()
        frame_bytes = self.analysis_width * self.analysis_height * 3
        index = 0
        try:
            while True:
                if self._should_cancel is not None and self._should_cancel():
                    raise _FaceTrackCancelled(
                        "El seguimiento facial se canceló."
                    )
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                # .copy(): frombuffer da un array de solo lectura y
                # cv2 puede exigir un buffer escribible.
                frame = (
                    np.frombuffer(data, dtype=np.uint8)
                    .reshape(self.analysis_height, self.analysis_width, 3)
                    .copy()
                )
                local_time = index / sample_rate
                yield _Sample(
                    clip_start + interval_start + local_time,
                    local_time,
                    frame,
                )
                index += 1
        finally:
            if timer is not None:
                timer.cancel()
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            drain_thread.join(timeout=10)
            detail = (
                b"".join(stderr_chunks)
                .decode("utf-8", "replace")
                .strip()
                .splitlines()
            )
            if detail:
                self.last_error_detail = detail[-1][:200]
        if killed_by_deadline:
            raise FaceTrackingError(
                "El seguimiento facial superó el tiempo máximo permitido "
                "para este video. Vuelve a intentarlo."
            )


def _analyze_with_source(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None,
    on_progress: Callable[[float], None] | None,
    sample_rate: float,
    source_name: str,
    deadline: float,
    should_cancel: Callable[[], bool] | None,
) -> tuple[list[list[FaceCropKeyframe] | None], dict[str, int]]:
    """Análisis facial con UNA fuente de fotogramas.

    Devuelve (rutas, estadísticas). Las estadísticas cuentan las
    muestras leídas, las únicas (sin duplicados del filtro fps), los
    intentos de detección y sus fallos: el llamador las usa para
    decidir si la vía de keyframes dio demasiadas pocas muestras
    únicas. Los fallos por muestra (lectura, detección, labios) no
    abortan el análisis: la muestra se omite y se sigue; solo se
    lanza si NO se leyó ningún fotograma, si la detección falló en
    TODAS las muestras, o si vence el plazo / se cancela.
    """
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

    stats = {
        "samples_read": 0,
        "unique_frames": 0,
        "detect_attempts": 0,
        "detect_errors": 0,
        "samples_failed": 0,
    }
    if source_name == "cv2":
        source: _Cv2FrameSource | _FFmpegFrameSource = _Cv2FrameSource(
            input_path, cv2
        )
        source_width, source_height = source.source_width, source.source_height
    else:
        source_width, source_height = _probe_video_dimensions(input_path)
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise _FrameSourceError(
                "ffmpeg no está disponible para extraer los fotogramas."
            )
        source = _FFmpegFrameSource(
            input_path,
            ffmpeg_path,
            source_width,
            source_height,
            source_name,
            deadline,
            should_cancel,
        )
    try:
        scale, max_x, max_y = _crop_geometry(
            source_width, source_height, output_width, output_height
        )
        if max_x < 1 and max_y < 1:
            if on_progress is not None:
                on_progress(100.0)
            return [None for _ in intervals], stats

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
        detector.setInputSize((source.analysis_width, source.analysis_height))
        media_pipe = None
        landmarker = None
        # Detección de orador por labios (MediaPipe): solo un extra
        # para elegir entre varios rostros. Si el modelo o MediaPipe
        # no están disponibles, el seguimiento sigue con YuNet solo
        # (sin abortar el análisis con un error): un rostro único se
        # sigue igual, y con varios se elige el de mayor área y
        # confianza, como sin orador preferido. En la vía de
        # keyframes se desactiva de entrada (ver _FFmpegFrameSource).
        lip_tracking_available = source.lip_tracking
        tracker = _SubjectTracker(
            source_width,
            source_height,
            output_width / scale,
        )
        speaker_tracker = _SpeakerActivityTracker(source_width, source_height)
        previous_thumbnail = None
        previous_faces: list[_Face] | None = None
        previous_preferred_face: _Face | None = None
        any_face_detected = False
        all_paths: list[list[FaceCropKeyframe] | None] = []
        total_samples = sum(
            max(1, math.ceil((interval_end - interval_start) * sample_rate))
            for interval_start, interval_end in intervals
        )
        if total_samples == 0:
            if on_progress is not None:
                on_progress(100.0)
            return [], stats
        completed_samples = 0
        if on_progress is not None:
            on_progress(0.0)
        # Intervalos de habla fusionados para la consulta por muestra
        # (O(log n) con bisect en vez de O(n) por muestra; ver
        # _merge_speech_intervals).
        speech_starts, speech_ends = _merge_speech_intervals(speech_intervals)

        def _report_progress() -> None:
            if on_progress is not None:
                on_progress(100 * completed_samples / total_samples)

        for interval_start, interval_end in intervals:
            interval_duration = interval_end - interval_start
            points: list[FaceCropKeyframe] = []
            consecutive_read_failures = 0
            samples = source.iter_interval(
                clip_start, interval_start, interval_end, sample_rate
            )
            try:
                for sample in samples:
                    _check_deadline_or_cancel(deadline, should_cancel)
                    completed_samples += 1
                    if sample.frame is None:
                        # Lectura fallida (fotograma corrupto o fin
                        # del video antes de lo que dice la duración
                        # de metadatos): se omite la muestra. Tras
                        # varias seguidas se da por terminado el
                        # intervalo en vez de insistir hasta el final.
                        stats["samples_failed"] += 1
                        consecutive_read_failures += 1
                        if (
                            consecutive_read_failures
                            > _MAX_CONSECUTIVE_READ_FAILURES
                        ):
                            logger.warning(
                                "El video dejó de leerse en el segundo "
                                "%.2f; se analiza hasta ahí.",
                                sample.absolute_time,
                            )
                            break
                        _report_progress()
                        continue
                    consecutive_read_failures = 0
                    stats["samples_read"] += 1
                    try:
                        frame = sample.frame
                        thumbnail = cv2.cvtColor(
                            cv2.resize(frame, (48, 27)), cv2.COLOR_BGR2GRAY
                        )
                        diff = (
                            float(
                                cv2.absdiff(
                                    previous_thumbnail, thumbnail
                                ).mean()
                            )
                            if previous_thumbnail is not None
                            else None
                        )
                        scene_changed = diff is not None and diff >= 52.0
                        # Fotograma duplicado por el filtro fps de la
                        # vía de keyframes (misma imagen que la
                        # muestra anterior, diff exactamente 0.0):
                        # se reutilizan los rostros y el orador sin
                        # volver a detectar ni a medir labios. El
                        # tracker se actualiza igual (misma posición,
                        # sin movimiento), así la rejilla de 2 fps y
                        # el hold de dropout funcionan como siempre.
                        duplicate = (
                            diff == 0.0 and previous_faces is not None
                        )
                        previous_thumbnail = thumbnail
                        if duplicate:
                            assert previous_faces is not None
                            faces = previous_faces
                            preferred_face = previous_preferred_face
                        else:
                            stats["unique_frames"] += 1
                            try:
                                stats["detect_attempts"] += 1
                                _, detections = detector.detect(frame)
                            except Exception as error:  # noqa: BLE001 - un fotograma malo no tumba el análisis
                                stats["detect_errors"] += 1
                                logger.warning(
                                    "La detección de rostros falló en el "
                                    "segundo %.2f: %s",
                                    sample.absolute_time,
                                    error,
                                )
                                detections = None
                            faces = (
                                [
                                    _Face(
                                        float(detection[0])
                                        * source.coord_scale,
                                        float(detection[1])
                                        * source.coord_scale,
                                        float(detection[2])
                                        * source.coord_scale,
                                        float(detection[3])
                                        * source.coord_scale,
                                        float(detection[-1]),
                                    )
                                    for detection in detections
                                ]
                                if detections is not None
                                else []
                            )
                            previous_faces = faces
                            any_face_detected |= bool(faces)
                            preferred_face = None
                            speech_active = _speech_active_at(
                                speech_starts,
                                speech_ends,
                                sample.absolute_time,
                            )
                            if scene_changed:
                                speaker_tracker.reset()
                            measurements: list[tuple[_Face, float]] = []
                            # Con menos de dos rostros la selección del
                            # orador no cambia nada (el único rostro se
                            # sigue igualmente), así que se omite el
                            # landmarker: en un video de una sola
                            # persona hablando ahorra una inferencia
                            # por muestra.
                            if (
                                speech_active
                                and len(faces) >= 2
                                and lip_tracking_available
                            ):
                                if landmarker is None:
                                    try:
                                        media_pipe, landmarker = (
                                            _get_lip_landmarker()
                                        )
                                    except FaceTrackingError as error:
                                        lip_tracking_available = False
                                        logger.warning(
                                            "Seguimiento facial sin "
                                            "detección de orador (se sigue "
                                            "con YuNet solo): %s",
                                            error,
                                        )
                                if (
                                    landmarker is not None
                                    and media_pipe is not None
                                ):
                                    try:
                                        rgb_frame = cv2.cvtColor(
                                            frame, cv2.COLOR_BGR2RGB
                                        )
                                        image = media_pipe.Image(
                                            image_format=media_pipe.ImageFormat.SRGB,
                                            data=rgb_frame,
                                        )
                                        with _LIP_LANDMARKER_LOCK:
                                            result = landmarker.detect(image)
                                        landmarks_by_face = (
                                            result.face_landmarks or []
                                        )
                                        measurements = _lip_measurements(
                                            landmarks_by_face,
                                            faces,
                                            source_width,
                                            source_height,
                                        )
                                    except Exception as error:  # noqa: BLE001 - un fallo del landmarker a mitad no tumba el análisis
                                        lip_tracking_available = False
                                        landmarker = None
                                        logger.warning(
                                            "La detección de orador falló "
                                            "en el segundo %.2f; se sigue "
                                            "con YuNet solo: %s",
                                            sample.absolute_time,
                                            error,
                                        )
                            preferred_face = speaker_tracker.update(
                                measurements,
                                sample.absolute_time,
                                speech_active,
                            )
                            previous_preferred_face = preferred_face
                        target = tracker.update(
                            faces,
                            sample.absolute_time,
                            scene_changed,
                            preferred_face=preferred_face,
                        )
                        if target is not None:
                            x, y = _crop_position(
                                target,
                                source_width,
                                source_height,
                                output_width,
                                output_height,
                            )
                            point_time = sample.local_time
                            if scene_changed and points:
                                previous = points[-1]
                                hold_time = max(previous.time, point_time - 0.02)
                                if hold_time > previous.time:
                                    points.append(
                                        FaceCropKeyframe(
                                            hold_time, previous.x, previous.y
                                        )
                                    )
                            points.append(FaceCropKeyframe(point_time, x, y))
                    except FaceTrackingError:
                        # Plazo/cancelación: propagar, nunca tragar.
                        raise
                    except Exception as error:  # noqa: BLE001 - una muestra mala no tumba el análisis
                        stats["samples_failed"] += 1
                        logger.warning(
                            "Se omitió una muestra del seguimiento facial "
                            "en el segundo %.2f: %s",
                            sample.absolute_time,
                            error,
                        )
                    _report_progress()
            finally:
                samples.close()

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

        if stats["samples_read"] == 0:
            if source_name == "cv2":
                raise FaceTrackingError(
                    "No se pudo leer ningún fotograma del video original. "
                    "El archivo podría estar dañado o en un formato no "
                    "compatible."
                )
            detail = source.last_error_detail
            raise _FrameSourceError(
                "ffmpeg no devolvió ningún fotograma del video."
                + (f" ({detail})" if detail else "")
            )
        if (
            stats["detect_attempts"] > 0
            and stats["detect_errors"] == stats["detect_attempts"]
        ):
            raise FaceTrackingError(
                "El detector facial no pudo analizar ningún fotograma del video."
            )
        if on_progress is not None:
            on_progress(100.0)
        return (
            all_paths if any_face_detected else [None for _ in intervals]
        ), stats
    except cv2.error as error:
        raise FaceTrackingError(
            f"El detector facial no pudo analizar el video: {error}"
        ) from error
    finally:
        source.close()


def _analyze_face_crop_paths_uncached(
    input_path: Path,
    clip_start: float,
    intervals: list[tuple[float, float]],
    output_width: int,
    output_height: int,
    speech_intervals: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    sample_rate: float = _SAMPLE_RATE,
    frame_source: str = "cv2",
    deadline_seconds: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[list[FaceCropKeyframe] | None]:
    """Análisis facial sin caché, con reintento entre fuentes de fotogramas.

    ``frame_source="auto"`` prueba keyframes → denso → cv2: una vía
    que no entrega ningún fotograma (ffmpeg ausente, video que esa
    vía no decodifica) da paso a la siguiente, y una vía de keyframes
    con demasiadas pocas muestras únicas (menos de 1 por cada 15 s:
    el recorte quedaría casi estático) reintenta con muestreo denso.
    El plazo es común a toda la cadena (los reintentos no reinician
    el reloj) y un plazo vencido o una cancelación NO reintentan.
    Con una fuente explícita ("cv2", "ffmpeg-keyframes",
    "ffmpeg-dense") no hay reintento.
    """
    valid_sources = ("cv2", "ffmpeg-keyframes", "ffmpeg-dense", "auto")
    if frame_source not in valid_sources:
        raise ValueError(
            f"Fuente de fotogramas no válida: {frame_source!r} "
            f"(usa una de {', '.join(valid_sources)})."
        )
    chain = (
        ("ffmpeg-keyframes", "ffmpeg-dense", "cv2")
        if frame_source == "auto"
        else (frame_source,)
    )
    total_span = sum(
        max(0.0, interval_end - interval_start)
        for interval_start, interval_end in intervals
    )
    budget = (
        deadline_seconds
        if deadline_seconds is not None
        else max(
            _FACE_TRACK_DEADLINE_MIN_SECONDS,
            _FACE_TRACK_DEADLINE_SECONDS_PER_MINUTE * total_span / 60.0,
        )
    )
    deadline = time.monotonic() + budget
    last_error: FaceTrackingError | None = None
    for index, source_name in enumerate(chain):
        try:
            paths, stats = _analyze_with_source(
                input_path,
                clip_start,
                intervals,
                output_width,
                output_height,
                speech_intervals,
                on_progress,
                sample_rate,
                source_name,
                deadline,
                should_cancel,
            )
        except _FrameSourceError as error:
            last_error = error
            logger.warning(
                "La fuente %s no pudo analizar el video (%s); se reintenta "
                "con otra vía.",
                source_name,
                error,
            )
            continue
        if (
            source_name == "ffmpeg-keyframes"
            and index + 1 < len(chain)
            and stats["unique_frames"]
            < max(1, math.ceil(total_span / _KEYFRAME_MIN_SECONDS_PER_SAMPLE))
        ):
            logger.warning(
                "La vía de keyframes solo encontró %d fotogramas únicos en "
                "%.0f s de video; se reintenta con muestreo denso.",
                stats["unique_frames"],
                total_span,
            )
            continue
        return paths
    raise last_error or FaceTrackingError(
        "No se pudo analizar el video para el seguimiento facial."
    )
