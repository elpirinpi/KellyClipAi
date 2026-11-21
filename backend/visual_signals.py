"""Señales visuales para el motor de momentos (Fase C, v1.2.0).

Extrae los timelines visuales 1 Hz que consume ``backend.moments``:
cortes de escena, energía de movimiento y presencia de rostro. Todo
Tier 0: 0 MB adicionales, sin descargas ni modelos nuevos — solo ffmpeg
(que la app ya tiene) y, para los rostros, el YuNet que ya trae
``backend/face_tracking.py`` (mismo modelo, mismos parámetros).

Extracción (un solo proceso ffmpeg para cortes + movimiento):
- Cortes: filtro ``select='gt(scene,0.4)'`` + ``showinfo`` (la vía
  estándar y parseable; la salida se lee del stderr). Un corte por
  segundo vale 0.63 en el timeline (curva de saturación ``1-exp(-n)``).
- Movimiento: frames grises a 2 fps por pipe (streaming, memoria O(1)),
  diferencia media absoluta entre frames consecutivos con numpy,
  normalizada con z-score robusto (mediana + MAD, como el canal de
  audio) y la misma curva que el pico de audio v1.1.1
  (``clip01((z - 0.5) / 2.0)``). La media cruda (niveles de gris 0-255)
  se conserva aparte: es una medida ABSOLUTA de actividad que usa la
  detección de perfil de contenido (el timeline normalizado es relativo
  al video y no sirve para eso).
- Rostros: frames BGR a 1 fps por pipe + YuNet vía cv2.dnn (importación
  perezosa: sin cv2 no hay canal de rostros, pero Tier 0 sigue intacto).

Velocidad (medido en esta VM, 1,5 CPU, video 720p 60 s): decodificación
completa 9,3 s; con ``-skip_frame noref -lowres 1`` 2,8 s (3,3x, con
detección de cortes IDÉNTICA en el video de prueba). Los TRES canales
comparten UNA decodificación (filter_complex con split en 3: movimiento
a un archivo temporal, rostros por stdout, cortes por showinfo en
stderr); antes los rostros usaban una segunda pasada y costaban otros
~46 s en un video de 320 s. La fusión da timelines bit-idénticos
(medido el 21-11-2026 en un video 1080p60 de 320 s con rostros en 314
de 321 segundos: diferencia máxima 0.0 en los tres canales). En el PC
de Kelly (Xeon de 14 núcleos) un video de 1 h queda en decenas de
segundos.

Calibración del movimiento crudo (MAD medio, niveles de gris 0-255,
medido en esta VM a 160x90 y 2 fps): video estático 0,0; patrón
testsrc2 (muchos elementos pequeños en movimiento) 3,6; cuadrado blanco
de 60x60 moviéndose rápido sobre negro 7,9; zoom de Mandelbrot 14,8;
paneo de cámara sobre detalle 90; presentación con un corte cada 2 s
22 (dominado por los cortes). De ahí los umbrales de perfil en
``moments.detect_content_profile``: <= 5 es "calmo" (cubre cabeza
parlante y fondo animado) y >= 8 es "activo".

Ninguna función pública de este módulo lanza por problemas de
extracción (ffmpeg ausente, archivo sin video, cv2 ausente): devuelve
None (canal ausente) y registra un aviso. None significa "sin datos";
un timeline de ceros significa "medido y sin señal" — la distinción
importa porque el scorer trata ambos casos distinto.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Cortes de escena --------------------------------------------------------
_SCENE_THRESHOLD = 0.4  # Valor estándar del filtro scene de ffmpeg.
# Líneas de frame de showinfo: "n:   0 pts:  30720 pts_time:2 ...". Las
# líneas de configuración ("config in/out") y de metadatos (SEI) no
# llevan "n: ... pts_time:" y se ignoran solas.
_SHOWINFO_FRAME = re.compile(r"\bn:\s*\d+\s+pts:\s*\d+\s+pts_time:([0-9]+(?:\.[0-9]+)?)")

# --- Movimiento --------------------------------------------------------------
_MOTION_FPS = 2.0
_MOTION_WIDTH = 160
_MOTION_HEIGHT = 90
# Piso de sigma para el z-score robusto: por debajo de ~1 nivel de gris
# de diferencia media, la "variación" es ruido de compresión, no
# movimiento (sin piso, un video casi estático amplificaría el ruido
# hasta 1.0).
_MOTION_SIGMA_FLOOR = 1.0

# --- Rostros -----------------------------------------------------------------
_FACE_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"  # Igual que face_tracking.
_FACE_FPS = 1.0
_FACE_FRAME_SIZE = 320  # Lienzo cuadrado 320x320 (sin distorsionar).
# Fracción del cuadro a partir de la cual un rostro cuenta como
# "prominente" para la ponderación por tamaño.
_FACE_MIN_AREA_FRACTION = 0.05
# Parámetros del detector YuNet: los mismos que face_tracking.py.
_FACE_SCORE_THRESHOLD = 0.55
_FACE_NMS_THRESHOLD = 0.3
_FACE_TOP_K = 5000


@dataclass(frozen=True)
class VisualStats:
    """Estadísticas visuales ABSOLUTAS para detectar el perfil de contenido.

    Los timelines normalizados son relativos al video (un video de
    gameplay con movimiento constante normaliza a ~0); estas cifras
    miden la actividad tal cual para clasificar talk / mixed / gameplay.
    """

    cut_rate_per_minute: float = 0.0
    mean_motion_raw: float = 0.0  # MAD medio en niveles de gris (0-255).


@dataclass
class VisualTimelines:
    """Timelines visuales 1 Hz + estadísticas (salida de la extracción).

    Cada timeline es None cuando su extracción falló o no aplica (canal
    ausente: el scorer lo ignora). ``combined`` es el timeline que
    alimenta el canal "visual" del scorer.
    """

    scene_cuts: np.ndarray | None = None
    motion: np.ndarray | None = None
    face: np.ndarray | None = None
    cut_count: int = 0
    mean_motion_raw: float = 0.0
    duration: float = 0.0

    @property
    def combined(self) -> np.ndarray | None:
        """Escenas + movimiento en un solo timeline (máximo por segundo)."""
        return combine_visual_timeline(self.scene_cuts, self.motion)

    @property
    def face_fraction(self) -> float:
        """Fracción de segundos con rostro en cuadro (0.0 sin datos)."""
        if self.face is None or self.face.size == 0:
            return 0.0
        return float(np.mean(self.face > 0.5))

    @property
    def stats(self) -> VisualStats:
        """Estadísticas absolutas para ``moments.detect_content_profile``."""
        cut_rate = self.cut_count / self.duration * 60.0 if self.duration > 0 else 0.0
        return VisualStats(cut_rate_per_minute=cut_rate, mean_motion_raw=self.mean_motion_raw)


# --- cortes de escena ----------------------------------------------------------


def parse_showinfo_cut_times(stderr_text: str) -> list[float]:
    """Tiempos de corte (s) desde la salida de ffmpeg con showinfo.

    Solo cuentan las líneas de frame (``n: … pts_time:…``). Función
    pura: no toca disco ni procesos, para tests y calibración.
    """
    times: list[float] = []
    for line in stderr_text.splitlines():
        if "showinfo" not in line:
            continue
        match = _SHOWINFO_FRAME.search(line)
        if match:
            times.append(float(match.group(1)))
    return times


def _scene_timeline_from_cut_times(
    cut_times: list[float], duration: float
) -> tuple[np.ndarray, int]:
    """Timeline 1 Hz de densidad de cortes + número de cortes.

    Curva de saturación ``1 - exp(-n)``: 1 corte en un segundo vale
    0.63, 2 valen 0.86, 3 o más ~0.95. Un segundo con un corte ya es
    notable; más cortes en el mismo segundo apenas añaden.
    """
    seconds = max(1, math.ceil(duration))
    counts = np.zeros(seconds, dtype=np.float64)
    inside = 0
    for cut_time in cut_times:
        second = math.floor(cut_time)
        if 0 <= second < seconds:
            counts[second] += 1
            inside += 1
    return 1.0 - np.exp(-counts), inside


def extract_scene_cut_timeline(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    scene_threshold: float = _SCENE_THRESHOLD,
) -> np.ndarray | None:
    """Timeline 1 Hz de cortes de escena (0–1) con ffmpeg select+showinfo.

    Devuelve None si ffmpeg falla o el archivo no tiene video; nunca
    lanza. Un video sin cortes devuelve ceros (medido, no ausente).
    """
    command = [
        ffmpeg_path,
        "-v",
        "info",
        "-nostats",
        "-skip_frame",
        "noref",
        "-lowres",
        "1",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        f"select='gt(scene,{scene_threshold})',showinfo",
        "-f",
        "null",
        os.devnull,
    ]
    try:
        # stdin DEVNULL: ffmpeg lee stdin para sus comandos interactivos
        # y se bloquea para siempre si hereda un pipe abierto sin EOF
        # (deadlock medido el 21-11-2026 al perfilar).
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        logger.warning("No se pudo detectar cortes de escena: %s", error)
        return None
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        logger.warning(
            "ffmpeg falló al detectar cortes de escena (%s).",
            detail[-1][:200] if detail else "sin pista de video",
        )
        return None
    cut_times = parse_showinfo_cut_times(result.stderr.decode("utf-8", "replace"))
    timeline, _ = _scene_timeline_from_cut_times(cut_times, duration)
    return timeline


# --- movimiento ------------------------------------------------------------------


class _MotionAccumulator:
    """Diferencia media absoluta entre frames grises consecutivos, por segundo."""

    def __init__(self, seconds: int, fps: float) -> None:
        self._sums = np.zeros(seconds, dtype=np.float64)
        self._counts = np.zeros(seconds, dtype=np.float64)
        self._fps = fps
        self._seconds = seconds
        self._prev: np.ndarray | None = None
        self._index = 0
        self.frames = 0

    def add(self, frame: np.ndarray) -> None:
        current = frame.astype(np.float32)
        if self._prev is not None:
            diff = float(np.mean(np.abs(current - self._prev)))
            second = min(int(self._index / self._fps), self._seconds - 1)
            self._sums[second] += diff
            self._counts[second] += 1
        self._prev = current
        self._index += 1
        self.frames += 1

    def raw_per_second(self) -> np.ndarray:
        return np.divide(
            self._sums,
            self._counts,
            out=np.zeros(self._seconds, dtype=np.float64),
            where=self._counts > 0,
        )


def _normalize_motion(raw_per_second: np.ndarray) -> tuple[np.ndarray, float]:
    """Z-score robusto (mediana + MAD) del MAD crudo -> timeline 0–1.

    Misma curva que el componente de pico del canal de audio (v1.1.1):
    ``clip01((z - 0.5) / 2.0)``. Devuelve (timeline, media cruda): la
    media cruda es la medida absoluta de actividad para el perfil.
    """
    mean_raw = float(np.mean(raw_per_second)) if raw_per_second.size else 0.0
    median = float(np.median(raw_per_second)) if raw_per_second.size else 0.0
    mad = float(np.median(np.abs(raw_per_second - median))) if raw_per_second.size else 0.0
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(raw_per_second)) if raw_per_second.size else 0.0
    sigma = max(sigma, _MOTION_SIGMA_FLOOR)
    z = (raw_per_second - median) / sigma
    return np.clip((z - 0.5) / 2.0, 0.0, 1.0), mean_raw


def extract_motion_timeline(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    fps: float = _MOTION_FPS,
    width: int = _MOTION_WIDTH,
    height: int = _MOTION_HEIGHT,
) -> np.ndarray | None:
    """Timeline 1 Hz de energía de movimiento (0–1) con ffmpeg + numpy.

    Decodifica frames grises a 2 fps por pipe (streaming, memoria O(1))
    y mide la diferencia media absoluta entre consecutivos. Devuelve
    None si ffmpeg falla o el archivo no tiene video; nunca lanza.
    """
    seconds = max(1, math.ceil(duration))
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-skip_frame",
        "noref",
        "-lowres",
        "1",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        # stdin DEVNULL: ffmpeg lee stdin para sus comandos interactivos
        # y se bloquea para siempre si hereda un pipe abierto sin EOF
        # (deadlock medido el 21-11-2026 al perfilar).
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        logger.warning("No se pudo extraer el movimiento: %s", error)
        return None
    assert process.stdout is not None
    frame_bytes = width * height
    accumulator = _MotionAccumulator(seconds, fps)
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            accumulator.add(np.frombuffer(data, dtype=np.uint8))
    except (OSError, ValueError) as error:
        logger.warning("La extracción de movimiento falló a medias: %s", error)
    finally:
        try:
            process.kill()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            stderr = b""
    if accumulator.frames == 0:
        detail = stderr.decode("utf-8", "replace").strip()[:200]
        logger.warning(
            "ffmpeg no devolvió frames para el movimiento (%s).",
            detail or "sin pista de video",
        )
        return None
    timeline, _ = _normalize_motion(accumulator.raw_per_second())
    return timeline


def combine_visual_timeline(
    scene_cuts: np.ndarray | None, motion: np.ndarray | None
) -> np.ndarray | None:
    """Máximo por segundo de cortes y movimiento (None si ambos ausentes).

    El máximo (no la media): un corte O un pico de movimiento bastan
    para marcar un segundo visualmente notable, y un corte suele
    coincidir con un pico de movimiento — la media lo contaría dos veces.
    """
    if scene_cuts is None and motion is None:
        return None
    if scene_cuts is None:
        return np.asarray(motion, dtype=np.float64)
    if motion is None:
        return np.asarray(scene_cuts, dtype=np.float64)
    length = max(scene_cuts.size, motion.size)
    combined = np.zeros(length, dtype=np.float64)
    combined[: scene_cuts.size] = np.maximum(combined[: scene_cuts.size], scene_cuts)
    combined[: motion.size] = np.maximum(combined[: motion.size], motion)
    return combined


# --- rostros ----------------------------------------------------------------------


def _face_frame_value(detections: Any, width: int, height: int) -> float:
    """Valor 0–1 de un frame: mejor rostro (confianza x tamaño x centralidad).

    ``detections`` es la salida de YuNet: filas [x, y, w, h, ..., confianza]
    (la confianza es el último valor, como en face_tracking). Un rostro
    grande y centrado vale su confianza; uno pequeño y en una esquina,
    la mitad. Sin rostros, 0.0. Función pura, para tests sin cv2.
    """
    if detections is None or len(detections) == 0:
        return 0.0
    best = 0.0
    for detection in detections:
        x, y, w, h = (float(detection[i]) for i in range(4))
        confidence = float(detection[-1])
        area_fraction = (w * h) / max(1.0, float(width * height))
        size_factor = min(1.0, area_fraction / _FACE_MIN_AREA_FRACTION)
        center_x, center_y = (x + w / 2) / width, (y + h / 2) / height
        centrality = 1.0 - min(
            1.0, math.hypot(center_x - 0.5, center_y - 0.5) / math.hypot(0.5, 0.5)
        )
        value = confidence * (0.5 + 0.3 * size_factor + 0.2 * centrality)
        best = max(best, value)
    return min(1.0, best)


def _create_face_detector(model_path: Path | str | None) -> Any | None:
    """Crea el detector YuNet (mismo modelo y parámetros que face_tracking).

    Importación perezosa de cv2: sin OpenCV no hay canal de rostros,
    pero Tier 0 sigue intacto. Devuelve None si falta cv2, el modelo o
    la creación falla; nunca lanza.
    """
    path = (
        Path(model_path)
        if model_path is not None
        else Path(__file__).with_name("models") / _FACE_MODEL_FILENAME
    )
    if not path.is_file():
        logger.warning("Falta el modelo de detección facial (%s); sin canal de rostros.", path)
        return None
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV no está instalado; sin canal de rostros.")
        return None
    try:
        detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (_FACE_FRAME_SIZE, _FACE_FRAME_SIZE),
            _FACE_SCORE_THRESHOLD,
            _FACE_NMS_THRESHOLD,
            _FACE_TOP_K,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        detector.setInputSize((_FACE_FRAME_SIZE, _FACE_FRAME_SIZE))
    except Exception as error:  # noqa: BLE001 - sin detector no hay canal; Tier 0 sigue
        logger.warning("No se pudo crear el detector facial YuNet: %s", error)
        return None
    return detector


def extract_face_timeline(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    fps: float = _FACE_FPS,
    model_path: Path | str | None = None,
    detector: Any | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """Timeline 1 Hz de presencia de rostro (0–1) con ffmpeg + YuNet.

    Muestrea frames a ~1 fps (lienzo cuadrado 320x320, sin distorsionar)
    y corre el detector YuNet que ya trae la app. ``detector`` es
    inyectable para los tests (sin cv2 en el entorno de CI). Devuelve
    None si falta cv2, el modelo, ffmpeg o el video; nunca lanza. Una
    cancelación a medias también devuelve None: un timeline parcial
    (ceros en la cola) sería datos falsos, peor que la ausencia.
    """
    seconds = max(1, math.ceil(duration))
    if detector is None:
        detector = _create_face_detector(model_path)
        if detector is None:
            return None
    size = _FACE_FRAME_SIZE
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-skip_frame",
        "noref",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        # stdin DEVNULL: ffmpeg lee stdin para sus comandos interactivos
        # y se bloquea para siempre si hereda un pipe abierto sin EOF
        # (deadlock medido el 21-11-2026 al perfilar).
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        logger.warning("No se pudo extraer frames para la detección de rostros: %s", error)
        return None
    assert process.stdout is not None
    frame_bytes = size * size * 3
    timeline = np.zeros(seconds, dtype=np.float64)
    frame_index = 0
    saw_frames = False
    cancelled = False
    try:
        while True:
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            saw_frames = True
            # .copy(): frombuffer da un array de solo lectura y cv2 puede
            # exigir un buffer escribible.
            frame = np.frombuffer(data, dtype=np.uint8).reshape(size, size, 3).copy()
            second = min(int(frame_index / fps), seconds - 1)
            try:
                _, detections = detector.detect(frame)
            except Exception as error:  # noqa: BLE001 - un frame malo no tumba el canal
                logger.warning("La detección de rostros falló en el segundo %d: %s", second, error)
            else:
                timeline[second] = max(timeline[second], _face_frame_value(detections, size, size))
            frame_index += 1
    except (OSError, ValueError) as error:
        logger.warning("La extracción de rostros falló a medias: %s", error)
    finally:
        try:
            process.kill()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            stderr = b""
    if cancelled or not saw_frames:
        if not saw_frames and not cancelled:
            detail = stderr.decode("utf-8", "replace").strip()[:200]
            logger.warning(
                "ffmpeg no devolvió frames para la detección de rostros (%s).",
                detail or "sin pista de video",
            )
        return None
    return timeline


# --- extracción combinada ------------------------------------------------------


def _extract_scene_and_motion(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    scene_threshold: float = _SCENE_THRESHOLD,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int, float]:
    """Cortes + movimiento en UN solo proceso ffmpeg (una decodificación).

    filter_complex con split: la rama de movimiento va por stdout
    (rawvideo gris a 2 fps) y la de cortes a null con showinfo (los
    tiempos se leen del stderr, que un hilo drena en paralelo para que
    ffmpeg no se bloquee al llenarse el buffer del pipe). Devuelve
    (cortes, movimiento, nº de cortes, media cruda de movimiento);
    (None, None, 0, 0.0) si no hubo frames o se canceló.
    """
    seconds = max(1, math.ceil(duration))
    command = [
        ffmpeg_path,
        "-v",
        "info",
        "-nostats",
        "-skip_frame",
        "noref",
        "-lowres",
        "1",
        "-i",
        str(input_path),
        "-an",
        "-filter_complex",
        (
            "[0:v]split=2[m_in][s_in];"
            f"[m_in]fps={_MOTION_FPS},scale={_MOTION_WIDTH}:{_MOTION_HEIGHT},format=gray[m];"
            f"[s_in]select='gt(scene,{scene_threshold})',showinfo[s]"
        ),
        "-map",
        "[m]",
        "-f",
        "rawvideo",
        "-",
        "-map",
        "[s]",
        "-f",
        "null",
        os.devnull,
    ]
    try:
        # stdin DEVNULL: ffmpeg lee stdin para sus comandos interactivos
        # y se bloquea para siempre si hereda un pipe abierto sin EOF
        # (deadlock medido el 21-11-2026 al perfilar).
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        logger.warning("No se pudo extraer las señales visuales: %s", error)
        return None, None, 0, 0.0
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
    frame_bytes = _MOTION_WIDTH * _MOTION_HEIGHT
    accumulator = _MotionAccumulator(seconds, _MOTION_FPS)
    cancelled = False
    try:
        while True:
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            accumulator.add(np.frombuffer(data, dtype=np.uint8))
    except (OSError, ValueError) as error:
        logger.warning("La extracción de señales visuales falló a medias: %s", error)
    finally:
        if cancelled:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        drain_thread.join(timeout=10)
    if cancelled or accumulator.frames == 0:
        if accumulator.frames == 0 and not cancelled:
            detail = b"".join(stderr_chunks).decode("utf-8", "replace").strip().splitlines()
            logger.warning(
                "ffmpeg no devolvió frames para las señales visuales (%s).",
                detail[-1][:200] if detail else "sin pista de video",
            )
        return None, None, 0, 0.0
    stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace")
    cut_times = parse_showinfo_cut_times(stderr_text)
    scene_timeline, cut_count = _scene_timeline_from_cut_times(cut_times, duration)
    motion_timeline, mean_raw = _normalize_motion(accumulator.raw_per_second())
    return scene_timeline, motion_timeline, cut_count, mean_raw


def _extract_all_visual(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    scene_threshold: float = _SCENE_THRESHOLD,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int, float]:
    """Cortes + movimiento + rostros en UN solo proceso ffmpeg (una decodificación).

    filter_complex con split en 3: la rama de movimiento va a un
    archivo temporal (gris 160x90 a 2 fps, ~29 KB/s: 9 MB en un video
    de 320 s, ~415 MB en el peor caso de 4 h, borrado al terminar), la
    de rostros por stdout (BGR 320x320 a 1 fps, detección YuNet en
    streaming) y la de cortes a null con showinfo (tiempos del stderr,
    drenado por un hilo). Un solo stdout obliga a sacar el movimiento
    por archivo: dos pipes no existen en Windows y un hstack
    complicaría el sincronizado de fps distintos.

    Los timelines son los mismos que los de las dos pasadas separadas:
    la rama de movimiento y la de cortes usan los mismos filtros y la
    misma decodificación (``-skip_frame noref -lowres 1``); la de
    rostros comparte esa decodificación a media resolución y en el
    video de prueba dio detecciones bit-idénticas (ver el docstring
    del módulo). Devuelve (cortes, movimiento, rostros, nº de cortes,
    media cruda de movimiento); todo None si se canceló o no hubo
    frames. Nunca lanza: cada canal que falla queda en None.
    """
    seconds = max(1, math.ceil(duration))
    if should_cancel is not None and should_cancel():
        return None, None, None, 0, 0.0
    detector = _create_face_detector(None)
    face_timeline: np.ndarray | None = (
        np.zeros(seconds, dtype=np.float64) if detector is not None else None
    )
    size = _FACE_FRAME_SIZE
    with tempfile.TemporaryDirectory(prefix="kca-visual-") as tmp_dir:
        motion_path = os.path.join(tmp_dir, "motion.raw")
        command = [
            ffmpeg_path,
            "-y",
            "-v",
            "info",
            "-nostats",
            "-skip_frame",
            "noref",
            "-lowres",
            "1",
            "-i",
            str(input_path),
            "-an",
            "-filter_complex",
            (
                "[0:v]split=3[m_in][s_in][f_in];"
                f"[m_in]fps={_MOTION_FPS},scale={_MOTION_WIDTH}:{_MOTION_HEIGHT},format=gray[m];"
                f"[s_in]select='gt(scene,{scene_threshold})',showinfo[s];"
                f"[f_in]fps={_FACE_FPS},scale={size}:{size}:force_original_aspect_ratio=decrease,"
                f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,format=bgr24[f]"
            ),
            "-map",
            "[m]",
            "-f",
            "rawvideo",
            motion_path,
            "-map",
            "[f]",
            "-f",
            "rawvideo",
            "-",
            "-map",
            "[s]",
            "-f",
            "null",
            os.devnull,
        ]
        try:
            # stdin DEVNULL: ffmpeg lee stdin para sus comandos
            # interactivos y se bloquea si hereda un pipe abierto sin EOF.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError as error:
            logger.warning("No se pudo extraer las señales visuales: %s", error)
            return None, None, None, 0, 0.0
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
        frame_bytes = size * size * 3
        frame_index = 0
        saw_face_frames = False
        cancelled = False
        try:
            while True:
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                saw_face_frames = True
                if detector is not None and face_timeline is not None:
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(size, size, 3).copy()
                    second = min(int(frame_index / _FACE_FPS), seconds - 1)
                    try:
                        _, detections = detector.detect(frame)
                    except Exception as error:  # noqa: BLE001 - un frame malo no tumba el canal
                        logger.warning(
                            "La detección de rostros falló en el segundo %d: %s",
                            second,
                            error,
                        )
                    else:
                        face_timeline[second] = max(
                            face_timeline[second],
                            _face_frame_value(detections, size, size),
                        )
                frame_index += 1
        except (OSError, ValueError) as error:
            logger.warning("La extracción de señales visuales falló a medias: %s", error)
        finally:
            if cancelled:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            drain_thread.join(timeout=10)
        if cancelled:
            return None, None, None, 0, 0.0
        stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace")
        cut_times = parse_showinfo_cut_times(stderr_text)
        scene_timeline, cut_count = _scene_timeline_from_cut_times(cut_times, duration)
        accumulator = _MotionAccumulator(seconds, _MOTION_FPS)
        motion_frame_bytes = _MOTION_WIDTH * _MOTION_HEIGHT
        try:
            with open(motion_path, "rb") as motion_file:
                while True:
                    data = motion_file.read(motion_frame_bytes)
                    if len(data) < motion_frame_bytes:
                        break
                    accumulator.add(np.frombuffer(data, dtype=np.uint8))
        except OSError as error:
            # ffmpeg falló antes de crear el archivo (p. ej. video
            # inexistente): sin frames de movimiento.
            logger.warning("No se pudo leer el movimiento extraído: %s", error)
        if accumulator.frames == 0:
            if not saw_face_frames:
                detail = stderr_text.strip().splitlines()
                logger.warning(
                    "ffmpeg no devolvió frames para las señales visuales (%s).",
                    detail[-1][:200] if detail else "sin pista de video",
                )
            return None, None, face_timeline if saw_face_frames else None, 0, 0.0
        motion_timeline, mean_raw = _normalize_motion(accumulator.raw_per_second())
        return (
            scene_timeline,
            motion_timeline,
            face_timeline if saw_face_frames else None,
            cut_count,
            mean_raw,
        )


def extract_visual_timelines(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    include_face: bool = True,
    scene_threshold: float = _SCENE_THRESHOLD,
    should_cancel: Callable[[], bool] | None = None,
) -> VisualTimelines:
    """Extrae todas las señales visuales (el punto de entrada del pipeline).

    Con rostros, los tres canales salen de UNA decodificación
    (``_extract_all_visual``); sin rostros, cortes + movimiento usan
    su propia pasada. Nunca lanza: cada canal que falla queda en None
    y el motor sigue con el resto.
    """
    if include_face:
        scene, motion, face, cut_count, mean_raw = _extract_all_visual(
            input_path,
            ffmpeg_path,
            duration,
            scene_threshold=scene_threshold,
            should_cancel=should_cancel,
        )
    else:
        scene, motion, cut_count, mean_raw = _extract_scene_and_motion(
            input_path,
            ffmpeg_path,
            duration,
            scene_threshold=scene_threshold,
            should_cancel=should_cancel,
        )
        face = None
    return VisualTimelines(
        scene_cuts=scene,
        motion=motion,
        face=face,
        cut_count=cut_count,
        mean_motion_raw=mean_raw,
        duration=duration,
    )
