"""Eventos de audio Tier 1 (Fase B, v1.1.0): YAMNet ONNX bajo demanda.

El canal de eventos del motor de señales: detecta risa, aplauso,
ovación y gritos en el audio y los convierte en el ``event_timeline``
(1 Hz, 0–1 por segundo) que consume ``backend.moments.detect_moments``
con peso 1.5. Sin este canal, el motor sigue con Tier 0 (señales
puras); Tier 1 es una mejora opcional, NUNCA un requisito.

Modelo elegido (verificado el 19-11-2026, no por teoría):
``audiomagic/yamnet-onnx`` (Hugging Face), exportación ONNX del YAMNet
de Google (Apache 2.0, sin pesos reentrenados). Archivo
``yamnet.onnx``: 16.093.355 bytes, sha256 verificado abajo. Cargado
con onnxruntime 1.32.0: entrada ``waveform`` float32 16 kHz mono,
salida ``output_0`` [frames, 521] (una fila por frame de 0,96 s con
salto de 0,48 s). Comportamiento de referencia reproducido: seno de
1 kHz -> "Beep, bleep" (0,91), silencio -> "Silence" (1,00), ruido
blanco -> "White noise" (0,20). Velocidad medida en esta VM (1,5 CPU):
~3,2 ms/frame, ~24 s para un video de 1 hora. EfficientAT MN10
(mAP 0.47, mejor que YAMNet 0.306) se descartó como primera opción
porque no hay un .onnx listo verificado: habría que exportarlo con
torch y distribuirlo aparte; si YAMNet decepciona en la Fase E, esa
es la vía documentada, sin cambiar esta interfaz (mismas clases
AudioSet, mismos umbrales por familia).

Descarga bajo demanda: el MODELO (~15 MB) se descarga en el primer
uso del Tier 1 con el mismo patrón que Whisper (``huggingface_hub``
+ callback de progreso + reintento); el RUNTIME (onnxruntime) sí va
en el instalador. Sin descarga, sin internet o sin onnxruntime, el
motor sigue con Tier 0 automáticamente: ``get_event_timeline`` nunca
lanza (salvo cancelación del usuario) y devuelve None.

Umbrales por clase (valores iniciales; la Fase E los calibra con
videos reales de Kelly): 0.5 es el punto de decisión natural de
YAMNet (puntuaciones por clase independientes en [0, 1]; el ejemplo
oficial de TensorFlow muestra como presentes las clases >= 0.5).
Risa/aplauso/ovación tienen firmas espectrales distintivas (armónicos
de risa, transitorios de palmas, masa de voces) y usan 0.5. La
familia grito se confunde con habla enfática fuerte — el contenido
más común en los videos — y usa 0.6 para recortar falsos positivos.
Cada segundo del timeline guarda la puntuación máxima por encima del
umbral (0.0 si ninguna clase lo supera), así el scorer distingue un
evento fuerte de uno débil.

Sin PyTorch jamás: todo corre en ONNX Runtime. Sin interruptor de
usuario en v1.1.0: Tier 1 es automático con fallback silencioso (el
parámetro ``enabled`` de ``get_event_timeline`` permite cablear un
ajuste después sin tocar este módulo).
"""

from __future__ import annotations

import hashlib
import logging
import math
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

# --- Modelo YAMNet ----------------------------------------------------------
_YAMNET_REPO_ID = "audiomagic/yamnet-onnx"
_YAMNET_FILENAME = "yamnet.onnx"
# sha256 del yamnet.onnx descargado de Hugging Face el 19-11-2026
# (16.093.355 bytes). Se verifica tras la descarga: un archivo corrupto
# en caché desactiva Tier 1 (Tier 0) en vez de producir eventos basura.
_YAMNET_SHA256 = "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e"
_YAMNET_SAMPLE_RATE = 16000
_YAMNET_FRAME_SECONDS = 0.96
_YAMNET_HOP_SECONDS = 0.48
_YAMNET_FRAME_SAMPLES = 15360  # 0.96 s a 16 kHz
_YAMNET_HOP_SAMPLES = 7680  # 0.48 s a 16 kHz
_YAMNET_NUM_CLASSES = 521
_YAMNET_INPUT_NAME = "waveform"
_YAMNET_SCORES_OUTPUT = "output_0"  # Solo se piden las puntuaciones:
# embeddings (1024) y log-mel (96x64 por frame) no se necesitan y pedirlos
# costaría memoria y tiempo en videos largos.
_YAMNET_DOWNLOAD_ATTEMPTS = 2
_YAMNET_RETRY_DELAY_SECONDS = 1.0
# Inferencia por bloques de 60 s (múltiplo exacto del salto: 125 frames
# por bloque) + 0,48 s de solape, para que la memoria sea O(1) aunque el
# video dure horas: el waveform completo de 4 h serían ~920 MB.
_CHUNK_SECONDS = 60.0
_CHUNK_SAMPLES = int(_CHUNK_SECONDS * _YAMNET_SAMPLE_RATE)
_FRAMES_PER_CHUNK = _CHUNK_SAMPLES // _YAMNET_HOP_SAMPLES
# Lectura de ffmpeg por bloques de 5 s (igual que moments.extract_audio_timeline).
_FFMPEG_READ_SECONDS = 5


@dataclass(frozen=True)
class _EventClass:
    """Una clase AudioSet de YAMNet con su umbral y su familia."""

    yamnet_index: int
    threshold: float
    kind: str  # Familia para los tramos y las etiquetas: risa/aplauso/ovación/grito.
    label: str  # Nombre legible de la familia.


# Índices según yamnet_class_map.csv del repo (verificado 19-11-2026):
# Laughter 13, Baby laughter 14, Giggle 15, Chuckle 18, Clapping 58,
# Cheering 61, Applause 62, Shout 6, Whoop 8, Yell 9, Children shouting
# 10, Screaming 11.
_EVENT_CLASSES: tuple[_EventClass, ...] = (
    _EventClass(13, 0.5, "risa", "Risa"),
    _EventClass(14, 0.5, "risa", "Risa"),
    _EventClass(15, 0.5, "risa", "Risa"),
    _EventClass(18, 0.5, "risa", "Risa"),
    _EventClass(62, 0.5, "aplauso", "Aplauso"),
    _EventClass(58, 0.5, "aplauso", "Aplauso"),
    _EventClass(61, 0.5, "ovación", "Ovación"),
    _EventClass(6, 0.6, "grito", "Grito"),
    _EventClass(9, 0.6, "grito", "Grito"),
    _EventClass(8, 0.6, "grito", "Grito"),
    _EventClass(10, 0.6, "grito", "Grito"),
    _EventClass(11, 0.6, "grito", "Grito"),
)
# Umbrales por familia, para la UI y los tests (derivado de _EVENT_CLASSES).
EVENT_THRESHOLDS: dict[str, float] = {
    kind: min(item.threshold for item in _EVENT_CLASSES if item.kind == kind)
    for kind in ("risa", "aplauso", "ovación", "grito")
}


class AudioEventError(Exception):
    """Tier 1 no disponible (sin runtime, sin modelo, sin audio)."""


class AudioEventsCancelled(Exception):
    """El usuario canceló la descarga o la inferencia del Tier 1.

    A diferencia de AudioEventError, NO se traga: el pipeline debe
    traducirla a su propia cancelación (ProjectCancelled).
    """


@dataclass(frozen=True)
class AudioEvent:
    """Un tramo continuo de la misma familia de evento."""

    start: float
    end: float
    kind: str  # "risa" | "aplauso" | "ovación" | "grito"
    label: str  # "Risa" | "Aplauso" | "Ovación" | "Grito"
    score: float  # Puntuación máxima del tramo (por encima del umbral).


@dataclass
class AudioEventResult:
    """Salida del Tier 1: timeline para el scorer + tramos para la UI."""

    timeline: np.ndarray  # 1 Hz, 0–1 por segundo (0.0 = sin evento).
    events: list[AudioEvent] = field(default_factory=list)


# --- descarga del modelo ----------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_yamnet_model(
    on_progress: ProgressCallback | None = None,
    *,
    cache_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Descarga yamnet.onnx en el primer uso (caché de huggingface_hub).

    Mismo patrón que la descarga de Whisper en media.py: snapshot con
    tqdm + callback de progreso (0–95 durante la descarga, 100 al
    terminar) y reintento ante fallos de red. Si el archivo ya está en
    caché, no descarga nada y devuelve su ruta. Lanza AudioEventError
    si la descarga falla o el sha256 no coincide; AudioEventsCancelled
    si el usuario cancela (no se reintenta).
    """
    if on_progress is not None:
        on_progress(0)

    class ModelDownloadProgress(tqdm):
        def update(self, n: int = 1) -> bool:
            if should_cancel is not None and should_cancel():
                raise AudioEventsCancelled("Descarga del modelo YAMNet cancelada.")
            updated = super().update(n)
            if self.total and on_progress is not None:
                percent = min(95.0, self.n / self.total * 95)
                on_progress(percent)
            return updated

    last_error: Exception | None = None
    for attempt in range(1, _YAMNET_DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_dir = snapshot_download(
                repo_id=_YAMNET_REPO_ID,
                allow_patterns=[_YAMNET_FILENAME],
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                max_workers=1,
                tqdm_class=ModelDownloadProgress,
            )
        except AudioEventsCancelled:
            raise
        except Exception as error:  # Red caída, repo no disponible, etc.
            last_error = error
            logger.warning(
                "Descarga de YAMNet fallida (intento %d/%d): %s",
                attempt,
                _YAMNET_DOWNLOAD_ATTEMPTS,
                error,
            )
            if attempt < _YAMNET_DOWNLOAD_ATTEMPTS:
                time.sleep(_YAMNET_RETRY_DELAY_SECONDS)
            continue
        model_path = Path(snapshot_dir) / _YAMNET_FILENAME
        if not model_path.is_file():
            last_error = AudioEventError(f"La descarga de YAMNet terminó sin {_YAMNET_FILENAME}.")
            break
        if _sha256_file(model_path) != _YAMNET_SHA256:
            # No se reintenta: la caché devolvería el mismo archivo
            # corrupto. Tier 0 hasta que se limpie la caché de HF.
            raise AudioEventError(
                "El modelo YAMNet descargado no coincide con su sha256; "
                "se usa Tier 0. Borra la caché de Hugging Face para reintentarlo."
            )
        if on_progress is not None:
            on_progress(100)
        return model_path
    raise AudioEventError(
        f"No se pudo descargar el modelo YAMNet tras {_YAMNET_DOWNLOAD_ATTEMPTS} "
        f"intentos: {last_error}"
    ) from last_error


def is_tier1_available() -> bool:
    """True si Tier 1 puede correr AHORA sin descargar nada.

    Para un futuro indicador en la UI: onnxruntime importable y modelo
    en caché. Nunca descarga ni lanza.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    try:
        hf_hub_download(
            repo_id=_YAMNET_REPO_ID,
            filename=_YAMNET_FILENAME,
            local_files_only=True,
        )
    except Exception:
        return False
    return True


# --- sesión ONNX ------------------------------------------------------------

_SESSION_CACHE: tuple[Path, Any] | None = None
_SESSION_LOCK = threading.Lock()


def _reset_session_cache() -> None:
    """Solo para tests: olvida la sesión cacheada."""
    global _SESSION_CACHE
    with _SESSION_LOCK:
        _SESSION_CACHE = None


def load_yamnet_session(model_path: Path | str) -> Any:
    """Carga (y cachea por ruta) la sesión ONNX de YAMNet.

    Importación perezosa de onnxruntime: este módulo debe importarse
    aunque el runtime no esté instalado (Tier 0). Lanza
    AudioEventError si falta onnxruntime o el modelo no carga.
    """
    global _SESSION_CACHE
    path = Path(model_path)
    with _SESSION_LOCK:
        if _SESSION_CACHE is not None and _SESSION_CACHE[0] == path:
            return _SESSION_CACHE[1]
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise AudioEventError(
            "El Tier 1 necesita onnxruntime, que no está instalado. Se usa Tier 0."
        ) from error
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise AudioEventError(f"No se pudo cargar el modelo YAMNet ({path}): {error}") from error
    with _SESSION_LOCK:
        _SESSION_CACHE = (path, session)
    return session


# --- puntuaciones -> timeline -------------------------------------------------


def scores_to_timeline(scores: np.ndarray, duration: float) -> AudioEventResult:
    """Convierte puntuaciones YAMNet [frames, 521] en timeline 1 Hz.

    Cada frame (0,96 s, salto 0,48 s) se asigna al segundo de su
    centro. El valor de cada segundo es la puntuación máxima entre las
    clases de evento que superan su umbral (0.0 si ninguna lo supera).
    Los tramos agrupan segundos consecutivos de la misma familia.
    Función pura: no toca red ni disco, para tests y calibración.
    """
    seconds = max(1, math.ceil(duration))
    timeline = np.zeros(seconds, dtype=np.float64)
    best_kind: list[str | None] = [None] * seconds
    best_label: list[str] = [""] * seconds
    if scores.size == 0:
        return AudioEventResult(timeline=timeline)
    # Frames más allá de la duración (p. ej. el frame de cola con
    # zero-padding) se descartan por el centro: su segundo queda fuera.
    for frame_idx in range(scores.shape[0]):
        center = frame_idx * _YAMNET_HOP_SECONDS + _YAMNET_FRAME_SECONDS / 2
        second = math.floor(center)
        if second < 0 or second >= seconds:
            continue
        row = scores[frame_idx]
        for event_class in _EVENT_CLASSES:
            if event_class.yamnet_index >= row.size:
                continue
            value = float(row[event_class.yamnet_index])
            if value >= event_class.threshold and value > timeline[second]:
                timeline[second] = value
                best_kind[second] = event_class.kind
                best_label[second] = event_class.label
    events: list[AudioEvent] = []
    span_start: int | None = None
    span_kind: str | None = None
    span_label = ""
    span_score = 0.0
    for second in range(seconds + 1):
        kind = best_kind[second] if second < seconds else None
        if kind is not None and kind == span_kind:
            span_score = max(span_score, float(timeline[second]))
            continue
        if span_start is not None and span_kind is not None:
            events.append(
                AudioEvent(
                    start=float(span_start),
                    end=min(float(second), float(duration)) if duration > 0 else float(second),
                    kind=span_kind,
                    label=span_label,
                    score=span_score,
                )
            )
        if kind is not None:
            span_start = second
            span_kind = kind
            span_label = best_label[second]
            span_score = float(timeline[second])
        else:
            span_start = None
            span_kind = None
    return AudioEventResult(timeline=timeline, events=events)


# --- inferencia por bloques ---------------------------------------------------


class _ChunkedInference:
    """Inferencia YAMNet en bloques de 60 s con solape de 0,48 s.

    Mantiene la memoria O(1): solo guarda el bloque actual más el
    solape. Cada bloque completo aporta exactamente 125 frames (los que
    empiezan dentro del bloque); el último bloque, más corto, aporta
    los frames que cubren la cola. ``feed`` devuelve los bloques de
    puntuaciones listos; ``finish`` procesa la cola pendiente.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._pending = np.empty(0, dtype=np.float32)

    def _run(self, waveform: np.ndarray) -> np.ndarray:
        outputs = self._session.run([_YAMNET_SCORES_OUTPUT], {_YAMNET_INPUT_NAME: waveform})
        return np.asarray(outputs[0], dtype=np.float64)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        if samples.size:
            self._pending = (
                np.concatenate([self._pending, samples])
                if self._pending.size
                else np.asarray(samples, dtype=np.float32)
            )
        blocks: list[np.ndarray] = []
        needed = _CHUNK_SAMPLES + _YAMNET_HOP_SAMPLES
        while self._pending.size >= needed:
            chunk = self._pending[:needed]
            scores = self._run(chunk)
            blocks.append(scores[:_FRAMES_PER_CHUNK])
            self._pending = self._pending[_CHUNK_SAMPLES:]
        return blocks

    def finish(self) -> list[np.ndarray]:
        if self._pending.size == 0:
            return []
        scores = self._run(self._pending)
        self._pending = np.empty(0, dtype=np.float32)
        return [scores]


# --- entrada principal --------------------------------------------------------


def detect_audio_events(
    input_path: Path | str,
    duration: float,
    ffmpeg_path: str,
    *,
    session: Any | None = None,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AudioEventResult:
    """Detecta eventos de audio en un video y devuelve timeline + tramos.

    Extrae PCM float32 16 kHz mono con ffmpeg (streaming) y corre
    YAMNet por bloques. Si no se pasa ``session``, descarga el modelo
    en el primer uso y cachea la sesión. Lanza AudioEventError si
    ffmpeg falla, el video no tiene audio o Tier 1 no está disponible;
    AudioEventsCancelled si el usuario cancela.
    """
    if session is None:
        model_path = ensure_yamnet_model(on_progress, should_cancel=should_cancel)
        session = load_yamnet_session(model_path)
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_YAMNET_SAMPLE_RATE),
        "-f",
        "f32le",
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
        raise AudioEventError(f"No se pudo ejecutar ffmpeg para el Tier 1: {error}") from error
    assert process.stdout is not None
    inference = _ChunkedInference(session)
    score_blocks: list[np.ndarray] = []
    saw_samples = False
    read_bytes = _YAMNET_SAMPLE_RATE * 4 * _FFMPEG_READ_SECONDS
    try:
        while True:
            if should_cancel is not None and should_cancel():
                raise AudioEventsCancelled("Detección de eventos de audio cancelada.")
            data = process.stdout.read(read_bytes)
            if not data:
                break
            saw_samples = True
            samples = np.frombuffer(data, dtype=np.float32)
            score_blocks.extend(inference.feed(samples))
        score_blocks.extend(inference.finish())
    except AudioEventsCancelled:
        raise
    except (OSError, ValueError) as error:
        raise AudioEventError(f"La extracción de audio para el Tier 1 falló: {error}") from error
    finally:
        try:
            process.kill()
        except OSError:
            pass
        try:
            _, stderr = process.communicate(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            stderr = b""
    if not saw_samples:
        detail = stderr.decode("utf-8", "replace").strip()[:200]
        raise AudioEventError(
            f"ffmpeg no devolvió audio para el Tier 1 ({detail or 'sin pista de audio'})."
        )
    if not score_blocks:
        return AudioEventResult(timeline=np.zeros(max(1, math.ceil(duration))))
    scores = score_blocks[0] if len(score_blocks) == 1 else np.concatenate(score_blocks, axis=0)
    return scores_to_timeline(scores, duration)


def get_event_timeline(
    input_path: Path | str | None,
    duration: float,
    ffmpeg_path: str | None,
    *,
    enabled: bool = True,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """Timeline de eventos para ``moments.detect_moments`` (o None).

    El punto de entrada que cablea el pipeline: nunca lanza por
    problemas del Tier 1 (sin runtime, sin modelo, sin red, sin audio)
    — registra un aviso y devuelve None para que el motor siga con
    Tier 0. Solo propaga AudioEventsCancelled, que el pipeline debe
    traducir a su propia cancelación.
    """
    if not enabled or input_path is None or not ffmpeg_path:
        return None
    try:
        result = detect_audio_events(
            input_path,
            duration,
            ffmpeg_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    except AudioEventsCancelled:
        raise
    except AudioEventError as error:
        logger.warning("Tier 1 no disponible (%s); el motor sigue con Tier 0.", error)
        return None
    except Exception as error:  # Defensa: Tier 1 jamás rompe un proyecto.
        logger.warning(
            "El Tier 1 falló de forma inesperada (%s); el motor sigue con Tier 0.",
            error,
        )
        return None
    logger.info(
        "Tier 1: %d evento(s) de audio detectados (%s).",
        len(result.events),
        ", ".join(f"{event.kind} {event.start:.0f}s" for event in result.events[:5]) or "ninguno",
    )
    return result.timeline
