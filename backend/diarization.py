"""Diarización de hablantes Tier 2 (Fase C, v1.2.0): sherpa-onnx sin sherpa-onnx.

El canal de diarización del motor de señales: detecta quién habla
cuándo y lo convierte en el ``speaker_timeline`` (1 Hz, 0–1 por
segundo, densidad de turnos de palabra) que consume
``backend.moments.detect_moments``. Sin este canal, el motor sigue
con Tier 0 + Tier 1; la diarización es una mejora opcional, NUNCA un
requisito.

DECISIÓN DE RUNTIME (verificada el 21-11-2026, no por teoría):
onnxruntime puro + numpy, SIN el paquete pip ``sherpa-onnx`` (0
dependencias nuevas). Los dos modelos ONNX de sherpa-onnx corren
directamente en onnxruntime 1.32.0; el único hueco era el fbank
estilo Kaldi que el extractor de embeddings espera, y está
reimplementado aquí en numpy (``_kaldi_fbank``) siguiendo
kaldi-native-fbank al detalle (ventana Povey, preénfasis 0.97,
80 bins mel Kaldi, log con suelo FLT_EPSILON, reflexión simétrica
en los bordes, media global restada). Verificación contra el
oráculo: embeddings míos vs ``sherpa_onnx.SpeakerEmbeddingExtractor``
sobre los mismos clips, similitud coseno 1.000000 en todos los
casos (16 s, 3 s, 1 s, 0,5 s). Y diarización completa vs
``sherpa_onnx.OfflineSpeakerDiarization``: en el wav de 16 s los
segmentos son IDÉNTICOS (100% de solape con etiqueta correcta); en
el de 34 s mi pipeline incluso detecta un turno real a ~24 s que el
oráculo con num_clusters=2 forzado se traga (confirmado con clips
limpios de 2 s: la cola 24,5–26,5 s se parece a sp1 con 0,63 y a
sp0 con 0,11). El wheel pip ``sherpa-onnx`` 2.0.2 existe (2,46 MB
win_amd64; 4,63 MB manylinux x86_64) y solo se usó como oráculo de
verificación en un venv aparte; la app no lo necesita.

Modelos elegidos (verificados el 21-11-2026, descargados y con
sha256 medido; copias de referencia en ~/workspace/kca-models/
diarization/, fuera del repo):

* Segmentación ``sherpa-onnx-pyannote-segmentation-3-0`` (tarball
  del release ``speaker-segmentation-models`` de k2-fsa/sherpa-onnx,
  conversión ONNX del pyannote/segmentation-3.0, MIT): se usa
  ``model.onnx`` (5.992.913 bytes, FP32). Entrada waveform float32
  16 kHz mono [N, 1, T]; salida [N, 589, 7] por ventana de 10 s
  (7 clases powerset de hasta 3 hablantes; salto de ventana 1 s;
  frame 16,875 ms). El ``model.int8.onnx`` (1,5 MB) carga y da la
  misma forma de salida, pero solo es ~15% más rápido y se queda
  como opción futura: el FP32 ya es pequeño frente al embedding.
* Embedding ``3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx``
  (release ``speaker-recongition-models`` — el tag lleva esa falta
  de ortografía —, 3D-Speaker ERes2Net entrenado en VoxCeleb,
  Apache 2.0, 26.485.264 bytes): entrada fbank [N, T, 80], salida
  embedding de 192 dimensiones. Elegido frente a WeSpeaker
  ResNet34-LM (mismo tamaño, separabilidad similar) y TitaNet-small
  (38 MB, peor separabilidad medida por terceros): es el menor de
  los buenos y el mejor medido (mismo hablante p50 0,52 vs distinto
  0,05 en ventanas de 1,5 s; mejor split sim 0,24).

Descarga total: ~33,4 MB bajo demanda en el primer uso, desde los
releases de GitHub de k2-fsa/sherpa-onnx (sin token ni cuenta),
con urllib (stdlib) + callback de progreso + reintento, igual
semántica que la descarga de YAMNet pero sin huggingface_hub
(los modelos no están en HF como artefactos oficiales de
sherpa-onnx). El runtime (onnxruntime) sí va en el instalador.
Sin descarga, sin internet o sin onnxruntime, el motor sigue sin
este canal: ``get_speaker_timeline`` nunca lanza (salvo
cancelación del usuario) y devuelve None.

Pipeline (réplica del script de referencia
speaker-diarization-onnx.py del tarball, con tres diferencias
documentadas): ventanas de segmentación de 10 s con salto de 1 s
en lotes de 32 -> clases powerset -> un embedding por par
(chunk, slot) activo >= 10 frames -> clustering aglomerativo de
enlace completo (el mismo algoritmo que FastClustering de sherpa:
disimilitud coseno, corte por umbral) -> voto mayoritario por
frame -> segmentos con min_duration_on/off 0,3/0,5 s (defaults
de sherpa) -> timeline de turnos 1 Hz.

Diferencias con la referencia (medidas, no caprichos):
1. Embeddings cada 2 chunks (stride 2) y con tope de 2,0 s de
   audio por par (fragmentos más largos primero): la identidad no
   necesita 10 s de voz ni un embedding por segundo; el coste de
   embeddings cae ~5x y cada frame sigue cubierto por ~5 ventanas
   embebidas para el voto. Verificado: los segmentos coinciden
   con el oráculo salvo ~1 s en un borde de turno.
2. Los pares no embebidos heredan la etiqueta del chunk embebido
   más cercano con el mismo slot (±5 chunks) SIEMPRE QUE la
   actividad del slot se solape en el tiempo (>= 50% de sus frames
   activos): las ventanas se solapan un 90% y el slot suele ser
   estable, pero en un turno el modelo puede reasignar slots entre
   ventanas y heredar a ciegas etiqueta mal (en el wav de 34 s
   movía el primer turno de 2,76 s a 4,03 s). Con solape exigido,
   la herencia solo refuerza votos donde el slot es el mismo
   hablante: el primer turno queda en 2,76 s y el de la cola en
   22,96 s (real ~24,0 s).
3. Tras el corte por umbral, se fusionan clusters cuyos centroides
   sean muy parecidos (sim > 0,5): corrige el sobre-fragmentado
   del enlace completo, que parte a un hablante en dos y crea
   turnos falsos en el timeline.

Umbrales (medidos el 21-11-2026 con etiquetas del oráculo):
umbral de clustering en disimilitud coseno 0,76 (sim 0,24: el
mejor split medido por terceros en reuniones reales; en el wav
de 16 s la separación es perfecta — mismo hablante p5 0,61 vs
distinto p95 0,13 — y en el de 34 s, más ruidoso, el mejor split
es 0,41 con un 8% de error por pares). Turnos: un turno es un
cambio de hablante entre segmentos consecutivos; el valor de cada
segundo es el número de turnos en una ventana centrada de ±10 s,
con 4 turnos = 1,0: una entrevista tranquila (un turno cada
~20 s) puntúa ~0,25, un debate vivo (un turno cada ~5 s) puntúa
1,0 y un monólogo 0,0. PESO SUGERIDO para el scorer: 0,75
(señal media según el informe; por debajo de los canales
principales 1,0 y de eventos 1,5, por encima de los visuales
débiles 0,5; el canal debe usar la misma forma que
``_events_channel``: 0,5 * pico + 0,5 * fracción por encima de
0,5). La Fase E lo calibra con videos reales de Kelly.

Velocidad medida en esta VM (1,5 vCPU compartidas, muy por debajo
del PC de Kelly): segmentación ~93 ms/ventana en lotes de 32
(~5,5 min por hora de video) y embeddings ~1,1 ms por frame de
10 ms (~2 s de audio por par embebido); RTF total ~0,22–0,32 en
esta VM, con sherpa-onnx en la misma máquina en 0,13–0,31 (el
oráculo no es más rápido aquí). En el Xeon de Kelly (14 núcleos)
se espera RTF ~0,05–0,09, en línea con lo publicado. Memoria O(1)
en el waveform (streaming por bloques de 5 s desde ffmpeg, buffer
pendiente < 41 s); solo crecen con la duración las etiquetas
compactas int8 (~2 KB/s) y los embeddings (~1,5 KB/s), más la
matriz de clustering, topada en 1500 embeddings por submuestra
uniforme + asignación por centroide más cercano.

Sin PyTorch jamás: todo corre en ONNX Runtime. Sin interruptor de
usuario en v1.2.0: la diarización es automática con fallback
silencioso (el parámetro ``enabled`` de ``get_speaker_timeline``
permite cablear un ajuste después sin tocar este módulo).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

# --- Modelos (verificados el 21-11-2026) --------------------------------------
_SEGMENTATION_TARBALL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_SEGMENTATION_TARBALL_FILENAME = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
_SEGMENTATION_TARBALL_SIZE = 6958444
_SEGMENTATION_TARBALL_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
_SEGMENTATION_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_SEGMENTATION_FILENAME = "pyannote-segmentation-3-0.onnx"
_SEGMENTATION_SIZE = 5992913
_SEGMENTATION_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
)
_EMBEDDING_FILENAME = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
_EMBEDDING_SIZE = 26485264
_EMBEDDING_SHA256 = "c4947caf150b8502fdd75bbe4089b971ab1ec42e90ee3ba547faca2cde44d9ee"
_DOWNLOAD_ATTEMPTS = 2
_DOWNLOAD_RETRY_DELAY_SECONDS = 1.0
_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_CHUNK_BYTES = 1 << 20
# Reparto del progreso 0–95 entre los dos archivos (proporcional al
# tamaño: el tarball es ~21% de los ~33,4 MB totales).
_SEGMENTATION_PROGRESS_END = 20.0

# --- Audio y segmentación ------------------------------------------------------
_SAMPLE_RATE = 16000
_WINDOW_SIZE = 160000  # 10 s por ventana de segmentación
_WINDOW_SHIFT = 16000  # 1 s de salto entre ventanas
_RECEPTIVE_FIELD_SHIFT = 270  # muestras por frame (16,875 ms)
_RECEPTIVE_FIELD_SIZE = 991
_FRAMES_PER_WINDOW = 589
_NUM_LOCAL_SPEAKERS = 3  # slots por ventana (pyannote-segmentation-3.0)
_NUM_CLASSES = 7  # clases powerset (silencio + 3 solos + 3 pares)
_SEG_BATCH = 32  # ventanas por lote de inferencia
_FFMPEG_READ_SECONDS = 5  # lectura de ffmpeg por bloques (como audio_events)

# --- Embeddings -----------------------------------------------------------------
_EMBEDDING_DIM = 192
_EMBED_STRIDE = 2  # un embedding cada 2 chunks (2 s)
# Hilos para los embeddings: medido en un video de 320 s, las 161
# llamadas secuenciales al modelo de embedding costaban 33,6 s de los
# 54 s de diarización (~0,2 s cada una, fbank + inferencia). Son
# independientes entre sí y la sesión ONNX admite run() concurrente,
# así que van a un pool; los resultados se recogen en orden de envío
# (finish), idénticos a la ejecución secuencial. 4 hilos es un
# equilibrio: en el PC de Kelly (28 hilos) sobra CPU para Whisper y
# ffmpeg a la vez; en equipos pequeños el pool no empeora nada frente
# a la secuencia (mismo trabajo total).
_EMBED_WORKERS = 4
_MAX_EMBED_SECONDS = 2.0  # tope de audio por par (chunk, slot)
_MIN_ACTIVE_FRAMES = 10  # regla del script de referencia (~0,17 s)
_MIN_EMBED_SECONDS = 0.25  # por debajo, el embedding no sirve
_MAX_CLUSTER_EMBEDDINGS = 1500  # tope de la matriz O(N²) de clustering
_CLUSTER_THRESHOLD = 0.76  # disimilitud coseno (sim 0,24)
_CENTROID_MERGE_SIM = 0.5  # fusión de clusters con centroides parecidos
_INHERIT_RADIUS = 5  # chunks para heredar la etiqueta de un par no embebido

# --- Fbank estilo Kaldi (ver _kaldi_fbank) ---------------------------------------
_FBANK_FRAME_LENGTH = 400  # 25 ms
_FBANK_FRAME_SHIFT = 160  # 10 ms
_FBANK_FFT_SIZE = 512  # round_to_power_of_two
_FBANK_NUM_MEL_BINS = 80
_FBANK_LOW_FREQ = 20.0
_FBANK_HIGH_FREQ = 7600.0  # Nyquist (8000) - 400
_FBANK_PREEMPH_COEFF = 0.97
_FBANK_LOG_FLOOR = float(np.finfo(np.float32).eps)  # 1.1920929e-07

# --- Segmentos y timeline ---------------------------------------------------------
_MIN_DURATION_ON = 0.3  # default de sherpa-onnx
_MIN_DURATION_OFF = 0.5  # default de sherpa-onnx
_TURN_WINDOW_SECONDS = 10.0  # ventana centrada ±10 s para densidad de turnos
_TURNS_FOR_MAX = 4.0  # 4 turnos en 20 s => 1.0 (debate vivo)

# Mapeo powerset (clase -> slots activos), igual que get_powerset_mapping
# del script de referencia de sherpa-onnx.
_POWERSET = np.zeros((_NUM_CLASSES, _NUM_LOCAL_SPEAKERS), dtype=np.int8)
_POWERSET[1] = (1, 0, 0)
_POWERSET[2] = (0, 1, 0)
_POWERSET[3] = (0, 0, 1)
_POWERSET[4] = (1, 1, 0)
_POWERSET[5] = (1, 0, 1)
_POWERSET[6] = (0, 1, 1)


class DiarizationError(Exception):
    """Diarización no disponible (sin runtime, sin modelo, sin audio)."""


class DiarizationCancelled(Exception):
    """El usuario canceló la descarga o la inferencia de diarización.

    A diferencia de DiarizationError, NO se traga: el pipeline debe
    traducirla a su propia cancelación (ProjectCancelled).
    """


@dataclass(frozen=True)
class SpeakerSegment:
    """Un tramo continuo de un mismo hablante."""

    start: float
    end: float
    speaker: int  # Id de cluster (0, 1, 2, ...), estable en el video.

    @property
    def label(self) -> str:
        """Etiqueta legible para la UI: SPEAKER_00, SPEAKER_01, ..."""
        return f"SPEAKER_{self.speaker:02d}"


@dataclass
class DiarizationResult:
    """Salida de la diarización: timeline para el scorer + tramos para la UI."""

    timeline: np.ndarray  # 1 Hz, 0–1 por segundo (0.0 = sin turnos cerca).
    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0


# --- descarga de los modelos ----------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    """Carpeta de caché por defecto: <data_dir>/models/diarization.

    Importación perezosa de settings para no acoplar este módulo a la
    configuración de la app; si falla, caché en el home del usuario.
    """
    try:
        from .settings import get_settings

        return get_settings().data_dir / "models" / "diarization"
    except Exception:
        return Path.home() / ".cache" / "kellyclipai" / "diarization"


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return _default_cache_dir()


def _model_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "segmentation": cache_dir / _SEGMENTATION_FILENAME,
        "embedding": cache_dir / _EMBEDDING_FILENAME,
    }


def _file_is_valid(path: Path, size: int, sha256: str) -> bool:
    """True si el archivo existe con el tamaño y sha256 esperados."""
    try:
        if not path.is_file() or path.stat().st_size != size:
            return False
        return _sha256_file(path) == sha256
    except OSError:
        return False


def _download_file(
    url: str,
    dest: Path,
    expected_size: int,
    expected_sha256: str,
    on_progress: ProgressCallback | None,
    progress_start: float,
    progress_end: float,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Descarga un archivo con progreso, cancelación y verificación.

    Escribe a un temporal y lo mueve al destino solo si el tamaño y
    el sha256 coinciden; un temporal a medias se borra. Lanza
    DiarizationError si la red falla o el archivo no coincide;
    DiarizationCancelled si el usuario cancela.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "KellyClipAi"})
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                total = response.length if response.length else expected_size
                downloaded = 0
                with tmp_path.open("wb") as handle:
                    while True:
                        if should_cancel is not None and should_cancel():
                            raise DiarizationCancelled(f"Descarga de {dest.name} cancelada.")
                        block = response.read(_DOWNLOAD_CHUNK_BYTES)
                        if not block:
                            break
                        handle.write(block)
                        downloaded += len(block)
                        if on_progress is not None and total > 0:
                            fraction = min(1.0, downloaded / total)
                            on_progress(progress_start + fraction * (progress_end - progress_start))
        except DiarizationCancelled:
            tmp_path.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            tmp_path.unlink(missing_ok=True)
            logger.warning(
                "Descarga de %s fallida (intento %d/%d): %s",
                dest.name,
                attempt,
                _DOWNLOAD_ATTEMPTS,
                error,
            )
            if attempt < _DOWNLOAD_ATTEMPTS:
                time.sleep(_DOWNLOAD_RETRY_DELAY_SECONDS)
            continue
        if not _file_is_valid(tmp_path, expected_size, expected_sha256):
            tmp_path.unlink(missing_ok=True)
            # No se reintenta: el servidor devolvería el mismo archivo.
            raise DiarizationError(
                f"El archivo {dest.name} descargado no coincide con su tamaño o sha256 esperado."
            )
        os.replace(tmp_path, dest)
        return
    raise DiarizationError(
        f"No se pudo descargar {dest.name} tras {_DOWNLOAD_ATTEMPTS} intentos: {last_error}"
    ) from last_error


def _extract_segmentation_model(tarball_path: Path, dest: Path) -> None:
    """Extrae model.onnx del tarball de segmentación y lo verifica.

    Solo se extrae el miembro esperado (nunca rutas arbitrarias del
    tarball). El tarball se borra tras una extracción correcta para
    no guardar ~7 MB duplicados.
    """
    try:
        with tarfile.open(tarball_path, "r:bz2") as archive:
            member = archive.getmember(_SEGMENTATION_MEMBER)
            if not member.isfile():
                raise DiarizationError(
                    f"El tarball de segmentación no trae {_SEGMENTATION_MEMBER}."
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise DiarizationError(f"No se pudo leer {_SEGMENTATION_MEMBER} del tarball.")
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            with tmp_path.open("wb") as handle:
                while True:
                    block = extracted.read(_DOWNLOAD_CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
    except (tarfile.TarError, OSError, KeyError) as error:
        raise DiarizationError(f"El tarball de segmentación está corrupto: {error}") from error
    if not _file_is_valid(tmp_path, _SEGMENTATION_SIZE, _SEGMENTATION_SHA256):
        tmp_path.unlink(missing_ok=True)
        raise DiarizationError("El modelo de segmentación extraído no coincide con su sha256.")
    os.replace(tmp_path, dest)
    tarball_path.unlink(missing_ok=True)


def ensure_diarization_models(
    on_progress: ProgressCallback | None = None,
    *,
    cache_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Descarga los dos modelos ONNX en el primer uso (~33,4 MB).

    Misma semántica que la descarga de YAMNet: progreso 0–95 durante
    la descarga (segmentación 0–20, embedding 20–95) y 100 al
    terminar, reintento ante fallos de red, verificación de tamaño y
    sha256 de CADA archivo. Si ya están en caché y son válidos, no
    descarga nada. Lanza DiarizationError si la descarga falla o un
    archivo no coincide; DiarizationCancelled si el usuario cancela
    (no se reintenta).
    """
    if on_progress is not None:
        on_progress(0)
    root = _resolve_cache_dir(cache_dir)
    paths = _model_paths(root)
    if _file_is_valid(
        paths["segmentation"], _SEGMENTATION_SIZE, _SEGMENTATION_SHA256
    ) and _file_is_valid(paths["embedding"], _EMBEDDING_SIZE, _EMBEDDING_SHA256):
        if on_progress is not None:
            on_progress(100)
        return paths
    if not _file_is_valid(paths["segmentation"], _SEGMENTATION_SIZE, _SEGMENTATION_SHA256):
        tarball_path = root / _SEGMENTATION_TARBALL_FILENAME
        if not _file_is_valid(
            tarball_path, _SEGMENTATION_TARBALL_SIZE, _SEGMENTATION_TARBALL_SHA256
        ):
            _download_file(
                _SEGMENTATION_TARBALL_URL,
                tarball_path,
                _SEGMENTATION_TARBALL_SIZE,
                _SEGMENTATION_TARBALL_SHA256,
                on_progress,
                0.0,
                _SEGMENTATION_PROGRESS_END,
                should_cancel,
            )
        elif on_progress is not None:
            on_progress(_SEGMENTATION_PROGRESS_END)
        _extract_segmentation_model(tarball_path, paths["segmentation"])
    elif on_progress is not None:
        on_progress(_SEGMENTATION_PROGRESS_END)
    if not _file_is_valid(paths["embedding"], _EMBEDDING_SIZE, _EMBEDDING_SHA256):
        _download_file(
            _EMBEDDING_URL,
            paths["embedding"],
            _EMBEDDING_SIZE,
            _EMBEDDING_SHA256,
            on_progress,
            _SEGMENTATION_PROGRESS_END,
            95.0,
            should_cancel,
        )
    if on_progress is not None:
        on_progress(100)
    return paths


def is_diarization_available(cache_dir: str | Path | None = None) -> bool:
    """True si la diarización puede correr AHORA sin descargar nada.

    Para un futuro indicador en la UI: onnxruntime importable y los
    dos modelos en caché con tamaño y sha256 correctos. Nunca
    descarga ni lanza.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    try:
        paths = _model_paths(_resolve_cache_dir(cache_dir))
        return _file_is_valid(
            paths["segmentation"], _SEGMENTATION_SIZE, _SEGMENTATION_SHA256
        ) and _file_is_valid(paths["embedding"], _EMBEDDING_SIZE, _EMBEDDING_SHA256)
    except Exception:
        return False


# --- sesiones ONNX ------------------------------------------------------------

_SESSION_CACHE: dict[tuple[str, Path], Any] = {}
_SESSION_LOCK = threading.Lock()


def _reset_session_cache() -> None:
    """Solo para tests: olvida las sesiones cacheadas."""
    with _SESSION_LOCK:
        _SESSION_CACHE.clear()


def _load_session(kind: str, model_path: Path) -> Any:
    """Carga (y cachea por tipo y ruta) una sesión ONNX.

    Importación perezosa de onnxruntime: este módulo debe importarse
    aunque el runtime no esté instalado (Tier 0/1). Lanza
    DiarizationError si falta onnxruntime o el modelo no carga.
    """
    with _SESSION_LOCK:
        cached = _SESSION_CACHE.get((kind, model_path))
        if cached is not None:
            return cached
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise DiarizationError(
            "La diarización necesita onnxruntime, que no está instalado."
        ) from error
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise DiarizationError(
            f"No se pudo cargar el modelo de diarización ({model_path}): {error}"
        ) from error
    with _SESSION_LOCK:
        _SESSION_CACHE[(kind, model_path)] = session
    return session


def load_diarization_sessions(model_paths: dict[str, Path]) -> tuple[Any, Any]:
    """Carga las sesiones de segmentación y embedding (cacheadas)."""
    segmentation = _load_session("segmentation", Path(model_paths["segmentation"]))
    embedding = _load_session("embedding", Path(model_paths["embedding"]))
    return segmentation, embedding


# --- fbank estilo Kaldi (numpy) -------------------------------------------------

# Replica kaldi-native-fbank (el extractor que sherpa-onnx usa para el
# embedding) con la configuración exacta de FeatureExtractorConfig para
# modelos 3D-Speaker: 16 kHz, ventana 25 ms, salto 10 ms,
# snip_edges=False, remove_dc_offset=True, preénfasis 0,97, ventana
# Povey, dither=0, 80 bins mel Kaldi (low 20 Hz, high 7600 Hz),
# log(max(energía, FLT_EPSILON)), FFT 512. Verificado contra el
# oráculo sherpa_onnx: cos 1.000000 en todos los clips probados.
_MEL_BANKS: np.ndarray | None = None
_POVEY_WINDOW: np.ndarray | None = None


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    """Escala mel de Kaldi: 1127 * ln(1 + f/700)."""
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _get_mel_banks() -> np.ndarray:
    """Matriz (80, 256) de filtros triangulares mel estilo Kaldi."""
    global _MEL_BANKS
    if _MEL_BANKS is not None:
        return _MEL_BANKS
    num_fft_bins = _FBANK_FFT_SIZE // 2  # 256: bins 0..255 (sin Nyquist)
    fft_bin_width = _SAMPLE_RATE / _FBANK_FFT_SIZE
    mel_low = float(_mel_scale(np.array(_FBANK_LOW_FREQ)))
    mel_high = float(_mel_scale(np.array(_FBANK_HIGH_FREQ)))
    mel_delta = (mel_high - mel_low) / (_FBANK_NUM_MEL_BINS + 1)
    freqs = fft_bin_width * np.arange(num_fft_bins, dtype=np.float64)
    mels = _mel_scale(freqs)
    banks = np.zeros((_FBANK_NUM_MEL_BINS, num_fft_bins), dtype=np.float32)
    for b in range(_FBANK_NUM_MEL_BINS):
        left = mel_low + b * mel_delta
        center = mel_low + (b + 1) * mel_delta
        right = mel_low + (b + 2) * mel_delta
        active = (mels > left) & (mels < right)
        weights = np.where(
            mels <= center,
            (mels - left) / (center - left),
            (right - mels) / (right - center),
        )
        banks[b, active] = weights[active].astype(np.float32)
    _MEL_BANKS = banks
    return banks


def _get_povey_window() -> np.ndarray:
    """Ventana Povey: (0,5 - 0,5*cos(2*pi*i/(N-1)))^0,85."""
    global _POVEY_WINDOW
    if _POVEY_WINDOW is not None:
        return _POVEY_WINDOW
    n = _FBANK_FRAME_LENGTH
    a = 2.0 * np.pi / (n - 1)
    i = np.arange(n, dtype=np.float64)
    _POVEY_WINDOW = ((0.5 - 0.5 * np.cos(a * i)) ** 0.85).astype(np.float32)
    return _POVEY_WINDOW


def _kaldi_fbank(waveform: np.ndarray) -> np.ndarray:
    """Fbank Kaldi (num_frames, 80) float32 para un waveform 16 kHz mono.

    Número de frames con snip_edges=False y flush: (N + 80) // 160.
    El primer frame empieza en la muestra -120 (ventana centrada) y
    los bordes se rellenan por reflexión simétrica, como
    ExtractWindow de Kaldi. Función pura (sin red ni disco).
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    n = waveform.shape[0]
    if n == 0:
        return np.empty((0, _FBANK_NUM_MEL_BINS), dtype=np.float32)
    num_frames = (n + _FBANK_FRAME_SHIFT // 2) // _FBANK_FRAME_SHIFT
    if num_frames == 0:
        return np.empty((0, _FBANK_NUM_MEL_BINS), dtype=np.float32)
    # FirstSampleOfFrame(f) = f*160 + 80 - 200 = f*160 - 120.
    first_start = -_FBANK_FRAME_LENGTH // 2 + _FBANK_FRAME_SHIFT // 2
    last_end = (num_frames - 1) * _FBANK_FRAME_SHIFT + first_start + _FBANK_FRAME_LENGTH
    pad_left = max(0, -first_start)
    pad_right = max(0, last_end - n)
    padded = np.pad(waveform, (pad_left, pad_right), mode="symmetric")
    frames = as_strided(
        padded,
        shape=(num_frames, _FBANK_FRAME_LENGTH),
        strides=(_FBANK_FRAME_SHIFT * 4, 4),
    ).astype(np.float32, copy=True)
    # ProcessWindow: dither (0) -> quitar DC -> preénfasis -> ventana.
    # El preénfasis usa la muestra ORIGINAL anterior (el bucle C++
    # recorre de atrás hacia adelante): vectorizado con copia previa.
    frames -= frames.mean(axis=1, keepdims=True)
    prev = frames[:, :-1].copy()
    frames[:, 1:] -= _FBANK_PREEMPH_COEFF * prev
    frames[:, 0] *= 1.0 - _FBANK_PREEMPH_COEFF
    frames *= _get_povey_window()
    spectrum = np.fft.rfft(frames, n=_FBANK_FFT_SIZE, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    power = power[:, : _FBANK_FFT_SIZE // 2]  # bins 0..255
    mel = power @ _get_mel_banks().T
    return np.log(np.maximum(mel, _FBANK_LOG_FLOOR)).astype(np.float32)


def _embedding_features(waveform: np.ndarray) -> np.ndarray:
    """Fbank + resta de la media global (feature_normalize_type=global-mean)."""
    feats = _kaldi_fbank(waveform)
    if feats.shape[0] == 0:
        return feats
    return feats - feats.mean(axis=0, keepdims=True)


# --- segmentación en streaming ----------------------------------------------------


def _active_fragments(
    frames: np.ndarray, window_audio: np.ndarray, max_seconds: float
) -> np.ndarray:
    """Audio concatenado de los fragmentos activos de un slot en un chunk.

    frames: (589,) int8 con 1 donde el slot está activo. El mapeo
    frame->muestra es el del script de referencia (lineal sobre la
    ventana: k/589 * 160000). Se toman los fragmentos más largos
    hasta max_seconds: la identidad del hablante no necesita más voz.
    """
    frags: list[tuple[int, int]] = []
    start: int | None = None
    for k, value in enumerate(frames):
        if value and start is None:
            start = k
        elif not value and start is not None:
            frags.append((start, k))
            start = None
    if start is not None:
        frags.append((start, len(frames)))
    if not frags:
        return np.empty(0, dtype=np.float32)
    frags.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    max_samples = int(max_seconds * _SAMPLE_RATE)
    pieces: list[np.ndarray] = []
    total = 0
    for a, b in frags:
        s0 = int(a / _FRAMES_PER_WINDOW * _WINDOW_SIZE)
        s1 = int(b / _FRAMES_PER_WINDOW * _WINDOW_SIZE)
        s0 = max(0, min(s0, window_audio.shape[0]))
        s1 = max(0, min(s1, window_audio.shape[0]))
        if s1 <= s0:
            continue
        take = min(s1 - s0, max_samples - total)
        if take <= 0:
            break
        pieces.append(window_audio[s0 : s0 + take])
        total += take
        if total >= max_samples:
            break
    if not pieces:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


class _StreamingDiarizer:
    """Primera pasada en streaming: segmentación + embeddings.

    Mantiene la memoria O(1) en el waveform: solo guarda el buffer
    pendiente (< 41 s) y las estructuras compactas que crecen con la
    duración (etiquetas int8 ~2 KB/s de audio, embeddings ~1,5 KB/s).
    ``feed`` recibe bloques de audio (p. ej. lecturas de 5 s de
    ffmpeg) y corre la segmentación en lotes de 32 ventanas de 10 s
    con salto de 1 s; ``finish`` procesa la cola y devuelve las
    etiquetas, los pares embebidos y sus embeddings. Los embeddings
    (fbank + inferencia, lo más caro de la diarización) van a un pool
    de ``embed_workers`` hilos cuando es > 1 y se recogen en orden en
    ``finish``; con 1, el camino es secuencial (el de siempre).
    """

    def __init__(
        self,
        seg_session: Any,
        emb_session: Any,
        *,
        embed_stride: int = _EMBED_STRIDE,
        embed_workers: int = _EMBED_WORKERS,
    ) -> None:
        self._seg_session = seg_session
        self._emb_session = emb_session
        self._embed_stride = max(1, embed_stride)
        self._embed_workers = max(1, embed_workers)
        self._seg_input = seg_session.get_inputs()[0].name
        self._emb_input = emb_session.get_inputs()[0].name
        self._emb_output = emb_session.get_outputs()[0].name
        self._pending = np.empty(0, dtype=np.float32)
        self._chunk_index = 0
        self._labels: list[np.ndarray] = []
        self._pairs: list[tuple[int, int]] = []
        self._embeddings: list[np.ndarray] = []
        self._total_samples = 0
        # Pool perezoso (sin trabajos de embedding no se crea ningún
        # hilo) + futuros en orden de envío (futuro, chunk, slot).
        self._embed_pool: ThreadPoolExecutor | None = None
        self._embed_futures: list[tuple[Future[np.ndarray | None], int, int]] = []

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def feed(self, samples: np.ndarray) -> None:
        """Añade audio y procesa los lotes de 32 ventanas completos."""
        if samples.size:
            self._pending = (
                np.concatenate([self._pending, samples])
                if self._pending.size
                else np.asarray(samples, dtype=np.float32)
            )
            self._total_samples += int(samples.size)
        batch_samples = (_SEG_BATCH - 1) * _WINDOW_SHIFT + _WINDOW_SIZE
        while self._pending.size >= batch_samples:
            windows = as_strided(
                self._pending,
                shape=(_SEG_BATCH, _WINDOW_SIZE),
                strides=(_WINDOW_SHIFT * 4, 4),
            )
            self._process_windows(np.ascontiguousarray(windows))
            self._pending = self._pending[_SEG_BATCH * _WINDOW_SHIFT :]
            self._chunk_index += _SEG_BATCH

    def finish(self) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, int]:
        """Procesa la cola: ventanas completas restantes + ventana final.

        La ventana final (rellenada con ceros) existe si el audio no
        termina en un múltiplo exacto del salto, igual que en el
        script de referencia. Devuelve (clases, pares, embeddings,
        muestras totales).
        """
        if self._pending.size >= _WINDOW_SIZE:
            num_full = (self._pending.size - _WINDOW_SIZE) // _WINDOW_SHIFT + 1
            windows = as_strided(
                self._pending,
                shape=(num_full, _WINDOW_SIZE),
                strides=(_WINDOW_SHIFT * 4, 4),
            )
            self._process_windows(np.ascontiguousarray(windows))
            self._pending = self._pending[num_full * _WINDOW_SHIFT :]
            self._chunk_index += num_full
        total = self._total_samples
        has_last = total < _WINDOW_SIZE or (total - _WINDOW_SIZE) % _WINDOW_SHIFT > 0
        if has_last and total > 0:
            padded = np.pad(self._pending, (0, _WINDOW_SIZE - self._pending.size))
            self._process_windows(np.ascontiguousarray(padded[None, :]))
            self._chunk_index += 1
        if self._labels:
            classes = np.stack(self._labels, axis=0)
        else:
            classes = np.empty((0, _FRAMES_PER_WINDOW), dtype=np.int8)
        self._gather_embeddings()
        if self._embeddings:
            embeddings = np.stack(self._embeddings)
        else:
            embeddings = np.empty((0, _EMBEDDING_DIM), dtype=np.float64)
        return classes, self._pairs, embeddings, total

    def _gather_embeddings(self) -> None:
        """Recoge los embeddings del pool en orden de envío y lo cierra.

        El orden de envío es el mismo en que la versión secuencial
        añadía los pares, así que (pares, embeddings) quedan idénticos.
        Una excepción en un trabajador se propaga aquí (el llamador la
        trata como un fallo del canal, igual que un fallo en feed).
        """
        try:
            for future, chunk, slot in self._embed_futures:
                embedding = future.result()
                if embedding is not None:
                    self._pairs.append((chunk, slot))
                    self._embeddings.append(embedding)
        finally:
            self._embed_futures.clear()
            self.close()

    def close(self) -> None:
        """Cierra el pool de embeddings (idempotente; cancela pendientes).

        ``finish`` ya lo cierra tras recoger los resultados;
        ``detect_speakers`` lo llama en sus vías de error para no dejar
        hilos si ``feed`` falla o se cancela. Las tareas en cola se
        cancelan; las que ya corrían terminan (cada una ~0,2 s).
        """
        pool, self._embed_pool = self._embed_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _process_windows(self, windows: np.ndarray) -> None:
        """Segmenta un lote de ventanas y embebe los pares con stride."""
        (raw,) = self._seg_session.run(
            None, {self._seg_input: windows[:, None, :].astype(np.float32)}
        )
        classes = np.asarray(raw).argmax(axis=-1).astype(np.int8)
        for b in range(windows.shape[0]):
            chunk = self._chunk_index + b
            self._labels.append(classes[b])
            if chunk % self._embed_stride == 0:
                self._embed_chunk(chunk, classes[b], windows[b])

    def _embed_chunk(self, chunk: int, classes: np.ndarray, window_audio: np.ndarray) -> None:
        """Un embedding por slot activo >= 10 frames en el chunk.

        La selección de slots y el recorte del clip son baratos y se
        hacen en el hilo de lectura; el fbank + la inferencia ONNX
        (lo caro: ~0,2 s por par) van al pool cuando embed_workers > 1
        y se recogen en orden en finish(). Con embed_workers=1 el
        embedding corre aquí mismo, como siempre.
        """
        labels = _POWERSET[classes]  # (589, 3)
        for slot in range(_NUM_LOCAL_SPEAKERS):
            frames = labels[:, slot]
            if int(frames.sum()) < _MIN_ACTIVE_FRAMES:
                continue
            clip = _active_fragments(frames, window_audio, _MAX_EMBED_SECONDS)
            if clip.size < int(_MIN_EMBED_SECONDS * _SAMPLE_RATE):
                continue
            if self._embed_workers <= 1:
                embedding = self._run_embedding(clip)
                if embedding is not None:
                    self._pairs.append((chunk, slot))
                    self._embeddings.append(embedding)
                continue
            pool = self._embed_pool
            if pool is None:
                pool = ThreadPoolExecutor(
                    max_workers=self._embed_workers,
                    thread_name_prefix="kca-embed",
                )
                self._embed_pool = pool
            self._embed_futures.append(
                (pool.submit(self._run_embedding, clip), chunk, slot)
            )

    def _run_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        """Fbank + inferencia de embedding para un clip (hilo de trabajo)."""
        feats = _embedding_features(clip)
        if feats.shape[0] == 0:
            return None
        (emb,) = self._emb_session.run(
            [self._emb_output], {self._emb_input: feats[None].astype(np.float32)}
        )
        return np.asarray(emb[0], dtype=np.float64)


# --- funciones puras: conteo, clustering, segmentos, timeline -----------------------


def speaker_count_per_frame(classes: np.ndarray) -> np.ndarray:
    """Núm. de hablantes simultáneos por frame global (16,875 ms).

    Promedia sobre las ventanas solapadas el conteo de slots activos
    por frame y redondea, como el script de referencia. Función pura.
    """
    if classes.shape[0] == 0:
        return np.zeros(0, dtype=np.int8)
    labels = _POWERSET[classes]  # (chunks, 589, 3)
    counts = labels.sum(axis=-1).astype(np.float64)
    num_frames = (
        int((_WINDOW_SIZE + (classes.shape[0] - 1) * _WINDOW_SHIFT) / _RECEPTIVE_FIELD_SHIFT) + 1
    )
    total = np.zeros(num_frames)
    denom = np.zeros(num_frames)
    for i in range(classes.shape[0]):
        start = int(i * _WINDOW_SHIFT / _RECEPTIVE_FIELD_SHIFT + 0.5)
        end = start + _FRAMES_PER_WINDOW
        total[start:end] += counts[i]
        denom[start:end] += 1
    total /= np.maximum(denom, 1e-12)
    return (total + 0.5).astype(np.int8)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _complete_linkage_labels(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Clustering aglomerativo de enlace completo, corte por umbral.

    Equivalente a FastClustering de sherpa (cutree_cdist): se fusiona
    mientras la menor distancia entre clusters <= umbral. Distancia =
    1 - similitud coseno sobre embeddings L2-normalizados. O(N²) en
    memoria y O(N) pasadas vectorizadas; por eso cluster_embeddings
    topa N en 1500.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)
    normed = _l2_normalize(embeddings)
    dist = np.clip(1.0 - (normed @ normed.T), 0.0, 2.0).astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    # members[c] = puntos del cluster c (c = id representativo, el
    # menor índice). Al fusionar j en i, la fila/columna j se marca inf.
    members: list[list[int]] = [[i] for i in range(n)]
    while True:
        flat = int(np.argmin(dist))
        i, j = flat // n, flat % n
        if not np.isfinite(dist[i, j]) or dist[i, j] > threshold:
            break
        if j < i:
            i, j = j, i
        dist[i, :] = np.maximum(dist[i, :], dist[j, :])
        dist[:, i] = dist[i, :]
        dist[i, i] = np.inf
        dist[j, :] = np.inf
        dist[:, j] = np.inf
        members[i].extend(members[j])
        members[j] = []
    out = np.empty(n, dtype=np.int32)
    remap: dict[int, int] = {}
    for rep, mem in enumerate(members):
        if not mem:
            continue
        remap[rep] = len(remap)
        for m in mem:
            out[m] = remap[rep]
    return out


def _merge_similar_centroids(
    embeddings: np.ndarray, labels: np.ndarray, min_sim: float
) -> np.ndarray:
    """Fusiona clusters con centroides casi idénticos (sim > min_sim).

    Corrige el sobre-fragmentado del enlace completo: un hablante
    partido en dos clusters crea turnos falsos en el timeline. Dos
    hablantes distintos rara vez superan 0,5 en sus centroides (en
    el wav de 34 s, el p95 entre hablantes distintos es 0,44).
    """
    normed = _l2_normalize(embeddings)
    clusters = sorted(set(labels.tolist()))
    centroids: dict[int, np.ndarray] = {}
    for c in clusters:
        members = normed[labels == c]
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[c] = centroid / norm if norm > 0 else centroid
    parent = {c: c for c in clusters}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            ca, cb = clusters[a], clusters[b]
            if float(centroids[ca] @ centroids[cb]) > min_sim:
                parent[find(cb)] = find(ca)
    remap: dict[int, int] = {}
    for c in clusters:
        root = find(c)
        if root not in remap:
            remap[root] = len(remap)
    return np.array([remap[find(int(x))] for x in labels], dtype=np.int32)


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = _CLUSTER_THRESHOLD,
    max_cluster: int = _MAX_CLUSTER_EMBEDDINGS,
    centroid_merge_sim: float = _CENTROID_MERGE_SIM,
) -> np.ndarray:
    """Etiquetas de cluster para cada embedding.

    Si hay más de max_cluster embeddings, se agrupa una submuestra
    uniforme y el resto se asigna al centroide más cercano (coseno):
    la matriz O(N²) queda topada y un video de 4 h no explota la
    memoria. Función pura.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n <= max_cluster:
        labels = _complete_linkage_labels(embeddings, threshold)
    else:
        idx = np.linspace(0, n - 1, max_cluster).astype(int)
        sub_labels = _complete_linkage_labels(embeddings[idx], threshold)
        normed = _l2_normalize(embeddings)
        centroids = []
        for c in range(int(sub_labels.max()) + 1):
            members = normed[idx][sub_labels == c]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm > 0 else centroid)
        sims = normed @ np.stack(centroids).T
        labels = np.argmax(sims, axis=1).astype(np.int32)
    if int(labels.max(initial=0)) > 0:
        labels = _merge_similar_centroids(embeddings, labels, centroid_merge_sim)
    return labels


def _runs_to_segments(
    frames: np.ndarray, speaker: int, scale: float, offset: float
) -> list[SpeakerSegment]:
    """Codifica por rachas los frames activos de un hablante.

    Fusiona huecos < min_duration_off y descarta segmentos <
    min_duration_on (defaults de sherpa-onnx).
    """
    raw: list[SpeakerSegment] = []
    start: int | None = None
    for i, value in enumerate(frames):
        if value and start is None:
            start = i
        elif not value and start is not None:
            raw.append(SpeakerSegment(start * scale + offset, i * scale + offset, speaker))
            start = None
    if start is not None:
        raw.append(SpeakerSegment(start * scale + offset, len(frames) * scale + offset, speaker))
    merged: list[SpeakerSegment] = []
    for seg in raw:
        if merged and seg.start - merged[-1].end < _MIN_DURATION_OFF:
            merged[-1] = SpeakerSegment(merged[-1].start, seg.end, speaker)
        else:
            merged.append(seg)
    return [s for s in merged if s.end - s.start >= _MIN_DURATION_ON]


def build_segments(
    classes: np.ndarray,
    pairs: list[tuple[int, int]],
    pair_labels: np.ndarray,
    total_samples: int,
    speakers_per_frame: np.ndarray,
) -> list[SpeakerSegment]:
    """Segmentos por hablante desde los votos de los pares embebidos.

    Cada par (chunk, slot) vota por su cluster en sus frames activos;
    cada frame global toma los clusters más votados, hasta el número
    de hablantes simultáneos que dice speaker_count_per_frame. Los
    pares no embebidos (por el stride) heredan la etiqueta del chunk
    embebido más cercano con el mismo slot (±5 chunks) siempre que
    la actividad del slot se solape en el tiempo (>= 50% de sus
    frames activos): en un turno el modelo puede reasignar slots
    entre ventanas y heredar a ciegas etiqueta mal. Sin embeddings
    pero con habla (audio muy corto o slots pobres), todo se
    etiqueta como un solo hablante. Función pura.
    """
    if classes.shape[0] == 0:
        return []
    num_clusters = int(pair_labels.max()) + 1 if pair_labels.size else 0
    num_frames = speakers_per_frame.shape[0]
    labels = _POWERSET[classes]
    scale = _RECEPTIVE_FIELD_SHIFT / _SAMPLE_RATE
    offset = _RECEPTIVE_FIELD_SIZE / _SAMPLE_RATE * 0.5
    if num_clusters == 0:
        stop = min(int(total_samples / _RECEPTIVE_FIELD_SHIFT), num_frames)
        return _runs_to_segments((speakers_per_frame[:stop] > 0).astype(np.int8), 0, scale, offset)
    pair_label = dict(zip(pairs, pair_labels.tolist(), strict=True))
    for i in range(classes.shape[0]):
        for j in range(_NUM_LOCAL_SPEAKERS):
            if (i, j) in pair_label:
                continue
            frames = labels[i, :, j]
            active = int(frames.sum())
            if active < _MIN_ACTIVE_FRAMES:
                continue
            base = int(i * _WINDOW_SHIFT / _RECEPTIVE_FIELD_SHIFT + 0.5)
            inherited = None
            for radius in range(1, _INHERIT_RADIUS + 1):
                for i2 in (i - radius, i + radius):
                    if (i2, j) not in pair_label:
                        continue
                    shift = int(i2 * _WINDOW_SHIFT / _RECEPTIVE_FIELD_SHIFT + 0.5) - base
                    other = labels[i2, :, j]
                    if shift >= 0:
                        overlap = int((frames[shift:] & other[: _FRAMES_PER_WINDOW - shift]).sum())
                    else:
                        overlap = int((frames[: _FRAMES_PER_WINDOW + shift] & other[-shift:]).sum())
                    if overlap >= 0.5 * active:
                        inherited = pair_label[(i2, j)]
                        break
                if inherited is not None:
                    break
            if inherited is not None:
                pair_label[(i, j)] = inherited
    votes = np.zeros((num_frames, num_clusters), dtype=np.float32)
    for (i, j), cluster in pair_label.items():
        frames = labels[i, :, j].astype(np.float32)
        start = int(i * _WINDOW_SHIFT / _RECEPTIVE_FIELD_SHIFT + 0.5)
        end = start + _FRAMES_PER_WINDOW
        votes[start:end, cluster] += frames
    stop_frame = min(int(total_samples / _RECEPTIVE_FIELD_SHIFT), num_frames)
    votes = votes[:stop_frame]
    spf = speakers_per_frame[:stop_frame]
    final = np.zeros((votes.shape[0], num_clusters), dtype=np.int8)
    order = np.argsort(-votes, axis=1)
    for f in range(votes.shape[0]):
        for k in range(min(int(spf[f]), num_clusters)):
            c = order[f, k]
            if votes[f, c] > 0:
                final[f, c] = 1
    segments: list[SpeakerSegment] = []
    for c in range(num_clusters):
        segments.extend(_runs_to_segments(final[:, c], c, scale, offset))
    segments.sort(key=lambda s: s.start)
    return segments


def segments_to_timeline(segments: list[SpeakerSegment], duration: float) -> np.ndarray:
    """Densidad de turnos 1 Hz (0–1) desde los segmentos.

    Un turno es un cambio de hablante entre segmentos consecutivos
    (ordenados por inicio). El valor de cada segundo es el número de
    turnos en una ventana centrada de ±10 s, con 4 turnos = 1,0:
    una entrevista tranquila (un turno cada ~20 s) puntúa ~0,25, un
    debate vivo (un turno cada ~5 s) puntúa 1,0 y un monólogo 0,0.
    Función pura: no toca red ni disco, para tests y calibración.
    """
    seconds = max(1, math.ceil(duration))
    timeline = np.zeros(seconds, dtype=np.float64)
    turns: list[float] = []
    prev: SpeakerSegment | None = None
    for seg in segments:
        if prev is not None and seg.speaker != prev.speaker:
            turns.append(seg.start)
        prev = seg
    if not turns:
        return timeline
    turns_arr = np.array(turns)
    for s in range(seconds):
        center = s + 0.5
        count = int(
            (
                (turns_arr >= center - _TURN_WINDOW_SECONDS)
                & (turns_arr <= center + _TURN_WINDOW_SECONDS)
            ).sum()
        )
        timeline[s] = min(1.0, count / _TURNS_FOR_MAX)
    return timeline


# --- entrada principal --------------------------------------------------------


def detect_speakers(
    input_path: Path | str,
    duration: float,
    ffmpeg_path: str,
    *,
    sessions: tuple[Any, Any] | None = None,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    embed_workers: int = _EMBED_WORKERS,
) -> DiarizationResult:
    """Diariza un video y devuelve timeline de turnos + segmentos.

    Extrae PCM float32 16 kHz mono con ffmpeg (streaming) y corre la
    segmentación por lotes y los embeddings por pares. Si no se pasan
    ``sessions``, descarga los modelos en el primer uso y cachea las
    sesiones. Lanza DiarizationError si ffmpeg falla, el video no
    tiene audio o la diarización no está disponible;
    DiarizationCancelled si el usuario cancela. Sin habla, devuelve
    un resultado vacío (timeline en ceros), no un error.
    ``embed_workers`` paraleliza los embeddings (ver _EMBED_WORKERS);
    con 1, secuencial.
    """
    if sessions is None:
        model_paths = ensure_diarization_models(on_progress, should_cancel=should_cancel)
        sessions = load_diarization_sessions(model_paths)
    seg_session, emb_session = sessions
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
        str(_SAMPLE_RATE),
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
        raise DiarizationError(
            f"No se pudo ejecutar ffmpeg para la diarización: {error}"
        ) from error
    assert process.stdout is not None
    diarizer = _StreamingDiarizer(
        seg_session, emb_session, embed_workers=embed_workers
    )
    saw_samples = False
    read_bytes = _SAMPLE_RATE * 4 * _FFMPEG_READ_SECONDS
    try:
        while True:
            if should_cancel is not None and should_cancel():
                raise DiarizationCancelled("Diarización cancelada.")
            data = process.stdout.read(read_bytes)
            if not data:
                break
            saw_samples = True
            samples = np.frombuffer(data, dtype=np.float32)
            diarizer.feed(samples)
    except DiarizationCancelled:
        diarizer.close()
        raise
    except (OSError, ValueError) as error:
        diarizer.close()
        raise DiarizationError(
            f"La extracción de audio para la diarización falló: {error}"
        ) from error
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
        diarizer.close()
        detail = stderr.decode("utf-8", "replace").strip()[:200]
        raise DiarizationError(
            f"ffmpeg no devolvió audio para la diarización ({detail or 'sin pista de audio'})."
        )
    if should_cancel is not None and should_cancel():
        diarizer.close()
        raise DiarizationCancelled("Diarización cancelada.")
    classes, pairs, embeddings, total_samples = diarizer.finish()
    speakers_per_frame = speaker_count_per_frame(classes)
    timeline = np.zeros(max(1, math.ceil(duration)), dtype=np.float64)
    if classes.shape[0] == 0 or int(speakers_per_frame.max(initial=0)) == 0:
        return DiarizationResult(timeline=timeline)
    if should_cancel is not None and should_cancel():
        raise DiarizationCancelled("Diarización cancelada.")
    pair_labels = cluster_embeddings(embeddings)
    segments = build_segments(classes, pairs, pair_labels, total_samples, speakers_per_frame)
    timeline = segments_to_timeline(segments, duration)
    return DiarizationResult(
        timeline=timeline,
        segments=segments,
        num_speakers=len({s.speaker for s in segments}),
    )


def get_speaker_result(
    input_path: Path | str | None,
    duration: float,
    ffmpeg_path: str | None,
    *,
    enabled: bool = True,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    embed_workers: int = _EMBED_WORKERS,
) -> DiarizationResult | None:
    """Diarización completa (timeline + segmentos) o None si no se pudo.

    Mismo contrato que ``get_speaker_timeline``: nunca lanza por
    problemas de la diarización (sin runtime, sin modelo, sin red,
    sin audio) — registra un aviso y devuelve None. Solo propaga
    DiarizationCancelled. El render usa los segmentos para el split
    dinámico (Fase 3, ítem 3b).
    """
    if not enabled or input_path is None or not ffmpeg_path:
        return None
    try:
        return detect_speakers(
            input_path,
            duration,
            ffmpeg_path,
            on_progress=on_progress,
            should_cancel=should_cancel,
            embed_workers=embed_workers,
        )
    except DiarizationCancelled:
        raise
    except DiarizationError as error:
        logger.warning("Diarización no disponible (%s); el motor sigue sin este canal.", error)
        return None
    except Exception as error:  # Defensa: la diarización jamás rompe un proyecto.
        logger.warning(
            "La diarización falló de forma inesperada (%s); el motor sigue sin este canal.",
            error,
        )
        return None


def get_speaker_segments(
    input_path: Path | str | None,
    duration: float,
    ffmpeg_path: str | None,
    *,
    enabled: bool = True,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    embed_workers: int = _EMBED_WORKERS,
) -> list[dict[str, float | int]]:
    """Segmentos de hablante como dicts JSON (para metadata y render)."""
    result = get_speaker_result(
        input_path,
        duration,
        ffmpeg_path,
        enabled=enabled,
        on_progress=on_progress,
        should_cancel=should_cancel,
        embed_workers=embed_workers,
    )
    if result is None:
        return []
    return [
        {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
        for segment in result.segments
    ]


def get_speaker_timeline(
    input_path: Path | str | None,
    duration: float,
    ffmpeg_path: str | None,
    *,
    enabled: bool = True,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    embed_workers: int = _EMBED_WORKERS,
) -> np.ndarray | None:
    """Timeline de turnos para ``moments.detect_moments`` (o None).

    El punto de entrada que cablea el pipeline: nunca lanza por
    problemas de la diarización (sin runtime, sin modelo, sin red,
    sin audio) — registra un aviso y devuelve None para que el motor
    siga sin este canal. Solo propaga DiarizationCancelled, que el
    pipeline debe traducir a su propia cancelación.
    ``embed_workers`` paraleliza los embeddings (ver _EMBED_WORKERS).
    """
    result = get_speaker_result(
        input_path,
        duration,
        ffmpeg_path,
        enabled=enabled,
        on_progress=on_progress,
        should_cancel=should_cancel,
        embed_workers=embed_workers,
    )
    return result.timeline if result is not None else None
    turns = sum(
        1
        for prev, cur in zip(result.segments, result.segments[1:], strict=False)
        if cur.speaker != prev.speaker
    )
    logger.info(
        "Diarización: %d hablante(s), %d segmento(s), %d turno(s).",
        result.num_speakers,
        len(result.segments),
        turns,
    )
    return result.timeline
