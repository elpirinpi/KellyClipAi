from __future__ import annotations

import copy
import html
import json
import logging
import os
import re
import signal
import subprocess
import textwrap
import threading
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from .face_tracking import (
    FaceCropKeyframe,
    FaceTrackingError,
    analyze_face_crop_paths,
)
from .schemas import ClipProposal, ProjectConfig, TranscriptSegment, Word

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")
_CENSORED_WORDS = {
    "asshole",
    "bitch",
    "cabrón",
    "cabron",
    "coño",
    "concha",
    "damn",
    "fuck",
    "fucking",
    "gilipollas",
    "joder",
    "mierda",
    "pendejo",
    "puta",
    "puto",
    "shit",
    "verga",
}
_FFMPEG_RENDER_TIMEOUT_SECONDS = 600
_WHISPER_PIPELINE_LOCK = threading.Lock()
_WHISPER_PIPELINE_CACHE: tuple[str, Any] | None = None

# --- NVENC (Fase 2, B-R1) ---------------------------------------------------
# Detección en runtime con fallback elegante: la app debe funcionar en
# CUALQUIER PC (NVIDIA, AMD, Intel, integradas o sin GPU). El resultado se
# cachea por proceso: la detección ejecuta ffmpeg dos veces y no tiene
# sentido repetirla en cada clip. KELLYCLIP_NVENC=0 la desactiva (soporte).
_NVENC_PROBE_TIMEOUT_SECONDS = 15
_NVENC_CACHE: bool | None = None
_NVENC_CACHE_LOCK = threading.Lock()

# --- Renders en curso (Fase 2, B-P14) ---------------------------------------
# Registro de los ffmpeg hijos de los renders activos, para que el apagado
# elegante pueda matarlos. Cada ffmpeg arranca en su propio grupo de
# procesos (start_new_session en POSIX, CREATE_NEW_PROCESS_GROUP en
# Windows) para no tocar a ningún otro proceso de la app.
_ACTIVE_RENDER_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_RENDER_PROCESSES_LOCK = threading.Lock()

# --- Caché de análisis de silencios (Fase 2, B-R4) --------------------------
# El análisis de silencios es un pase completo de ffmpeg sobre el clip;
# el editor lo repetía en cada re-render del mismo clip. Clave: identidad
# del archivo (ruta + tamaño + mtime) + rango + binario ffmpeg. Si el
# archivo cambia, la clave cambia y la caché se invalida sola. Acotada
# (LRU simple por orden de inserción) para no crecer sin límite.
_SILENCE_CACHE: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
_SILENCE_CACHE_LOCK = threading.Lock()
_SILENCE_CACHE_MAX_ENTRIES = 32


def _file_identity(path: Path) -> tuple[str, int, int] | None:
    """Identidad barata de un archivo para las cachés de análisis.

    (ruta resuelta, tamaño, mtime en ns). None si no se puede leer: en
    ese caso la caché se omite y se analiza de nuevo, sin fallar.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def clear_analysis_caches() -> None:
    """Vacía las cachés de análisis de este módulo (tests)."""
    with _SILENCE_CACHE_LOCK:
        _SILENCE_CACHE.clear()


def is_nvenc_available(ffmpeg_path: str) -> bool:
    """Indica si este ffmpeg puede codificar con NVENC en este equipo.

    Dos comprobaciones: el codificador debe estar compilado en el ffmpeg
    (`-encoders`) Y un encode de prueba corto debe terminar bien. La
    segunda es imprescindible: un ffmpeg con NVENC compilado en un equipo
    sin GPU NVIDIA (o sin driver) lista el codificador pero el encode
    falla. El resultado se cachea por proceso.
    """
    if os.environ.get("KELLYCLIP_NVENC", "").strip() == "0":
        return False
    global _NVENC_CACHE
    with _NVENC_CACHE_LOCK:
        if _NVENC_CACHE is not None:
            return _NVENC_CACHE
        available = False
        try:
            encoders = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_NVENC_PROBE_TIMEOUT_SECONDS,
                **_subprocess_options(),
            )
            if "h264_nvenc" in (encoders.stdout or ""):
                # Encode de prueba con los mismos parámetros que el render
                # real: si este pasa, el render con NVENC también puede.
                probe = subprocess.run(
                    [
                        ffmpeg_path,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc=duration=0.2:size=128x128:rate=10",
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p4",
                        "-cq",
                        "21",
                        "-pix_fmt",
                        "yuv420p",
                        "-f",
                        "null",
                        "-",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_NVENC_PROBE_TIMEOUT_SECONDS,
                    **_subprocess_options(),
                )
                available = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as error:
            logger.debug("La detección de NVENC falló; se usará libx264: %s", error)
            available = False
        _NVENC_CACHE = available
        return available


def reset_nvenc_cache() -> None:
    """Olvida el resultado cacheado de la detección de NVENC (tests)."""
    global _NVENC_CACHE
    with _NVENC_CACHE_LOCK:
        _NVENC_CACHE = None


def _video_encoder_args(ffmpeg_path: str) -> list[str]:
    """Argumentos del codificador de video: NVENC si está disponible.

    NVENC usa `-cq` (calidad constante, equivalente al CRF de x264) y
    presets p1–p7 (p4 ≈ veryfast). Sin NVENC, libx264 exactamente como
    antes. `-pix_fmt yuv420p` solo se añade con NVENC: el codificador
    de NVIDIA lo exige y libx264 ya lo usa por defecto.
    """
    if is_nvenc_available(ffmpeg_path):
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "21",
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]


def _register_render_process(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_RENDER_PROCESSES_LOCK:
        _ACTIVE_RENDER_PROCESSES.add(process)


def _unregister_render_process(process: subprocess.Popen[str]) -> None:
    with _ACTIVE_RENDER_PROCESSES_LOCK:
        _ACTIVE_RENDER_PROCESSES.discard(process)


def _kill_render_process(process: subprocess.Popen[str]) -> None:
    """Mata un ffmpeg de render junto con su grupo de procesos.

    Multiplataforma y defensiva: en POSIX se mata el grupo completo
    (el ffmpeg arranca con start_new_session); en Windows se mata el
    proceso (ffmpeg no genera hijos y arranca en su propio grupo con
    CREATE_NEW_PROCESS_GROUP). Nunca lanza: se llama desde el apagado.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
    except Exception as error:  # noqa: BLE001 - el apagado no debe fallar
        logger.debug("No se pudo terminar el render ffmpeg %s: %s", process.pid, error)
        try:
            process.kill()
        except Exception:  # noqa: BLE001 - último intento, sin propagar
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - el apagado no debe fallar
            pass


def terminate_active_renders() -> int:
    """Mata todos los ffmpeg de renders en curso. Devuelve cuántos había.

    La usa el apagado elegante de la cola de procesamiento cuando el
    drenado agota su timeout: sin esto, cancelar la tarea asyncio no
    detiene el hilo del render y el ffmpeg quedaba huérfano (B9).
    """
    with _ACTIVE_RENDER_PROCESSES_LOCK:
        processes = list(_ACTIVE_RENDER_PROCESSES)
    for process in processes:
        _kill_render_process(process)
    return len(processes)


class MediaProcessingError(RuntimeError):
    pass


def _subprocess_options() -> dict[str, int]:
    # stdin DEVNULL en TODOS los ffmpeg: ffmpeg lee stdin para sus
    # comandos interactivos ('q', ...) y, si hereda un pipe abierto sin
    # EOF, se bloquea para siempre en la lectura (deadlock medido el
    # 21-11-2026 al perfilar; el proceso quedaba al 0 % de CPU).
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW, "stdin": subprocess.DEVNULL}
    return {"stdin": subprocess.DEVNULL}


def _render_subprocess_options() -> dict[str, object]:
    """Opciones del ffmpeg de render: su propio grupo de procesos.

    En POSIX, start_new_session=True crea un grupo nuevo; en Windows,
    CREATE_NEW_PROCESS_GROUP hace lo mismo. Así el apagado elegante
    puede matar el grupo sin tocar al resto de la app (B-P14).
    stdin DEVNULL: ver _subprocess_options (ffmpeg se bloquea leyendo
    un stdin heredado que sea un pipe abierto sin EOF).
    """
    if os.name == "nt":
        create_new_process_group = int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        )
        return {
            "creationflags": subprocess.CREATE_NO_WINDOW | create_new_process_group,
            "stdin": subprocess.DEVNULL,
        }
    return {"start_new_session": True, "stdin": subprocess.DEVNULL}


def _run(arguments: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            **_subprocess_options(),
        )
    except FileNotFoundError as error:
        raise MediaProcessingError(
            f"No se encontró {arguments[0]}. Instala FFmpeg y ffprobe y vuelve a intentarlo."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise MediaProcessingError(f"Se agotó el tiempo al {purpose}.") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        raise MediaProcessingError(f"No se pudo {purpose}: {' '.join(detail)}")
    return result


def _probe_media_pyav(input_path: Path) -> dict[str, object]:
    """Lee duración, dimensiones y audio con PyAV (sin ffprobe).

    PyAV ya es dependencia de la app y trae sus propias DLL de FFmpeg,
    así que este camino permite prescindir de ffprobe.exe (~85 MB del
    instalador, B-R2/B37). Devuelve exactamente las mismas claves que
    la sonda por ffprobe. Lanza MediaProcessingError ante cualquier
    anomalía; el llamador decide si reintenta con ffprobe.
    """
    try:
        import av
    except ImportError as error:
        raise MediaProcessingError("PyAV no está disponible.") from error
    try:
        with av.open(str(input_path)) as container:
            video = next(iter(container.streams.video), None)
            audio = next(iter(container.streams.audio), None)
            if video is None or audio is None:
                raise MediaProcessingError(
                    "El archivo debe ser un video reproducible que incluya pista de audio."
                )
            duration: float | None = None
            if container.duration is not None:
                duration = container.duration / 1_000_000
            elif video.duration is not None and video.time_base is not None:
                duration = float(video.duration * video.time_base)
            width, height = int(video.width or 0), int(video.height or 0)
    except MediaProcessingError:
        raise
    except Exception as error:
        raise MediaProcessingError(f"PyAV no pudo leer el archivo: {error}") from error
    if duration is None or duration <= 0 or width <= 0 or height <= 0:
        raise MediaProcessingError("PyAV devolvió una duración o resolución no válida.")
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_audio": audio is not None,
    }


def _probe_media_ffprobe(input_path: Path, ffprobe_path: str) -> dict[str, object]:
    result = _run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            str(input_path),
        ],
        "leer los metadatos con ffprobe",
    )
    try:
        data = json.loads(result.stdout)
        streams = data["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
        duration = float(data["format"]["duration"])
        width, height = int(video["width"]), int(video["height"])
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MediaProcessingError(
            "El archivo debe ser un video reproducible que incluya pista de audio."
        ) from error
    if duration <= 0 or width <= 0 or height <= 0:
        raise MediaProcessingError("ffprobe devolvió una duración o resolución no válida.")
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "has_audio": audio is not None,
    }


def probe_media(input_path: Path, ffprobe_path: str) -> dict[str, object]:
    """Metadatos del video: PyAV primero, ffprobe como respaldo.

    El camino PyAV es el preferido (ver _probe_media_pyav). Si PyAV no
    está instalado o no puede leer el archivo, se reintenta con ffprobe
    antes de fallar: un formato que PyAV no abre pero ffprobe sí no debe
    tumbar el proyecto. `ffprobe_path` se conserva por eso y por
    compatibilidad con los llamadores actuales.
    """
    try:
        return _probe_media_pyav(input_path)
    except MediaProcessingError as error:
        logger.debug(
            "La sonda PyAV falló para %s (%s); reintentando con ffprobe.",
            input_path,
            error,
        )
    return _probe_media_ffprobe(input_path, ffprobe_path)


def create_thumbnail(
    input_path: Path,
    output_path: Path,
    duration: float,
    ffmpeg_path: str,
) -> None:
    timestamp = min(1.0, duration / 2)
    _run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-q:v",
            "3",
            "-y",
            str(output_path),
        ],
        "crear la miniatura del video",
    )


def create_frame_thumbnail(
    input_path: Path,
    output_path: Path,
    timestamp: float,
    ffmpeg_path: str,
) -> None:
    """Un fotograma del video en `timestamp`, guardado como JPEG.

    Como create_thumbnail pero en un instante elegido: las propuestas
    sin editar (edición automática apagada, v1.3.2) no tienen clip
    renderizado, así que su vista previa es el fotograma central del
    momento detectado (hotfix v1.4.1). El escalado es el mismo
    (640 px de ancho); el recorte vertical 9:16 lo hace el CSS
    (object-fit: cover). El directorio de salida debe existir.
    """
    _run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-q:v",
            "3",
            "-y",
            str(output_path),
        ],
        "extraer el fotograma de la propuesta",
    )


def transcribe(
    input_path: Path,
    model_name: str,
    on_progress: ProgressCallback,
    language: str = "auto",
    on_model_download_progress: ProgressCallback | None = None,
) -> tuple[list[TranscriptSegment], str]:
    try:
        pipeline = _get_whisper_pipeline(
            model_name, on_model_download_progress
        )
        segments, info = pipeline.transcribe(
            str(input_path),
            language=None if language == "auto" else language,
            word_timestamps=True,
            without_timestamps=False,
            vad_filter=True,
            beam_size=5,
            batch_size=4,
        )
        transcript: list[TranscriptSegment] = []
        last_percent = -1
        duration = max(float(info.duration or 0), 0.001)
        for segment in segments:
            words = [
                Word(
                    start=max(float(word.start), 0.0),
                    end=max(float(word.end), 0.001),
                    text=word.word,
                )
                for word in (segment.words or [])
                if word.end > word.start
            ]
            transcript.append(
                TranscriptSegment(
                    start=max(float(segment.start), 0.0),
                    end=max(float(segment.end), 0.001),
                    text=segment.text.strip(),
                    words=words,
                )
            )
            percent = min(100, max(0, int(float(segment.end) / duration * 100)))
            if percent > last_percent:
                on_progress(percent)
                last_percent = percent
        if not any(segment.words for segment in transcript):
            raise MediaProcessingError(
                "Whisper no detectó voz con marcas de tiempo. Comprueba que el video tenga audio inteligible."
            )
        on_progress(100)
        return transcript, info.language or "und"
    except MediaProcessingError:
        raise
    except Exception as error:
        raise MediaProcessingError(f"Falló la transcripción con faster-whisper: {error}") from error


def _download_whisper_model(
    model_name: str,
    on_progress: ProgressCallback | None,
) -> str:
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    if on_progress is not None:
        on_progress(0)

    class ModelDownloadProgress(tqdm):
        def update(self, n: int = 1) -> bool:
            updated = super().update(n)
            if self.total and on_progress is not None:
                percent = min(95.0, self.n / self.total * 95)
                on_progress(percent)
            return updated

    return str(
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{model_name}",
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ],
            max_workers=1,
            tqdm_class=ModelDownloadProgress,
        )
    )


def _get_whisper_pipeline(
    model_name: str,
    on_download_progress: ProgressCallback | None,
) -> Any:
    global _WHISPER_PIPELINE_CACHE
    with _WHISPER_PIPELINE_LOCK:
        if (
            _WHISPER_PIPELINE_CACHE is not None
            and _WHISPER_PIPELINE_CACHE[0] == model_name
        ):
            return _WHISPER_PIPELINE_CACHE[1]

        from faster_whisper import BatchedInferencePipeline, WhisperModel

        _WHISPER_PIPELINE_CACHE = None
        model_path = _download_whisper_model(model_name, on_download_progress)
        model = WhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
            num_workers=2,
        )
        pipeline = BatchedInferencePipeline(model=model)
        _WHISPER_PIPELINE_CACHE = (model_name, pipeline)
        if on_download_progress is not None:
            on_download_progress(100)
        return pipeline


def _silence_intervals(
    input_path: Path,
    start: float,
    duration: float,
    ffmpeg_path: str,
) -> list[tuple[float, float]]:
    identity = _file_identity(input_path)
    cache_key = (
        (identity, round(start, 3), round(duration, 3), ffmpeg_path)
        if identity is not None
        else None
    )
    if cache_key is not None:
        with _SILENCE_CACHE_LOCK:
            cached = _SILENCE_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
    intervals = _detect_silence_intervals(input_path, start, duration, ffmpeg_path)
    if cache_key is not None:
        with _SILENCE_CACHE_LOCK:
            _SILENCE_CACHE[cache_key] = list(intervals)
            while len(_SILENCE_CACHE) > _SILENCE_CACHE_MAX_ENTRIES:
                _SILENCE_CACHE.pop(next(iter(_SILENCE_CACHE)))
    return intervals


def _detect_silence_intervals(
    input_path: Path,
    start: float,
    duration: float,
    ffmpeg_path: str,
) -> list[tuple[float, float]]:
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "info",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-af",
                "silencedetect=noise=-35dB:d=0.45",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise MediaProcessingError(f"No se pudo analizar los silencios: {error}") from error
    if result.returncode:
        detail = (result.stderr or "").strip().splitlines()[-8:]
        raise MediaProcessingError(f"FFmpeg no pudo detectar silencios: {' '.join(detail)}")

    kept: list[tuple[float, float]] = []
    cursor = 0.0
    silence_open = False
    for line in (result.stderr or "").splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            silence_start = min(duration, max(0.0, float(start_match.group(1))))
            if silence_start > cursor:
                kept.append((cursor, silence_start))
            cursor = silence_start
            silence_open = True
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and silence_open:
            cursor = min(duration, max(cursor, float(end_match.group(1))))
            silence_open = False
    if not silence_open and cursor < duration:
        kept.append((cursor, duration))
    return [(left, right) for left, right in kept if right - left >= 0.08]


def _retained_intervals(
    input_path: Path,
    proposal: ClipProposal,
    remove_silence: bool,
    ffmpeg_path: str,
) -> list[tuple[float, float]]:
    duration = proposal.fin - proposal.inicio
    if not remove_silence:
        return [(0.0, duration)]
    intervals = _silence_intervals(input_path, proposal.inicio, duration, ffmpeg_path)
    if not intervals:
        raise MediaProcessingError(
            "El análisis de silencios no encontró voz suficiente dentro de un clip propuesto."
        )
    return intervals


def _subtract_intervals(
    intervals: list[tuple[float, float]],
    exclusions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for interval_start, interval_end in intervals:
        cursor = interval_start
        for exclusion_start, exclusion_end in sorted(exclusions):
            start = max(interval_start, exclusion_start)
            end = min(interval_end, exclusion_end)
            if end <= cursor or start >= interval_end:
                continue
            if start > cursor:
                kept.append((cursor, start))
            cursor = max(cursor, end)
            if cursor >= interval_end:
                break
        if interval_end - cursor >= 0.08:
            kept.append((cursor, interval_end))
    return [(start, end) for start, end in kept if end - start >= 0.08]


def _remap_time(timestamp: float, intervals: list[tuple[float, float]]) -> float | None:
    elapsed = 0.0
    for start, end in intervals:
        if start <= timestamp <= end:
            return elapsed + timestamp - start
        elapsed += end - start
    return None


def _clip_words(
    transcript: list[TranscriptSegment],
    proposal: ClipProposal,
    intervals: list[tuple[float, float]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for segment in transcript:
        for word in segment.words:
            if word.end <= proposal.inicio or word.start >= proposal.fin:
                continue
            local_start = max(0.0, word.start - proposal.inicio)
            local_end = min(proposal.fin - proposal.inicio, word.end - proposal.inicio)
            remapped_start = _remap_time(local_start, intervals)
            remapped_end = _remap_time(local_end, intervals)
            if remapped_start is None or remapped_end is None or remapped_end <= remapped_start:
                continue
            output.append(
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "preview_start": round(remapped_start, 3),
                    "preview_end": round(remapped_end, 3),
                    "text": word.text,
                }
            )
    return output


def _apply_subtitle_overrides(
    words: list[dict[str, object]],
    overrides: list[tuple[float, float, str]],
    proposal: ClipProposal,
    intervals: list[tuple[float, float]],
) -> list[dict[str, object]]:
    output = list(words)
    for start, end, text in overrides:
        start = max(0.0, start)
        end = min(proposal.fin - proposal.inicio, end)
        if end <= start or not text.strip():
            continue
        preview_start = _remap_time(start, intervals)
        preview_end = _remap_time(end, intervals)
        if (
            preview_start is None
            or preview_end is None
            or preview_end <= preview_start
        ):
            continue
        absolute_start = proposal.inicio + start
        absolute_end = proposal.inicio + end
        output = [
            word
            for word in output
            if not (
                float(word["start"]) < absolute_end
                and float(word["end"]) > absolute_start
            )
        ]
        output.append(
            {
                "start": round(absolute_start, 3),
                "end": round(absolute_end, 3),
                "preview_start": round(preview_start, 3),
                "preview_end": round(preview_end, 3),
                "text": text.strip(),
            }
        )
    return sorted(output, key=lambda word: (float(word["start"]), float(word["end"])))


def _ratio_size(ratio: str, resolution: int = 720) -> tuple[int, int]:
    base_width, base_height = {
        "9:16": (720, 1280),
        "16:9": (1280, 720),
        "1:1": (720, 720),
        "4:5": (720, 900),
    }[ratio]
    scale = resolution / 720
    return round(base_width * scale), round(base_height * scale)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def _censored(text: str) -> bool:
    normalized = re.sub(r"[^\w]", "", text.casefold())
    return normalized in _CENSORED_WORDS


def _caption_groups(
    words: list[dict[str, object]],
    max_chars: int = 36,
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0
    for word in words:
        text = str(word["text"])
        if current and (
            len(current) >= 6
            or current_chars + len(text) > max_chars
            or float(word["preview_start"]) - float(current[0]["preview_start"]) > 2.5
        ):
            groups.append(current)
            current, current_chars = [], 0
        current.append(word)
        current_chars += len(text) + 1
    if current:
        groups.append(current)
    return groups


def _subtitle_timestamp(seconds: float, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{separator}{milliseconds:03d}"
    )


def format_subtitles(
    words: list[dict[str, object]],
    format_name: str,
) -> str:
    if format_name not in {"srt", "vtt"}:
        raise ValueError("El formato debe ser SRT o WebVTT.")
    separator = "," if format_name == "srt" else "."
    entries: list[str] = []
    for index, group in enumerate(_caption_groups(words), start=1):
        start = float(group[0]["preview_start"])
        end = max(start + 0.08, float(group[-1]["preview_end"]))
        text = " ".join(str(word["text"]).strip() for word in group)
        if format_name == "vtt":
            text = html.escape(text, quote=False)
        timing = (
            f"{_subtitle_timestamp(start, separator)} --> "
            f"{_subtitle_timestamp(end, separator)}"
        )
        entries.append(
            f"{index}\n{timing}\n{text}"
            if format_name == "srt"
            else f"{timing}\n{text}"
        )
    prefix = "WEBVTT\n\n" if format_name == "vtt" else ""
    return prefix + "\n\n".join(entries) + ("\n" if entries else "")


def _caption_max_chars(width: int, font_size: int, margin: int = 48) -> int:
    return max(12, int((width - 2 * margin) / (font_size * 0.68)))


def _ass_color(color: str | None) -> str | None:
    if color is None:
        return None
    red, green, blue = color[1:3], color[3:5], color[5:7]
    return f"&H00{blue}{green}{red}".upper()


def _write_ass(
    output_path: Path,
    words: list[dict[str, object]],
    proposal: ClipProposal,
    config: ProjectConfig,
    clip_duration: float,
    retained: list[tuple[float, float]],
    resolution: int = 720,
    brand: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    width, height = _ratio_size(config.ratio, resolution)
    scale = resolution / 720
    style = {
        "Default": ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2),
        "Modern": ("Arial", 56, "&H0000FFFF", "&H00FFFFFF", 2, 1, 2),
        "Bouncy": ("Arial Black", 60, "&H0088FF88", "&H00FFFFFF", 4, 1, 2),
        "MrBeast": ("Arial Black", 64, "&H0000FFFF", "&H000000FF", 5, 1, 2),
        "Business": ("Arial", 46, "&H00202020", "&H00FFFFFF", 1, 0, 2),
        "Esencial": ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2),
        "Editorial": ("Georgia", 48, "&H00FFFFFF", "&H00FFFFFF", 2, 0, 2),
        "Neón": ("Arial Black", 60, "&H00FFFF66", "&H00FFFFFF", 4, 1, 2),
    }.get(config.subtitle_template, ("Arial", 58, "&H00FFFFFF", "&H0000FFFF", 3, 1, 2))
    font, size, primary, secondary, outline, shadow, alignment = style
    # Plantilla de marca (Fase 3, ítem 5): los colores y la fuente de
    # marca sustituyen a los de la plantilla; los ajustes manuales de
    # estilo (abajo) siguen ganando a la marca. Los colores se validan
    # (la API ya los valida, pero el kit puede venir de datos viejos).
    if brand:
        font = str(brand.get("font_family") or font)
        brand_primary = str(brand.get("primary_color") or "")
        brand_accent = str(brand.get("accent_color") or "")
        if re.fullmatch(r"#[0-9a-fA-F]{6}", brand_primary):
            primary = _ass_color(brand_primary) or primary
        if re.fullmatch(r"#[0-9a-fA-F]{6}", brand_accent):
            secondary = _ass_color(brand_accent) or secondary
    overrides = config.subtitle_style
    font = overrides.font_family or font
    size = overrides.font_size or size
    primary = _ass_color(overrides.font_color) or primary
    secondary = _ass_color(overrides.highlight_color) or secondary
    alignment = {
        "top": 8,
        "center": 5,
        "bottom": 2,
    }.get(overrides.position, alignment)
    size = round(size * scale)
    caption_max_chars = _caption_max_chars(width, size)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},{primary},{secondary},&H00101010,&H99000000,-1,0,1,{outline},{shadow},{alignment},48,48,96,1
Style: Watermark,Arial,{round(20 * scale)},&H00FFFFFF,&H00FFFFFF,&H66000000,&H55000000,0,0,1,1,0,3,18,18,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogues: list[str] = []
    censor_ranges: list[tuple[float, float]] = []
    # Copia profunda antes de censurar: la transcripción guardada del clip
    # debe conservar las palabras originales y solo el .ass renderizado
    # sale censurado. Sin la copia, _render_clip devolvería las palabras
    # ya reemplazadas por "****".
    words = copy.deepcopy(words)
    for word in words:
        if _censored(str(word["text"])) and config.options.censor:
            censor_ranges.append((float(word["preview_start"]), float(word["preview_end"])))
            word["text"] = "****"
    groups = _caption_groups(words, caption_max_chars)
    for index, group in enumerate(groups):
        begin = float(group[0]["preview_start"])
        end = max(float(group[-1]["preview_end"]), begin + 0.08)
        end = min(end, clip_duration)
        parts: list[str] = []
        for word in group:
            text = _ass_escape(str(word["text"]))
            if len(text) > caption_max_chars:
                text = r"\N".join(
                    textwrap.wrap(
                        text,
                        width=caption_max_chars,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                )
            word_duration = max(
                1,
                round((float(word["preview_end"]) - float(word["preview_start"])) * 100),
            )
            if config.options.keywords and any(
                re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE)
                for keyword in proposal.palabras_clave
                if keyword
            ):
                parts.append(
                    "{\\c"
                    + secondary
                    + "&}"
                    + text
                    + "{\\c"
                    + primary
                    + "&}"
                )
            else:
                parts.append(f"{{\\kf{word_duration}}}{text}")
        caption = " ".join(parts)
        if config.options.emojis and index == len(groups) - 1 and proposal.emojis:
            caption += " " + _ass_escape("".join(proposal.emojis))
        dialogues.append(
            f"Dialogue: 0,{_ass_time(begin)},{_ass_time(end)},Caption,,0,0,0,,{caption}"
        )
    watermark_start = _ass_time(0)
    watermark_end = _ass_time(clip_duration)
    dialogues.append(
        f"Dialogue: 1,{watermark_start},{watermark_end},Watermark,,0,0,0,,hecho con kellyClipAi"
    )
    output_path.write_text(header + "\n".join(dialogues) + "\n", encoding="utf-8")
    return censor_ranges


def _filter_time(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def _broll_cut_ranges(duration: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    cut_duration = min(4.0, max(1.0, duration * 0.2))
    if duration <= 8:
        start = (duration - cut_duration) / 2
        return [(start, start + cut_duration)]
    ranges: list[tuple[float, float]] = []
    start = 4.0
    while start + cut_duration <= duration:
        ranges.append((start, start + cut_duration))
        start += 12.0
    if not ranges:
        start = (duration - cut_duration) / 2
        ranges.append((start, start + cut_duration))
    return ranges


def _crop_coordinate_expression(
    points: list[FaceCropKeyframe],
    coordinate: str,
) -> str:
    if not points:
        raise ValueError("El seguimiento facial no produjo posiciones de recorte.")
    if coordinate not in {"x", "y"}:
        raise ValueError(f"Coordenada de recorte no válida: {coordinate}.")

    expression = f"{getattr(points[-1], coordinate):.3f}"
    for current, following in reversed(list(pairwise(points))):
        duration = following.time - current.time
        if duration <= 0:
            continue
        start_value = getattr(current, coordinate)
        end_value = getattr(following, coordinate)
        interpolation = (
            f"({start_value:.3f}+({end_value - start_value:.3f})"
            f"*(t-{current.time:.3f})/{duration:.3f})"
        )
        expression = (
            f"if(lt(t,{following.time:.3f}),{interpolation},{expression})"
        )
    return expression


def _build_filter(
    intervals: list[tuple[float, float]],
    width: int,
    height: int,
    subtitle_path: Path,
    clip_duration: float,
    censor_ranges: list[tuple[float, float]],
    enhance_audio: bool = False,
    include_broll: bool = False,
    crop_paths: list[list[FaceCropKeyframe] | None] | None = None,
    framing: str = "speaker",
    punch_in_ranges: list[tuple[float, float]] | None = None,
    logo_input: int | None = None,
) -> tuple[str, str, str]:
    parts: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []
    for index, (start, end) in enumerate(intervals):
        video_label, audio_label = f"v{index}", f"a{index}"
        if framing == "split":
            top_source, bottom_source = f"split_top_src{index}", f"split_bottom_src{index}"
            top_label, bottom_label = f"split_top{index}", f"split_bottom{index}"
            pane_height = height // 2
            parts.append(f"[0:v]split=2[{top_source}][{bottom_source}]")
            for source, label, horizontal_crop in (
                (top_source, top_label, "0"),
                (bottom_source, bottom_label, "iw-ow"),
            ):
                parts.append(
                    f"[{source}]trim=start={_filter_time(start)}:end={_filter_time(end)},"
                    f"setpts=PTS-STARTPTS,scale={width}:{pane_height}:"
                    "force_original_aspect_ratio=increase,"
                    f"crop={width}:{pane_height}:x='{horizontal_crop}':y='(ih-oh)/2'"
                    f"[{label}]"
                )
            parts.append(
                f"[{top_label}][{bottom_label}]vstack=inputs=2[{video_label}]"
            )
        elif framing == "blur":
            background_source, foreground_source = (
                f"blur_bg_src{index}",
                f"blur_fg_src{index}",
            )
            background_label, foreground_label = f"blur_bg{index}", f"blur_fg{index}"
            parts.append(f"[0:v]split=2[{background_source}][{foreground_source}]")
            parts.append(
                f"[{background_source}]trim=start={_filter_time(start)}:end={_filter_time(end)},"
                f"setpts=PTS-STARTPTS,scale={width}:{height}:"
                "force_original_aspect_ratio=increase,"
                f"crop={width}:{height},boxblur=20:10[{background_label}]"
            )
            parts.append(
                f"[{foreground_source}]trim=start={_filter_time(start)}:end={_filter_time(end)},"
                f"setpts=PTS-STARTPTS,scale={round(width * 0.8)}:{round(height * 0.8)}:"
                "force_original_aspect_ratio=decrease,format=rgba,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0,"
                f"setsar=1[{foreground_label}]"
            )
            parts.append(
                f"[{background_label}][{foreground_label}]overlay=shortest=1[{video_label}]"
            )
        else:
            crop_path = crop_paths[index] if crop_paths else None
            crop_filter = (
                f"crop={width}:{height}:"
                f"x='{_crop_coordinate_expression(crop_path, 'x')}':"
                f"y='{_crop_coordinate_expression(crop_path, 'y')}'"
                if crop_path and framing == "speaker"
                else f"crop={width}:{height}"
            )
            parts.append(
                f"[0:v]trim=start={_filter_time(start)}:end={_filter_time(end)},"
                f"setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"{crop_filter},setsar=1[{video_label}]"
            )
        parts.append(
            f"[0:a]atrim=start={_filter_time(start)}:end={_filter_time(end)},"
            f"asetpts=PTS-STARTPTS[{audio_label}]"
        )
        video_labels.append(f"[{video_label}]")
        audio_labels.append(f"[{audio_label}]")
    if len(intervals) == 1:
        video_input, audio_input = video_labels[0], audio_labels[0]
    else:
        parts.append(
            "".join(
                label
                for pair in zip(video_labels, audio_labels, strict=True)
                for label in pair
            )
            + f"concat=n={len(intervals)}:v=1:a=1[vcat][acat]"
        )
        video_input, audio_input = "[vcat]", "[acat]"
    if punch_in_ranges:
        # Zoom punch-in (Fase 3, ítem 3b): en los rangos de alta
        # energía se recorta un 15 % (zoom x1.18) y se reescala al
        # tamaño del clip. Las expresiones se evalúan por fotograma
        # sobre la línea de tiempo CONCATENADA (tras quitar silencios
        # y muletillas); el llamador remapea los rangos. Las
        # dimensiones se fuerzan a pares (trunc/2*2) y las posiciones
        # a pares para el submuestreo de croma yuv420p.
        enable = "+".join(
            f"between(t,{_filter_time(start)},{_filter_time(end)})"
            for start, end in punch_in_ranges
        )
        parts.append(
            f"{video_input}crop="
            "w='if(" + enable + ",trunc(iw/1.18/2)*2,iw)':"
            "h='if(" + enable + ",trunc(ih/1.18/2)*2,ih)':"
            "x='trunc((iw-ow)/4)*2':y='trunc((ih-oh)/4)*2',"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}[vzoom]"
        )
        video_input = "[vzoom]"
    if include_broll:
        cut_ranges = _broll_cut_ranges(clip_duration)
        if cut_ranges:
            enable = "+".join(
                f"between(t,{_filter_time(start)},{_filter_time(end)})"
                for start, end in cut_ranges
            )
            parts.append(
                f"[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},trim=duration={_filter_time(clip_duration)},"
                "setpts=PTS-STARTPTS,setsar=1[broll]"
            )
            parts.append(
                f"{video_input}[broll]overlay=shortest=0:eof_action=repeat:"
                f"enable='{enable}'[brollbase]"
            )
            video_input = "[brollbase]"
    parts.append(
        f"{video_input}subtitles=filename='{subtitle_path.name}'[vout]"
    )
    video_output = "[vout]"
    if logo_input is not None:
        # Logo de marca (Fase 3, ítem 5): esquina superior derecha, a
        # un cuarto del ancho del clip. La entrada del logo es una
        # imagen en bucle (-loop 1), así que nunca se agota.
        logo_width = max(64, width // 4)
        parts.append(f"[{logo_input}:v]scale={logo_width}:-1[logo]")
        parts.append(f"{video_output}[logo]overlay=x=W-w-24:y=24[vlogo]")
        video_output = "[vlogo]"
    if censor_ranges:
        enable = "+".join(
            f"between(t,{_filter_time(start)},{_filter_time(end)})"
            for start, end in censor_ranges
        )
        parts.append(f"{audio_input}volume=0:enable='{enable}'[amut]")
        parts.append(
            f"aevalsrc=0.15*sin(2*PI*1000*t):s=48000:d={_filter_time(clip_duration)},"
            f"volume=0.15:enable='{enable}'[beep]"
        )
        parts.append("[amut][beep]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        audio_output = "[aout]"
    else:
        audio_output = audio_input
    if enhance_audio:
        parts.append(
            f"{audio_output}highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,"
            "acompressor=threshold=-20dB:ratio=1.6:attack=20:release=200:makeup=1.5"
            "[aenhanced]"
        )
        audio_output = "[aenhanced]"
    return ";".join(parts), video_output, audio_output


def _render_clip(
    input_path: Path,
    output_path: Path,
    subtitle_path: Path,
    proposal: ClipProposal,
    config: ProjectConfig,
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    on_progress: ProgressCallback,
    removed_ranges: list[tuple[float, float]] | None = None,
    subtitle_overrides: list[tuple[float, float, str]] | None = None,
    output_resolution: int = 720,
    enhance_audio: bool = False,
    broll_path: Path | None = None,
    on_analysis_progress: ProgressCallback | None = None,
    speaker_segments: list[Any] | None = None,
    brand_kit: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    source_duration = proposal.fin - proposal.inicio
    # Split dinámico (Fase 3, ítem 3b): con diarización disponible, un
    # clip de entrevista se renderiza en pantalla dividida aunque el
    # encuadre configurado sea "speaker".
    framing = config.framing
    if config.options.dynamic_split and speaker_segments:
        from .speaker_framing import choose_framing

        framing = choose_framing(
            speaker_segments, proposal.inicio, proposal.fin, config.framing
        )
    intervals = _retained_intervals(input_path, proposal, config.options.silence, ffmpeg_path)
    if removed_ranges:
        intervals = _subtract_intervals(intervals, removed_ranges)
    if config.options.remove_fillers:
        # Muletillas y micro-pausas (Fase 3, ítem 8): mismos intervalos
        # que la eliminación de silencios. Salvaguarda: si quitarlas
        # recortase más del 40 % del clip, se ignora (una transcripción
        # ruidosa no debe vaciar el clip).
        from .fillers import detect_filler_exclusions

        exclusions = detect_filler_exclusions(transcript, proposal)
        if exclusions:
            kept_total = sum(end - start for start, end in intervals)
            removed_total = sum(
                max(0.0, min(end, ex_end) - max(start, ex_start))
                for start, end in intervals
                for ex_start, ex_end in exclusions
            )
            if kept_total > 0 and removed_total <= 0.4 * kept_total:
                intervals = _subtract_intervals(intervals, exclusions)
            else:
                logger.warning(
                    "Las muletillas recortarían más del 40 %% del clip "
                    "(%.1fs de %.1fs); se ignoran.",
                    removed_total,
                    kept_total,
                )
    render_duration = sum(end - start for start, end in intervals)
    if render_duration <= 0.08:
        raise MediaProcessingError("El clip no contiene suficiente audio para exportarse.")
    words = _clip_words(transcript, proposal, intervals)
    if subtitle_overrides:
        words = _apply_subtitle_overrides(words, subtitle_overrides, proposal, intervals)
    censor_ranges = _write_ass(
        subtitle_path,
        words,
        proposal,
        config,
        render_duration,
        intervals,
        output_resolution,
        brand=brand_kit,
    )
    width, height = _ratio_size(config.ratio, output_resolution)
    crop_paths = None
    if config.options.face_tracking and framing == "speaker":
        try:
            speech_intervals = [
                (word.start, word.end)
                for segment in transcript
                for word in segment.words
                if word.start < proposal.fin and word.end > proposal.inicio
            ]
            # Hotfix v1.4.1 (bug 5): el render decodificaba con cv2, que
            # no decodifica AV1 (el video de Kelly) y tumbaba el render
            # con MediaProcessingError. "ffmpeg-dense" mantiene el
            # muestreo denso y la detección de orador por labios
            # (paridad con cv2), cambiando solo el decodificador.
            # NO "auto": su vía de keyframes desactiva los labios y
            # el encuadre de orador degradaría a la cara más grande.
            crop_paths = analyze_face_crop_paths(
                input_path,
                proposal.inicio,
                intervals,
                width,
                height,
                speech_intervals=speech_intervals,
                on_progress=on_analysis_progress,
                frame_source="ffmpeg-dense",
            )
        except FaceTrackingError as error:
            raise MediaProcessingError(str(error)) from error
    # Punch-in (Fase 3, ítem 3b): los picos de energía se detectan en
    # tiempo local del clip y se remapean a la línea de tiempo
    # concatenada (tras quitar silencios y muletillas); un pico que
    # cae en un hueco eliminado se descarta.
    punch_in_ranges: list[tuple[float, float]] | None = None
    if config.options.punch_in:
        from .speaker_framing import energy_peak_ranges

        raw_ranges = energy_peak_ranges(
            input_path, proposal.inicio, source_duration, ffmpeg_path
        )
        punch_in_ranges = []
        for range_start, range_end in raw_ranges:
            mapped_start = _remap_time(range_start, intervals)
            mapped_end = _remap_time(range_end, intervals)
            if (
                mapped_start is not None
                and mapped_end is not None
                and mapped_end - mapped_start >= 0.3
            ):
                punch_in_ranges.append((mapped_start, mapped_end))
        if not punch_in_ranges:
            punch_in_ranges = None
    logo_path = None
    logo_input = None
    if brand_kit:
        candidate = brand_kit.get("logo_path")
        if candidate and Path(str(candidate)).is_file():
            logo_path = Path(str(candidate))
            logo_input = 2 if broll_path is not None else 1
    filter_graph, video_output, audio_output = _build_filter(
        intervals,
        width,
        height,
        subtitle_path,
        render_duration,
        censor_ranges,
        enhance_audio,
        broll_path is not None,
        crop_paths,
        framing,
        punch_in_ranges=punch_in_ranges,
        logo_input=logo_input,
    )
    arguments = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{proposal.inicio:.3f}",
        "-i",
        str(input_path.resolve()),
    ]
    if broll_path is not None:
        if not broll_path.is_file():
            raise MediaProcessingError("No se encontró el archivo de B-roll descargado.")
        arguments.extend(["-stream_loop", "-1", "-i", str(broll_path.resolve())])
    if logo_path is not None:
        arguments.extend(["-loop", "1", "-i", str(logo_path.resolve())])
    arguments.extend([
        "-t",
        f"{source_duration:.3f}",
        "-filter_complex",
        filter_graph,
        "-map",
        video_output,
        "-map",
        audio_output,
        *_video_encoder_args(ffmpeg_path),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(output_path.resolve()),
    ])
    process: subprocess.Popen[str] | None = None
    timer: threading.Timer | None = None
    timed_out = threading.Event()
    try:
        process = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=subtitle_path.resolve().parent,
            **_render_subprocess_options(),
        )
        _register_render_process(process)
        if process.stdout is None:
            raise MediaProcessingError("No se pudo leer el progreso de FFmpeg.")

        def terminate_on_timeout() -> None:
            if process is not None and process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(
            _FFMPEG_RENDER_TIMEOUT_SECONDS, terminate_on_timeout
        )
        timer.daemon = True
        timer.start()
        last_lines: list[str] = []
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                last_lines.append(stripped)
                last_lines = last_lines[-12:]
            if stripped.startswith("out_time_ms="):
                try:
                    processed_seconds = int(stripped.split("=", 1)[1]) / 1_000_000
                    on_progress(min(100.0, max(0.0, processed_seconds / render_duration * 100)))
                except ValueError:
                    continue
        return_code = process.wait()
        if timed_out.is_set():
            raise MediaProcessingError(
                "FFmpeg superó el tiempo máximo de renderizado de 600 segundos."
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaProcessingError(f"FFmpeg no pudo exportar el clip: {error}") from error
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            _unregister_render_process(process)
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()
    if return_code:
        raise MediaProcessingError(f"FFmpeg no pudo exportar el clip: {' '.join(last_lines[-6:])}")
    clip_transcript = words
    return render_duration, clip_transcript


def render_clip(
    input_path: Path,
    output_path: Path,
    subtitle_path: Path,
    proposal: ClipProposal,
    config: ProjectConfig,
    transcript: list[TranscriptSegment],
    ffmpeg_path: str,
    on_progress: ProgressCallback,
    removed_ranges: list[tuple[float, float]] | None = None,
    subtitle_overrides: list[tuple[float, float, str]] | None = None,
    output_resolution: int = 720,
    enhance_audio: bool = False,
    broll_path: Path | None = None,
    on_analysis_progress: ProgressCallback | None = None,
    speaker_segments: list[Any] | None = None,
    brand_kit: dict[str, Any] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _render_clip(
        input_path,
        output_path,
        subtitle_path,
        proposal,
        config,
        transcript,
        ffmpeg_path,
        on_progress,
        removed_ranges,
        subtitle_overrides,
        output_resolution,
        enhance_audio,
        broll_path,
        on_analysis_progress,
        speaker_segments,
        brand_kit,
    )
