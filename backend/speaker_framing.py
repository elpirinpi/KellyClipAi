"""Rostros multi-orador: split dinámico y punch-in (Fase 3, ítem 3b).

Dos piezas independientes:

1. ``choose_framing``: decide el encuadre de un clip a partir de los
   turnos de diarización. En una entrevista (dos hablantes que
   alternan), el encuadre "speaker" (seguir a una sola cara) pierde a
   quien escucha; el "split" (pantalla dividida, ya implementado en
   ``media._build_filter``) muestra a los dos. La decisión es por
   clip: solo se asciende de "speaker" a "split", nunca se toca un
   encuadre elegido a mano ("blur", "center" o "split" explícito).

2. ``energy_peak_ranges``: momentos de alta energía del audio para el
   zoom punch-in. Decodifica solo el rango del clip con ffmpeg (PCM
   s16le mono 8 kHz, en streaming, memoria O(1)) y marca las ventanas
   cuyo RMS supera el percentil 85. Nunca lanza: sin audio o sin
   ffmpeg devuelve [] y el render sigue sin punch-in.

Límite honesto: el seguimiento facial (``face_tracking``) produce UNA
sola pista de recorte por clip (la cara activa por movimiento labial).
Un split con seguimiento por hablante individual (cada panel siguiendo
a su cara) exigiría rehacer esa pista; aquí el split usa los paneles
estáticos izquierda/derecha que ya existen.
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Condiciones para ascender a "split" en un clip.
_MIN_SPEAKER_SHARE = 0.20  # cada hablante, al menos el 20 % del habla
_MIN_TURNS_PER_SPEAKER = 2
_MIN_ALTERNATIONS = 3
# Energía: ventana de RMS y percentil de corte para el punch-in.
ENERGY_WINDOW_SECONDS = 0.25
ENERGY_PERCENTILE = 85.0
ENERGY_MIN_RANGE_SECONDS = 0.5
ENERGY_MAX_RANGES = 3
_ENERGY_SAMPLE_RATE = 8000
_ENERGY_CHUNK_SECONDS = 5


def _segment_tuple(segment: Any) -> tuple[float, float, int] | None:
    """Un segmento de diarización como (inicio, fin, hablante)."""
    if isinstance(segment, Mapping):
        start, end, speaker = (
            segment.get("start"),
            segment.get("end"),
            segment.get("speaker"),
        )
    else:
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        speaker = getattr(segment, "speaker", None)
    try:
        start_f, end_f, speaker_i = float(start), float(end), int(speaker)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start_f) and math.isfinite(end_f)) or end_f <= start_f:
        return None
    return start_f, end_f, speaker_i


def speaker_turns_in_clip(
    segments: list[Any] | None,
    clip_start: float,
    clip_end: float,
) -> list[tuple[float, float, int]]:
    """Turnos recortados al clip, ordenados por inicio."""
    turns: list[tuple[float, float, int]] = []
    for segment in segments or []:
        parsed = _segment_tuple(segment)
        if parsed is None:
            continue
        start, end, speaker = parsed
        start = max(start, clip_start)
        end = min(end, clip_end)
        if end > start:
            turns.append((start, end, speaker))
    turns.sort(key=lambda turn: (turn[0], turn[1]))
    return turns


def choose_framing(
    segments: list[Any] | None,
    clip_start: float,
    clip_end: float,
    configured: str,
) -> str:
    """Encuadre efectivo de un clip con diarización disponible.

    Solo asciende "speaker" a "split" cuando dos hablantes alternan
    de verdad dentro del clip (cada uno habla al menos el 20 % del
    tiempo con voz, con 2+ turnos cada uno y 3+ alternancias). Sin
    segmentos, con un solo hablante o con otro encuadre configurado,
    devuelve el configurado tal cual.
    """
    if configured != "speaker":
        return configured
    turns = speaker_turns_in_clip(segments, clip_start, clip_end)
    if len(turns) < _MIN_ALTERNATIONS + 1:
        return configured
    speech_by_speaker: dict[int, float] = {}
    for start, end, speaker in turns:
        speech_by_speaker[speaker] = speech_by_speaker.get(speaker, 0.0) + (end - start)
    total_speech = sum(speech_by_speaker.values())
    if total_speech <= 0:
        return configured
    turns_by_speaker: dict[int, int] = {}
    alternations = 0
    for index, (_, _, speaker) in enumerate(turns):
        turns_by_speaker[speaker] = turns_by_speaker.get(speaker, 0) + 1
        if index and speaker != turns[index - 1][2]:
            alternations += 1
    dominant = [
        speaker
        for speaker, seconds in speech_by_speaker.items()
        if seconds / total_speech >= _MIN_SPEAKER_SHARE
        and turns_by_speaker.get(speaker, 0) >= _MIN_TURNS_PER_SPEAKER
    ]
    if len(dominant) >= 2 and alternations >= _MIN_ALTERNATIONS:
        return "split"
    return configured


def ranges_from_rms(
    rms: np.ndarray,
    window_seconds: float,
    *,
    percentile: float = ENERGY_PERCENTILE,
    min_range_seconds: float = ENERGY_MIN_RANGE_SECONDS,
    max_ranges: int = ENERGY_MAX_RANGES,
) -> list[tuple[float, float]]:
    """Ventanas de alta energía a partir de un RMS por ventana.

    Las ventanas por encima del percentil se fusionan en rangos
    contiguos; los rangos se ordenan por energía total y se queda con
    los ``max_ranges`` mejores (en orden cronológico). Con energía
    constante (p. ej. un tono puro) no hay ninguna ventana por encima
    del percentil y devuelve [].
    """
    if rms.size == 0:
        return []
    # Umbral: el percentil, con suelo en 1.6x la mediana. Sin el suelo,
    # un audio de energía casi constante (un tono, ruido de fondo) da
    # "picos" por variaciones mínimas de codificación.
    threshold = max(
        float(np.percentile(rms, percentile)),
        float(np.median(rms)) * 1.6,
    )
    above = rms > threshold
    if not above.any():
        return []
    ranges: list[tuple[float, float, float]] = []
    start_index = 0
    in_range = False
    for index, hot in enumerate(above):
        if hot and not in_range:
            start_index, in_range = index, True
        elif not hot and in_range:
            ranges.append(
                (
                    start_index * window_seconds,
                    index * window_seconds,
                    float(rms[start_index:index].sum()),
                )
            )
            in_range = False
    if in_range:
        ranges.append(
            (
                start_index * window_seconds,
                len(rms) * window_seconds,
                float(rms[start_index:].sum()),
            )
        )
    # Extender los rangos cortos al mínimo (hacia adelante, sin pasar
    # del final) para que el zoom se note.
    total = len(rms) * window_seconds
    extended: list[tuple[float, float, float]] = []
    for start, end, energy in ranges:
        if end - start < min_range_seconds:
            end = min(total, start + min_range_seconds)
        if end - start >= min_range_seconds * 0.6:
            extended.append((start, end, energy))
    extended.sort(key=lambda item: -item[2])
    chosen = sorted(extended[:max_ranges], key=lambda item: item[0])
    return [(start, end) for start, end, _ in chosen]


def energy_peak_ranges(
    input_path: Any,
    start: float,
    duration: float,
    ffmpeg_path: str,
    *,
    window_seconds: float = ENERGY_WINDOW_SECONDS,
    percentile: float = ENERGY_PERCENTILE,
    max_ranges: int = ENERGY_MAX_RANGES,
) -> list[tuple[float, float]]:
    """Picos de energía del audio en tiempos LOCALES del clip.

    Decodifica solo [start, start+duration) con ffmpeg (s16le mono
    8 kHz, en streaming). Nunca lanza: cualquier fallo devuelve [].
    """
    if duration <= 0 or not ffmpeg_path:
        return []
    window_samples = max(1, int(_ENERGY_SAMPLE_RATE * window_seconds))
    chunk_samples = _ENERGY_SAMPLE_RATE * _ENERGY_CHUNK_SECONDS
    rms_values: list[float] = []
    leftover = np.empty(0, dtype=np.float64)
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_ENERGY_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError) as error:
        logger.warning("No se pudo analizar la energía del audio (%s).", error)
        return []
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(chunk_samples * 2)
            if not chunk:
                break
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float64) / 32768.0
            samples = np.concatenate([leftover, samples]) if leftover.size else samples
            complete = (samples.size // window_samples) * window_samples
            if complete:
                windows = samples[:complete].reshape(-1, window_samples)
                rms_values.extend(
                    float(value)
                    for value in np.sqrt(np.mean(windows * windows, axis=1))
                )
            leftover = samples[complete:]
        process.wait(timeout=120)
    except (subprocess.TimeoutExpired, OSError) as error:
        logger.warning("El análisis de energía del audio falló (%s).", error)
        try:
            process.kill()
        except OSError:
            pass
        return []
    if process.returncode != 0:
        return []
    if leftover.size >= window_samples // 2:
        rms_values.append(float(np.sqrt(np.mean(leftover * leftover))))
    return ranges_from_rms(
        np.asarray(rms_values, dtype=np.float64),
        window_seconds,
        percentile=percentile,
        max_ranges=max_ranges,
    )
