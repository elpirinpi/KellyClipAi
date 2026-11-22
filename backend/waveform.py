"""Forma de onda para la línea de tiempo del Studio (Fase 3, ítem 4b).

El navegador no debería decodificar el audio de un video largo solo
para dibujar la onda (memoria y tiempo); el backend calcula los picos
con ffmpeg (PCM f32le mono 8 kHz, en streaming, memoria O(1)) y la
UI dibuja las barras en un <canvas>.

Los picos son el máximo absoluto por cubo, normalizados a 0–1 por el
máximo global del rango: la onda muestra la forma relativa, no el
volumen absoluto. Con caché por identidad de archivo + rango + cubos
(el mismo patrón que el análisis de silencios de ``media.py``).
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_WAVEFORM_SAMPLE_RATE = 8000
_WAVEFORM_CHUNK_SECONDS = 10
_WAVEFORM_CACHE_MAX_ENTRIES = 8
_WAVEFORM_CACHE: dict[tuple[Any, ...], list[float]] = {}
_WAVEFORM_CACHE_LOCK = threading.Lock()


def _file_identity(path: Path) -> tuple[float, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def peaks_from_samples(samples: np.ndarray, buckets: int) -> list[float]:
    """Máximo absoluto por cubo, normalizado a 0–1 por el máximo global."""
    if buckets <= 0 or samples.size == 0:
        return [0.0] * max(0, buckets)
    magnitudes = np.abs(samples.astype(np.float64))
    edges = np.linspace(0, magnitudes.size, buckets + 1).astype(np.int64)
    peaks = np.zeros(buckets, dtype=np.float64)
    for index in range(buckets):
        chunk = magnitudes[edges[index] : edges[index + 1]]
        if chunk.size:
            peaks[index] = float(chunk.max())
    top = float(peaks.max()) if peaks.size else 0.0
    if top > 0:
        peaks /= top
    return [round(float(value), 4) for value in peaks]


def compute_waveform_peaks(
    input_path: Path,
    start: float,
    duration: float,
    ffmpeg_path: str,
    *,
    buckets: int = 1200,
) -> list[float]:
    """Picos de la onda de [start, start+duration), normalizados 0–1.

    Lanza ``ValueError`` con rangos inválidos; sin audio o con ffmpeg
    roto devuelve ceros (la UI dibuja una pista vacía en vez de
    fallar).
    """
    if buckets <= 0:
        raise ValueError("El número de cubos debe ser mayor que cero.")
    if duration <= 0 or start < 0:
        raise ValueError("El rango de la forma de onda no es válido.")
    identity = _file_identity(input_path)
    cache_key = (
        (identity, round(start, 3), round(duration, 3), buckets, ffmpeg_path)
        if identity is not None
        else None
    )
    if cache_key is not None:
        with _WAVEFORM_CACHE_LOCK:
            cached = _WAVEFORM_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
    peaks = _decode_peaks(input_path, start, duration, ffmpeg_path, buckets)
    if cache_key is not None:
        with _WAVEFORM_CACHE_LOCK:
            _WAVEFORM_CACHE[cache_key] = list(peaks)
            while len(_WAVEFORM_CACHE) > _WAVEFORM_CACHE_MAX_ENTRIES:
                _WAVEFORM_CACHE.pop(next(iter(_WAVEFORM_CACHE)))
    return peaks


def _decode_peaks(
    input_path: Path,
    start: float,
    duration: float,
    ffmpeg_path: str,
    buckets: int,
) -> list[float]:
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
        str(_WAVEFORM_SAMPLE_RATE),
        "-f",
        "f32le",
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
        logger.warning("No se pudo calcular la forma de onda (%s).", error)
        return [0.0] * buckets
    chunks: list[np.ndarray] = []
    try:
        assert process.stdout is not None
        chunk_bytes = _WAVEFORM_SAMPLE_RATE * _WAVEFORM_CHUNK_SECONDS * 4
        while True:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            chunks.append(np.frombuffer(data, dtype=np.float32))
        process.wait(timeout=300)
    except (subprocess.TimeoutExpired, OSError) as error:
        logger.warning("El cálculo de la forma de onda falló (%s).", error)
        try:
            process.kill()
        except OSError:
            pass
        return [0.0] * buckets
    if process.returncode != 0 or not chunks:
        return [0.0] * buckets
    samples = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return peaks_from_samples(samples, buckets)
