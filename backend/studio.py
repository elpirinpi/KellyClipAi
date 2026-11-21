"""Conversión de transcripción por palabra a cues para el editor visual.

El editor ShortStudio muestra subtítulos sincronizados con el video que
está editando. Los cues se derivan de la transcripción por palabra que
ya guarda el proyecto (``transcript_json``) o el clip
(``clip["transcript"]``); este módulo es puro (sin E/S ni dependencias
del resto del backend) para poder probarlo en aislamiento.

Regla de agrupación
-------------------
- Se ignoran las palabras sin tiempos válidos (sin ``start``/``end``,
  no numéricos, ``end <= start`` o ``start < 0``) y las sin texto.
- Las palabras se ordenan por inicio y se agrupan en cues de texto
  plano, sin marcadores de resaltado.
- Un cue se cierra antes de añadir una palabra cuando se cumple alguna
  de estas condiciones:

  * añadirla superaría ~42 caracteres (sin partir palabras: una palabra
    sola más larga que el tope forma su propio cue);
  * la duración del cue superaría ~6 s (del inicio del cue al fin de
    la palabra candidata);
  * la palabra anterior cierra una frase (``.`` ``!`` ``?``) y el cue
    ya tiene ~20 caracteres (corte preferente en fin de frase).

- Cada cue es ``{"id": "cue-N", "start": float, "end": float,
  "text": str}``, con los tiempos redondeados a 3 decimales como en el
  resto del backend.

Base de tiempos
---------------
Todos los cues usan tiempos ABSOLUTOS del video fuente, tanto los de
un clip como los de un rango del proyecto: el editor visual reproduce
siempre el video fuente (nunca el clip renderizado) y el renderizador
del canvas empareja cada cue contra ``video.currentTime``. Los tiempos
locales ``preview_start``/``preview_end`` que guarda ``render_clip``
se ignoran aquí (siguen usándose en el registro del clip exportado).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Regla de agrupación (ver el docstring del módulo).
DEFAULT_MAX_CUE_CHARS = 42
DEFAULT_SENTENCE_BREAK_MIN_CHARS = 20
DEFAULT_MAX_CUE_DURATION = 6.0
_SENTENCE_END = ".!?"


def _as_float(value: Any) -> float | None:
    """Un número finito como float, o None si no es un número válido."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _word_field(word: Any, name: str) -> Any:
    if isinstance(word, Mapping):
        return word.get(name)
    return getattr(word, name, None)


def _normalize_word(word: Any) -> dict[str, Any] | None:
    """Una palabra con tiempos válidos y texto, o None si no sirve."""
    start = _as_float(_word_field(word, "start"))
    end = _as_float(_word_field(word, "end"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    text = _word_field(word, "text")
    text = str(text).strip() if text is not None else ""
    if not text:
        return None
    return {"start": start, "end": end, "text": text}


def words_to_cues(
    words: list[Any] | None,
    *,
    max_chars: int = DEFAULT_MAX_CUE_CHARS,
    sentence_break_min_chars: int = DEFAULT_SENTENCE_BREAK_MIN_CHARS,
    max_duration: float = DEFAULT_MAX_CUE_DURATION,
) -> list[dict[str, Any]]:
    """Agrupa palabras con ``start``/``end``/``text`` en cues (ver arriba)."""
    normalized = [
        word for word in (_normalize_word(item) for item in words or []) if word is not None
    ]
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    cues: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_text = ""

    def flush() -> None:
        nonlocal current, current_text
        if not current:
            return
        cues.append(
            {
                "id": f"cue-{len(cues) + 1}",
                "start": round(float(current[0]["start"]), 3),
                "end": round(float(current[-1]["end"]), 3),
                "text": current_text,
            }
        )
        current = []
        current_text = ""

    for word in normalized:
        if current:
            candidate = f"{current_text} {word['text']}"
            too_long = len(candidate) > max_chars
            too_old = float(word["end"]) - float(current[0]["start"]) > max_duration
            sentence_break = (
                current_text[-1] in _SENTENCE_END and len(current_text) >= sentence_break_min_chars
            )
            if too_long or too_old or sentence_break:
                flush()
        current.append(word)
        current_text = f"{current_text} {word['text']}" if current_text else str(word["text"])
    flush()
    return cues


def clip_transcript_words(clip: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Palabras del transcript del clip en tiempo ABSOLUTO del fuente.

    ``render_clip`` guarda tiempos absolutos (``start``/``end``, en el
    video fuente) y locales (``preview_start``/``preview_end``, en el
    clip renderizado). El editor visual reproduce siempre el video
    fuente, así que los cues usan los absolutos y se ignoran los
    locales. Las palabras sin tiempos utilizables se devuelven tal cual
    y las descarta ``words_to_cues``.
    """
    words: list[dict[str, Any]] = []
    for raw in clip.get("transcript") or []:
        if not isinstance(raw, Mapping):
            continue
        start = _as_float(raw.get("start"))
        end = _as_float(raw.get("end"))
        if start is None or end is None:
            continue
        words.append({"start": start, "end": end, "text": raw.get("text")})
    return words


def project_transcript_words(
    transcript: list[Any] | None,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Palabras del transcript del proyecto que se solapan con [start, end).

    Tiempos ABSOLUTOS (video fuente): el editor de clips manuales
    reproduce el video fuente. Se incluye una palabra si se solapa con
    el rango aunque empiece antes o termine después de él.
    """
    words: list[dict[str, Any]] = []
    for segment in transcript or []:
        segment_words = _word_field(segment, "words")
        for raw in segment_words or []:
            word = _normalize_word(raw)
            if word is None:
                continue
            if float(word["end"]) <= start or float(word["start"]) >= end:
                continue
            words.append(word)
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def project_duration(project: Mapping[str, Any]) -> float | None:
    """Duración del video fuente, o None si no se conoce.

    Primero ``metadata["duration"]`` (la sonda del pipeline); si falta,
    el fin de la última palabra de la transcripción.
    """
    metadata = project.get("metadata")
    duration = _as_float(metadata.get("duration")) if isinstance(metadata, Mapping) else None
    if duration is not None and duration > 0:
        return duration
    max_end = 0.0
    for segment in project.get("transcript") or []:
        for raw in _word_field(segment, "words") or []:
            word = _normalize_word(raw)
            if word is not None:
                max_end = max(max_end, float(word["end"]))
    return max_end if max_end > 0 else None


def local_transcript_words(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Palabras de un rango como transcript de clip (forma de render_clip).

    Cada palabra conserva sus tiempos absolutos (``start``/``end``) y
    añade los locales (``preview_start``/``preview_end``) recortados al
    rango [start, end). Lo usa el export del editor para que el clip
    resultante siga teniendo cues válidos sin re-transcribir.
    """
    output: list[dict[str, Any]] = []
    for word in words:
        preview_start = max(0.0, float(word["start"]) - start)
        preview_end = min(end - start, float(word["end"]) - start)
        if preview_end <= preview_start:
            continue
        output.append(
            {
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "preview_start": round(preview_start, 3),
                "preview_end": round(preview_end, 3),
                "text": str(word["text"]),
            }
        )
    return output
