"""Traducción de subtítulos con el LLM (Fase 3, ítem 7).

Traduce la transcripción del clip y exporta SRT/VTT en otro idioma
reutilizando ``media.format_subtitles``. El flujo:

1. Las palabras del clip se agrupan en cues (``studio.words_to_cues``,
   con los tiempos LOCALES del clip renderizado).
2. Los cues se traducen en UNA llamada al LLM, numerados ("1. texto")
   para que la respuesta vuelva alineada por número.
3. El texto traducido de cada cue se reparte en palabras con tiempos
   distribuidos uniformemente sobre el tramo del cue, en la misma
   forma que ``render_clip`` guarda (``preview_start``/``preview_end``),
   lista para ``format_subtitles``.

Reglas de fallo: sin clave de IA configurada, el endpoint responde
409 con un mensaje claro (la traducción es explícita del usuario, no
un pulido opcional). Si el LLM devuelve menos líneas o líneas sin
número, los cues sin traducción conservan el texto original: nunca se
pierde un subtítulo por un fallo de formato.
"""

from __future__ import annotations

import re
from typing import Any

from .llm import ResolvedLLM, chat_completion_text
from .studio import words_to_cues

# Idiomas destino ofrecidos en la UI (código -> nombre en español).
SUPPORTED_LANGUAGES = {
    "en": "inglés",
    "fr": "francés",
    "de": "alemán",
    "it": "italiano",
    "pt": "portugués",
    "es": "español",
}
_NUMBERED_LINE = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$")


def clip_words_to_cues(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Palabras del clip (tiempos locales) agrupadas en cues."""
    local = [
        {
            "start": word.get("preview_start"),
            "end": word.get("preview_end"),
            "text": word.get("text"),
        }
        for word in words
    ]
    return words_to_cues(local)


def _numbered_prompt(cues: list[dict[str, Any]], language_name: str) -> str:
    lines = "\n".join(
        f"{index}. {cue['text']}" for index, cue in enumerate(cues, start=1)
    )
    return (
        f"Traduce cada línea al {language_name}. Son subtítulos de un video: "
        "mantén el sentido y el registro coloquial, sin añadir explicaciones. "
        "Devuelve EXACTAMENTE las mismas líneas numeradas, en el mismo orden, "
        "una por línea, con el formato \"N. texto traducido\". No añadas "
        "líneas sin numerar ni texto fuera de la lista.\n\n"
        f"{lines}"
    )


def _parse_numbered_response(
    content: str,
    cues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cues con el texto traducido; los sin traducción conservan el original."""
    translated: dict[int, str] = {}
    for line in content.splitlines():
        match = _NUMBERED_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        text = match.group(2).strip()
        if 1 <= index <= len(cues) and text:
            translated[index] = text
    output: list[dict[str, Any]] = []
    for position, cue in enumerate(cues, start=1):
        output.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": translated.get(position, str(cue["text"])),
            }
        )
    return output


def translate_cues(
    cues: list[dict[str, Any]],
    target_language: str,
    llm: ResolvedLLM,
) -> list[dict[str, Any]]:
    """Traduce los cues con el LLM (una sola llamada)."""
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Idioma destino no admitido: {target_language}.")
    if not cues:
        return []
    language_name = SUPPORTED_LANGUAGES[target_language]
    content = chat_completion_text(
        llm,
        system=(
            "Eres un traductor de subtítulos. Traduce con precisión, "
            "manteniendo el tono coloquial del original."
        ),
        user=_numbered_prompt(cues, language_name),
    )
    return _parse_numbered_response(content, cues)


def translated_words(cues: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Cues traducidos como palabras con tiempos (forma de render_clip).

    Cada palabra del texto traducido recibe una porción uniforme del
    tramo del cue. ``format_subtitles`` solo usa ``preview_start``/
    ``preview_end``; ``start``/``end`` se rellenan con los mismos
    valores locales para que la estructura sea completa.
    """
    words: list[dict[str, object]] = []
    for cue in cues:
        start = float(cue["start"])
        end = float(cue["end"])
        tokens = str(cue["text"]).split()
        if not tokens or end <= start:
            continue
        span = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            word_start = round(start + index * span, 3)
            word_end = round(start + (index + 1) * span, 3)
            words.append(
                {
                    "start": word_start,
                    "end": word_end,
                    "preview_start": word_start,
                    "preview_end": word_end,
                    "text": token,
                }
            )
    return words
