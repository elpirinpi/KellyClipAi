"""Detección de muletillas y micro-pausas para jump cuts (Fase 3, ítem 8).

Reutiliza el mecanismo de intervalos de la eliminación de silencios
(``media._subtract_intervals``): este módulo solo calcula QUÉ quitar,
en tiempos locales del clip (relativos a ``proposal.inicio``); el
render resta esos intervalos de los retenidos, igual que con los
rangos quitados a mano en el editor.

Muletillas
----------
Palabras de relleno del español hablado ("eh", "em", "este", "o sea").
"Este" también es demostrativo, así que una palabra solo cuenta como
muletilla si está AISLADA: con una pausa (hueco entre palabras) de al
menos 0,12 s a un lado. Un "este" pegado a la frase siguiente no se
toca.

Micro-pausas
------------
Huecos entre palabras de 0,25–0,9 s: demasiado cortos para la
eliminación de silencios (que corta desde 0,45 s de silencio real y
solo cuando está activada) pero visibles como titubeos en vertical.
Se recortan dejando 0,05 s a cada lado para no pegar las palabras.

Puro (sin E/S ni ffmpeg): trabaja sobre la transcripción por palabra
que ya guarda el proyecto.
"""

from __future__ import annotations

from .schemas import ClipProposal, TranscriptSegment

# Muletillas de una palabra (minúsculas, sin puntuación). "bueno" y
# "pues" NO están: aparecen demasiado como conectores con contenido.
FILLER_WORDS = {
    "eh", "ehh", "ehhh", "em", "emm", "mmm", "mm", "ah", "ahh",
    "este", "estee", "esto", "um", "uh", "er",
}
# Muletillas de dos palabras (pares consecutivos).
FILLER_BIGRAMS = {("o", "sea"), ("o", "sea,")}
# Hueco mínimo a un lado de la palabra para considerarla aislada.
MIN_ISOLATION_GAP = 0.12
# Rango de micro-pausas que se recortan (segundos entre palabras).
MICRO_PAUSE_MIN = 0.25
MICRO_PAUSE_MAX = 0.9
# Respiro que se conserva a cada lado de una micro-pausa recortada.
MICRO_PAUSE_KEEP = 0.05
# Fragmento mínimo que merece un intervalo (igual que en media.py).
_MIN_FRAGMENT = 0.08


def _clean(text: str) -> str:
    return text.strip().casefold().strip(".,;:!?¿¡\"'()[]")


def _clip_words(
    transcript: list[TranscriptSegment],
    proposal: ClipProposal,
) -> list[tuple[float, float, str]]:
    """Palabras del clip en tiempo LOCAL, ordenadas por inicio."""
    words: list[tuple[float, float, str]] = []
    for segment in transcript:
        for word in segment.words:
            if word.end <= proposal.inicio or word.start >= proposal.fin:
                continue
            start = max(0.0, word.start - proposal.inicio)
            end = min(proposal.fin - proposal.inicio, word.end - proposal.inicio)
            if end <= start:
                continue
            words.append((start, end, word.text))
    words.sort(key=lambda item: (item[0], item[1]))
    return words


def detect_filler_exclusions(
    transcript: list[TranscriptSegment],
    proposal: ClipProposal,
    *,
    min_isolation_gap: float = MIN_ISOLATION_GAP,
    micro_pause_min: float = MICRO_PAUSE_MIN,
    micro_pause_max: float = MICRO_PAUSE_MAX,
    micro_pause_keep: float = MICRO_PAUSE_KEEP,
) -> list[tuple[float, float]]:
    """Intervalos locales a quitar: muletillas + micro-pausas.

    Devuelve intervalos ordenados y sin solapes (las muletillas que
    caen dentro de una micro-pausa ya recortada no se duplican).
    """
    words = _clip_words(transcript, proposal)
    if len(words) < 2:
        return []
    exclusions: list[tuple[float, float]] = []

    # Micro-pausas entre palabras consecutivas.
    for (_, prev_end, _), (next_start, _, _) in zip(words, words[1:], strict=False):
        gap = next_start - prev_end
        if micro_pause_min <= gap <= micro_pause_max:
            start = prev_end + micro_pause_keep
            end = next_start - micro_pause_keep
            if end - start >= _MIN_FRAGMENT:
                exclusions.append((start, end))

    # Muletillas aisladas (una palabra o un bigrama).
    index = 0
    while index < len(words):
        start, end, text = words[index]
        gap_before = start - words[index - 1][1] if index > 0 else 0.0
        gap_after = (
            words[index + 1][0] - end if index + 1 < len(words) else 0.0
        )
        isolated = gap_before >= min_isolation_gap or gap_after >= min_isolation_gap
        # Bigrama ("o sea"): las dos palabras seguidas y aisladas.
        if (
            isolated
            and index + 1 < len(words)
            and (_clean(text), _clean(words[index + 1][2])) in FILLER_BIGRAMS
        ):
            bigram_end = words[index + 1][1]
            gap_after_bigram = (
                words[index + 2][0] - bigram_end
                if index + 2 < len(words)
                else 0.0
            )
            if gap_before >= min_isolation_gap or gap_after_bigram >= min_isolation_gap:
                exclusions.append((start, bigram_end))
                index += 2
                continue
        if isolated and _clean(text) in FILLER_WORDS:
            exclusions.append((start, end))
        index += 1

    # Ordenar y fusionar solapes (una muletilla puede caer en una
    # micro-pausa ya recortada).
    exclusions.sort()
    merged: list[tuple[float, float]] = []
    for start, end in exclusions:
        if end - start < _MIN_FRAGMENT:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
