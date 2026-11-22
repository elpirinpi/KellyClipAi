"""Agrupación de clips por tema y series numeradas (Fase 3, ítem 2).

Los clips de un mismo video largo suelen repetir tema (el mismo asunto
reaparece en varios momentos). Este módulo agrupa las propuestas por
solape de vocabulario y numera los grupos con más de un clip como una
serie ("Parte 1", "Parte 2"...), para publicarlas en orden.

Puro (sin E/S): recibe ``ClipProposal`` y devuelve copias con los
campos ``topic``/``series``/``part_number`` y el título numerado.
Es idempotente: un título que ya trae " · Parte N" se limpia antes de
reagrupar, así que puede aplicarse tras el pulido de títulos con IA
(que reescribe los títulos) sin duplicar el sufijo.
"""

from __future__ import annotations

import re
from collections import Counter

from .schemas import ClipProposal

# Umbral de solape (coeficiente de solape sobre el vocabulario del clip
# más corto) para considerar que dos clips tratan el mismo tema.
TOPIC_OVERLAP_THRESHOLD = 0.34
_PART_SUFFIX = re.compile(r"\s*·\s*Parte\s+\d+\s*$")
_TOKEN = re.compile(r"[a-záéíóúñü0-9]+")
_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "y", "o", "a", "con", "por", "para", "que", "es", "son", "se",
    "su", "sus", "al", "lo", "como", "más", "pero", "este", "esta",
    "esto", "eso", "esa", "mi", "mis", "tu", "tus", "yo", "tú", "él",
    "ella", "nosotros", "nosotras", "ustedes", "ellos", "ellas", "me",
    "te", "nos", "le", "les", "the", "and", "of", "to", "in", "for", "on",
    # Marcadores del habla coloquial: aparecen en casi cualquier clip
    # y no dicen nada del tema ("vale", "aunque", "vamos"…).
    "aunque", "vale", "vamos", "vaya", "bueno", "pues", "mira", "miren",
    "oye", "oiga", "claro", "sabes", "sabe", "verdad", "entonces",
    "ahora", "siempre", "nunca", "también", "tampoco", "mucho", "poco",
    "cosa", "cosas", "así", "bien", "mal", "creo", "quiero", "puedo",
    "puede", "dice", "digo", "voy", "ves", "gente", "chicos", "hola",
    "ok", "eh",
    # Palabras genéricas del propio producto: aparecen en casi cualquier
    # título y no definen un tema ("Clip uno" y "Clip dos" no son una
    # serie solo por compartir "clip").
    "clip", "clips", "video", "videos", "momento", "momentos", "parte",
    "partes", "short", "shorts", "reel", "reels", "tiktok",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _proposal_tokens(proposal: ClipProposal) -> set[str]:
    tokens = _tokens(proposal.titulo)
    for keyword in proposal.palabras_clave:
        tokens |= _tokens(keyword)
    return tokens


def _base_title(title: str) -> str:
    """Título sin el sufijo de serie (para reagrupar sin duplicarlo)."""
    return _PART_SUFFIX.sub("", title).strip()


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / min(len(left), len(right))


def _group_label(members: list[ClipProposal]) -> str:
    """Etiqueta del tema: el token con más peso del grupo.

    Los tokens del título pesan el doble que los de las palabras clave
    (el título es la frase más contundente del clip); todo filtrado
    por las stopwords (incluidos los marcadores coloquiales: una
    palabra clave como "vale" o "aunque" no puede ser la etiqueta).
    El desempate es alfabético para que la etiqueta sea determinista.
    """
    token_counts: Counter[str] = Counter()
    for proposal in members:
        for token in _tokens(proposal.titulo):
            token_counts[token] += 2
        for keyword in proposal.palabras_clave:
            for token in _tokens(keyword):
                token_counts[token] += 1
    if token_counts:
        return sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return "Momentos"


def group_by_topic(
    proposals: list[ClipProposal],
    *,
    threshold: float = TOPIC_OVERLAP_THRESHOLD,
) -> list[list[ClipProposal]]:
    """Agrupa propuestas por tema (solape de vocabulario).

    Voraz y cronológico: cada propuesta se une al grupo con mayor
    solape si supera el umbral; si no, abre un grupo nuevo. El orden
    dentro de cada grupo es cronológico (por ``inicio``).
    """
    ordered = sorted(proposals, key=lambda proposal: proposal.inicio)
    groups: list[list[ClipProposal]] = []
    group_tokens: list[set[str]] = []
    for proposal in ordered:
        tokens = _proposal_tokens(proposal)
        best_index = -1
        best_overlap = 0.0
        for index, existing in enumerate(group_tokens):
            overlap = _overlap(tokens, existing)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index >= 0 and best_overlap >= threshold:
            groups[best_index].append(proposal)
            group_tokens[best_index] |= tokens
        else:
            groups.append([proposal])
            group_tokens.append(set(tokens))
    return groups


def _with_part_suffix(title: str, part: int) -> str:
    """Título con " · Parte N", recortado al tope de 160 caracteres."""
    base = _base_title(title)
    suffix = f" · Parte {part}"
    if len(base) + len(suffix) > 160:
        base = base[: 160 - len(suffix) - 1].rstrip()
    return base + suffix


def assign_series(proposals: list[ClipProposal]) -> list[ClipProposal]:
    """Numera las series y anota el tema de cada propuesta.

    Los grupos de 2+ clips reciben ``series`` (etiqueta del tema),
    ``part_number`` (1, 2, 3… en orden cronológico) y el título
    numerado; los grupos de un solo clip solo reciben ``topic``.
    Devuelve copias; la entrada no se modifica. Reejecutable: los
    sufijos previos se limpian antes de reagrupar.
    """
    if not proposals:
        return []
    cleaned = [
        proposal.model_copy(update={"titulo": _base_title(proposal.titulo)})
        for proposal in proposals
    ]
    output: list[ClipProposal] = []
    for group in group_by_topic(cleaned):
        label = _group_label(group)
        if len(group) < 2:
            output.append(group[0].model_copy(update={"topic": label}))
            continue
        for index, proposal in enumerate(group, start=1):
            output.append(
                proposal.model_copy(
                    update={
                        "titulo": _with_part_suffix(proposal.titulo, index),
                        "topic": label,
                        "series": label,
                        "part_number": index,
                    }
                )
            )
    return sorted(output, key=lambda proposal: proposal.inicio)
