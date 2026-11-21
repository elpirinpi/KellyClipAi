"""Motor de detección de momentos por señales (Tier 0, Fase A, v1.1.0).

Selecciona los mejores momentos de un video midiendo señales — energía de
audio, densidad y ritmo de habla, ganchos de texto — en vez de pedirle a
un LLM que lea la transcripción. Es el motor POR DEFECTO de la app: no
necesita internet, clave API ni descargas (0 MB adicionales, solo numpy +
ffmpeg, que la app ya tiene). Sin PyTorch jamás.

Diseño (patrón del proyecto Chopify, MIT):
- Unidad base: la frase (de faster-whisper con word-timestamps). Cada
  frase es un posible inicio; el final es el final natural de una frase
  posterior dentro del rango de duración pedido. Los clips nunca cortan
  una idea a la mitad.
- Puntuación ABSOLUTA 0–10 + umbral (por defecto 5.0, ajustable en
  Ajustes): un video aburrido da pocos clips o ninguno — "cero clips"
  es un resultado válido, mejor que inventar clips malos.
  Calibración v1.1.1: las curvas de los canales se ajustaron contra
  contenido conversacional realista tras el reporte de Kelly de que la
  v1.1.0 no encontraba ningún clip ni con el umbral en 5.0 (las curvas
  originales solo se habían validado con ejemplos caricaturescos: picos
  de audio de 6x y texto saturado de ganchos).
- Diversidad temporal (MMR) + deduplicación semántica por TF-IDF
  implementado a mano (sin sklearn): los clips se reparten por el video
  y no repiten la misma idea.
- Explicabilidad: cada clip guarda qué canales lo eligieron
  (``signal_breakdown``) para el "por qué este clip" de la UI.

Canales y pesos iniciales (informe INVESTIGACION_MOTOR_LOCAL_KELLYCLIPAI.md;
la calibración con videos reales es la Fase E):
- energía de audio 1.0, densidad/ritmo de habla 1.0, ganchos de texto 1.5,
  gancho inicial 1.0, completitud como multiplicador.
- El canal de eventos de audio (risa, aplauso, ovación, grito; peso
  1.5) lo aporta la Fase B (Tier 1, YAMNet ONNX): ``backend.audio_events``
  produce el ``event_timeline`` y este scorer lo incluye en la media
  ponderada solo cuando está presente.
- La Fase C (v1.2.0) añade dos canales visuales Tier 0 (0 MB):
  "visual" (cortes de escena + movimiento, peso 0.5) y "rostro"
  (presencia de rostro, peso 0.5), producidos por
  ``backend.visual_signals``. Un canal sin variación no entra en la
  media (misma idea que el audio sin audio utilizable, v1.1.1): un
  rostro siempre (o nunca) en cuadro no distingue momentos.
- La Fase C también detecta el PERFIL de contenido (talk / mixed /
  gameplay) por densidad de habla y actividad visual, y ajusta los
  pesos: talk sube texto (2.0) y baja visual (0.25); gameplay sube
  audio (1.5) y visual (1.0) y baja texto (1.0); mixed usa los pesos
  v1.1.1 sin cambios. Sin estadísticas visuales no hay evidencia
  visual y el perfil es "mixed": los fixtures de test sin timelines
  visuales mantienen exactamente el comportamiento v1.1.1.
- La Fase C añade además el canal "hablantes" (peso 0.75, Tier 2
  opcional): ``backend.diarization`` produce el ``speaker_timeline``
  (1 Hz, densidad de turnos de hablante) con sherpa-onnx bajo demanda
  (~33 MB); sin él, el canal no participa. Como los visuales, un
  timeline sin turnos no entra en la media (un monólogo no debe
  perder puntos por no tener cambios de hablante).
- Calibración v1.2.1: los canales opcionales (eventos, visual, rostro,
  hablantes) pasaron de componentes de la media a BONIFICACIÓN ACOTADA
  sobre la media de los 4 canales base. En el video real de Kelly (GTA 6
  en PS4, 320 s) la v1.2.0 daba 2 clips en vez de 6 porque el timeline
  de eventos llegaba en ceros (sin risa ni aplauso en todo el video) y
  entraba en la media con peso 1.5, y el rostro (bimodal: cara a cámara
  o pantalla) arrastraba las ventanas sin rostro: quitar solo el canal
  de eventos multiplicaba por 8 los candidatos sobre el umbral. Un canal
  opcional que no detecta nada no debe restar: solo suma cuando detecta
  algo (y solo si está activo: eventos con algún pico, visual con algún
  segundo notable, rostro con variación, hablantes con algún turno).
  Sin canales opcionales, la puntuación es idéntica a la v1.1.1.
  La misma calibración bajó el suelo de separación MMR de 45 s a 30 s:
  en un video de 5 min, dos momentos de 7.5 a 11–13 s de otro elegido
  quedaban suprimidos por la penalización (45 s de separación exigen
  huecos que un video corto no tiene).
- Fin de clip (v1.2.1): el fin se ajusta (snap) al fin de la última
  palabra hablada + una cola de ~0.4 s acotada al inicio de la palabra
  siguiente. Los timestamps por palabra de Whisper pueden solaparse y el
  fin de frase caía dentro de la palabra siguiente (en el video de Kelly
  el clip 1 terminaba dentro de "poco"); sin timestamps por palabra, el
  fin se extiende hasta el siguiente hueco de silencio.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .llm import (
    ResolvedLLM,
    _extract_json,
    _request_clip_content,
)
from .schemas import ClipProposal, ProjectConfig, TranscriptSegment, Word
from .visual_signals import VisualStats, combine_visual_timeline, extract_visual_timelines

logger = logging.getLogger(__name__)

# --- Pesos de los canales (ver docstring del módulo) ------------------------
_WEIGHT_AUDIO = 1.0
_WEIGHT_SPEECH = 1.0
_WEIGHT_TEXT = 1.5
_WEIGHT_OPENING = 1.0
_WEIGHT_EVENTS = 1.5  # Solo cuando la Fase B aporta event_timeline.
_WEIGHT_VISUAL = 0.5  # Fase C: cortes + movimiento (Tier 0).
_WEIGHT_FACE = 0.5  # Fase C: presencia de rostro (Tier 0).
_WEIGHT_SPEAKERS = 0.75  # Fase C: turnos de hablante (Tier 2, diarización).

# Pesos por perfil de contenido (Fase C). "mixed" es el perfil por
# defecto y usa EXACTAMENTE los pesos v1.1.1: sin estadísticas visuales
# no hay evidencia para otro perfil y el comportamiento no cambia.
_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "mixed": {
        "audio": _WEIGHT_AUDIO,
        "texto": _WEIGHT_TEXT,
        "ritmo": _WEIGHT_SPEECH,
        "gancho_inicial": _WEIGHT_OPENING,
        "eventos": _WEIGHT_EVENTS,
        "visual": _WEIGHT_VISUAL,
        "rostro": _WEIGHT_FACE,
        "hablantes": _WEIGHT_SPEAKERS,
    },
    # Charla a cámara: lo que importa es lo que se dice; lo visual
    # apenas varía y no debe pesar.
    "talk": {
        "audio": _WEIGHT_AUDIO,
        "texto": 2.0,
        "ritmo": _WEIGHT_SPEECH,
        "gancho_inicial": _WEIGHT_OPENING,
        "eventos": _WEIGHT_EVENTS,
        "visual": 0.25,
        "rostro": _WEIGHT_FACE,
        "hablantes": _WEIGHT_SPEAKERS,
    },
    # Gameplay / acción: hay poco que decir y mucho que ver y oír.
    "gameplay": {
        "audio": 1.5,
        "texto": 1.0,
        "ritmo": _WEIGHT_SPEECH,
        "gancho_inicial": _WEIGHT_OPENING,
        "eventos": _WEIGHT_EVENTS,
        "visual": 1.0,
        "rostro": _WEIGHT_FACE,
        "hablantes": _WEIGHT_SPEAKERS,
    },
}

# Umbrales de perfil (Fase C, documentados; la calibración fina es la
# Fase E). Habla densa de video conversacional: 120–180 palabras/min
# mientras se habla y ~85 % del video con habla; un gameplay con
# comentarios dispersos queda muy por debajo. Movimiento crudo medido en
# la VM (visual_signals.py): estático 0,0; testsrc2 3,6; objeto pequeño
# en movimiento rápido 7,9; zoom 14,8; paneo 90. Los umbrales son
# conservadores a propósito: ante la duda, "mixed" (pesos sin cambios).
_PROFILE_TALK_MIN_WPM = 100.0
_PROFILE_TALK_MIN_SPEECH_FRACTION = 0.5
_PROFILE_TALK_MAX_CUT_RATE = 2.0  # Cortes por minuto.
_PROFILE_TALK_MAX_MOTION_RAW = 5.0  # MAD medio en niveles de gris.
_PROFILE_GAMEPLAY_MAX_WPM = 60.0
_PROFILE_GAMEPLAY_MAX_SPEECH_FRACTION = 0.35
_PROFILE_GAMEPLAY_MIN_CUT_RATE = 6.0  # Cortes por minuto.
_PROFILE_GAMEPLAY_MIN_MOTION_RAW = 8.0  # MAD medio en niveles de gris.

# Activación de los canales opcionales: un canal sin señal no bonifica
# (v1.2.1; antes "no entra en la media"). Eventos: basta un segundo con
# un evento claro (puntuación > 0.5, el mismo umbral que el canal usa
# para "segundo notable"). Visual: basta un segundo notable (máximo >
# 0.1). Rostro: la presencia debe variar (desviación típica > 0.05).
# Hablantes: basta un turno en el video (el timeline vale 0.25 con un
# solo turno en ±10 s; un monólogo queda en 0 y no bonifica).
_EVENTS_MIN_PEAK = 0.5
_VISUAL_MIN_PEAK = 0.1
_FACE_MIN_STD = 0.05
_SPEAKER_MIN_PEAK = 0.1

# Bonificación de los canales opcionales (v1.2.1): cada canal activo suma
# valor × peso × factor, con un tope total. Con factor 1.0, un canal a
# plena señal suma como máximo su peso en puntos (eventos 1.5, visual
# 0.5, rostro 0.5, hablantes 0.75) y el tope de 1.5 puntos evita que
# varias señales coincidentes creen clips de la nada (el video aburrido
# con canales visuales queda en 3.1/10 como mucho, medido en la
# calibración v1.2.1).
_BONUS_FACTOR = 1.0
_BONUS_MAX_POINTS = 1.5

# Cola tras la última palabra al ajustar el fin del clip (v1.2.1): los
# timestamps de Whisper pueden cortar el último fonema; 0.4 s de cola,
# siempre acotada al inicio de la palabra siguiente, lo cubren. Sin
# timestamps por palabra, el fin se extiende hasta el siguiente hueco de
# silencio, como máximo 2 s.
_WORD_END_TAIL_SECONDS = 0.4
_SILENCE_FALLBACK_SECONDS = 2.0

# Separación MMR por defecto: los clips elegidos tienden a estar al menos
# a esta distancia entre sí (proporcional a la duración, con topes).
# Suelo 30 s (v1.2.1, antes 45 s): en un video de 5 min, 45 s de hueco
# mínimo suprimían momentos de 7.5 a 11–13 s de otro elegido; con 30 s
# el video de Kelly da 7 clips buenos y repartidos, y el test de
# supresión de duplicados cercanos (dos momentos a 20 s, umbral 8.0)
# sigue pasando.
_MMR_MIN_SECONDS = 30.0
_MMR_MAX_SECONDS = 90.0
_MMR_DURATION_DIVISOR = 15.0
# Similitud coseno TF-IDF por encima de la cual dos clips "dicen lo mismo".
_TFIDF_DUPLICATE_SIMILARITY = 0.55
# Rango de duración preferido cuando el proyecto pide "Cualquiera" (igual
# que la instrucción del prompt LLM: prefiere 15–45 s, evita < 12 s).
_DEFAULT_MIN_CLIP_SECONDS = 15.0
_DEFAULT_MAX_CLIP_SECONDS = 45.0
# Rangos de duración por elección del proyecto (espejo de
# pipeline._duration_bounds; "Cualquiera" usa el rango preferido 15–45 s
# en vez de (3, ∞) para no generar clips de 3 s ni de 10 minutos).
_DURATION_BOUNDS: dict[str, tuple[float, float]] = {
    "Cualquiera": (_DEFAULT_MIN_CLIP_SECONDS, _DEFAULT_MAX_CLIP_SECONDS),
    "<30s": (3.0, 29.9),
    "30–60s": (30.0, 60.0),
    "60–90s": (60.0, 90.0),
    "90s–3min": (90.0, 180.0),
}
_MAX_CANDIDATES = 60_000  # Guarda: videos larguísimos con frases densas.
_MAX_TITLE_CHARS = 60
_MAX_DESCRIPTION_CHARS = 300

# Palabras-gancho en español (lista curada): absolutos, exclusividad,
# urgencia, curiosidad y llamadas de atención. Solo afectan a la
# puntuación del canal de texto; nunca se inventan en títulos.
_HOOK_WORDS = frozenset(
    {
        "increíble",
        "impresionante",
        "brutal",
        "épico",
        "locura",
        "alucinante",
        "secreto",
        "gratis",
        "nunca",
        "siempre",
        "nadie",
        "todos",
        "primero",
        "último",
        "único",
        "única",
        "mejor",
        "peor",
        "verdad",
        "mentira",
        "truco",
        "fácil",
        "rápido",
        "urgente",
        "ahora",
        "mira",
        "imagina",
        "escucha",
        "atención",
        "cuidado",
        "ojo",
        "wow",
        "importante",
        "clave",
        "error",
        "peligro",
        "regalo",
        "sorpresa",
        "exclusivo",
        "prohibido",
        "oculto",
        "finalmente",
    }
)
# Ordinales de lista ("primero..., segundo..., tercero...").
_LIST_WORDS = frozenset({"primero", "segundo", "tercero", "cuarto", "quinto", "sexto"})
_QUESTION_STARTERS = frozenset(
    {"qué", "cómo", "por", "cuándo", "dónde", "quién", "cuál", "cuánto", "cuánta"}
)
# Stopwords en español para TF-IDF y palabras clave (lista corta y
# conservadora: solo palabras funcionales, nunca contenido).
_STOPWORDS = frozenset(
    {
        "de",
        "la",
        "el",
        "en",
        "y",
        "a",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "que",
        "es",
        "se",
        "su",
        "sus",
        "con",
        "por",
        "para",
        "del",
        "al",
        "lo",
        "como",
        "más",
        "pero",
        "este",
        "esta",
        "esto",
        "eso",
        "ese",
        "esa",
        "son",
        "fue",
        "han",
        "o",
        "si",
        "no",
        "sí",
        "ya",
        "muy",
        "mi",
        "mis",
        "tu",
        "tus",
        "te",
        "me",
        "le",
        "les",
        "nos",
        "os",
        "yo",
        "tú",
        "él",
        "ella",
        "nosotros",
        "ustedes",
        "ellos",
        "ellas",
        "aquí",
        "ahí",
        "allí",
        "cuando",
        "donde",
        "porque",
        "pues",
        "entonces",
        "también",
        "tampoco",
        "bien",
        "mal",
        "hay",
        "está",
        "están",
        "estoy",
        "ser",
        "estar",
        "tener",
        "haber",
        "hacer",
        "decir",
        "ir",
        "ver",
        "dar",
        "saber",
        "querer",
        "poder",
        "deber",
    }
)
_WORD_PATTERN = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)?", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\d")
_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
# ffmpeg: PCM 16 kHz mono para el timeline de audio 1 Hz. Se lee por
# bloques (streaming) para no cargar en RAM el audio de un video de horas.
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_CHUNK_SECONDS = 5


@dataclass(frozen=True)
class Phrase:
    """Una frase de la transcripción, con tiempos por palabra."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.words) if self.words else len(self.text.split())

    @property
    def is_question(self) -> bool:
        return "?" in self.text

    @property
    def has_exclamation(self) -> bool:
        return "!" in self.text


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'”’)]}").endswith(_SENTENCE_END)


def build_phrases(transcript: list[TranscriptSegment]) -> list[Phrase]:
    """Agrupa las palabras de la transcripción en frases.

    Una frase se cierra cuando una palabra termina en signo de puntuación
    de fin de oración, o al terminar el segmento. Los segmentos sin
    palabras se usan tal cual como frase (defensivo: el pipeline ya exige
    tiempos por palabra, pero la caché de transcripción podría no traerlos).
    """
    phrases: list[Phrase] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(word.text.strip() for word in current).strip()
        if text:
            phrases.append(
                Phrase(
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                    words=tuple(current),
                )
            )
        current.clear()

    for segment in sorted(transcript, key=lambda item: item.start):
        if not segment.words:
            flush()
            text = segment.text.strip()
            if text:
                phrases.append(Phrase(start=segment.start, end=segment.end, text=text))
            continue
        for word in sorted(segment.words, key=lambda item: item.start):
            if not word.text.strip():
                continue
            current.append(word)
            if _ends_sentence(word.text.strip()):
                flush()
        flush()
    flush()
    return [phrase for phrase in phrases if phrase.end > phrase.start]


def extract_audio_timeline(
    input_path: Path | str,
    ffmpeg_path: str,
    duration: float,
    *,
    sample_rate: int = _AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """RMS de audio por segundo (timeline 1 Hz) con ffmpeg + numpy.

    Decodifica a PCM s16le mono por stdout y lo lee por bloques de 5 s:
    la memoria es O(1) aunque el video dure horas. Si ffmpeg falla o el
    video no tiene audio, devuelve ceros (el motor sigue con el canal de
    texto) y registra un aviso; nunca lanza.
    """
    seconds = max(1, math.ceil(duration))
    rms = np.zeros(seconds, dtype=np.float64)
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
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as error:
        logger.warning("No se pudo extraer el audio para el motor de señales: %s", error)
        return rms
    assert process.stdout is not None
    chunk_bytes = sample_rate * 2 * _AUDIO_CHUNK_SECONDS
    leftover = np.empty(0, dtype=np.float32)
    second = 0
    saw_samples = False
    try:
        while second < seconds:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            saw_samples = True
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
            if leftover.size:
                samples = np.concatenate([leftover, samples])
            complete = samples.size // sample_rate
            if complete:
                usable = min(complete, seconds - second)
                block = samples[: usable * sample_rate].reshape(usable, sample_rate)
                rms[second : second + usable] = np.sqrt(np.mean(block * block, axis=1))
                second += usable
                samples = samples[usable * sample_rate :]
                # Si el audio supera la duración probada, se ignora el resto.
                if second >= seconds:
                    break
                # Segundos completos restantes en el bloque (más allá de
                # `usable` no puede haber: usable == complete aquí).
                leftover = samples
            else:
                leftover = samples
        if second < seconds and leftover.size:
            rms[second] = float(np.sqrt(np.mean(leftover * leftover)))
    except (OSError, ValueError) as error:
        logger.warning("La extracción de audio para señales falló a medias: %s", error)
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
        logger.warning(
            "ffmpeg no devolvió audio para el motor de señales (%s); "
            "se usa solo el canal de texto.",
            detail or "sin pista de audio",
        )
    return rms


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_PATTERN.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in _STOPWORDS and len(token) >= 2]


def _phrase_hook_points(phrase: Phrase) -> tuple[float, list[str]]:
    """Puntos de gancho de una frase + etiquetas legibles (para razon_viral).

    Pregunta 2.0, exclamación 1.0, cada palabra-gancho distinta 1.0 (tope
    3), números 1.0, ordinales de lista 0.5 (tope 1), inicio interrogativo
    0.5, palabra en MAYÚSCULAS (énfasis) 0.5.
    """
    text = phrase.text
    points = 0.0
    labels: list[str] = []
    if phrase.is_question:
        points += 2.0
        labels.append("pregunta")
    if phrase.has_exclamation:
        points += 1.0
        labels.append("exclamación")
    tokens = set(_tokenize(text))
    hooks = sorted(token for token in tokens if token in _HOOK_WORDS)
    if hooks:
        points += min(3.0, float(len(hooks)))
        labels.extend(f"gancho:{hook}" for hook in hooks[:2])
    if _NUMBER_PATTERN.search(text):
        points += 1.0
        labels.append("números")
    list_hits = tokens & _LIST_WORDS
    if list_hits:
        points += min(1.0, 0.5 * len(list_hits))
        labels.append("lista")
    first = _tokenize(text)[:1]
    if first and first[0] in _QUESTION_STARTERS and not phrase.is_question:
        points += 0.5
        labels.append("pregunta")
    if any(len(token) >= 4 and token.isupper() for token in _WORD_PATTERN.findall(text)):
        points += 0.5
        labels.append("énfasis")
    return points, labels


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def detect_content_profile(
    phrases: list[Phrase],
    duration: float,
    *,
    cut_rate_per_minute: float | None = None,
    mean_motion_raw: float | None = None,
) -> str:
    """Clasifica el video: "talk", "mixed" o "gameplay" (determinista).

    Por densidad de habla (palabras/minuto sobre el video completo y
    fracción de segundos con habla) y actividad visual (tasa de cortes
    y movimiento medio crudo, de ``visual_signals.VisualStats``).
    - talk: habla densa + imagen calma (cabeza parlante).
    - gameplay: habla dispersa + imagen activa (cortes rápidos o mucho
      movimiento).
    - mixed: todo lo demás, y SIEMPRE que no haya estadísticas visuales
      (sin evidencia visual no se toca ningún peso: los fixtures de test
      y los videos sin timelines visuales mantienen el comportamiento
      v1.1.1). Un solo dato visual ausente no bloquea la clasificación:
      se usa el que haya; para "gameplay" basta con que UNO de los dos
      marque actividad alta.
    """
    if duration <= 0 or not phrases:
        return "mixed"
    if cut_rate_per_minute is None and mean_motion_raw is None:
        return "mixed"
    total_words = sum(phrase.word_count for phrase in phrases)
    speech_seconds = sum(max(0.1, phrase.end - phrase.start) for phrase in phrases)
    wpm = total_words / duration * 60.0
    speech_fraction = min(1.0, speech_seconds / duration)
    calm = (cut_rate_per_minute is None or cut_rate_per_minute <= _PROFILE_TALK_MAX_CUT_RATE) and (
        mean_motion_raw is None or mean_motion_raw <= _PROFILE_TALK_MAX_MOTION_RAW
    )
    active = (
        cut_rate_per_minute is not None and cut_rate_per_minute >= _PROFILE_GAMEPLAY_MIN_CUT_RATE
    ) or (mean_motion_raw is not None and mean_motion_raw >= _PROFILE_GAMEPLAY_MIN_MOTION_RAW)
    if (
        wpm >= _PROFILE_TALK_MIN_WPM
        and speech_fraction >= _PROFILE_TALK_MIN_SPEECH_FRACTION
        and calm
    ):
        return "talk"
    if (
        wpm <= _PROFILE_GAMEPLAY_MAX_WPM
        and speech_fraction <= _PROFILE_GAMEPLAY_MAX_SPEECH_FRACTION
        and active
    ):
        return "gameplay"
    return "mixed"


class MomentScorer:
    """Canales de señal precalculados + puntuación de ventanas.

    Las sumas prefijas hacen que puntuar una ventana sea O(1) (salvo el
    pico de energía, un máximo numpy sobre ~30–180 valores). ``event_timeline``
    es el hook de la Fase B: por segundo, 0–1 de eventos de audio (risa,
    aplauso, grito); ausente en Tier 0, en cuyo caso el canal no bonifica.
    ``visual_timeline`` (Fase C) es el timeline combinado de cortes +
    movimiento; si en vez de él llegan los componentes
    (``scene_cut_timeline`` / ``motion_timeline``), se combinan aquí.
    ``face_timeline`` (Fase C) es la presencia de rostro 0–1 por segundo.
    ``profile`` elige los pesos (ver ``_PROFILE_WEIGHTS``); un perfil
    desconocido cae a "mixed".

    La puntuación es la media ponderada de los 4 canales base (audio,
    texto, ritmo, gancho inicial) por la completitud, MÁS una
    bonificación acotada de los canales opcionales activos (eventos,
    visual, rostro, hablantes; v1.2.1): un canal opcional que no detecta
    nada no resta, solo suma cuando detecta algo.
    """

    def __init__(
        self,
        phrases: list[Phrase],
        audio_rms: np.ndarray,
        duration: float,
        event_timeline: np.ndarray | None = None,
        visual_timeline: np.ndarray | None = None,
        face_timeline: np.ndarray | None = None,
        scene_cut_timeline: np.ndarray | None = None,
        motion_timeline: np.ndarray | None = None,
        speaker_timeline: np.ndarray | None = None,
        profile: str = "mixed",
    ) -> None:
        self.phrases = phrases
        self.duration = duration
        self.audio_rms = audio_rms
        self.event_timeline = event_timeline
        self.speaker_timeline = speaker_timeline
        self.scene_cut_timeline = scene_cut_timeline
        self.motion_timeline = motion_timeline
        # El timeline visual combinado: si llega ya calculado (el
        # pipeline lo combina), se usa tal cual; si solo llegan los
        # componentes, se combinan aquí (máximo por segundo).
        if visual_timeline is not None:
            self.visual_timeline = visual_timeline
        else:
            self.visual_timeline = combine_visual_timeline(scene_cut_timeline, motion_timeline)
        self.face_timeline = face_timeline
        self.profile = profile if profile in _PROFILE_WEIGHTS else "mixed"
        self._weights = _PROFILE_WEIGHTS[self.profile]
        # Un canal opcional sin señal no bonifica (v1.2.1; antes "no entra
        # en la media"). Eventos: basta un segundo con un evento claro;
        # sin esta guarda, un video sin risa ni aplauso perdería ~1.4
        # puntos en todas sus ventanas por un canal en cero con peso
        # 1.5 (el video de Kelly, v1.2.0: 2 clips en vez de 6). Visual:
        # basta un segundo notable. Rostro: la presencia debe variar;
        # un rostro siempre (o nunca) en cuadro no distingue momentos.
        self._events_active = (
            event_timeline is not None
            and event_timeline.size > 0
            and float(np.max(event_timeline)) > _EVENTS_MIN_PEAK
        )
        self._visual_active = (
            self.visual_timeline is not None
            and self.visual_timeline.size > 0
            and float(np.max(self.visual_timeline)) > _VISUAL_MIN_PEAK
        )
        self._face_active = (
            face_timeline is not None
            and face_timeline.size > 0
            and float(np.std(face_timeline)) > _FACE_MIN_STD
        )
        self._speaker_active = (
            speaker_timeline is not None
            and speaker_timeline.size > 0
            and float(np.max(speaker_timeline)) > _SPEAKER_MIN_PEAK
        )
        self._face_mean = (
            float(np.mean(face_timeline))
            if face_timeline is not None and face_timeline.size
            else 0.0
        )
        # Índice de palabras ordenado por inicio, para ajustar el fin del
        # clip a una frontera de palabra (v1.2.1, `snap_end`): los
        # timestamps de Whisper pueden solaparse, así que el fin de frase
        # puede caer dentro de la palabra siguiente. Se guardan los
        # inicios aparte para buscar por bisección.
        self._words: list[tuple[float, float]] = sorted(
            (word.start, word.end)
            for phrase in phrases
            for word in phrase.words
            if word.end > word.start
        )
        self._word_starts = [start for start, _ in self._words]
        # Audio utilizable: si el timeline es todo ceros (ffmpeg falló o
        # el video no tiene pista de audio), el canal de audio NO entra
        # en la media ponderada (un 0 arrastraría la puntuación) y no se
        # recortan bordes por silencio (con RMS en cero todo parecería
        # silencio). El desglose sigue mostrando audio 0.0.
        self._has_audio = bool(audio_rms.size and float(np.max(audio_rms)) > 1e-3)
        # Z-score robusto del RMS por segundo (mediana + MAD): cuánto
        # destaca cada segundo respecto a la línea base DEL VIDEO.
        median = float(np.median(audio_rms)) if audio_rms.size else 0.0
        mad = float(np.median(np.abs(audio_rms - median))) if audio_rms.size else 0.0
        sigma = 1.4826 * mad
        if sigma < 1e-6:
            sigma = float(np.std(audio_rms)) if audio_rms.size else 0.0
        if sigma < 1e-6:
            sigma = 1e-6
        self.audio_z = (audio_rms - median) / sigma
        self._hot_prefix = np.concatenate(
            [[0.0], np.cumsum((self.audio_z > 1.0).astype(np.float64))]
        )
        # Silencio para recortar bordes: segundos con RMS por debajo del
        # 25 % de la mediana (con suelo absoluto 0.005).
        self._silence_threshold = max(0.005, 0.25 * median)
        # Puntos de gancho y palabras por frase, con sumas prefijas.
        hook_points = []
        hook_labels: list[list[str]] = []
        for phrase in phrases:
            points, labels = _phrase_hook_points(phrase)
            hook_points.append(points)
            hook_labels.append(labels)
        self._hook_points = hook_points
        self._hook_labels = hook_labels
        self._hook_prefix = np.concatenate(
            [[0.0], np.cumsum(np.asarray(hook_points, dtype=np.float64))]
        )
        word_counts = np.asarray([phrase.word_count for phrase in phrases], dtype=np.float64)
        self._words_prefix = np.concatenate([[0.0], np.cumsum(word_counts)])
        # Línea base de ritmo: palabras por minuto del video completo.
        speech_seconds = sum(max(0.1, phrase.end - phrase.start) for phrase in phrases)
        total_words = float(word_counts.sum()) if word_counts.size else 0.0
        self._baseline_wpm = total_words / speech_seconds * 60.0 if speech_seconds > 0 else 0.0

    # -- canales -----------------------------------------------------------
    def _audio_channel(self, start: float, end: float) -> tuple[float, float]:
        """(puntuación 0–1, pico z). Pico + fracción de segundos calientes."""
        first = max(0, math.floor(start))
        last = min(self.audio_z.size, math.ceil(end))
        if last <= first:
            return 0.0, 0.0
        peak_z = float(np.max(self.audio_z[first:last]))
        # v1.1.1: divisor 2.0 (antes 3.0): un pico realista (z ~2, un
        # énfasis claro) debe puntuar alto, no a la mitad.
        peak_component = _clip01((peak_z - 0.5) / 2.0)
        hot_fraction = float((self._hot_prefix[last] - self._hot_prefix[first]) / (last - first))
        return 0.6 * peak_component + 0.4 * hot_fraction, peak_z

    def _speech_channel(
        self, first_idx: int, last_idx: int, start: float, end: float
    ) -> tuple[float, dict[str, float]]:
        """Densidad y ritmo de habla 0–1 + detalles para las etiquetas."""
        if self._baseline_wpm <= 0:
            return 0.0, {}
        words = float(self._words_prefix[last_idx + 1] - self._words_prefix[first_idx])
        # v1.1.1: el WPM se mide sobre el tiempo DE HABLA (suma de las
        # duraciones de las frases), no sobre el lapso de pared: en
        # ventanas con pausas largas (gameplay, comentarios dispersos),
        # el lapso de pared deflactaba el WPM y la rama "lenta" premiaba
        # el silencio como si fuera énfasis. La línea base ya se calcula
        # así (_baseline_wpm), de modo que el ratio compara hablar con
        # hablar.
        speech_span = sum(
            max(0.1, phrase.end - phrase.start) for phrase in self.phrases[first_idx : last_idx + 1]
        )
        wpm = words / max(0.1, speech_span) * 60.0
        ratio = wpm / self._baseline_wpm
        # v1.1.1: divisor 0.35 (antes 0.5): una subida realista del ritmo
        # (+25–35 %) debe puntuar alto, no a la mitad.
        fast = _clip01((ratio - 1.0) / 0.35)
        # Habla lenta y pausada también es señal (énfasis), pero más débil.
        slow = _clip01((0.8 - ratio) / 0.4) * 0.7
        rhythm = 0.0
        count = last_idx - first_idx + 1
        if count >= 4:
            half = first_idx + count // 2
            words_first = float(self._words_prefix[half] - self._words_prefix[first_idx])
            words_second = float(self._words_prefix[last_idx + 1] - self._words_prefix[half])
            span_first = sum(
                max(0.1, phrase.end - phrase.start) for phrase in self.phrases[first_idx:half]
            )
            span_second = sum(
                max(0.1, phrase.end - phrase.start) for phrase in self.phrases[half : last_idx + 1]
            )
            wpm_first = words_first / max(0.1, span_first) * 60.0
            wpm_second = words_second / max(0.1, span_second) * 60.0
            # v1.1.1: divisor 0.5 (antes 0.8).
            rhythm = _clip01(abs(wpm_first - wpm_second) / self._baseline_wpm / 0.5)
        score = 0.6 * max(fast, slow) + 0.4 * rhythm
        return score, {"ratio": ratio, "fast": fast, "slow": slow, "rhythm": rhythm}

    def _text_channel(self, first_idx: int, last_idx: int) -> tuple[float, list[str]]:
        """Ganchos de texto 0–1 + etiquetas (pregunta→respuesta incluida)."""
        points = float(self._hook_prefix[last_idx + 1] - self._hook_prefix[first_idx])
        # v1.1.1: divisor 2.0 (antes 3.0): 2 puntos de gancho (p. ej. una
        # pregunta, o dos palabras-gancho) ya son señal clara en habla
        # real; con 3.0 apenas llegaban a la mitad.
        score = 1.0 - math.exp(-points / 2.0)
        labels: list[str] = []
        for idx in range(first_idx, last_idx + 1):
            labels.extend(self._hook_labels[idx])
        # Estructura pregunta→respuesta: la ventana empieza con una
        # pregunta y hay frases después que la responden.
        if self.phrases[first_idx].is_question and last_idx - first_idx >= 2:
            score = min(1.0, score + 0.1)
            labels.append("pregunta-respuesta")
        return score, labels

    def _opening_channel(self, first_idx: int, last_idx: int, start: float) -> float:
        """Fuerza de los primeros 3 s (la retención depende del gancho)."""
        opening_end = start + 3.0
        points = 0.0
        for idx in range(first_idx, last_idx + 1):
            if self.phrases[idx].start >= opening_end:
                break
            points += self._hook_points[idx]
        # v1.1.1: divisor 1.5 (antes 2.0) y umbral de audio 0.25 (antes
        # 0.5): los primeros 3 s rara vez acumulan muchos puntos, pero
        # un gancho ahí es justo lo que retiene.
        text_open = 1.0 - math.exp(-points / 1.5)
        if self.phrases[first_idx].is_question:
            text_open = max(text_open, 0.85)
        first = max(0, math.floor(start))
        last = min(self.audio_z.size, math.ceil(opening_end))
        if last > first:
            mean_z = float(np.mean(self.audio_z[first:last]))
        else:
            mean_z = 0.0
        audio_open = _clip01((mean_z - 0.25) / 1.5)
        # El gancho es fuerte si las palabras enganchan O la energía
        # sube: basta una de las dos (max, no media).
        return max(text_open, audio_open)

    def _events_channel(self, start: float, end: float) -> float:
        """Canal de eventos de audio (Fase B). En Tier 0 no se llama."""
        timeline = self.event_timeline
        if timeline is None:
            return 0.0
        first = max(0, math.floor(start))
        last = min(timeline.size, math.ceil(end))
        if last <= first:
            return 0.0
        window = timeline[first:last]
        peak = float(np.max(window))
        frac = float(np.mean(window > 0.5))
        return 0.5 * _clip01((peak - 0.3) / 0.5) + 0.5 * frac

    def _speaker_channel(self, start: float, end: float) -> float:
        """Canal de turnos de hablante (Fase C, diarización). Misma forma
        que eventos: pico + fracción de segundos con actividad alta.
        Un debate vivo puntúa alto; un monólogo, cero."""
        timeline = self.speaker_timeline
        if timeline is None:
            return 0.0
        first = max(0, math.floor(start))
        last = min(timeline.size, math.ceil(end))
        if last <= first:
            return 0.0
        window = timeline[first:last]
        peak = float(np.max(window))
        frac = float(np.mean(window > 0.5))
        return 0.5 * _clip01((peak - 0.3) / 0.5) + 0.5 * frac

    def _visual_channel(self, start: float, end: float) -> float:
        """Canal visual (Fase C): cortes + movimiento, misma forma que eventos.

        Pico + fracción de segundos notables: un corte aislado en una
        ventana larga puntúa menos que una racha de cortes o de acción.
        """
        timeline = self.visual_timeline
        if timeline is None:
            return 0.0
        first = max(0, math.floor(start))
        last = min(timeline.size, math.ceil(end))
        if last <= first:
            return 0.0
        window = timeline[first:last]
        peak = float(np.max(window))
        frac = float(np.mean(window > 0.5))
        return 0.5 * _clip01((peak - 0.3) / 0.5) + 0.5 * frac

    def _face_channel(self, start: float, end: float) -> float:
        """Canal de rostro (Fase C): presencia RELATIVA al video.

        Un rostro en cuadro puntúa cuando aparece AQUÍ más que en el
        resto del video: en una cabeza parlante (rostro siempre
        presente) el canal vale ~0 en todas partes y no infla nada (y
        la guarda de variación lo excluye de la media).
        """
        timeline = self.face_timeline
        if timeline is None:
            return 0.0
        first = max(0, math.floor(start))
        last = min(timeline.size, math.ceil(end))
        if last <= first:
            return 0.0
        window_mean = float(np.mean(timeline[first:last]))
        return _clip01((window_mean - self._face_mean) / 0.5)

    def _visual_component_peaks(self, start: float, end: float) -> tuple[float, float]:
        """(Pico de cortes, pico de movimiento) en la ventana (0.0 si ausentes).

        Solo para las etiquetas: el canal "visual" no distingue qué lo
        disparó, pero las etiquetas sí ("Cambio de escena" / "Momento
        de mucha acción").
        """
        scene_peak = 0.0
        if self.scene_cut_timeline is not None:
            first = max(0, math.floor(start))
            last = min(self.scene_cut_timeline.size, math.ceil(end))
            if last > first:
                scene_peak = float(np.max(self.scene_cut_timeline[first:last]))
        motion_peak = 0.0
        if self.motion_timeline is not None:
            first = max(0, math.floor(start))
            last = min(self.motion_timeline.size, math.ceil(end))
            if last > first:
                motion_peak = float(np.max(self.motion_timeline[first:last]))
        return scene_peak, motion_peak

    # -- bordes ------------------------------------------------------------
    def trim_edges(self, start: float, end: float) -> tuple[float, float, float]:
        """Recorta silencios en los bordes (máx. 2 s por lado).

        Devuelve (inicio, fin, multiplicador de completitud): 1.0 si el
        recorte es pequeño, 0.85 si movió un borde más de 1 s (la frontera
        de frase caía en una pausa). Sin audio utilizable no se recorta
        nada (con RMS en cero todo parecería silencio) y el multiplicador
        es 1.0.
        """
        if not self._has_audio:
            return start, end, 1.0
        trimmed = 0.0
        new_start = start
        for _ in range(2):
            second = math.floor(new_start)
            if (
                second < 0
                or second >= self.audio_rms.size
                or self.audio_rms[second] >= self._silence_threshold
                or new_start + 1.0 >= end - 3.0
            ):
                break
            new_start = min(end - 3.0, second + 1.0)
            trimmed += 1.0
        new_end = end
        for _ in range(2):
            second = math.ceil(new_end) - 1
            if (
                second < 0
                or second >= self.audio_rms.size
                or self.audio_rms[second] >= self._silence_threshold
                or new_end - 1.0 <= new_start + 3.0
            ):
                break
            new_end = max(new_start + 3.0, float(second))
            trimmed += 1.0
        multiplier = 0.85 if trimmed > 1.0 else 1.0
        return new_start, new_end, multiplier

    def snap_end(self, end: float, *, max_end: float | None = None) -> float:
        """Ajusta el fin del clip para no cortar una palabra por la mitad.

        Los timestamps por palabra de Whisper pueden solaparse: el fin de
        frase puede caer dentro de la palabra siguiente (en el video de
        Kelly el clip 1 terminaba en 33.16, dentro de "poco"). Si ``end``
        cae dentro de una palabra, se extiende hasta el fin de esa
        palabra (repitiendo si los solapes encadenan varias); después se
        añade una cola de hasta 0.4 s, acotada al inicio de la palabra
        siguiente para no entrar en ella (la cola cubre el último fonema
        cuando Whisper marca el fin de palabra un poco antes). ``max_end``
        acota el resultado (duración máxima pedida y clip siguiente):
        nunca se supera, aunque eso deje una palabra incompleta en el
        caso límite de un clip ya al máximo de duración.
        Sin timestamps por palabra, el fin se extiende hasta el siguiente
        hueco de silencio (máx. 2 s); sin audio utilizable no hay forma
        de saber dónde hay silencio y el fin se deja como está. Nunca
        retrocede ni supera la duración del video.
        """
        end = min(end, self.duration)
        limit = self.duration if max_end is None else min(max_end, self.duration)
        if end >= limit:
            return min(end, limit)
        if self._words:
            starts = self._word_starts
            candidate = end
            # Extender mientras el fin caiga dentro de una palabra.
            for _ in range(5):
                idx = bisect.bisect_right(starts, candidate) - 1
                containing_end = 0.0
                j = idx
                while j >= 0 and starts[j] > candidate - 5.0:
                    word_start, word_end = self._words[j]
                    if word_start <= candidate < word_end:
                        containing_end = max(containing_end, word_end)
                    j -= 1
                if containing_end <= candidate:
                    break
                candidate = containing_end
            # Cola hasta 0.4 s, acotada al inicio de la palabra
            # siguiente (nunca dentro de ella).
            tail = _WORD_END_TAIL_SECONDS
            next_idx = bisect.bisect_left(starts, candidate)
            if next_idx < len(starts):
                tail = min(tail, max(0.0, starts[next_idx] - candidate))
            return min(candidate + tail, limit)
        if not self._has_audio:
            return min(end, limit)
        second = math.floor(end) + 1
        last = min(self.audio_rms.size, math.ceil(end + _SILENCE_FALLBACK_SECONDS))
        while second < last:
            if self.audio_rms[second] < self._silence_threshold:
                return min(float(second), limit)
            second += 1
        return min(end, limit)

    # -- puntuación --------------------------------------------------------
    def score_window(
        self, first_idx: int, last_idx: int
    ) -> tuple[float, dict[str, float], list[str]]:
        """Puntuación absoluta 0–10 de la ventana + desglose + etiquetas."""
        start = self.phrases[first_idx].start
        end = self.phrases[last_idx].end
        audio, peak_z = self._audio_channel(start, end)
        speech, speech_detail = self._speech_channel(first_idx, last_idx, start, end)
        text, text_labels = self._text_channel(first_idx, last_idx)
        opening = self._opening_channel(first_idx, last_idx, start)
        weights = self._weights
        channels: list[tuple[str, float, float]] = [
            ("audio", audio, weights["audio"]),
            ("texto", text, weights["texto"]),
            ("ritmo", speech, weights["ritmo"]),
            ("gancho_inicial", opening, weights["gancho_inicial"]),
        ]
        if self.event_timeline is not None:
            events_score = self._events_channel(start, end)
            channels.append(("eventos", events_score, weights["eventos"]))
        else:
            events_score = 0.0
        if self.visual_timeline is not None:
            visual_score = self._visual_channel(start, end)
            channels.append(("visual", visual_score, weights["visual"]))
        else:
            visual_score = 0.0
        if self.face_timeline is not None:
            face_score = self._face_channel(start, end)
            channels.append(("rostro", face_score, weights["rostro"]))
        else:
            face_score = 0.0
        if self.speaker_timeline is not None:
            speaker_score = self._speaker_channel(start, end)
            channels.append(("hablantes", speaker_score, weights["hablantes"]))
        else:
            speaker_score = 0.0
        # Media base: solo los 4 canales principales. Sin audio
        # utilizable, el canal de audio (0.0) no entra en la media: un
        # dato ausente no debe arrastrar la puntuación (v1.1.1).
        scoring = [
            (name, value, weight)
            for name, value, weight in channels[:4]
            if self._has_audio or name != "audio"
        ]
        total_weight = sum(weight for _, _, weight in scoring)
        weighted = sum(value * weight for _, value, weight in scoring) / total_weight
        _, _, completeness = self.trim_edges(start, end)
        score = weighted * 10.0 * completeness
        # Bonificación de los canales opcionales (v1.2.1): cada canal
        # ACTIVO suma valor × peso × factor, con tope total. Un canal
        # opcional que no detecta nada no resta: en la v1.2.0 el canal
        # de eventos en cero (sin risa ni aplauso en el video) entraba
        # en la media con peso 1.5 y costaba ~1.4 puntos por ventana.
        # El desglose sigue mostrando todos los valores (ver breakdown
        # abajo), activos o no.
        bonus = 0.0
        for name, value, is_active in (
            ("eventos", events_score, self._events_active),
            ("visual", visual_score, self._visual_active),
            ("rostro", face_score, self._face_active),
            ("hablantes", speaker_score, self._speaker_active),
        ):
            if is_active:
                bonus += value * weights[name] * _BONUS_FACTOR
        score = min(10.0, max(0.0, score + min(_BONUS_MAX_POINTS, bonus)))
        breakdown = {name: round(value * 10.0, 1) for name, value, _ in channels}
        breakdown["completitud"] = round(completeness, 2)
        scene_peak, motion_peak = self._visual_component_peaks(start, end)
        labels = self._build_labels(
            text_labels,
            peak_z,
            speech_detail,
            opening,
            first_idx,
            events_score,
            visual_score,
            face_score,
            scene_peak,
            motion_peak,
            speaker_score,
        )
        return score, breakdown, labels

    def _build_labels(
        self,
        text_labels: list[str],
        peak_z: float,
        speech_detail: dict[str, float],
        opening: float,
        first_idx: int,
        events_score: float = 0.0,
        visual_score: float = 0.0,
        face_score: float = 0.0,
        scene_peak: float = 0.0,
        motion_peak: float = 0.0,
        speaker_score: float = 0.0,
    ) -> list[str]:
        """Etiquetas legibles para razon_viral, en orden de prioridad."""
        labels: list[str] = []
        if self.phrases[first_idx].is_question:
            labels.append("Empieza con una pregunta")
        elif "pregunta" in text_labels:
            labels.append("Pregunta directa")
        hooks = sorted(
            {label.split(":", 1)[1] for label in text_labels if label.startswith("gancho:")}
        )
        if hooks:
            quoted = ", ".join(f"«{hook}»" for hook in hooks[:2])
            labels.append(f"Palabras gancho: {quoted}")
        if events_score >= 0.5:
            # Antes que "Pico de energía": una reacción detectada (risa,
            # aplauso) es más específica que un pico de volumen, y
            # razon_viral solo muestra las 3 primeras etiquetas.
            labels.append("Reacción del público (risa o aplauso)")
        if self.scene_cut_timeline is not None or self.motion_timeline is not None:
            # Componentes disponibles: etiqueta precisa de lo que disparó
            # el canal visual (pueden ser las dos a la vez: un corte
            # suele coincidir con un pico de movimiento).
            if scene_peak >= 0.5:
                labels.append("Cambio de escena")
            if motion_peak >= 0.5:
                labels.append("Momento de mucha acción")
        elif visual_score >= 0.5:
            # Solo llegó el timeline combinado: sin saber qué lo disparó.
            labels.append("Cambio visual brusco")
        if face_score >= 0.5:
            labels.append("Rostro en cuadro")
        if speaker_score >= 0.5:
            labels.append("Intercambio animado entre hablantes")
        if peak_z >= 2.0:
            labels.append("Pico de energía")
        if speech_detail.get("fast", 0.0) >= 0.6:
            labels.append("Ritmo de habla acelerado")
        elif speech_detail.get("slow", 0.0) >= 0.5:
            labels.append("Ritmo pausado con énfasis")
        if "números" in text_labels:
            labels.append("Datos y cifras")
        if opening >= 0.7 and "Empieza con una pregunta" not in labels:
            labels.append("Gancho inicial fuerte")
        if "exclamación" in text_labels:
            labels.append("Exclamación")
        if speech_detail.get("rhythm", 0.0) >= 0.6:
            labels.append("Cambio brusco de ritmo")
        if "lista" in text_labels:
            labels.append("Estructura de lista")
        return labels[:4]


# --- selección --------------------------------------------------------------


@dataclass
class _ScoredCandidate:
    score: float
    first_idx: int
    last_idx: int
    start: float
    end: float
    breakdown: dict[str, float]
    labels: list[str]
    text: str


def _candidate_bounds(choice: str, duration: float) -> tuple[float, float] | None:
    """Rango de duración de los candidatos para la elección del proyecto.

    En videos más cortos que el mínimo preferido, el rango se encoge al
    video (un video de 10 s con "Cualquiera" da candidatos de 10 s).
    None si el video es demasiado corto para cualquier clip.
    """
    minimum, maximum = _DURATION_BOUNDS.get(
        choice, (_DEFAULT_MIN_CLIP_SECONDS, _DEFAULT_MAX_CLIP_SECONDS)
    )
    maximum = min(maximum, duration)
    minimum = min(minimum, maximum)
    if maximum < 3.0:
        return None
    return minimum, maximum


def _default_separation(duration: float) -> float:
    """Separación MMR por defecto: proporcional a la duración, con topes."""
    return min(
        _MMR_MAX_SECONDS,
        max(_MMR_MIN_SECONDS, duration / _MMR_DURATION_DIVISOR),
    )


def _tfidf_vectors(texts: list[str], idf_corpus: list[str]) -> list[dict[str, float]]:
    """Vectores TF-IDF (implementación a mano, sin sklearn).

    TF sobre el texto del candidato; IDF sobre TODAS las frases del video
    (no solo sobre los candidatos): así las palabras del momento — raras
    en el video — pesan mucho y la similitud refleja solape de contenido
    distintivo, no de tema general. Si el IDF se calculase solo sobre los
    candidatos que superaron el umbral (ventanas solapadas sobre los
    mismos momentos), las palabras compartidas parecerían comunes y la
    deduplicación no detectaría repeticiones.
    """
    tokenized = [_content_tokens(text) for text in texts]
    corpus_tokenized = [_content_tokens(text) for text in idf_corpus]
    doc_count = len(corpus_tokenized)
    doc_freq: dict[str, int] = {}
    for tokens in corpus_tokenized:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    idf = {token: math.log((doc_count + 1) / (freq + 1)) + 1.0 for token, freq in doc_freq.items()}
    default_idf = math.log(doc_count + 1) + 1.0
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        total = max(1, len(tokens))
        vector = {
            token: (count / total) * idf.get(token, default_idf) for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm > 0:
            vector = {token: value / norm for token, value in vector.items()}
        vectors.append(vector)
    return vectors


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(token, 0.0) for token, value in left.items()))


def _select_diverse(
    candidates: list[_ScoredCandidate],
    *,
    threshold: float,
    separation_seconds: float,
    max_clips: int,
    idf_corpus: list[str],
    enable_semantic: bool = False,
    semantic_threshold: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[_ScoredCandidate]:
    """Elige hasta max_clips por puntuación, con diversidad MMR + TF-IDF.

    - Solapes: un candidato que se solapa con uno elegido se descarta
      (la validación del pipeline también los rechaza).
    - MMR: la puntuación efectiva se penaliza según la cercanía al clip
      elegido más próximo (factor 0.5 si es adyacente, 1.0 a >= X s); si
      la penalizada cae bajo el umbral, el candidato se descarta. Solo un
      momento excepcional puede quedar junto a otro elegido.
    - TF-IDF: si el texto es casi idéntico al de un clip elegido
      (similitud > 0.55), se queda el de mayor puntuación.
    - Semántico (Fase D, solo si ``enable_semantic``): además del
      TF-IDF, se descarta un candidato si su embedding MiniLM es casi
      idéntico al de un clip elegido (coseno > umbral, 0.85 por
      defecto): caza paráfrasis (misma idea con otras palabras) que el
      TF-IDF no ve. Filtro ADICIONAL, nunca reemplazo: si los
      embeddings no están disponibles, la selección sigue solo con
      TF-IDF. La cancelación del usuario (``SemanticCancelled``) NO se
      traga: se propaga para que el pipeline la traduzca.
    """
    ordered = sorted(
        (cand for cand in candidates if cand.score >= threshold),
        key=lambda cand: (-cand.score, cand.start),
    )
    if not ordered:
        return []
    vectors = _tfidf_vectors([cand.text for cand in ordered], idf_corpus)
    embeddings: np.ndarray | None = None
    if enable_semantic:
        try:
            from . import semantic
        except ImportError:
            semantic = None  # type: ignore[assignment]
        if semantic is not None:
            if semantic_threshold is None:
                semantic_threshold = semantic.DEFAULT_PARAPHRASE_THRESHOLD
            embeddings = semantic.embed_texts(
                [cand.text for cand in ordered], should_cancel=should_cancel
            )
            if embeddings is not None and embeddings.shape[0] != len(ordered):
                logger.warning(
                    "Los embeddings semánticos no cubren todos los candidatos; "
                    "se sigue solo con TF-IDF."
                )
                embeddings = None
    selected: list[_ScoredCandidate] = []
    selected_vectors: list[dict[str, float]] = []
    selected_embeddings: list[np.ndarray] = []
    for index, candidate in enumerate(ordered):
        if len(selected) >= max_clips:
            break
        overlaps = any(
            candidate.start < chosen.end and chosen.start < candidate.end for chosen in selected
        )
        if overlaps:
            continue
        if selected:
            gap = min(
                max(
                    candidate.start - chosen.end,
                    chosen.start - candidate.end,
                    0.0,
                )
                for chosen in selected
            )
            factor = 0.5 + 0.5 * min(1.0, gap / separation_seconds)
            if candidate.score * factor < threshold:
                continue
        vector = vectors[index]
        if any(
            _cosine_similarity(vector, chosen) > _TFIDF_DUPLICATE_SIMILARITY
            for chosen in selected_vectors
        ):
            continue
        if (
            embeddings is not None
            and selected_embeddings
            and semantic_threshold is not None
            and any(
                float(embeddings[index] @ chosen) > semantic_threshold
                for chosen in selected_embeddings
            )
        ):
            continue
        selected.append(candidate)
        selected_vectors.append(vector)
        if embeddings is not None:
            selected_embeddings.append(embeddings[index])
    return sorted(selected, key=lambda cand: cand.start)


# --- construcción de ClipProposal --------------------------------------------


def _trim_text(text: str, max_chars: int) -> str:
    """Recorta a max_chars en frontera de palabra, con '…' si recorta."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0]
    return (cut or text[: max_chars - 1]).rstrip(".,;:") + "…"


def _build_title(scorer: MomentScorer, first_idx: int, last_idx: int) -> str:
    """Título por plantilla: la frase más contundente del clip.

    Si el clip empieza con una pregunta, la pregunta es el título.
    Recortado a ~60 caracteres.
    """
    phrases = scorer.phrases
    if phrases[first_idx].is_question:
        best = phrases[first_idx]
    else:
        best_idx = max(
            range(first_idx, last_idx + 1),
            key=lambda idx: scorer._hook_points[idx],
        )
        best = phrases[best_idx]
    title = _trim_text(best.text, _MAX_TITLE_CHARS)
    # Quitar el punto final (las preguntas/exclamaciones se conservan).
    if title.endswith("."):
        title = title[:-1]
    return title or "Momento destacado"


def _build_keywords(text: str) -> list[str]:
    """Palabras clave: contenido más frecuente del clip (incluye números)."""
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    for position, token in enumerate(_content_tokens(text)):
        counts[token] = counts.get(token, 0) + 1
        order.setdefault(token, position)
    ranked = sorted(counts, key=lambda token: (-counts[token], order.get(token, 0)))
    return ranked[:12]


def _build_proposal(
    scorer: MomentScorer,
    candidate: _ScoredCandidate,
    theme_tokens: list[str],
    max_end: float | None = None,
) -> ClipProposal:
    phrases = scorer.phrases
    start, end, _ = scorer.trim_edges(candidate.start, candidate.end)
    if end - start < 3.0:
        # El recorte de silencios no debe dejar un clip inválido.
        start, end = candidate.start, candidate.end
    # El fin no debe cortar una palabra por la mitad (v1.2.1): se ajusta
    # al fin de la última palabra + cola, sin superar `max_end`.
    end = scorer.snap_end(end, max_end=max_end)
    title = _build_title(scorer, candidate.first_idx, candidate.last_idx)
    description = _trim_text(
        " ".join(phrase.text for phrase in phrases[candidate.first_idx : candidate.last_idx + 1])[
            : _MAX_DESCRIPTION_CHARS * 2
        ],
        _MAX_DESCRIPTION_CHARS,
    )
    keywords = _build_keywords(candidate.text)
    # Hashtags: lo más frecuente del clip + genéricos del tema del video.
    hashtag_words = keywords[:6]
    hashtag_words.extend(token for token in theme_tokens if token not in hashtag_words)
    hashtags = ["#" + keyword for keyword in hashtag_words[:8]]
    hashtag_map = (
        {"youtube": hashtags, "tiktok": hashtags, "instagram": hashtags} if hashtags else {}
    )
    reason_parts = [f"Puntuación {candidate.score:.1f}/10"] + candidate.labels[:3]
    return ClipProposal(
        inicio=round(start, 3),
        fin=round(end, 3),
        titulo=title[:160],
        puntuacion=round(candidate.score, 2),
        razon_viral=" · ".join(reason_parts)[:1400],
        descripcion=description,
        hashtags=hashtag_map,
        palabras_clave=keywords,
        emojis=[],
        signal_breakdown={key: float(value) for key, value in candidate.breakdown.items()},
    )


def detect_moments(
    transcript: list[TranscriptSegment],
    project: ProjectConfig,
    duration: float,
    *,
    input_path: Path | str | None = None,
    ffmpeg_path: str | None = None,
    audio_timeline: np.ndarray | None = None,
    event_timeline: np.ndarray | None = None,
    visual_timeline: np.ndarray | None = None,
    face_timeline: np.ndarray | None = None,
    scene_cut_timeline: np.ndarray | None = None,
    motion_timeline: np.ndarray | None = None,
    speaker_timeline: np.ndarray | None = None,
    visual_stats: VisualStats | None = None,
    threshold: float | None = None,
    min_separation_seconds: float | None = None,
    max_clips: int = 10,
    should_cancel: Callable[[], bool] | None = None,
    on_profile: Callable[[str], None] | None = None,
    enable_semantic: bool = False,
) -> list[ClipProposal]:
    """Detecta momentos por señales sobre el video COMPLETO (sin ventanas).

    El ventaneado de 5 min era una limitación del LLM por TPM; las
    señales no la tienen. ``audio_timeline`` es inyectable para los tests;
    si no se pasa y hay ``input_path`` + ``ffmpeg_path``, se extrae con
    ffmpeg; sin ninguno, el motor trabaja solo con texto (canal de audio
    en cero). ``event_timeline`` es el hook del Tier 1 (Fase B): lo
    produce ``backend.audio_events.get_event_timeline`` y es None cuando
    Tier 1 no está disponible. ``visual_timeline`` (combinado) /
    ``scene_cut_timeline`` / ``motion_timeline`` / ``face_timeline`` son
    los hooks visuales de la Fase C: si no se inyecta NINGUNO y hay
    ``input_path`` + ``ffmpeg_path``, se extraen aquí en una sola pasada
    (``backend.visual_signals.extract_visual_timelines``); si el
    llamador inyecta al menos uno, se asume que el pipeline ya hizo la
    extracción y no se duplica trabajo. ``visual_stats`` alimenta la
    detección de perfil (sin él, el perfil es "mixed"); ``on_profile``
    recibe el perfil detectado para que el llamador lo guarde (p. ej.
    en la metadata del proyecto). ``speaker_timeline`` es el hook de
    diarización (Fase C, Tier 2): lo produce
    ``backend.diarization.get_speaker_timeline`` y es None cuando la
    diarización no está disponible; aquí NO se extrae (la descarga de
    ~33 MB la gestiona el pipeline, como el Tier 1). ``enable_semantic``
    (Fase D) activa
    la deduplicación por paráfrasis con MiniLM en la selección; por
    defecto False (sin descargas en llamadas directas ni en tests): el
    pipeline lo activa tras precalentar el modelo. Devuelve [] si ningún
    candidato supera el umbral: "cero clips" es un resultado válido, no
    un error.
    """
    threshold = project.signal_score_threshold if threshold is None else threshold
    phrases = build_phrases(transcript)
    if not phrases:
        logger.info("El motor de señales no encontró frases en la transcripción.")
        return []
    if audio_timeline is None:
        if input_path is not None and ffmpeg_path:
            audio_timeline = extract_audio_timeline(input_path, ffmpeg_path, duration)
        else:
            logger.info(
                "El motor de señales trabaja sin timeline de audio "
                "(sin ruta de video o sin ffmpeg); el canal de audio puntúa 0."
            )
            audio_timeline = np.zeros(max(1, math.ceil(duration)))
    bounds = _candidate_bounds(project.duration, duration)
    if bounds is None:
        logger.info(
            "El video (%.1f s) es demasiado corto para el motor de señales.",
            duration,
        )
        return []
    minimum, maximum = bounds
    if (
        visual_timeline is None
        and face_timeline is None
        and scene_cut_timeline is None
        and motion_timeline is None
        and input_path is not None
        and ffmpeg_path
    ):
        # El pipeline puede inyectar los timelines (evita duplicar la
        # extracción); si no inyectó ninguno, se extraen aquí. La
        # extracción visual jamás rompe Tier 0: un fallo deja los
        # canales en None y el motor sigue con texto/audio.
        try:
            bundle = extract_visual_timelines(
                input_path, ffmpeg_path, duration, should_cancel=should_cancel
            )
        except Exception as error:  # noqa: BLE001 - la extracción visual jamás rompe Tier 0
            logger.warning("La extracción de señales visuales falló: %s", error)
            bundle = None
        if bundle is not None:
            visual_timeline = bundle.combined
            face_timeline = bundle.face
            scene_cut_timeline = bundle.scene_cuts
            motion_timeline = bundle.motion
            if visual_stats is None:
                visual_stats = bundle.stats
    # Perfil de contenido (Fase C): ajusta los pesos del scorer. Sin
    # estadísticas visuales no hay evidencia visual y el perfil es
    # "mixed" (pesos v1.1.1 sin cambios).
    cut_rate = visual_stats.cut_rate_per_minute if visual_stats is not None else None
    mean_motion = visual_stats.mean_motion_raw if visual_stats is not None else None
    if cut_rate is None and scene_cut_timeline is not None and duration > 0:
        # Respaldo: si el llamador inyectó los cortes pero no las
        # estadísticas, la tasa se estima contando los segundos con corte.
        cut_rate = float(np.count_nonzero(scene_cut_timeline > 0.5)) / duration * 60.0
    profile = detect_content_profile(
        phrases, duration, cut_rate_per_minute=cut_rate, mean_motion_raw=mean_motion
    )
    logger.info("Perfil de contenido detectado: %s.", profile)
    if on_profile is not None:
        try:
            on_profile(profile)
        except Exception as error:  # noqa: BLE001 - el callback no rompe la detección
            logger.warning("El callback de perfil falló: %s", error)
    scorer = MomentScorer(
        phrases,
        audio_timeline,
        duration,
        event_timeline,
        visual_timeline=visual_timeline,
        face_timeline=face_timeline,
        scene_cut_timeline=scene_cut_timeline,
        motion_timeline=motion_timeline,
        speaker_timeline=speaker_timeline,
        profile=profile,
    )
    # Candidatos: cada frase es un inicio; el final es el final natural de
    # una frase posterior dentro del rango de duración pedido.
    candidates: list[_ScoredCandidate] = []
    best_score = 0.0
    stride = 1
    estimated = len(phrases) * max(1, int((maximum - minimum) / 5))
    if estimated > _MAX_CANDIDATES:
        stride = math.ceil(estimated / _MAX_CANDIDATES)
    for step, first_idx in enumerate(range(0, len(phrases), stride)):
        if should_cancel is not None and step % 200 == 0 and should_cancel():
            logger.info("La detección de momentos se canceló.")
            return []
        start = phrases[first_idx].start
        for last_idx in range(first_idx, len(phrases)):
            clip_end = phrases[last_idx].end
            clip_duration = clip_end - start
            if clip_duration > maximum:
                break
            if clip_duration < minimum:
                continue
            score, breakdown, labels = scorer.score_window(first_idx, last_idx)
            best_score = max(best_score, score)
            if score < threshold:
                continue
            text = " ".join(phrase.text for phrase in phrases[first_idx : last_idx + 1])
            candidates.append(
                _ScoredCandidate(
                    score=score,
                    first_idx=first_idx,
                    last_idx=last_idx,
                    start=start,
                    end=clip_end,
                    breakdown=breakdown,
                    labels=labels,
                    text=text,
                )
            )
    if not candidates:
        # Se registra la mejor puntuación encontrada: si el usuario
        # reporta "cero clips", el log dice a cuánto quedó de pasar.
        logger.info(
            "El motor de señales no encontró momentos por encima del umbral "
            "%.1f (cero clips, resultado válido; mejor ventana: %.1f).",
            threshold,
            best_score,
        )
        return []
    separation = (
        min_separation_seconds
        if min_separation_seconds is not None
        else _default_separation(duration)
    )
    selected = _select_diverse(
        candidates,
        threshold=threshold,
        separation_seconds=separation,
        max_clips=max_clips,
        idf_corpus=[phrase.text for phrase in phrases],
        enable_semantic=enable_semantic,
        should_cancel=should_cancel,
    )
    # Tema del video: contenido más frecuente de toda la transcripción,
    # para hashtags genéricos del tema en cada clip.
    all_text = " ".join(phrase.text for phrase in phrases)
    theme_counts: dict[str, int] = {}
    for token in _content_tokens(all_text):
        theme_counts[token] = theme_counts.get(token, 0) + 1
    theme_tokens = [
        token for token, _ in sorted(theme_counts.items(), key=lambda item: -item[1])[:3]
    ]
    proposals = []
    for index, candidate in enumerate(selected):
        # El fin ajustado (snap) no debe superar la duración máxima
        # pedida ni invadir el clip siguiente (la validación del
        # pipeline descarta clips fuera de rango o solapados).
        max_end = candidate.start + maximum
        if index + 1 < len(selected):
            max_end = min(max_end, selected[index + 1].start)
        proposals.append(_build_proposal(scorer, candidate, theme_tokens, max_end=max_end))
    logger.info(
        "El motor de señales eligió %d clip(s) de %d candidato(s) por encima del umbral %.1f.",
        len(proposals),
        len(candidates),
        threshold,
    )
    return proposals


# --- pulido con IA (modo "signals+ai") ----------------------------------------


def polish_titles(
    proposals: list[ClipProposal],
    transcript: list[TranscriptSegment],
    llm: ResolvedLLM,
    *,
    on_rate_limit: Callable[[float, int], None] | None = None,
) -> list[ClipProposal] | None:
    """Pule títulos/descripciones de clips YA ELEGIDOS con UNA llamada corta.

    Reutiliza el cliente LLM existente (reintentos, 429, fallback sin modo
    JSON). Devuelve None si el pulido falla por cualquier motivo o no
    cambia ningún título: el llamador conserva los títulos por plantilla.
    Nunca lanza.
    """
    if not proposals:
        return None
    try:
        items: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals):
            text = " ".join(
                segment.text
                for segment in transcript
                if segment.start < proposal.fin and segment.end > proposal.inicio
            )[:600]
            items.append(
                {
                    "indice": index,
                    "inicio": round(proposal.inicio, 1),
                    "fin": round(proposal.fin, 1),
                    "titulo_actual": proposal.titulo,
                    "texto": text,
                }
            )
        system = (
            "Eres un editor de clips cortos para redes sociales. Te doy clips "
            "ya elegidos con su transcripción y su título actual. Reescribe "
            "cada título para que sea más atractivo y claro, sin inventar "
            "datos ni cambiar el sentido, y escribe una descripción breve "
            "de 1–2 frases. Usa el idioma de la transcripción. Responde "
            'únicamente con un objeto JSON con una única clave "clips", '
            "cuyo valor sea un array de objetos con las claves indice, "
            "titulo, descripcion y palabras_clave (array, máximo 12)."
        )
        user = "Clips a pulir (conserva cada indice tal cual):\n" + json.dumps(
            items, ensure_ascii=False
        )
        timeout = httpx.Timeout(90, connect=15)
        with httpx.Client(timeout=timeout) as client:
            content, _finish_reason = _request_clip_content(
                client, llm, system, user, on_rate_limit=on_rate_limit
            )
        parsed = _extract_json(content)
        if isinstance(parsed, dict):
            parsed = parsed.get("clips")
        if not isinstance(parsed, list):
            logger.warning("El pulido con IA devolvió un JSON sin lista de clips.")
            return None
        by_index = {
            item["indice"]: item
            for item in parsed
            if isinstance(item, dict) and isinstance(item.get("indice"), int)
        }
        polished: list[ClipProposal] = []
        changed = False
        for index, proposal in enumerate(proposals):
            item = by_index.get(index)
            if not item:
                polished.append(proposal)
                continue
            title = str(item.get("titulo") or "").strip()
            if not title:
                polished.append(proposal)
                continue
            update: dict[str, Any] = {"titulo": title[:160]}
            description = str(item.get("descripcion") or "").strip()
            if description:
                update["descripcion"] = description[:4000]
            keywords = [
                str(keyword).strip()
                for keyword in (item.get("palabras_clave") or [])
                if str(keyword).strip()
            ][:12]
            if keywords:
                update["palabras_clave"] = keywords
            updated = proposal.model_copy(update=update)
            if updated.titulo != proposal.titulo:
                changed = True
            polished.append(updated)
        if not changed:
            return None
        return polished
    except Exception as error:  # noqa: BLE001 - el pulido nunca tumba el proyecto
        logger.warning(
            "El pulido de títulos con IA falló; se conservan los títulos por plantilla: %s",
            error,
        )
        return None
