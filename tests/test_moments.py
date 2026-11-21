"""Tests del motor de detección de momentos por señales (Fase A, v1.1.0).

Todo sintético o con mocks: sin videos reales ni red. El audio se inyecta
como timeline 1 Hz, salvo un test que genera un tono con ffmpeg (lavfi).
numpy es requisito del motor (en producción llega vía mediapipe/opencv/
faster-whisper); sin numpy, el módulo entero se salta en vez de fallar.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from backend import database, moments, pipeline  # noqa: E402
from backend.llm import ResolvedLLM  # noqa: E402
from backend.moments import (  # noqa: E402
    MomentScorer,
    _default_separation,
    _ScoredCandidate,
    _select_diverse,
    build_phrases,
)
from backend.schemas import (  # noqa: E402
    ClipProposal,
    ProjectConfig,
    TranscriptSegment,
    Word,
)
from backend.settings import AppSettings  # noqa: E402

_FLAT_TOPICS = [
    "El tema de hoy es el clima en la ciudad y las noticias locales.",
    "Vamos a hablar del tráfico y de los horarios del transporte público.",
    "La reunión de ayer trató varios asuntos administrativos pendientes.",
    "El informe describe el procedimiento habitual de cada departamento.",
    "Recordamos que el horario de oficina es de ocho a cinco.",
]
_EXCITING_TEXTS = [
    "¿Sabes cuál es el secreto increíble para ganar dinero gratis?",
    "Mira, esto es brutal y nadie te lo cuenta, es impresionante.",
    "Primero anotas 3 ideas, segundo las ordenas y tercero actúas ya.",
    "¡Imagínalo! En 2024 esto cambió todo para siempre, es una locura.",
]


def _segment(start: float, text: str, *, wpm: float = 150) -> TranscriptSegment:
    tokens = text.split()
    span = max(1.0, len(tokens) * 60.0 / wpm)
    per_word = span / max(1, len(tokens))
    words = []
    cursor = start
    for token in tokens:
        words.append(Word(start=round(cursor, 3), end=round(cursor + per_word, 3), text=token))
        cursor += per_word
    return TranscriptSegment(start=start, end=round(start + span, 3), text=text, words=words)


def _transcript(duration: float, *, exciting_at: float | None = None) -> list[TranscriptSegment]:
    """Transcripción sintética plana, con un bloque excitante opcional."""
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    index = 0
    inserted = False
    while cursor < duration - 6:
        if exciting_at is not None and not inserted and cursor >= exciting_at:
            for text in _EXCITING_TEXTS:
                segment = _segment(cursor, text, wpm=210)
                segments.append(segment)
                cursor = segment.end + 0.4
            index += 4
            inserted = True
            continue
        segment = _segment(cursor, _FLAT_TOPICS[index % len(_FLAT_TOPICS)])
        segments.append(segment)
        cursor = segment.end + 0.8
        index += 1
    return segments


def _audio(duration: float, *, peak_at: float | None = None, peak_rms: float = 0.6) -> Any:
    rng = np.random.default_rng(7)
    rms = 0.10 + 0.01 * rng.standard_normal(int(duration))
    if peak_at is not None:
        start = int(peak_at)
        rms[start : start + 20] = peak_rms
    return np.clip(rms, 0.001, None)


def _settings(data_dir: Path, **overrides: Any) -> AppSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "cors_origins": (),
        "llm_provider": "Groq",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_base_url": "",
        "whisper_model": "small",
    }
    values.update(overrides)
    return AppSettings(**values)


def _create_ready_project(
    settings: AppSettings,
    project_id: str,
    *,
    config: ProjectConfig,
    transcript: list[TranscriptSegment],
    duration: float,
) -> None:
    project_dir = settings.data_dir / "projects" / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "source").write_bytes(b"video-bytes")
    database.create_project(
        settings.database_path,
        project_id,
        "Proyecto de prueba",
        project_dir / "source",
        config.model_dump(mode="json"),
        "test-user",
    )
    database.update_project(
        settings.database_path,
        project_id,
        metadata={"duration": duration},
        transcript=[segment.model_dump(mode="json") for segment in transcript],
    )


# ---------------------------------------------------------------------------
# Frases y scorer
# ---------------------------------------------------------------------------


def test_build_phrases_splits_sentences() -> None:
    transcript = [
        TranscriptSegment(
            start=0,
            end=10,
            text="Hola a todos. ¿Cómo están hoy? Muy bien.",
            words=[
                Word(start=0, end=1, text="Hola"),
                Word(start=1, end=2, text="a"),
                Word(start=2, end=3, text="todos."),
                Word(start=3, end=4, text="¿Cómo"),
                Word(start=4, end=5, text="están"),
                Word(start=5, end=6, text="hoy?"),
                Word(start=6, end=7, text="Muy"),
                Word(start=7, end=8, text="bien."),
            ],
        )
    ]
    phrases = build_phrases(transcript)
    assert [phrase.text for phrase in phrases] == [
        "Hola a todos.",
        "¿Cómo están hoy?",
        "Muy bien.",
    ]
    assert phrases[1].is_question
    assert phrases[1].start == 3
    assert phrases[1].end == 6


def test_scorer_exciting_moment_beats_flat_one() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    scorer = MomentScorer(phrases, audio, 600)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    flat_idx = 0
    exciting_score, exciting_breakdown, _ = scorer.score_window(exciting_idx, exciting_idx + 3)
    flat_score, _, _ = scorer.score_window(flat_idx, flat_idx + 3)
    assert exciting_score > flat_score
    assert exciting_score >= 8.0
    assert flat_score < 5.0
    assert exciting_breakdown["texto"] >= 8.0
    assert exciting_breakdown["audio"] >= 8.0
    assert exciting_breakdown["gancho_inicial"] >= 8.0


def test_events_timeline_hook_changes_score() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    without_events = MomentScorer(phrases, audio, 600)
    score_plain, breakdown_plain, _ = without_events.score_window(exciting_idx, exciting_idx + 3)
    assert "eventos" not in breakdown_plain
    events = np.zeros(600)
    events[300:320] = 0.9  # Risa/aplauso en el momento (Fase B).
    with_events = MomentScorer(phrases, audio, 600, events)
    score_events, breakdown_events, _ = with_events.score_window(exciting_idx, exciting_idx + 3)
    assert "eventos" in breakdown_events
    assert breakdown_events["eventos"] >= 8.0
    assert score_events > score_plain


def test_trim_edges_cuts_silence() -> None:
    transcript = _transcript(120)
    audio = np.full(120, 0.1)
    audio[0:2] = 0.0  # Silencio al inicio.
    audio[118:120] = 0.0  # Silencio al final.
    phrases = build_phrases(transcript)
    scorer = MomentScorer(phrases, audio, 120)
    start, end, multiplier = scorer.trim_edges(0.0, 119.0)
    assert start >= 1.0
    assert end <= 118.0
    assert multiplier == 0.85  # Recorte total > 1 s.
    start2, end2, multiplier2 = scorer.trim_edges(10.0, 20.0)
    assert (start2, end2) == (10.0, 20.0)
    assert multiplier2 == 1.0


# ---------------------------------------------------------------------------
# detect_moments
# ---------------------------------------------------------------------------


def test_detect_moments_finds_exciting_moment() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    clips = moments.detect_moments(transcript, ProjectConfig(), 600, audio_timeline=audio)
    assert len(clips) == 1
    clip = clips[0]
    assert 295 <= clip.inicio <= 305
    assert clip.puntuacion >= 8.0
    assert clip.titulo.startswith("¿Sabes cuál es el secreto")
    assert "pregunta" in clip.razon_viral.lower()
    assert clip.signal_breakdown is not None
    assert clip.signal_breakdown["texto"] >= 8.0
    assert clip.descripcion
    assert clip.palabras_clave
    # Mismo formato ClipProposal que propose_clips: valida sin cambios.
    ClipProposal.model_validate(clip.model_dump(mode="json"))


def test_boring_video_produces_zero_clips() -> None:
    transcript = _transcript(600)
    audio = _audio(600)
    assert moments.detect_moments(transcript, ProjectConfig(), 600, audio_timeline=audio) == []
    # Ni siquiera con un umbral más bajo: no hay ninguna señal.
    assert (
        moments.detect_moments(
            transcript,
            ProjectConfig(),
            600,
            audio_timeline=audio,
            threshold=5.0,
        )
        == []
    )


# Frases conversacionales realistas (v1.1.1): habla normal en español con
# densidad de ganchos natural — alguna pregunta, alguna exclamación, algún
# número — y picos de audio de 1.5–1.8x (no los 6x del test caricaturesco).
# Regresión del reporte de Kelly (21-11-2026): la v1.1.0 no encontraba
# ningún clip en videos reales ni con el umbral en 5.0.
_REALISTIC_TEXTS = [
    "Bueno, hoy vamos a hablar de un tema que me preguntan mucho en los comentarios.",
    "Lo primero que hay que entender es cómo funciona el proceso completo.",
    "Yo empecé con esto hace un par de años y al principio cometí muchos errores.",
    "La placa que tengo aquí es una de las más comunes en el mercado actual.",
    "Mucha gente piensa que es complicado, pero en realidad es bastante directo.",
    "Ahora vamos a ver qué pasa cuando encendemos el equipo por primera vez.",
    "El ventilador hace un poco de ruido al arrancar, eso es normal.",
    "¿Cuál es la diferencia entre estos dos modelos? Vamos a verlo en detalle.",
    "Yo les recomiendo que revisen bien las especificaciones antes de comprar.",
    "Miren este resultado, la diferencia es bastante clara en la práctica.",
    "Cuando lo probé la primera vez no me lo esperaba, la verdad.",
    "Hay tres cosas que siempre reviso antes de instalar cualquier pieza.",
    "Primero la compatibilidad, segundo el consumo y tercero el tamaño físico.",
    "¡Ojo con esto! Si lo conectas al revés puedes dañar la placa.",
    "Para los que juegan, esto hace una diferencia notable en los cuadros por segundo.",
    "Yo juego casi todos los días y nunca he tenido problemas de calentamiento.",
    "Como pueden ver, el rendimiento es bastante bueno para el precio que tiene.",
    "¿Vale la pena gastar el doble por el modelo superior? Depende de tu uso.",
    "En resumen, es una buena opción si buscas algo equilibrado y sin gastar mucho.",
    "Déjenme en los comentarios qué les parece y si quieren que lo compare con otro.",
]


def _realistic_transcript(duration: float) -> list[TranscriptSegment]:
    rng = np.random.default_rng(42)
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    index = 0
    while cursor < duration - 8:
        segment = _segment(
            cursor,
            _REALISTIC_TEXTS[index % len(_REALISTIC_TEXTS)],
            wpm=float(rng.uniform(130, 170)),
        )
        segments.append(segment)
        cursor = segment.end + float(rng.uniform(0.3, 1.0))
        index += 1
    return segments


def _realistic_audio(duration: float) -> Any:
    rng = np.random.default_rng(42)
    rms = 0.10 + 0.02 * rng.standard_normal(int(duration))
    for at, length, level in [(90, 8, 0.16), (210, 6, 0.18), (350, 10, 0.15), (500, 7, 0.17)]:
        rms[at : at + length] = level
    return np.clip(rms, 0.001, None)


def test_realistic_conversational_video_finds_clips_at_5() -> None:
    # Un video conversacional normal (no caricaturesco) debe dar clips
    # con el umbral en 5.0: la v1.1.0 daba cero.
    transcript = _realistic_transcript(600)
    audio = _realistic_audio(600)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        audio_timeline=audio,
    )
    assert len(clips) >= 2
    assert all(clip.puntuacion >= 5.0 for clip in clips)


def test_realistic_video_without_audio_finds_clips_at_5() -> None:
    # Sin timeline de audio (ffmpeg falló o video sin pista), el canal de
    # audio no debe arrastrar la puntuación: el video realista debe dar
    # clips solo con texto y ritmo. La v1.1.0 daba cero (mejor ventana 4.4).
    transcript = _realistic_transcript(600)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        audio_timeline=np.zeros(600),
    )
    assert len(clips) >= 2
    assert all(clip.puntuacion >= 5.0 for clip in clips)
    assert all(clip.signal_breakdown is not None for clip in clips)
    assert all(clip.signal_breakdown["audio"] == 0.0 for clip in clips)


def test_threshold_is_respected() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    scorer = MomentScorer(phrases, audio, 600)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    score, _, _ = scorer.score_window(exciting_idx, exciting_idx + 3)
    assert (
        moments.detect_moments(
            transcript,
            ProjectConfig(),
            600,
            audio_timeline=audio,
            threshold=score + 0.5,
        )
        == []
    )
    found = moments.detect_moments(
        transcript,
        ProjectConfig(),
        600,
        audio_timeline=audio,
        threshold=score - 0.5,
    )
    assert len(found) >= 1
    assert found[0].puntuacion >= score - 0.5


def test_mmr_spreads_clips_across_video() -> None:
    # Dos momentos excitantes alejados (120 s y 420 s): salen los dos.
    transcript = _transcript(600, exciting_at=120)
    extra = []
    cursor = 420.0
    for text in [
        "¿Sabías que el arroz perfecto necesita exactamente 18 minutos de cocción?",
        "El chef reveló el truco exclusivo en su cocina y fue una sorpresa total.",
        "Primero sofríes la cebolla, segundo añades el caldo y tercero tapas la olla.",
        "El resultado fue espectacular, fácil y rápido, digno de un restaurante famoso.",
    ]:
        segment = _segment(cursor, text, wpm=210)
        extra.append(segment)
        cursor = segment.end + 0.4
    merged = [
        segment for segment in transcript if segment.end <= 418 or segment.start >= cursor + 1
    ]
    merged.extend(extra)
    merged.sort(key=lambda item: item.start)
    audio = _audio(600, peak_at=120)
    audio[420:440] = 0.6
    clips = moments.detect_moments(merged, ProjectConfig(), 600, audio_timeline=audio)
    assert len(clips) == 2
    assert clips[0].inicio < 200
    assert clips[1].inicio > 350


def test_mmr_suppresses_nearby_duplicate_moment() -> None:
    # Dos momentos excitantes a 20 s: el segundo queda penalizado por MMR
    # (separación por defecto 45 s en un video de 600 s) y solo sale uno.
    # Umbral explícito 8.0: con el umbral por defecto (5.0 desde v1.1.1),
    # un momento excepcional (9.5 x 0.56 de penalización = 5.3) supera el
    # umbral por diseño ("solo un momento excepcional puede quedar junto
    # a otro elegido"); aquí lo que se prueba es la supresión MMR.
    transcript = _transcript(600, exciting_at=300)
    extra = []
    cursor = 325.0
    for text in _EXCITING_TEXTS:
        segment = _segment(cursor, text, wpm=210)
        extra.append(segment)
        cursor = segment.end + 0.4
    merged = [
        segment for segment in transcript if segment.end <= 323 or segment.start >= cursor + 1
    ]
    merged.extend(extra)
    merged.sort(key=lambda item: item.start)
    audio = _audio(600, peak_at=300)
    audio[325:345] = 0.6
    clips = moments.detect_moments(
        merged,
        ProjectConfig(signal_score_threshold=8.0),
        600,
        audio_timeline=audio,
    )
    assert len(clips) == 1


def test_tfidf_dedup_keeps_highest_scoring_copy() -> None:
    # El MISMO bloque excitante repetido lejos (120 s y 420 s): MMR lo
    # permitiría (280 s de separación), pero TF-IDF detecta que dicen lo
    # mismo y se queda solo con el de mayor puntuación.
    # Umbral explícito 8.0: con el umbral por defecto (5.0 desde v1.1.1),
    # una ventana parcial del segundo bloque (7.96, solo la última frase
    # excitante + cola plana, similitud 0.46 < 0.55) también sería
    # candidata; aquí lo que se prueba es la deduplicación TF-IDF.
    transcript = _transcript(600, exciting_at=120)
    extra = []
    cursor = 420.0
    for text in _EXCITING_TEXTS:
        segment = _segment(cursor, text, wpm=210)
        extra.append(segment)
        cursor = segment.end + 0.4
    merged = [
        segment for segment in transcript if segment.end <= 418 or segment.start >= cursor + 1
    ]
    merged.extend(extra)
    merged.sort(key=lambda item: item.start)
    audio = _audio(600, peak_at=120)
    audio[420:440] = 0.6
    clips = moments.detect_moments(
        merged,
        ProjectConfig(signal_score_threshold=8.0),
        600,
        audio_timeline=audio,
    )
    assert len(clips) == 1


def test_tier0_works_without_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    # Quitar onnxruntime del proceso si otro test lo importó: lo que se
    # prueba es que detect_moments (Tier 0) no lo importa por sí mismo.
    # monkeypatch lo restaura al terminar el test.
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    clips = moments.detect_moments(transcript, ProjectConfig(), 600, audio_timeline=audio)
    assert len(clips) == 1
    assert "onnxruntime" not in sys.modules


def test_detect_moments_without_audio_uses_text_only() -> None:
    transcript = _transcript(600, exciting_at=300)
    clips = moments.detect_moments(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
    )
    assert len(clips) == 1
    assert clips[0].signal_breakdown is not None
    assert clips[0].signal_breakdown["audio"] == 0.0


# ---------------------------------------------------------------------------
# Timeline de audio con ffmpeg (tono generado, sin videos reales)
# ---------------------------------------------------------------------------


def test_extract_audio_timeline_measures_generated_tone(tmp_path: Path) -> None:
    wav_path = tmp_path / "tone.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3,volume=0.5",
            "-y",
            str(wav_path),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    rms = moments.extract_audio_timeline(wav_path, "ffmpeg", 3.0)
    assert rms.shape == (3,)
    # La fuente sine de lavfi genera amplitud ~0.125 (RMS ≈ 0.088); con
    # volume=0.5 queda la mitad (≈ 0.044), estable en los 3 segundos.
    assert all(0.04 < value < 0.05 for value in rms)
    assert max(rms) - min(rms) < 0.005


def test_extract_audio_timeline_missing_file_returns_zeros(tmp_path: Path) -> None:
    rms = moments.extract_audio_timeline(tmp_path / "no-existe.mp4", "ffmpeg", 10.0)
    assert rms.shape == (10,)
    assert bool((rms == 0).all())


# ---------------------------------------------------------------------------
# Pulido con IA (signals+ai)
# ---------------------------------------------------------------------------


def _polished_proposal() -> ClipProposal:
    return ClipProposal(
        inicio=300,
        fin=320,
        titulo="Título por plantilla",
        puntuacion=8.5,
        razon_viral="Puntuación 8.5/10 · Pregunta directa",
        descripcion="Primera frase del clip.",
        palabras_clave=["secreto"],
    )


@pytest.fixture
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quita las variables de proxy del entorno: httpx.Client() las lee
    (trust_env, igual que llm.py) y en esta VM el no_proxy trae IPv6
    entre corchetes que su parser rechaza ("Invalid port")."""
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_polish_titles_rewrites_with_llm(
    monkeypatch: pytest.MonkeyPatch, _no_proxy_env: None
) -> None:
    def fake_request(_client, _llm, _system, _user, **_kwargs):
        return (
            '{"clips": [{"indice": 0, "titulo": "El secreto que nadie cuenta", '
            '"descripcion": "Descubre el truco en 20 segundos.", '
            '"palabras_clave": ["secreto", "truco"]}]}',
            "stop",
        )

    monkeypatch.setattr(moments, "_request_clip_content", fake_request)
    proposal = _polished_proposal()
    polished = moments.polish_titles(
        [proposal],
        [],
        ResolvedLLM("Groq", "k", "m", "https://llm.example"),
    )
    assert polished is not None
    assert polished[0].titulo == "El secreto que nadie cuenta"
    assert polished[0].descripcion == "Descubre el truco en 20 segundos."
    assert polished[0].palabras_clave == ["secreto", "truco"]
    assert polished[0].inicio == 300
    assert polished[0].fin == 320
    assert polished[0].puntuacion == 8.5


def test_polish_titles_failure_keeps_templates(
    monkeypatch: pytest.MonkeyPatch, _no_proxy_env: None
) -> None:
    def failing_request(*_args, **_kwargs):
        raise RuntimeError("Groq no responde")

    monkeypatch.setattr(moments, "_request_clip_content", failing_request)
    assert (
        moments.polish_titles(
            [_polished_proposal()],
            [],
            ResolvedLLM("Groq", "k", "m", "https://llm.example"),
        )
        is None
    )


def test_polish_titles_empty_returns_none() -> None:
    assert (
        moments.polish_titles([], [], ResolvedLLM("Groq", "k", "m", "https://llm.example")) is None
    )


# ---------------------------------------------------------------------------
# Respeto del ajuste moment_engine (dispatch + integración)
# ---------------------------------------------------------------------------


def test_signals_mode_never_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_propose(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("El modo signals no debe llamar al LLM")

    monkeypatch.setattr(pipeline, "propose_clips", forbidden_propose)
    transcript = _transcript(600, exciting_at=300)
    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        None,
    )
    assert len(proposals) == 1
    assert proposals[0].signal_breakdown is not None


def test_signals_mode_wires_tier1_event_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """El pipeline pasa el timeline Tier 1 al scorer (Fase B)."""
    audio_events = pytest.importorskip("backend.audio_events")
    seen: dict[str, Any] = {}

    def fake_timeline(*args: Any, **_kwargs: Any) -> Any:
        seen["input_path"] = args[0]
        seen["duration"] = args[1]
        timeline = np.zeros(600)
        timeline[295:310] = 0.9  # risa/aplauso sobre el bloque excitante
        return timeline

    monkeypatch.setattr(audio_events, "get_event_timeline", fake_timeline)
    transcript = _transcript(600, exciting_at=300)
    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        None,
        input_path=Path("/tmp/no-existe.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert seen["input_path"] == Path("/tmp/no-existe.mp4")
    assert seen["duration"] == 600
    assert len(proposals) == 1
    assert proposals[0].signal_breakdown is not None
    assert proposals[0].signal_breakdown.get("eventos", 0) > 0


def test_signals_mode_tier1_unavailable_falls_back_to_tier0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 1 no disponible (None) -> el motor sigue con Tier 0."""
    audio_events = pytest.importorskip("backend.audio_events")
    monkeypatch.setattr(audio_events, "get_event_timeline", lambda *a, **k: None)
    transcript = _transcript(600, exciting_at=300)
    proposals = pipeline._propose_valid_clips(
        transcript,
        ProjectConfig(signal_score_threshold=5.0),
        600,
        None,
        input_path=Path("/tmp/no-existe.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert len(proposals) == 1
    assert proposals[0].signal_breakdown is not None
    assert proposals[0].signal_breakdown.get("eventos", 0) == 0


def test_signals_mode_tier1_cancel_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelar durante Tier 1 cancela el proyecto (no cae a Tier 0)."""
    audio_events = pytest.importorskip("backend.audio_events")

    def cancelled(*_args: Any, **_kwargs: Any) -> Any:
        raise audio_events.AudioEventsCancelled()

    monkeypatch.setattr(audio_events, "get_event_timeline", cancelled)
    with pytest.raises(pipeline.ProjectCancelled):
        pipeline._propose_valid_clips(
            _transcript(600, exciting_at=300),
            ProjectConfig(signal_score_threshold=5.0),
            600,
            None,
            input_path=Path("/tmp/no-existe.mp4"),
            ffmpeg_path="ffmpeg",
        )


def test_llm_mode_uses_current_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_propose(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls.append(1)
        return [
            {
                "inicio": 10,
                "fin": 30,
                "titulo": "Clip LLM",
                "puntuacion": 8,
                "razon_viral": "Una idea completa.",
            }
        ]

    monkeypatch.setattr(pipeline, "propose_clips", fake_propose)
    proposals = pipeline._propose_valid_clips(
        _transcript(60),
        ProjectConfig(moment_engine="llm"),
        60,
        ResolvedLLM("Groq", "k", "m", "https://llm.example"),
    )
    assert calls
    assert [proposal.titulo for proposal in proposals] == ["Clip LLM"]
    assert proposals[0].signal_breakdown is None


def _mock_render_projects(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []

    def fake_render_projects(*args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        records = args[4]
        for record in records:
            rendered.append(
                {
                    "id": str(record["id"]),
                    "title": record["proposal"]["titulo"],
                    "status": "ready",
                }
            )
        return rendered

    monkeypatch.setattr(pipeline, "_render_projects", fake_render_projects)
    return rendered


def test_process_project_signals_records_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El Tier 1 no es objeto de este test: sin mock, el pipeline intentaría
    # descargar el modelo YAMNet real e importar onnxruntime.
    audio_events = pytest.importorskip("backend.audio_events")
    monkeypatch.setattr(audio_events, "get_event_timeline", lambda *a, **k: None)
    settings = _settings(tmp_path)
    database.initialize(settings.database_path)
    project_id = "signals-project"
    _create_ready_project(
        settings,
        project_id,
        config=ProjectConfig(signal_score_threshold=5.0),
        transcript=_transcript(600, exciting_at=300),
        duration=600,
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffmpeg")

    def forbidden_propose(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("El modo signals no debe llamar al LLM")

    monkeypatch.setattr(pipeline, "propose_clips", forbidden_propose)
    _mock_render_projects(monkeypatch)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["metadata"]["moment_engine"] == "signals"
    assert stored["metadata"]["signal_score_threshold"] == 5.0
    assert len(stored["clips"]) == 1
    assert stored["clips"][0]["title"].startswith("¿Sabes cuál es el secreto")


def test_process_project_signals_ai_without_key_keeps_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El Tier 1 no es objeto de este test: sin mock, el pipeline intentaría
    # descargar el modelo YAMNet real e importar onnxruntime.
    audio_events = pytest.importorskip("backend.audio_events")
    monkeypatch.setattr(audio_events, "get_event_timeline", lambda *a, **k: None)
    settings = _settings(tmp_path, llm_api_key="")
    database.initialize(settings.database_path)
    project_id = "signals-ai-no-key"
    _create_ready_project(
        settings,
        project_id,
        config=ProjectConfig(moment_engine="signals+ai", signal_score_threshold=5.0),
        transcript=_transcript(600, exciting_at=300),
        duration=600,
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffmpeg")
    _mock_render_projects(monkeypatch)

    pipeline.process_project(project_id, None, settings)

    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    # Sin clave no hay pulido: el motor efectivo es "signals".
    assert stored["metadata"]["moment_engine"] == "signals"
    proposals = stored["metadata"]["clip_proposals"]
    assert len(proposals) == 1
    assert proposals[0]["proposal"]["titulo"].startswith("¿Sabes cuál es el secreto")


def test_process_project_llm_records_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    database.initialize(settings.database_path)
    project_id = "llm-project"
    _create_ready_project(
        settings,
        project_id,
        config=ProjectConfig(moment_engine="llm"),
        transcript=_transcript(60),
        duration=60,
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: "ffmpeg")

    calls: list[int] = []

    def fake_propose(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls.append(1)
        return [
            {
                "inicio": 10,
                "fin": 30,
                "titulo": "Clip LLM",
                "puntuacion": 8,
                "razon_viral": "Una idea completa.",
            }
        ]

    monkeypatch.setattr(pipeline, "propose_clips", fake_propose)
    _mock_render_projects(monkeypatch)

    pipeline.process_project(project_id, None, settings)

    assert calls
    stored = database.get_project(settings.database_path, project_id)
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["metadata"]["moment_engine"] == "llm"
    assert stored["clips"][0]["title"] == "Clip LLM"


# ---------------------------------------------------------------------------
# Calibración v1.2.1 (video real de Kelly: 2 clips en vez de 6, y finales
# que cortaban palabras). Los canales opcionales son bonificación, no
# componentes de la media; el fin del clip se ajusta a la palabra.
# ---------------------------------------------------------------------------


def test_events_timeline_all_zeros_does_not_drag_score() -> None:
    # Regresión v1.2.1: en el video de Kelly el timeline de eventos
    # llegaba en ceros (sin risa ni aplauso) y entraba en la media con
    # peso 1.5, costando ~1.4 puntos por ventana. Un canal en cero no
    # debe restar: la puntuación es EXACTA a la base.
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    exciting_idx = next(
        idx for idx, phrase in enumerate(phrases) if phrase.text.startswith("¿Sabes")
    )
    plain = MomentScorer(phrases, audio, 600)
    score_plain, _, _ = plain.score_window(exciting_idx, exciting_idx + 3)
    with_zeros = MomentScorer(phrases, audio, 600, np.zeros(600))
    score_zeros, breakdown_zeros, _ = with_zeros.score_window(exciting_idx, exciting_idx + 3)
    assert score_zeros == score_plain
    assert breakdown_zeros["eventos"] == 0.0
    # Una ventana plana tampoco pierde puntos por el canal en cero.
    flat_plain, _, _ = plain.score_window(0, 3)
    flat_zeros, _, _ = with_zeros.score_window(0, 3)
    assert flat_zeros == flat_plain


def test_quiet_window_with_active_optional_channels_is_unaffected() -> None:
    # Canales opcionales activos EN OTRA PARTE del video (debate en
    # 300–320 s): una ventana tranquila sin señal no gana ni pierde
    # puntos por ellos (la bonificación es por ventana).
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    phrases = build_phrases(transcript)
    speakers = np.zeros(600)
    speakers[300:320] = 1.0
    visual = np.zeros(600)
    visual[300:320] = 0.9
    plain = MomentScorer(phrases, audio, 600)
    with_channels = MomentScorer(
        phrases, audio, 600, speaker_timeline=speakers, visual_timeline=visual
    )
    flat_plain, _, _ = plain.score_window(0, 3)
    flat_channels, breakdown, _ = with_channels.score_window(0, 3)
    assert flat_channels == flat_plain
    assert breakdown["hablantes"] == 0.0
    assert breakdown["visual"] == 0.0


def test_default_separation_short_video_is_30_seconds() -> None:
    # v1.2.1: suelo 30 s (antes 45 s). En un video de 5 min, 45 s de
    # hueco mínimo suprimían momentos de 7.5 a 11–13 s de otro elegido.
    assert _default_separation(320) == pytest.approx(30.0)
    assert _default_separation(600) == pytest.approx(40.0)
    assert _default_separation(1800) == pytest.approx(90.0)


def _candidate(score: float, start: float, end: float, text: str) -> _ScoredCandidate:
    return _ScoredCandidate(
        score=score,
        first_idx=0,
        last_idx=0,
        start=start,
        end=end,
        breakdown={},
        labels=[],
        text=text,
    )


def test_mmr_30s_keeps_nearby_strong_moments_in_short_video() -> None:
    # Dos momentos de 7.5 a 11 s de distancia en un video de 320 s: con
    # la separación por defecto (30 s) salen los dos; con 45 s el
    # segundo quedaba suprimido (7.5 x 0.62 = 4.67 < 5.0).
    first = _candidate(
        7.5, 100.0, 120.0, "El chef revela el truco secreto del arroz perfecto en su cocina."
    )
    second = _candidate(
        7.5, 131.0, 151.0, "El estadio entero celebró el gol del delantero en el minuto final."
    )
    corpus = [first.text, second.text, "El clima de hoy es templado y sin lluvia."]
    selected_30 = _select_diverse(
        [first, second],
        threshold=5.0,
        separation_seconds=_default_separation(320),
        max_clips=10,
        idf_corpus=corpus,
    )
    assert len(selected_30) == 2
    selected_45 = _select_diverse(
        [first, second],
        threshold=5.0,
        separation_seconds=45.0,
        max_clips=10,
        idf_corpus=corpus,
    )
    assert len(selected_45) == 1


def _overlap_segment() -> TranscriptSegment:
    # Palabras con timestamps solapados como los de Whisper: "poco"
    # empieza (33.05) antes de que termine "un" (33.16), así que el fin
    # de frase 33.16 cae dentro de "poco". Caso real del video de Kelly.
    return TranscriptSegment(
        start=30.0,
        end=40.0,
        text="una un poco truquito falso",
        words=[
            Word(start=30.0, end=31.0, text="una"),
            Word(start=31.5, end=33.16, text="un"),
            Word(start=33.05, end=33.45, text="poco"),
            Word(start=33.6, end=34.0, text="truquito"),
            Word(start=34.5, end=35.0, text="falso"),
        ],
    )


def test_snap_end_extends_mid_word_cut() -> None:
    phrases = build_phrases([_overlap_segment()])
    scorer = MomentScorer(phrases, np.zeros(60), 60.0)
    # Fin dentro de "poco" -> fin de "poco" (33.45) + cola acotada al
    # inicio de "truquito" (33.6): 33.45 + 0.15 = 33.6.
    assert scorer.snap_end(33.16) == pytest.approx(33.6)


def test_snap_end_tail_never_enters_next_word() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=20.0,
        text="hola mundo",
        words=[
            Word(start=10.0, end=10.5, text="hola"),
            Word(start=10.6, end=11.0, text="mundo"),
        ],
    )
    phrases = build_phrases([segment])
    scorer = MomentScorer(phrases, np.zeros(60), 60.0)
    # Fin exacto en "hola" (10.5) + cola de 0.4, acotada al inicio de
    # "mundo" (10.6): el fin queda en 10.6, no dentro de "mundo".
    assert scorer.snap_end(10.5) == pytest.approx(10.6)


def test_snap_end_without_word_timestamps_uses_silence() -> None:
    segment = TranscriptSegment(start=0.0, end=20.0, text="Habla sin marcas de tiempo.")
    phrases = build_phrases([segment])
    audio = np.full(60, 0.1)
    audio[12] = 0.0  # Segundo 12 en silencio.
    scorer = MomentScorer(phrases, audio, 60.0)
    assert scorer.snap_end(10.5) == pytest.approx(12.0)
    # Sin silencio en los 2 s siguientes, el fin no se mueve.
    scorer2 = MomentScorer(phrases, np.full(60, 0.1), 60.0)
    assert scorer2.snap_end(10.5) == pytest.approx(10.5)


def test_build_proposal_snaps_end_to_word() -> None:
    phrases = build_phrases([_overlap_segment()])
    scorer = MomentScorer(phrases, np.zeros(60), 60.0)
    candidate = _ScoredCandidate(
        score=8.0,
        first_idx=0,
        last_idx=0,
        start=30.0,
        end=33.16,
        breakdown={"audio": 5.0, "texto": 8.0, "ritmo": 5.0, "gancho_inicial": 5.0},
        labels=[],
        text="una un poco truquito falso",
    )
    proposal = moments._build_proposal(scorer, candidate, ["truquito"])
    assert proposal.inicio == pytest.approx(30.0)
    assert proposal.fin == pytest.approx(33.6)


def test_detect_moments_clip_ends_do_not_cut_words() -> None:
    # Invariante de punta a punta: con timestamps solapados (como los de
    # Whisper), ningún fin de clip debe caer dentro de una palabra.
    transcript = _transcript(600, exciting_at=300)
    segments = sorted(transcript, key=lambda item: item.start)
    for index, segment in enumerate(segments):
        if segment.start >= 320 and segment.words:
            previous_end = segments[index - 1].end
            first = segment.words[0]
            segment.words[0] = Word(start=previous_end - 0.3, end=first.end, text=first.text)
            break
    audio = _audio(600, peak_at=300)
    clips = moments.detect_moments(transcript, ProjectConfig(), 600, audio_timeline=audio)
    assert len(clips) >= 1
    all_words = [word for item in transcript for word in item.words]
    for clip in clips:
        assert not any(word.start < clip.fin < word.end for word in all_words)


# --- selección con semántica perezosa (v1.3.1) --------------------------------


def test_select_diverse_only_touches_embeddings_it_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con max_clips=2, _select_diverse solo pide 2 embeddings de 70."""
    from backend import semantic

    touched: list[int] = []
    seen_texts: list[list[str]] = []

    class _RecordingLazy:
        def __init__(self, texts: list[str], **kwargs: Any) -> None:
            seen_texts.append(texts)

        @property
        def available(self) -> bool:
            return True

        def get(self, index: int) -> np.ndarray:
            touched.append(index)
            vector = np.zeros(384, dtype=np.float32)
            vector[index % 384] = 1.0  # Ortogonales: sin filtrado por paráfrasis.
            return vector

    monkeypatch.setattr(semantic, "LazyEmbeddings", _RecordingLazy)
    candidates = [
        _candidate(
            7.5,
            float(i * 60),
            float(i * 60 + 20),
            # Sufijo de letras (el tokenizador TF-IDF quita dígitos):
            # cada candidato tiene tokens propios, sin duplicados TF-IDF.
            " ".join(
                f"{stem}{chr(97 + i % 26)}{chr(97 + i // 26)}"
                for stem in ("palabra", "unica", "distinta", "tema", "especial", "raro")
            ),
        )
        for i in range(70)
    ]
    selected = _select_diverse(
        candidates,
        threshold=5.0,
        separation_seconds=0.001,  # Sin supresión por cercanía.
        max_clips=2,
        idf_corpus=[c.text for c in candidates],
        enable_semantic=True,
    )
    assert len(selected) == 2
    assert len(seen_texts[0]) == 70  # Recibió todos los textos...
    assert set(touched) == {0, 1}  # ...pero solo embebió los 2 necesarios.
