"""Tests de integración de las Fases C y D (v1.2.0), a cargo del coordinador.

Cubre el cableado que NO pertenece a ningún trabajador: el hook semántico
en ``moments._select_diverse`` / ``detect_moments`` y la integración en
``pipeline._propose_valid_signals`` (visuales + precalentado MiniLM +
perfil de contenido). Los modelos reales nunca se descargan aquí: todo lo
que tocaría red o ffmpeg está mockeado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.diarization import DiarizationCancelled
from backend.moments import (
    MomentScorer,
    _ScoredCandidate,
    _select_diverse,
    build_phrases,
    detect_moments,
)
from backend.pipeline import ProjectCancelled, _propose_valid_clips
from backend.schemas import ProjectConfig, TranscriptSegment, Word
from backend.semantic import SemanticCancelled, SemanticError
from backend.visual_signals import VisualTimelines


def _segment(start: float, text: str, *, wpm: float = 150.0) -> TranscriptSegment:
    tokens = text.split()
    per_word = 60.0 / wpm
    words = [
        Word(
            start=round(start + index * per_word, 3),
            end=round(start + (index + 1) * per_word, 3),
            text=token,
        )
        for index, token in enumerate(tokens)
    ]
    return TranscriptSegment(
        start=start, end=round(start + len(tokens) * per_word, 3), text=text, words=words
    )


def _candidate(score: float, start: float, text: str) -> _ScoredCandidate:
    return _ScoredCandidate(
        score=score,
        first_idx=0,
        last_idx=0,
        start=start,
        end=start + 30.0,
        breakdown={},
        labels=[],
        text=text,
    )


# Textos paráfrasis sin tokens de contenido compartidos: el TF-IDF no los
# detecta como duplicados (similitud 0), solo lo semántico los caza.
_PARA_A = "me encanta este lugar maravilloso"
_PARA_B = "adoro aquel rincón fantástico"


def _unit_embeddings() -> np.ndarray:
    return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)


# --- hook semántico en _select_diverse ---------------------------------------


def test_semantic_dedup_removes_paraphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dos paráfrasis (TF-IDF 0) se deduplican con enable_semantic."""
    monkeypatch.setattr("backend.semantic.embed_texts", lambda texts, **_: _unit_embeddings())
    ordered = [_candidate(9.0, 0.0, _PARA_A), _candidate(8.0, 120.0, _PARA_B)]
    selected = _select_diverse(
        ordered,
        threshold=5.0,
        separation_seconds=45.0,
        max_clips=10,
        idf_corpus=[_PARA_A, _PARA_B],
        enable_semantic=True,
    )
    assert [cand.text for cand in selected] == [_PARA_A]


def test_semantic_disabled_keeps_paraphrase() -> None:
    """Sin enable_semantic, las paráfrasis se quedan (TF-IDF no las ve)."""
    ordered = [_candidate(9.0, 0.0, _PARA_A), _candidate(8.0, 120.0, _PARA_B)]
    selected = _select_diverse(
        ordered,
        threshold=5.0,
        separation_seconds=45.0,
        max_clips=10,
        idf_corpus=[_PARA_A, _PARA_B],
    )
    assert [cand.text for cand in selected] == [_PARA_A, _PARA_B]


def test_semantic_orthogonal_embeddings_keep_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embeddings distintos (coseno 0) no deduplican."""
    monkeypatch.setattr(
        "backend.semantic.embed_texts",
        lambda texts, **_: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    ordered = [_candidate(9.0, 0.0, _PARA_A), _candidate(8.0, 120.0, _PARA_B)]
    selected = _select_diverse(
        ordered,
        threshold=5.0,
        separation_seconds=45.0,
        max_clips=10,
        idf_corpus=[_PARA_A, _PARA_B],
        enable_semantic=True,
    )
    assert len(selected) == 2


def test_semantic_none_falls_back_to_tfidf(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_texts None (sin modelo) → solo TF-IDF, sin error."""
    monkeypatch.setattr("backend.semantic.embed_texts", lambda texts, **_: None)
    ordered = [_candidate(9.0, 0.0, _PARA_A), _candidate(8.0, 120.0, _PARA_B)]
    selected = _select_diverse(
        ordered,
        threshold=5.0,
        separation_seconds=45.0,
        max_clips=10,
        idf_corpus=[_PARA_A, _PARA_B],
        enable_semantic=True,
    )
    assert len(selected) == 2


def test_semantic_cancelled_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """SemanticCancelled NO se traga: se propaga al pipeline."""

    def _cancelled(texts: list[str], **_: Any) -> None:
        raise SemanticCancelled("cancelado")

    monkeypatch.setattr("backend.semantic.embed_texts", _cancelled)
    ordered = [_candidate(9.0, 0.0, _PARA_A), _candidate(8.0, 120.0, _PARA_B)]
    with pytest.raises(SemanticCancelled):
        _select_diverse(
            ordered,
            threshold=5.0,
            separation_seconds=45.0,
            max_clips=10,
            idf_corpus=[_PARA_A, _PARA_B],
            enable_semantic=True,
        )


# --- detect_moments: enable_semantic por defecto apagado ----------------------


def test_detect_moments_does_not_touch_semantic_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Por defecto no se llama a embed_texts (sin descargas en tests)."""

    def _boom(texts: list[str], **_: Any) -> None:
        raise AssertionError("embed_texts no debe llamarse con enable_semantic=False")

    monkeypatch.setattr("backend.semantic.embed_texts", _boom)
    transcript = [
        _segment(0.0, "¿Sabías que este truco es increíble y gratis? Mira esto."),
        _segment(20.0, "Te explico el secreto paso a paso, es muy fácil."),
        _segment(40.0, "Y al final te espera una sorpresa importante."),
        _segment(60.0, "Nadie te cuenta la verdad completa, pero yo sí."),
        _segment(80.0, "Recuerda suscribirte para más contenido como este."),
        _segment(100.0, "Nos vemos en el próximo video, cuídate mucho."),
    ]
    proposals = detect_moments(
        transcript,
        ProjectConfig(),
        120.0,
        audio_timeline=np.zeros(120),
    )
    assert isinstance(proposals, list)


def test_detect_moments_semantic_none_matches_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """enable_semantic con embed_texts None da lo mismo que apagado."""
    transcript = [
        _segment(0.0, "¿Sabías que este truco es increíble y gratis? Mira esto."),
        _segment(20.0, "Te explico el secreto paso a paso, es muy fácil."),
        _segment(40.0, "Y al final te espera una sorpresa importante."),
        _segment(60.0, "Nadie te cuenta la verdad completa, pero yo sí."),
        _segment(80.0, "Recuerda suscribirte para más contenido como este."),
        _segment(100.0, "Nos vemos en el próximo video, cuídate mucho."),
    ]
    baseline = detect_moments(transcript, ProjectConfig(), 120.0, audio_timeline=np.zeros(120))
    monkeypatch.setattr("backend.semantic.embed_texts", lambda texts, **_: None)
    with_semantic = detect_moments(
        transcript,
        ProjectConfig(),
        120.0,
        audio_timeline=np.zeros(120),
        enable_semantic=True,
    )
    assert [(p.inicio, p.fin) for p in with_semantic] == [(p.inicio, p.fin) for p in baseline]


# --- cableado en pipeline._propose_valid_signals ------------------------------


def _fake_visual() -> VisualTimelines:
    return VisualTimelines(
        scene_cuts=np.zeros(120),
        motion=np.zeros(120),
        face=np.zeros(120),
        cut_count=12,
        mean_motion_raw=3.5,
        duration=120.0,
    )


def _transcript() -> list[TranscriptSegment]:
    return [
        _segment(0.0, "Hola, este es un video de prueba con contenido."),
        _segment(30.0, "Seguimos hablando de cosas interesantes aquí."),
    ]


def test_pipeline_wires_visual_semantic_and_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """_propose_valid_clips pasa visuales + semántica + perfil a detect_moments."""
    seen: dict[str, Any] = {}
    profiles: list[str] = []
    progress_calls: list[float] = []

    def _fake_detect(transcript: Any, project: Any, duration: float, **kwargs: Any) -> list:
        seen.update(kwargs)
        if kwargs.get("on_profile") is not None:
            kwargs["on_profile"]("gameplay")
        return []

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr(
        "backend.semantic.ensure_minilm_model",
        lambda *a, **k: Path("minilm.onnx"),
    )
    monkeypatch.setattr("backend.moments.detect_moments", _fake_detect)
    result = _propose_valid_clips(
        _transcript(),
        ProjectConfig(),
        120.0,
        None,
        input_path=Path("video.mp4"),
        ffmpeg_path="ffmpeg",
        on_profile=profiles.append,
        on_semantic_download_progress=progress_calls.append,
    )
    assert result == []
    assert seen["enable_semantic"] is True
    assert seen["face_timeline"] is not None
    assert seen["scene_cut_timeline"] is not None
    assert seen["motion_timeline"] is not None
    assert seen["visual_timeline"] is not None
    assert seen["visual_stats"].cut_rate_per_minute == pytest.approx(6.0)
    assert seen["visual_stats"].mean_motion_raw == pytest.approx(3.5)
    assert profiles == ["gameplay"]


def test_pipeline_semantic_failure_disables_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si ensure_minilm_model falla, enable_semantic=False y todo sigue."""
    seen: dict[str, Any] = {}

    def _fake_detect(transcript: Any, project: Any, duration: float, **kwargs: Any) -> list:
        seen.update(kwargs)
        return []

    def _fail(*a: Any, **k: Any) -> None:
        raise SemanticError("sin red")

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr("backend.semantic.ensure_minilm_model", _fail)
    monkeypatch.setattr("backend.moments.detect_moments", _fake_detect)
    result = _propose_valid_clips(
        _transcript(),
        ProjectConfig(),
        120.0,
        None,
        input_path=Path("video.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert result == []
    assert seen["enable_semantic"] is False


def test_pipeline_semantic_cancelled_from_ensure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelación durante la descarga de MiniLM → ProjectCancelled."""

    def _cancelled(*a: Any, **k: Any) -> None:
        raise SemanticCancelled("cancelado")

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr("backend.semantic.ensure_minilm_model", _cancelled)
    with pytest.raises(ProjectCancelled):
        _propose_valid_clips(
            _transcript(),
            ProjectConfig(),
            120.0,
            None,
            input_path=Path("video.mp4"),
            ffmpeg_path="ffmpeg",
        )


def test_pipeline_semantic_cancelled_from_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelación durante el embedding → ProjectCancelled."""

    def _cancelled_detect(*a: Any, **k: Any) -> None:
        raise SemanticCancelled("cancelado")

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr(
        "backend.semantic.ensure_minilm_model",
        lambda *a, **k: Path("minilm.onnx"),
    )
    monkeypatch.setattr("backend.moments.detect_moments", _cancelled_detect)
    with pytest.raises(ProjectCancelled):
        _propose_valid_clips(
            _transcript(),
            ProjectConfig(),
            120.0,
            None,
            input_path=Path("video.mp4"),
            ffmpeg_path="ffmpeg",
        )


def test_pipeline_visual_failure_keeps_tier0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la extracción visual falla, detect_moments recibe Nones y sigue."""
    seen: dict[str, Any] = {}

    def _fake_detect(transcript: Any, project: Any, duration: float, **kwargs: Any) -> list:
        seen.update(kwargs)
        return []

    def _boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("ffmpeg roto")

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.visual_signals.extract_visual_timelines", _boom)
    monkeypatch.setattr(
        "backend.semantic.ensure_minilm_model",
        lambda *a, **k: Path("minilm.onnx"),
    )
    monkeypatch.setattr("backend.moments.detect_moments", _fake_detect)
    result = _propose_valid_clips(
        _transcript(),
        ProjectConfig(),
        120.0,
        None,
        input_path=Path("video.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert result == []
    assert seen["visual_timeline"] is None
    assert seen["face_timeline"] is None
    assert seen["visual_stats"] is None
    assert seen["enable_semantic"] is True


def test_pipeline_empty_transcript_skips_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin transcripción no se descargan modelos ni se extrae nada."""
    calls: list[str] = []

    def _record(name: str) -> Any:
        def _fn(*a: Any, **k: Any) -> Any:
            calls.append(name)
            return None

        return _fn

    monkeypatch.setattr("backend.audio_events.get_event_timeline", _record("events"))
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", _record("diarization"))
    monkeypatch.setattr("backend.visual_signals.extract_visual_timelines", _record("visual"))
    monkeypatch.setattr("backend.semantic.ensure_minilm_model", _record("semantic"))
    monkeypatch.setattr("backend.moments.detect_moments", lambda *a, **k: [])
    result = _propose_valid_clips([], ProjectConfig(), 120.0, None)
    assert result == []
    assert calls == []


# --- canal de diarización en el scorer ---------------------------------------


def _debate_fixture() -> tuple[list, np.ndarray]:
    """Transcripción conversacional de 120 s + audio plano (RMS constante)."""
    transcript = [
        _segment(0.0, "Bienvenidos al debate de hoy, tenemos mucho que discutir."),
        _segment(20.0, "Yo creo que el primer punto es el más importante de todos."),
        _segment(40.0, "No estoy de acuerdo, mira los datos que tengo aquí."),
        _segment(60.0, "Espera, deja que termine de explicar mi postura completa."),
        _segment(80.0, "Vale, pero dime entonces cuál es tu propuesta concreta."),
        _segment(100.0, "Para cerrar, un resumen rápido de lo que acordamos hoy."),
    ]
    phrases = build_phrases(transcript)
    audio = np.full(120, 0.1)
    return phrases, audio


def test_speaker_channel_lifts_debate_window() -> None:
    """Una ventana con turnos de hablante puntúa más que sin diarización."""
    phrases, audio = _debate_fixture()
    speakers = np.zeros(120)
    speakers[20:60] = 1.0  # Debate vivo en 20–60 s.
    plain = MomentScorer(phrases, audio, 120.0)
    with_speakers = MomentScorer(phrases, audio, 120.0, speaker_timeline=speakers)
    base_score, _, _ = plain.score_window(1, 2)
    live_score, breakdown, labels = with_speakers.score_window(1, 2)
    assert live_score > base_score
    assert breakdown["hablantes"] > 0
    assert "Intercambio animado entre hablantes" in labels


def test_speaker_channel_quiet_window_unaffected() -> None:
    """Una ventana sin turnos no gana puntos por el canal de hablantes."""
    phrases, audio = _debate_fixture()
    speakers = np.zeros(120)
    speakers[20:60] = 1.0
    with_speakers = MomentScorer(phrases, audio, 120.0, speaker_timeline=speakers)
    _, breakdown, labels = with_speakers.score_window(4, 5)
    assert breakdown["hablantes"] == 0.0
    assert "Intercambio animado entre hablantes" not in labels


def test_speaker_timeline_all_zeros_does_not_drag_score() -> None:
    """Un monólogo (timeline en ceros) puntúa igual que sin diarización."""
    phrases, audio = _debate_fixture()
    plain = MomentScorer(phrases, audio, 120.0)
    monologue = MomentScorer(phrases, audio, 120.0, speaker_timeline=np.zeros(120))
    base_score, _, _ = plain.score_window(1, 2)
    mono_score, breakdown, _ = monologue.score_window(1, 2)
    assert mono_score == pytest.approx(base_score)
    assert breakdown["hablantes"] == 0.0


# --- cableado de diarización en pipeline -------------------------------------


def test_pipeline_wires_speaker_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """_propose_valid_clips pasa el speaker_timeline a detect_moments."""
    seen: dict[str, Any] = {}
    timeline = np.zeros(120)
    timeline[30:60] = 0.75

    def _fake_detect(transcript: Any, project: Any, duration: float, **kwargs: Any) -> list:
        seen.update(kwargs)
        return []

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: timeline)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr(
        "backend.semantic.ensure_minilm_model",
        lambda *a, **k: Path("minilm.onnx"),
    )
    monkeypatch.setattr("backend.moments.detect_moments", _fake_detect)
    result = _propose_valid_clips(
        _transcript(),
        ProjectConfig(),
        120.0,
        None,
        input_path=Path("video.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert result == []
    assert seen["speaker_timeline"] is timeline


def test_pipeline_diarization_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelación durante la diarización → ProjectCancelled."""

    def _cancelled(*a: Any, **k: Any) -> None:
        raise DiarizationCancelled("cancelado")

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", _cancelled)
    with pytest.raises(ProjectCancelled):
        _propose_valid_clips(
            _transcript(),
            ProjectConfig(),
            120.0,
            None,
            input_path=Path("video.mp4"),
            ffmpeg_path="ffmpeg",
        )


def test_pipeline_diarization_none_keeps_working(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin diarización (None), detect_moments recibe None y todo sigue."""
    seen: dict[str, Any] = {}

    def _fake_detect(transcript: Any, project: Any, duration: float, **kwargs: Any) -> list:
        seen.update(kwargs)
        return []

    monkeypatch.setattr("backend.audio_events.get_event_timeline", lambda *a, **k: None)
    monkeypatch.setattr("backend.diarization.get_speaker_timeline", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.visual_signals.extract_visual_timelines",
        lambda *a, **k: _fake_visual(),
    )
    monkeypatch.setattr(
        "backend.semantic.ensure_minilm_model",
        lambda *a, **k: Path("minilm.onnx"),
    )
    monkeypatch.setattr("backend.moments.detect_moments", _fake_detect)
    result = _propose_valid_clips(
        _transcript(),
        ProjectConfig(),
        120.0,
        None,
        input_path=Path("video.mp4"),
        ffmpeg_path="ffmpeg",
    )
    assert result == []
    assert seen["speaker_timeline"] is None
