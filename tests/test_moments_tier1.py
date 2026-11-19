"""Tests del Tier 1 del motor de momentos (Fase B, v1.1.0): YAMNet ONNX.

Todo con mocks o audio sintético: sin red ni modelo real, salvo UN
test de integración que usa el modelo real si la variable
KELLYCLIPAI_YAMNET_TEST_MODEL apunta a un yamnet.onnx válido (se
salta en CI, donde el modelo no está). numpy es requisito del motor
(ver tests/test_moments.py); sin numpy, el módulo entero se salta.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from backend import audio_events, moments  # noqa: E402
from backend.audio_events import (  # noqa: E402
    EVENT_THRESHOLDS,
    AudioEventError,
    AudioEventsCancelled,
    _ChunkedInference,
    scores_to_timeline,
)
from backend.moments import MomentScorer, build_phrases  # noqa: E402
from backend.schemas import ProjectConfig, TranscriptSegment, Word  # noqa: E402

_LAUGHTER = 13  # Laughter
_APPLAUSE = 62  # Applause
_SHOUT = 6  # Shout
_BEEP = 475  # "Beep, bleep" (referencia de verificación del modelo)
_SILENCE = 494  # Silence
_NUM_CLASSES = 521
_SAMPLE_RATE = 16000


@pytest.fixture(autouse=True)
def _clean_session_cache() -> Any:
    audio_events._reset_session_cache()
    yield
    audio_events._reset_session_cache()


@pytest.fixture(autouse=True)
def _no_download_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_events, "_YAMNET_RETRY_DELAY_SECONDS", 0.0)


def _frame_count(num_samples: int) -> int:
    """Frames YAMNet para un waveform (fórmula del README del modelo)."""
    if num_samples <= 0:
        return 0
    return 1 + max(0, math.ceil((num_samples - 15360) / 7680))


def _scores(frames: int, hits: list[tuple[int, int, float]]) -> np.ndarray:
    """Matriz [frames, 521] con las puntuaciones indicadas."""
    matrix = np.zeros((frames, _NUM_CLASSES), dtype=np.float64)
    for frame_idx, class_idx, value in hits:
        matrix[frame_idx, class_idx] = value
    return matrix


class _FakeSession:
    """Sesión ONNX simulada: en la columna 0, el índice de frame relativo.

    Permite verificar los límites de bloque: cada bloque completo debe
    aportar 125 frames numerados 0..124, sin duplicados ni huecos.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []  # Longitudes de waveform recibidas.

    def run(self, _output_names: list[str], feed: dict[str, Any]) -> list[np.ndarray]:
        waveform = feed["waveform"]
        self.calls.append(int(waveform.size))
        frames = _frame_count(int(waveform.size))
        matrix = np.zeros((frames, _NUM_CLASSES), dtype=np.float64)
        matrix[:, 0] = np.arange(frames, dtype=np.float64)
        return [matrix]


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
    flat = "El tema de hoy es el clima en la ciudad y las noticias locales."
    exciting = "¿Sabes cuál es el secreto increíble para ganar dinero gratis?"
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    while cursor < duration:
        text = exciting if exciting_at is not None and abs(cursor - exciting_at) < 30 else flat
        segment = _segment(cursor, text)
        segments.append(segment)
        cursor = segment.end + 2.0
    return segments


def _audio(seconds: int, *, peak_at: float | None = None) -> np.ndarray:
    timeline = np.full(seconds, 0.05, dtype=np.float64)
    if peak_at is not None:
        start = int(peak_at)
        timeline[start : start + 20] = 0.6
    return timeline


# --- el módulo no exige onnxruntime para importarse ---------------------------


def test_import_audio_events_does_not_import_onnxruntime() -> None:
    """Importar audio_events no debe importar onnxruntime (perezoso).

    Se comprueba en un intérprete nuevo: en la suite, otros tests ya
    pueden haber importado onnxruntime (p. ej. al ejercitar el Tier 1
    real), y lo que importa es que el import en limpio no lo arrastre.
    """
    code = (
        "import sys\n"
        "from backend import audio_events\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "assert audio_events._YAMNET_REPO_ID == 'audiomagic/yamnet-onnx'\n"
        "assert 'onnxruntime' not in sys.modules\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr


# --- umbrales por clase --------------------------------------------------------


def test_thresholds_are_documented() -> None:
    assert EVENT_THRESHOLDS == {
        "risa": 0.5,
        "aplauso": 0.5,
        "ovación": 0.5,
        "grito": 0.6,
    }


def test_scores_below_threshold_produce_no_event() -> None:
    # Frame 10: centro 5,28 s -> segundo 5. Risa 0,49 < umbral 0,5.
    result = scores_to_timeline(_scores(30, [(10, _LAUGHTER, 0.49)]), 20.0)
    assert result.timeline.shape == (20,)
    assert float(np.max(result.timeline)) == 0.0
    assert result.events == []


def test_scores_above_threshold_produce_event() -> None:
    result = scores_to_timeline(_scores(30, [(10, _LAUGHTER, 0.75)]), 20.0)
    assert result.timeline[5] == pytest.approx(0.75)
    assert float(np.max(np.delete(result.timeline, 5))) == 0.0
    assert len(result.events) == 1
    event = result.events[0]
    assert event.kind == "risa"
    assert event.label == "Risa"
    assert event.start == 5.0
    assert event.end == 6.0
    assert event.score == pytest.approx(0.75)


def test_shout_needs_higher_threshold_than_laughter() -> None:
    # Grito 0,55: por encima del umbral de risa pero por debajo del de grito.
    result = scores_to_timeline(_scores(30, [(10, _SHOUT, 0.55)]), 20.0)
    assert result.events == []
    result = scores_to_timeline(_scores(30, [(10, _SHOUT, 0.61)]), 20.0)
    assert len(result.events) == 1
    assert result.events[0].kind == "grito"


def test_applause_uses_laughter_threshold() -> None:
    result = scores_to_timeline(_scores(30, [(10, _APPLAUSE, 0.51)]), 20.0)
    assert len(result.events) == 1
    assert result.events[0].kind == "aplauso"


def test_events_merge_consecutive_seconds_of_same_kind() -> None:
    # Frames 10-11 -> segundo 5; 12-13 -> segundo 6; 14 -> segundo 7.
    hits = [
        (10, _LAUGHTER, 0.7),
        (12, _LAUGHTER, 0.8),
        (14, _LAUGHTER, 0.6),
    ]
    result = scores_to_timeline(_scores(30, hits), 20.0)
    assert len(result.events) == 1
    assert result.events[0].start == 5.0
    assert result.events[0].end == 8.0
    assert result.events[0].score == pytest.approx(0.8)


def test_different_kinds_do_not_merge() -> None:
    hits = [(10, _LAUGHTER, 0.7), (12, _APPLAUSE, 0.9)]
    result = scores_to_timeline(_scores(30, hits), 20.0)
    assert [event.kind for event in result.events] == ["risa", "aplauso"]


def test_frames_beyond_duration_are_ignored() -> None:
    # 30 frames cubren ~15 s; con duración 5 s solo valen los primeros.
    hits = [(25, _LAUGHTER, 0.9)]
    result = scores_to_timeline(_scores(30, hits), 5.0)
    assert result.events == []


def test_empty_scores_give_empty_timeline() -> None:
    result = scores_to_timeline(np.zeros((0, _NUM_CLASSES)), 10.0)
    assert result.timeline.shape == (10,)
    assert result.events == []


# --- inferencia por bloques ----------------------------------------------------


def test_chunked_inference_matches_single_run() -> None:
    session = _FakeSession()
    inference = _ChunkedInference(session)
    total_samples = 130 * _SAMPLE_RATE
    waveform = np.zeros(total_samples, dtype=np.float32)
    blocks: list[np.ndarray] = []
    # Alimentación en lecturas desiguales, como las de ffmpeg.
    cursor = 0
    for read_seconds in (5, 5, 17, 3, 60, 40):
        chunk = waveform[cursor : cursor + read_seconds * _SAMPLE_RATE]
        cursor += chunk.size
        blocks.extend(inference.feed(chunk))
    blocks.extend(inference.finish())
    assert cursor == total_samples
    scores = np.concatenate(blocks, axis=0)
    assert scores.shape == (_frame_count(total_samples), _NUM_CLASSES)
    # Dos bloques completos de 125 frames + cola de 20: sin duplicados.
    expected = np.concatenate(
        [np.arange(125), np.arange(125), np.arange(_frame_count(total_samples) - 250)]
    )
    assert list(scores[:, 0]) == list(expected.astype(np.float64))
    assert session.calls == [
        60 * _SAMPLE_RATE + 7680,
        60 * _SAMPLE_RATE + 7680,
        10 * _SAMPLE_RATE,
    ]


def test_chunked_inference_short_audio_is_a_single_run() -> None:
    session = _FakeSession()
    inference = _ChunkedInference(session)
    waveform = np.zeros(3 * _SAMPLE_RATE, dtype=np.float32)
    assert inference.feed(waveform) == []
    blocks = inference.finish()
    assert len(blocks) == 1
    assert blocks[0].shape == (6, _NUM_CLASSES)
    assert session.calls == [3 * _SAMPLE_RATE]


def test_chunked_inference_empty_audio() -> None:
    session = _FakeSession()
    inference = _ChunkedInference(session)
    assert inference.feed(np.zeros(0, dtype=np.float32)) == []
    assert inference.finish() == []
    assert session.calls == []


# --- descarga bajo demanda -----------------------------------------------------


def _fake_model_file(tmp_path: Path, content: bytes = b"modelo-falso") -> Path:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    model_path = model_dir / "yamnet.onnx"
    model_path.write_bytes(content)
    return model_path


def test_download_success_reports_progress_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _fake_model_file(tmp_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    monkeypatch.setattr(audio_events, "_YAMNET_SHA256", digest)
    seen: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> str:
        seen.update(kwargs)
        progress = kwargs["tqdm_class"](total=1000)
        progress.update(500)
        progress.update(500)
        progress.close()
        return str(model_path.parent)

    monkeypatch.setattr(audio_events, "snapshot_download", fake_download)
    calls: list[float] = []
    result = audio_events.ensure_yamnet_model(calls.append)
    assert result == model_path
    assert seen["repo_id"] == "audiomagic/yamnet-onnx"
    assert seen["allow_patterns"] == ["yamnet.onnx"]
    assert calls[0] == 0
    assert calls[-1] == 100
    assert any(0 < value < 100 for value in calls)


def test_download_failure_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def failing_download(**kwargs: Any) -> str:
        attempts.append(1)
        raise RuntimeError("sin red")

    monkeypatch.setattr(audio_events, "snapshot_download", failing_download)
    with pytest.raises(AudioEventError, match="No se pudo descargar"):
        audio_events.ensure_yamnet_model()
    assert len(attempts) == 2


def test_download_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = _fake_model_file(tmp_path)
    monkeypatch.setattr(
        audio_events, "_YAMNET_SHA256", hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    attempts: list[int] = []

    def flaky_download(**kwargs: Any) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("corte momentáneo")
        return str(model_path.parent)

    monkeypatch.setattr(audio_events, "snapshot_download", flaky_download)
    assert audio_events.ensure_yamnet_model() == model_path
    assert len(attempts) == 2


def test_sha_mismatch_rejected_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _fake_model_file(tmp_path, content=b"contenido-corrupto")
    attempts: list[int] = []

    def fake_download(**kwargs: Any) -> str:
        attempts.append(1)
        return str(model_path.parent)

    monkeypatch.setattr(audio_events, "snapshot_download", fake_download)
    with pytest.raises(AudioEventError, match="sha256"):
        audio_events.ensure_yamnet_model()
    assert len(attempts) == 1  # Reintentar devolvería el mismo archivo corrupto.


def test_cancelled_download_raises_and_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_download(**kwargs: Any) -> str:
        attempts.append(1)
        progress = kwargs["tqdm_class"](total=1000, disable=True)
        progress.update(10)
        return "/no-importa"

    monkeypatch.setattr(audio_events, "snapshot_download", fake_download)
    with pytest.raises(AudioEventsCancelled):
        audio_events.ensure_yamnet_model(should_cancel=lambda: True)
    assert len(attempts) == 1


# --- fallback a Tier 0 ---------------------------------------------------------


def test_get_event_timeline_falls_back_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio_events,
        "snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("sin red")),
    )
    assert audio_events.get_event_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_get_event_timeline_falls_back_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "yamnet.onnx"
    model_path.write_bytes(b"irrelevante")
    monkeypatch.setattr(audio_events, "ensure_yamnet_model", lambda *a, **k: model_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert audio_events.get_event_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_load_session_without_onnxruntime_raises_audio_event_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(AudioEventError, match="onnxruntime"):
        audio_events.load_yamnet_session(tmp_path / "yamnet.onnx")


def test_get_event_timeline_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Tier 1 desactivado: no debe descargar nada")

    monkeypatch.setattr(audio_events, "ensure_yamnet_model", fail_if_called)
    assert audio_events.get_event_timeline("video.mp4", 60.0, "ffmpeg", enabled=False) is None


def test_get_event_timeline_without_ffmpeg_returns_none() -> None:
    assert audio_events.get_event_timeline("video.mp4", 60.0, None) is None
    assert audio_events.get_event_timeline(None, 60.0, "ffmpeg") is None


def test_get_event_timeline_swallows_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fallo inesperado")

    monkeypatch.setattr(audio_events, "detect_audio_events", boom)
    assert audio_events.get_event_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_get_event_timeline_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    def cancelled(*args: Any, **kwargs: Any) -> Any:
        raise AudioEventsCancelled("cancelado")

    monkeypatch.setattr(audio_events, "detect_audio_events", cancelled)
    with pytest.raises(AudioEventsCancelled):
        audio_events.get_event_timeline("video.mp4", 60.0, "ffmpeg")


def test_is_tier1_available_false_without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert audio_events.is_tier1_available() is False


def test_is_tier1_available_checks_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setattr(audio_events, "hf_hub_download", lambda **kwargs: "/cache/yamnet.onnx")
    assert audio_events.is_tier1_available() is True
    monkeypatch.setattr(
        audio_events,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no en caché")),
    )
    assert audio_events.is_tier1_available() is False


# --- el canal de eventos en el scorer ------------------------------------------


def test_laughter_raises_clip_score_and_labels_it() -> None:
    transcript = _transcript(120, exciting_at=60)
    phrases = build_phrases(transcript)
    audio = _audio(120, peak_at=60)
    first_idx = next(i for i, p in enumerate(phrases) if p.start >= 55)
    last_idx = next(i for i, p in enumerate(phrases) if p.end >= 85)
    scorer_plain = MomentScorer(phrases, audio, 120.0)
    score_plain, breakdown_plain, _ = scorer_plain.score_window(first_idx, last_idx)
    assert "eventos" not in breakdown_plain
    events = np.zeros(120, dtype=np.float64)
    events[60:80] = 0.8  # Risa sostenida durante la ventana.
    scorer_events = MomentScorer(phrases, audio, 120.0, events)
    score_events, breakdown_events, labels = scorer_events.score_window(first_idx, last_idx)
    assert score_events > score_plain
    assert breakdown_events["eventos"] > 0
    assert "Reacción del público (risa o aplauso)" in labels


def test_events_timeline_without_events_does_not_boost() -> None:
    transcript = _transcript(120, exciting_at=60)
    phrases = build_phrases(transcript)
    audio = _audio(120, peak_at=60)
    first_idx = next(i for i, p in enumerate(phrases) if p.start >= 55)
    last_idx = next(i for i, p in enumerate(phrases) if p.end >= 85)
    scorer = MomentScorer(phrases, audio, 120.0, np.zeros(120))
    _, breakdown, labels = scorer.score_window(first_idx, last_idx)
    assert breakdown["eventos"] == 0.0
    assert "Reacción del público (risa o aplauso)" not in labels


def test_detect_moments_includes_events_breakdown() -> None:
    transcript = _transcript(600, exciting_at=300)
    audio = _audio(600, peak_at=300)
    events = np.zeros(600, dtype=np.float64)
    events[290:320] = 0.8
    clips = moments.detect_moments(
        transcript, ProjectConfig(), 600, audio_timeline=audio, event_timeline=events
    )
    assert len(clips) == 1
    assert clips[0].signal_breakdown.get("eventos", 0) > 0
    assert "Reacción del público" in clips[0].razon_viral


# --- extracción con ffmpeg (sesión simulada, tono real) ------------------------


def test_detect_audio_events_with_ffmpeg_tone(tmp_path: Path) -> None:
    wav_path = tmp_path / "tono.wav"
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
    session = _FakeSession()
    detected = audio_events.detect_audio_events(wav_path, 3.0, "ffmpeg", session=session)
    assert detected.timeline.shape == (3,)
    assert detected.events == []  # La sesión simulada no emite eventos.
    assert session.calls == [3 * _SAMPLE_RATE]


def test_detect_audio_events_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioEventError):
        audio_events.detect_audio_events(
            tmp_path / "no-existe.mp4", 10.0, "ffmpeg", session=_FakeSession()
        )


# --- integración con el modelo real (salta sin modelo ni runtime) --------------


def test_real_yamnet_inference() -> None:
    """Carga el .onnx real y reproduce el comportamiento de referencia.

    Requiere KELLYCLIPAI_YAMNET_TEST_MODEL=<ruta a yamnet.onnx> y
    onnxruntime; en CI se salta (el modelo no se commitea al repo).
    """
    model = os.environ.get("KELLYCLIPAI_YAMNET_TEST_MODEL", "")
    if not model or not Path(model).is_file():
        pytest.skip("modelo YAMNet real no disponible")
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "waveform"
    seconds = np.arange(3 * _SAMPLE_RATE, dtype=np.float32) / _SAMPLE_RATE
    sine = (0.5 * np.sin(2 * np.pi * 1000 * seconds)).astype(np.float32)
    silence = np.zeros(3 * _SAMPLE_RATE, dtype=np.float32)
    scores, _, _ = session.run(None, {"waveform": sine})
    assert scores.shape == (6, _NUM_CLASSES)
    assert int(np.argmax(scores[0])) == _BEEP
    scores_silence, _, _ = session.run(None, {"waveform": silence})
    assert int(np.argmax(scores_silence[0])) == _SILENCE
    assert float(scores_silence[0, _SILENCE]) == pytest.approx(1.0, abs=0.01)
    # El silencio no es un evento: timeline en cero.
    result = scores_to_timeline(scores_silence, 3.0)
    assert result.events == []
    assert float(np.max(result.timeline)) == 0.0
