"""Tests de la diarización Tier 2 (Fase C, v1.2.0): sherpa-onnx en onnxruntime.

Todo con mocks o audio sintético: sin red ni modelo real, salvo los
tests de integración que usan los modelos reales si la variable
KELLYCLIPAI_DIARIZATION_TEST_MODELS apunta a una carpeta con los dos
.onnx en layout de caché (se saltan en CI, donde los modelos no
están) y KELLYCLIPAI_DIARIZATION_TEST_WAV a un wav de 2 hablantes.
numpy es requisito del motor (ver tests/test_moments.py); sin numpy,
el módulo entero se salta.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from backend import diarization  # noqa: E402
from backend.diarization import (  # noqa: E402
    DiarizationCancelled,
    DiarizationError,
    SpeakerSegment,
    _kaldi_fbank,
    _merge_similar_centroids,
    _StreamingDiarizer,
    build_segments,
    cluster_embeddings,
    segments_to_timeline,
    speaker_count_per_frame,
)

_SAMPLE_RATE = 16000
_FRAMES_PER_WINDOW = 589
_WINDOW_SIZE = 160000
_WINDOW_SHIFT = 16000


@pytest.fixture(autouse=True)
def _clean_session_cache() -> Any:
    diarization._reset_session_cache()
    yield
    diarization._reset_session_cache()


@pytest.fixture(autouse=True)
def _no_download_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diarization, "_DOWNLOAD_RETRY_DELAY_SECONDS", 0.0)


# --- el módulo no exige onnxruntime para importarse ---------------------------


def test_import_diarization_does_not_import_onnxruntime() -> None:
    """Importar diarization no debe importar onnxruntime (perezoso)."""
    code = (
        "import sys\n"
        "from backend import diarization\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "assert diarization._EMBEDDING_DIM == 192\n"
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


# --- timeline de turnos (función pura) ------------------------------------------


def _alternating_segments(
    duration: float, turn_every: float, start_speaker: int = 0
) -> list[SpeakerSegment]:
    segments: list[SpeakerSegment] = []
    cursor = 0.0
    speaker = start_speaker
    while cursor < duration:
        end = min(duration, cursor + turn_every)
        segments.append(SpeakerSegment(cursor, end, speaker))
        speaker = 1 - speaker
        cursor = end
    return segments


def test_single_speaker_timeline_is_zero() -> None:
    segments = [SpeakerSegment(0.0, 60.0, 0)]
    timeline = segments_to_timeline(segments, 60.0)
    assert timeline.shape == (60,)
    assert float(np.max(timeline)) == 0.0


def test_no_segments_timeline_is_zero() -> None:
    timeline = segments_to_timeline([], 30.0)
    assert timeline.shape == (30,)
    assert float(np.max(timeline)) == 0.0


def test_fast_turns_score_high() -> None:
    # Turno cada 2 s: debate vivo, 10 turnos en cada ventana de ±10 s.
    segments = _alternating_segments(60.0, 2.0)
    timeline = segments_to_timeline(segments, 60.0)
    assert timeline.shape == (60,)
    assert float(timeline[30]) == pytest.approx(1.0)
    assert float(np.mean(timeline[10:50])) == pytest.approx(1.0)


def test_calm_interview_scores_low_but_not_zero() -> None:
    # Un turno cada 20 s: ~1 turno por ventana de ±10 s -> 0,25.
    segments = _alternating_segments(120.0, 20.0)
    timeline = segments_to_timeline(segments, 120.0)
    assert float(timeline[60]) == pytest.approx(0.25)
    assert float(np.max(timeline)) <= 0.5


def test_same_speaker_gap_is_not_a_turn() -> None:
    # Mismo hablante antes y después de una pausa: no hay turno.
    segments = [SpeakerSegment(0.0, 10.0, 0), SpeakerSegment(12.0, 22.0, 0)]
    timeline = segments_to_timeline(segments, 30.0)
    assert float(np.max(timeline)) == 0.0


def test_turns_outside_window_do_not_count() -> None:
    # Un solo turno en 5 s: los segundos a más de 10 s quedan en cero.
    segments = [SpeakerSegment(0.0, 5.0, 0), SpeakerSegment(5.0, 40.0, 1)]
    timeline = segments_to_timeline(segments, 40.0)
    assert float(timeline[5]) == pytest.approx(0.25)
    assert float(timeline[30]) == 0.0


def test_timeline_length_is_ceiled_duration() -> None:
    assert segments_to_timeline([], 10.2).shape == (11,)
    assert segments_to_timeline([], 0.0).shape == (1,)


def test_speaker_segment_label() -> None:
    assert SpeakerSegment(0.0, 1.0, 0).label == "SPEAKER_00"
    assert SpeakerSegment(0.0, 1.0, 3).label == "SPEAKER_03"


# --- fbank estilo Kaldi (función pura) ------------------------------------------


def test_kaldi_fbank_shape() -> None:
    # Frames con snip_edges=False: (N + 80) // 160.
    assert _kaldi_fbank(np.zeros(_SAMPLE_RATE, dtype=np.float32)).shape == (100, 80)
    assert _kaldi_fbank(np.zeros(8000, dtype=np.float32)).shape == (50, 80)
    assert _kaldi_fbank(np.zeros(0, dtype=np.float32)).shape == (0, 80)
    assert _kaldi_fbank(np.zeros(79, dtype=np.float32)).shape == (0, 80)


def test_kaldi_fbank_silence_is_log_floor() -> None:
    feats = _kaldi_fbank(np.zeros(_SAMPLE_RATE, dtype=np.float32))
    expected = math.log(float(np.finfo(np.float32).eps))
    assert np.allclose(feats, expected)
    assert np.isfinite(feats).all()


def test_kaldi_fbank_is_deterministic_and_finite() -> None:
    rng = np.random.RandomState(0)
    audio = rng.randn(2 * _SAMPLE_RATE).astype(np.float32) * 0.1
    first = _kaldi_fbank(audio)
    second = _kaldi_fbank(audio)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(np.max(first)) > float(np.min(first))


def test_kaldi_fbank_tone_has_energy_above_silence() -> None:
    seconds = np.arange(_SAMPLE_RATE, dtype=np.float32) / _SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * seconds)).astype(np.float32)
    feats = _kaldi_fbank(tone)
    silence = _kaldi_fbank(np.zeros(_SAMPLE_RATE, dtype=np.float32))
    assert float(np.mean(feats)) > float(np.mean(silence))


# --- conteo de hablantes por frame ----------------------------------------------


def _classes(chunks: list[np.ndarray]) -> np.ndarray:
    return np.stack(chunks).astype(np.int8)


def test_speaker_count_silence_is_zero() -> None:
    classes = _classes([np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)])
    counts = speaker_count_per_frame(classes)
    assert counts.shape[0] > 0
    assert int(counts.max()) == 0


def test_speaker_count_one_slot_is_one() -> None:
    chunk = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk[:300] = 1  # slot 0 activo en la primera mitad
    counts = speaker_count_per_frame(_classes([chunk]))
    assert int(counts[100]) == 1
    assert int(counts[400]) == 0


def test_speaker_count_overlap_is_two() -> None:
    chunk = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk[:300] = 4  # slots 0 y 1 activos (clase powerset 4)
    counts = speaker_count_per_frame(_classes([chunk]))
    assert int(counts[100]) == 2


def test_speaker_count_empty_classes() -> None:
    counts = speaker_count_per_frame(np.empty((0, _FRAMES_PER_WINDOW), dtype=np.int8))
    assert counts.shape == (0,)


# --- clustering (funciones puras) -------------------------------------------------


def _group_embeddings(center: np.ndarray, count: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    points = center + noise * rng.randn(count, center.size)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def test_cluster_two_separated_groups() -> None:
    dim = 192
    e0 = np.zeros(dim)
    e0[0] = 1.0
    e1 = np.zeros(dim)
    e1[1] = 1.0
    group_a = _group_embeddings(e0, 8, 0.05, seed=1)
    group_b = _group_embeddings(e1, 8, 0.05, seed=2)
    labels = cluster_embeddings(np.vstack([group_a, group_b]))
    assert len(set(labels[:8].tolist())) == 1
    assert len(set(labels[8:].tolist())) == 1
    assert labels[0] != labels[8]


def test_cluster_single_and_empty() -> None:
    assert cluster_embeddings(np.zeros((1, 192))).tolist() == [0]
    assert cluster_embeddings(np.zeros((0, 192))).shape == (0,)


def test_cluster_subsample_assigns_by_centroid() -> None:
    dim = 192
    e0 = np.zeros(dim)
    e0[0] = 1.0
    e1 = np.zeros(dim)
    e1[1] = 1.0
    group_a = _group_embeddings(e0, 25, 0.05, seed=3)
    group_b = _group_embeddings(e1, 25, 0.05, seed=4)
    # max_cluster=10 fuerza la submuestra + asignación por centroide.
    labels = cluster_embeddings(np.vstack([group_a, group_b]), max_cluster=10)
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:25].tolist())) == 1
    assert len(set(labels[25:].tolist())) == 1
    assert labels[0] != labels[25]


def test_merge_similar_centroids_merges_only_when_close() -> None:
    dim = 192
    e0 = np.zeros(dim)
    e0[0] = 1.0
    # Centroide a sim 0,6 de e0.
    w = 0.6 * e0
    w[1] = 0.8
    group_a = _group_embeddings(e0, 4, 0.01, seed=5)
    group_b = _group_embeddings(w, 4, 0.01, seed=6)
    embeddings = np.vstack([group_a, group_b])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    merged = _merge_similar_centroids(embeddings, labels, 0.5)
    assert len(set(merged.tolist())) == 1
    kept = _merge_similar_centroids(embeddings, labels, 0.9)
    assert len(set(kept.tolist())) == 2


# --- construcción de segmentos (función pura) ---------------------------------------


def _two_slot_chunk(split: int = 295) -> np.ndarray:
    """Un chunk: slot 0 en [0, split), slot 1 en [split, 589)."""
    chunk = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk[:split] = 1
    chunk[split:] = 2
    return chunk


def test_build_segments_two_speakers_one_chunk() -> None:
    classes = _classes([_two_slot_chunk()])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(
        classes,
        [(0, 0), (0, 1)],
        np.array([0, 1], dtype=np.int32),
        _WINDOW_SIZE,
        spf,
    )
    assert len(segments) == 2
    assert segments[0].speaker == 0
    assert segments[0].start == pytest.approx(0.031, abs=0.02)
    assert segments[0].end == pytest.approx(5.009, abs=0.02)
    assert segments[1].speaker == 1
    assert segments[1].start == pytest.approx(5.009, abs=0.02)
    assert segments[1].end == pytest.approx(9.970, abs=0.02)


def test_build_segments_short_runs_are_dropped() -> None:
    # Un parpadeo de 5 frames (~0,08 s) del slot 1: bajo min_duration_on.
    chunk = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk[:] = 1
    chunk[300:305] = 2
    classes = _classes([chunk])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(
        classes,
        [(0, 0), (0, 1)],
        np.array([0, 1], dtype=np.int32),
        _WINDOW_SIZE,
        spf,
    )
    assert [s.speaker for s in segments] == [0]


def test_build_segments_small_gap_is_merged() -> None:
    # Slot 0 en dos tramos separados por 20 frames (~0,34 s < 0,5 s).
    chunk = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk[100:300] = 1
    chunk[320:500] = 1
    classes = _classes([chunk])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(classes, [(0, 0)], np.array([0], dtype=np.int32), _WINDOW_SIZE, spf)
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(100 * 270 / 16000 + 991 / 32000, abs=0.02)
    assert segments[0].end == pytest.approx(500 * 270 / 16000 + 991 / 32000, abs=0.02)


def test_build_segments_without_embeddings_is_single_speaker() -> None:
    classes = _classes([_two_slot_chunk()])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(classes, [], np.empty(0, dtype=np.int32), _WINDOW_SIZE, spf)
    assert len(segments) == 1
    assert segments[0].speaker == 0


def test_build_segments_inheritance_extends_tail() -> None:
    # Chunk 0 embebido; chunk 1 hereda (sus slots se solapan >= 50%
    # con los del chunk 0) y alarga el segmento del hablante 1 hasta
    # ~10,95 s; sin herencia terminaría en ~9,95 s.
    classes = _classes([_two_slot_chunk(), _two_slot_chunk()])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(
        classes,
        [(0, 0), (0, 1)],
        np.array([0, 1], dtype=np.int32),
        2 * _WINDOW_SHIFT + _WINDOW_SIZE,
        spf,
    )
    assert len(segments) == 2
    assert segments[1].speaker == 1
    assert segments[1].end == pytest.approx(10.95, abs=0.05)


def test_build_segments_no_inheritance_without_overlap() -> None:
    # Chunk 1 con los slots en tiempos disjuntos a los del chunk 0:
    # sin solape no hay herencia y los segmentos no se alargan.
    chunk1 = np.zeros(_FRAMES_PER_WINDOW, dtype=np.int8)
    chunk1[:400] = 2  # slot 1 en [0, 400)
    chunk1[400:] = 1  # slot 0 en [400, 589)
    classes = _classes([_two_slot_chunk(), chunk1])
    spf = speaker_count_per_frame(classes)
    segments = build_segments(
        classes,
        [(0, 0), (0, 1)],
        np.array([0, 1], dtype=np.int32),
        2 * _WINDOW_SHIFT + _WINDOW_SIZE,
        spf,
    )
    assert len(segments) == 2
    assert segments[1].end == pytest.approx(9.97, abs=0.05)


# --- inferencia en streaming (sesiones simuladas) ---------------------------------


class _FakeIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSegSession:
    """Sesión de segmentación simulada con patrón de clases fijo.

    ``pattern`` es una función frame -> clase; por defecto, silencio
    (clase 0) en todas partes.
    """

    def __init__(self, pattern: Any | None = None) -> None:
        self.pattern = pattern or (lambda frame: 0)
        self.calls: list[int] = []  # Ventanas por llamada.

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("x")]

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        windows = feed["x"]
        self.calls.append(int(windows.shape[0]))
        batch = windows.shape[0]
        logits = np.zeros((batch, _FRAMES_PER_WINDOW, 7), dtype=np.float32)
        for frame in range(_FRAMES_PER_WINDOW):
            logits[:, frame, self.pattern(frame)] = 1.0
        return [logits]


class _FakeEmbSession:
    """Sesión de embedding simulada: alterna dos vectores ortogonales.

    Segura para hilos: el diarizador corre los embeddings en un pool
    por defecto, y sin el lock dos llamadas concurrentes podrían leer
    el mismo contador y devolver el mismo vector.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("x")]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO("embedding")]

    def run(self, _outputs: list[str], _feed: dict[str, Any]) -> list[np.ndarray]:
        with self._lock:
            emb = np.zeros((1, 192), dtype=np.float32)
            emb[0, self.calls % 2] = 1.0
            self.calls += 1
        return [emb]


def test_streaming_matches_single_feed() -> None:
    seg = _FakeSegSession()
    emb = _FakeEmbSession()
    total = 130 * _SAMPLE_RATE
    audio = np.zeros(total, dtype=np.float32)
    streamer = _StreamingDiarizer(seg, emb)
    cursor = 0
    for read_seconds in (5, 5, 17, 3, 60, 40):
        chunk = audio[cursor : cursor + read_seconds * _SAMPLE_RATE]
        cursor += chunk.size
        streamer.feed(chunk)
    classes_chunked, _, _, _ = streamer.finish()
    assert cursor == total
    streamer2 = _StreamingDiarizer(_FakeSegSession(), _FakeEmbSession())
    streamer2.feed(audio)
    classes_single, _, _, _ = streamer2.finish()
    assert classes_chunked.shape == classes_single.shape
    assert np.array_equal(classes_chunked, classes_single)
    # 130 s: (2080000-160000)//16000+1 = 121 chunks, sin cola.
    assert classes_chunked.shape == (121, _FRAMES_PER_WINDOW)


def test_streaming_short_audio_is_a_single_padded_window() -> None:
    seg = _FakeSegSession()
    emb = _FakeEmbSession()
    streamer = _StreamingDiarizer(seg, emb)
    streamer.feed(np.zeros(3 * _SAMPLE_RATE, dtype=np.float32))
    classes, pairs, embeddings, total = streamer.finish()
    assert classes.shape == (1, _FRAMES_PER_WINDOW)
    assert pairs == []
    assert embeddings.shape == (0, 192)
    assert total == 3 * _SAMPLE_RATE
    assert seg.calls == [1]


def test_streaming_empty_audio() -> None:
    seg = _FakeSegSession()
    emb = _FakeEmbSession()
    streamer = _StreamingDiarizer(seg, emb)
    streamer.feed(np.zeros(0, dtype=np.float32))
    classes, pairs, embeddings, total = streamer.finish()
    assert classes.shape == (0, _FRAMES_PER_WINDOW)
    assert pairs == []
    assert total == 0
    assert seg.calls == []


def test_streaming_embeds_only_stride_chunks() -> None:
    # Slot 0 activo en la primera mitad de cada ventana.
    seg = _FakeSegSession(pattern=lambda frame: 1 if frame < 295 else 0)
    emb = _FakeEmbSession()
    streamer = _StreamingDiarizer(seg, emb)
    streamer.feed(np.zeros(12 * _SAMPLE_RATE, dtype=np.float32))
    _classes, pairs, embeddings, _total = streamer.finish()
    # 12 s: 3 chunks; con stride 2 se embeben el 0 y el 2.
    assert pairs == [(0, 0), (2, 0)]
    assert embeddings.shape == (2, 192)
    assert emb.calls == 2


# --- descarga bajo demanda (urllib simulado) ---------------------------------------


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.length = len(data)
        self._pos = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def _make_tarball_bytes(member: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def _fake_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Contenido falso + constantes parcheadas para los tests de descarga."""
    seg_content = b"modelo-segmentacion-falso"
    emb_content = b"modelo-embedding-falso"
    tarball = _make_tarball_bytes("sherpa-onnx-pyannote-segmentation-3-0/model.onnx", seg_content)
    monkeypatch.setattr(diarization, "_SEGMENTATION_TARBALL_SIZE", len(tarball))
    monkeypatch.setattr(
        diarization,
        "_SEGMENTATION_TARBALL_SHA256",
        hashlib.sha256(tarball).hexdigest(),
    )
    monkeypatch.setattr(diarization, "_SEGMENTATION_SIZE", len(seg_content))
    monkeypatch.setattr(
        diarization,
        "_SEGMENTATION_SHA256",
        hashlib.sha256(seg_content).hexdigest(),
    )
    monkeypatch.setattr(diarization, "_EMBEDDING_SIZE", len(emb_content))
    monkeypatch.setattr(diarization, "_EMBEDDING_SHA256", hashlib.sha256(emb_content).hexdigest())
    return {"tarball": tarball, "seg": seg_content, "emb": emb_content}


def _urlopen_dispatch(files: dict[str, Any], calls: list[str]) -> Any:
    def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        if url.endswith(".tar.bz2"):
            return _FakeResponse(files["tarball"])
        return _FakeResponse(files["emb"])

    return fake_urlopen


def test_download_success_reports_progress_and_returns_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_dispatch(_fake_models, calls))
    progress: list[float] = []
    paths = diarization.ensure_diarization_models(progress.append, cache_dir=tmp_path)
    assert paths["segmentation"].read_bytes() == _fake_models["seg"]
    assert paths["embedding"].read_bytes() == _fake_models["emb"]
    # El tarball se borra tras extraer.
    assert not (tmp_path / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2").exists()
    assert progress[0] == 0
    assert progress[-1] == 100
    assert any(0 < value < 100 for value in progress)
    assert len(calls) == 2


def test_download_cached_files_skip_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(_fake_models["seg"])
    (tmp_path / "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx").write_bytes(
        _fake_models["emb"]
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("En caché: no debe descargar nada")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)
    progress: list[float] = []
    paths = diarization.ensure_diarization_models(progress.append, cache_dir=tmp_path)
    assert paths["segmentation"].is_file()
    assert progress == [0, 100]


def test_download_sha_mismatch_rejected_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    calls: list[str] = []
    files = dict(_fake_models)
    files["emb"] = b"contenido-corrupto"

    monkeypatch.setattr("urllib.request.urlopen", _urlopen_dispatch(files, calls))
    with pytest.raises(DiarizationError, match="sha256"):
        diarization.ensure_diarization_models(cache_dir=tmp_path)
    # El tarball (1) + el embedding corrupto (1, sin reintento).
    assert len(calls) == 2


def test_download_failure_raises_after_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def failing_urlopen(*args: Any, **kwargs: Any) -> Any:
        attempts.append(1)
        raise OSError("sin red")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    with pytest.raises(DiarizationError, match="No se pudo descargar"):
        diarization.ensure_diarization_models(cache_dir=tmp_path)
    assert len(attempts) == 2


def test_download_retry_then_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    calls: list[str] = []
    state = {"failed": False}

    def flaky_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        calls.append("x")
        if not state["failed"]:
            state["failed"] = True
            raise OSError("corte momentáneo")
        return _urlopen_dispatch(_fake_models, [])(request, timeout)

    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    paths = diarization.ensure_diarization_models(cache_dir=tmp_path)
    assert paths["embedding"].is_file()
    assert len(calls) == 3  # 1 fallo + tarball + embedding


def test_download_cancelled_raises_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def fake_urlopen(*args: Any, **kwargs: Any) -> _FakeResponse:
        attempts.append(1)
        return _FakeResponse(b"x" * 100)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(DiarizationCancelled):
        diarization.ensure_diarization_models(cache_dir=tmp_path, should_cancel=lambda: True)
    assert len(attempts) == 1


def test_download_corrupt_tarball_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    files = dict(_fake_models)
    files["tarball"] = b"esto-no-es-un-tarball"
    monkeypatch.setattr(
        diarization,
        "_SEGMENTATION_TARBALL_SIZE",
        len(files["tarball"]),
    )
    monkeypatch.setattr(
        diarization,
        "_SEGMENTATION_TARBALL_SHA256",
        hashlib.sha256(files["tarball"]).hexdigest(),
    )
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_dispatch(files, []))
    with pytest.raises(DiarizationError, match="tarball"):
        diarization.ensure_diarization_models(cache_dir=tmp_path)


# --- fallback (get_speaker_timeline nunca rompe un proyecto) ------------------------


def test_get_speaker_timeline_falls_back_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("sin red")),
    )
    assert diarization.get_speaker_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_get_speaker_timeline_falls_back_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = {
        "segmentation": tmp_path / "seg.onnx",
        "embedding": tmp_path / "emb.onnx",
    }
    monkeypatch.setattr(diarization, "ensure_diarization_models", lambda *a, **k: paths)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert diarization.get_speaker_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_load_sessions_without_onnxruntime_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(DiarizationError, match="onnxruntime"):
        diarization.load_diarization_sessions(
            {"segmentation": tmp_path / "s.onnx", "embedding": tmp_path / "e.onnx"}
        )


def test_get_speaker_timeline_disabled_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Diarización desactivada: no debe descargar nada")

    monkeypatch.setattr(diarization, "ensure_diarization_models", fail_if_called)
    assert diarization.get_speaker_timeline("video.mp4", 60.0, "ffmpeg", enabled=False) is None


def test_get_speaker_timeline_without_ffmpeg_returns_none() -> None:
    assert diarization.get_speaker_timeline("video.mp4", 60.0, None) is None
    assert diarization.get_speaker_timeline(None, 60.0, "ffmpeg") is None


def test_get_speaker_timeline_swallows_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("fallo inesperado")

    monkeypatch.setattr(diarization, "detect_speakers", boom)
    assert diarization.get_speaker_timeline("video.mp4", 60.0, "ffmpeg") is None


def test_get_speaker_timeline_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cancelled(*args: Any, **kwargs: Any) -> Any:
        raise DiarizationCancelled("cancelado")

    monkeypatch.setattr(diarization, "detect_speakers", cancelled)
    with pytest.raises(DiarizationCancelled):
        diarization.get_speaker_timeline("video.mp4", 60.0, "ffmpeg")


def test_is_diarization_available_false_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert diarization.is_diarization_available() is False


def test_is_diarization_available_checks_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fake_models: dict[str, Any],
) -> None:
    import types

    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    assert diarization.is_diarization_available(cache_dir=tmp_path) is False
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(_fake_models["seg"])
    (tmp_path / "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx").write_bytes(
        _fake_models["emb"]
    )
    assert diarization.is_diarization_available(cache_dir=tmp_path) is True
    # Un archivo corrupto invalida la caché.
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"corrupto")
    assert diarization.is_diarization_available(cache_dir=tmp_path) is False


# --- extracción con ffmpeg (sesiones simuladas, tono real) --------------------------


def _make_tone_wav(path: Path, seconds: int = 12) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds},volume=0.5",
            "-y",
            str(path),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def test_detect_speakers_silence_pattern_gives_empty_result(tmp_path: Path) -> None:
    wav_path = tmp_path / "tono.wav"
    _make_tone_wav(wav_path, seconds=12)
    result = diarization.detect_speakers(
        wav_path, 12.0, "ffmpeg", sessions=(_FakeSegSession(), _FakeEmbSession())
    )
    assert result.timeline.shape == (12,)
    assert float(np.max(result.timeline)) == 0.0
    assert result.segments == []
    assert result.num_speakers == 0


def test_detect_speakers_two_slot_pattern_gives_two_speakers(tmp_path: Path) -> None:
    wav_path = tmp_path / "tono.wav"
    _make_tone_wav(wav_path, seconds=12)
    seg = _FakeSegSession(pattern=lambda frame: 1 if frame < 295 else 2)
    result = diarization.detect_speakers(
        wav_path, 12.0, "ffmpeg", sessions=(seg, _FakeEmbSession())
    )
    assert result.num_speakers == 2
    assert len(result.segments) >= 2
    assert float(np.max(result.timeline)) > 0.0


def test_detect_speakers_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DiarizationError):
        diarization.detect_speakers(
            tmp_path / "no-existe.mp4",
            10.0,
            "ffmpeg",
            sessions=(_FakeSegSession(), _FakeEmbSession()),
        )


# --- integración con los modelos reales (salta sin modelos ni runtime) -------------


def _real_model_paths() -> dict[str, Path] | None:
    """Rutas de los modelos reales si KELLYCLIPAI_DIARIZATION_TEST_MODELS existe."""
    models_dir = os.environ.get("KELLYCLIPAI_DIARIZATION_TEST_MODELS", "")
    if not models_dir:
        return None
    root = Path(models_dir)
    paths = diarization._model_paths(root)
    if not paths["segmentation"].is_file() or not paths["embedding"].is_file():
        return None
    return paths


def test_real_diarization_model_shapes() -> None:
    """Los .onnx reales cargan y tienen las entradas/salidas esperadas."""
    paths = _real_model_paths()
    if paths is None:
        pytest.skip("modelos de diarización reales no disponibles")
    ort = pytest.importorskip("onnxruntime")
    seg = ort.InferenceSession(str(paths["segmentation"]), providers=["CPUExecutionProvider"])
    assert seg.get_inputs()[0].name == "x"
    zeros = np.zeros((1, 1, _WINDOW_SIZE), dtype=np.float32)
    (out,) = seg.run(None, {"x": zeros})
    assert np.asarray(out).shape == (1, _FRAMES_PER_WINDOW, 7)
    emb = ort.InferenceSession(str(paths["embedding"]), providers=["CPUExecutionProvider"])
    assert emb.get_inputs()[0].name == "x"
    feats = np.zeros((1, 10, 80), dtype=np.float32)
    (vec,) = emb.run(["embedding"], {"x": feats})
    assert np.asarray(vec).shape == (1, 192)


def test_real_diarization_two_speakers() -> None:
    """Diarización completa sobre el wav de test de sherpa (2 hablantes).

    Reproduce el comportamiento verificado el 21-11-2026 contra el
    oráculo sherpa_onnx: 4 segmentos con los mismos bordes (±0,2 s).
    """
    paths = _real_model_paths()
    wav = os.environ.get("KELLYCLIPAI_DIARIZATION_TEST_WAV", "")
    if paths is None or not wav or not Path(wav).is_file():
        pytest.skip("modelos o wav de test de diarización no disponibles")
    pytest.importorskip("onnxruntime")
    found = diarization.ensure_diarization_models(cache_dir=Path(paths["segmentation"]).parent)
    assert found == paths  # Vía rápida: ya en caché y válidos.
    sessions = diarization.load_diarization_sessions(paths)
    result = diarization.detect_speakers(Path(wav), 16.0, "ffmpeg", sessions=sessions)
    assert result.num_speakers == 2
    assert len(result.segments) == 4
    expected = [(1.58, 3.41, 0), (4.40, 6.46, 0), (9.35, 11.47, 1), (12.16, 14.64, 1)]
    for segment, (start, end, speaker) in zip(result.segments, expected, strict=True):
        assert segment.start == pytest.approx(start, abs=0.2)
        assert segment.end == pytest.approx(end, abs=0.2)
        assert segment.speaker == speaker
    assert result.timeline.shape == (16,)
    assert float(np.max(result.timeline)) == pytest.approx(0.25)


def test_real_diarization_tone_is_no_speech(tmp_path: Path) -> None:
    """Un tono puro con los modelos reales: sin habla, resultado vacío."""
    paths = _real_model_paths()
    if paths is None:
        pytest.skip("modelos de diarización reales no disponibles")
    pytest.importorskip("onnxruntime")
    wav_path = tmp_path / "tono.wav"
    _make_tone_wav(wav_path, seconds=12)
    sessions = diarization.load_diarization_sessions(paths)
    result = diarization.detect_speakers(wav_path, 12.0, "ffmpeg", sessions=sessions)
    assert result.num_speakers == 0
    assert result.segments == []
    assert float(np.max(result.timeline)) == 0.0


# --- embeddings en paralelo (v1.3.1): mismo resultado que en serie -----------

class _DeterministicFakeEmbSession:
    """Embedding derivado solo del feed: misma entrada -> mismo vector.

    El fake alternante (_FakeEmbSession) depende del orden de llamada,
    que con el pool no es determinista; este permite comparar en serie
    y en paralelo sin flakiness.
    """

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("x")]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO("embedding")]

    def run(self, _outputs: list[str], feed: dict[str, Any]) -> list[np.ndarray]:
        feats = np.asarray(feed["x"], dtype=np.float64)
        emb = np.zeros((1, 192), dtype=np.float32)
        emb[0, 0] = float(feats.mean())
        emb[0, 1] = float(feats.std())
        return [emb]


def test_parallel_embeddings_match_sequential() -> None:
    """embed_workers=4 da las mismas clases, pares y embeddings que =1."""
    rng = np.random.default_rng(7)
    audio = rng.normal(0, 0.1, 90 * _SAMPLE_RATE).astype(np.float32)
    seg_seq = _FakeSegSession(pattern=lambda frame: frame % 3)
    diar_seq = _StreamingDiarizer(seg_seq, _DeterministicFakeEmbSession(), embed_workers=1)
    diar_seq.feed(audio)
    classes_seq, pairs_seq, embs_seq, total_seq = diar_seq.finish()
    seg_par = _FakeSegSession(pattern=lambda frame: frame % 3)
    diar_par = _StreamingDiarizer(seg_par, _DeterministicFakeEmbSession(), embed_workers=4)
    diar_par.feed(audio)
    classes_par, pairs_par, embs_par, total_par = diar_par.finish()
    assert total_seq == total_par
    assert np.array_equal(classes_seq, classes_par)
    assert pairs_seq == pairs_par
    assert embs_seq.shape == embs_par.shape
    assert np.array_equal(embs_seq, embs_par)
    assert len(pairs_seq) > 0  # El test ejercita de verdad el pool.


def test_diarizer_close_is_idempotent() -> None:
    diar = _StreamingDiarizer(_FakeSegSession(), _FakeEmbSession())
    diar.feed(np.zeros(20 * _SAMPLE_RATE, dtype=np.float32))
    diar.close()
    diar.close()  # Sin error la segunda vez.
