"""Tests de la puntuación semántica Tier 2 (Fase D, v1.2.0): MiniLM ONNX.

Todo con mocks o un vocabulario sintético, salvo los tests de
integración que usan el modelo real si está en la caché de Hugging
Face (``is_semantic_available()``): en CI se saltan. Sin cv2, sin
PyTorch, sin sentence-transformers. numpy es requisito del motor (ver
tests/test_moments.py); sin numpy, el módulo entero se salta.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from backend import semantic  # noqa: E402
from backend.semantic import (  # noqa: E402
    DEFAULT_PARAPHRASE_THRESHOLD,
    EMBEDDING_DIM,
    LazyEmbeddings,
    SemanticCancelled,
    SemanticError,
    _load_tokenizer,
    _vocab_path_for_model,
    _WordPieceTokenizer,
    embed_texts,
    find_paraphrase_duplicates,
    preload_minilm_model,
    semantic_similarity,
)

# Vocabulario sintético mínimo (índice = línea): cubre los casos del
# tokenizador sin necesitar el vocab.txt real (30.522 tokens).
_TINY_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "una",
    "##ffa",
    "##ble",
    "hola",
    "me",
    "en",
    "##can",
    "##ta",
    "este",
    "lu",
    "##gar",
    "!",
    "?",
    ",",
    "don",
    "##de",
    "est",
    "##a",
    "la",
    "can",
    "##cion",
    "¿",
    "你",
    "好",
]

# Pares de calibración realista en español (medidos el 21-11-2026 con
# el modelo Xenova; los tests de integración exigen márgenes amplios
# por debajo de lo medido, no los valores exactos).
_PARA_A = (
    "El presupuesto del próximo año va a ser complicado porque tenemos que recortar "
    "gastos en varias áreas y lo más importante es no tocar la parte de educación."
)
_PARA_B = (
    "El presupuesto del año que viene será difícil porque hay que recortar gastos "
    "en varias áreas, pero lo más importante es no tocar educación."
)  # Medido: 0,943.
_DIFF_A = (
    "El presupuesto del próximo año va a ser complicado porque tenemos que recortar "
    "gastos en varias áreas y lo más importante es no tocar la parte de educación."
)
_DIFF_B = (
    "La reunión de hoy fue sobre el horario de verano y los turnos de vacaciones. "
    "Al final quedamos en que cada equipo propone sus fechas antes del viernes."
)  # Medido: 0,455.
_NEAR_DUP_A = (
    "Bueno, entonces llegamos al restaurante y pedimos la especialidad de la casa, "
    "que era un pescado frito con tostones."
)
_NEAR_DUP_B = (
    "Bueno entonces llegamos al restaurante y pedimos la especialidad de la casa, "
    "que era un pescado frito con tostones"
)  # Medido: 0,972.


@pytest.fixture(autouse=True)
def _clean_caches() -> Any:
    semantic._reset_caches()
    yield
    semantic._reset_caches()


@pytest.fixture(autouse=True)
def _no_download_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantic, "_MINILM_RETRY_DELAY_SECONDS", 0.0)


@pytest.fixture()
def tiny_vocab(tmp_path: Path) -> Path:
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(_TINY_VOCAB) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def tiny_tokenizer(tiny_vocab: Path) -> _WordPieceTokenizer:
    return _WordPieceTokenizer(tiny_vocab)


def _require_real_model() -> None:
    if not semantic.is_semantic_available():
        pytest.skip("modelo MiniLM real no disponible en caché")


# --- el módulo no exige onnxruntime para importarse ---------------------------


def test_import_semantic_does_not_import_onnxruntime() -> None:
    """Importar semantic no debe importar onnxruntime (perezoso).

    Se comprueba en un intérprete nuevo: en la suite, otros tests ya
    pueden haber importado onnxruntime, y lo que importa es que el
    import en limpio no lo arrastre.
    """
    code = (
        "import sys\n"
        "from backend import semantic\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "assert semantic._MINILM_REPO_ID == 'Xenova/all-MiniLM-L6-v2'\n"
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


# --- tokenizador WordPiece (vocabulario sintético) ----------------------------


def test_tokenizer_matches_canonical_bert_example(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("unaffable") == ["una", "##ffa", "##ble"]


def test_tokenizer_spanish_sentence(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("Me encanta este lugar!") == [
        "me",
        "en",
        "##can",
        "##ta",
        "este",
        "lu",
        "##gar",
        "!",
    ]


def test_tokenizer_strips_accents_and_splits_punctuation(
    tiny_tokenizer: _WordPieceTokenizer,
) -> None:
    # "dónde" -> "don" + "##de", "está" -> "est" + "##a": sin quitar los
    # acentos, ambas palabras serían [UNK] (el vocabulario BERT uncased
    # no tiene ningún token con acento latino).
    assert tiny_tokenizer.tokenize("¿Dónde está la canción?") == [
        "¿",
        "don",
        "##de",
        "est",
        "##a",
        "la",
        "can",
        "##cion",
        "?",
    ]


def test_tokenizer_unknown_word_is_unk(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("zzq") == ["[UNK]"]


def test_tokenizer_word_over_100_chars_is_unk(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("x" * 101) == ["[UNK]"]


def test_tokenizer_cjk_chars_are_split(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("你好") == ["你", "好"]


def test_tokenizer_empty_text(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert tiny_tokenizer.tokenize("") == []
    assert tiny_tokenizer.tokenize("   ") == []


def test_tokenizer_len(tiny_tokenizer: _WordPieceTokenizer) -> None:
    assert len(tiny_tokenizer) == len(_TINY_VOCAB)


def test_tokenizer_missing_special_token_raises(tmp_path: Path) -> None:
    path = tmp_path / "vocab.txt"
    path.write_text("[PAD]\n[UNK]\n[CLS]\n", encoding="utf-8")  # Sin [SEP].
    with pytest.raises(SemanticError, match="SEP"):
        _WordPieceTokenizer(path)


def test_encode_adds_special_tokens_and_pads(tiny_tokenizer: _WordPieceTokenizer) -> None:
    input_ids, mask = tiny_tokenizer.encode(["hola", "hola!"])
    # [CLS]=2, hola=8, [SEP]=3, [PAD]=0, "!"=16.
    assert input_ids.tolist() == [[2, 8, 3, 0], [2, 8, 16, 3]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert input_ids.dtype == np.int64


def test_encode_truncates_to_256_keeping_sep(tiny_tokenizer: _WordPieceTokenizer) -> None:
    input_ids, mask = tiny_tokenizer.encode(["hola " * 300])
    assert input_ids.shape == (1, 256)
    assert int(input_ids[0, -1]) == 3  # [SEP] siempre al final.
    assert int(mask.sum()) == 256


def test_encode_empty_text_is_cls_sep(tiny_tokenizer: _WordPieceTokenizer) -> None:
    input_ids, mask = tiny_tokenizer.encode([""])
    assert input_ids.tolist() == [[2, 3]]
    assert mask.tolist() == [[1, 1]]


def test_vocab_path_for_model() -> None:
    model = Path("/cache/snapshots/abc/onnx/model_quantized.onnx")
    assert _vocab_path_for_model(model) == Path("/cache/snapshots/abc/vocab.txt")


# --- sesión ONNX simulada: pooling, lotes, formas ------------------------------


class _FakeIO:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Sesión ONNX simulada: hidden 2,0 en tokens reales, 7,0 en relleno.

    Si el mean pooling ignorase la máscara, el 7,0 del relleno
    contaminaría el resultado; si la respeta, cada componente del
    embedding es 1/sqrt(384).
    """

    def __init__(
        self,
        input_names: tuple[str, ...] = ("input_ids", "attention_mask", "token_type_ids"),
        hidden_dim: int = EMBEDDING_DIM,
    ) -> None:
        self.input_names = input_names
        self.hidden_dim = hidden_dim
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO(name) for name in self.input_names]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO("last_hidden_state")]

    def run(self, _outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        mask = feeds["attention_mask"]
        hidden = np.where(mask[:, :, None] == 1, 2.0, 7.0).astype(np.float32)
        hidden = np.repeat(hidden, self.hidden_dim, axis=2)
        return [hidden]


def test_embed_mean_pool_ignores_padding(
    tiny_tokenizer: _WordPieceTokenizer,
) -> None:
    session = _FakeSession()
    embeddings = embed_texts(["hola", "hola!"], session=session, tokenizer=tiny_tokenizer)
    assert embeddings is not None
    assert embeddings.shape == (2, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    assert embeddings == pytest.approx(1.0 / math.sqrt(EMBEDDING_DIM), abs=1e-6)
    norms = np.linalg.norm(embeddings, axis=1)
    assert norms == pytest.approx(1.0, abs=1e-6)


def test_embed_feeds_token_type_ids_as_zeros(tiny_tokenizer: _WordPieceTokenizer) -> None:
    session = _FakeSession()
    embed_texts(["hola"], session=session, tokenizer=tiny_tokenizer)
    feeds = session.feeds[0]
    assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}
    assert np.all(feeds["token_type_ids"] == 0)
    assert feeds["token_type_ids"].shape == feeds["input_ids"].shape


def test_embed_skips_token_type_ids_when_model_lacks_them(
    tiny_tokenizer: _WordPieceTokenizer,
) -> None:
    session = _FakeSession(input_names=("input_ids", "attention_mask"))
    embeddings = embed_texts(["hola"], session=session, tokenizer=tiny_tokenizer)
    assert embeddings is not None
    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_embed_batches_in_32s(tiny_tokenizer: _WordPieceTokenizer) -> None:
    session = _FakeSession()
    texts = ["hola"] * 40
    embeddings = embed_texts(texts, session=session, tokenizer=tiny_tokenizer)
    assert embeddings is not None
    assert embeddings.shape == (40, EMBEDDING_DIM)
    assert [feed["input_ids"].shape[0] for feed in session.feeds] == [32, 8]


def test_embed_reports_progress_50_to_100(tiny_tokenizer: _WordPieceTokenizer) -> None:
    session = _FakeSession()
    calls: list[float] = []
    embed_texts(["hola"] * 40, on_progress=calls.append, session=session, tokenizer=tiny_tokenizer)
    # Sin descarga (sesión inyectada), el progreso cubre solo el
    # embedding: 50-100.
    assert calls[0] == pytest.approx(75.0)  # Primer lote: 50 + 50*1/2.
    assert calls[-1] == 100.0


def test_embed_wrong_output_dim_returns_none(tiny_tokenizer: _WordPieceTokenizer) -> None:
    session = _FakeSession(hidden_dim=128)
    assert embed_texts(["hola"], session=session, tokenizer=tiny_tokenizer) is None


def test_embed_unexpected_session_error_returns_none(
    tiny_tokenizer: _WordPieceTokenizer,
) -> None:
    class _BoomSession(_FakeSession):
        def run(self, _outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            raise RuntimeError("fallo inesperado")

    assert embed_texts(["hola"], session=_BoomSession(), tokenizer=tiny_tokenizer) is None


def test_embed_empty_list_returns_empty_array(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("sin textos no debe descargar nada")

    monkeypatch.setattr(semantic, "ensure_minilm_model", fail_if_called)
    result = embed_texts([])
    assert result is not None
    assert result.shape == (0, EMBEDDING_DIM)
    assert result.dtype == np.float32


# --- fallback: sin modelo, sin runtime, cancelación ----------------------------


def test_embed_returns_none_when_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        semantic,
        "ensure_minilm_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(SemanticError("sin red")),
    )
    assert embed_texts(["hola"]) is None


def test_embed_returns_none_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model_quantized.onnx"
    model_path.write_bytes(b"irrelevante")
    monkeypatch.setattr(semantic, "ensure_minilm_model", lambda *a, **k: model_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert embed_texts(["hola"]) is None


def test_load_session_without_onnxruntime_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(SemanticError, match="onnxruntime"):
        semantic.load_minilm_session(tmp_path / "model_quantized.onnx")


def test_load_session_with_invalid_file_raises(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    bad = tmp_path / "model_quantized.onnx"
    bad.write_bytes(b"esto no es un modelo onnx")
    with pytest.raises(SemanticError, match="No se pudo cargar"):
        semantic.load_minilm_session(bad)


def test_embed_propagates_cancellation_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cancelado antes de empezar: no debe descargar nada")

    monkeypatch.setattr(semantic, "ensure_minilm_model", fail_if_called)
    with pytest.raises(SemanticCancelled):
        embed_texts(["hola"], should_cancel=lambda: True)


def test_embed_propagates_cancellation_between_batches(
    tiny_tokenizer: _WordPieceTokenizer,
) -> None:
    session = _FakeSession()
    checks: list[int] = []

    def should_cancel() -> bool:
        checks.append(1)
        return len(checks) >= 3  # False, False, True: cancela en el 2º lote.

    with pytest.raises(SemanticCancelled):
        embed_texts(
            ["hola"] * 40,
            should_cancel=should_cancel,
            session=session,
            tokenizer=tiny_tokenizer,
        )
    assert len(session.feeds) == 1  # Solo el primer lote llegó a correr.


# --- descarga (simulada) -------------------------------------------------------


def _fake_snapshot(
    tmp_path: Path, model_bytes: bytes = b"modelo-falso", vocab_bytes: bytes = b"vocab-falso"
) -> tuple[Path, Path, Path]:
    root = tmp_path / "snapshot"
    model_path = root / "onnx" / "model_quantized.onnx"
    vocab_path = root / "vocab.txt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(model_bytes)
    vocab_path.write_bytes(vocab_bytes)
    return root, model_path, vocab_path


def _patch_fake_hashes(monkeypatch: pytest.MonkeyPatch, model_path: Path, vocab_path: Path) -> None:
    monkeypatch.setattr(
        semantic, "_MINILM_SHA256", hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(semantic, "_MINILM_SIZE_BYTES", model_path.stat().st_size)
    monkeypatch.setattr(
        semantic, "_VOCAB_SHA256", hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(semantic, "_VOCAB_SIZE_BYTES", vocab_path.stat().st_size)


def test_download_success_reports_progress_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, model_path, vocab_path = _fake_snapshot(tmp_path)
    _patch_fake_hashes(monkeypatch, model_path, vocab_path)
    seen: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> str:
        seen.update(kwargs)
        progress = kwargs["tqdm_class"](total=1000)
        progress.update(500)
        progress.update(500)
        progress.close()
        return str(root)

    monkeypatch.setattr(semantic, "snapshot_download", fake_download)
    calls: list[float] = []
    result = semantic.ensure_minilm_model(calls.append, cache_dir=tmp_path / "cache")
    assert result == model_path
    assert seen["repo_id"] == "Xenova/all-MiniLM-L6-v2"
    assert seen["allow_patterns"] == ["onnx/model_quantized.onnx", "vocab.txt"]
    assert seen["cache_dir"] == str(tmp_path / "cache")
    assert calls[0] == 0
    assert calls[-1] == 100
    assert any(0 < value < 100 for value in calls)


def test_download_failure_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def failing_download(**kwargs: Any) -> str:
        attempts.append(1)
        raise RuntimeError("sin red")

    monkeypatch.setattr(semantic, "snapshot_download", failing_download)
    with pytest.raises(SemanticError, match="No se pudo descargar"):
        semantic.ensure_minilm_model()
    assert len(attempts) == 2


def test_download_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, model_path, vocab_path = _fake_snapshot(tmp_path)
    _patch_fake_hashes(monkeypatch, model_path, vocab_path)
    attempts: list[int] = []

    def flaky_download(**kwargs: Any) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("corte momentáneo")
        return str(root)

    monkeypatch.setattr(semantic, "snapshot_download", flaky_download)
    assert semantic.ensure_minilm_model() == model_path
    assert len(attempts) == 2


def test_sha_mismatch_rejected_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, model_path, vocab_path = _fake_snapshot(tmp_path, model_bytes=b"contenido-corrupto")
    # Tamaños correctos para llegar a la comprobación de sha256; el sha
    # no coincide con el archivo corrupto.
    _patch_fake_hashes(monkeypatch, model_path, vocab_path)
    monkeypatch.setattr(semantic, "_MINILM_SHA256", "0" * 64)
    attempts: list[int] = []

    def fake_download(**kwargs: Any) -> str:
        attempts.append(1)
        return str(root)

    monkeypatch.setattr(semantic, "snapshot_download", fake_download)
    with pytest.raises(SemanticError, match="sha256"):
        semantic.ensure_minilm_model()
    assert len(attempts) == 1  # Reintentar devolvería el mismo archivo corrupto.


def test_size_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, model_path, vocab_path = _fake_snapshot(tmp_path)
    _patch_fake_hashes(monkeypatch, model_path, vocab_path)
    monkeypatch.setattr(semantic, "_MINILM_SIZE_BYTES", 999999)

    def fake_download(**kwargs: Any) -> str:
        return str(root)

    monkeypatch.setattr(semantic, "snapshot_download", fake_download)
    with pytest.raises(SemanticError, match="bytes"):
        semantic.ensure_minilm_model()


def test_missing_vocab_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "snapshot"
    model_path = root / "onnx" / "model_quantized.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"modelo-falso")
    monkeypatch.setattr(
        semantic, "_MINILM_SHA256", hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(semantic, "_MINILM_SIZE_BYTES", model_path.stat().st_size)

    def fake_download(**kwargs: Any) -> str:
        return str(root)

    monkeypatch.setattr(semantic, "snapshot_download", fake_download)
    with pytest.raises(SemanticError, match="vocab.txt"):
        semantic.ensure_minilm_model()


def test_cancelled_download_raises_and_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_download(**kwargs: Any) -> str:
        attempts.append(1)
        progress = kwargs["tqdm_class"](total=1000, disable=True)
        progress.update(10)
        return "/no-importa"

    monkeypatch.setattr(semantic, "snapshot_download", fake_download)
    with pytest.raises(SemanticCancelled):
        semantic.ensure_minilm_model(should_cancel=lambda: True)
    assert len(attempts) == 1


# --- disponibilidad ------------------------------------------------------------


def test_is_semantic_available_false_without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert semantic.is_semantic_available() is False


def test_is_semantic_available_checks_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    seen: list[dict[str, Any]] = []

    def fake_hub_download(**kwargs: Any) -> str:
        seen.append(kwargs)
        return "/cache/archivo"

    monkeypatch.setattr(semantic, "hf_hub_download", fake_hub_download)
    assert semantic.is_semantic_available() is True
    assert [call["filename"] for call in seen] == ["onnx/model_quantized.onnx", "vocab.txt"]
    assert all(call["local_files_only"] is True for call in seen)
    assert all(call["repo_id"] == "Xenova/all-MiniLM-L6-v2" for call in seen)
    monkeypatch.setattr(
        semantic,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no en caché")),
    )
    assert semantic.is_semantic_available() is False


def test_is_semantic_available_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setattr(
        semantic,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fallo raro")),
    )
    assert semantic.is_semantic_available() is False


# --- similitud y pares de paráfrasis (vectores sintéticos) ---------------------


def test_semantic_similarity_values() -> None:
    a = np.array([1.0, 0.0, 0.0])
    assert semantic_similarity(a, a) == pytest.approx(1.0)
    assert semantic_similarity(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert semantic_similarity(a, -a) == pytest.approx(-1.0)
    assert semantic_similarity(a, np.array([0.0, 0.0, 0.0])) == 0.0
    # Sin normalizar también funciona (coseno real, no producto punto).
    assert semantic_similarity(np.array([2.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_semantic_similarity_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="formas distintas"):
        semantic_similarity(np.zeros(3), np.zeros(4))


def test_find_paraphrase_duplicates_basic() -> None:
    # v0 y v1 con coseno 0,95; v2 ortogonal a v0.
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.95, math.sqrt(1 - 0.95**2), 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    embeddings = np.stack([v0, v1, v2])
    pairs = find_paraphrase_duplicates(embeddings)
    assert len(pairs) == 1
    i, j, similarity = pairs[0]
    assert (i, j) == (0, 1)
    assert similarity == pytest.approx(0.95)
    assert find_paraphrase_duplicates(embeddings, threshold=0.99) == []


def test_find_paraphrase_duplicates_accepts_unnormalized() -> None:
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.95, math.sqrt(1 - 0.95**2), 0.0]) * 3.0
    pairs = find_paraphrase_duplicates(np.stack([v0, v1]))
    assert len(pairs) == 1
    assert pairs[0][2] == pytest.approx(0.95)


def test_find_paraphrase_duplicates_sorted_and_strict() -> None:
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.95, math.sqrt(1 - 0.95**2), 0.0])
    v2 = np.array([0.99, math.sqrt(1 - 0.99**2), 0.0])
    pairs = find_paraphrase_duplicates(np.stack([v0, v1, v2]), threshold=0.85)
    similarities = [pair[2] for pair in pairs]
    assert similarities == sorted(similarities, reverse=True)
    assert pairs[0][:2] == (0, 2)  # El par más similar va primero.
    # Umbral estricto (>): una similitud igual al umbral no cuenta.
    assert find_paraphrase_duplicates(np.stack([v0, v1]), threshold=0.95) == []


def test_find_paraphrase_duplicates_edge_cases() -> None:
    assert find_paraphrase_duplicates(np.empty((0, EMBEDDING_DIM))) == []
    assert find_paraphrase_duplicates(np.zeros((1, EMBEDDING_DIM))) == []
    with pytest.raises(ValueError, match="matriz"):
        find_paraphrase_duplicates(np.zeros(EMBEDDING_DIM))


def test_default_threshold_is_conservative() -> None:
    # 0,85: cero falsos positivos medidos en la calibración del
    # 21-11-2026 (ver docstring de backend/semantic.py).
    assert DEFAULT_PARAPHRASE_THRESHOLD == 0.85


# --- integración con el modelo real (salta sin modelo en caché) ----------------


def test_real_minilm_tokenizer_matches_bert() -> None:
    _require_real_model()
    model_path = semantic.ensure_minilm_model()
    tokenizer = _load_tokenizer(_vocab_path_for_model(model_path))
    assert len(tokenizer) == 30522
    assert tokenizer.tokenize("unaffable") == ["una", "##ffa", "##ble"]
    assert tokenizer.tokenize("Me encanta este lugar!") == [
        "me",
        "en",
        "##can",
        "##ta",
        "este",
        "lu",
        "##gar",
        "!",
    ]
    assert tokenizer.tokenize("¿Dónde está la canción?") == [
        "¿",
        "don",
        "##de",
        "est",
        "##a",
        "la",
        "can",
        "##cion",
        "?",
    ]


def test_real_minilm_embeddings_shape_and_norms() -> None:
    _require_real_model()
    embeddings = embed_texts(["hola mundo", "el presupuesto es importante", ""])
    assert embeddings is not None
    assert embeddings.shape == (3, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    assert norms == pytest.approx(1.0, abs=1e-5)


def test_real_minilm_paraphrase_beats_different() -> None:
    """Calibración: la paráfrasis puntúa claramente por encima del par distinto.

    Medido el 21-11-2026: paráfrasis 0,943, distintos 0,455. Los
    umbrales del test dejan margen amplio (0,8 / 0,7 / 0,15).
    """
    _require_real_model()
    embeddings = embed_texts([_PARA_A, _PARA_B, _DIFF_B])
    assert embeddings is not None
    para = semantic_similarity(embeddings[0], embeddings[1])
    diff = semantic_similarity(embeddings[0], embeddings[2])
    assert para > 0.8
    assert diff < 0.7
    assert para - diff > 0.15


def test_real_minilm_find_duplicates() -> None:
    _require_real_model()
    embeddings = embed_texts([_NEAR_DUP_A, _NEAR_DUP_B, _DIFF_B])
    assert embeddings is not None
    pairs = find_paraphrase_duplicates(embeddings)
    assert len(pairs) == 1
    i, j, similarity = pairs[0]
    assert (i, j) == (0, 1)
    assert similarity > 0.9  # Medido: 0,972.


def test_real_minilm_session_is_cached() -> None:
    _require_real_model()
    model_path = semantic.ensure_minilm_model()
    first = semantic.load_minilm_session(model_path)
    second = semantic.load_minilm_session(model_path)
    assert first is second


def test_real_minilm_progress_reaches_100() -> None:
    _require_real_model()
    calls: list[float] = []
    embeddings = embed_texts(["hola mundo"], on_progress=calls.append)
    assert embeddings is not None
    assert calls[0] == 0
    assert calls[-1] == 100


# --- embeddings perezosos (v1.3.1): solo se embeben los lotes tocados -------


def _seed_caches(
    monkeypatch: pytest.MonkeyPatch,
    session: Any,
    tokenizer: _WordPieceTokenizer,
) -> Path:
    """Sesión + tokenizador falsos en las cachés del módulo (sin descargas)."""
    model_path = Path("/cache/minilm.onnx")
    semantic._SESSION_CACHE = (model_path, session)  # type: ignore[attr-defined]
    semantic._TOKENIZER_CACHE = (model_path, tokenizer)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        semantic, "ensure_minilm_model", lambda *a, **k: model_path
    )
    return model_path


def test_lazy_embeddings_matches_eager(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LazyEmbeddings devuelve los mismos vectores que embed_texts directo."""
    session = _FakeSession()
    _seed_caches(monkeypatch, session, tiny_tokenizer)
    texts = [f"hola mundo {i}" for i in range(70)]
    eager = embed_texts(texts, session=session, tokenizer=tiny_tokenizer)
    assert eager is not None
    lazy = LazyEmbeddings(texts)
    assert lazy.available
    for index in (0, 31, 32, 33, 63, 64, 69):
        assert np.array_equal(lazy.get(index), eager[index])


def test_lazy_embeddings_only_embeds_touched_batches(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El constructor embebe el lote 0; tocar otro lote embebe solo ese."""
    session = _FakeSession()
    _seed_caches(monkeypatch, session, tiny_tokenizer)
    texts = [f"hola mundo {i}" for i in range(70)]
    lazy = LazyEmbeddings(texts)
    assert len(session.feeds) == 1  # Lote 0, eager en el constructor.
    assert session.feeds[0]["input_ids"].shape[0] == 32
    lazy.get(0)
    lazy.get(31)  # Mismo lote: sin inferencia nueva.
    assert len(session.feeds) == 1
    lazy.get(64)  # Tercer lote (solo 6 textos).
    assert len(session.feeds) == 2
    assert session.feeds[1]["input_ids"].shape[0] == 6


def test_lazy_embeddings_empty_get_is_none() -> None:
    lazy = LazyEmbeddings([])
    assert lazy.get(0) is None
    assert lazy.get(-1) is None


def test_lazy_embeddings_failed_first_batch_is_unavailable(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el primer lote falla, available es False y get devuelve None."""

    class _FailingSession(_FakeSession):
        def run(self, _outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            raise RuntimeError("sesión rota")

    _seed_caches(monkeypatch, _FailingSession(), tiny_tokenizer)
    lazy = LazyEmbeddings(["hola"])
    assert not lazy.available
    assert lazy.get(0) is None


def test_lazy_embeddings_get_out_of_range_is_none(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_caches(monkeypatch, _FakeSession(), tiny_tokenizer)
    lazy = LazyEmbeddings(["hola"])
    assert lazy.get(-1) is None
    assert lazy.get(1) is None


def test_embed_texts_skips_ensure_when_session_cached(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con sesión + tokenizador en caché, embed_texts no llama a ensure."""
    model_path = Path("/cache/minilm.onnx")
    semantic._SESSION_CACHE = (model_path, _FakeSession())  # type: ignore[attr-defined]
    semantic._TOKENIZER_CACHE = (model_path, tiny_tokenizer)  # type: ignore[attr-defined]

    def _must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ensure_minilm_model no debe llamarse con caché llena")

    monkeypatch.setattr(semantic, "ensure_minilm_model", _must_not_run)
    embeddings = embed_texts(["hola"])
    assert embeddings is not None
    assert embeddings.shape == (1, EMBEDDING_DIM)


def test_preload_minilm_model_loads_session_and_tokenizer(
    tiny_tokenizer: _WordPieceTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preload asegura el modelo y deja cargados sesión + tokenizador."""
    model_path = Path("/cache/minilm.onnx")
    session = _FakeSession()
    monkeypatch.setattr(semantic, "ensure_minilm_model", lambda *a, **k: model_path)
    monkeypatch.setattr(semantic, "load_minilm_session", lambda path: session)
    monkeypatch.setattr(semantic, "_load_tokenizer", lambda path: tiny_tokenizer)
    assert preload_minilm_model() == model_path


def test_preload_minilm_model_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si ensure falla, preload propaga SemanticError (el pipeline lo caza)."""

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise SemanticError("sin modelo")

    monkeypatch.setattr(semantic, "ensure_minilm_model", _fail)
    with pytest.raises(SemanticError):
        preload_minilm_model()


def test_preload_minilm_model_cancelled_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _cancelled(*args: Any, **kwargs: Any) -> Any:
        raise SemanticCancelled("cancelado")

    monkeypatch.setattr(semantic, "ensure_minilm_model", _cancelled)
    with pytest.raises(SemanticCancelled):
        preload_minilm_model()
