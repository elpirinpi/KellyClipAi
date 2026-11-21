"""Puntuación semántica Tier 2 (Fase D, v1.2.0): MiniLM ONNX INT8.

Detecta PARÁFRASIS entre los textos de los candidatos a clip (misma
idea con distintas palabras), que el TF-IDF de ``backend.moments``
(``_select_diverse``) no puede ver porque solo compara palabras
idénticas. Es un tier OPCIONAL: sin modelo, sin runtime o sin red, la
deduplicación sigue con TF-IDF exactamente como hasta ahora.
``embed_texts`` nunca lanza (salvo cancelación del usuario) y devuelve
None cuando lo semántico no está disponible.

Modelo elegido (verificado el 21-11-2026, no por teoría):
``Xenova/all-MiniLM-L6-v2`` (Hugging Face), exportación ONNX de
``sentence-transformers/all-MiniLM-L6-v2`` para Transformers.js
(Apache 2.0). Archivo ``onnx/model_quantized.onnx``: 22.972.370 bytes,
sha256 verificado abajo. Cargado con onnxruntime 1.32.0: entradas
``input_ids`` / ``attention_mask`` / ``token_type_ids`` (int64),
salida ``last_hidden_state`` [lote, secuencia, 384]. Tokenizador
WordPiece en Python puro (0 dependencias nuevas) desde ``vocab.txt``
del mismo repo (231.508 bytes, sha256 abajo; idéntico al de
sentence-transformers): minúsculas + quitar acentos (BERT uncased),
``[CLS]`` / ``[SEP]`` / ``[PAD]`` / ``[UNK]``, longitud máxima 256.
Verificado contra los ejemplos canónicos de BERT: ``unaffable`` ->
``una`` + ``##ffa`` + ``##ble``. Embeddings: mean pooling con
``attention_mask`` + normalización L2 (igual que sentence-transformers
y Transformers.js: ``pooling: mean, normalize: true``).

Por qué Xenova y no la exportación de sentence-transformers: el
candidato ``onnx/model_quint8_avx2.onnx`` (23.046.789 bytes, sha256
b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21)
también carga y da embeddings correctos, pero es 2,2x más lento en
esta VM (9,9 ms/frase frente a 4,4 ms/frase) y su cuantización está
orientada a AVX2; el ``model_quantized.onnx`` de Xenova es INT8
dinámico sin requisito de instrucciones concretas, lo que encaja con
el principio "cualquier PC". Calidad medida idéntica en la práctica
(paráfrasis en español: media 0,74 con ambos).

Calibración REALISTA en español (21-11-2026, modelo Xenova; el modelo
se entrenó en inglés y en español funciona de forma aproximada, como
avisaba la investigación — en inglés la separación es limpia:
paráfrasis 0,74-0,87 frente a textos distintos 0,04-0,15):

- casi-duplicados (puntuación/acentos): 0,97-1,00
- rewording moderado (repetición realista del hablante): 0,845-0,943
- paráfrasis libre (mucha reescritura): 0,756-0,821
- mismo tema, contenido distinto (peor caso): 0,629-0,756
- mismo video, subtema distinto: 0,455-0,597
- tema distinto: 0,36-0,51

Las paráfrasis libres y los textos distintos del mismo tema SE SOLAPAN
(0,76 en ambos lados): ningún umbral los separa limpiamente. Por eso
el umbral por defecto es CONSERVADOR, ``0.85``: cero falsos positivos
medidos (el par no-duplicado más alto fue 0,811, un par corto de 4-5
palabras; en textos largos de clip, 0,756) y captura los
casi-duplicados y la mayoría de rewordings moderados (3 de 4 medidos).
Un umbral de 0,80 capturaría más paráfrasis libres a cambio de falsos
positivos en textos cortos; la Fase E puede recalibrarlo con videos
reales de Kelly. El umbral es un parámetro en todas partes
(``DEFAULT_PARAPHRASE_THRESHOLD``), no un valor enterrado.

Velocidad medida en esta VM (1,5 CPU): ~8 ms/texto en lotes de 32
para textos de ~60 tokens; 100 candidatos ≈ 1 s. Carga del modelo
~0,4 s. La sesión ONNX y el tokenizador se cachean.

Descarga bajo demanda: el MODELO (~23 MB) + ``vocab.txt`` (~0,2 MB)
se descargan en el primer uso con el mismo patrón que Whisper y YAMNet
(``huggingface_hub`` + callback de progreso + reintento); el RUNTIME
(onnxruntime) sí va en el instalador. Sin PyTorch jamás: todo corre
en ONNX Runtime.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

# --- Modelo MiniLM ----------------------------------------------------------
_MINILM_REPO_ID = "Xenova/all-MiniLM-L6-v2"
_MINILM_FILENAME = "onnx/model_quantized.onnx"
# sha256 del model_quantized.onnx descargado de Hugging Face el 21-11-2026
# (22.972.370 bytes). Se verifica tras la descarga: un archivo corrupto
# en caché desactiva lo semántico (TF-IDF) en vez de producir
# embeddings basura.
_MINILM_SHA256 = "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"
_MINILM_SIZE_BYTES = 22972370
# Vocabulario WordPiece del mismo repo (idéntico al de
# sentence-transformers/all-MiniLM-L6-v2: mismo sha256 en ambos).
_VOCAB_FILENAME = "vocab.txt"
_VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
_VOCAB_SIZE_BYTES = 231508
_MINILM_DOWNLOAD_ATTEMPTS = 2
_MINILM_RETRY_DELAY_SECONDS = 1.0
# Nombres de entrada/salida del grafo ONNX (verificados el 21-11-2026).
_MINILM_INPUT_IDS = "input_ids"
_MINILM_ATTENTION_MASK = "attention_mask"
_MINILM_TOKEN_TYPE_IDS = "token_type_ids"
_MINILM_OUTPUT = "last_hidden_state"
# Dimensión de embedding de all-MiniLM-L6-v2 (hidden_size 384).
EMBEDDING_DIM = 384
# Longitud máxima de secuencia: 256 tokens (el modelo admite 512, pero
# los textos de clip rara vez pasan de ~150 tokens; 256 acota la
# memoria del lote sin recortar contenido real).
MAX_SEQUENCE_LENGTH = 256
# Lote de inferencia: 32 textos (~60 tokens) tardan ~0,26 s en esta VM;
# la memoria del lote es O(32 * 256 * 384) y no depende del video.
_EMBED_BATCH_SIZE = 32
# Umbral de paráfrasis por defecto: ver la calibración del docstring
# del módulo (0,85 = conservador, cero falsos positivos medidos).
DEFAULT_PARAPHRASE_THRESHOLD = 0.85


class SemanticError(Exception):
    """Lo semántico no disponible (sin runtime, sin modelo, sin vocabulario)."""


class SemanticCancelled(Exception):
    """El usuario canceló la descarga o la inferencia semántica.

    A diferencia de SemanticError, NO se traga: el pipeline debe
    traducirla a su propia cancelación (ProjectCancelled).
    """


# --- tokenizador WordPiece (Python puro, 0 dependencias) --------------------
#
# Réplica del BertTokenizer (uncased) de transformers para este modelo:
# do_lower_case=true, strip_accents (None -> quitar acentos al pasar a
# minúsculas, como en el BERT original), tokenize_chinese_chars=true,
# never_split=null. El vocabulario no contiene ningún token con acento
# latino (verificado el 21-11-2026: 0 de 30.522), así que quitar los
# acentos es obligatorio: sin hacerlo, "está" sería [UNK] entero.
# No se hace reparación de mojibake (ftfy): las transcripciones de
# faster-whisper llegan en UTF-8 limpio.

_MAX_WORDPIECE_CHARS = 100  # Como en el BertTokenizer original.


def _is_control(char: str) -> bool:
    return unicodedata.category(char) in ("Cc", "Cf") and char not in "\t\n\r"


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    return (
        (33 <= code <= 47)
        or (58 <= code <= 64)
        or (91 <= code <= 96)
        or (123 <= code <= 126)
        or unicodedata.category(char).startswith("P")
    )


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        (0x4E00 <= code <= 0x9FFF)
        or (0x3400 <= code <= 0x4DBF)
        or (0x20000 <= code <= 0x2A6DF)
        or (0x2A700 <= code <= 0x2B73F)
        or (0x2B740 <= code <= 0x2B81F)
        or (0x2B820 <= code <= 0x2CEAF)
        or (0xF900 <= code <= 0xFAFF)
        or (0x2F800 <= code <= 0x2FA1F)
    )


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


class _WordPieceTokenizer:
    """Tokenizador WordPiece BERT (uncased) desde un vocab.txt.

    ``tokenize`` devuelve los sub-tokens (con prefijo ``##``); ``encode``
    devuelve ``(input_ids, attention_mask)`` int64 con ``[CLS]`` /
    ``[SEP]`` y relleno ``[PAD]`` hasta el texto más largo del lote.
    """

    def __init__(self, vocab_path: Path | str) -> None:
        self.vocab: dict[str, int] = {}
        with Path(vocab_path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                self.vocab[line.rstrip("\n")] = index
        try:
            self.pad_id = self.vocab["[PAD]"]
            self.unk_id = self.vocab["[UNK]"]
            self.cls_id = self.vocab["[CLS]"]
            self.sep_id = self.vocab["[SEP]"]
        except KeyError as error:
            raise SemanticError(
                f"El vocabulario {vocab_path} no trae el token especial {error}."
            ) from error

    def __len__(self) -> int:
        return len(self.vocab)

    def _basic_tokenize(self, text: str) -> list[str]:
        cleaned = "".join(char for char in text if not _is_control(char))
        spaced = "".join(f" {char} " if _is_cjk(char) else char for char in cleaned)
        tokens: list[str] = []
        for word in spaced.split():
            lowered = _strip_accents(word.lower())
            current: list[str] = []
            for char in lowered:
                if _is_punctuation(char):
                    if current:
                        tokens.append("".join(current))
                        current = []
                    tokens.append(char)
                else:
                    current.append(char)
            if current:
                tokens.append("".join(current))
        return tokens

    def _wordpiece(self, token: str) -> list[str]:
        if len(token) > _MAX_WORDPIECE_CHARS:
            return ["[UNK]"]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            found: str | None = None
            while start < end:
                sub = token[start:end] if start == 0 else "##" + token[start:end]
                if sub in self.vocab:
                    found = sub
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in self._basic_tokenize(text):
            tokens.extend(self._wordpiece(word))
        return tokens

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Codifica un lote: ids + máscara, truncado a 256 tokens.

        El truncado recorta el contenido a 254 tokens y SIEMPRE añade
        ``[SEP]`` al final, para que ningún texto pierda su cierre.
        """
        all_ids: list[list[int]] = []
        for text in texts:
            content = self.tokenize(text)[: MAX_SEQUENCE_LENGTH - 2]
            ids = [self.cls_id]
            ids.extend(self.vocab.get(token, self.unk_id) for token in content)
            ids.append(self.sep_id)
            all_ids.append(ids)
        width = max((len(ids) for ids in all_ids), default=0)
        input_ids = np.full((len(texts), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, ids in enumerate(all_ids):
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        return input_ids, attention_mask


# --- descarga del modelo ----------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot_file(snapshot_dir: Path, filename: str, sha256: str, size: int) -> Path:
    path = snapshot_dir / filename
    if not path.is_file():
        raise SemanticError(f"La descarga de MiniLM terminó sin {filename}.")
    if path.stat().st_size != size:
        raise SemanticError(
            f"El archivo {filename} descargado pesa {path.stat().st_size} bytes, "
            f"se esperaban {size}."
        )
    if _sha256_file(path) != sha256:
        # No se reintenta: la caché devolvería el mismo archivo corrupto.
        # Lo semántico queda desactivado (TF-IDF) hasta que se limpie la
        # caché de Hugging Face.
        raise SemanticError(
            f"El archivo {filename} descargado no coincide con su sha256; "
            "se usa TF-IDF. Borra la caché de Hugging Face para reintentarlo."
        )
    return path


def ensure_minilm_model(
    on_progress: ProgressCallback | None = None,
    *,
    cache_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Descarga model_quantized.onnx + vocab.txt en el primer uso.

    Caché de huggingface_hub, mismo patrón que la descarga de Whisper
    y de YAMNet: snapshot con tqdm + callback de progreso (0-95 durante
    la descarga, 100 al terminar) y reintento ante fallos de red. Si
    los archivos ya están en caché, no descarga nada y devuelve la ruta
    del modelo. Verifica tamaño y sha256 de AMBOS archivos. Lanza
    SemanticError si la descarga falla o un archivo no coincide;
    SemanticCancelled si el usuario cancela (no se reintenta).
    """
    if on_progress is not None:
        on_progress(0)

    class ModelDownloadProgress(tqdm):
        def update(self, n: int = 1) -> bool:
            if should_cancel is not None and should_cancel():
                raise SemanticCancelled("Descarga del modelo MiniLM cancelada.")
            updated = super().update(n)
            if self.total and on_progress is not None:
                percent = min(95.0, self.n / self.total * 95)
                on_progress(percent)
            return updated

    last_error: Exception | None = None
    for attempt in range(1, _MINILM_DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_dir = snapshot_download(
                repo_id=_MINILM_REPO_ID,
                allow_patterns=[_MINILM_FILENAME, _VOCAB_FILENAME],
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                max_workers=1,
                tqdm_class=ModelDownloadProgress,
            )
        except SemanticCancelled:
            raise
        except Exception as error:  # Red caída, repo no disponible, etc.
            last_error = error
            logger.warning(
                "Descarga de MiniLM fallida (intento %d/%d): %s",
                attempt,
                _MINILM_DOWNLOAD_ATTEMPTS,
                error,
            )
            if attempt < _MINILM_DOWNLOAD_ATTEMPTS:
                time.sleep(_MINILM_RETRY_DELAY_SECONDS)
            continue
        root = Path(snapshot_dir)
        model_path = _verify_snapshot_file(
            root, _MINILM_FILENAME, _MINILM_SHA256, _MINILM_SIZE_BYTES
        )
        _verify_snapshot_file(root, _VOCAB_FILENAME, _VOCAB_SHA256, _VOCAB_SIZE_BYTES)
        if on_progress is not None:
            on_progress(100)
        return model_path
    raise SemanticError(
        f"No se pudo descargar el modelo MiniLM tras {_MINILM_DOWNLOAD_ATTEMPTS} "
        f"intentos: {last_error}"
    ) from last_error


def is_semantic_available() -> bool:
    """True si lo semántico puede correr AHORA sin descargar nada.

    Para un futuro indicador en la UI y para que el hook de
    deduplicación decida si merece intentarlo: onnxruntime importable
    y modelo + vocabulario en caché. Nunca descarga ni lanza.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    try:
        for filename in (_MINILM_FILENAME, _VOCAB_FILENAME):
            hf_hub_download(
                repo_id=_MINILM_REPO_ID,
                filename=filename,
                local_files_only=True,
            )
    except Exception:
        return False
    return True


def _vocab_path_for_model(model_path: Path) -> Path:
    """Ruta de vocab.txt dentro del mismo snapshot que el modelo.

    El modelo vive en ``<snapshot>/onnx/model_quantized.onnx`` y el
    vocabulario en ``<snapshot>/vocab.txt``.
    """
    return model_path.parent.parent / _VOCAB_FILENAME


# --- sesión ONNX + tokenizador (cacheados) ----------------------------------


_SESSION_CACHE: tuple[Path, Any] | None = None
_SESSION_LOCK = threading.Lock()
_TOKENIZER_CACHE: tuple[Path, _WordPieceTokenizer] | None = None
_TOKENIZER_LOCK = threading.Lock()


def _reset_caches() -> None:
    """Solo para tests: olvida la sesión y el tokenizador cacheados."""
    global _SESSION_CACHE, _TOKENIZER_CACHE
    with _SESSION_LOCK:
        _SESSION_CACHE = None
    with _TOKENIZER_LOCK:
        _TOKENIZER_CACHE = None


def load_minilm_session(model_path: Path | str) -> Any:
    """Carga (y cachea por ruta) la sesión ONNX de MiniLM.

    Importación perezosa de onnxruntime: este módulo debe importarse
    aunque el runtime no esté instalado (TF-IDF). Lanza SemanticError
    si falta onnxruntime o el modelo no carga.
    """
    global _SESSION_CACHE
    path = Path(model_path)
    with _SESSION_LOCK:
        if _SESSION_CACHE is not None and _SESSION_CACHE[0] == path:
            return _SESSION_CACHE[1]
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise SemanticError(
            "Lo semántico necesita onnxruntime, que no está instalado. Se usa TF-IDF."
        ) from error
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise SemanticError(f"No se pudo cargar el modelo MiniLM ({path}): {error}") from error
    with _SESSION_LOCK:
        _SESSION_CACHE = (path, session)
    return session


def _load_tokenizer(vocab_path: Path | str) -> _WordPieceTokenizer:
    """Carga (y cachea por ruta) el tokenizador WordPiece."""
    global _TOKENIZER_CACHE
    path = Path(vocab_path)
    with _TOKENIZER_LOCK:
        if _TOKENIZER_CACHE is not None and _TOKENIZER_CACHE[0] == path:
            return _TOKENIZER_CACHE[1]
    try:
        tokenizer = _WordPieceTokenizer(path)
    except (OSError, SemanticError) as error:
        raise SemanticError(f"No se pudo cargar el vocabulario MiniLM ({path}): {error}") from error
    with _TOKENIZER_LOCK:
        _TOKENIZER_CACHE = (path, tokenizer)
    return tokenizer


# --- inferencia ---------------------------------------------------------------


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling con máscara + normalización L2 (sentence-transformers).

    ``last_hidden_state`` [lote, secuencia, 384], ``attention_mask``
    [lote, secuencia]. Devuelve [lote, 384] float32 con norma 1.
    """
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden_state.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)


def embed_texts(
    texts: list[str],
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    session: Any | None = None,
    tokenizer: _WordPieceTokenizer | None = None,
) -> np.ndarray | None:
    """Embeddings L2-normalizados [n, 384] (float32) para los textos.

    El punto de entrada que cablea el hook de deduplicación: descarga
    el modelo en el primer uso (progreso 0-50) y luego embebe por lotes
    de 32 (progreso 50-100). Devuelve None (con aviso en el log) si lo
    semántico no está disponible — sin runtime, sin modelo, sin red o
    con un modelo corrupto — para que la deduplicación siga con TF-IDF.
    NUNCA lanza salvo SemanticCancelled, que el pipeline debe traducir
    a su propia cancelación. Una lista vacía devuelve un array vacío
    [0, 384], no None. ``session`` / ``tokenizer`` solo existen para
    los tests (evitan la descarga).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    if should_cancel is not None and should_cancel():
        raise SemanticCancelled("Embeddings semánticos cancelados.")
    try:
        if session is None or tokenizer is None:
            if on_progress is not None:

                def _download_progress(percent: float) -> None:
                    if on_progress is not None:
                        on_progress(percent * 0.5)

            else:
                _download_progress = None
            model_path = ensure_minilm_model(_download_progress, should_cancel=should_cancel)
            if session is None:
                session = load_minilm_session(model_path)
            if tokenizer is None:
                tokenizer = _load_tokenizer(_vocab_path_for_model(model_path))
        input_names = {item.name for item in session.get_inputs()}
        output_name = session.get_outputs()[0].name
        batches: list[np.ndarray] = []
        total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
        for batch_index in range(total_batches):
            if should_cancel is not None and should_cancel():
                raise SemanticCancelled("Embeddings semánticos cancelados.")
            batch = texts[batch_index * _EMBED_BATCH_SIZE : (batch_index + 1) * _EMBED_BATCH_SIZE]
            input_ids, attention_mask = tokenizer.encode(batch)
            feeds: dict[str, np.ndarray] = {}
            if _MINILM_INPUT_IDS in input_names:
                feeds[_MINILM_INPUT_IDS] = input_ids
            if _MINILM_ATTENTION_MASK in input_names:
                feeds[_MINILM_ATTENTION_MASK] = attention_mask
            if _MINILM_TOKEN_TYPE_IDS in input_names:
                feeds[_MINILM_TOKEN_TYPE_IDS] = np.zeros_like(input_ids)
            hidden = np.asarray(session.run([output_name], feeds)[0])
            if hidden.ndim != 3 or hidden.shape[-1] != EMBEDDING_DIM:
                raise SemanticError(
                    f"El modelo MiniLM devolvió forma {hidden.shape}, "
                    f"se esperaba (*, *, {EMBEDDING_DIM})."
                )
            batches.append(_mean_pool(hidden, attention_mask))
            if on_progress is not None:
                on_progress(50.0 + 50.0 * (batch_index + 1) / total_batches)
    except SemanticCancelled:
        raise
    except SemanticError as error:
        logger.warning("Lo semántico no está disponible (%s); se usa TF-IDF.", error)
        return None
    except Exception as error:  # Defensa: lo semántico jamás rompe un proyecto.
        logger.warning(
            "La deduplicación semántica falló de forma inesperada (%s); se usa TF-IDF.",
            error,
        )
        return None
    return batches[0] if len(batches) == 1 else np.concatenate(batches, axis=0)


# --- similitud ----------------------------------------------------------------


def semantic_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos embeddings (vectores 1-D).

    Coseno real (no solo producto punto): funciona también si los
    vectores no están normalizados. Un vector cero da 0,0. Lanza
    ValueError si las formas no coinciden.
    """
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Los embeddings tienen formas distintas: {left.shape} y {right.shape}.")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def find_paraphrase_duplicates(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_PARAPHRASE_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Pares (i, j, similitud) con similitud coseno > umbral, i < j.

    Función pura para el hook de deduplicación y para tests: normaliza
    las filas por seguridad (los embeddings de ``embed_texts`` ya lo
    están) y calcula la matriz de Gram. Devuelve los pares ordenados
    por similitud descendente. O(n²) en memoria: con los pocos cientos
    de candidatos de un video es despreciable. Lanza ValueError si la
    entrada no es 2-D.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Se esperaba una matriz [n, {EMBEDDING_DIM}], llegó {matrix.shape}.")
    count = matrix.shape[0]
    if count < 2:
        return []
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-9, None)
    gram = normalized @ normalized.T
    pairs: list[tuple[int, int, float]] = []
    for i in range(count):
        for j in range(i + 1, count):
            similarity = float(gram[i, j])
            if similarity > threshold:
                pairs.append((i, j, similarity))
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return pairs
