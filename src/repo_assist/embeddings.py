"""Local text embeddings (ONNX via fastembed), with a content-hash cache in SQLite.

A small local model (bge-small, 33M params) is used because ingestion embeds
thousands of chunks: hosted free tiers rate-limit that to a crawl, and larger
local models are 3-10x slower on CPU. Exact identifier matching is covered by
BM25 and grep, so dense retrieval only needs to capture meaning.
"""

import hashlib
import sqlite3
import threading
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from repo_assist.config import get_settings

# bge models expect this instruction on queries (not on documents).
_QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}
BATCH_SIZE = 32


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # deferred: loading onnxruntime is slow

        self.name = model_name
        self._model = TextEmbedding(model_name, cache_dir=str(get_settings().model_cache_dir))
        self._prefix = _QUERY_PREFIX.get(model_name, "")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(list(texts), batch_size=BATCH_SIZE))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed([self._prefix + text]))))[0]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = FastEmbedder(get_settings().embedding_model)
        return _embedder


def embed_with_cache(conn: sqlite3.Connection, embedder: Embedder, texts: Sequence[str],
                     progress: Callable[[int, int], None] = lambda done, total: None) -> np.ndarray:
    """Embed `texts`, reusing vectors cached from earlier ingests of identical text."""
    keys = [hashlib.sha256(f"{embedder.name}\0{t}".encode()).hexdigest() for t in texts]
    cached: dict[str, np.ndarray] = {}
    for i in range(0, len(keys), 500):
        batch = keys[i:i + 500]
        rows = conn.execute(
            f"SELECT key, vector FROM embedding_cache WHERE key IN ({','.join('?' * len(batch))})", batch
        )
        cached.update((k, np.frombuffer(v, dtype=np.float32)) for k, v in rows)

    text_for = dict(zip(keys, texts))
    # Batches are padded to their longest text, so embedding in length order avoids
    # wasting compute on padding (about 3x faster on mixed-length history threads).
    missing = sorted({k for k in keys if k not in cached}, key=lambda k: len(text_for[k]))
    step = BATCH_SIZE * 4
    for i in range(0, len(missing), step):
        batch = missing[i:i + step]
        vectors = embedder.embed_documents([text_for[k] for k in batch])
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)",
            [(k, v.tobytes()) for k, v in zip(batch, vectors)],
        )
        conn.commit()  # persist progress so an interrupted ingest resumes cheaply
        cached.update(zip(batch, vectors))
        progress(min(i + step, len(missing)), len(missing))

    return np.stack([cached[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
