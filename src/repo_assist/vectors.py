"""Dense vector search behind one interface, with two backends.

- `QdrantIndex` (used when REPO_ASSIST_QDRANT_URL is set, as in docker-compose): one
  Qdrant collection per repository with an HNSW index and a `kind` payload index, so
  "code and docs only" or "history only" is filtered inside the vector database.
- `ExactIndex` (the default without Qdrant): cosine similarity against every vector,
  in-process with numpy. At the size of one repository (a few thousand chunks) this
  takes about a millisecond and needs no extra service.

SQLite stays the source of record for embeddings (`chunks.embedding`); a Qdrant
collection is a derived index, rebuilt from SQLite whenever a repository is
re-ingested or the collection is missing. A rebuild is blue/green: the new collection
is filled alongside the old one and an alias (`repo_assist_<repo>`) is repointed to it
in one atomic operation only after SQLite has committed, so searches never see a
half-built index or chunk ids SQLite doesn't have.

`evals/vector_bench.py` compares the two backends on speed and on how many of the exact
top results HNSW's approximate search returns.
"""

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from repo_assist.config import get_settings

log = logging.getLogger(__name__)

UPSERT_BATCH = 512


def stored_vectors(conn: sqlite3.Connection, repo_id: str) -> tuple[list[int], list[str], np.ndarray]:
    """All embedded chunks of a repo (passages have no vector): ids, kinds and an (n, dim) matrix."""
    rows = conn.execute(
        "SELECT id, kind, embedding FROM chunks WHERE repo_id = ? AND embedding IS NOT NULL ORDER BY id", (repo_id,)
    ).fetchall()
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else \
        np.zeros((0, 1), dtype=np.float32)
    return [r["id"] for r in rows], [r["kind"] for r in rows], matrix


class VectorIndex(Protocol):
    name: str

    def prepare(self, conn: sqlite3.Connection, repo_id: str) -> str | None:
        """Build a new index from the chunks visible on `conn` (it may be mid-transaction). Returns a handle."""

    def activate(self, repo_id: str, handle: str | None) -> None:
        """Make a prepared index the one searches use, after SQLite has committed."""

    def discard(self, handle: str | None) -> None:
        """Throw away a prepared index whose SQLite transaction failed."""

    def search(self, conn: sqlite3.Connection, repo_id: str, query: np.ndarray, kinds: tuple[str, ...],
               limit: int) -> list[tuple[int, float]]:
        """Chunk ids and similarity scores of the nearest vectors among chunks of the given kinds."""


# ---------------------------------------------------------------------------
# Exact search (numpy, in-process)
# ---------------------------------------------------------------------------

@dataclass
class _Matrix:
    version: tuple
    ids: np.ndarray
    kinds: np.ndarray
    matrix: np.ndarray


class ExactIndex:
    """Reads vectors straight from SQLite; nothing to build, so prepare/activate are no-ops."""

    name = "exact"

    def __init__(self) -> None:
        self._cache: dict[str, _Matrix] = {}
        self._lock = threading.Lock()

    def prepare(self, conn, repo_id):
        return None

    def activate(self, repo_id, handle):
        pass

    def discard(self, handle):
        pass

    def _matrix(self, conn: sqlite3.Connection, repo_id: str) -> _Matrix:
        version = tuple(conn.execute(
            "SELECT count(*), coalesce(max(id), 0) FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone())
        with self._lock:
            cached = self._cache.get(repo_id)
            if cached is None or cached.version != version:
                ids, kinds, matrix = stored_vectors(conn, repo_id)
                cached = self._cache[repo_id] = _Matrix(version, np.array(ids), np.array(kinds), matrix)
            return cached

    def search(self, conn, repo_id, query, kinds, limit):
        index = self._matrix(conn, repo_id)
        if len(index.ids) == 0:
            return []
        scores = index.matrix @ query
        scores[~np.isin(index.kinds, kinds)] = -np.inf
        top = np.argsort(-scores)[:limit]
        return [(int(index.ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantIndex:
    name = "qdrant"

    def __init__(self, client) -> None:
        self.client = client
        self._lock = threading.Lock()  # one rebuild at a time per process

    @staticmethod
    def alias(repo_id: str) -> str:
        return f"repo_assist_{repo_id}"

    def _target(self, repo_id: str) -> str | None:
        """The collection a repo's alias currently points to."""
        alias = self.alias(repo_id)
        return next((a.collection_name for a in self.client.get_aliases().aliases if a.alias_name == alias), None)

    def prepare(self, conn, repo_id):
        from qdrant_client import models

        ids, kinds, matrix = stored_vectors(conn, repo_id)
        if not ids:
            return None
        collection = f"{self.alias(repo_id)}_{uuid.uuid4().hex[:8]}"
        self.client.create_collection(
            collection, vectors_config=models.VectorParams(size=matrix.shape[1], distance=models.Distance.COSINE))
        self.client.create_payload_index(collection, "kind", models.PayloadSchemaType.KEYWORD)
        for start in range(0, len(ids), UPSERT_BATCH):
            stop = start + UPSERT_BATCH
            self.client.upsert(collection, wait=True, points=models.Batch(
                ids=ids[start:stop], vectors=matrix[start:stop].tolist(),
                payloads=[{"kind": kind} for kind in kinds[start:stop]]))
        return collection

    def activate(self, repo_id, handle):
        from qdrant_client import models

        if handle is None:
            return
        alias, previous = self.alias(repo_id), self._target(repo_id)
        operations = [models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))] if previous else []
        operations.append(models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=handle, alias_name=alias)))
        self.client.update_collection_aliases(change_aliases_operations=operations)  # atomic switch
        if previous and previous != handle:
            self.client.delete_collection(previous)

    def discard(self, handle):
        if handle is not None:
            self.client.delete_collection(handle)

    def _ensure(self, conn: sqlite3.Connection, repo_id: str) -> bool:
        """Build a missing collection from SQLite (e.g. a repo indexed before Qdrant was configured)."""
        if self._target(repo_id):
            return True
        with self._lock:
            if self._target(repo_id):
                return True
            log.info("Building the Qdrant collection for %s from SQLite", repo_id)
            handle = self.prepare(conn, repo_id)
            self.activate(repo_id, handle)
            return handle is not None

    def search(self, conn, repo_id, query, kinds, limit):
        from qdrant_client import models

        if not self._ensure(conn, repo_id):
            return []
        result = self.client.query_points(
            self.alias(repo_id), query=query.tolist(), limit=limit, with_payload=False,
            query_filter=models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds)))]),
        )
        return [(int(p.id), float(p.score)) for p in result.points]


_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        url = get_settings().qdrant_url
        if url:
            from qdrant_client import QdrantClient

            _index = QdrantIndex(QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url))
        else:
            _index = ExactIndex()
    return _index
