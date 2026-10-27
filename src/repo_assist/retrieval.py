"""Evidence retrieval: hybrid search (BM25 + dense, fused with RRF), symbol lookup, grep, file reads.

Every result carries a `ref` (e.g. "httpx/_client.py:760-769", "PR#1274") that the
agent cites; the citation validator only accepts refs that a retrieval call
actually returned during the run.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import numpy as np

from repo_assist import store
from repo_assist.config import get_settings
from repo_assist.embeddings import Embedder, get_embedder

CODE_KINDS = ("code", "doc")
HISTORY_KINDS = ("commit", "issue", "pr", "discussion", "release")
RRF_K = 60            # standard reciprocal-rank-fusion constant
CANDIDATES = 50       # per-retriever candidates fed into fusion

SearchMode = Literal["hybrid", "bm25", "dense"]

_STOPWORDS = set("""
a an and are as at be but by can could did do does doing for from had has have how i if in into is it
its me my not of on or our should so that the their them then there these they this to too was we were
what when where which who whom why will with would you your about exactly specific specifically
explain describe list name
""".split())


@dataclass
class Hit:
    id: int
    ref: str
    kind: str
    path: str | None
    start_line: int | None
    end_line: int | None
    symbol: str | None
    title: str | None
    text: str
    created_at: str | None
    parent: str | None = None
    score: float = 0.0
    passage_offset: int | None = None  # history hits found via a passage: where in the thread it is


# ---------------------------------------------------------------------------
# BM25 (SQLite FTS5)
# ---------------------------------------------------------------------------

def fts_query(query: str) -> str | None:
    terms = [t for t in dict.fromkeys(store.search_terms(query)) if t not in _STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms) or None


def bm25_search(conn: sqlite3.Connection, repo_id: str, query: str, kinds: tuple[str, ...],
                limit: int = CANDIDATES) -> list[tuple[int, float]]:
    match = fts_query(query)
    if match is None:
        return []
    rows = conn.execute(
        f"""SELECT c.id, bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ? AND c.repo_id = ? AND c.kind IN ({','.join('?' * len(kinds))})
            ORDER BY score LIMIT ?""",
        (match, repo_id, *kinds, limit),
    ).fetchall()
    return [(r["id"], -r["score"]) for r in rows]  # FTS5 bm25: lower is better


# ---------------------------------------------------------------------------
# Dense (exact cosine over an in-memory matrix per repo)
# ---------------------------------------------------------------------------

@dataclass
class _VectorIndex:
    version: tuple
    ids: np.ndarray
    kinds: np.ndarray
    matrix: np.ndarray


_vector_cache: dict[str, _VectorIndex] = {}
_vector_lock = threading.Lock()


def _vector_index(conn: sqlite3.Connection, repo_id: str) -> _VectorIndex:
    version = tuple(conn.execute(
        "SELECT count(*), coalesce(max(id), 0) FROM chunks WHERE repo_id = ?", (repo_id,)
    ).fetchone())
    with _vector_lock:
        cached = _vector_cache.get(repo_id)
        if cached is not None and cached.version == version:
            return cached
        rows = conn.execute(
            "SELECT id, kind, embedding FROM chunks WHERE repo_id = ? AND embedding IS NOT NULL", (repo_id,)
        ).fetchall()
        if rows:
            matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        else:
            matrix = np.zeros((0, 1), dtype=np.float32)
        index = _VectorIndex(version, np.array([r["id"] for r in rows]), np.array([r["kind"] for r in rows]), matrix)
        _vector_cache[repo_id] = index
        return index


def dense_search(conn: sqlite3.Connection, repo_id: str, query: str, kinds: tuple[str, ...],
                 embedder: Embedder | None = None, limit: int = CANDIDATES) -> list[tuple[int, float]]:
    index = _vector_index(conn, repo_id)
    if len(index.ids) == 0:
        return []
    query_vector = (embedder or get_embedder()).embed_query(query)
    scores = index.matrix @ query_vector
    scores[~np.isin(index.kinds, kinds)] = -np.inf
    top = np.argsort(-scores)[:limit]
    return [(int(index.ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Score(d) = sum over rankings of 1 / (k + rank(d)). Robust to incomparable score scales."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def search(repo_id: str, query: str, kinds: tuple[str, ...] = CODE_KINDS, k: int = 8,
           mode: SearchMode = "hybrid", embedder: Embedder | None = None,
           conn: sqlite3.Connection | None = None) -> list[Hit]:
    own_conn = conn is None
    conn = conn or store.connect()
    try:
        if mode == "bm25":
            ranked = bm25_search(conn, repo_id, query, kinds)
        elif mode == "dense":
            ranked = dense_search(conn, repo_id, query, kinds, embedder)
        else:
            lexical = [i for i, _ in bm25_search(conn, repo_id, query, kinds)]
            semantic = [i for i, _ in dense_search(conn, repo_id, query, kinds, embedder)]
            ranked = reciprocal_rank_fusion([lexical, semantic])
        return _load_hits(conn, ranked[:k])
    finally:
        if own_conn:
            conn.close()


def _load_hits(conn: sqlite3.Connection, ranked: list[tuple[int, float]]) -> list[Hit]:
    if not ranked:
        return []
    ids = [i for i, _ in ranked]
    rows = conn.execute(
        f"""SELECT id, ref, kind, path, start_line, end_line, symbol, title, text, created_at, parent
            FROM chunks WHERE id IN ({','.join('?' * len(ids))})""", ids,
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [Hit(**dict(by_id[i]), score=s) for i, s in ranked if i in by_id]


def search_history(repo_id: str, query: str, k: int = 8, mode: SearchMode = "hybrid",
                   embedder: Embedder | None = None, conn: sqlite3.Connection | None = None) -> list[Hit]:
    """Search threads, commits and releases, matching keywords at passage (comment) level.

    Returns one hit per thread, best first. When a passage matched, the hit's text is
    that passage and `passage_offset` says where it sits in the thread.
    """
    own_conn = conn is None
    conn = conn or store.connect()
    try:
        candidates = search(repo_id, query, (*HISTORY_KINDS, "passage"), k=k * 4, mode=mode,
                            embedder=embedder, conn=conn)
        results: dict[str, Hit] = {}
        for hit in candidates:
            parent_ref = hit.parent or hit.ref
            if parent_ref in results:
                continue
            if hit.kind == "passage":
                thread = conn.execute(
                    "SELECT kind, title, created_at FROM chunks WHERE repo_id = ? AND ref = ?", (repo_id, parent_ref)
                ).fetchone()
                if thread is None:
                    continue
                hit = Hit(id=hit.id, ref=parent_ref, kind=thread["kind"], path=None, start_line=None, end_line=None,
                          symbol=None, title=thread["title"], text=hit.text, created_at=thread["created_at"],
                          score=hit.score, passage_offset=int(hit.ref.rsplit("@", 1)[1]))
            results[parent_ref] = hit
            if len(results) == k:
                break
        return list(results.values())
    finally:
        if own_conn:
            conn.close()


def get_chunk(repo_id: str, ref: str) -> Hit | None:
    with store.session() as conn:
        row = conn.execute(
            """SELECT id, ref, kind, path, start_line, end_line, symbol, title, text, created_at, parent
               FROM chunks WHERE repo_id = ? AND lower(ref) = lower(?)""", (repo_id, ref),
        ).fetchone()
    return Hit(**dict(row)) if row else None


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

def find_definitions(repo_id: str, name: str, limit: int = 10) -> list[dict]:
    """Match `name` against simple and qualified names ("get", "Client.get", "httpx.Client.get")."""
    short = name.strip().strip("`").split("(")[0].split(".")[-1]
    with store.session() as conn:
        rows = conn.execute(
            "SELECT * FROM symbols WHERE repo_id = ? AND name = ? ORDER BY length(qualified_name) LIMIT 50",
            (repo_id, short),
        ).fetchall()
    qualified = name.strip().strip("`").split("(")[0]
    exact = [dict(r) for r in rows if qualified.endswith(r["qualified_name"])]
    return (exact or [dict(r) for r in rows])[:limit]


def find_callers(repo_id: str, name: str, limit: int = 30) -> list[dict]:
    short = name.strip().strip("`").split("(")[0].split(".")[-1]
    with store.session() as conn:
        rows = conn.execute(
            "SELECT caller, path, line FROM calls WHERE repo_id = ? AND callee = ? ORDER BY path, line LIMIT ?",
            (repo_id, short, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Raw repository access (pinned checkout)
# ---------------------------------------------------------------------------

_tracked_cache: dict[str, tuple[str, list[str]]] = {}


def repo_dir(repo_id: str) -> Path:
    return get_settings().repos_dir / repo_id


def tracked_files(repo_id: str) -> list[str]:
    from repo_assist.ingest.git import _git

    root = repo_dir(repo_id)
    head = _git(root, "rev-parse", "HEAD").strip()
    cached = _tracked_cache.get(repo_id)
    if cached is None or cached[0] != head:
        files = [f for f in _git(root, "ls-files").splitlines() if f]
        _tracked_cache[repo_id] = cached = (head, files)
    return cached[1]


def read_lines(repo_id: str, path: str, start: int, end: int) -> tuple[list[str], int, int, int]:
    """Return (lines, start, end, total_lines) for a tracked file. Raises KeyError if not tracked."""
    path = path.strip().lstrip("./")
    if path not in set(tracked_files(repo_id)):
        raise KeyError(path)
    lines = (repo_dir(repo_id) / path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return lines[start - 1:end], start, end, len(lines)


def grep(repo_id: str, pattern: str, path_glob: str | None = None, ignore_case: bool = False,
         max_matches: int = 40) -> tuple[list[tuple[str, int, str]], int]:
    """Regex search over tracked text files. Returns (matches, total_match_count)."""
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    root = repo_dir(repo_id)
    matches, total = [], 0
    for path in tracked_files(repo_id):
        if path_glob and not fnmatch(path, path_glob):
            continue
        file_path = root / path
        try:
            if file_path.stat().st_size > 1_000_000:
                continue
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                total += 1
                if len(matches) < max_matches:
                    matches.append((path, lineno, line.strip()[:200]))
    return matches, total
