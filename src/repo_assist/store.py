"""SQLite storage: one file holds repos, chunks, the BM25 index, vectors, and the symbol table.

Why one SQLite file instead of a vector database: an indexed repository is a few
thousand chunks. Exact cosine search over that with numpy takes about a millisecond,
and FTS5 gives BM25 in the same file. A separate vector service would add
deployment and consistency work without improving results at this scale.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from repo_assist.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    commit_sha  TEXT,
    commit_date TEXT,
    status      TEXT NOT NULL,          -- queued | ingesting | ready | failed
    stage       TEXT,                   -- human-readable progress message
    error       TEXT,
    stats       TEXT,                   -- JSON counts per chunk kind
    updated_at  TEXT NOT NULL
);

-- One row per retrievable unit of evidence. `ref` is the citation id shown to
-- the model and the user: "path:start-end" for code/docs, "PR#12", "issue#3",
-- "discussion#7", "commit:abc1234" for history.
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    repo_id    TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,           -- code | doc | commit | issue | pr | discussion
    ref        TEXT NOT NULL,
    path       TEXT,
    start_line INTEGER,
    end_line   INTEGER,
    symbol     TEXT,
    title      TEXT,
    text       TEXT NOT NULL,
    created_at TEXT,
    embedding  BLOB,
    UNIQUE (repo_id, ref)
);
CREATE INDEX IF NOT EXISTS chunks_repo_kind ON chunks (repo_id, kind);

-- BM25 index. rowid = chunks.id; `body` is search text with identifiers split
-- into sub-words so "build request" matches `build_request` and `buildRequest`.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, tokenize = 'unicode61');

CREATE TABLE IF NOT EXISTS symbols (
    repo_id        TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind           TEXT NOT NULL,
    path           TEXT NOT NULL,
    start_line     INTEGER NOT NULL,
    end_line       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols (repo_id, name);

-- Call sites by callee name (unresolved). Imprecise for common names, but the
-- agent can read the cited lines to confirm, which is what a human would do.
CREATE TABLE IF NOT EXISTS calls (
    repo_id TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    callee  TEXT NOT NULL,
    caller  TEXT NOT NULL,
    path    TEXT NOT NULL,
    line    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_callee ON calls (repo_id, callee);

CREATE TABLE IF NOT EXISTS embedding_cache (
    key    TEXT PRIMARY KEY,            -- sha256(model + text)
    vector BLOB NOT NULL
);
"""


@dataclass
class Chunk:
    kind: str
    ref: str
    text: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None
    title: str | None = None
    created_at: str | None = None
    # Text given to the embedding model; defaults to `text`. Code chunks prepend
    # file/symbol context, long threads are cut to their opening.
    embed_text: str | None = None
    extra_search_terms: list[str] = field(default_factory=list)


@dataclass
class Symbol:
    name: str
    qualified_name: str
    kind: str
    path: str
    start_line: int
    end_line: int


@dataclass
class CallSite:
    callee: str
    caller: str
    path: str
    line: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_settings().db_path
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with session(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Search-text normalisation (shared by indexing and querying)
# ---------------------------------------------------------------------------

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(token: str) -> list[str]:
    """`_transport_for_url` -> [transport, for, url]; `beforeRetry` -> [before, retry]."""
    parts = []
    for piece in token.split("_"):
        parts.extend(p for p in _CAMEL_BOUNDARY.split(piece) if p)
    return [p.lower() for p in parts]


def search_terms(text: str) -> list[str]:
    """Lower-cased sub-word terms of every identifier-like token in `text`."""
    terms = []
    for token in _IDENTIFIER.findall(text):
        terms.extend(split_identifier(token))
    return terms


def fts_body(chunk: Chunk) -> str:
    """Text indexed by FTS5: the raw text plus camelCase splits, path and symbol."""
    header = " ".join(filter(None, [chunk.title, chunk.symbol, chunk.path]))
    camel_parts = " ".join(
        " ".join(split_identifier(tok))
        for tok in _IDENTIFIER.findall(f"{header} {chunk.text}")
        if _CAMEL_BOUNDARY.search(tok)
    )
    return " ".join([header, chunk.text, camel_parts, *chunk.extra_search_terms])


# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------

def upsert_repo(conn: sqlite3.Connection, repo_id: str, owner: str, name: str, url: str) -> None:
    conn.execute(
        """INSERT INTO repos (id, owner, name, url, status, stage, updated_at)
           VALUES (?, ?, ?, ?, 'queued', 'Queued', ?)
           ON CONFLICT(id) DO UPDATE SET status='queued', stage='Queued', error=NULL,
                                         updated_at=excluded.updated_at""",
        (repo_id, owner, name, url, now_iso()),
    )


def update_repo(conn: sqlite3.Connection, repo_id: str, **fields) -> None:
    if "stats" in fields and not isinstance(fields["stats"], str):
        fields["stats"] = json.dumps(fields["stats"])
    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE repos SET {assignments} WHERE id = ?", (*fields.values(), repo_id))


def get_repo(conn: sqlite3.Connection, repo_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
    return _repo_dict(row) if row else None


def list_repos(conn: sqlite3.Connection) -> list[dict]:
    return [_repo_dict(r) for r in conn.execute("SELECT * FROM repos ORDER BY id")]


def _repo_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["stats"] = json.loads(d["stats"]) if d["stats"] else {}
    return d


def clear_repo_index(conn: sqlite3.Connection, repo_id: str) -> None:
    """Remove all indexed content for a repo (keeps the repos row)."""
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE repo_id = ?)", (repo_id,)
    )
    for table in ("chunks", "symbols", "calls"):
        conn.execute(f"DELETE FROM {table} WHERE repo_id = ?", (repo_id,))


def delete_repo(conn: sqlite3.Connection, repo_id: str) -> None:
    clear_repo_index(conn, repo_id)
    conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))


# ---------------------------------------------------------------------------
# Index writes
# ---------------------------------------------------------------------------

def insert_chunks(conn: sqlite3.Connection, repo_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
    for chunk, vector in zip(chunks, vectors, strict=True):
        cur = conn.execute(
            """INSERT INTO chunks (repo_id, kind, ref, path, start_line, end_line, symbol, title,
                                   text, created_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(repo_id, ref) DO NOTHING""",
            (repo_id, chunk.kind, chunk.ref, chunk.path, chunk.start_line, chunk.end_line,
             chunk.symbol, chunk.title, chunk.text, chunk.created_at,
             np.asarray(vector, dtype=np.float32).tobytes()),
        )
        if cur.rowcount:
            conn.execute("INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)", (cur.lastrowid, fts_body(chunk)))


def insert_symbols(conn: sqlite3.Connection, repo_id: str, symbols: list[Symbol], calls: list[CallSite]) -> None:
    conn.executemany(
        "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(repo_id, s.name, s.qualified_name, s.kind, s.path, s.start_line, s.end_line) for s in symbols],
    )
    conn.executemany(
        "INSERT INTO calls VALUES (?, ?, ?, ?, ?)",
        [(repo_id, c.callee, c.caller, c.path, c.line) for c in calls],
    )
