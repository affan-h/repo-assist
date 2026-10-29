"""Ingestion pipeline: clone -> parse code & docs -> collect history -> embed -> index.

Everything is written in a single transaction at the end, so a repo being
re-ingested stays queryable on its previous index until the new one is ready.
"""

import logging
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone

import httpx

from repo_assist import store, vectors
from repo_assist.config import get_settings
from repo_assist.embeddings import Embedder, embed_with_cache, get_embedder
from repo_assist.ingest import code, docs, git, github

log = logging.getLogger(__name__)


def resolve_repo_id(owner: str, name: str) -> str:
    """Short ids ("httpx") unless that name is already taken by another owner."""
    repo_id = name.lower()
    with store.session() as conn:
        existing = store.get_repo(conn, repo_id)
    if existing and existing["owner"].lower() != owner.lower():
        repo_id = f"{owner}-{name}".lower()
    return repo_id


def register_repo(url: str) -> str:
    """Create (or reset) the repos row in 'queued' state and return its id."""
    store.init_db()
    owner, name = git.parse_repo_url(url)
    repo_id = resolve_repo_id(owner, name)
    with store.session() as conn:
        store.upsert_repo(conn, repo_id, owner, name, f"https://github.com/{owner}/{name}")
    return repo_id


def ingest(repo_id: str, ref: str | None = None, *, include_history: bool = True,
           embedder: Embedder | None = None, on_progress: Callable[[str], None] = lambda _: None) -> dict:
    """Run the full pipeline for a registered repo. Returns per-kind chunk counts."""
    settings = get_settings()
    conn = store.connect()
    started = time.monotonic()

    def stage(message: str) -> None:
        log.info("[%s] %s", repo_id, message)
        on_progress(message)
        store.update_repo(conn, repo_id, status="ingesting", stage=message)
        conn.commit()

    vector_index, prepared = vectors.get_index(), None
    try:
        repo = store.get_repo(conn, repo_id)
        if repo is None:
            raise ValueError(f"Unknown repo {repo_id!r}; call register_repo first")
        owner, name = repo["owner"], repo["name"]

        stage(f"Cloning {owner}/{name}")
        checkout = git.clone(owner, name, settings.repos_dir / repo_id, ref)
        store.update_repo(conn, repo_id, commit_sha=checkout.sha, commit_date=checkout.date)

        stage("Parsing source code and docs")
        chunks, symbols, calls = [], [], []
        for path in git.tracked_files(checkout.path):
            file_path = checkout.path / path
            if code.is_indexable_source(path):
                if file_path.stat().st_size > code.MAX_FILE_BYTES:
                    continue
                parsed = code.parse_file(path, file_path.read_bytes())
                chunks += parsed.chunks
                symbols += parsed.symbols
                calls += parsed.calls
            elif docs.is_indexable_doc(path):
                chunks += docs.chunk_document(path, file_path.read_text(encoding="utf-8", errors="replace"))

        warnings: list[str] = []
        if include_history:
            stage("Reading commit history")
            chunks += git.commit_chunks(checkout.path, settings.max_commits)
            if settings.github_token:
                stage("Fetching releases, issues, pull requests and discussions")
                cutoff = datetime.fromisoformat(checkout.date).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                try:
                    chunks += github.fetch_thread_chunks(owner, name, settings.github_token, cutoff,
                                                         settings.max_threads, progress=stage)
                except (httpx.HTTPError, RuntimeError) as exc:  # history is optional; keep code+docs+commits
                    warnings.append(f"GitHub history skipped: {exc}")
                    log.warning("[%s] %s", repo_id, warnings[-1])
            else:
                warnings.append("GITHUB_TOKEN not set: issues, PRs, discussions and releases not indexed")
                log.warning("[%s] %s", repo_id, warnings[-1])

        # Duplicate refs can occur (e.g. a one-line class header and method); keep the first.
        unique: dict[str, store.Chunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.ref, chunk)
        chunks = list(unique.values())

        embedder = embedder or get_embedder()
        stage(f"Embedding {len(chunks)} chunks")
        chunk_vectors = list(embed_with_cache(
            conn, embedder, [c.embed_text or c.text for c in chunks],
            progress=lambda done, total: stage(f"Embedding chunks ({done}/{total} new)"),
        ))

        # Comment-level passages of long threads: keyword-searchable, not embedded.
        passages = [p for c in chunks for p in github.split_passages(c)]
        chunks += passages
        chunk_vectors += [None] * len(passages)

        stage("Writing index")
        stats = dict(Counter(c.kind for c in chunks))
        stats["symbols"] = len(symbols)
        stats["ingest_seconds"] = round(time.monotonic() - started)
        if warnings:
            stats["warnings"] = warnings
        # Single SQLite transaction swaps the old index for the new one. The vector index is built
        # from inside it and switched over only once SQLite has committed (blue/green).
        with conn:
            store.clear_repo_index(conn, repo_id)
            store.insert_chunks(conn, repo_id, chunks, chunk_vectors)
            store.insert_symbols(conn, repo_id, symbols, calls)
            prepared = vector_index.prepare(conn, repo_id)
            store.update_repo(conn, repo_id, status="ready", stage="Ready", error=None, stats=stats)
        vector_index.activate(repo_id, prepared)
        prepared = None
        log.info("[%s] ready: %s", repo_id, stats)
        return stats
    except Exception as exc:
        conn.rollback()
        vector_index.discard(prepared)
        store.update_repo(conn, repo_id, status="failed", stage="Failed", error=f"{type(exc).__name__}: {exc}")
        conn.commit()
        raise
    finally:
        conn.close()
