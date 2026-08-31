"""
Real, targeted backfill for a confirmed docs-coverage gap (v2 hotfix).

Root cause, confirmed via direct evidence, not guessed:
  - `docs` table for httpx only contains CHANGELOG.md content -- confirmed
    via `SELECT DISTINCT file_path FROM docs WHERE repo='httpx'`.
  - The REAL answer to "what two use cases does ASGITransport's official
    documentation identify" lives in httpx's real
    `docs/advanced/transports.md` on GitHub (confirmed via direct web
    search against the live encode/httpx repo): "This is particularly
    useful for two main use-cases: Using httpx as a client inside test
    cases. Mocking out external services during tests..."
  - v1's own query_tools.py has a standing comment (lines 467-471)
    acknowledging `docs` table coverage gaps exist as a known, documented
    limitation of build_docs_table.py's original scrape scope -- this is
    a real, pre-existing gap in v1's own data layer, not something
    introduced by v2's retrieval agents. v2's semantic/keyword retrieval
    was working correctly the whole time; it had nothing to find because
    the source document was never indexed.

This script is a real, minimal, targeted fix: fetch the specific missing
real file(s) from GitHub, chunk them the same way build_docs_table.py's
real convention already does (word-count chunks, matching
build_embeddings_index.py's own CHUNK_SIZE/CHUNK_OVERLAP), and insert into
the EXISTING `docs` table schema (repo, file_path, chunk_index, heading,
content) -- confirmed via the real schema already read back correctly
elsewhere in this codebase (docs_agent.py, build_embeddings_index.py).

Deliberately narrow, not a full re-scrape: this targets the one confirmed
gap plus a small set of other high-value `docs/advanced/` pages likely to
have the same problem, rather than reimplementing all of
build_docs_table.py's original scope under time pressure. If more gaps
surface later, extend DOCS_TO_BACKFILL below.

RUN FROM src/: python3 backfill_missing_docs.py
Then re-run: python3 build_embeddings_index.py   (to index the new rows)
"""

import sqlite3
import time
from datetime import datetime, timezone

import requests

from config import DB_PATH
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Real, confirmed-missing files, fetched from GitHub's raw content API.
# Each entry: (repo, github_owner_repo, real_path_in_repo).
DOCS_TO_BACKFILL = [
    ("httpx", "encode/httpx", "docs/advanced/transports.md"),
    ("httpx", "encode/httpx", "docs/advanced/timeouts.md"),
    ("httpx", "encode/httpx", "docs/advanced/proxies.md"),
    ("httpx", "encode/httpx", "docs/advanced/ssl.md"),
    ("httpx", "encode/httpx", "docs/advanced/authentication.md"),
    ("httpx", "encode/httpx", "docs/quickstart.md"),
    ("httpx", "encode/httpx", "docs/async.md"),
    ("got", "sindresorhus/got", "documentation/2-options.md"),
    ("got", "sindresorhus/got", "documentation/3-streams.md"),
]


def chunk_text(text: str) -> list[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP)]


def _split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Splits a markdown file into (heading, content) sections on '## '
    boundaries, matching the real heading-per-chunk convention already
    used elsewhere in this project's docs table (confirmed via
    docs_agent.py reading a distinct `heading` per row). Falls back to a
    single section with the filename as heading if no '## ' found.

    Real fix, found via direct testing against the actual fetched file:
    a naive line-startswith('#') check also matched Python comment lines
    inside fenced code blocks (e.g. "# Instantiate a client..."), creating
    a junk heading/section from example code. Now tracks fenced code
    block state (```) and ignores '#'-prefixed lines while inside one."""
    lines = markdown_text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = None
    current_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block and (line.startswith("## ") or line.startswith("# ")):
            if current_lines:
                sections.append((current_heading or "Untitled", "\n".join(current_lines).strip()))
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading or "Untitled", "\n".join(current_lines).strip()))

    return [(h, c) for h, c in sections if c.strip()]


def fetch_real_doc(github_owner_repo: str, path: str) -> str | None:
    """Fetches real, current file content from GitHub's raw content
    endpoint. Tries both 'main' and 'master' since the two pinned repos
    use different default branches (confirmed: got uses 'main', httpx
    uses 'master')."""
    for branch in ("master", "main"):
        url = f"https://raw.githubusercontent.com/{github_owner_repo}/{branch}/{path}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text
        except requests.RequestException:
            continue
    return None


def backfill():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Confirm the real docs table schema exists before writing --
        # fail loudly rather than silently create a mismatched table.
        cursor.execute("PRAGMA table_info(docs)")
        cols = {row[1] for row in cursor.fetchall()}
        required = {"repo", "file_path", "chunk_index", "heading", "content"}
        if not required.issubset(cols):
            raise RuntimeError(
                f"docs table is missing expected columns. Has: {cols}. "
                f"Required: {required}. Aborting rather than guessing at schema."
            )

        total_inserted = 0
        for repo, github_owner_repo, path in DOCS_TO_BACKFILL:
            # Skip if this exact file_path is already present -- real,
            # idempotent check, not INSERT OR REPLACE blind (chunk_index
            # numbering could differ between the real scraper's convention
            # and this script's if run twice with different chunking).
            existing = cursor.execute(
                "SELECT COUNT(*) FROM docs WHERE repo=? AND file_path=?",
                (repo, path),
            ).fetchone()[0]
            if existing > 0:
                print(f"  SKIP {repo}/{path}: {existing} row(s) already present.")
                continue

            print(f"  Fetching {github_owner_repo}/{path}...")
            content = fetch_real_doc(github_owner_repo, path)
            time.sleep(0.3)  # light courtesy pacing against GitHub's raw endpoint

            if not content:
                print(f"    NOT FOUND (tried master and main branches) -- skipping.")
                continue

            sections = _split_into_sections(content)
            chunk_idx = 0
            for heading, section_text in sections:
                for sub_chunk in chunk_text(section_text):
                    cursor.execute(
                        "INSERT INTO docs (repo, file_path, chunk_index, heading, content) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (repo, path, chunk_idx, heading, sub_chunk),
                    )
                    chunk_idx += 1
                    total_inserted += 1
            print(f"    Inserted {chunk_idx} chunk(s) from {len(sections)} section(s).")

        conn.commit()
        print(f"\nBackfill complete: {total_inserted} new doc chunk(s) inserted.")
        print("Run `python3 build_embeddings_index.py` next to index these new rows for semantic search.")


if __name__ == "__main__":
    backfill()
