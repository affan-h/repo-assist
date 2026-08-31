"""
Builds a `docs` table in data/code_graph.db from real on-disk documentation:
httpx's README.md + CHANGELOG.md, got's readme.md + documentation/**/*.md
(including migration-guides/).

RUN FROM src/: python3 build_docs_table.py

CONFIRMED REAL STRUCTURE (verified via direct heading-count check, not
assumed) before writing this:
  - All files use ATX headings (#, ##, ###...) -- zero setext (===/---)
    headings found anywhere in either repo's docs.
  - got's docs/readme: 2-tier structure, ## = file topic, ###/#### = the
    actual answerable unit (e.g. "Merge behavior explained" is a ####
    nested under an option name under "## Options"). Chunk boundary: ###
    (with any deeper #### content folded into its ### parent's chunk).
  - httpx README.md: flat, only ## headings, 5 sections. Chunk boundary: ##.
  - httpx CHANGELOG.md: ## = release (e.g. "[UNRELEASED]"), ### = category
    (Added/Removed/Fixed) WITHIN that release. Chunk boundary: ## -- a
    changelog citation needs the release context, not just "Added" alone,
    which is meaningless without knowing which release it's in. This is a
    deliberate per-file exception, not an inconsistency.

KNOWN, EXPLICITLY LOGGED GAP (not silently missing): got's release notes
(cited in Y9/Y10/Y11 ground truth as "v9/v10/v12 release notes") are NOT
in this table. They don't exist as files in the cloned repo -- confirmed
via directory listing, no CHANGELOG.md anywhere under got/. They live on
GitHub's Releases page and need a separate live-fetch tool (see
fetch_got_releases.py placeholder below), same pattern as pr_cache. Do not
assume this table covers them.
"""

import re
import sqlite3

from config import DB_PATH

# (repo, disk_path_for_reading, stored_file_path_matching_repo_convention, chunk-boundary_level)
# httpx convention (confirmed via symbols table): paths prefixed "httpx/..."
# got convention (confirmed via risk_scores table): paths unprefixed, e.g. "source/core/index.ts"
SOURCES = [
    ("httpx", "../repos/httpx/README.md", "httpx/README.md", 2),
    ("httpx", "../repos/httpx/CHANGELOG.md", "httpx/CHANGELOG.md", 2),   # exception: ## = release, see docstring
    ("got", "../repos/got/readme.md", "readme.md", 3),
    ("got", "../repos/got/documentation/1-promise.md", "documentation/1-promise.md", 3),
    ("got", "../repos/got/documentation/2-options.md", "documentation/2-options.md", 3),
    ("got", "../repos/got/documentation/3-streams.md", "documentation/3-streams.md", 3),
    ("got", "../repos/got/documentation/4-pagination.md", "documentation/4-pagination.md", 3),
    ("got", "../repos/got/documentation/5-https.md", "documentation/5-https.md", 3),
    ("got", "../repos/got/documentation/6-timeout.md", "documentation/6-timeout.md", 3),
    ("got", "../repos/got/documentation/7-retry.md", "documentation/7-retry.md", 3),
    ("got", "../repos/got/documentation/8-errors.md", "documentation/8-errors.md", 3),
    ("got", "../repos/got/documentation/9-hooks.md", "documentation/9-hooks.md", 3),
    ("got", "../repos/got/documentation/10-instances.md", "documentation/10-instances.md", 3),
    ("got", "../repos/got/documentation/async-stack-traces.md", "documentation/async-stack-traces.md", 3),
    ("got", "../repos/got/documentation/cache.md", "documentation/cache.md", 3),
    ("got", "../repos/got/documentation/diagnostics-channel.md", "documentation/diagnostics-channel.md", 3),
    ("got", "../repos/got/documentation/lets-make-a-plugin.md", "documentation/lets-make-a-plugin.md", 3),
    ("got", "../repos/got/documentation/quick-start.md", "documentation/quick-start.md", 2),
    ("got", "../repos/got/documentation/tips.md", "documentation/tips.md", 3),
    ("got", "../repos/got/documentation/typescript.md", "documentation/typescript.md", 3),
    ("got", "../repos/got/documentation/migration-guides/axios.md", "documentation/migration-guides/axios.md", 3),
    ("got", "../repos/got/documentation/migration-guides/nodejs.md", "documentation/migration-guides/nodejs.md", 3),
    ("got", "../repos/got/documentation/migration-guides/request.md", "documentation/migration-guides/request.md", 3),
]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_file(text: str, boundary_level: int) -> list[tuple[str, str]]:
    """Splits on headings at `boundary_level` or shallower (e.g. level=3
    means ### and ## and # all start new chunks, #### and deeper fold in).
    Returns list of (heading_text, chunk_content)."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("(no heading)", text.strip())] if text.strip() else []

    chunks = []
    # content before the first heading, if any, gets its own chunk
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            chunks.append(("(preamble)", pre))

    boundaries = [
        i for i, m in enumerate(matches) if len(m.group(1)) <= boundary_level
    ]
    if not boundaries:
        # nothing at or above boundary_level (e.g. whole file is deeper) --
        # fall back to chunking at whatever the shallowest level present is
        shallowest = min(len(m.group(1)) for m in matches)
        boundaries = [i for i, m in enumerate(matches) if len(m.group(1)) == shallowest]

    for idx, b in enumerate(boundaries):
        start = matches[b].start()
        end = matches[boundaries[idx + 1]].start() if idx + 1 < len(boundaries) else len(text)
        heading = matches[b].group(2)
        content = text[start:end].strip()
        chunks.append((heading, content))

    return chunks


def build():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            heading TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            PRIMARY KEY (repo, file_path, chunk_index)
        )
    """)
    conn.execute("DELETE FROM docs")  # idempotent full rebuild, matches project's init_*_table pattern

    total = 0
    for repo, disk_path, stored_path, level in SOURCES:
        try:
            text = open(disk_path, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            print(f"MISSING (skipped, real gap logged): {disk_path}")
            continue
        chunks = chunk_file(text, level)
        for i, (heading, content) in enumerate(chunks):
            conn.execute(
                "INSERT INTO docs (repo, file_path, heading, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                (repo, stored_path, heading, i, content),
            )
        total += len(chunks)
        print(f"{stored_path}: {len(chunks)} chunks")

    conn.commit()
    print(f"\nTotal chunks indexed: {total}")
    conn.close()


if __name__ == "__main__":
    build()
