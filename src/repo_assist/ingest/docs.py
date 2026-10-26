"""Split documentation (Markdown/reStructuredText/plain text) into heading-scoped chunks."""

import re
from pathlib import PurePosixPath

from repo_assist.ingest.code import is_test_path
from repo_assist.store import Chunk

DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
MAX_SECTION_LINES = 60
MIN_SECTION_CHARS = 120   # smaller sections are merged into the next one

_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def is_indexable_doc(path: str) -> bool:
    p = PurePosixPath(path)
    if p.suffix.lower() not in DOC_SUFFIXES or is_test_path(path):
        return False
    if p.suffix.lower() == ".txt":  # only top-level prose like README.txt, not data files
        return p.stem.upper() in {"README", "CHANGELOG", "CHANGES", "HISTORY"}
    return not any(part in {"node_modules", ".github"} for part in p.parts)


def chunk_document(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    if PurePosixPath(path).suffix.lower() in {".md", ".mdx"}:
        sections = _markdown_sections(lines)
    else:
        sections = [(None, 1, len(lines))]

    chunks = []
    for title, start, end in _merge_small(sections, lines):
        for s, e in _split(lines, start, end):
            body = "\n".join(lines[s - 1:e]).strip()
            if not body:
                continue
            context = f"{path}\n{title}" if title else path
            chunks.append(Chunk(
                kind="doc", ref=f"{path}:{s}-{e}", path=path, start_line=s, end_line=e,
                title=title, text=body, embed_text=f"{context}\n{body}"[:4000],
            ))
    return chunks


def _markdown_sections(lines: list[str]) -> list[tuple[str | None, int, int]]:
    """Return (breadcrumb, start_line, end_line) per heading section, 1-based inclusive."""
    sections, trail = [], []
    current_title, current_start, in_fence = None, 1, False
    for i, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if not match:
            continue
        if i > current_start:
            sections.append((current_title, current_start, i - 1))
        level, heading = len(match.group(1)), match.group(2).strip()
        trail = [t for t in trail if t[0] < level] + [(level, heading)]
        current_title, current_start = " > ".join(h for _, h in trail), i
    sections.append((current_title, current_start, len(lines)))
    return sections


def _merge_small(sections, lines):
    merged, pending = [], None
    for title, start, end in sections:
        if pending is not None:
            # The later heading's breadcrumb already includes its parents.
            start, title = pending[1], title or pending[0]
            pending = None
        size = sum(len(lines[n - 1]) for n in range(start, end + 1))
        if size < MIN_SECTION_CHARS and end < len(lines):
            pending = (title, start)
            continue
        merged.append((title, start, end))
    if pending is not None:
        merged.append((pending[0], pending[1], len(lines)))
    return merged


def _split(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Split long sections at blank lines into pieces of at most MAX_SECTION_LINES."""
    spans, s = [], start
    while end - s + 1 > MAX_SECTION_LINES:
        cut = s + MAX_SECTION_LINES - 1
        for candidate in range(cut, s + MAX_SECTION_LINES // 2, -1):
            if not lines[candidate - 1].strip():
                cut = candidate
                break
        spans.append((s, cut))
        s = cut + 1
    spans.append((s, end))
    return spans
