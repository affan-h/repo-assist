"""Agent tools and the evidence ledger they write to.

Every tool records exactly what it showed the model in `Deps.evidence`; the
grounding checks (grounding.py) only accept citations and quotes backed by it.
"""

import re
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch

from pydantic_ai import RunContext

from repo_assist import retrieval
from repo_assist.grounding import Evidence, normalise_ref
from repo_assist.ingest import git

MAX_READ_LINES = 150
SEARCH_RESULTS = 8
SNIPPET_LINES = 30
THREAD_PAGE_CHARS = 6000
HISTORY_PREVIEW_CHARS = 700
MAX_LISTED_FILES = 200
REPEAT_MESSAGE = ("You already made this exact call; its results are earlier in the conversation. "
                  "Use different terms, another tool, or answer with the evidence you have.")


@dataclass
class ToolCall:
    id: str
    tool: str
    args: dict
    summary: str = ""
    refs: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Deps:
    repo: dict
    question: str = ""  # the user's question (the rationale guardrail keys off it)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    trace: dict[str, ToolCall] = field(default_factory=dict)
    seen_calls: set[str] = field(default_factory=set)

    @property
    def repo_id(self) -> str:
        return self.repo["id"]

    def add(self, item: Evidence) -> None:
        """Record evidence shown to the model. Different excerpts of one thread accumulate."""
        existing = self.evidence.get(item.ref)
        if existing is None or existing.text in item.text:
            self.evidence[item.ref] = item
        elif item.text not in existing.text and existing.path is None:  # history: keep every excerpt seen
            existing.text = f"{existing.text}\n...\n{item.text}"

    def is_repeat(self, ctx: RunContext, tool: str, args: dict) -> bool:
        """Small models sometimes loop on an identical call; tell them instead of re-running it."""
        key = f"{tool}:{sorted(args.items())}"
        if key in self.seen_calls:
            self.record(ctx, tool, args, "repeated call skipped", [], time.monotonic())
            return True
        self.seen_calls.add(key)
        return False

    def record(self, ctx: RunContext, tool: str, args: dict, summary: str, refs: list[str], started: float):
        call_id = ctx.tool_call_id or f"call-{len(self.trace)}"
        self.trace[call_id] = ToolCall(call_id, tool, args, summary, refs, round(time.monotonic() - started, 3))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def hit_evidence(hit: retrieval.Hit) -> Evidence:
    return Evidence(ref=hit.ref, kind=hit.kind, text=hit.text, path=hit.path,
                    start_line=hit.start_line, end_line=hit.end_line, title=hit.title or hit.symbol)


def format_code_hit(hit: retrieval.Hit) -> str:
    label = hit.symbol or hit.title or ""
    lines = hit.text.splitlines()
    body = "\n".join(lines[:SNIPPET_LINES]) + ("\n..." if len(lines) > SNIPPET_LINES else "")
    return f"[{hit.ref}] ({hit.kind}{': ' + label if label else ''})\n{body}"


def format_history_hit(hit: retrieval.Hit, preview_chars: int = HISTORY_PREVIEW_CHARS) -> str:
    date = (hit.created_at or "")[:10]
    body = hit.text[:preview_chars] + ("..." if len(hit.text) > preview_chars else "")
    where = (f" (matching passage at char {hit.passage_offset}; read_thread('{hit.ref}', offset="
             f"{hit.passage_offset}) to read on)") if hit.passage_offset else ""
    return f"[{hit.ref}] {date} {hit.title or ''}{where}\n{body}"


def _number(lines: list[str], start: int) -> str:
    return "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(lines))


def _linked_refs(repo_id: str, numbers: list[int]) -> list[str]:
    """Which '#123' mentions are indexed issues, PRs or discussions."""
    refs = []
    for n in numbers[:10]:
        for prefix in ("PR#", "issue#", "discussion#"):
            if retrieval.get_chunk(repo_id, f"{prefix}{n}") is not None:
                refs.append(f"{prefix}{n}")
                break
    return refs


# ---------------------------------------------------------------------------
# Index-backed tools
# ---------------------------------------------------------------------------

def search_code(ctx: RunContext[Deps], query: str) -> str:
    """Search the repository's source code and documentation by meaning and keywords.
    Returns the best-matching chunks with their refs. Use natural language or identifiers."""
    if ctx.deps.is_repeat(ctx, "search_code", {"query": query}):
        return REPEAT_MESSAGE
    started = time.monotonic()
    hits = retrieval.search(ctx.deps.repo_id, query, retrieval.CODE_KINDS, k=SEARCH_RESULTS)
    for hit in hits:
        ctx.deps.add(hit_evidence(hit))
    ctx.deps.record(ctx, "search_code", {"query": query}, f"{len(hits)} results", [h.ref for h in hits], started)
    return "\n\n".join(format_code_hit(h) for h in hits) or "No results."


def search_history(ctx: RunContext[Deps], query: str) -> str:
    """Search project history: commit messages, issues, pull requests, discussions and release
    notes, down to individual comments. Use for questions about why something was designed or
    changed. Results show the best-matching passage; use read_thread to read more."""
    if ctx.deps.is_repeat(ctx, "search_history", {"query": query}):
        return REPEAT_MESSAGE
    started = time.monotonic()
    hits = retrieval.search_history(ctx.deps.repo_id, query, k=SEARCH_RESULTS)
    for hit in hits:
        ctx.deps.add(Evidence(ref=hit.ref, kind=hit.kind, text=hit.text[:HISTORY_PREVIEW_CHARS], title=hit.title))
    ctx.deps.record(ctx, "search_history", {"query": query}, f"{len(hits)} results", [h.ref for h in hits], started)
    return "\n\n".join(format_history_hit(h) for h in hits) or "No results."


def _ref_candidates(ref: str) -> list[str]:
    """Accept loose release refs: 'v9.0.0', '9.0.0', 'release:9.0.0' -> 'release:v9.0.0' etc."""
    ref = normalise_ref(ref).split("@", 1)[0]
    tag = ref[len("release:"):] if ref.lower().startswith("release:") else ref
    if ref.lower().startswith("release:") or re.match(r"^v?\d+(\.\d+)+", ref):
        return [ref, f"release:{tag}", f"release:v{tag.lstrip('v')}", f"release:{tag.lstrip('v')}"]
    return [ref]


def read_thread(ctx: RunContext[Deps], ref: str, offset: int = 0) -> str:
    """Read an issue, pull request, discussion (with comments), release notes or commit message by
    ref, e.g. 'PR#1274', 'issue#572', 'discussion#1633', 'release:v10.0.0', 'commit:b5addb64f0'.
    Long threads are returned in pages; pass the offset given at the end of a page (or by a
    search result) to continue there. Lists issues/PRs the thread links to."""
    started = time.monotonic()
    hit = next(filter(None, (retrieval.get_chunk(ctx.deps.repo_id, r) for r in _ref_candidates(ref))), None)
    if hit is None or hit.kind not in retrieval.HISTORY_KINDS:
        ctx.deps.record(ctx, "read_thread", {"ref": ref}, "not found", [], started)
        return f"No issue/PR/discussion/release/commit with ref {ref!r} is indexed."
    offset = max(0, offset)
    page = hit.text[offset:offset + THREAD_PAGE_CHARS]
    ctx.deps.add(Evidence(ref=hit.ref, kind=hit.kind, title=hit.title, text=page))
    end = offset + len(page)
    numbers = sorted({int(n) for n in re.findall(r"(?<![\w/])#(\d+)\b", page)} - {_number_of(hit.ref)})
    links = _linked_refs(ctx.deps.repo_id, numbers)
    ctx.deps.record(ctx, "read_thread", {"ref": ref, "offset": offset},
                    f"{hit.title or hit.ref} (chars {offset}-{end} of {len(hit.text)})", [hit.ref], started)
    more = f"\n\n[thread continues: call read_thread with offset={end}]" if end < len(hit.text) else ""
    linked = f"\n\nLinked threads mentioned here: {', '.join(links)}" if links else ""
    return f"[{hit.ref}] {(hit.created_at or '')[:10]}\n{page}{more}{linked}"


def _number_of(ref: str) -> int | None:
    match = re.search(r"#(\d+)$", ref)
    return int(match.group(1)) if match else None


def find_symbol(ctx: RunContext[Deps], name: str) -> str:
    """Find where a function/class/method is defined (with its code) and where it is called.
    Accepts 'name', 'Class.method' or 'module.Class.method'."""
    if ctx.deps.is_repeat(ctx, "find_symbol", {"name": name}):
        return REPEAT_MESSAGE
    started = time.monotonic()
    definitions = retrieval.find_definitions(ctx.deps.repo_id, name)
    callers = retrieval.find_callers(ctx.deps.repo_id, name)
    parts, refs = [], []
    for d in definitions[:3]:
        end = min(d["end_line"], d["start_line"] + 59)
        lines, start, end, _ = retrieval.read_lines(ctx.deps.repo_id, d["path"], d["start_line"], end)
        ref = f"{d['path']}:{start}-{end}"
        ctx.deps.add(Evidence(ref=ref, kind="code", text="\n".join(lines), path=d["path"],
                              start_line=start, end_line=end, title=d["qualified_name"]))
        refs.append(ref)
        truncated = f" (definition continues to line {d['end_line']})" if d["end_line"] > end else ""
        parts.append(f"[{ref}] {d['kind']} {d['qualified_name']}{truncated}\n" + _number(lines, start))
    if len(definitions) > 3:
        parts.append("Other definitions: " + ", ".join(
            f"{d['qualified_name']} at {d['path']}:{d['start_line']}-{d['end_line']}" for d in definitions[3:]))
    if callers:
        call_lines = []
        for c in callers:
            text = retrieval.read_lines(ctx.deps.repo_id, c["path"], c["line"], c["line"])[0]
            ref = f"{c['path']}:{c['line']}-{c['line']}"
            ctx.deps.add(Evidence(ref=ref, kind="code", text=text[0] if text else "", path=c["path"],
                                  start_line=c["line"], end_line=c["line"], title=f"call in {c['caller']}"))
            call_lines.append(f"[{ref}] in {c['caller']}: {text[0].strip() if text else ''}")
        parts.append(f"Call sites of '{name.split('.')[-1]}' (matched by name, may include unrelated "
                     f"functions with the same name):\n" + "\n".join(call_lines))
    summary = f"{len(definitions)} definitions, {len(callers)} call sites"
    ctx.deps.record(ctx, "find_symbol", {"name": name}, summary, refs, started)
    return "\n\n".join(parts) or f"No symbol named {name!r} found. Try search_code or grep."


# ---------------------------------------------------------------------------
# Checkout-backed tools
# ---------------------------------------------------------------------------

def read_file(ctx: RunContext[Deps], path: str, start_line: int = 1, end_line: int = 120) -> str:
    """Read lines from a file in the repository (any tracked file, including tests and config).
    At most 150 lines per call."""
    started = time.monotonic()
    if end_line < start_line:  # models sometimes pass a line count or swap the bounds
        end_line = start_line + 119
    end_line = min(end_line, start_line + MAX_READ_LINES - 1)
    try:
        lines, start, end, total = retrieval.read_lines(ctx.deps.repo_id, path, start_line, end_line)
    except KeyError:
        ctx.deps.record(ctx, "read_file", {"path": path}, "not found", [], started)
        return f"{path!r} is not a file in this repository. Use list_files or grep to find paths."
    ref = f"{path.strip().lstrip('./')}:{start}-{end}"
    ctx.deps.add(Evidence(ref=ref, kind="code", text="\n".join(lines), path=ref.rsplit(":", 1)[0],
                          start_line=start, end_line=end))
    ctx.deps.record(ctx, "read_file", {"path": path, "start_line": start_line, "end_line": end_line},
                    f"lines {start}-{end} of {total}", [ref], started)
    return f"[{ref}] ({total} lines in file)\n" + _number(lines, start)


def grep(ctx: RunContext[Deps], pattern: str, path_glob: str | None = None) -> str:
    """Regex search (Python syntax) over every tracked file, e.g. pattern 'def build_request',
    path_glob 'httpx/*.py'. Best for exact identifiers, imports and string literals."""
    if ctx.deps.is_repeat(ctx, "grep", {"pattern": pattern, "path_glob": path_glob}):
        return REPEAT_MESSAGE
    started = time.monotonic()
    try:
        matches, total = retrieval.grep(ctx.deps.repo_id, pattern, path_glob)
    except re.error as exc:
        ctx.deps.record(ctx, "grep", {"pattern": pattern}, "invalid regex", [], started)
        return f"Invalid regex: {exc}"
    refs = []
    for path, line, text in matches:
        ref = f"{path}:{line}-{line}"
        ctx.deps.add(Evidence(ref=ref, kind="code", text=text, path=path, start_line=line, end_line=line))
        refs.append(ref)
    ctx.deps.record(ctx, "grep", {"pattern": pattern, "path_glob": path_glob}, f"{total} matches", refs, started)
    if not matches:
        return "No matches."
    more = f"\n({total - len(matches)} more matches not shown; narrow the pattern or path_glob)" if total > len(matches) else ""
    return "\n".join(f"[{p}:{n}-{n}] {t}" for p, n, t in matches) + more


def list_files(ctx: RunContext[Deps], path_glob: str = "*") -> str:
    """List tracked files matching a glob, e.g. 'httpx/*.py', '*.md', 'src/**'. Use to explore the
    repository layout."""
    if ctx.deps.is_repeat(ctx, "list_files", {"path_glob": path_glob}):
        return REPEAT_MESSAGE
    started = time.monotonic()
    files = [f for f in retrieval.tracked_files(ctx.deps.repo_id) if fnmatch(f, path_glob)]
    ctx.deps.record(ctx, "list_files", {"path_glob": path_glob}, f"{len(files)} files", [], started)
    if not files:
        return "No files match."
    more = f"\n... and {len(files) - MAX_LISTED_FILES} more" if len(files) > MAX_LISTED_FILES else ""
    return "\n".join(files[:MAX_LISTED_FILES]) + more


def _commit_lines(ctx: RunContext[Deps], commits: list[git.Commit]) -> list[str]:
    lines = []
    for c in commits:
        ctx.deps.add(Evidence(ref=c.ref, kind="commit", text=c.message, title=c.subject))
        links = _linked_refs(ctx.deps.repo_id, c.linked_numbers())
        linked = f" (linked: {', '.join(links)})" if links else ""
        body = f"\n    {c.body[:300]}" if c.body else ""
        lines.append(f"[{c.ref}] {c.date[:10]} {c.subject}{linked}{body}")
    return lines


def code_history(ctx: RunContext[Deps], path: str, start_line: int, end_line: int) -> str:
    """Trace lines of code back through the commits that changed them (git log -L), newest first,
    with the issues/PRs each commit references. The way to answer "why is this code like this?":
    find the code, call code_history on its lines, then read_thread the linked PRs/issues."""
    args = {"path": path, "start_line": start_line, "end_line": end_line}
    if ctx.deps.is_repeat(ctx, "code_history", args):
        return REPEAT_MESSAGE
    started = time.monotonic()
    path = path.strip().lstrip("./")
    if path not in set(retrieval.tracked_files(ctx.deps.repo_id)):
        ctx.deps.record(ctx, "code_history", args, "not found", [], started)
        return f"{path!r} is not a file in this repository."
    try:
        commits = git.line_history(retrieval.repo_dir(ctx.deps.repo_id), path, start_line, max(start_line, end_line))
    except RuntimeError as exc:
        ctx.deps.record(ctx, "code_history", args, "failed", [], started)
        return f"git log -L failed: {exc}"
    lines = _commit_lines(ctx, commits)
    ctx.deps.record(ctx, "code_history", args, f"{len(commits)} commits", [c.ref for c in commits], started)
    return "\n".join(lines) or "No commits found for those lines."


TOOLS = [search_code, search_history, read_thread, code_history, find_symbol, read_file, grep, list_files]
