"""Question answering: a tool-using agent, plus a single-shot RAG baseline for comparison.

Both modes share one answer schema and one citation validator. Every tool
records the evidence it showed the model; a final answer is accepted only if
each citation points at evidence from this run (a code citation must fall
inside a line range the model actually saw). Otherwise the model gets a
retry with the list of valid refs, so ungrounded citations can't reach the user.
"""

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from repo_assist import retrieval, store
from repo_assist.config import get_settings

Mode = Literal["agent", "single_shot"]

MAX_READ_LINES = 150
SEARCH_RESULTS = 8
SNIPPET_LINES = 30
THREAD_PAGE_CHARS = 6000
AGENT_LIMITS = UsageLimits(request_limit=12, tool_calls_limit=15)


class Answer(BaseModel):
    answer: str = Field(description=(
        "Concise Markdown answer. Support claims with inline markers like [1], [2] that index "
        "(1-based) into `citations`."))
    citations: list[str] = Field(description=(
        "Evidence refs exactly as shown in tool results, e.g. 'httpx/_client.py:760-769', "
        "'PR#1274', 'issue#572', 'discussion#1633', 'release:v10.0.0', 'commit:b5addb64f0'. A code/doc ref may "
        "narrow to a sub-range of lines you were shown."))
    abstained: bool = Field(description=(
        "True if the retrieved evidence does not answer the question. Never guess."))


@dataclass
class Evidence:
    ref: str
    kind: str
    text: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    title: str | None = None


@dataclass
class ToolCall:
    id: str
    tool: str
    args: dict
    summary: str = ""
    refs: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class Citation:
    ref: str
    kind: str
    url: str
    title: str | None
    path: str | None
    start_line: int | None
    end_line: int | None
    snippet: str


@dataclass
class Result:
    question: str
    repo_id: str
    mode: Mode
    answer: str
    citations: list[Citation]
    abstained: bool
    trace: list[ToolCall]
    evidence: dict[str, Evidence]
    usage: dict
    latency_s: float
    error: str | None = None

    def to_dict(self, include_evidence: bool = False) -> dict:
        data = asdict(self)
        if not include_evidence:
            data.pop("evidence")
        return data


@dataclass
class Deps:
    repo: dict
    evidence: dict[str, Evidence] = field(default_factory=dict)
    trace: dict[str, ToolCall] = field(default_factory=dict)

    @property
    def repo_id(self) -> str:
        return self.repo["id"]

    def add(self, item: Evidence) -> None:
        existing = self.evidence.get(item.ref)
        if existing is None or len(item.text) > len(existing.text):
            self.evidence[item.ref] = item

    def record(self, ctx: RunContext, tool: str, args: dict, summary: str, refs: list[str], started: float):
        call_id = ctx.tool_call_id or f"call-{len(self.trace)}"
        self.trace[call_id] = ToolCall(call_id, tool, args, summary, refs, round(time.monotonic() - started, 3))


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

_RANGE_REF = re.compile(r"^(?P<path>.+?)(?::|#L)(?P<start>\d+)(?:-L?(?P<end>\d+))?$")
_MARKER = re.compile(r"\[(\d+)\]")


def normalise_ref(ref: str) -> str:
    """Strip formatting and accept GitHub-style anchors: 'a.py#L3-L9' -> 'a.py:3-9'."""
    ref = ref.strip().strip("`[]() ")
    return ref.replace("#L", ":", 1).replace("-L", "-") if "#L" in ref else ref


def resolve_citation(ref: str, evidence: dict[str, Evidence]) -> tuple[str, Evidence] | None:
    """Return (canonical_ref, evidence) if `ref` is backed by evidence seen this run."""
    ref = normalise_ref(ref)
    lowered = {k.lower(): v for k, v in evidence.items()}
    if ref.lower() in lowered:
        item = lowered[ref.lower()]
        return item.ref, item
    if ref.lower().startswith("commit:"):  # allow abbreviated SHAs (>= 7 chars)
        sha = ref.lower().removeprefix("commit:")
        for key, item in lowered.items():
            if len(sha) >= 7 and key.startswith("commit:") and (key[7:].startswith(sha) or sha.startswith(key[7:])):
                return item.ref, item
        return None
    match = _RANGE_REF.match(ref)
    if not match:
        return None
    path, start = match["path"], int(match["start"])
    end = int(match["end"] or start)
    for item in evidence.values():
        if item.path == path and item.start_line is not None and item.start_line <= start <= end <= item.end_line:
            return f"{path}:{start}-{end}", item
    return None


def validate_answer(answer: Answer, evidence: dict[str, Evidence]) -> list[str]:
    """Return a list of problems; empty means the answer is acceptable."""
    problems = []
    for ref in answer.citations:
        if resolve_citation(ref, evidence) is None:
            problems.append(f"citation {ref!r} is not a ref (or line range) returned by a tool in this session")
    if not answer.abstained and not answer.citations:
        problems.append("a non-abstained answer must cite at least one piece of evidence")
    markers = {int(m) for m in _MARKER.findall(answer.answer)}
    out_of_range = sorted(m for m in markers if not 1 <= m <= len(answer.citations))
    if out_of_range:
        problems.append(f"markers {out_of_range} have no matching entry in `citations`")
    return problems


async def _check_citations(ctx: RunContext[Deps], answer: Answer) -> Answer:
    problems = validate_answer(answer, ctx.deps.evidence)
    if problems:
        sample = ", ".join(list(ctx.deps.evidence)[:25]) or "(none yet - use the tools first)"
        raise ModelRetry("Fix these citation problems: " + "; ".join(problems) + f". Valid refs include: {sample}")
    return answer


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _hit_evidence(hit: retrieval.Hit) -> Evidence:
    return Evidence(ref=hit.ref, kind=hit.kind, text=hit.text, path=hit.path,
                    start_line=hit.start_line, end_line=hit.end_line, title=hit.title or hit.symbol)


def _format_code_hit(hit: retrieval.Hit) -> str:
    label = hit.symbol or hit.title or ""
    lines = hit.text.splitlines()
    body = "\n".join(lines[:SNIPPET_LINES]) + ("\n..." if len(lines) > SNIPPET_LINES else "")
    return f"[{hit.ref}] ({hit.kind}{': ' + label if label else ''})\n{body}"


def _format_history_hit(hit: retrieval.Hit) -> str:
    date = (hit.created_at or "")[:10]
    body = hit.text[:700] + ("..." if len(hit.text) > 700 else "")
    return f"[{hit.ref}] {date} {hit.title or ''}\n{body}"


def search_code(ctx: RunContext[Deps], query: str) -> str:
    """Hybrid (keyword + semantic) search over the repository's source code and documentation.
    Returns the best-matching chunks with their refs. Use natural language or identifiers."""
    started = time.monotonic()
    hits = retrieval.search(ctx.deps.repo_id, query, retrieval.CODE_KINDS, k=SEARCH_RESULTS)
    for hit in hits:
        ctx.deps.add(_hit_evidence(hit))
    ctx.deps.record(ctx, "search_code", {"query": query}, f"{len(hits)} results", [h.ref for h in hits], started)
    return "\n\n".join(_format_code_hit(h) for h in hits) or "No results."


def search_history(ctx: RunContext[Deps], query: str) -> str:
    """Hybrid search over project history: commit messages, issues, pull requests, discussions and
    release notes.
    Use for questions about why something was designed or changed. Snippets are truncated;
    use read_thread to see a full thread."""
    started = time.monotonic()
    hits = retrieval.search(ctx.deps.repo_id, query, retrieval.HISTORY_KINDS, k=SEARCH_RESULTS)
    for hit in hits:
        shown = Evidence(ref=hit.ref, kind=hit.kind, text=hit.text[:700], title=hit.title)
        ctx.deps.add(shown)
    ctx.deps.record(ctx, "search_history", {"query": query}, f"{len(hits)} results", [h.ref for h in hits], started)
    return "\n\n".join(_format_history_hit(h) for h in hits) or "No results."


def read_thread(ctx: RunContext[Deps], ref: str, offset: int = 0) -> str:
    """Read an issue, pull request, discussion (with comments), release notes or commit message by
    ref, e.g. 'PR#1274', 'issue#572', 'discussion#1633', 'release:v10.0.0', 'commit:b5addb64f0'. Long threads are
    returned in pages; pass the offset given at the end of a page to continue."""
    started = time.monotonic()
    hit = retrieval.get_chunk(ctx.deps.repo_id, normalise_ref(ref))
    if hit is None or hit.kind not in retrieval.HISTORY_KINDS:
        ctx.deps.record(ctx, "read_thread", {"ref": ref}, "not found", [], started)
        return f"No issue/PR/discussion/release/commit with ref {ref!r} is indexed."
    offset = max(0, offset)
    page = hit.text[offset:offset + THREAD_PAGE_CHARS]
    existing = ctx.deps.evidence.get(hit.ref)
    seen = existing.text if existing and existing.kind == hit.kind else ""
    if not seen:
        text = page
    else:  # accumulate every page the model has read, for citation snippets and grading
        text = seen if page in seen else f"{seen}\n...\n{page}"
    ctx.deps.evidence[hit.ref] = Evidence(ref=hit.ref, kind=hit.kind, title=hit.title, text=text)
    end = offset + len(page)
    ctx.deps.record(ctx, "read_thread", {"ref": ref, "offset": offset},
                    f"{hit.title or hit.ref} (chars {offset}-{end} of {len(hit.text)})", [hit.ref], started)
    more = f"\n\n[thread continues: call read_thread with offset={end}]" if end < len(hit.text) else ""
    return f"[{hit.ref}] {(hit.created_at or '')[:10]}\n{page}{more}"


def find_symbol(ctx: RunContext[Deps], name: str) -> str:
    """Find where a function/class/method is defined (with its code) and where it is called.
    Accepts 'name', 'Class.method' or 'module.Class.method'."""
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
        return f"{path!r} is not a file in this repository. Use grep or search_code to find paths."
    ref = f"{path.strip().lstrip('./')}:{start}-{end}"
    ctx.deps.add(Evidence(ref=ref, kind="code", text="\n".join(lines), path=ref.rsplit(":", 1)[0],
                          start_line=start, end_line=end))
    ctx.deps.record(ctx, "read_file", {"path": path, "start_line": start_line, "end_line": end_line},
                    f"lines {start}-{end} of {total}", [ref], started)
    return f"[{ref}] ({total} lines in file)\n" + _number(lines, start)


def grep(ctx: RunContext[Deps], pattern: str, path_glob: str | None = None) -> str:
    """Regex search (Python syntax) over every tracked file, e.g. pattern 'def build_request',
    path_glob 'httpx/*.py'. Best for exact identifiers, imports and string literals."""
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


def _number(lines: list[str], start: int) -> str:
    return "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(lines))


TOOLS = [search_code, search_history, read_thread, find_symbol, read_file, grep]

# ---------------------------------------------------------------------------
# Prompts and agents
# ---------------------------------------------------------------------------

_ANSWER_RULES = """\
Answer rules:
- Base every claim on evidence shown to you in this session, never on prior knowledge of the project.
- Cite with [n] markers; `citations` lists refs exactly as shown (you may narrow a code/doc ref to a
  sub-range of the lines you were shown, e.g. 'httpx/_client.py:762-765').
- If the evidence does not answer the question, set abstained=true and briefly say what you looked for.
  For "why" questions, a description of current behaviour is not a reason: only cite a maintainer's
  stated rationale (issue, PR, discussion, commit message, docs). Do not invent plausible reasons.
- If the question contains a false premise, say so and cite the evidence that contradicts it.
- Be concise: a short paragraph or a list. Name the relevant files, functions and line ranges."""

AGENT_INSTRUCTIONS = """\
You answer questions about the GitHub repository {owner}/{name} at commit {sha}.
Investigate with tools before answering:
- search_code: keyword+semantic search over source and docs. Good first step for most questions.
- find_symbol: definition and call sites of a named function/class/method.
- read_file / grep: verify details in the actual code, follow calls, find imports or literals.
- search_history / read_thread: commits, issues, PRs, discussions and release notes; use for "why" and
  "when" questions.
Search results are excerpts: read the code before relying on details. Stop once the evidence is
sufficient (usually 2-6 tool calls).

""" + _ANSWER_RULES

SINGLE_SHOT_INSTRUCTIONS = """\
You answer questions about the GitHub repository {owner}/{name} at commit {sha}, using only the
retrieved evidence provided in the message.

""" + _ANSWER_RULES


def _instructions(template: str) -> Callable[[RunContext[Deps]], str]:
    def render(ctx: RunContext[Deps]) -> str:
        repo = ctx.deps.repo
        return template.format(owner=repo["owner"], name=repo["name"], sha=(repo["commit_sha"] or "HEAD")[:10])
    return render


def _build_agent(mode: Mode) -> Agent[Deps, Answer]:
    template, tools = (AGENT_INSTRUCTIONS, TOOLS) if mode == "agent" else (SINGLE_SHOT_INSTRUCTIONS, [])
    agent = Agent(
        None,  # model is chosen per run
        output_type=Answer,
        deps_type=Deps,
        instructions=_instructions(template),
        tools=tools,
        retries=3,
    )
    agent.output_validator(_check_citations)
    return agent


AGENTS: dict[Mode, Agent[Deps, Answer]] = {"agent": _build_agent("agent"), "single_shot": _build_agent("single_shot")}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_THREAD_URL_SEGMENT = {"pr": "pull", "issue": "issues", "discussion": "discussions"}


def _github_url(repo: dict, ev: Evidence, ref: str) -> str:
    base = f"https://github.com/{repo['owner']}/{repo['name']}"
    if ev.kind in ("code", "doc"):
        path, span = ref.rsplit(":", 1)
        start, _, end = span.partition("-")
        return f"{base}/blob/{repo['commit_sha']}/{path}#L{start}-L{end or start}"
    if ev.kind == "commit":
        return f"{base}/commit/{ref.split(':', 1)[1]}"
    if ev.kind == "release":
        return f"{base}/releases/tag/{ref.split(':', 1)[1]}"
    return f"{base}/{_THREAD_URL_SEGMENT[ev.kind]}/{ref.split('#', 1)[1]}"


def _citations(repo: dict, refs: list[str], evidence: dict[str, Evidence]) -> list[Citation]:
    citations = []
    for ref in refs:
        resolved = resolve_citation(ref, evidence)
        if resolved is None:
            continue
        canonical, ev = resolved
        snippet = ev.text
        if ev.start_line is not None and ":" in canonical:
            start, _, end = canonical.rsplit(":", 1)[1].partition("-")
            offset = int(start) - ev.start_line
            snippet = "\n".join(ev.text.splitlines()[offset:offset + int(end or start) - int(start) + 1])
        path, start, end = ev.path, ev.start_line, ev.end_line
        if ev.path:
            start, end = (int(x) for x in canonical.rsplit(":", 1)[1].split("-"))
        citations.append(Citation(canonical, ev.kind, _github_url(repo, ev, canonical), ev.title,
                                  path, start, end, snippet[:2000]))
    return citations


def _usage_dict(usage) -> dict:
    return {"requests": usage.requests, "tool_calls": usage.tool_calls,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def _single_shot_context(deps: Deps, question: str) -> str:
    code_hits = retrieval.search(deps.repo_id, question, retrieval.CODE_KINDS, k=6)
    history_hits = retrieval.search(deps.repo_id, question, retrieval.HISTORY_KINDS, k=4)
    blocks = []
    for hit in code_hits:
        deps.add(_hit_evidence(hit))
        blocks.append(_format_code_hit(hit))
    for hit in history_hits:
        deps.add(Evidence(ref=hit.ref, kind=hit.kind, text=hit.text[:1500], title=hit.title))
        blocks.append(f"[{hit.ref}] {(hit.created_at or '')[:10]} {hit.title or ''}\n{hit.text[:1500]}")
    return "\n\n".join(blocks)


def _emit_tool_event(event, deps: Deps, emit: Callable[[dict], None]) -> None:
    if isinstance(event, FunctionToolCallEvent):
        emit({"type": "tool_call", "id": event.part.tool_call_id, "tool": event.part.tool_name,
              "args": event.part.args_as_dict()})
    elif isinstance(event, FunctionToolResultEvent):
        call = deps.trace.get(event.part.tool_call_id)
        emit({"type": "tool_result", "id": event.part.tool_call_id,
              "summary": call.summary if call else "error", "refs": call.refs[:10] if call else []})


async def answer(repo_id: str, question: str, mode: Mode = "agent", model: "str | Model | None" = None,
                 on_event: Callable[[dict], None] | None = None) -> Result:
    """Answer `question` about an ingested repo. `on_event` receives live trace events (agent mode)."""
    with store.session() as conn:
        repo = store.get_repo(conn, repo_id)
    if repo is None or not repo["stats"]:
        raise ValueError(f"Repository {repo_id!r} is not indexed")

    deps = Deps(repo=repo)
    emit = on_event or (lambda event: None)
    started = time.monotonic()

    if mode == "agent":
        prompt = question
    else:
        prompt = f"Question: {question}\n\nRetrieved evidence:\n\n{_single_shot_context(deps, question)}"

    output, error, run = None, None, None
    try:
        async with AGENTS[mode].iter(prompt, model=model or get_settings().agent_model, deps=deps,
                                     usage_limits=AGENT_LIMITS) as run:
            async for node in run:
                if Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as events:
                        async for event in events:
                            _emit_tool_event(event, deps, emit)
            output = run.result.output
    except UsageLimitExceeded as exc:
        error = f"Stopped at step budget: {exc}"
    except UnexpectedModelBehavior as exc:  # e.g. citations still invalid after all retries
        error = f"Model failed to produce a valid answer: {exc}"

    if output is None:
        output = Answer(answer="I couldn't reach a grounded answer within the step budget.", citations=[], abstained=True)
    final = Result(
        question=question, repo_id=repo_id, mode=mode, answer=output.answer,
        citations=_citations(repo, output.citations, deps.evidence), abstained=output.abstained,
        trace=list(deps.trace.values()), evidence=deps.evidence,
        usage=_usage_dict(run.usage) if run else {}, latency_s=round(time.monotonic() - started, 2), error=error,
    )
    emit({"type": "answer", "result": final.to_dict()})
    return final


def answer_sync(repo_id: str, question: str, mode: Mode = "agent", model: "str | Model | None" = None,
                on_event: Callable[[dict], None] | None = None) -> Result:
    return asyncio.run(answer(repo_id, question, mode, model, on_event))
