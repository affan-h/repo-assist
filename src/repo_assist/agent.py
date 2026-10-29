"""Question answering: the tool-using agent, plus a single-shot RAG baseline used by the evaluation.

Both share one answer schema and the grounding checks in grounding.py: a final
answer is accepted only if its citations and rationale quotes are backed by
evidence a tool returned during this run; otherwise the model retries with the
problems spelled out, so ungrounded answers can't reach the user.

  agent     the product: tools over the hybrid index, the checkout and git history
  baseline  classic RAG for comparison: one hybrid retrieval, one LLM call, no tools
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from repo_assist import retrieval, store
from repo_assist.config import get_settings
from repo_assist.grounding import Answer, Evidence, resolve_citation, validate_answer
from repo_assist.retry import RetryingModel
from repo_assist.tools import TOOLS, Deps, ToolCall, format_code_hit, format_history_hit, hit_evidence

Mode = Literal["agent", "baseline"]

AGENT_LIMITS = UsageLimits(request_limit=12, tool_calls_limit=15)


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
    quotes: list[dict]
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


# ---------------------------------------------------------------------------
# Prompts and agents
# ---------------------------------------------------------------------------

_ANSWER_RULES = """\
Answer rules:
- Base every claim on evidence shown to you in this session, never on prior knowledge of the project.
- Cite with [n] markers; `citations` lists refs exactly as shown (you may narrow a code/doc ref to a
  sub-range of the lines you were shown, e.g. 'httpx/_client.py:762-765').
- "Why" questions: copy the sentence(s) in which a maintainer, commit, release note or doc states the
  reason into `rationale_quotes`, verbatim, with the ref they came from. If no evidence states the reason
  in its own words, set abstained=true and say what you checked. Describing what the code does, or that a
  change happened, is not a reason; do not infer plausible motives.
- If the evidence does not answer the question, set abstained=true and briefly say what you looked for.
- If the question contains a false premise, say so and cite the evidence that contradicts it.
- Be concise: a short paragraph or a list. Name the relevant files, functions and line ranges."""

_AGENT_INSTRUCTIONS = """\
You answer questions about the GitHub repository {owner}/{name} at commit {sha}.
Investigate with tools before answering:
- search_code: search source and docs by meaning and keywords. Good first step for most questions.
- find_symbol: definition and call sites of a named function/class/method.
- read_file / grep / list_files: verify details in the actual code, follow calls, find imports or literals.
- search_history / read_thread: commits, issues, PRs, discussions and release notes. Search results point
  at the matching comment; read_thread with the given offset to read it in context. For a specific
  version, read its notes directly: read_thread('release:v<version>').
- code_history: the commits that changed specific lines, with the PRs/issues they reference. The most
  direct route from code to the reason behind it: find the code, trace its lines, read the linked threads.
Search results are excerpts: read the code before relying on details. Stop once the evidence is
sufficient (usually 2-6 tool calls).

""" + _ANSWER_RULES

_BASELINE_INSTRUCTIONS = """\
You answer questions about the GitHub repository {owner}/{name} at commit {sha}, using only the
retrieved evidence provided in the message.

""" + _ANSWER_RULES


def _instructions(template: str) -> Callable[[RunContext[Deps]], str]:
    def render(ctx: RunContext[Deps]) -> str:
        repo = ctx.deps.repo
        return template.format(owner=repo["owner"], name=repo["name"], sha=(repo["commit_sha"] or "HEAD")[:10])
    return render


async def _check_grounding(ctx: RunContext[Deps], answer: Answer) -> Answer:
    problems = validate_answer(answer, ctx.deps.evidence, ctx.deps.question)
    if problems:
        sample = ", ".join(list(ctx.deps.evidence)[:25]) or "(none yet - use the tools first)"
        raise ModelRetry("Fix these problems: " + "; ".join(problems) + f". Valid refs include: {sample}")
    return answer


def _build_agent(instructions: str, tools: list) -> Agent[Deps, Answer]:
    agent = Agent(None, output_type=Answer, deps_type=Deps, instructions=_instructions(instructions),
                  tools=tools, retries=3)  # model is chosen per run
    agent.output_validator(_check_grounding)
    return agent


AGENTS: dict[Mode, Agent[Deps, Answer]] = {
    "agent": _build_agent(_AGENT_INSTRUCTIONS, TOOLS),
    "baseline": _build_agent(_BASELINE_INSTRUCTIONS, []),  # also writes the step-budget fallback answer
}


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


def _usage_dict(*usages) -> dict:
    usages = [u for u in usages if u is not None]
    return {key: sum(getattr(u, key) for u in usages)
            for key in ("requests", "tool_calls", "input_tokens", "output_tokens")}


async def _answer_from_gathered_evidence(deps: Deps, question: str, model) -> tuple[Answer | None, object]:
    """Final, tool-less answer over the evidence the agent collected before its budget ran out."""
    blocks, size = [], 0
    for item in reversed(list(deps.evidence.values())):  # most recent evidence first
        block = f"[{item.ref}] {item.title or ''}\n{item.text[:2000]}"
        if size + len(block) > 40_000:
            break
        blocks.append(block)
        size += len(block)
    prompt = (f"Question: {question}\n\nEvidence gathered while investigating (the investigation ran out of "
              f"steps; answer from this evidence or abstain):\n\n" + "\n\n".join(blocks))
    try:
        result = await AGENTS["baseline"].run(prompt, model=model, deps=deps)
        return result.output, result.usage
    except UnexpectedModelBehavior:
        return None, None


def _baseline_context(deps: Deps, question: str) -> str:
    code_hits = retrieval.search(deps.repo_id, question, retrieval.CODE_KINDS, k=6)
    history_hits = retrieval.search_history(deps.repo_id, question, k=4)
    blocks = []
    for hit in code_hits:
        deps.add(hit_evidence(hit))
        blocks.append(format_code_hit(hit))
    for hit in history_hits:
        deps.add(Evidence(ref=hit.ref, kind=hit.kind, text=hit.text[:1500], title=hit.title))
        blocks.append(format_history_hit(hit, preview_chars=1500))
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
    """Answer `question` about an ingested repo. `on_event` receives live tool events (agent mode)."""
    with store.session() as conn:
        repo = store.get_repo(conn, repo_id)
    if repo is None or not repo["stats"]:
        raise ValueError(f"Repository {repo_id!r} is not indexed")

    deps = Deps(repo=repo, question=question)
    emit = on_event or (lambda event: None)
    started = time.monotonic()

    if mode == "baseline":
        prompt = f"Question: {question}\n\nRetrieved evidence:\n\n{_baseline_context(deps, question)}"
    else:
        prompt = question

    model = RetryingModel(model or get_settings().agent_model)  # free tier: 15 requests/minute
    output, error, run, usage = None, None, None, None
    try:
        async with AGENTS[mode].iter(prompt, model=model, deps=deps, usage_limits=AGENT_LIMITS) as run:
            async for node in run:
                if Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as events:
                        async for event in events:
                            _emit_tool_event(event, deps, emit)
            output = run.result.output
    except UsageLimitExceeded:
        # Out of steps: answer from what was gathered rather than returning nothing.
        error = "Step budget reached; answered from the evidence gathered so far"
        if deps.evidence:
            output, usage = await _answer_from_gathered_evidence(deps, question, model)
    except UnexpectedModelBehavior as exc:  # e.g. citations still invalid after all retries
        error = f"Model failed to produce a valid answer: {exc}"

    if output is None:
        output = Answer(answer="I couldn't reach a grounded answer within the step budget.", citations=[], abstained=True)
    final = Result(
        question=question, repo_id=repo_id, mode=mode, answer=output.answer,
        citations=_citations(repo, output.citations, deps.evidence),
        quotes=[q.model_dump() for q in output.rationale_quotes], abstained=output.abstained,
        trace=list(deps.trace.values()), evidence=deps.evidence,
        usage=_usage_dict(run.usage, usage) if run else {}, latency_s=round(time.monotonic() - started, 2),
        error=error,
    )
    emit({"type": "answer", "result": final.to_dict()})
    return final


def answer_sync(repo_id: str, question: str, mode: Mode = "agent", model: "str | Model | None" = None,
                on_event: Callable[[dict], None] | None = None) -> Result:
    return asyncio.run(answer(repo_id, question, mode, model, on_event))
