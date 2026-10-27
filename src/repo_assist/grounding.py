"""Answer schema and the checks that keep answers grounded in evidence the model was shown.

Two guardrails run on every final answer, in code rather than in the prompt:

1. Citations must resolve to evidence returned by a tool during this run (a code
   citation must fall inside a line range the model actually read).
2. Answers to rationale questions ("why ...?") must quote the stated reason
   verbatim from that evidence. A reason that can't be quoted from a maintainer's
   words is an inference, and the model is told to abstain instead.

Failed checks are returned as problems; the agent turns them into a retry.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

MIN_QUOTE_CHARS = 20


class Quote(BaseModel):
    ref: str = Field(description="Ref of the evidence the quote is copied from, e.g. 'issue#572'.")
    quote: str = Field(description="Exact text copied from that evidence (a sentence or clause).")


class Answer(BaseModel):
    answer: str = Field(description=(
        "Concise Markdown answer. Support claims with inline markers like [1], [2] that index "
        "(1-based) into `citations`."))
    citations: list[str] = Field(description=(
        "Evidence refs exactly as shown in tool results, e.g. 'httpx/_client.py:760-769', "
        "'PR#1274', 'issue#572', 'discussion#1633', 'release:v10.0.0', 'commit:b5addb64f0'. A code/doc ref may "
        "narrow to a sub-range of lines you were shown."))
    rationale_quotes: list[Quote] = Field(default_factory=list, description=(
        "For 'why' questions: verbatim quotes in which a maintainer or the docs state the reason. "
        "Required for a non-abstained answer to a 'why' question; leave empty otherwise."))
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


_RANGE_REF = re.compile(r"^(?P<path>.+?)(?::|#L)(?P<start>\d+)(?:-L?(?P<end>\d+))?$")
_MARKER = re.compile(r"\[(\d+)\]")
_RATIONALE = re.compile(r"\bwhy\b|\breasons?\b|\brationale\b|\bmotivat|\btrade-?offs?\b", re.IGNORECASE)


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


def is_rationale_question(question: str) -> bool:
    return bool(_RATIONALE.search(question))


def _normalise_text(text: str) -> str:
    """Compare quotes modulo whitespace, case, Markdown emphasis and quote styles."""
    text = re.sub(r"[`*_>#]", "", text.lower()).replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip(" .")


def quote_is_supported(quote: Quote, evidence: dict[str, Evidence]) -> bool:
    resolved = resolve_citation(quote.ref, evidence)
    if resolved is None:
        return False
    needle = _normalise_text(quote.quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in _normalise_text(resolved[1].text)


def validate_answer(answer: Answer, evidence: dict[str, Evidence], question: str = "") -> list[str]:
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
    for q in answer.rationale_quotes:
        if not quote_is_supported(q, evidence):
            problems.append(f"rationale quote {q.quote[:60]!r} was not found verbatim in the evidence for {q.ref!r} "
                            f"(quotes must be copied exactly, at least {MIN_QUOTE_CHARS} characters)")
    if is_rationale_question(question) and not answer.abstained and not answer.rationale_quotes:
        problems.append("this asks for a reason: add rationale_quotes copied verbatim from evidence that states the "
                        "reason, or set abstained=true if no source states it")
    return problems
