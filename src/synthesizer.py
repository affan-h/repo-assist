"""
Synthesizer stage of the query/verifier layer. Takes a question + raw tool
results (already gathered by the router/executor from query_tools.py) and
produces a schema-validated answer.

Deliberately generic over model string -- Agent(model_string, ...) -- so we
can run the same synthesizer against multiple free-tier models and compare
real judged scores before picking one, rather than guessing which is "best"
from reputation. See test_structured_output.py for the real availability
check that grounded these model strings.

RUN FROM src/: python3 synthesizer.py   (runs a small manual smoke test)
"""

import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class SynthesizedAnswer(BaseModel):
    """Shape driven directly by grader.py's real rubric, not guessed:
    - why: rationale + citation must be separable (rule: "rationale without
      citation = 0", "wrong rationale with citation = 0").
    - unanswerable_why: needs an explicit, stated-reason decline (rule:
      "declines with a stated reason" = 1 vs "confident explanation without
      citing" = 0; declines without clear reason = 0.5).
    Building these as distinct fields means grading (ours or grader.py's)
    reads structured data, not prose that has to be parsed apart after
    the fact.

    CITATION REDESIGN (confirmed real failure mode, full eval runs): the
    original free-text `citation` field let the model name a plausible-
    sounding but WRONG source (e.g. "cited PR #2380 instead of Discussion
    #1633", "cited documentation/2-options.md instead of documentation/
    3-streams.md") -- a known, documented RAG failure pattern (post-hoc
    citation matching is unreliable; see Pleias-RAG's finding that models
    should cite AS they reason, not attach a citation after the fact).
    Fix: citation_source_id must be one of the EXPLICIT source identifiers
    actually present in this call's evidence (injected into the prompt as
    a numbered list built from real tool_results, not invented by the
    model) -- a closed-list pick, not free text, so a wrong citation is
    structurally impossible rather than merely discouraged by instruction.
    """
    answer: str = Field(description="The direct answer to the question, in plain prose. "
                                     "NEVER leave this blank, even when abstained=True -- "
                                     "in that case, put a short statement like 'I don't know' "
                                     "or 'No documented rationale was found' here.")
    citation_source_id: str | None = Field(
        default=None,
        description="The EXACT source_id (e.g. 'PR#2306', 'DISCUSSION#3007', "
                    "'ISSUE#572', 'DOC#3') of the evidence source that supports "
                    "this answer, copied EXACTLY from the numbered source list "
                    "given in the prompt. Do NOT invent a source_id or citation "
                    "that isn't in that list -- if the real citation you want to "
                    "name isn't in the list, that means it wasn't actually found "
                    "in the evidence, so you should abstain instead. Null if no "
                    "specific citation applies (e.g. a 'what'/'where' factual "
                    "answer verifiable directly from the graph data) or if abstaining."
    )
    abstained: bool = Field(
        description="True if the provided tool results do not contain enough "
                    "evidence to answer confidently. Must be True whenever the "
                    "honest answer is 'I don't know' or 'no documented rationale "
                    "found' -- do not guess or use general knowledge to fill gaps "
                    "the tool results don't cover."
    )
    abstain_reason: str | None = Field(
        default=None,
        description="If abstained=True, a specific, stated reason naming what "
                    "was searched and not found (e.g. 'no PR, commit, or discussion "
                    "in the provided data documents why this exists'). Required "
                    "whenever abstained=True."
    )


SYSTEM_PROMPT = """You are answering questions about a codebase using ONLY the \
tool results provided to you in the user message. You do not have access to \
the actual repository, and you must not use general knowledge, memory, or \
assumptions about this codebase (even if you recognize it) to fill in facts \
that are not present in the provided tool results.

Rules you must follow exactly:

1. Base your answer strictly on the provided tool results. If a fact isn't in
   them, you do not know it.
2. A numbered SOURCE LIST is provided below the tool results, built directly
   from real evidence -- this includes CODE sources (e.g. "[CODE#Client.get]
   method in httpx/_client.py...", real resolved symbols/snippets) as well as
   "why"-style sources (e.g. "[PR#2306] Title: ...", "[DISCUSSION#3007] Title:
   ..."). Two distinct cases:
   (a) DIRECT FACTS about what code does, where it lives, or how it works
       (what/where/how-style questions) are answerable straight from the
       tool results (summaries, source snippets, graph edges) EVEN IF there
       is no PR/Issue/Discussion/doc citation for them -- a citation is
       OPTIONAL here (citation_source_id may be a real "CODE#..." source_id
       if one directly supports it, or null if the graph/code data itself
       is the evidence). Do NOT abstain just because no PR/Issue/Discussion
       citation exists for a direct code fact -- abstaining on a real,
       answerable code-structure question is itself an error.
   (b) RATIONALE/reasoning ("why" questions) genuinely requires a citation
       from the source list (PR/Issue/Discussion/doc/release) -- for these,
       if the tool results contain a clear, direct answer: answer
       confidently, set abstained=False, and set citation_source_id to the
       EXACT source_id string (e.g. "PR#2306") from the list. Copy it
       exactly -- do not paraphrase, reformat, or invent a source_id that
       isn't printed in that list. If the rationale source you'd want to
       cite isn't in the list, that means it wasn't actually found in the
       evidence -- treat this the same as rule 3 (abstain) rather than
       naming an unlisted source.
   Put any necessary explanation of the rationale in the `answer` field, in
   your own words -- never quote source text verbatim.
3. If the tool results do NOT contain enough evidence to answer confidently --
   especially for "why" questions where no PR, commit message, discussion, or
   doc section actually explains the rationale -- you MUST set abstained=True
   and give a specific abstain_reason naming what you looked for and didn't
   find (e.g. "no PR, commit, or discussion in the provided data documents why
   this exists"). Do NOT produce a plausible-sounding guess. A confident-sounding
   wrong or unsupported answer is worse than an honest "I don't know." Even when
   abstained=True, the `answer` field must still contain a short, plain-language
   statement (e.g. "I don't know" or "No documented rationale was found for this")
   -- never leave `answer` blank, so abstained responses look consistent
   regardless of which model produced them.
4. Never cite a source_id you are not certain appears verbatim in the numbered
   source list. When MULTIPLE sources seem plausible (e.g. several
   DISCUSSION# entries), do not just pick the first or most recent one --
   check each candidate's title/content against what you are ABOUT TO WRITE
   as your rationale, and prefer the one whose content most specifically and
   directly matches your actual answer. Where a relevance_score is shown,
   treat a higher score as a signal of stronger topical relevance, but the
   real test is always: does this specific source's title/content actually
   support the specific claim you are making, not just "is it about a
   related topic."
"""


def build_agent(model_string: str) -> Agent:
    """Fresh Agent per model string -- lets us compare providers side by side
    without any shared state between them.

    FAST, REAL FIX: temperature=0 explicitly set. CONFIRMED real problem
    (identical question, identical evidence, 3 runs): default sampling
    randomness caused the SAME input to fabricate a rationale 1 out of 3
    times on an unanswerable_why question -- a real, safety-relevant
    reliability gap, not a data/retrieval bug. Low temperature won't
    eliminate this entirely (structured-output models still have some
    inherent variance), but should meaningfully reduce it for this
    grounding/abstention task, which needs determinism, not creativity."""
    from pydantic_ai.settings import ModelSettings
    return Agent(
        model_string,
        output_type=SynthesizedAnswer,
        instructions=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.0),
    )


def _build_source_list(tool_results: dict) -> tuple[str, set[str]]:
    """Builds the numbered, explicit source list injected into the prompt,
    and returns the set of valid source_ids for post-generation validation.
    Only sources with a real, distinct identifier are listed -- summaries/
    graph data don't get a source_id since they're not "why"-style citations,
    they're direct facts (matches the schema's "null if a graph-verifiable
    fact" allowance)."""
    lines = []
    valid_ids = set()

    for sym in tool_results.get("resolve_symbol_reference", []) or []:
        sid = f"CODE#{sym.get('qualified_name')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] {sym.get('kind')} in {sym.get('file_path')} (lines {sym.get('start_line')}-{sym.get('end_line')})")

    for snippet in tool_results.get("get_source_snippet", []) or []:
        sid = f"CODE#{snippet.get('file_path')}:{snippet.get('start_line')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] Real source code, {snippet.get('file_path')} lines {snippet.get('start_line')}-{snippet.get('end_line')}")

    for src in tool_results.get("search_source_code", []) or []:
        sid = f"CODE#{src.get('file_path')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] Real source file (full-text matched), {src.get('file_path')}")

    for pr in tool_results.get("get_pr", []) or []:
        sid = f"PR#{pr.get('pr_number')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] Title: {pr.get('title')}")

    for issue in tool_results.get("get_issue", []) or []:
        sid = f"ISSUE#{issue.get('issue_number')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] Title: {issue.get('title')}")

    # CONFIRMED REAL FIX: discussions previously showed NO relevance score
    # to the model, even when we'd already computed one -- confirmed real
    # case (Y4): the correct discussion (#1530, score 17.45, exact topical
    # match) was present in the list alongside a less-relevant one (#2119,
    # about API naming, not resource-leak rationale), and the model picked
    # the wrong one because nothing in the prompt signaled which was more
    # relevant. Fix: show the real score, sort discussions by it (highest
    # first), and instruct the model explicitly to prefer higher-scored,
    # more topically specific sources over ones just earlier in the list.
    discussions = tool_results.get("discussion_candidates_full", []) or []
    discussions_sorted = sorted(discussions, key=lambda d: -(d.get("_score") or 0))
    for disc in discussions_sorted:
        sid = f"DISCUSSION#{disc.get('discussion_number')}"
        valid_ids.add(sid)
        score_val = disc.get("_score")
        score_note = f", relevance_score={score_val}" if score_val is not None else ""
        lines.append(f"[{sid}] Title: {disc.get('title')}{score_note}")

    for release in tool_results.get("search_releases", []) or []:
        sid = f"RELEASE#{release.get('tag_name')}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] Name: {release.get('name')}")

    for i, doc in enumerate(tool_results.get("search_docs", []) or []):
        sid = f"DOC#{i}"
        valid_ids.add(sid)
        lines.append(f"[{sid}] File: {doc.get('file_path')}, Section: {doc.get('heading')}")

    if not lines:
        return "(no citable sources at all in evidence -- if this is a 'why' question with no code/doc/PR/issue/discussion evidence whatsoever, you should likely abstain; for what/where/how questions, answer directly from the tool results without a citation)", valid_ids

    return "\n".join(lines), valid_ids


def synthesize(model_string: str, question: str, repo: str, tool_results: dict) -> SynthesizedAnswer:
    """tool_results: raw dict of {tool_name: result} as gathered by the
    router/executor -- e.g. {"search_symbols": [...], "get_commit_history": [...]}.
    Formatted compactly and handed to the model as the only source of truth."""
    agent = build_agent(model_string)

    source_list_text, valid_source_ids = _build_source_list(tool_results)

    context_block = f"Repo: {repo}\nQuestion: {question}\n\nTool results:\n"
    for tool_name, result in tool_results.items():
        context_block += f"\n--- {tool_name} ---\n{result}\n"

    context_block += (
        f"\n\nNUMBERED SOURCE LIST (citation_source_id MUST be copied exactly "
        f"from this list, or left null):\n{source_list_text}\n"
    )

    result = agent.run_sync(context_block)
    output = result.output

    # POST-GENERATION VALIDATION -- defense in depth, not just trusting the
    # prompt (same claim-level grounding-check principle found in real RAG
    # research: verify a cited source is actually present, don't just ask
    # nicely and hope). If the model names a source_id that isn't in the
    # real, injected list, that's a hallucinated citation -- correct it to
    # an honest abstention rather than silently passing it through.
    #
    # CONFIRMED REAL BUG, FIXED: a model (Groq) cited "[CODE#extend]" --
    # WITH surrounding brackets included in the field value -- for a
    # genuinely real, present source ("CODE#extend"). The exact-match check
    # below incorrectly flagged this as invalid and overwrote a correct
    # answer with a forced abstention. Fixed by normalizing (stripping
    # brackets/whitespace) before comparing, since the model copying the
    # bracketed DISPLAY format instead of just the bare ID is a formatting
    # quirk, not evidence of a hallucinated source.
    raw_citation = output.citation_source_id
    normalized_citation = raw_citation.strip().strip("[]").strip() if raw_citation else None

    if normalized_citation and normalized_citation not in valid_source_ids:
        output.abstain_reason = (
            f"Model cited '{raw_citation}', which is not a real "
            f"source present in the gathered evidence -- treated as an invalid/"
            f"hallucinated citation and corrected to abstain."
        )
        output.citation_source_id = None
        output.abstained = True
        output.answer = "I don't know (citation could not be verified against real evidence)."
    elif normalized_citation:
        output.citation_source_id = normalized_citation  # store the clean form

    return output


def synthesize_with_fallback(
    primary_model: str,
    question: str,
    repo: str,
    tool_results: dict,
    fallback_model: str = "groq:llama-3.3-70b-versatile",
    max_retries: int = 2,
    retry_delay_seconds: float = 2.0,
) -> tuple[SynthesizedAnswer, str]:
    """Retries the primary model up to max_retries times (with a short delay
    between attempts) before falling back to a different model entirely.

    REAL, CONFIRMED NEED: compare_models.py showed Cerebras returning
    queue_exceeded/429 errors on 2 of 3 real test questions -- a real,
    observed infrastructure flakiness, not hypothetical. Retrying first
    catches the common "brief traffic blip" case without silently changing
    which model produced the answer; only falling back after retries are
    exhausted keeps model-switching as a last resort, not a first response
    to any transient error.

    Returns (answer, model_actually_used) -- the model string is ALWAYS
    returned explicitly, never hidden, so a caller building an eval report
    can see exactly which questions were answered by the fallback model
    rather than assuming every answer came from primary_model.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return synthesize(primary_model, question, repo, tool_results), primary_model
        except Exception as e:
            last_error = e
            print(f"  [{primary_model} attempt {attempt + 1}/{max_retries} failed: "
                  f"{type(e).__name__}: {e}]")
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)

    print(f"  [{primary_model} exhausted {max_retries} retries, falling back to {fallback_model}]")
    try:
        return synthesize(fallback_model, question, repo, tool_results), fallback_model
    except Exception as e:
        # both primary and fallback failed -- surface this honestly rather
        # than swallow it; the caller needs to know this question got NO answer.
        raise RuntimeError(
            f"Both primary ({primary_model}) and fallback ({fallback_model}) failed. "
            f"Primary's last error: {last_error}. Fallback's error: {e}"
        ) from e


if __name__ == "__main__":
    # Manual smoke test with fabricated tool results -- NOT real query_tools.py
    # output yet. Purpose: confirm the synthesizer's schema/prompt behave
    # sensibly end-to-end before wiring it to the real router/executor.
    fake_results_answerable = {
        "get_commit_history": [
            {"commit_hash": "abc123", "message": "Require Node.js 10 (#633651f)",
             "pr_number": None}
        ],
        "get_pr": None,
        "search_releases": [
            {"tag_name": "v10.0.0", "name": "v10.0.0",
             "body": "Why: This is so that we can use stream.pipeline for more "
                      "reliable stream handling. Node.js 8 will be out of LTS "
                      "at the end of this month anyway."}
        ],
    }
    fake_results_unanswerable = {
        "get_commit_history": [
            {"commit_hash": "def456", "message": "Fix typo", "pr_number": None}
        ],
        "get_pr": None,
        "find_linked_discussion": [],
        "search_docs": [],
        "search_releases": [],
    }

    for label, model in [("Groq", "groq:llama-3.3-70b-versatile"),
                          ("Cerebras gpt-oss-120b", "cerebras:gpt-oss-120b")]:
        print(f"\n=== {label}: answerable case (got Node.js 10 requirement) ===")
        out = synthesize(model, "Why does got require Node.js 10?", "got", fake_results_answerable)
        print(out)

        print(f"\n=== {label}: unanswerable case (fabricated empty evidence) ===")
        out = synthesize(model, "Why was the retry backoff formula changed to exponential?", "got", fake_results_unanswerable)
        print(out)
