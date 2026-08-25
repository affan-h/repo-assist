"""
Multi-Agent Orchestrator (V2 Engine) -- §4.4/§5.1/§5.2

Real fixes vs. the previous draft (see conversation history for the full
audit -- summarized here since this file is the actual deliverable):

  1. The planner now emits a CLOSED Literal ENUM (`agents_to_invoke`) as
     specified in §4.4/§3.5, not free-text search strings. This is the
     actual security mitigation §3.5 describes: a poisoned/weird question
     can only cause the planner to pick among 4 fixed specialists, never
     invoke arbitrary retrieval.
  2. Specialist agents (agents/structural_agent.py, history_agent.py,
     docs_agent.py) are ACTUALLY IMPORTED AND CALLED per the plan, wired
     to the fixed enum -- previously they were dead code; orchestrator.py
     had its own parallel, disconnected retrieval wrappers instead.
  3. v1's real synthesizer.py is reused UNCHANGED (§5.2's explicit
     requirement), not replaced with a second, weaker PydanticAI agent.
     This also means the real CODE#/PR#/ISSUE#/DISCUSSION#/RELEASE#/DOC#
     citation convention (see grader.py's call_your_agent) keeps working
     identically for both engines.
  4. Evidence is merged into the SAME tool_results shape router.py's
     plan_* functions already produce (resolve_symbol_reference,
     get_source_snippet, search_source_code, get_commit_history, get_pr,
     get_issue, discussion_candidates_full, discussion_candidates_brief,
     search_docs, search_releases) -- required because synthesizer.py and
     grader.py's citation-readback code are written against those exact
     keys and are NOT being modified.
  5. verifier_agent.execute() is called with its real (draft, evidence,
     synthesizer_provider) signature and real VerificationResult schema
     (claims_supported / unsupported_claims / missing_agent_suspected),
     enabling the real §4.4 retry branch: a missing-agent (planning) error
     triggers invoking exactly that one extra specialist and re-synthesizing;
     a synthesis error triggers a synthesizer retry with the objection
     appended. Exactly one retry total, never both (§4.4).
  6. `orchestration_runs` (§5.1) is created and written to on every run --
     previously this table didn't exist anywhere, so none of §3.8's
     provider-diversity guarantee was checkable.
  7. Return type kept IDENTICAL to what grader.py's run_eval already
     expects (`run_query(repo, question) -> object with .answer,
     .abstained, .abstain_reason`) -- grader.py itself is not modified.
"""

import json
import os
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import List, Literal, Optional

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pydantic import BaseModel, Field
from pydantic_ai import Agent

import synthesizer
from agents import structural_agent, history_agent, docs_agent, verifier_agent

DB_PATH = "../data/code_graph.db"

# Real fix, Aug 2026: Cerebras now requires payment (402 payment_required
# on gpt-oss-120b) and the Groq fallback path isn't confirmed working
# either -- the only provider confirmed actually working, by direct test,
# is Gemini (via GEMINI_API_KEY, "google:" prefix -- matches what was
# already confirmed working in the original verifier_agent.py draft, kept
# identical here rather than switching to the also-valid "google-gla:"
# alias, to avoid introducing a second untested variable at once).
# Switched planner+synthesizer to Gemini as primary, with a same-vendor
# different-model "fallback" (flash, not flash-lite) rather than a
# cross-vendor one that isn't currently available. This is a real, stated
# degradation from the plan's original Cerebras/Groq design, not a silent
# one -- see verifier_agent.py's docstring for the matching, more
# consequential change to §3.8's provider-diversity guarantee.
PLANNER_PRIMARY = "google:gemini-3.5-flash-lite"
PLANNER_FALLBACK = "google:gemini-3.5-flash"
SYNTH_PRIMARY = "google:gemini-3.5-flash-lite"
SYNTH_FALLBACK = "google:gemini-3.5-flash"

AgentChoice = Literal["structural", "history", "docs_semantic", "docs_keyword"]


class OrchestrationPlan(BaseModel):
    agents_to_invoke: List[AgentChoice] = Field(
        default_factory=lambda: ["structural", "history"],
        description="Which specialists to run for this question. Closed enum -- section 3.5's security mitigation.",
    )
    focus_notes: str = Field(default="", description="Short, free-text hint passed to each invoked agent about what to look for.")
    reasoning: str = Field(default="", description="The planner's own stated reasoning, logged every run.")


PLANNER_INSTRUCTIONS = (
    "You are the planning agent for a codebase-intelligence system covering two repos "
    "(httpx, a Python HTTP client; got, a TypeScript HTTP client).\n"
    "Given a user's question, decide which specialist agents should run:\n"
    "  - 'structural': the question is about WHAT code does, WHERE it lives, or the "
    "call/topology structure (e.g. 'what does X do', 'where is Y defined', 'what calls Z').\n"
    "  - 'history': the question is about WHY something exists or was changed -- commit/PR/"
    "issue/discussion rationale.\n"
    "  - 'docs_semantic': the question is conceptual/behavioral and may not share exact "
    "keywords with the docs (e.g. 'how does retry behavior work' without using the word "
    "'retry' the docs use).\n"
    "  - 'docs_keyword': the question uses specific technical terms likely to appear "
    "verbatim in README/CHANGELOG/docs.\n"
    "Most questions need 'structural' plus one of the docs/history modes. 'why' questions "
    "should almost always include 'history'. Pick 1-3 agents, not all 4 by default -- "
    "only include an agent if it's plausibly relevant.\n"
    "State your reasoning briefly, and give a short focus_notes hint (what specifically to "
    "look for) for the invoked agents."
)

_planner_agents: dict = {}


def _get_planner(model: str) -> Agent:
    if model not in _planner_agents:
        _planner_agents[model] = Agent(model, output_type=OrchestrationPlan, retries=1, instructions=PLANNER_INSTRUCTIONS)
    return _planner_agents[model]


class GraderResponse:
    """Kept identical to what grader.py's run_eval already reads
    (v2_raw.answer / v2_raw.abstained / v2_raw.abstain_reason)."""
    def __init__(self, answer, abstained=False, abstain_reason=""):
        self.answer = answer
        self.abstained = abstained
        self.abstain_reason = abstain_reason


ABSTAIN_TEXT = "I searched the codebase, documentation, issues, and PRs but found no documented rationale for this design decision."


def _init_orchestration_runs(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orchestration_runs (
            run_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            question TEXT NOT NULL,
            engine TEXT NOT NULL,
            plan_json TEXT,
            verification_json TEXT,
            synthesizer_provider TEXT,
            verifier_provider TEXT,
            retry_kind TEXT,
            final_answer TEXT NOT NULL,
            citation TEXT,
            abstained INTEGER NOT NULL,
            latency_seconds REAL NOT NULL,
            llm_call_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _log_run(repo, question, plan, verification, synth_provider, verifier_provider,
             retry_kind, final_answer, citation, abstained, latency, llm_calls):
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            _init_orchestration_runs(conn)
            conn.execute("""
                INSERT INTO orchestration_runs
                (run_id, repo, question, engine, plan_json, verification_json,
                 synthesizer_provider, verifier_provider, retry_kind, final_answer,
                 citation, abstained, latency_seconds, llm_call_count, created_at)
                VALUES (?, ?, ?, 'v2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), repo, question,
                json.dumps(plan.model_dump()) if plan else None,
                json.dumps(verification.model_dump()) if verification else None,
                synth_provider, verifier_provider, retry_kind,
                final_answer, citation, int(bool(abstained)), latency, llm_calls,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
    except Exception as e:
        print(f"  [orchestration_runs] failed to log run: {e}")


def call_with_backoff(agent: Agent, prompt: str, agent_name: str, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return agent.run_sync(prompt)
        except Exception as e:
            err = str(e).lower()

            # REAL FIX, found via a live 56-question run: PerDay quota
            # exhaustion was being treated the same as a transient 429,
            # burning ~155s of backoff (5+10+20+40+80) PER call, TWICE
            # (primary then fallback) -- ~5 minutes wasted per question on
            # a condition that cannot resolve until the daily window
            # rolls over, hours away. Daily-quota errors are distinguished
            # from per-minute/transient ones by the real string
            # "PerDay" in Gemini's quotaId (confirmed from the actual
            # error body: "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
            # and fail IMMEDIATELY, no backoff -- there is nothing to wait
            # for within this run. Per-minute/transient 429s and real
            # network blips still get full backoff below.
            if "perday" in err.replace(" ", ""):
                print(f"  [{agent_name}] DAILY QUOTA exhausted for this model -- failing fast, no backoff (real, unrecoverable within this run).")
                raise

            # Real fix, found via a live 56-question run: a transient DNS/
            # network blip ("nodename nor servname provided, or not known",
            # "Name or service not known", "Connection refused", "Max
            # retries exceeded") partway through a long run previously did
            # NOT match this retry-trigger list -- it fell straight through
            # to the non-retryable path and killed 23 of 56 questions
            # outright (W9 through T6, per that run's log) instead of
            # waiting a few seconds for the network to come back. DNS/
            # connection failures are exactly the kind of transient,
            # recoverable condition retry logic exists for -- a real gap,
            # not a hypothetical one, now closed.
            retryable = (
                "429", "503", "502", "500", "unavailable", "rate limit",
                "exhausted", "queue_exceeded", "timeout",
                "nodename nor servname", "name or service not known",
                "name resolution", "nameresolutionerror", "connection refused",
                "connection reset", "max retries exceeded", "connection aborted",
                "network is unreachable", "temporary failure in name resolution",
                "all connection attempts failed", "connection error",
            )
            if any(c in err for c in retryable):
                if attempt < max_retries - 1:
                    sleep_s = 5 * (2 ** attempt)
                    print(f"  [{agent_name} rate-limited/network-blip] waiting {sleep_s}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_s)
                    continue
            # Real fix: non-retryable errors (bad model name, missing API
            # key, payment_required, etc.) were previously re-raised
            # silently here with no print -- found via a real 402 from
            # Cerebras where the FALLBACK provider's own failure never
            # printed anything, making it look like a mysterious total
            # abstention rather than "both providers failed, here's why
            # each one failed." Always print before re-raising.
            print(f"  [{agent_name}] non-retryable error: {str(e)[:300]}")
            raise


def _run_specialist(agent_name: str, repo: str, question: str, focus_notes: str) -> dict:
    if agent_name == "structural":
        return structural_agent.execute(repo, question, focus_notes)
    elif agent_name == "history":
        return history_agent.execute(repo, question, focus_notes)
    elif agent_name in ("docs_semantic", "docs_keyword"):
        return docs_agent.execute(repo, question, focus_notes)
    return {}


def _merge_into_tool_results(tool_results: dict, agent_name: str, agent_out: dict) -> None:
    """Real merge into router.py's tool_results shape, so synthesizer.py
    (unchanged v1 code) and grader.py's citation readback keep working."""
    if agent_name == "structural":
        tool_results.setdefault("resolve_symbol_reference", [])
        tool_results["resolve_symbol_reference"] += agent_out.get("resolved_symbols", [])
        tool_results.setdefault("get_source_snippet", [])
        tool_results["get_source_snippet"] += agent_out.get("source_snippets", [])
        tool_results.setdefault("search_source_code", [])
        tool_results["search_source_code"] += agent_out.get("source_search", [])
        if agent_out.get("call_chain") is not None:
            tool_results["_trace_call_chain"] = agent_out["call_chain"]
        tool_results.setdefault("get_callers", [])
        tool_results["get_callers"] += agent_out.get("callers", [])
        tool_results.setdefault("get_callees", [])
        tool_results["get_callees"] += agent_out.get("callees", [])
        if agent_out.get("centrality"):
            tool_results["centrality"] = agent_out["centrality"]

    elif agent_name == "history":
        tool_results.setdefault("get_commit_history", [])
        tool_results["get_commit_history"] += agent_out.get("commits", [])
        tool_results.setdefault("get_pr", [])
        tool_results["get_pr"] += agent_out.get("prs", [])
        tool_results.setdefault("get_issue", [])
        tool_results["get_issue"] += agent_out.get("issues", [])
        tool_results.setdefault("discussion_candidates_full", [])
        tool_results["discussion_candidates_full"] += agent_out.get("discussions_full", [])
        tool_results.setdefault("discussion_candidates_brief", [])
        tool_results["discussion_candidates_brief"] += agent_out.get("discussions_brief", [])
        tool_results.setdefault("search_docs", [])
        tool_results["search_docs"] += agent_out.get("doc_hits", [])
        tool_results.setdefault("search_releases", [])
        tool_results["search_releases"] += agent_out.get("release_hits", [])

    elif agent_name in ("docs_semantic", "docs_keyword"):
        tool_results.setdefault("search_docs", [])
        tool_results["search_docs"] += agent_out.get("keyword_docs", [])

        # REAL BUG FIX, found via a live grader run: semantic_results were
        # being merged into tool_results["search_semantic"], but
        # synthesizer.py's _build_source_list() only ever reads
        # tool_results.get("search_docs", []) -- it has NO knowledge of a
        # "search_semantic" key. This meant every semantic hit (the entire
        # point of v2's embeddings layer, §4.1) was silently invisible to
        # the synthesizer from day one -- confirmed directly: docs_agent
        # found ASGITransport's real summary at score 0.613 in a live
        # trace, but the synthesizer said "the provided tool results do
        # not include" that information, because it was never in a key
        # the source-list builder reads. Per §5.2, synthesizer.py stays
        # unmodified -- so semantic hits are now reshaped into the SAME
        # {file_path, heading, ...} shape _build_source_list() already
        # knows how to read from search_docs, tagged with their real
        # source_type/score so they're still distinguishable in the
        # source list text, and merged into that same key rather than a
        # parallel one the synthesizer can't see.
        tool_results.setdefault("search_docs", [])
        for hit in agent_out.get("semantic_results", []):
            src_type = hit.get("source_type", "semantic")
            src_id = hit.get("source_id", "")
            score = hit.get("score")
            text = hit.get("text", "")
            tool_results["search_docs"].append({
                "file_path": src_id,
                "heading": f"[semantic:{src_type}, score={score:.3f}]" if isinstance(score, float) else f"[semantic:{src_type}]",
                "content": text,
                "chunk_index": None,
            })


def _agent_for_missing(missing: str) -> Optional[str]:
    """Maps the verifier's missing_agent_suspected value to a real
    specialist name -- 'none' means no planning-error retry applies."""
    if missing in ("structural", "history", "docs_semantic", "docs_keyword"):
        return missing
    return None


def run_query(repo: str, question: str) -> GraderResponse:
    """Entry point -- signature UNCHANGED, grader.py calls run_query(repo, question)."""
    repo = repo.split("/")[-1]
    start = time.monotonic()
    llm_calls = 0
    time.sleep(1)  # minimal pacing, section 7.3

    print(f"\n=== V2 ORCHESTRATION RUN: {repo} ===")
    print(f"Question: {question}")

    # 1. Planning -- single-shot, bounded (section 3.3), closed enum (section 3.5)
    plan = None
    try:
        planner = _get_planner(PLANNER_PRIMARY)
        plan_res = call_with_backoff(planner, f"Repository: {repo}\nQuestion: {question}", "Planner")
        plan = plan_res.output
        llm_calls += 1
    except Exception as e:
        print(f"  [Planner] primary failed ({e}), trying fallback provider...")
        try:
            planner = _get_planner(PLANNER_FALLBACK)
            plan_res = call_with_backoff(planner, f"Repository: {repo}\nQuestion: {question}", "Planner")
            plan = plan_res.output
            llm_calls += 1
        except Exception as e2:
            latency = time.monotonic() - start
            reason = f"Planner primary fail: {e}; fallback fail: {e2}"
            _log_run(repo, question, None, None, None, None, "none", ABSTAIN_TEXT, None, True, latency, llm_calls)
            return GraderResponse(ABSTAIN_TEXT, True, reason)

    print(f"  [Planner] agents_to_invoke={plan.agents_to_invoke} reasoning={plan.reasoning[:150]!r}")

    # 2. Specialists run sequentially per plan.agents_to_invoke (section 7.2, 5.2)
    tool_results: dict = {}
    for agent_name in plan.agents_to_invoke:
        try:
            out = _run_specialist(agent_name, repo, question, plan.focus_notes)
            _merge_into_tool_results(tool_results, agent_name, out)
        except Exception as e:
            print(f"  [{agent_name}] agent failed: {e}")

    trace_had_evidence = any(v for k, v in tool_results.items() if isinstance(v, list) and v)
    if not trace_had_evidence:
        latency = time.monotonic() - start
        _log_run(repo, question, plan, None, None, None, "none", ABSTAIN_TEXT, None, True, latency, llm_calls)
        return GraderResponse(ABSTAIN_TEXT, True, "No evidence retrieved by any invoked specialist.")

    # 3. Synthesis -- v1's real synthesizer.py, UNCHANGED (section 5.2)
    try:
        answer, synth_provider = synthesizer.synthesize_with_fallback(
            SYNTH_PRIMARY, question, repo, tool_results, fallback_model=SYNTH_FALLBACK,
        )
        llm_calls += 1
    except Exception as e:
        latency = time.monotonic() - start
        _log_run(repo, question, plan, None, None, None, "none", ABSTAIN_TEXT, None, True, latency, llm_calls)
        return GraderResponse(ABSTAIN_TEXT, True, f"Synthesis fail: {e}")

    if getattr(answer, "abstained", False):
        latency = time.monotonic() - start
        _log_run(repo, question, plan, None, synth_provider, None, "none",
                  answer.answer, getattr(answer, "citation_source_id", None), True, latency, llm_calls)
        return GraderResponse(answer.answer, True, getattr(answer, "abstain_reason", ""))

    # 4. Verification -- different provider than synthesis this run (section 3.8)
    evidence_text = json.dumps(
        {k: (v[:10] if isinstance(v, list) else v) for k, v in tool_results.items()}, default=str
    )[:12000]
    try:
        verification, verifier_provider = verifier_agent.execute(answer.answer, evidence_text, synth_provider)
        llm_calls += 1
    except Exception as e:
        print(f"  [Verifier] failed entirely ({e}); accepting synthesizer output unverified.")
        latency = time.monotonic() - start
        _log_run(repo, question, plan, None, synth_provider, None, "none",
                  answer.answer, getattr(answer, "citation_source_id", None), False, latency, llm_calls)
        return GraderResponse(answer.answer, False, "")

    retry_kind = "none"

    # 5. Exactly one retry, branching on planning-error vs synthesis-error (section 4.4)
    if not verification.claims_supported:
        missing = _agent_for_missing(verification.missing_agent_suspected)

        if missing and missing not in plan.agents_to_invoke:
            print(f"  [Verifier] flagged a planning error -- invoking missing specialist '{missing}' once.")
            retry_kind = "planning_error"
            try:
                out = _run_specialist(missing, repo, question, plan.focus_notes)
                _merge_into_tool_results(tool_results, missing, out)
                answer, synth_provider = synthesizer.synthesize_with_fallback(
                    SYNTH_PRIMARY, question, repo, tool_results, fallback_model=SYNTH_FALLBACK,
                )
                llm_calls += 1
            except Exception as e:
                print(f"  [Retry] planning-error retry failed: {e}")
        else:
            print("  [Verifier] flagged a synthesis error -- retrying synthesis with objection.")
            retry_kind = "synthesis_error"
            try:
                tool_results["_verifier_objection"] = (
                    f"NOTE: a prior draft made unsupported claims: {verification.unsupported_claims}. "
                    f"Only state what the evidence actually supports."
                )
                answer, synth_provider = synthesizer.synthesize_with_fallback(
                    SYNTH_PRIMARY, question, repo, tool_results, fallback_model=SYNTH_FALLBACK,
                )
                llm_calls += 1
            except Exception as e:
                print(f"  [Retry] synthesis-error retry failed: {e}")

        if getattr(answer, "abstained", False):
            latency = time.monotonic() - start
            _log_run(repo, question, plan, verification, synth_provider, verifier_provider,
                      retry_kind, answer.answer, getattr(answer, "citation_source_id", None), True, latency, llm_calls)
            return GraderResponse(answer.answer, True, getattr(answer, "abstain_reason", ""))

    final_ans = answer.answer
    citation = getattr(answer, "citation_source_id", None)

    latency = time.monotonic() - start
    _log_run(repo, question, plan, verification, synth_provider, verifier_provider,
              retry_kind, final_ans, citation, False, latency, llm_calls)

    print(f"  [Done] {llm_calls} LLM calls, {latency:.1f}s, retry_kind={retry_kind}")
    return GraderResponse(final_ans)
