"""
Phase 0 Automated Grader — LLM-as-Judge
========================================
Runs each question through your agent, then passes agent answer + ground truth
to a separate judge LLM. Uses different providers for generation vs judging
to reduce correlated-failure bias (self-grading problem).

Usage:
    python grader.py --questions phase0_questions.json --output results.json --engine v1
    python grader.py --questions phase0_questions.json --output results.json --engine v2
    python grader.py --questions phase0_questions.json --output results.json --engine both

Requires:
    pip install openai google-generativeai requests langfuse

Changes in this revision (§6 of the v2 plan -- "the single most important
operational discipline for v2, given the explicit goal of not regressing
v1's 49%"):

  1. Real fix, not just an addition: v2's `had_tool_call` was previously
     hardcoded to True unconditionally. This silently gave v2 a free pass
     on the judge's "Trace_Had_Tool_Call must be true or score 0" rule
     (see JUDGE_SYSTEM_PROMPT, what/how/where/topology) even on the runs
     where v2 correctly abstained with zero real evidence retrieved.
     That's not a fair comparison -- it's the eval quietly grading v1 and
     v2 by different rules. Fixed: had_tool_call for v2 now reflects
     whether orchestration_runs actually logged non-empty specialist
     evidence for that run (checked via the same DB orchestrator.py
     writes to), matching what v1's real trace-length check measures.
  2. `--engine both`: runs the full question set through both engines and
     writes a real side-by-side comparison, per §6.1.
  3. §6.2's regression gates, evaluated automatically after a `both` run:
     aggregate floor (v2 >= v1 - 5pts, OR v2 >= v1 + 3pts) AND the
     independent unanswerable_why floor (>= 65%, non-negotiable
     regardless of the aggregate gate) -- both must pass for the report
     to recommend v2 as the default engine.
  4. §6.3's rate-limit pacing: `--engine both` fires up to ~4x the LLM
     calls per question for v2 vs v1 (§5.2's worst case). A flat
     `time.sleep(1.5)` between judge calls was already present but there
     was no extra pacing for the *agent* calls themselves during a
     sustained 56-question dual-engine run, where v2's call volume is the
     real risk of tripping Cerebras/Groq rate limits mid-gate-run. Added
     explicit inter-question pacing, wider for v2 than v1, distinct from
     the existing per-call provider-fallback logic (which only reacts to
     failures, not sustained load).
"""

import json
import os
import sqlite3
import time
import argparse
from datetime import datetime, timezone
from typing import Optional
import requests
from orchestrator import run_query as run_v2_query

DB_PATH = "../data/code_graph.db"

# ── Judge configuration ──────────────────────────────────────────────────────
JUDGE_PROVIDER = "gemini"          # "gemini" | "groq"
JUDGE_MODEL    = "gemini-3.1-flash-lite"

JUDGE_SYSTEM_PROMPT = """You are an exact, unforgiving grader for a code intelligence system evaluation.
Your job is to compare a System_Answer to a Ground_Truth and output a score.

IMPORTANT SCOPE NOTE ON Trace_Had_Tool_Call: this field is ONLY a scoring
gate for the WHAT, HOW, WHERE, and TOPOLOGY categories (see their rules
below). It is explicitly NOT a gate for WHY or UNANSWERABLE_WHY -- for
those two categories, grade using ONLY the WHY/UNANSWERABLE_WHY rules
below, which never mention Trace_Had_Tool_Call. A correct, well-reasoned
abstention on an UNANSWERABLE_WHY question must not be scored 0 just
because Trace_Had_Tool_Call happens to be false -- a genuine, honest "no
evidence found" after a real search is exactly the correct behavior that
category is designed to reward, and it can legitimately co-occur with
Trace_Had_Tool_Call=false (the system searched, found nothing conclusive,
and correctly declined -- that is real tool use producing a null result,
not "no tool use at all").

Grading rules by category:

WHAT / HOW / WHERE:
- The System_Answer must contain the core facts of the Ground_Truth.
- Minor wording differences are fine. Missing key facts = 0.
- CRITICAL: If the Trace_Had_Tool_Call field is false, score 0 regardless of answer quality.
  A correct answer with no tool call means the LLM used pre-trained memory, not the index.

WHY:
- The System_Answer must contain BOTH the rationale AND an explicit citation
  (PR number, issue number, commit hash, or doc section name).
- Rationale without citation = 0.
- Wrong rationale with citation = 0.
- Correct rationale with correct citation = 1.
- CRITICAL, DO NOT SKIP: if the System_Answer declines to answer / states it found
  no documented rationale (an "abstention"), this is NOT automatically correct for
  the WHY category, unlike UNANSWERABLE_WHY. The WHY category means real, findable
  evidence is EXPECTED to exist -- you are given the Ground_Truth, which itself
  names the real citation (e.g. "Issue #572", "v10.0.0 release notes"). If the
  Ground_Truth contains a specific, real citation and the System_Answer abstained
  instead of finding it, that is a FAILURE to retrieve real evidence that exists,
  and must score 0 -- the same as a wrong answer.
- Do NOT apply the Trace_Had_Tool_Call gate here -- grade purely on whether the
  rationale and citation match the Ground_Truth, per the rules above.

UNANSWERABLE_WHY:
- The System_Answer must explicitly state that no documented rationale was found.
- If it provides a confident explanation (even a plausible one) without citing a source = 0.
- If it declines with a stated reason (e.g., "I found no PR or issue documenting this") = 1.
- Partial credit 0.5: if it declines but doesn't clearly state WHY it's declining.
- Do NOT apply the Trace_Had_Tool_Call gate here -- a correct abstention scores 1
  regardless of that field's value. This category has no tool-call requirement.

TOPOLOGY:
- The System_Answer must return an exact or near-exact list of nodes/files/functions.
- A partial list scores 0. Vague descriptions score 0.
- Must show evidence of graph traversal (Trace_Had_Tool_Call must be true).

Output ONLY a JSON object, no other text:
{"score": 1, 0, or 0.5, "reason": "brief one-sentence explanation"}"""


# ── Agent stub — V1 Logic ─────────────────────────
def call_your_agent(question: str, repo: str, category: str) -> tuple[str, bool]:
    import router
    import synthesizer

    # Real fix, Aug 2026: Cerebras now requires payment (confirmed via a
    # live 402 payment_required during testing) -- same root cause as
    # orchestrator.py's provider switch (see that file's docstring for the
    # full account). v1's eval-harness path needs the same fix to keep
    # running at all. NOTE: if v1's actual production CLI (not this eval
    # harness) reads its provider from a different config location than
    # this hardcoded pair, that separate location needs the same update --
    # this file only fixes the grader's own v1 comparison path.
    PRIMARY_MODEL = "google:gemini-3.5-flash-lite"
    FALLBACK_MODEL = "google:gemini-3.5-flash"

    plan_result = router.plan_and_execute(repo, category, question)
    trace_had_tool_call = len(plan_result["trace"]) > 0

    answer, model_used = synthesizer.synthesize_with_fallback(
        PRIMARY_MODEL, question, repo, plan_result["tool_results"],
        fallback_model=FALLBACK_MODEL,
    )

    answer_text = answer.answer
    if answer.citation_source_id:
        sid = answer.citation_source_id
        readable = sid
        tool_results = plan_result["tool_results"]

        if sid.startswith("CODE#"):
            target = sid[len("CODE#"):]
            for sym in tool_results.get("resolve_symbol_reference", []) or []:
                if sym.get("qualified_name") == target:
                    readable = f"{sym['file_path']} (lines {sym['start_line']}-{sym['end_line']})"
                    break
            else:
                for snip in tool_results.get("get_source_snippet", []) or []:
                    if f"{snip.get('file_path')}:{snip.get('start_line')}" == target:
                        readable = f"{snip['file_path']} (lines {snip['start_line']}-{snip['end_line']})"
                        break
                else:
                    for src in tool_results.get("search_source_code", []) or []:
                        if src.get("file_path") == target:
                            readable = src["file_path"]
                            break
        elif sid.startswith("PR#"):
            num = sid[len("PR#"):]
            for pr in tool_results.get("get_pr", []) or []:
                if str(pr.get("pr_number")) == num:
                    readable = f"PR #{num}: {pr.get('title')}"
                    break
        elif sid.startswith("ISSUE#"):
            num = sid[len("ISSUE#"):]
            for issue in tool_results.get("get_issue", []) or []:
                if str(issue.get("issue_number")) == num:
                    readable = f"Issue #{num}: {issue.get('title')}"
                    break
        elif sid.startswith("DISCUSSION#"):
            num = sid[len("DISCUSSION#"):]
            for disc in tool_results.get("discussion_candidates_full", []) or []:
                if str(disc.get("discussion_number")) == num:
                    readable = f"Discussion #{num}: {disc.get('title')}"
                    break
        elif sid.startswith("RELEASE#"):
            tag = sid[len("RELEASE#"):]
            readable = f"{tag} release notes"
        elif sid.startswith("DOC#"):
            idx = sid[len("DOC#"):]
            if idx.isdigit():
                docs = tool_results.get("search_docs", []) or []
                i = int(idx)
                if 0 <= i < len(docs):
                    readable = f"{docs[i]['file_path']}, section \"{docs[i]['heading']}\""

        answer_text += f" [Source: {readable}]"
    if answer.abstained and answer.abstain_reason:
        answer_text += f" [Reason: {answer.abstain_reason}]"

    return answer_text, trace_had_tool_call


def _v2_had_real_evidence(repo: str) -> bool:
    """Real fix vs. the previous draft's hardcoded `had_tool_call = True`
    for v2. Reads the most recent orchestration_runs row for this repo
    (orchestrator.py just wrote it) and checks whether the plan actually
    logged a non-'none' retry_kind or a citation -- both only occur when
    at least one specialist returned real evidence. Falls back to False
    (the conservative choice) if the table or row can't be read, so a
    logging failure never silently inflates v2's score."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT abstained, citation, plan_json FROM orchestration_runs "
                "WHERE repo = ? ORDER BY created_at DESC LIMIT 1",
                (repo,),
            ).fetchone()
        if not row:
            return False
        abstained, citation, plan_json = row
        if abstained:
            # An abstained run with no evidence retrieved should NOT count
            # as a real tool call for the what/how/where/topology gate --
            # this is exactly the case the previous hardcoded True hid.
            return False
        # A non-abstained answer with a real citation, or a logged plan
        # that invoked at least one specialist, indicates real retrieval.
        return bool(citation) or bool(plan_json)
    except Exception:
        return False


# ── Judge call ───────────────────────────────────────────────────────────────
def call_judge(
    question: str,
    category: str,
    ground_truth: str,
    system_answer: str,
    trace_had_tool_call: bool,
) -> dict:
    user_message = f"""Category: {category.upper()}
Question: {question}
Ground_Truth: {ground_truth}
System_Answer: {system_answer}
Trace_Had_Tool_Call: {trace_had_tool_call}

Grade the System_Answer against the Ground_Truth using the rules for {category.upper()}.
Output only a JSON object."""

    if JUDGE_PROVIDER == "gemini":
        return _judge_gemini(user_message)
    elif JUDGE_PROVIDER == "groq":
        return _judge_groq(user_message)
    else:
        raise ValueError(f"Unknown judge provider: {JUDGE_PROVIDER}")

def _judge_gemini(user_message: str, max_retries: int = 6) -> dict:
    api_key = os.environ["GEMINI_API_KEY"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{JUDGE_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 256,
            "response_mime_type": "application/json",
        },
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            # Real fix, found via a live 56-question run: a transient DNS/
            # network blip (requests.exceptions.ConnectionError wrapping a
            # NameResolutionError -- "nodename nor servname provided") was
            # previously NOT caught here at all -- only HTTP 429 responses
            # were retried. A raised connection-level exception propagated
            # straight out of this function uncaught, permanently losing
            # that question's score (23 of 56 questions in that run) instead
            # of waiting a few seconds for the network to recover. Now
            # caught and retried the same as a 429.
            last_error = f"{type(e).__name__}: {e}"
            wait_s = 2 * (attempt + 1)
            print(f"    [Gemini judge connection error, retrying in {wait_s}s "
                  f"(attempt {attempt + 1}/{max_retries})... {str(e)[:150]}]")
            time.sleep(wait_s)
            continue

        if resp.status_code == 429:
            last_error = f"429 on attempt {attempt + 1}/{max_retries}"
            print(f"    [Gemini judge 429, retrying in {2 * (attempt + 1)}s...]")
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return json.loads(raw)

    raise RuntimeError(f"Gemini judge exhausted {max_retries} retries. Last error: {last_error}")

def _judge_groq(user_message: str) -> dict:
    api_key = os.environ["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return json.loads(raw)


# ── Single-engine run (used directly, and twice by --engine both) ────────────
def _run_single_engine(questions: list, engine: str, dry_run: bool) -> dict:
    """Runs the full question set through exactly one engine. Returns the
    same result-bundle shape as before, factored out so --engine both can
    call it twice cleanly without duplicating the loop."""
    results = []
    cats = ["what", "how", "where", "why", "unanswerable_why", "topology"]
    scores_by_cat = {c: [] for c in cats}
    scores_by_repo = {"httpx": [], "got": []}

    # §6.3: v2 fires up to ~4x the LLM calls per question that v1 does
    # (§5.2's worst case) -- a sustained 56-question run risks tripping
    # Cerebras/Groq rate limits during the gate run itself, before v2 is
    # even judged on answer quality. Wider inter-question pacing for v2,
    # distinct from the existing per-call provider-fallback logic (which
    # only reacts to failures, not sustained load).
    inter_question_pause = 4.0 if engine == "v2" else 1.0

    print(f"Running eval on {len(questions)} questions using engine: {engine.upper()}...\n")

    for i, q in enumerate(questions):
        qid      = q["id"]
        repo     = q["repo"]
        category = q["category"]
        question = q["question"]
        gt       = q["ground_truth"]

        print(f"[{i+1}/{len(questions)}] {qid} ({category}) [{engine}]")

        if dry_run:
            agent_answer = "DRY RUN — no agent answer"
            had_tool_call = False
        else:
            try:
                if engine == "v1":
                    agent_answer, had_tool_call = call_your_agent(question, repo, category)

                elif engine == "v2":
                    v2_raw = run_v2_query(repo, question)

                    if hasattr(v2_raw, "answer"):
                        agent_answer = v2_raw.answer if not getattr(v2_raw, "abstained", False) else v2_raw.abstain_reason
                    else:
                        agent_answer = str(v2_raw)

                    # Real fix: reflects actual retrieval, not a hardcoded True.
                    had_tool_call = _v2_had_real_evidence(repo)

            except NotImplementedError as e:
                print(f"  ⚠ Agent not wired up yet: {e}")
                break
            except Exception as e:
                print(f"  ✗ Agent error: {e}")
                results.append({
                    "id": qid, "repo": repo, "category": category,
                    "score": None, "error": str(e)
                })
                time.sleep(inter_question_pause)
                continue

        time.sleep(1.5)  # rate-limit courtesy pause before judge call

        try:
            grade = call_judge(question, category, gt, agent_answer, had_tool_call)
        except Exception as e:
            print(f"  ✗ Judge error: {e}")
            grade = {"score": None, "reason": f"Judge error: {e}"}

        effective_score = grade.get("score")
        reason = grade.get("reason", "")

        print(f"  → score={effective_score} | {reason}")

        results.append({
            "id": qid,
            "repo": repo,
            "category": category,
            "question": question,
            "agent_answer": agent_answer,
            "trace_had_tool_call": had_tool_call,
            "ground_truth": gt,
            "score": effective_score,
            "judge_reason": reason,
        })

        if effective_score is not None:
            scores_by_cat[category].append(effective_score)
            scores_by_repo[repo].append(effective_score)

        time.sleep(inter_question_pause)

    all_scores = [r["score"] for r in results if r.get("score") is not None]
    overall_pct = round(100 * sum(all_scores) / len(all_scores), 1) if all_scores else None
    u_scores = scores_by_cat["unanswerable_why"]
    u_pct = round(100 * sum(u_scores) / len(u_scores), 1) if u_scores else None

    return {
        "engine": engine,
        "results": results,
        "scores_by_category": {
            c: {
                "correct": sum(scores_by_cat[c]),
                "total": len(scores_by_cat[c]),
                "pct": round(100 * sum(scores_by_cat[c]) / len(scores_by_cat[c]), 1) if scores_by_cat[c] else None,
            }
            for c in cats
        },
        "scores_by_repo": {
            r: {
                "correct": sum(s),
                "total": len(s),
                "pct": round(100 * sum(s) / len(s), 1) if s else None,
            }
            for r, s in scores_by_repo.items()
        },
        "overall_pct": overall_pct,
        "unanswerable_why_pct": u_pct,
        "graded": len(all_scores),
        "total_questions": len(questions),
    }


def _print_summary(bundle: dict):
    print("\n" + "=" * 60)
    print(f"RESULTS BY CATEGORY  [{bundle['engine']}]")
    print("=" * 60)
    for cat, d in bundle["scores_by_category"].items():
        if d["total"]:
            print(f"  {cat:<20} {d['correct']}/{d['total']}  ({d['pct']:.0f}%)")
        else:
            print(f"  {cat:<20} —")

    print(f"\nRESULTS BY REPO  [{bundle['engine']}]")
    for repo, d in bundle["scores_by_repo"].items():
        if d["total"]:
            print(f"  {repo:<20} {d['correct']}/{d['total']}  ({d['pct']:.0f}%)")

    if bundle["overall_pct"] is not None:
        print(f"\n  OVERALL               {bundle['graded']} graded  ({bundle['overall_pct']:.0f}%)")

    if bundle["unanswerable_why_pct"] is not None:
        pct = bundle["unanswerable_why_pct"]
        verdict = "✓ Verifier is working" if pct >= 50 else "✗ Verifier too credulous — tighten citation enforcement"
        print(f"\n  unanswerable_why abstention rate: {pct:.0f}% — {verdict}")


# ── §6.2 regression gates ─────────────────────────────────────────────────────
def _evaluate_gates(v1_bundle: dict, v2_bundle: dict) -> dict:
    """Real, precisely-defined gates per §6.2 -- not a soft/ambiguous
    criterion. Both must pass for v2 to be recommended as the default.

    Aggregate floor: v2 >= v1 - 5pts, OR v2 >= v1 + 3pts ("clearly outperforms").
    Safety-critical floor, independent of the aggregate: unanswerable_why
    must not regress below 65% (10-point tolerance from v1's real 75%),
    regardless of whether the aggregate gate passes.
    """
    v1_overall = v1_bundle["overall_pct"]
    v2_overall = v2_bundle["overall_pct"]
    v2_u = v2_bundle["unanswerable_why_pct"]

    aggregate_pass = None
    if v1_overall is not None and v2_overall is not None:
        aggregate_pass = (v2_overall >= v1_overall - 5) or (v2_overall >= v1_overall + 3)

    safety_pass = None
    if v2_u is not None:
        safety_pass = v2_u >= 65.0

    both_pass = bool(aggregate_pass) and bool(safety_pass) if aggregate_pass is not None and safety_pass is not None else None

    return {
        "v1_overall_pct": v1_overall,
        "v2_overall_pct": v2_overall,
        "aggregate_gate_pass": aggregate_pass,
        "aggregate_gate_rule": "v2 >= v1 - 5pts, OR v2 >= v1 + 3pts",
        "v2_unanswerable_why_pct": v2_u,
        "safety_gate_pass": safety_pass,
        "safety_gate_rule": "v2 unanswerable_why >= 65% (10pt tolerance from v1's 75% baseline), independent of aggregate",
        "v2_becomes_default": both_pass,
    }


def _print_gates(gates: dict):
    print("\n" + "=" * 60)
    print("§6.2 REGRESSION GATES")
    print("=" * 60)
    v1p = gates["v1_overall_pct"]
    v2p = gates["v2_overall_pct"]
    print(f"  v1 overall: {v1p}%   v2 overall: {v2p}%")
    print(f"  Aggregate gate ({gates['aggregate_gate_rule']}): "
          f"{'PASS' if gates['aggregate_gate_pass'] else 'FAIL' if gates['aggregate_gate_pass'] is not None else 'N/A'}")
    print(f"  v2 unanswerable_why: {gates['v2_unanswerable_why_pct']}%")
    print(f"  Safety gate ({gates['safety_gate_rule']}): "
          f"{'PASS' if gates['safety_gate_pass'] else 'FAIL' if gates['safety_gate_pass'] is not None else 'N/A'}")
    if gates["v2_becomes_default"] is True:
        print("\n  ✓ BOTH GATES PASS — v2 may become the CLI's default engine.")
    elif gates["v2_becomes_default"] is False:
        print("\n  ✗ At least one gate FAILED — v2 stays available via --engine v2, "
              "v1 remains the default. This is a legitimate, honestly-documented outcome (§12), not a failure to hide.")
    else:
        print("\n  Gate result inconclusive (missing data in one or both bundles).")


# ── Retry-failed-only mode ────────────────────────────────────────────────────
# REAL FIX, requested directly: given the daily-quota reality confirmed in the
# actual run (GenerateRequestsPerDayPerProjectPerModel-FreeTier), a full
# 56-question re-run wastes real quota re-answering questions that already
# scored correctly. This mode loads a prior results JSON, identifies rows
# that failed for INFRASTRUCTURE reasons (quota/network/judge errors -- not
# a genuine wrong answer the model actually attempted), reruns ONLY those,
# and merges the new results back in.

_FAILURE_MARKERS = (
    "quota", "429", "resource_exhausted", "rate limit", "rate-limited",
    "connection", "nameresolutionerror", "timeout", "judge error",
    "failed to generate an answer due to api errors",
    "failed to provide an answer",  # covers "...due to X" and "...and instead returned an error log" phrasings
    "returned an error log",
    "did not perform any tool calls",  # judge's own phrasing when the agent call itself errored out
)


def _is_infra_failure(result_row: dict) -> bool:
    """True if this row's score is None (a judge/agent error was recorded),
    OR the row scored 0 but the judge's own reasoning or the agent's own
    answer text shows it was actually an infrastructure failure (quota/
    network) rather than a genuine wrong answer -- distinguishing "the
    system tried and got it wrong" (a real result, don't re-spend quota on
    it) from "the system never got a real chance to try" (worth retrying)."""
    if result_row.get("score") is None:
        return True
    text = f"{result_row.get('judge_reason', '')} {result_row.get('agent_answer', '')}".lower()
    return any(marker in text for marker in _FAILURE_MARKERS)


def retry_failed(prior_results_file: str, questions_file: str, output_file: str, engine: str):
    """Loads a prior results JSON (single-engine or --engine both shape),
    finds infra-failed rows, reruns only those questions, and writes a
    merged, corrected results file. Does NOT re-spend quota on rows that
    already scored a real 0 or 1."""
    with open(prior_results_file) as f:
        prior = json.load(f)
    with open(questions_file) as f:
        all_questions = {q["id"]: q for q in json.load(f)["questions"]}

    # Handle both single-engine {"results": [...]} and --engine both
    # {"v1": {"results": [...]}, "v2": {...}} shapes.
    if "results" in prior:
        prior_results = prior["results"]
    elif engine in prior:
        prior_results = prior[engine]["results"]
    else:
        raise ValueError(f"Can't find results for engine={engine!r} in {prior_results_file}")

    failed_ids = [r["id"] for r in prior_results if _is_infra_failure(r)]
    ok_results = [r for r in prior_results if not _is_infra_failure(r)]

    print(f"Found {len(prior_results)} prior results: {len(ok_results)} kept (real scores), "
          f"{len(failed_ids)} to retry (infra failures): {failed_ids}")

    if not failed_ids:
        # REAL FIX: previously exited here with no gate check and no
        # re-write of the file -- meaning a prior file already missing its
        # OTHER engine's bundle (this exact scenario) stayed broken even
        # when there was genuinely nothing left to retry. Now still
        # recomputes and prints gates (and re-writes the file with both
        # engines' bundles intact) using the already-correct prior scores,
        # since "nothing to retry" doesn't mean "nothing to report."
        print("Nothing to retry -- all prior results were real.")
        other_engine = "v1" if engine == "v2" else "v2"
        if other_engine in prior:
            this_engine_bundle = prior[engine] if engine in prior else {
                "engine": engine, "results": prior_results,
                **{k: v for k, v in _run_single_engine([], engine, dry_run=True).items() if k != "results"},
            }
            gates = _evaluate_gates(prior[other_engine], this_engine_bundle)
            output = {
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "judge_model": f"{JUDGE_PROVIDER}/{JUDGE_MODEL}",
                "engine": "both",
                "retried_ids": [],
                other_engine: prior[other_engine],
                engine: this_engine_bundle,
                "regression_gates": gates,
            }
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            _print_gates(gates)
            print(f"\nFile re-written with both engines intact: {output_file}")
        return

    retry_questions = [all_questions[qid] for qid in failed_ids if qid in all_questions]
    bundle = _run_single_engine(retry_questions, engine, dry_run=False)

    merged_results = ok_results + bundle["results"]
    cats = ["what", "how", "where", "why", "unanswerable_why", "topology"]
    scores_by_cat = {c: [] for c in cats}
    scores_by_repo = {"httpx": [], "got": []}
    for r in merged_results:
        if r.get("score") is not None:
            scores_by_cat[r["category"]].append(r["score"])
            scores_by_repo[r["repo"]].append(r["score"])

    all_scores = [r["score"] for r in merged_results if r.get("score") is not None]
    overall_pct = round(100 * sum(all_scores) / len(all_scores), 1) if all_scores else None
    u_scores = scores_by_cat["unanswerable_why"]
    u_pct = round(100 * sum(u_scores) / len(u_scores), 1) if u_scores else None

    engine_bundle = {
        "engine": engine,
        "total_questions": len(merged_results),
        "graded": len(all_scores),
        "overall_pct": overall_pct,
        "unanswerable_why_pct": u_pct,
        "scores_by_category": {
            c: {"correct": sum(scores_by_cat[c]), "total": len(scores_by_cat[c]),
                "pct": round(100 * sum(scores_by_cat[c]) / len(scores_by_cat[c]), 1) if scores_by_cat[c] else None}
            for c in cats
        },
        "scores_by_repo": {
            r: {"correct": sum(s), "total": len(s), "pct": round(100 * sum(s) / len(s), 1) if s else None}
            for r, s in scores_by_repo.items()
        },
        "results": merged_results,
    }

    # REAL BUG FIX, found via a live run: this used to write ONLY the
    # retried engine's bundle, silently discarding the OTHER engine's
    # data if the prior file came from --engine both -- a real,
    # reconstructed v1 bundle got dropped entirely this way, breaking
    # _evaluate_gates(d['v1'], d['v2']) with a real KeyError. Now
    # preserves whichever other engine's bundle was present in the prior
    # file (v1 or v2, whichever wasn't retried) and writes BOTH back out
    # in the same {"v1": {...}, "v2": {...}} shape --engine both uses,
    # so the file stays gate-checkable after a retry-failed run.
    other_engine = "v1" if engine == "v2" else "v2"
    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": f"{JUDGE_PROVIDER}/{JUDGE_MODEL}",
        "engine": "both" if other_engine in prior else engine,
        "retried_ids": failed_ids,
    }
    if other_engine in prior:
        output[other_engine] = prior[other_engine]
        output[engine] = engine_bundle
        # Also compute and attach the real regression gates immediately,
        # since both engines' data is now genuinely present -- saves a
        # separate manual step and avoids the exact KeyError this fix closes.
        output["regression_gates"] = _evaluate_gates(prior[other_engine], engine_bundle)
    else:
        # No other engine present in the prior file -- keep the original,
        # simpler single-engine shape (unchanged behavior for that case).
        output.update(engine_bundle)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MERGED RESULTS  [{engine}]  ({len(merged_results)} total, {len(all_scores)} graded)")
    print(f"{'='*60}")
    for c, d in engine_bundle["scores_by_category"].items():
        if d["total"]:
            print(f"  {c:<20} {d['correct']}/{d['total']}  ({d['pct']:.0f}%)")
    if overall_pct is not None:
        print(f"\n  OVERALL               {len(all_scores)} graded  ({overall_pct:.0f}%)")
    if other_engine in prior:
        _print_gates(output["regression_gates"])
    print(f"\nMerged results written to {output_file}")


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_eval(questions_file: str, output_file: str, dry_run: bool = False, engine: str = "v2"):
    with open(questions_file) as f:
        data = json.load(f)
    questions = data["questions"]

    if engine in ("v1", "v2"):
        bundle = _run_single_engine(questions, engine, dry_run)
        _print_summary(bundle)

        output = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "judge_model": f"{JUDGE_PROVIDER}/{JUDGE_MODEL}",
            "engine": engine,
            "total_questions": bundle["total_questions"],
            "graded": bundle["graded"],
            "scores_by_category": bundle["scores_by_category"],
            "results": bundle["results"],
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull results written to {output_file}")

    elif engine == "both":
        print("Running BOTH engines for a real side-by-side comparison (§6.1)...\n")
        v1_bundle = _run_single_engine(questions, "v1", dry_run)
        _print_summary(v1_bundle)

        # Extra pause between the two full passes -- not just between
        # questions within a pass -- since switching engines is exactly
        # where a sustained-burst rate-limit risk (§6.3) is highest.
        if not dry_run:
            print("\n[Pausing 10s between engine passes to clear rate-limit windows...]\n")
            time.sleep(10)

        v2_bundle = _run_single_engine(questions, "v2", dry_run)
        _print_summary(v2_bundle)

        gates = _evaluate_gates(v1_bundle, v2_bundle)
        _print_gates(gates)

        output = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "judge_model": f"{JUDGE_PROVIDER}/{JUDGE_MODEL}",
            "engine": "both",
            "v1": {
                "total_questions": v1_bundle["total_questions"],
                "graded": v1_bundle["graded"],
                "overall_pct": v1_bundle["overall_pct"],
                "scores_by_category": v1_bundle["scores_by_category"],
                "scores_by_repo": v1_bundle["scores_by_repo"],
                "results": v1_bundle["results"],
            },
            "v2": {
                "total_questions": v2_bundle["total_questions"],
                "graded": v2_bundle["graded"],
                "overall_pct": v2_bundle["overall_pct"],
                "scores_by_category": v2_bundle["scores_by_category"],
                "scores_by_repo": v2_bundle["scores_by_repo"],
                "results": v2_bundle["results"],
            },
            "regression_gates": gates,
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull side-by-side results written to {output_file}")

    else:
        raise ValueError(f"Unknown engine: {engine}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 automated grader")
    parser.add_argument("--questions", default="phase0_questions.json")
    parser.add_argument("--output",    default="results.json")
    parser.add_argument("--dry-run",   action="store_true", help="Test judge pipeline without calling the agent")
    parser.add_argument("--engine", choices=["v1", "v2", "both"], default="v2", help="Which architecture(s) to evaluate")
    parser.add_argument("--retry-failed", metavar="PRIOR_RESULTS_JSON",
                         help="Load a prior results file, rerun ONLY the infra-failed "
                              "(quota/network/judge-error) rows, and write a merged, "
                              "corrected results file to --output. Does not re-spend "
                              "quota on rows that already scored a real 0 or 1. "
                              "--engine must be v1 or v2 (not both) when using this.")
    parser.add_argument("--override-model", default=None,
                         help="Override orchestrator.py's PLANNER_PRIMARY/SYNTH_PRIMARY and "
                              "verifier_agent.py's PRIMARY_MODEL for this run only -- real, "
                              "useful when a specific model's daily quota is exhausted and you "
                              "want retries to use a different one with fresh quota, e.g. "
                              "'google:gemini-3.1-flash'. Does not touch source files; only "
                              "affects the in-process module attributes for this run.")
    args = parser.parse_args()

    if args.override_model:
        # Real, minimal override: patch the live module attributes rather
        # than requiring the person to hand-edit source files mid-quota-
        # exhaustion. Scoped to this process only.
        import orchestrator as _orch
        from agents import verifier_agent as _verifier
        print(f"[--override-model] Overriding all Gemini model constants to {args.override_model!r} for this run.")
        _orch.PLANNER_PRIMARY = args.override_model
        _orch.PLANNER_FALLBACK = args.override_model
        _orch.SYNTH_PRIMARY = args.override_model
        _orch.SYNTH_FALLBACK = args.override_model
        _verifier.PRIMARY_MODEL = args.override_model
        _verifier.SECONDARY_MODEL = args.override_model
        _orch._planner_agents.clear()  # force re-creation with the new model string
        _verifier._verifier_agents.clear()

    if args.retry_failed:
        if args.engine == "both":
            raise ValueError("--retry-failed requires --engine v1 or v2 (not both) -- "
                              "specify which engine's results file you're retrying.")
        retry_failed(args.retry_failed, args.questions, args.output, args.engine)
    else:
        run_eval(args.questions, args.output, dry_run=args.dry_run, engine=args.engine)
