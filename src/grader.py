"""
Phase 0 Automated Grader — LLM-as-Judge
========================================
Runs each question through your agent, then passes agent answer + ground truth
to a separate judge LLM. Uses different providers for generation vs judging
to reduce correlated-failure bias (self-grading problem).

Usage:
    python grader.py --questions phase0_questions.json --output results.json

Requires:
    pip install openai google-generativeai requests langfuse

Environment variables:
    GROQ_API_KEY        — for the agent (generation)
    GEMINI_API_KEY      — for the judge
    LANGFUSE_SECRET_KEY — for trace validation
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_HOST       — your self-hosted Langfuse URL (e.g. http://localhost:3000)
"""

import json
import os
import time
import argparse
from datetime import datetime
from typing import Optional
import requests

# ── Judge configuration ──────────────────────────────────────────────────────
# IMPORTANT: Judge model must be a DIFFERENT provider than your generation model.
# If your agent uses Groq/Llama → judge with Gemini Flash (and vice versa).
# This prevents the same model from grading its own style of output.

JUDGE_PROVIDER = "gemini"          # "gemini" | "groq"
JUDGE_MODEL    = "gemini-3.1-flash-lite"  # confirmed via direct curl test to
                                            # actually respond (200, real text)
                                            # on this account -- gemini-2.5-flash
                                            # hit a real 20/day RPD cap (confirmed
                                            # via exact quota error), gemini-3.5-flash
                                            # returned 503 UNAVAILABLE (overloaded).
                                            # Multiple independent sources report
                                            # 3.1-flash-lite around 15-30 RPM /
                                            # ~1,000-1,500 RPD, but treat that as a
                                            # starting estimate, not gospel -- verify
                                            # against real usage during the dry run.

JUDGE_SYSTEM_PROMPT = """You are an exact, unforgiving grader for a code intelligence system evaluation.
Your job is to compare a System_Answer to a Ground_Truth and output a score.

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
  and must score 0 -- the same as a wrong answer. Do not give credit for "honest
  abstention" on a WHY question just because it sounds careful; check whether the
  Ground_Truth actually names a findable source first. Only treat an abstention on
  a WHY question as reasonable (partial credit at most, 0.5) in the rare case the
  Ground_Truth itself indicates real ambiguity or sparse documentation.

UNANSWERABLE_WHY:
- The System_Answer must explicitly state that no documented rationale was found.
- If it provides a confident explanation (even a plausible one) without citing a source = 0.
- If it declines with a stated reason (e.g., "I found no PR or issue documenting this") = 1.
- Partial credit 0.5: if it declines but doesn't clearly state WHY it's declining.

TOPOLOGY:
- The System_Answer must return an exact or near-exact list of nodes/files/functions.
- A partial list scores 0. Vague descriptions score 0.
- Must show evidence of graph traversal (Trace_Had_Tool_Call must be true).

Output ONLY a JSON object, no other text:
{"score": 1, 0, or 0.5, "reason": "brief one-sentence explanation"}"""


# ── Agent stub — replace with your actual agent call ─────────────────────────

def call_your_agent(question: str, repo: str, category: str) -> tuple[str, bool]:
    """
    REAL IMPLEMENTATION (replacing the original skeleton).

    NOTE ON ARCHITECTURE MISMATCH WITH THE ORIGINAL DOCSTRING BELOW: the
    original skeleton assumed an LLM-driven PydanticAI agent that calls
    tools itself, checking ToolCallPart in result.all_messages() for
    trace_had_tool_call. Our actual system is different: ROUTING is
    hand-rolled Python (router.py), not an LLM deciding which tools to call
    -- only the final SYNTHESIS step uses a PydanticAI Agent, and that
    Agent has no tools of its own (it only receives pre-gathered data), so
    result.all_messages() on it would show zero tool calls always,
    regardless of whether real retrieval happened.

    trace_had_tool_call therefore comes from router.py's OWN trace list
    instead -- arguably stronger evidence of real retrieval than
    ToolCallPart would be, since it shows exactly which query_tools.py
    functions ran and how many real rows came back, which is the eval's
    actual stated intent (catching pre-trained recall vs real index use),
    just correctly implemented for this system's real architecture.

    `category` was added to this function's signature (grader.py's
    run_eval() call site was updated to pass it) because the router needs
    it to select the right tool plan -- the original skeleton's signature
    (question, repo) alone isn't enough information to route correctly.

    ORIGINAL DOCSTRING'S PYDANTICAI NOTES (kept for reference, describe a
    different architecture than what's actually implemented):
      trace_had_tool_call: checked from PydanticAI's LOCAL message history,
      not from Langfuse. Langfuse flushes asynchronously — querying it
      immediately after an agent call returns an empty trace. PydanticAI's
      result.all_messages() is synchronous and available instantly.
      In PydanticAI's message model:
        - ModelRequest  = what we sent to the LLM (user prompt, tool results)
        - ModelResponse = what the LLM sent back (text, tool calls)
      Tool calls appear as ToolCallPart inside a ModelResponse, NOT in a
      ModelRequest. Check ModelResponse parts, not ModelRequest.
    """
    import router
    import synthesizer

    PRIMARY_MODEL = "cerebras:gpt-oss-120b"
    FALLBACK_MODEL = "groq:llama-3.3-70b-versatile"

    plan_result = router.plan_and_execute(repo, category, question)
    trace_had_tool_call = len(plan_result["trace"]) > 0

    answer, model_used = synthesizer.synthesize_with_fallback(
        PRIMARY_MODEL, question, repo, plan_result["tool_results"],
        fallback_model=FALLBACK_MODEL,
    )

    # Fold citation/abstain_reason into one text string for the judge, since
    # the judge prompt expects a single "System_Answer" string and does not
    # read our structured SynthesizedAnswer object's separate fields.
    #
    # CONFIRMED REAL BUG, FIXED: previously printed the raw internal
    # source_id (e.g. "DOC#0", "PR#2306") directly into the answer text --
    # a real, technically-grounded citation, but illegible to a human/judge
    # reading only the answer text without also seeing our internal source
    # list. Fixed by resolving the source_id back to a real, human-readable
    # description using the SAME source-list data the synthesizer itself
    # built (source of truth, not a second guess at formatting).
    answer_text = answer.answer
    if answer.citation_source_id:
        # CONFIRMED REAL PROBLEM (full eval run): resolving to the raw
        # source-list LINE (e.g. "[CODE#extend] function in source/
        # create.ts...") still led with bracketed internal-ID notation,
        # which the judge repeatedly did not recognize as a real citation
        # even when the underlying source was genuinely correct (H2: "failed
        # to provide required citations... e.g. httpx/_client.py" -- exactly
        # what WAS in the evidence, just not surfaced in judge-recognizable
        # form). Fix: extract and present ONLY the human-meaningful part
        # (file path, or "PR #N: title", etc.), matching how a grader
        # actually expects a citation to read, with the internal ID
        # completely dropped from the visible text.
        sid = answer.citation_source_id
        readable = sid  # fallback if we can't parse a nicer form
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


# ── Judge call ───────────────────────────────────────────────────────────────

def call_judge(
    question: str,
    category: str,
    ground_truth: str,
    system_answer: str,
    trace_had_tool_call: bool,
) -> dict:
    """Send to judge LLM, return parsed score dict."""

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


def _judge_gemini(user_message: str, max_retries: int = 3) -> dict:
    """Call Gemini Flash as judge via the free REST API.

    RETRY-WITH-BACKOFF ADDED: confirmed real, current Gemini free-tier 429s
    happen even on gemini-2.5-flash (Google's own developer forum shows
    multiple real reports of spurious 429s well under documented quotas,
    not just a hard RPM ceiling being cleanly hit). Same defense-in-depth
    pattern already proven working for Cerebras in synthesizer.py's
    synthesize_with_fallback -- retry a few times with increasing delay
    before giving up, since a 429 here is often transient."""
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
            "response_mime_type": "application/json",  # forces native JSON output
        },
    }

    last_error = None
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 429:
            last_error = f"429 on attempt {attempt + 1}/{max_retries}"
            print(f"    [Gemini judge 429, retrying in {2 * (attempt + 1)}s...]")
            time.sleep(2 * (attempt + 1))  # increasing backoff: 2s, 4s, 6s
            continue
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return json.loads(raw)  # no manual markdown-fence splitting needed

    raise RuntimeError(f"Gemini judge exhausted {max_retries} retries, all 429. Last: {last_error}")


def _judge_groq(user_message: str) -> dict:
    """Call Groq as judge (used when your agent runs on Gemini)."""
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
        "response_format": {"type": "json_object"},  # Groq strict JSON mode
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return json.loads(raw)  # no manual markdown-fence splitting needed


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_eval(questions_file: str, output_file: str, dry_run: bool = False):
    with open(questions_file) as f:
        data = json.load(f)

    questions = data["questions"]
    results = []
    
    # Category accumulators
    cats = ["what", "how", "where", "why", "unanswerable_why", "topology"]
    scores_by_cat = {c: [] for c in cats}
    scores_by_repo = {"httpx": [], "got": []}

    print(f"Running eval on {len(questions)} questions...\n")

    for i, q in enumerate(questions):
        qid      = q["id"]
        repo     = q["repo"]
        category = q["category"]
        question = q["question"]
        gt       = q["ground_truth"]

        print(f"[{i+1}/{len(questions)}] {qid} ({category})")

        if dry_run:
            # Dry run: skip agent, just test the judge pipeline
            agent_answer = "DRY RUN — no agent answer"
            had_tool_call = False
        else:
            try:
                agent_answer, had_tool_call = call_your_agent(question, repo, category)
            except NotImplementedError as e:
                print(f"  ⚠ Agent not wired up yet: {e}")
                break
            except Exception as e:
                print(f"  ✗ Agent error: {e}")
                results.append({
                    "id": qid, "repo": repo, "category": category,
                    "score": None, "error": str(e)
                })
                continue

        # Rate-limit courtesy pause between judge calls
        time.sleep(1.5)

        try:
            grade = call_judge(question, category, gt, agent_answer, had_tool_call)
        except Exception as e:
            print(f"  ✗ Judge error: {e}")
            grade = {"score": None, "reason": f"Judge error: {e}"}

        effective_score = grade.get("score")
        reason = grade.get("reason", "")

        print(f"  → score={effective_score} | {reason}")

        result = {
            "id": qid,
            "repo": repo,
            "category": category,
            "question": question,
            "agent_answer": agent_answer,
            "trace_had_tool_call": had_tool_call,
            "ground_truth": gt,
            "score": effective_score,
            "judge_reason": reason,
        }
        results.append(result)

        if effective_score is not None:
            scores_by_cat[category].append(effective_score)
            scores_by_repo[repo].append(effective_score)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS BY CATEGORY")
    print("="*60)
    for cat in cats:
        s = scores_by_cat[cat]
        if s:
            print(f"  {cat:<20} {sum(s)}/{len(s)}  ({100*sum(s)/len(s):.0f}%)")
        else:
            print(f"  {cat:<20} —")

    print("\nRESULTS BY REPO")
    for repo, s in scores_by_repo.items():
        if s:
            print(f"  {repo:<20} {sum(s)}/{len(s)}  ({100*sum(s)/len(s):.0f}%)")

    all_scores = [r["score"] for r in results if r.get("score") is not None]
    if all_scores:
        print(f"\n  OVERALL               {sum(all_scores)}/{len(all_scores)}  ({100*sum(all_scores)/len(all_scores):.0f}%)")

    # Unanswerable-why specific diagnostic
    u_scores = scores_by_cat["unanswerable_why"]
    if u_scores:
        u_pct = 100 * sum(u_scores) / len(u_scores)
        verdict = "✓ Verifier is working" if u_pct >= 50 else "✗ Verifier too credulous — tighten citation enforcement"
        print(f"\n  unanswerable_why abstention rate: {u_pct:.0f}% — {verdict}")

    # Topology leakage diagnostic
    topo = [r for r in results if r["category"] == "topology"]
    topo_no_tool = [r for r in topo if not r.get("trace_had_tool_call") and r.get("score", 0) == 1]
    if topo_no_tool:
        print(f"\n  ⚠ WARNING: {len(topo_no_tool)} topology question(s) answered correctly with NO tool call.")
        print("    These are likely pre-trained leakage — graph edges may not be tested.")

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "judge_model": f"{JUDGE_PROVIDER}/{JUDGE_MODEL}",
        "total_questions": len(questions),
        "graded": len(all_scores),
        "scores_by_category": {
            c: {
                "correct": sum(scores_by_cat[c]),
                "total": len(scores_by_cat[c]),
                "pct": round(100 * sum(scores_by_cat[c]) / len(scores_by_cat[c]), 1) if scores_by_cat[c] else None
            }
            for c in cats
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results written to {output_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 automated grader")
    parser.add_argument("--questions", default="phase0_questions.json")
    parser.add_argument("--output",    default="results.json")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Test judge pipeline without calling the agent")
    args = parser.parse_args()

    run_eval(args.questions, args.output, dry_run=args.dry_run)
