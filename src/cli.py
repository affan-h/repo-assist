"""
repo-assist CLI -- ask real questions about httpx or got from the terminal.

Usage:
    repo-assist ask got "Why does got default to 2 retries?"
    repo-assist ask httpx "Where does httpx decode response content?" --category where
    repo-assist ask httpx "What is ASGITransport?" --engine v2

If --category is omitted (v1 only), a lightweight heuristic guesses it from
the question's phrasing (why/where/what/how/topology keywords) -- pass
--category explicitly if the guess is wrong, since the router's plan
selection depends on getting this right. --category has no effect under
--engine v2, since v2's planner decides which specialists to invoke itself
per question rather than needing a category hint (see §4.4).

Real fix vs. the previous draft: PRIMARY_MODEL/FALLBACK_MODEL were hardcoded
to Cerebras/Groq, which now require payment (confirmed 402 payment_required,
Aug 2026) -- every invocation would have failed immediately. Switched to the
same Gemini models confirmed working throughout this project's real
evaluation runs (grader.py, orchestrator.py). See §12 of repo-assist-v2-plan.md
for why v2 is available but not yet the default: the real, full 56-question
regression-gate run (§6.2) did not pass -- v1 remains the safer, proven
default (41.1% vs v2's 35.7%, both gates failed) until a documented fix for
v2's why/unanswerable_why regression lands.
"""

import argparse
import re
import sys

import router
import synthesizer

PRIMARY_MODEL = "google:gemini-3.5-flash-lite"
FALLBACK_MODEL = "google:gemini-3.5-flash"

CATEGORY_KEYWORDS = [
    ("why", ["why"]),
    ("topology", [
        "call chain", "trace the", "which files import", "files import",
        "directly import", "import from", "import the",
    ]),
    ("where", ["where"]),
    ("how", ["how does", "how do", "how is"]),
    ("what", ["what does", "what is", "what are"]),
]


def guess_category(question: str) -> str:
    q_lower = question.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if category == "where":
            # Strip parenthetical clauses (e.g. "(where RequestError is defined)")
            # so secondary location references don't hijack questions of other categories.
            q_no_parens = re.sub(r"\(.*?\)", "", q_lower)
            if re.search(r"\bwhere\b", q_no_parens):
                return "where"
        elif any(kw in q_lower for kw in keywords):
            return category
    return "what"  # reasonable default -- most questions are factual lookups



def ask_v1(repo: str, question: str, category: str | None = None, verbose: bool = False) -> int:
    resolved_category = category or guess_category(question)
    if verbose:
        print(f"[engine: v1] [category: {resolved_category}{'  (guessed)' if not category else ''}]", file=sys.stderr)

    try:
        plan_result = router.plan_and_execute(repo, resolved_category, question)
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"[confidence: {plan_result['resolution_confidence']}, "
              f"{len(plan_result['trace'])} tool calls]", file=sys.stderr)

    answer, model_used = synthesizer.synthesize_with_fallback(
        PRIMARY_MODEL, question, repo, plan_result["tool_results"],
        fallback_model=FALLBACK_MODEL,
    )

    if verbose and model_used != PRIMARY_MODEL:
        print(f"[answered by fallback model: {model_used}]", file=sys.stderr)

    print()
    print(answer.answer)
    if answer.citation_source_id:
        print(f"\nSource: {answer.citation_source_id}")
    if answer.abstained and answer.abstain_reason:
        print(f"\n(Abstained: {answer.abstain_reason})")

    return 0


def ask_v2(repo: str, question: str, verbose: bool = False) -> int:
    # Local import -- keeps v1-only usage free of v2's real, heavier
    # dependency surface (onnxruntime, transformers, rustworkx) until
    # someone actually asks for --engine v2.
    import orchestrator

    if verbose:
        print(f"[engine: v2]", file=sys.stderr)

    result = orchestrator.run_query(repo, question)

    print()
    print(result.answer)
    if result.abstained and result.abstain_reason:
        print(f"\n(Abstained: {result.abstain_reason})")

    return 0


import sqlite3
from config import DB_PATH


def get_ingested_repos(db_path: str = DB_PATH) -> list[str]:
    """Return list of distinct ingested repositories found in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT repo FROM files ORDER BY repo")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def ask(repo: str, question: str, category: str | None = None, verbose: bool = False, engine: str = "v1") -> int:
    ingested = get_ingested_repos()
    if repo not in ingested:
        repos_str = ", ".join(ingested) if ingested else "none"
        print(f"Error: repo {repo!r} not found -- ingested repos are: {repos_str}", file=sys.stderr)
        return 1

    if engine == "v2":
        if category is not None and verbose:
            print("[note: --category has no effect under --engine v2; ignored]", file=sys.stderr)
        return ask_v2(repo, question, verbose)
    return ask_v1(repo, question, category, verbose)


def main():
    parser = argparse.ArgumentParser(
        prog="repo-assist",
        description="Ask real questions about ingested codebases, grounded in a real "
                     "structural graph, commit/PR/discussion history, docs, and release notes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question about an ingested repo")
    ask_parser.add_argument("repo", help="Which repo to ask about (must be ingested)")
    ask_parser.add_argument("question", help="Your question, in plain English "
                             "(quote code identifiers in backticks, e.g. `Client.get`)")
    ask_parser.add_argument("--category", choices=["what", "how", "where", "why", "unanswerable_why", "topology"],
                             default=None, help="Override the auto-guessed question category (v1 only, ignored under --engine v2)")
    ask_parser.add_argument("--engine", choices=["v1", "v2"], default="v1",
                             help="v1 (default): rule-based router, real 41.1%% eval score, proven and stable. "
                                  "v2: multi-agent orchestrator with semantic retrieval, real 35.7%% eval score "
                                  "as of the last full regression-gate run -- available for comparison/experimentation, "
                                  "not yet the default. See README.md's 'v1 vs v2' section.")
    ask_parser.add_argument("--verbose", "-v", action="store_true", help="Show routing/confidence diagnostics")

    args = parser.parse_args()

    if args.command == "ask":
        sys.exit(ask(args.repo, args.question, args.category, args.verbose, args.engine))


if __name__ == "__main__":
    main()
