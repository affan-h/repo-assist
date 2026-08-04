"""
repo-assist CLI -- ask real questions about httpx or got from the terminal.

Usage:
    repo-assist ask got "Why does got default to 2 retries?"
    repo-assist ask httpx "Where does httpx decode response content?" --category where

If --category is omitted, a lightweight heuristic guesses it from the
question's phrasing (why/where/what/how/topology keywords) -- pass
--category explicitly if the guess is wrong, since the router's plan
selection depends on getting this right.
"""

import argparse
import sys

import router
import synthesizer

PRIMARY_MODEL = "cerebras:gpt-oss-120b"
FALLBACK_MODEL = "groq:llama-3.3-70b-versatile"

CATEGORY_KEYWORDS = [
    ("why", ["why"]),
    ("where", ["where"]),
    ("topology", ["call chain", "trace the", "which files import", "import from"]),
    ("how", ["how does", "how do", "how is"]),
    ("what", ["what does", "what is", "what are"]),
]


def guess_category(question: str) -> str:
    q_lower = question.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in q_lower for kw in keywords):
            return category
    return "what"  # reasonable default -- most questions are factual lookups


def ask(repo: str, question: str, category: str | None = None, verbose: bool = False) -> int:
    if repo not in ("httpx", "got"):
        print(f"Error: repo must be 'httpx' or 'got' (this tool is currently scoped to these two "
              f"pinned repos -- see project_context.md for why). Got: {repo!r}", file=sys.stderr)
        return 1

    resolved_category = category or guess_category(question)
    if verbose:
        print(f"[category: {resolved_category}{'  (guessed)' if not category else ''}]", file=sys.stderr)

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


def main():
    parser = argparse.ArgumentParser(
        prog="repo-assist",
        description="Ask real questions about the httpx or got codebases, grounded in a real "
                     "structural graph, commit/PR/discussion history, docs, and release notes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question about a repo")
    ask_parser.add_argument("repo", choices=["httpx", "got"], help="Which repo to ask about")
    ask_parser.add_argument("question", help="Your question, in plain English "
                             "(quote code identifiers in backticks, e.g. `Client.get`)")
    ask_parser.add_argument("--category", choices=["what", "how", "where", "why", "unanswerable_why", "topology"],
                             default=None, help="Override the auto-guessed question category")
    ask_parser.add_argument("--verbose", "-v", action="store_true", help="Show routing/confidence diagnostics")

    args = parser.parse_args()

    if args.command == "ask":
        sys.exit(ask(args.repo, args.question, args.category, args.verbose))


if __name__ == "__main__":
    main()
