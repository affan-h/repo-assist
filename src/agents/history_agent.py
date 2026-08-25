"""
History specialist agent (v2, §4.3).

Real fix vs. the previous draft: blindly `inspect.getmembers`-scanning
query_tools and calling anything with "commit"/"pr"/"issue"/etc. in its name
ignores router.py's real, six-round-debugged evidence-gathering pipeline
(_gather_why_evidence / plan_why) -- global PR/issue caps, relevance ranking,
the confidence-gated symbol-vs-file-path fallback, and the discussion linker's
tie-handling are all real fixes already earned in v1. Reinventing a generic
scan here would silently throw all of that away and risk repeating the exact
token-budget blowout (§1) router.py's own docstrings describe fixing.

This agent wraps router.plan_why(...) directly.
"""

import router


def execute(repo: str, question: str, focus_notes: str = "") -> dict:
    print(f"  [Agent: History] Mining commits/PRs/issues/discussions/releases...")
    result = router.plan_why(repo, question)
    tool_results = result.get("tool_results", {})

    return {
        "focus_notes": focus_notes,
        "resolution_confidence": result.get("resolution_confidence"),
        "commits": tool_results.get("get_commit_history", []),
        "prs": tool_results.get("get_pr", []),
        "issues": tool_results.get("get_issue", []),
        "discussions_full": tool_results.get("discussion_candidates_full", []),
        "discussions_brief": tool_results.get("discussion_candidates_brief", []),
        "doc_hits": tool_results.get("search_docs", []),
        "release_hits": tool_results.get("search_releases", []),
    }
