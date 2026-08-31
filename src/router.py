"""
Router/executor for the query/verifier layer. Rule-based (no LLM) tool-plan
dispatch by Phase-0 question category, built on top of query_tools.py.

Starts with where/what/topology (simplest plans) to prove out the trace-
logging mechanism before adding why/how's more complex multi-source plans.

RUN FROM src/: python3 router.py   (runs against real code_graph.db)
"""

import re

from config import REPOS_DIR
import query_tools as tools


# ---------------------------------------------------------------------------
# Shared symbol resolution -- used by every category, built once here.
# ---------------------------------------------------------------------------

BACKTICK_RE = re.compile(r"`([^`]+)`")
# Detects a real file path inside backticks, e.g. `source/core/index.ts` or
# `httpx/_client.py` -- distinguished from a symbol name by containing a
# path separator and a file extension. Used only by why/unanswerable_why's
# file-level fallback (see _gather_why_evidence) -- where/what/how/topology
# don't need this, they're always asking about a specific symbol.
FILE_PATH_RE = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|md)\b")
# crude heuristic for "looks like an identifier" in non-backticked text --
# snake_case, camelCase, or PascalCase tokens of reasonable length. Used only
# as a last-resort fallback (see resolve_symbol_reference step 4).
IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")

# Common English words that would otherwise look like identifiers but never
# refer to real symbols -- prevents fallback step 4 from wasting calls/noise
# on words like "does", "what", "logic". Not exhaustive; extend if real
# questions surface more false positives.
STOPWORDS = {
    "the", "and", "for", "with", "does", "what", "how", "why", "when", "this",
    "that", "logic", "method", "function", "class", "return", "returns",
    "does", "which", "these", "those", "from", "into", "used", "using",
}


def _clean_symbol_term(raw_term: str) -> list[str]:
    """PERMANENT, SHARED FIX for a recurring real bug (confirmed twice:
    topology's `_resolve_chain_start`, and now the general resolver on
    "`got.extend()`"): a backticked term often carries call-syntax noise
    ("(url, options)") and/or a leading module/package prefix ("got.",
    "httpx.") that our symbols table's qualified_name convention never
    includes (confirmed via every real search_symbols result seen: e.g.
    "Client._transport_for_url", never "httpx.Client._transport_for_url").
    Both defeat exact AND fuzzy matching if not stripped first.

    Returns a list of candidate strings to try, most-specific first:
    the cleaned full term, then (if dotted) the term with its first
    segment dropped, then just the last segment alone (e.g. "extend" from
    "got.extend"). Callers should try each in order and stop at the first
    real match -- this function only prepares candidates, it doesn't
    resolve anything itself."""
    # strip a trailing call expression: "Client.get(url)" -> "Client.get"
    cleaned = re.sub(r"\([^)]*\)\s*$", "", raw_term).strip()
    # strip any leftover quote artifacts sometimes captured by the outer regex
    cleaned = cleaned.strip("'\"")

    candidates = [cleaned]
    parts = cleaned.split(".")
    if len(parts) > 1:
        candidates.append(".".join(parts[1:]))  # drop first segment (likely module/repo name)
        candidates.append(parts[-1])            # last segment alone (e.g. "extend")

    # de-duplicate while preserving order
    seen = set()
    result = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def search_source_code(repo: str, query: str, limit: int = 3) -> list[dict]:
    """FAST, ROBUST FIX for plain-English behavior questions that don't
    name a specific symbol/identifier: greps real source file CONTENT
    directly for question keywords, bypassing symbol-NAME matching
    entirely (which was confirmed unreliable for phrases like "closed
    client cannot send requests" -- no single word or word-pair matches
    the real symbol name ClientState).

    Real, simple approach: score each source file by how many distinct
    query keywords appear in its actual text, return the file(s) with the
    most hits plus the specific matching lines with context."""
    import os
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    if not query_terms:
        return []

    root = REPOS_DIR / repo
    if not root.is_dir():
        return []

    skip_dirs = {"node_modules", ".git", ".venv", "venv", "__pycache__", "tests", "test", "dist", "build", ".pytest_cache", ".mypy_cache"}
    scored_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            if not (fname.endswith(".py") or fname.endswith(".ts") or fname.endswith(".tsx") or fname.endswith(".js") or fname.endswith(".jsx")):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            content_lower = content.lower()
            score = sum(1 for t in query_terms if t in content_lower)
            if score >= 2:  # require at least 2 distinct real keyword hits, avoid pure noise
                scored_files.append((score, fpath, content))

    scored_files.sort(key=lambda x: -x[0])
    results = []
    for score, fpath, content in scored_files[:limit]:
        # find the single line window with the most keyword density, real
        # excerpt not just a prefix
        lines = content.split("\n")
        best_start, best_window_score = 0, -1
        window = 30
        for i in range(0, max(1, len(lines) - window), 10):
            chunk = "\n".join(lines[i:i + window]).lower()
            s = sum(chunk.count(t) for t in query_terms)
            if s > best_window_score:
                best_window_score = s
                best_start = i
        excerpt = "\n".join(lines[best_start:best_start + window])
        rel_path = os.path.relpath(fpath, root)
        results.append({"file_path": rel_path, "score": score, "excerpt": excerpt[:1500]})
    return results


def resolve_symbol_reference(repo: str, question_text: str) -> dict:
    """Tries, in order, to find real symbol(s) the question is about.
    Returns {"matches": [...], "trace": [...]} -- matches is a list of real
    symbol rows from query_tools (possibly empty if nothing resolves),
    trace records every attempt made (for grading/debugging), not just the
    winning one.

    CONFIRMED REAL PATTERN (phase0_eval_questions.md): most what/how/where/
    topology questions backtick-quote the symbol/class/param name directly
    (e.g. `_transport_for_url`). Some why questions name a BEHAVIOR instead
    (e.g. "the logic that merges per-request `auth`...") where the backticked
    term is a parameter, not the target symbol -- step 4's fallback exists
    specifically for that case, though it's lower-confidence and should be
    treated as such downstream.

    CONFIRMED REAL FIX (ground-truth audit, W6 "`got.extend()`" question):
    backticked terms with call-syntax noise ("()") or a leading module
    prefix ("got.") previously failed BOTH exact and fuzzy matching,
    falling through to the unreliable word-scan fallback. Every attempt
    below now also tries _clean_symbol_term()'s cleaned candidates, not
    just the raw backticked text.
    """
    trace = []
    backticked = BACKTICK_RE.findall(question_text)

    # Step 1: exact qualified_name match on each backticked term (raw AND cleaned)
    for term in backticked:
        for candidate in [term] + _clean_symbol_term(term):
            rows = tools.get_symbol(repo, candidate)
            trace.append({"tool": "get_symbol", "args": {"repo": repo, "qualified_name": candidate}, "result_count": len(rows)})
            if rows:
                return {"matches": rows, "trace": trace, "confidence": "high", "method": "exact_backtick"}

    # Step 2: fuzzy search_symbols on each backticked term (raw AND cleaned;
    # catches e.g. `Limits` when the real qualified_name is "Limits.__init__")
    for term in backticked:
        for candidate in [term] + _clean_symbol_term(term):
            rows = tools.search_symbols(repo, candidate)
            trace.append({"tool": "search_symbols", "args": {"repo": repo, "name_substring": candidate}, "result_count": len(rows)})
            if rows:
                return {"matches": rows, "trace": trace, "confidence": "medium", "method": "fuzzy_backtick"}

    # Step 3: try backticked terms that might be bare filenames (e.g.
    # `httpx/_client.py`) against get_imports/search -- not a symbol at all.
    # Skipped for now: none of the real questions we've seen need this yet;
    # revisit if a real miss shows this is needed (same evidence-driven
    # discipline as the rest of the project, not preemptive build-out).

    # Step 4: last-resort fallback -- scan non-backticked words for anything
    # identifier-shaped, lower confidence, logged clearly as such.
    candidates = [w for w in IDENTIFIER_RE.findall(question_text) if w.lower() not in STOPWORDS]
    for term in candidates:
        rows = tools.search_symbols(repo, term)
        trace.append({"tool": "search_symbols", "args": {"repo": repo, "name_substring": term}, "result_count": len(rows)})
        if rows:
            return {"matches": rows, "trace": trace, "confidence": "low", "method": "fallback_word_scan"}

    # Step 5: FAST FIX for a confirmed systemic gap -- plain-English
    # questions describing behavior (e.g. "closed client cannot send
    # further requests") never match a real symbol via single generic
    # words ("client" matches too much noise or nothing precise), but the
    # REAL symbol name is often a natural concatenation of two adjacent
    # candidate words (e.g. "closed"+"state" -> "ClientState", "send"+
    # "single"+"request" -> "_send_single_request"). Try adjacent-pair
    # concatenations against search_symbols before giving up entirely.
    for i in range(len(candidates) - 1):
        pair = candidates[i] + candidates[i + 1]
        rows = tools.search_symbols(repo, pair)
        trace.append({"tool": "search_symbols", "args": {"repo": repo, "name_substring": pair}, "result_count": len(rows)})
        if rows:
            return {"matches": rows, "trace": trace, "confidence": "low", "method": "fallback_pair_scan"}

    return {"matches": [], "trace": trace, "confidence": "none", "method": None}


# ---------------------------------------------------------------------------
# Category plans
# ---------------------------------------------------------------------------

def plan_where(repo: str, question: str) -> dict:
    """'where' questions: locate the real definition, and pull a real source
    snippet as supporting evidence. Real ground truth for these is typically
    'verifiable in <file>' -- so returning the resolved symbol's
    file_path/start_line/end_line IS most of the answer's core content, but
    a real snippet helps the synthesizer describe the actual mechanism at
    that location (e.g. WH4's ClientState enum check) rather than just
    naming a file/line range with no supporting text."""
    resolution = resolve_symbol_reference(repo, question)
    trace = list(resolution["trace"])
    snippets = []
    for sym in resolution["matches"]:
        snippet = tools.get_source_snippet(repo, sym["file_path"], sym["start_line"], sym["end_line"])
        trace.append({
            "tool": "get_source_snippet",
            "args": {"repo": repo, "file_path": sym["file_path"], "start_line": sym["start_line"], "end_line": sym["end_line"]},
            "result_count": 1 if snippet else 0,
        })
        if snippet:
            snippets.append(snippet)

    doc_hits_raw = tools.search_docs(repo, question)
    trace.append({"tool": "search_docs", "args": {"repo": repo, "query": question}, "result_count": len(doc_hits_raw)})
    doc_hits = [
        {"file_path": h["file_path"], "heading": h["heading"],
         "content": h["content"] if i == 0 else h["content"][:600], "score": h["score"]}
        for i, h in enumerate(doc_hits_raw)
    ]

    # FAST FIX for confirmed systemic gap: when symbol resolution is
    # unreliable (low/none), fall back to full-text source-code search
    # instead of trusting a wrong fuzzy symbol match.
    source_hits = []
    if resolution["confidence"] in ("low", "none"):
        source_hits = search_source_code(repo, question)
        trace.append({"tool": "search_source_code", "args": {"repo": repo, "query": question}, "result_count": len(source_hits)})

    return {
        "tool_results": {
            "resolve_symbol_reference": resolution["matches"] if resolution["confidence"] in ("high", "medium") else [],
            "get_source_snippet": snippets if resolution["confidence"] in ("high", "medium") else [],
            "search_docs": doc_hits,
            "search_source_code": source_hits,
        },
        "trace": trace,
        "resolution_confidence": resolution["confidence"],
    }


def plan_what(repo: str, question: str) -> dict:
    """'what' questions: resolve symbol, pull its summary, real source
    snippet, AND search docs.

    CONFIRMED REAL GAP (full 56-question eval run): every got 'what' question
    (W6-W10) failed, and every one of their ground truths cites a real
    documentation/*.md file (hooks, pagination, retry, options) -- content
    search_docs() was already proven able to find (the merge-behavior test).
    This plan never called search_docs() at all. Adding it here since prose
    documentation, not code summaries, is where got's factual answers
    actually live.

    SECOND CONFIRMED GAP: W2 (Limits class's 3 parameter default values)
    failed even with correct resolution + a real summary -- Phase 2's
    summaries are prose relational descriptions, not verbatim signatures.
    get_source_snippet() now pulls the REAL code text directly, giving the
    synthesizer the actual default values/types to read, not a paraphrase."""
    resolution = resolve_symbol_reference(repo, question)
    trace = list(resolution["trace"])
    summaries = []
    snippets = []
    for sym in resolution["matches"]:
        summary = tools.get_summary(repo, sym["file_path"], sym["qualified_name"], sym["start_line"])
        trace.append({
            "tool": "get_summary",
            "args": {"repo": repo, "file_path": sym["file_path"], "qualified_name": sym["qualified_name"]},
            "result_count": 1 if summary else 0,
        })
        if summary:
            summaries.append(summary)

        snippet = tools.get_source_snippet(repo, sym["file_path"], sym["start_line"], sym["end_line"])
        trace.append({
            "tool": "get_source_snippet",
            "args": {"repo": repo, "file_path": sym["file_path"], "start_line": sym["start_line"], "end_line": sym["end_line"]},
            "result_count": 1 if snippet else 0,
        })
        if snippet:
            snippets.append(snippet)

    doc_hits_raw = tools.search_docs(repo, question)
    trace.append({"tool": "search_docs", "args": {"repo": repo, "query": question}, "result_count": len(doc_hits_raw)})
    # CONFIRMED REAL BUG, FIXED (ground-truth audit, W7 "got hook points"
    # question): a flat 600-char cap truncated the real, correct doc chunk
    # (12,374 real chars) before it even finished describing the FIRST of
    # several real hook points -- the model never had a chance to answer
    # completely regardless of prompt/model quality. what/how questions
    # often need to read ONE chunk in full, unlike why (comparing across
    # many candidate sources, where a tighter per-source cap makes sense).
    # Fix: full content for the top-ranked match, shorter for the rest
    # (supporting context, not usually where the answer lives).
    doc_hits = [
        {"file_path": h["file_path"], "heading": h["heading"],
         "content": h["content"] if i == 0 else h["content"][:600], "score": h["score"]}
        for i, h in enumerate(doc_hits_raw)
    ]

    source_hits = []
    if resolution["confidence"] in ("low", "none"):
        source_hits = search_source_code(repo, question)
        trace.append({"tool": "search_source_code", "args": {"repo": repo, "query": question}, "result_count": len(source_hits)})

    return {
        "tool_results": {
            "resolve_symbol_reference": resolution["matches"],
            "get_summary": summaries,
            "get_source_snippet": snippets,
            "search_docs": doc_hits,
            "search_source_code": source_hits,
        },
        "trace": trace,
        "resolution_confidence": resolution["confidence"],
    }


def _trace_call_chain(repo: str, start_symbol: dict, max_hops: int = 8) -> dict:
    """Traces a single ordered path from start_symbol, following callees hop
    by hop, until either (a) no further callees exist (real end reached), or
    (b) the next callee isn't in our own symbols table (external library
    hand-off -- exactly what T1/T5 ask us to find, e.g. 'handed off to
    httpcore', 'Node's http.request invoked').

    BRANCH-PICKING FIX (was: always take branch[0] blindly, confirmed real
    problem -- T1's real chain needed _send_handling_redirects, branch[0]
    picked Auth.sync_auth_flow instead). Now, when a hop has 2+ callees,
    each branch is explored a few hops ahead (cheap -- just following
    single-callee chains, no recursion into further branches) and the
    branch that reaches an EXTERNAL hand-off soonest (or goes deepest, if
    none hit external) is preferred -- since T1/T5 specifically ask for the
    chain "to the point of hand-off," the correct branch is the one that
    actually gets there, not an arbitrary pick. Still records ALL branches
    in branch_points for transparency -- this is a smarter default choice,
    not a silent guess with no trace."""
    path = []
    branch_points = []
    current = start_symbol
    visited = set()

    def _peek_depth(start_sym, depth_budget):
        """Cheap lookahead: follows single-callee chains from start_sym,
        returns (depth_reached, hit_external). Does not recurse into
        further branches -- just picks callees[0] at each step, since this
        is only used to SCORE candidate branches relative to each other,
        not to produce the final answer."""
        cur = start_sym
        seen = set()
        for d in range(depth_budget):
            k = (cur["file_path"], cur["qualified_name"], cur["start_line"])
            if k in seen:
                return d, False
            seen.add(k)
            callees = tools.get_callees(repo, *k)
            if not callees:
                return d, False
            nxt = callees[0]
            rows = tools.get_symbol(repo, nxt["to_qualified_name"])
            matches = [r for r in rows if r["file_path"] == nxt["to_file"] and r["start_line"] == nxt["to_start_line"]]
            if not matches:
                return d + 1, True  # hit external -- exactly what T1/T5 want
            cur = matches[0]
        return depth_budget, False

    for _hop in range(max_hops):
        key = (current["file_path"], current["qualified_name"], current["start_line"])
        if key in visited:
            path.append({"qualified_name": current["qualified_name"], "file": current["file_path"], "note": "CYCLE DETECTED, stopping"})
            break
        visited.add(key)
        path.append({"qualified_name": current["qualified_name"], "file": current["file_path"]})

        callees = tools.get_callees(repo, *key)
        if not callees:
            path.append({"note": "no further callees in the graph -- real end of traceable path"})
            break

        if len(callees) > 1:
            branch_points.append({
                "at": current["qualified_name"],
                "options": [{"qualified_name": c["to_qualified_name"], "file": c["to_file"]} for c in callees],
            })
            # Score each branch by cheap lookahead, prefer one that reaches
            # an external hand-off, tie-break by depth reached.
            scored = []
            for c in callees:
                rows = tools.get_symbol(repo, c["to_qualified_name"])
                matches = [r for r in rows if r["file_path"] == c["to_file"] and r["start_line"] == c["to_start_line"]]
                if not matches:
                    scored.append((1, True, c))  # immediately external -- strong candidate
                    continue
                depth, hit_external = _peek_depth(matches[0], depth_budget=4)
                scored.append((depth, hit_external, c))
            scored.sort(key=lambda x: (-int(x[1]), -x[0]))  # external-hit first, then deepest
            best_callee = scored[0][2]
            callees = [best_callee] + [c for c in callees if c is not best_callee]

        next_callee = callees[0]

        # Check whether the next hop is a real symbol we can keep tracing,
        # or an external hand-off (the actual thing T1/T5 want identified).
        next_symbol_rows = tools.get_symbol(repo, next_callee["to_qualified_name"])
        matching_rows = [
            r for r in next_symbol_rows
            if r["file_path"] == next_callee["to_file"] and r["start_line"] == next_callee["to_start_line"]
        ]
        if not matching_rows:
            path.append({
                "qualified_name": next_callee["to_qualified_name"],
                "file": next_callee["to_file"],
                "note": "EXTERNAL (not in our own symbol graph) -- likely hand-off point to an external library",
            })
            break

        current = matching_rows[0]

    return {"path": path, "branch_points": branch_points}


def _resolve_chain_start(repo: str, question_text: str) -> dict:
    """CONFIRMED REAL BUG (T1 test): resolve_symbol_reference tries EVERY
    backticked term and returns the FIRST ONE THAT MATCHES ANYTHING -- for
    T1's "from `httpx.Client.get(url)` to ... `httpcore`", the real start
    term (`httpx.Client.get(url)`) failed to match at all (call-syntax/
    module-prefix noise), so it fell through to matching `httpcore` instead
    -- the DESTINATION reference, not the start point, producing a
    completely wrong traced chain rooted at an unrelated function.

    FIX: for chain-tracing specifically, take ONLY the first backticked
    term (confirmed real convention in both T1 and T5: "from `X` to `Y`"),
    and resolve it via the SHARED _clean_symbol_term() candidates (same
    permanent fix now used by resolve_symbol_reference generally) rather
    than trying every backtick and accepting whichever matches first."""
    backticked = BACKTICK_RE.findall(question_text)
    if not backticked:
        return {"matches": [], "trace": [], "confidence": "none"}

    candidates = _clean_symbol_term(backticked[0])
    trace = []

    for term in candidates:
        rows = tools.get_symbol(repo, term)
        trace.append({"tool": "get_symbol", "args": {"repo": repo, "qualified_name": term}, "result_count": len(rows)})
        if rows:
            return {"matches": rows, "trace": trace, "confidence": "high"}

    for term in candidates:
        rows = tools.search_symbols(repo, term.split(".")[-1])
        trace.append({"tool": "search_symbols", "args": {"repo": repo, "name_substring": term.split('.')[-1]}, "result_count": len(rows)})
        if rows:
            return {"matches": rows, "trace": trace, "confidence": "medium"}

    return {"matches": [], "trace": trace, "confidence": "none"}


def plan_topology(repo: str, question: str, max_hops: int = 4) -> dict:
    """'topology' questions want an EXACT node list from real graph traversal
    (confirmed rubric: partial list or no tool call = 0).

    CONFIRMED REAL GAP (full 56-question eval run, 0/6): T2, T3, T4, T6 all
    ask "which files import X" -- a REVERSE IMPORT lookup this plan never
    attempted (only CALLS/INSTANTIATES/EXTENDS traversal existed). Fixed by
    detecting import-style questions (keyword "import" + a real backticked
    file/class reference) and using the new get_files_importing() reverse
    lookup instead of call-graph traversal for those.

    T1/T5-style questions ("call chain", "trace... to the point where")
    still use the CALLS/INSTANTIATES/EXTENDS BFS below."""
    trace = []

    if "import" in question.lower():
        # Try a direct backticked file path first (T2, T4, T6 style:
        # `httpx/_client.py`, `source/core/errors.ts`)
        real_file = _extract_real_file_path(repo, question)
        target_files = [real_file] if real_file else []

        # T3-style: backticks a CLASS name, not a file path directly
        # ("import the `Timeout` class") -- resolve the class to its file.
        if not target_files:
            resolution = resolve_symbol_reference(repo, question)
            trace.extend(resolution["trace"])
            target_files = list({m["file_path"] for m in resolution["matches"]})

        importers = []
        for target_file in target_files:
            rows = tools.get_files_importing(repo, target_file)
            trace.append({"tool": "get_files_importing", "args": {"repo": repo, "target_file": target_file}, "result_count": len(rows)})
            importers.extend(rows)

        return {
            "tool_results": {
                "resolve_symbol_reference": [],
                "get_files_importing": importers,
                "get_callers": [],
                "get_callees": [],
                "traced_call_chain": [],
                "call_chain_branch_points": [],
            },
            "trace": trace,
            "resolution_confidence": "high" if importers else "low",
        }

    # Non-import topology question (T1/T5-style call chain) -- use the
    # dedicated chain-start resolver (fixes the confirmed real bug where
    # generic resolve_symbol_reference picked the destination reference
    # instead of the start point).
    resolution = _resolve_chain_start(repo, question)
    trace = list(resolution["trace"])

    all_callers = []
    all_callees = []
    frontier = resolution["matches"][:1]  # start from best match only, avoid combinatorial blowup
    visited = set()

    for _hop in range(max_hops):
        next_frontier = []
        for sym in frontier:
            key = (sym["file_path"], sym["qualified_name"], sym["start_line"])
            if key in visited:
                continue
            visited.add(key)

            callers = tools.get_callers(repo, *key)
            trace.append({"tool": "get_callers", "args": {"repo": repo, "symbol": key}, "result_count": len(callers)})
            all_callers.extend(callers)

            callees = tools.get_callees(repo, *key)
            trace.append({"tool": "get_callees", "args": {"repo": repo, "symbol": key}, "result_count": len(callees)})
            all_callees.extend(callees)

            # expand frontier along callees only (natural "what does this lead to" direction);
            # callers aren't expanded further to avoid unbounded fan-in blowup on popular symbols
            for edge in callees:
                next_frontier.append({
                    "file_path": edge["to_file"],
                    "qualified_name": edge["to_qualified_name"],
                    "start_line": edge["to_start_line"],
                })
        frontier = next_frontier
        if not frontier:
            break

    chain_result = {"path": [], "branch_points": []}
    if resolution["matches"]:
        chain_result = _trace_call_chain(repo, resolution["matches"][0])
        trace.append({
            "tool": "_trace_call_chain",
            "args": {"repo": repo, "start": resolution["matches"][0]["qualified_name"]},
            "result_count": len(chain_result["path"]),
        })

    return {
        "tool_results": {
            "resolve_symbol_reference": resolution["matches"],
            "get_callers": all_callers,
            "get_callees": all_callees,
            "get_files_importing": [],
            "traced_call_chain": chain_result["path"],
            "call_chain_branch_points": chain_result["branch_points"],
        },
        "trace": trace,
        "resolution_confidence": resolution["confidence"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plan_how(repo: str, question: str) -> dict:
    """'how' questions: resolve symbol, pull summary (like 'what'), pull
    immediate callees, AND search docs.

    CONFIRMED REAL GAP (full 56-question eval run): every got 'how' question
    (H6-H9) failed with the identical root cause as plan_what's got failures
    -- ground truths citing documentation/*.md content (retry delay
    calculation, afterResponse hook mechanics, pagination stop conditions,
    hook execution order) that this plan never searched for."""
    resolution = resolve_symbol_reference(repo, question)
    trace = list(resolution["trace"])
    summaries = []
    snippets = []
    callees_all = []
    for sym in resolution["matches"]:
        key = (sym["file_path"], sym["qualified_name"], sym["start_line"])

        summary = tools.get_summary(repo, *key)
        trace.append({"tool": "get_summary", "args": {"repo": repo, "symbol": key}, "result_count": 1 if summary else 0})
        if summary:
            summaries.append(summary)

        snippet = tools.get_source_snippet(repo, sym["file_path"], sym["start_line"], sym["end_line"])
        trace.append({"tool": "get_source_snippet", "args": {"repo": repo, "symbol": key}, "result_count": 1 if snippet else 0})
        if snippet:
            snippets.append(snippet)

        callees = tools.get_callees(repo, *key)
        trace.append({"tool": "get_callees", "args": {"repo": repo, "symbol": key}, "result_count": len(callees)})
        callees_all.extend(callees)

    doc_hits_raw = tools.search_docs(repo, question)
    trace.append({"tool": "search_docs", "args": {"repo": repo, "query": question}, "result_count": len(doc_hits_raw)})
    # Same fix as plan_what -- full content for top match, capped for the rest.
    doc_hits = [
        {"file_path": h["file_path"], "heading": h["heading"],
         "content": h["content"] if i == 0 else h["content"][:600], "score": h["score"]}
        for i, h in enumerate(doc_hits_raw)
    ]

    source_hits = []
    if resolution["confidence"] in ("low", "none"):
        source_hits = search_source_code(repo, question)
        trace.append({"tool": "search_source_code", "args": {"repo": repo, "query": question}, "result_count": len(source_hits)})

    return {
        "tool_results": {
            "resolve_symbol_reference": resolution["matches"],
            "get_summary": summaries,
            "get_source_snippet": snippets,
            "get_callees": callees_all,
            "search_docs": doc_hits,
            "search_source_code": source_hits,
        },
        "trace": trace,
        "resolution_confidence": resolution["confidence"],
    }


def _extract_real_file_path(repo: str, question_text: str) -> str | None:
    """Looks for a backticked, real file-extension path in the question
    (e.g. `source/core/index.ts`) and confirms it's an ACTUAL file in the
    graph via query_tools' files table -- not just regex-shaped text that
    happens to look like a path. Returns the confirmed real file_path or
    None."""
    candidates = FILE_PATH_RE.findall(question_text)
    if not candidates:
        return None
    with tools._connect() as conn:
        for candidate in candidates:
            # try as-is, and stripped of backticks/punctuation the regex might've grabbed
            for variant in (candidate, candidate.strip("`.,")):
                row = conn.execute(
                    "SELECT path FROM files WHERE repo = ? AND path = ?", (repo, variant)
                ).fetchone()
                if row:
                    return row[0]
    return None


MAX_PRS_TO_FETCH = 8
# Real, confirmed problem (compare_models.py, both why-family questions):
# fetching every linked PR's full body/comments/reviews for a high-churn
# file (up to 66 PRs, 214 commits) blew through every provider's real token
# limits -- Groq 12k TPM, Cerebras per-request quotas, zai-glm-4.7's 8192
# context window. This is a genuine retrieval problem, not a display
# truncation: we need to SELECT which evidence is worth showing, the same
# way a human investigating would skim commit messages before opening
# PRs, not open all 66.


def _score_commit_relevance(commit: dict, question_terms: list[str]) -> float:
    """Cheap, no-fetch relevance score using only the commit MESSAGE
    (already in hand from get_commit_history -- no extra tool call needed).
    Same IDF-free simple term-match principle as search_releases() -- kept
    simple until real evidence shows it needs the fuller IDF treatment
    search_docs() got. Deliberately down-weights trivial-sounding commits
    (confirmed real pattern: "chore: fix typo"-style messages carry near-zero
    rationale signal) via a short deny-list of low-signal message prefixes."""
    message = (commit.get("message") or "").lower()
    if not message:
        return 0.0

    score = sum(1 for t in question_terms if t.lower() in message)

    trivial_prefixes = ("fix typo", "chore:", "docs:", "style:", "bump ", "release ")
    if any(message.startswith(p) for p in trivial_prefixes):
        score -= 1  # down-weight, don't hard-exclude -- a real rationale could still follow a "chore:" prefix

    # having a real PR number at all is itself weak positive signal (more
    # likely to have discussion/review context worth reading)
    if commit.get("pr_number"):
        score += 0.5

    return score


def _compact_issue(issue: dict) -> dict:
    """Same compaction principle as _compact_pr -- keep title/body/top
    comments, drop the rest. Issues have no reviews_json (that's PR-only),
    so nothing to drop there."""
    compact = {
        "issue_number": issue.get("issue_number"),
        "title": issue.get("title"),
        "body": (issue.get("body") or "")[:800],
        "state": issue.get("state"),
        "closed_at": issue.get("closed_at"),
    }
    comments = issue.get("comments_json") or []
    if isinstance(comments, list):
        compact["comments"] = [
            {"author": (c.get("author") or {}).get("login"), "body": (c.get("body") or "")[:400]}
            for c in comments[:5]
        ]
    return compact


def _compact_pr(pr: dict) -> dict:
    """Strips low-signal fields before handing a PR to the synthesizer.
    CONFIRMED real, near-zero rationale signal in reviews_json (real
    example: PR #2306's reviews were APPROVED/CHANGES_REQUESTED with empty
    bodies) -- dropped entirely. comments_json kept but compacted to just
    author login + body text, since the synthesizer needs the substance,
    not nested timestamp/metadata objects."""
    compact = {
        "pr_number": pr.get("pr_number"),
        "title": pr.get("title"),
        "body": (pr.get("body") or "")[:800],  # real bodies can be long release-checklists; cap defensively
        "state": pr.get("state"),
        "merged_at": pr.get("merged_at"),
    }
    comments = pr.get("comments_json") or []
    if isinstance(comments, list):
        compact["comments"] = [
            {"author": (c.get("author") or {}).get("login"), "body": (c.get("body") or "")[:400]}
            for c in comments[:5]  # cap comment count too -- a long thread's first few comments usually carry the core rationale
        ]
    return compact


def _compact_discussion(disc: dict) -> dict:
    """Same compaction principle as _compact_pr -- keep title/body/top
    comments, drop the rest."""
    compact = {
        "discussion_number": disc.get("discussion_number"),
        "title": disc.get("title"),
        "body": (disc.get("body") or "")[:800],
        "answer_body": (disc.get("answer_body") or "")[:800] if disc.get("answer_body") else None,
        "_score": disc.get("_score"),
        "_tied_with_count": disc.get("_tied_with_count"),
    }
    comments = disc.get("comments_json") or []
    if isinstance(comments, list):
        compact["comments"] = [
            {"author": (c.get("author") or {}).get("login"), "body": (c.get("body") or "")[:400]}
            for c in comments[:5]
        ]
    return compact


def _gather_why_evidence(repo: str, question: str) -> dict:
    """Shared by plan_why and plan_unanswerable_why -- same evidence-gathering
    process; the only real difference between the two categories is what the
    SYNTHESIZER concludes from the evidence (confident rationale+citation vs.
    honest abstention), not how the router searches. Building one shared
    gatherer avoids duplicating this multi-source logic twice, and matches
    the project's existing principle that unanswerable_why should be tested
    with the SAME real pipeline, not a special-cased shortcut.

    CONFIRMED REAL GAP (found via direct testing, both got Node.js-version and
    naming-convention questions): resolve_symbol_reference's low-confidence
    fallback (step 4, generic word scan) produces COINCIDENTAL matches for
    file/project-level why questions that aren't about any specific symbol at
    all -- e.g. "got" (the repo's own name in the question) matched the real
    function `isGotInstance`, entirely unrelated to the actual question.
    Confirmed via diagnose_fallback.py: these matches are noise, not signal,
    and following their commit history pollutes the synthesizer's evidence
    with irrelevant commits (70-455 in the two real cases tested).

    FIX: symbol-level evidence (commit history via a resolved symbol's file,
    PR/discussion linking) is only gathered when resolution reaches at least
    "medium" confidence -- i.e. a real backtick-matched symbol, not a
    coincidental word-scan hit. When confidence is "low" or "none", this
    falls back to: (a) a real, confirmed file path mentioned in the question
    (e.g. `source/core/index.ts`) if one exists, pulling ITS commit history
    directly rather than via a coincidentally-matched symbol; or (b) skipping
    commit/PR/discussion evidence entirely and relying on search_docs/
    search_releases alone, which are always run regardless (cheap, and
    confirmed to work well for real file/project-level cases like got's
    Node.js version requirement).

    Real source order, cheapest/most-structural first (matches the project's
    own finding that docs answer "what" better than "why" for at least one
    real case -- code/history leads, docs/releases supplement):
      1. resolve symbol (only trusted at medium/high confidence)
      2. commit history for its file (or a confirmed real file path fallback)
      3. PR lookup for commits that have a real pr_number
      4. discussion linker for each such PR (ties handled per project decision:
         no numeric tiebreak, top 2-3 candidates get full detail, rest get
         title/date only)
      5. search_docs / search_releases as supplementary, always run (cheap,
         doesn't depend on step 2-4 succeeding) -- covers cases like got's
         release-note-only rationale (Y9/Y10/Y11) where there's no PR/commit
         trail at all worth following.
    """
    resolution = resolve_symbol_reference(repo, question)
    trace = list(resolution["trace"])

    commits_all = []
    prs = []
    discussion_candidates_full = []
    discussion_candidates_brief = []

    files_to_check = []  # list of real file_paths to pull commit history for

    if resolution["confidence"] in ("high", "medium"):
        files_to_check = [sym["file_path"] for sym in resolution["matches"]]
    else:
        # low/none confidence -- don't trust the symbol match (confirmed
        # coincidental in real testing). Try a real, confirmed file path
        # mentioned directly in the question instead.
        real_file = _extract_real_file_path(repo, question)
        trace.append({"tool": "_extract_real_file_path", "args": {"repo": repo}, "result_count": 1 if real_file else 0})
        if real_file:
            files_to_check = [real_file]
        # else: no symbol-level or file-level evidence source at all --
        # commits_all/prs/discussions stay empty, search_docs/search_releases
        # (below) carry the whole answer. This is an honest, logged outcome,
        # not a silent failure -- the trace shows exactly why.

    # RANK before fetching -- cheap (uses only commit message text already in
    # hand), fixes the real confirmed problem of dumping every linked PR
    # regardless of relevance. Collected across ALL files_to_check first, then
    # capped GLOBALLY -- CONFIRMED REAL BUG (Y2/Y5 hard failures, full 56-
    # question eval run): when resolve_symbol_reference returns multiple
    # matches (e.g. Client.X and AsyncClient.X), files_to_check had 2+ entries,
    # and the old code capped MAX_PRS_TO_FETCH PER FILE inside this loop --
    # so 2 files could fetch up to 16 PRs total, blowing through both
    # Cerebras' TPM window and Groq's fallback 12k-token limit in the same
    # request. Global collection-then-cap fixes this at the root.
    question_terms = [t for t in question.split() if len(t) > 2]
    all_commits_with_pr = []
    all_issue_refs = []  # (issue_number, commit) pairs, for ranking by the commit's relevance

    for file_path in files_to_check:
        commits = tools.get_commit_history(repo, file_path)
        trace.append({"tool": "get_commit_history", "args": {"repo": repo, "file_path": file_path}, "result_count": len(commits)})
        commits_all.extend(commits)
        all_commits_with_pr.extend(c for c in commits if c.get("pr_number"))

        # CONFIRMED REAL FIELD (Phase 2 already populates this correctly,
        # verified via direct query): related_issue_refs distinguishes a
        # commit's own PR from OTHER issue numbers mentioned in the commit
        # body (e.g. "Fixes #572"). This is exactly what's needed for
        # ground truths citing bare Issue numbers (Y1, Y3, Y7) -- commit
        # MESSAGES alone don't contain these (confirmed via direct testing:
        # every #N in 15 sampled real commit messages was just the commit's
        # own PR number), so this column is the real, only source for them.
        for c in commits:
            refs = c.get("related_issue_refs")
            if refs:
                for ref in str(refs).split(","):
                    ref = ref.strip()
                    if ref.isdigit():
                        all_issue_refs.append((int(ref), c))

    all_commits_with_pr.sort(key=lambda c: -_score_commit_relevance(c, question_terms))
    all_issue_refs.sort(key=lambda pair: -_score_commit_relevance(pair[1], question_terms))

    seen_prs = set()
    ranked_pr_numbers = []
    for c in all_commits_with_pr:
        pr_num = c["pr_number"]
        if pr_num not in seen_prs:
            seen_prs.add(pr_num)
            ranked_pr_numbers.append(pr_num)
        if len(ranked_pr_numbers) >= MAX_PRS_TO_FETCH:
            break

    MAX_ISSUES_TO_FETCH = 4  # smaller than PRs -- issue refs are a secondary signal, not the primary evidence path
    seen_issues = set()
    ranked_issue_numbers = []
    for issue_num, _c in all_issue_refs:
        if issue_num not in seen_issues:
            seen_issues.add(issue_num)
            ranked_issue_numbers.append(issue_num)
        if len(ranked_issue_numbers) >= MAX_ISSUES_TO_FETCH:
            break

    trace.append({
        "tool": "_score_commit_relevance",
        "args": {"repo": repo, "total_files_checked": len(files_to_check), "total_prs_available": len({c["pr_number"] for c in all_commits_with_pr})},
        "result_count": len(ranked_pr_numbers),
    })

    issues = []
    for issue_number in ranked_issue_numbers:
        issue = tools.get_issue(repo, issue_number)
        trace.append({"tool": "get_issue", "args": {"repo": repo, "issue_number": issue_number}, "result_count": 1 if issue else 0})
        if issue:
            issues.append(_compact_issue(issue))
        # else: honest miss -- either the issue truly doesn't exist, OR (confirmed
        # real case for httpx specifically) the repo's Issues access is closed
        # externally (see fetch_issue.py's docstring) -- either way, get_issue()
        # returning None here is correctly logged in the trace, not silently lost.

    MAX_DISCUSSIONS_FULL = 4  # global cap, not per-PR -- see docstring note below
    discussions_full_count = 0

    for pr_number in ranked_pr_numbers:
        pr = tools.get_pr(repo, pr_number)
        trace.append({"tool": "get_pr", "args": {"repo": repo, "pr_number": pr_number}, "result_count": 1 if pr else 0})
        if pr:
            prs.append(_compact_pr(pr))

        candidates = tools.find_linked_discussion(repo, pr_number)
        trace.append({"tool": "find_linked_discussion", "args": {"repo": repo, "pr_number": pr_number}, "result_count": len(candidates)})

        for i, cand in enumerate(candidates):
            # CONFIRMED REAL BUG (Y5 token-budget diagnosis, same class as
            # the PR cap fix): this cap was "i < 3" -- per PR, not global.
            # With MAX_PRS_TO_FETCH=8 PRs each contributing up to 3 full
            # discussions, real observed count was 11 full discussions for
            # one question (20,516 of the 53,714-char total). Now capped
            # globally across ALL PRs for this question.
            if i < 3 and discussions_full_count < MAX_DISCUSSIONS_FULL:
                full = tools.get_discussion(repo, cand["discussion_number"])
                trace.append({"tool": "get_discussion", "args": {"repo": repo, "discussion_number": cand["discussion_number"]}, "result_count": 1 if full else 0})
                if full:
                    full["_score"] = cand["score"]
                    full["_tied_with_count"] = cand.get("tied_with_count", 1)
                    discussion_candidates_full.append(_compact_discussion(full))
                    discussions_full_count += 1
            else:  # rest: title/date only, no extra tool call needed (data already in `cand`)
                discussion_candidates_brief.append({
                    "discussion_number": cand["discussion_number"],
                    "title": cand["title"],
                    "created_at": cand["created_at"],
                    "score": cand["score"],
                })

    # supplementary sources -- always run, cheap, independent of the above
    query_text = question  # docs/releases search directly on question text, same as resolve_symbol_reference's raw input
    doc_hits_raw = tools.search_docs(repo, query_text)
    trace.append({"tool": "search_docs", "args": {"repo": repo, "query": query_text}, "result_count": len(doc_hits_raw)})
    # CONFIRMED REAL BUG, FIXED (Y6/Y9/Y10 investigation): flat character
    # caps miss content buried past the cutoff. Same fix as release bodies:
    # _extract_relevant_excerpt() finds the real window with the highest
    # concentration of the QUESTION's own terms, instead of a fixed prefix.
    # Kept to top-3 (not all results) -- why's evidence set already compares
    # across many source types and previously hit real hard token-budget
    # failures (Y2/Y5) when nothing was capped.
    question_terms_for_doc_excerpt = [t for t in query_text.split() if len(t) > 2]
    doc_hits = [
        {"file_path": h["file_path"], "heading": h["heading"],
         "content": tools._extract_relevant_excerpt(h["content"], question_terms_for_doc_excerpt) if i < 3 else h["content"][:600],
         "score": h["score"]}
        for i, h in enumerate(doc_hits_raw)
    ]

    release_hits_raw = tools.search_releases(repo, query_text)
    trace.append({"tool": "search_releases", "args": {"repo": repo, "query": query_text}, "result_count": len(release_hits_raw)})
    # CONFIRMED REAL BUG, FIXED (Y9/Y10 investigation): flat character caps
    # (even a generous 2000, even widened to top-3) can still miss the
    # actually-relevant part of a long release body -- real case: got's
    # v10.0.0 has ~10+ separate "Why:" annotations, one per breaking change,
    # and the specific one for a given question ("retries" -> "calculateDelay"
    # rename) can sit well past any flat cutoff while an EARLIER, unrelated
    # "Why:" (e.g. about Node.js version) gets shown instead. Fixed by using
    # _extract_relevant_excerpt() to find the real window of text with the
    # highest concentration of the QUESTION's own terms, instead of always
    # taking a fixed prefix.
    question_terms_for_excerpt = [t for t in query_text.split() if len(t) > 2]
    release_hits = [
        {"tag_name": r.get("tag_name"), "name": r.get("name"),
         "body": tools._extract_relevant_excerpt(r.get("body") or "", question_terms_for_excerpt) if i < 3 else (r.get("body") or "")[:600],
         "match_count": r.get("match_count")}
        for i, r in enumerate(release_hits_raw)
    ]

    # CONFIRMED REAL GAP FIX: find_linked_discussion only ever searches
    # discussions THROUGH a specific PR's title/body -- it never had a way
    # to search discussions directly by topic. Real case: Discussion #1530
    # was fully indexed/fetchable but never surfaced because no PR in the
    # commit-history-derived evidence happened to reference it. This direct
    # search runs independently, same as search_docs/search_releases.
    direct_discussion_hits_raw = tools.search_discussions(repo, query_text)
    trace.append({"tool": "search_discussions", "args": {"repo": repo, "query": query_text}, "result_count": len(direct_discussion_hits_raw)})
    direct_discussion_hits = [_compact_discussion(d) for d in direct_discussion_hits_raw[:3]]  # cap same as the PR-linked full-detail cap

    # Merge into discussion_candidates_full (same schema/citation-source-id
    # format the synthesizer already expects for DISCUSSION# sources) --
    # avoid re-adding a discussion already found via the PR-linked path.
    already_have = {d.get("discussion_number") for d in discussion_candidates_full}
    for d in direct_discussion_hits:
        if d.get("discussion_number") not in already_have:
            discussion_candidates_full.append(d)
            already_have.add(d.get("discussion_number"))

    # Compact commits before returning -- author email, added/deleted line
    # counts, is_merge flags etc. are real fields but not needed by the
    # synthesizer; message + hash + pr_number is what actually carries
    # rationale signal. Cap total count too (214 commits' worth of even
    # compacted data is still excessive) -- keep the highest-relevance-
    # scored ones, same scoring already computed above, falling back to
    # most-recent for commits with no PR number to score by.
    #
    # CONFIRMED REAL CASE (Y5 token-budget diagnosis): version-bump/release
    # commits can carry an entire multi-paragraph CHANGELOG diff as their
    # commit message (real example: "Version 0.15.0 (#1301)" carried ~600
    # chars of unrelated deprecation notes). Message truncated to 200 chars
    # -- the first line/paragraph usually carries the real gist; a real
    # rationale worth keeping is rarely buried 400+ chars into a message.
    question_terms_final = [t for t in question.split() if len(t) > 2]
    commits_all.sort(key=lambda c: -_score_commit_relevance(c, question_terms_final))
    MAX_COMMITS_TO_KEEP = 30
    commits_compact = [
        {"commit_hash": c.get("commit_hash"), "message": (c.get("message") or "")[:200],
         "pr_number": c.get("pr_number"), "author_date": c.get("author_date")}
        for c in commits_all[:MAX_COMMITS_TO_KEEP]
    ]

    return {
        "tool_results": {
            "resolve_symbol_reference": resolution["matches"] if resolution["confidence"] in ("high", "medium") else [],
            "get_commit_history": commits_compact,
            "get_pr": prs,
            "get_issue": issues,
            "discussion_candidates_full": discussion_candidates_full,
            "discussion_candidates_brief": discussion_candidates_brief,
            "search_docs": doc_hits,
            "search_releases": release_hits,
        },
        "trace": trace,
        "resolution_confidence": resolution["confidence"],
    }


def plan_why(repo: str, question: str) -> dict:
    return _gather_why_evidence(repo, question)


def plan_unanswerable_why(repo: str, question: str) -> dict:
    """Same evidence-gathering as plan_why -- see _gather_why_evidence's
    docstring for why these two categories share one gatherer. The synthesizer
    (not the router) is responsible for correctly abstaining when this
    evidence doesn't support a confident rationale."""
    return _gather_why_evidence(repo, question)


CATEGORY_PLANS = {
    "where": plan_where,
    "what": plan_what,
    "topology": plan_topology,
    "how": plan_how,
    "why": plan_why,
    "unanswerable_why": plan_unanswerable_why,
}


def plan_and_execute(repo: str, category: str, question: str) -> dict:
    if category not in CATEGORY_PLANS:
        raise NotImplementedError(f"No plan yet for category={category!r} -- only {list(CATEGORY_PLANS)} implemented so far")
    return CATEGORY_PLANS[category](repo, question)


if __name__ == "__main__":
    # Smoke test against REAL questions from phase0_eval_questions.md
    print("=== where: W1's underlying symbol (via a 'how' question reusing it) ===")
    result = plan_and_execute("httpx", "what", "What does the `_transport_for_url` method in `httpx/_client.py` do?")
    print("confidence:", result["resolution_confidence"])
    print("trace:", result["trace"])
    print("matches:", result["tool_results"]["resolve_symbol_reference"])
    print()

    print("=== topology smoke test (Limits class) ===")
    result = plan_and_execute("httpx", "topology", "What does `Limits` call or get called by?")
    print("confidence:", result["resolution_confidence"])
    print("num callers found:", len(result["tool_results"]["get_callers"]))
    print("num callees found:", len(result["tool_results"]["get_callees"]))
    print()

    print("=== why smoke test (got Node.js requirement -- real answerable case) ===")
    result = plan_and_execute("got", "why", "Why does got require a certain Node.js version?")
    print("confidence:", result["resolution_confidence"])
    print("commits found:", len(result["tool_results"]["get_commit_history"]))
    print("PRs found:", len(result["tool_results"]["get_pr"]))
    print("release hits:", len(result["tool_results"]["search_releases"]))
    print("doc hits:", len(result["tool_results"]["search_docs"]))
    print()

    print("=== unanswerable_why smoke test (fabricated, expect thin/no evidence) ===")
    result = plan_and_execute("got", "unanswerable_why", "Why was the internal variable naming convention chosen in `source/core/index.ts`?")
    print("confidence:", result["resolution_confidence"])
    print("commits found:", len(result["tool_results"]["get_commit_history"]))
    print("PRs found:", len(result["tool_results"]["get_pr"]))
    print("discussion candidates (full):", len(result["tool_results"]["discussion_candidates_full"]))
