"""
Data-access layer for repo-assist's query/verifier phase.

Every function here is a thin, read-only wrapper around a real table in
data/code_graph.db. No LLM calls happen in this file. The planner will call
these as "tools"; the verifier will re-call them to check claims.

ASSUMPTIONS MADE FROM SCHEMA (flagged for verification against real data,
not guessed silently):

  A1. `imports(repo, from_file, to_file)`: assumed from_file IMPORTS to_file
      (i.e. to_file is the dependency). Verify with a known real import.
  A2. `symbols.name` is the bare identifier (e.g. "get") and
      `qualified_name` is the dotted/class-scoped path (e.g. "Client.get").
      search_symbols() searches `name`; get_symbol() searches `qualified_name`.
  A3. Multiple rows can share (repo, qualified_name) across different
      file_path/start_line (overloads, or same name in different classes).
      Every lookup function returns a LIST, never assumes uniqueness.
  A4. `summaries.delegates_to_verified` / `delegates_to_unverified` are
      stored as JSON-encoded lists of strings (matching Phase 2's Pydantic
      schema). If they're actually comma-separated strings instead, the
      parse helper below needs a one-line change.
  A5. The PR<->Discussion linker is an existing function in your codebase,
      not reimplemented here. Import path below is a PLACEHOLDER --
      point it at wherever that function actually lives.
"""

import json
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

from config import DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Phase 1: structural graph
# ---------------------------------------------------------------------------

def search_symbols(repo: str, name_substring: str, limit: int = 20) -> list[dict]:
    """Fuzzy search by bare name. Use when the question names a function/class
    informally (e.g. "the retry logic") rather than a known qualified name.
    FAST FIX: exact name matches now sort first, before substring matches --
    confirmed real case: bare "got" as a search term matched isGotInstance
    (substring) ahead of the real, exact top-level `got` function, since the
    old query had no exact-match preference at all."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM symbols
               WHERE repo = ? AND name LIKE ?
               ORDER BY (name != ?), name LIMIT ?""",
            (repo, f"%{name_substring}%", name_substring, limit),
        )
        return _rows(cur)


def get_symbol(repo: str, qualified_name: str) -> list[dict]:
    """Exact qualified-name lookup. May return >1 row (A3) -- caller must
    disambiguate by file_path/start_line if needed."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM symbols WHERE repo = ? AND qualified_name = ?""",
            (repo, qualified_name),
        )
        return _rows(cur)


def get_callers(repo: str, file_path: str, qualified_name: str, start_line: int) -> list[dict]:
    """Who calls this symbol. Returns symbol_edges rows where this symbol is
    the target (to_*), across all edge_types unless filtered."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM symbol_edges
               WHERE repo = ? AND to_file = ? AND to_qualified_name = ? AND to_start_line = ?""",
            (repo, file_path, qualified_name, start_line),
        )
        return _rows(cur)


def get_callees(repo: str, file_path: str, qualified_name: str, start_line: int) -> list[dict]:
    """What this symbol calls/instantiates/extends (from_* side)."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM symbol_edges
               WHERE repo = ? AND from_file = ? AND from_qualified_name = ? AND from_start_line = ?""",
            (repo, file_path, qualified_name, start_line),
        )
        return _rows(cur)


def get_extends_chain(repo: str, file_path: str, qualified_name: str, start_line: int) -> list[dict]:
    """Parent classes (EXTENDS edges only), one level. Walk repeatedly for
    full MRO -- deliberately not recursive here per the project's documented
    scoping decision against full MRO resolution."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM symbol_edges
               WHERE repo = ? AND edge_type = 'EXTENDS'
                 AND from_file = ? AND from_qualified_name = ? AND from_start_line = ?""",
            (repo, file_path, qualified_name, start_line),
        )
        return _rows(cur)


def get_imports(repo: str, file_path: str) -> list[dict]:
    """Files this file imports (see assumption A1)."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM imports WHERE repo = ? AND from_file = ?""",
            (repo, file_path),
        )
        return _rows(cur)


def get_files_importing(repo: str, target_file: str) -> list[dict]:
    """REVERSE of get_imports(): which files import/depend on target_file.
    CONFIRMED REAL GAP (full 56-question eval run): T2, T3, T4, T6 all ask
    'which files import X' -- this lookup direction was entirely missing
    from the tool layer, and plan_topology never attempted it, causing a
    total 0/6 topology score. This is the direct fix."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM imports WHERE repo = ? AND to_file = ?""",
            (repo, target_file),
        )
        return _rows(cur)


# ---------------------------------------------------------------------------
# Phase 2: history / provenance
# ---------------------------------------------------------------------------

def get_commit_history(repo: str, file_path: str, limit: Optional[int] = None) -> list[dict]:
    """Commits touching this file, most recent first."""
    q = """SELECT * FROM commits WHERE repo = ? AND file_path = ?
           ORDER BY author_date DESC"""
    params: tuple = (repo, file_path)
    if limit:
        q += " LIMIT ?"
        params = (repo, file_path, limit)
    with _connect() as conn:
        cur = conn.execute(q, params)
        return _rows(cur)


# Maps internal repo string -> real GitHub (owner, repo) slug, with dynamic
# fallback resolving from the git remote URL in repos/<repo>/.git/config.
GITHUB_OWNER_REPO = {
    "httpx": ("encode", "httpx"),
    "got": ("sindresorhus", "got"),
}


def resolve_github_owner_repo(repo: str) -> Optional[tuple[str, str]]:
    if repo in GITHUB_OWNER_REPO:
        return GITHUB_OWNER_REPO[repo]
    for base in [os.path.join("..", "repos", repo), os.path.join("repos", repo)]:
        git_dir = os.path.join(base, ".git")
        if os.path.isdir(git_dir):
            try:
                import subprocess
                res = subprocess.run(
                    ["git", "-C", base, "remote", "get-url", "origin"],
                    capture_output=True, text=True, timeout=5
                )
                if res.returncode == 0:
                    url = res.stdout.strip()
                    m = re.search(r"github\.com[:/]([^/]+)/([^/\.]+)", url)
                    if m:
                        slug = (m.group(1), m.group(2))
                        GITHUB_OWNER_REPO[repo] = slug
                        return slug
            except Exception:
                pass
    return None


def get_pr(repo: str, pr_number: int, allow_live_fetch: bool = True) -> Optional[dict]:
    """Cached PR data. Parses comments_json/reviews_json into real objects.

    REAL GAP FOUND AND FIXED (via router testing on got's why-question
    evidence gathering): this function was originally read-only against
    pr_cache, silently returning None on every cache miss. Confirmed via
    direct testing: pr_cache had ZERO rows for got and only 1 for httpx
    (the single PR we manually tested earlier) -- meaning every real
    why-question needing PR context got silently starved of evidence.

    fetch_pr.py (Phase 2, already built and working) implements the actual
    lazy fetch-on-miss design your project intended -- it just was never
    wired to this read path. Fixed by falling back to that real function on
    a genuine cache miss, matching the original "lazy PR fetching" design
    from the project context doc, rather than reimplementing fetch logic
    here a second time.

    allow_live_fetch=False lets a caller force read-only behavior (e.g. for
    a dry offline test) without touching GITHUB_TOKEN/network at all.
    """
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM pr_cache WHERE repo = ? AND pr_number = ?""",
            (repo, pr_number),
        )
        row = cur.fetchone()

    if row is None:
        if not allow_live_fetch:
            return None
        slug = resolve_github_owner_repo(repo)
        if slug is None:
            return None  # no known real GitHub slug for this repo -- can't fetch live
        owner, gh_repo = slug
        try:
            from fetch_pr import get_pr as live_get_pr
            live_get_pr(owner, gh_repo, pr_number, db_path=DB_PATH)
        except Exception:
            return None  # network error, missing token, PR truly doesn't exist, etc. -- honest miss, not a crash
        # re-read from cache now that (if successful) it was just populated
        with _connect() as conn:
            cur = conn.execute(
                """SELECT * FROM pr_cache WHERE repo = ? AND pr_number = ?""",
                (repo, pr_number),
            )
            row = cur.fetchone()
        if row is None:
            return None

    d = dict(row)
    for key in ("comments_json", "reviews_json"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass  # leave as raw string if not valid JSON
    return d


def get_issue(repo: str, issue_number: int, allow_live_fetch: bool = True) -> Optional[dict]:
    """Cached GitHub Issue data (distinct from PRs). Mirrors get_pr()'s exact
    lazy fetch-on-miss pattern, wired to fetch_issue.py.

    REAL, EXTERNAL, CONFIRMED CONSTRAINT (not a bug): httpx's maintainer
    closed off Issues access on encode/httpx (see fetch_issue.py's docstring
    for full details, confirmed via direct GraphQL testing). This means
    get_issue('httpx', N) will correctly return None for any real issue
    number -- an honest reflection of a real external state, not a failure
    of this function. get_issue('got', N) works normally since got's Issues
    remain open (confirmed via direct testing)."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM issue_cache WHERE repo = ? AND issue_number = ?""",
            (repo, issue_number),
        )
        row = cur.fetchone()

    if row is None:
        if not allow_live_fetch:
            return None
        slug = resolve_github_owner_repo(repo)
        if slug is None:
            return None
        owner, gh_repo = slug
        try:
            from fetch_issue import get_issue as live_get_issue
            live_get_issue(owner, gh_repo, issue_number, db_path=DB_PATH)
        except Exception:
            return None  # network error, missing token, issue closed-off/doesn't exist -- honest miss, not a crash
        with _connect() as conn:
            cur = conn.execute(
                """SELECT * FROM issue_cache WHERE repo = ? AND issue_number = ?""",
                (repo, issue_number),
            )
            row = cur.fetchone()
        if row is None:
            return None

    d = dict(row)
    if d.get("comments_json"):
        try:
            d["comments_json"] = json.loads(d["comments_json"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def get_discussion(repo: str, discussion_number: int) -> Optional[dict]:
    """Full indexed discussion, including nested reply comments_json."""
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM discussions_index WHERE repo = ? AND discussion_number = ?""",
            (repo, discussion_number),
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("comments_json"):
            try:
                d["comments_json"] = json.loads(d["comments_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


def find_linked_discussion(repo: str, pr_number: int, top_n: Optional[int] = None) -> list[dict]:
    """Runs the real, six-round-debugged linker live: link_pr_to_discussions.py's
    find_candidate_discussions(). Returns a RANKED LIST, title-weighted IDF score
    first (highest first).

    CONFIRMED REAL GAP (found via direct testing, httpx PR #3319): the scorer
    ties multiple discussions at identical score when they share the same single
    discriminating title term (e.g. 5-way tie on "SSLContext", including the
    real answer #3007 alongside 4 unrelated discussions).

    TRIED AND RULED OUT: time-proximity between PR and discussion date as a
    tiebreak. Tested against the one real ground-truth case (PR #3319 <-> #3007):
    it picked #3470 instead, because #3470's date happened to be numerically
    closer to the PR's date than #3007's, despite #3007 being the real answer.
    "Nearest preceding discussion only" was also considered and rejected before
    implementation -- confirmed (by direct recollection of real project history,
    not assumption) that the true rationale discussion can happen AFTER the PR
    too, so a precedes-PR constraint would silently exclude valid real answers.

    DECISION: no automatic numeric tiebreak. Disambiguating "which of several
    equally-scored discussions is the real design rationale" is a judgment call
    best made by reading actual discussion content (title, body, top comments),
    which is exactly what the planner/verifier LLM layer is for. Building a
    second hand-tuned formula here would repeat the same trap the original
    6-round scoring effort was trying to escape.

    Returns each candidate with a `tied_with_count` field: how many other
    candidates share its exact score. The planner should treat tied_with_count
    > 1 as a signal to fetch and compare full discussion content (via
    get_discussion()) before treating any single one as confident, rather than
    picking the first list item as if it were an unambiguous top match.
    """
    from link_pr_to_discussions import find_candidate_discussions

    pr = get_pr(repo, pr_number)
    if pr is None:
        return []

    with _connect() as conn:
        kwargs = {"top_n": top_n} if top_n is not None else {}
        candidates = find_candidate_discussions(
            conn, repo, pr.get("title") or "", pr.get("body") or "", **kwargs
        )

    score_counts: dict[float, int] = {}
    for c in candidates:
        score_counts[c["score"]] = score_counts.get(c["score"], 0) + 1
    for c in candidates:
        c["tied_with_count"] = score_counts[c["score"]]

    return candidates


# ---------------------------------------------------------------------------
# Phase 2: local summaries
# ---------------------------------------------------------------------------

def _parse_delegates(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [s.strip() for s in raw.split(",") if s.strip()]  # fallback per A4


def get_summary(repo: str, file_path: str, qualified_name: str, start_line: int) -> Optional[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM summaries
               WHERE repo = ? AND file_path = ? AND qualified_name = ? AND start_line = ?""",
            (repo, file_path, qualified_name, start_line),
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["delegates_to_verified"] = _parse_delegates(d.get("delegates_to_verified"))
        d["delegates_to_unverified"] = _parse_delegates(d.get("delegates_to_unverified"))
        return d


def get_source_snippet(repo: str, file_path: str, start_line: int, end_line: int, context_lines: int = 0) -> Optional[dict]:
    """Reads REAL source code lines directly from the checked-out repo on
    disk -- no summary, no LLM, ground-truth text. Addresses a CONFIRMED
    REAL GAP (full 56-question eval runs): questions needing exact facts
    (parameter default values, precise mechanism details like "insertion
    order not priority-sorted") failed even with correct symbol resolution
    and a real summary, because Phase 2's summaries are prose-level
    relational descriptions, not verbatim signatures/values. One confirmed
    case: H3 gave a confidently WRONG answer about mount-matching priority
    because no evidence source contained the actual literal code showing
    plain iteration order.

    file_path is the value as STORED in the symbols table (per-repo
    convention already handled here). start_line/end_line likewise come
    directly from a real symbols row.

    Per-repo path convention (confirmed via direct testing, same
    convention noted in build_docs_table.py): httpx's stored file_path is
    prefixed with the repo name itself (e.g. "httpx/_client.py"), got's is
    NOT prefixed (e.g. "source/core/index.ts"). The actual checkout lives
    at ../repos/httpx/... and ../repos/got/... respectively -- so the real
    disk path needs repo-specific handling, not a single naive join.
    """
    candidates = [
        os.path.join("..", "repos", repo, file_path),
        os.path.join("repos", repo, file_path),
    ]
    disk_path = None
    for c in candidates:
        if os.path.isfile(c):
            disk_path = c
            break

    if not disk_path:
        return None

    with open(disk_path, encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    lo = max(1, start_line - context_lines)
    hi = min(len(all_lines), end_line + context_lines)
    snippet_lines = all_lines[lo - 1:hi]  # 1-indexed start_line/end_line -> 0-indexed slice

    return {
        "repo": repo,
        "file_path": file_path,
        "start_line": lo,
        "end_line": hi,
        "content": "".join(snippet_lines),
    }


# ---------------------------------------------------------------------------
# Phase 3: risk / blast-radius
# ---------------------------------------------------------------------------

def get_risk(repo: str, file_path: str) -> Optional[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM risk_scores WHERE repo = ? AND file_path = ?""",
            (repo, file_path),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_top_risk_files(repo: str, limit: int = 10) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM risk_scores WHERE repo = ?
               ORDER BY risk_score DESC LIMIT ?""",
            (repo, limit),
        )
        return _rows(cur)


def get_risk_validation(repo: str, file_path: str) -> Optional[dict]:
    with _connect() as conn:
        cur = conn.execute(
            """SELECT * FROM risk_validation WHERE repo = ? AND file_path = ?""",
            (repo, file_path),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Docs (README/CHANGELOG/documentation) and got release notes
# Both built separately (build_docs_table.py, fetch_got_releases.py) --
# see those files for how `docs` and `release_cache` were populated and
# their known coverage gaps (e.g. docs table does NOT include got's
# release notes -- that's release_cache's job, fetched live from GitHub).
# ---------------------------------------------------------------------------

def _score_text_match(query_terms: list[str], text: str) -> int:
    """Simple count of how many query terms appear (case-insensitive).
    Used by search_releases() only -- search_docs() moved to IDF weighting
    after a confirmed real failure (see diagnose_search_docs.py output).
    search_releases() hasn't shown the same failure yet on real queries, so
    it stays simple until evidence says otherwise -- same discipline as
    before, not an oversight."""
    text_lower = text.lower()
    return sum(1 for t in query_terms if t.lower() in text_lower)


def _build_doc_corpus_frequency(rows: list[dict], query_terms: list[str]) -> dict:
    """How many chunks (out of the whole repo's docs corpus) contain each
    query term. Mirrors link_pr_to_discussions.py's build_corpus_term_frequency,
    but scoped to just the query's own terms (cheap -- no need to index the
    whole vocabulary when we only ever score against a handful of terms per
    call)."""
    freq = {}
    for t in query_terms:
        t_lower = t.lower()
        freq[t] = sum(1 for r in rows if t_lower in f"{r['heading']} {r['content']}".lower())
    return freq


def _score_text_match_idf(query_terms: list[str], text: str, term_freq: dict, doc_count: int, heading_terms: set) -> float:
    """IDF-weighted score, same principle as link_pr_to_discussions.py's
    find_candidate_discussions: rare terms count far more than common ones
    (fixes 'options' at 100/268 chunks drowning out 'auth' at 16/268 --
    confirmed real problem via diagnose_search_docs.py), and a term matching
    in the HEADING counts more than one only in body text (heading plays the
    role PR-title played in the linker: the author's own explicit label for
    what this chunk is about)."""
    text_lower = text.lower()
    score = 0.0
    for t in query_terms:
        t_lower = t.lower()
        if t_lower in text_lower:
            freq = max(1, term_freq.get(t, 1))
            idf = math.log(doc_count / freq)
            weight = 2.0 if t_lower in heading_terms else 1.0
            score += idf * weight
    return round(score, 2)


def search_docs(repo: str, query: str, limit: int = 5, widen_threshold: float = 2.0) -> list[dict]:
    """IDF-weighted search over doc chunks (README/CHANGELOG/documentation/
    migration-guides), heading-match weighted like a title.

    WIDENING (2nd fix from the same real failure): if the top match's score
    is weak (< widen_threshold), the real answer may be split across an
    adjacent chunk in the same file -- confirmed real case: got's
    `options.merge()` chunk describes the mechanism but contains zero
    occurrences of "auth", because the auth-specific rationale lives in a
    neighboring chunk within the same file, not this one. When widening
    triggers, the immediately preceding and following chunk (same file_path,
    chunk_index +/-1) are pulled in as `context` alongside the matched chunk,
    not re-scored independently -- they're supporting context, not separate
    hits."""
    query_terms = [t for t in query.split() if len(t) > 2]
    if not query_terms:
        return []
    with _connect() as conn:
        all_rows = _rows(conn.execute("SELECT * FROM docs WHERE repo = ?", (repo,)))

    term_freq = _build_doc_corpus_frequency(all_rows, query_terms)
    doc_count = len(all_rows)

    scored = []
    for r in all_rows:
        heading_terms = {t.lower() for t in query_terms if t.lower() in r["heading"].lower()}
        combined = f"{r['heading']} {r['content']}"
        score = _score_text_match_idf(query_terms, combined, term_freq, doc_count, heading_terms)
        if score > 0:
            r["score"] = score
            scored.append(r)

    scored.sort(key=lambda r: -r["score"])
    top = scored[:limit]

    if top and top[0]["score"] < widen_threshold:
        with _connect() as conn:
            for r in top:
                neighbors = _rows(conn.execute(
                    """SELECT * FROM docs WHERE repo = ? AND file_path = ?
                       AND chunk_index IN (?, ?)""",
                    (repo, r["file_path"], r["chunk_index"] - 1, r["chunk_index"] + 1),
                ))
                r["context"] = neighbors

    return top


def search_discussions(repo: str, query: str, limit: int = 5) -> list[dict]:
    """Direct, PR-INDEPENDENT search over discussions_index (bulk-indexed
    in Phase 2). Same IDF-weighted, title-weighted scoring principle as
    search_docs() and link_pr_to_discussions.py's real linker.

    CONFIRMED REAL GAP THIS FIXES: our existing find_linked_discussion()
    only ever searches discussions THROUGH a specific PR's title/body --
    it has no way to search discussions directly by topic. Confirmed real
    case: Discussion #1530 ("On switching the Transport API to a context-
    managed style") is fully indexed and fetchable, an exact real match for
    a real why-question about context managers -- but it was NEVER found,
    because no PR in that question's commit-history-derived evidence set
    happened to reference it. A direct search, independent of any PR,
    closes this gap."""
    query_terms = [t for t in query.split() if len(t) > 2]
    if not query_terms:
        return []
    with _connect() as conn:
        all_rows = _rows(conn.execute("SELECT * FROM discussions_index WHERE repo = ?", (repo,)))

    term_freq = {}
    for t in query_terms:
        t_lower = t.lower()
        term_freq[t] = sum(
            1 for r in all_rows
            if t_lower in f"{r.get('title') or ''} {r.get('body') or ''}".lower()
        )
    doc_count = max(1, len(all_rows))

    scored = []
    for r in all_rows:
        title_lower = (r.get("title") or "").lower()
        title_terms = {t.lower() for t in query_terms if t.lower() in title_lower}
        combined = f"{r.get('title') or ''} {r.get('body') or ''}"
        combined_lower = combined.lower()

        score = 0.0
        for t in query_terms:
            t_lower = t.lower()
            if t_lower in combined_lower:
                freq = max(1, term_freq.get(t, 1))
                idf = math.log(doc_count / freq)
                weight = 2.0 if t_lower in title_terms else 1.0
                score += idf * weight

        if score > 0:
            r["score"] = round(score, 2)
            r["_score"] = r["score"]  # match _compact_discussion's expected field name (same convention as find_linked_discussion's candidates)
            if r.get("comments_json"):
                try:
                    r["comments_json"] = json.loads(r["comments_json"])
                except (json.JSONDecodeError, TypeError):
                    pass  # leave as raw string if not valid JSON, same fallback as get_discussion()
            scored.append(r)

    scored.sort(key=lambda r: -r["score"])
    return scored[:limit]


def _extract_relevant_excerpt(text: str, query_terms: list[str], window_size: int = 800, prefix_chars: int = 400) -> str:
    """CONFIRMED REAL FIX: flat character caps (even a generous 2000) can
    still miss the actually-relevant part of a long document -- real case:
    got's v10.0.0 release notes are ~5000+ chars, and the specific "Why:"
    line for a particular renamed option can sit well past any reasonable
    flat cutoff, while unrelated earlier "Why:" lines (for a DIFFERENT
    rename) get shown instead.

    Instead of truncating from the start, this scans the FULL text for the
    window with the highest concentration of real query terms, and returns
    that window (with some surrounding context) instead. Always includes
    the first `prefix_chars` too (real intros/context, e.g. "why this
    release exists" framing, are often worth keeping regardless of which
    window scores highest)."""
    if not text or not query_terms:
        return text[:prefix_chars] if text else ""

    text_lower = text.lower()
    terms_lower = [t.lower() for t in query_terms]

    best_start = 0
    best_score = -1
    step = window_size // 2  # overlapping windows, so a match spanning a boundary isn't missed
    for start in range(0, max(1, len(text) - window_size), step):
        window = text_lower[start:start + window_size]
        score = sum(window.count(t) for t in terms_lower)
        if score > best_score:
            best_score = score
            best_start = start

    if best_score <= 0:
        # no real match found anywhere -- fall back to a plain prefix, same
        # as the old behavior, rather than returning an arbitrary window
        return text[:prefix_chars]

    prefix = text[:prefix_chars]
    excerpt_start = max(best_start - 100, prefix_chars)  # don't re-include the prefix twice
    excerpt = text[excerpt_start:excerpt_start + window_size + 200]

    if excerpt_start <= prefix_chars:
        return prefix + excerpt if excerpt_start == prefix_chars else prefix
    return f"{prefix}\n...\n{excerpt}"


def search_releases(repo: str, query: str, limit: int = 5) -> list[dict]:
    """Keyword search over cached GitHub release bodies (currently only
    populated for got -- see fetch_got_releases.py). Same simple match-count
    scoring as search_docs(), for the same reason: unverified whether a more
    complex ranking is even needed here yet."""
    query_terms = [t for t in query.split() if len(t) > 2]
    if not query_terms:
        return []
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM release_cache WHERE repo = ?", (repo,))
        rows = _rows(cur)

    scored = []
    for r in rows:
        combined = f"{r.get('name') or ''} {r.get('body') or ''}"
        score = _score_text_match(query_terms, combined)
        if score > 0:
            r["match_count"] = score
            scored.append(r)

    scored.sort(key=lambda r: -r["match_count"])
    return scored[:limit]
