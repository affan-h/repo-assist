"""
Phase 2, final component -- local symbol summarization via Ollama.

Design decisions, each verified against real evidence before building:

  - SEQUENTIAL, not concurrent: benchmarked directly on this hardware
    (2015 dual-core i5, no GPU). 4 concurrent calls via asyncio+thread-
    executor took 27.0s vs 29.5s sequential -- an ~8% difference, within
    noise, not a real speedup. Confirmed cause: no GPU to overlap
    matrix-multiply work across requests, so Ollama serializes the
    actual compute regardless of how many requests Python sends
    concurrently. Concurrent code was NOT built, since it would add
    real complexity (async/await, thread-executor error handling,
    harder-to-read progress logs) for no measured benefit.

  - keep_alive is set explicitly on every call, using a long duration
    (1 hour) so the model never unloads between symbols during a
    multi-hour batch run. Verified this parameter is real and
    supported directly by the native `ollama` Python library (not the
    OpenAI-compatible shim, where a GitHub issue documents keep_alive
    being silently ignored -- we deliberately use the native library,
    not that shim, avoiding that specific documented bug).

  - qwen2.5-coder:1.5b confirmed via direct test: ~7-8s per short
    prompt, accurate and appropriately-scoped output quality (correctly
    explained Python decorators with concrete examples, no rambling).

  - Docstrings extracted via tree-sitter (Step 7's extractor already
    captures these) are fed into the prompt as context, per the
    original Phase 2 design -- NOT relied on alone, since a docstring
    describes intent in isolation while we also want the model to
    incorporate real callers/callees from the graph (Phase 1's CALLS
    edges) for relational context, not just restate the docstring.

Storage: a new 'summaries' table, keyed by (repo, file_path,
qualified_name, start_line) -- same composite key discipline as
graph_schema.py's symbols table, since two symbols can share a
qualified_name (the getter/setter bug from Step 8).

Run with:
    python3 src/summarize_symbols.py [--limit N] [--repo httpx|got]
    (--limit for a quick test batch before committing to the full run)
"""

import sys
import time
import sqlite3
import argparse

from config import DB_PATH, REPOS_DIR
# NOTE: `ollama` is deliberately NOT imported at module level. It's imported
# inside summarize_one() instead, so that the schema/symbol-fetching/
# snippet-extraction/prompt-building logic in this file can be tested
# independently of whether Ollama is installed or running -- found this
# mattered directly: importing it at the top made the entire file fail to
# even load in a test environment without Ollama, blocking verification of
# unrelated logic that has nothing to do with the model call itself.


MODEL = "qwen2.5-coder:1.5b"
KEEP_ALIVE_SECONDS = 3600  # 1 hour -- long enough to cover a multi-hour batch
                            # run without the model unloading between calls


def init_summaries_table(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS summaries (
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            summary TEXT,
            model TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (repo, file_path, qualified_name, start_line)
        );
    """)

    # Migration guard, same pattern as extract_pr_numbers.py's
    # pr_number/related_issue_refs columns: CREATE TABLE IF NOT EXISTS
    # correctly does nothing when the table already exists from an
    # earlier version of this script (confirmed this is exactly what
    # happened -- the real error was "table summaries has no column
    # named delegates_to" on a pre-existing database), so a new column
    # needs its own explicit ALTER TABLE, guarded by checking it isn't
    # already present (ALTER TABLE ADD COLUMN errors if run twice).
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(summaries)")
    existing_columns = {row[1] for row in cur.fetchall()}
    if "delegates_to" not in existing_columns:
        cur.execute("ALTER TABLE summaries ADD COLUMN delegates_to TEXT")
    # Split delegates_to into verified/unverified, per the third-way fix:
    # rather than trust or suppress the model's claims uniformly, check
    # each against our real known-symbol table after generation.
    if "delegates_to_verified" not in existing_columns:
        cur.execute("ALTER TABLE summaries ADD COLUMN delegates_to_verified TEXT")
    if "delegates_to_unverified" not in existing_columns:
        cur.execute("ALTER TABLE summaries ADD COLUMN delegates_to_unverified TEXT")

    conn.commit()
    return conn


def get_all_known_symbol_names(conn: sqlite3.Connection, repo: str) -> set[str]:
    """
    Real, third fix for the delegates_to noise problem (stdlib/builtin
    calls like randomUUID(), sys.exit(1), Math.random() being mixed in
    with genuine project-symbol relationships): rather than choose
    between trusting the model's claims uniformly or making it more
    conservative via prompting (neither of which distinguishes "real
    project symbol we didn't index" from "not a project symbol at all"),
    build a lookup set of every symbol name we actually know about --
    both bare names (e.g. "request") and qualified names (e.g.
    "Client.request") -- so the model's claims can be checked against
    real ground truth after generation, not filtered by asking the
    model to self-censor.
    """
    cur = conn.cursor()
    cur.execute("SELECT name, qualified_name FROM symbols WHERE repo = ?", (repo,))
    names = set()
    for name, qualified_name in cur.fetchall():
        names.add(name)
        names.add(qualified_name)
    return names


def verify_delegates_to(claimed: list[str], known_names: set[str]) -> tuple[list[str], list[str]]:
    """
    Splits the model's claimed delegates_to entries into (verified,
    unverified) by checking each against known_names. Two normalization
    steps, both found necessary via direct testing against real output:

      1. Strip call syntax: 'randomUUID()' or 'sys.exit(1)' -> compare
         the part before the first '(', since the model sometimes
         includes call syntax that a bare symbol name lookup wouldn't
         match even for genuinely real, indexed symbols.

      2. REAL BUG FOUND AND FIXED: 'httpx.request' (module-qualified)
         was being marked unverified even though 'request' IS a real,
         known symbol -- our qualified-name scheme is "ClassName.method"
         (e.g. "Client.request"), not "modulename.function", so a
         model claim using module-style qualification never matched.
         Confirmed via real output: put()'s claim 'httpx.request' was
         wrongly unverified while the graph's own ground truth for the
         same relationship is plainly 'request'. Now also tries the
         LAST dot-segment as a fallback (e.g. "httpx.request" ->
         "request") before giving up, since this is a legitimate,
         common way to reference a top-level function, not a
         hallucination.
    """
    verified = []
    unverified = []
    for claim in claimed:
        normalized = claim.split("(")[0].strip()
        last_segment = normalized.split(".")[-1] if "." in normalized else None

        if normalized in known_names or claim in known_names or (last_segment and last_segment in known_names):
            verified.append(claim)
        else:
            unverified.append(claim)
    return verified, unverified


def get_symbols_to_summarize(conn: sqlite3.Connection, repo: str | None, limit: int | None):
    cur = conn.cursor()
    query = "SELECT repo, file_path, qualified_name, name, kind, start_line, end_line, parent_class FROM symbols"
    params = []
    if repo:
        query += " WHERE repo = ?"
        params.append(repo)
    query += " ORDER BY repo, file_path, start_line"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    cur.execute(query, params)
    return cur.fetchall()


def already_summarized(conn: sqlite3.Connection, repo: str, file_path: str, qualified_name: str, start_line: int) -> bool:
    """Skip symbols already summarized -- makes the pipeline safely
    resumable if interrupted partway through a multi-hour run."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM summaries WHERE repo=? AND file_path=? AND qualified_name=? AND start_line=?",
        (repo, file_path, qualified_name, start_line),
    )
    return cur.fetchone() is not None


def get_source_snippet(repo_root: str, file_path: str, start_line: int, end_line: int) -> str:
    """Read the actual source lines for this symbol directly from disk."""
    full_path = f"{repo_root}/{file_path}"
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # start_line/end_line are 1-indexed, inclusive
        snippet_lines = lines[start_line - 1:end_line]
        return "".join(snippet_lines)
    except (FileNotFoundError, IndexError):
        return ""


def get_callees(conn: sqlite3.Connection, repo: str, qualified_name: str) -> tuple[list[str], list[str]]:
    """
    Real fix for a quality gap found in actual test output: summaries
    like get()'s only restated "sends a GET request" -- mechanical
    behavior already visible in the source -- and never mentioned that
    get() delegates to request(), even though that's exactly the kind
    of relational fact Phase 1's CALLS graph already gives us for free.
    This fetches the real callees so the prompt can ask the model to
    incorporate them explicitly, rather than re-derive relationships
    the graph already knows from scratch (and likely miss them, as the
    test run showed).

    Returns (called_names, instantiated_names) SEPARATELY -- found via
    direct testing that bundling both edge types into one generic
    "(calls)"/"(instantiates)" suffix produced an awkward, redundant
    instruction once the prompt's own wording also used a fixed verb
    (e.g. "calls request (calls)"). Keeping them separate lets the
    prompt phrase each naturally: "calls X" vs "creates an instance of Y".
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT to_qualified_name, edge_type FROM symbol_edges
        WHERE repo = ? AND from_qualified_name = ? AND edge_type IN ('CALLS', 'INSTANTIATES')
    """, (repo, qualified_name))
    rows = cur.fetchall()
    called = [r[0] for r in rows if r[1] == "CALLS"]
    instantiated = [r[0] for r in rows if r[1] == "INSTANTIATES"]
    return called, instantiated


from pydantic import BaseModel

# REAL CHANGE FROM PROSE-BASED PROMPTING, documented here rather than in
# the Pydantic docstring below (which gets serialized into the JSON
# schema sent to the model on every call -- keeping it out saves real
# prompt tokens and avoids feeding the model our own implementation
# commentary as if it were part of the task):
#
# Two prior attempts (a soft conditional instruction, then a direct
# "MUST" command) both produced inconsistent results on real test data --
# only ~2 of 7 symbols with a known real callee relationship (confirmed
# correct via get_callees()) mentioned it in the free-form summary. This
# is not a prompt-wording problem we kept failing to solve; it's evidence
# that PROSE-BASED instruction-following is unreliable on this 1.5B model
# for this specific task.
#
# Switching to Ollama's `format` parameter (verified via research: it
# uses grammar-constrained decoding, masking invalid tokens at the
# sampling level -- NOT the same mechanism as asking nicely in a prompt)
# makes "delegates_to" a REQUIRED part of the output structure. The model
# cannot skip it the way it could skip a sentence in free prose, because
# token generation is constrained to only produce valid JSON matching
# this schema at every step.


class SymbolSummary(BaseModel):
    """A structured summary of one code symbol."""
    purpose: str  # 1-2 sentence description of what this symbol does
    delegates_to: list[str]  # symbols this one calls/instantiates, per the
                              # graph-provided context -- REQUIRED field,
                              # not an optional mention buried in prose


def build_prompt(name: str, kind: str, parent_class: str | None, source_code: str, called: list[str], instantiated: list[str]) -> str:
    context = f"a {kind}"
    if parent_class:
        context += f" on class {parent_class}"

    known_relationships = called + instantiated
    relationship_note = ""
    if known_relationships:
        relationship_note = f"\nKnown relationships from static analysis: {', '.join(known_relationships)}\n"

    return f"""You are documenting a codebase. Analyze this {kind} and provide structured information about it.

Name: {name} ({context})

Source:
```
{source_code}
```
{relationship_note}
Provide the purpose (1-2 sentences) and list every symbol this {kind} delegates to (calls or instantiates), including any listed in the known relationships above."""


def summarize_one(
    name: str, kind: str, parent_class: str | None, source_code: str,
    called: list[str], instantiated: list[str],
) -> dict:
    import ollama  # deferred import -- see module-level note

    if not source_code.strip():
        return {"purpose": "(source unavailable)", "delegates_to": []}

    prompt = build_prompt(name, kind, parent_class, source_code, called, instantiated)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=SymbolSummary.model_json_schema(),
        options={"temperature": 0},  # verified via research: low/zero
                                       # temperature is recommended for
                                       # reliable schema adherence
        keep_alive=KEEP_ALIVE_SECONDS,
    )

    try:
        parsed = SymbolSummary.model_validate_json(response["message"]["content"])
        return {"purpose": parsed.purpose, "delegates_to": parsed.delegates_to}
    except Exception as e:
        # Even grammar-constrained decoding can occasionally produce
        # invalid output (e.g. truncation mid-generation) -- confirmed
        # via research this is a real, if rarer, failure mode, not
        # something to silently assume never happens. Fall back to
        # raw text rather than crash the whole batch run over one symbol.
        return {"purpose": f"(parse error: {e}) raw={response['message']['content'][:200]}", "delegates_to": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only summarize the first N symbols (for testing)")
    parser.add_argument("--repo", type=str, default=None, help="Only summarize this repo")
    args = parser.parse_args()

    conn = init_summaries_table(DB_PATH)

    symbols = get_symbols_to_summarize(conn, args.repo, args.limit)
    print(f"Found {len(symbols)} symbol(s) to process.\n")

    known_names_cache: dict[str, set[str]] = {}

    done = 0
    skipped = 0
    start_time = time.time()

    for repo, file_path, qualified_name, name, kind, start_line, end_line, parent_class in symbols:
        if already_summarized(conn, repo, file_path, qualified_name, start_line):
            skipped += 1
            continue

        if repo not in known_names_cache:
            known_names_cache[repo] = get_all_known_symbol_names(conn, repo)

        repo_root = str(REPOS_DIR / repo)
        source_code = get_source_snippet(repo_root, file_path, start_line, end_line)
        called, instantiated = get_callees(conn, repo, qualified_name)
        result = summarize_one(name, kind, parent_class, source_code, called, instantiated)

        verified, unverified = verify_delegates_to(result["delegates_to"], known_names_cache[repo])

        import json as json_module
        verified_json = json_module.dumps(verified)
        unverified_json = json_module.dumps(unverified)

        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO summaries
               (repo, file_path, qualified_name, start_line, summary,
                delegates_to_verified, delegates_to_unverified, model, generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (repo, file_path, qualified_name, start_line, result["purpose"],
             verified_json, unverified_json, MODEL, str(time.time())),
        )
        conn.commit()

        done += 1
        elapsed = time.time() - start_time
        avg = elapsed / done
        remaining = len(symbols) - skipped - done
        eta_minutes = (remaining * avg) / 60

        print(f"[{done}/{len(symbols) - skipped}] {repo}:{qualified_name} "
              f"({avg:.1f}s/symbol avg, ~{eta_minutes:.0f}min remaining)")
        print(f"    purpose: {result['purpose'][:120]}")
        print(f"    delegates_to (verified against known symbols): {verified}")
        if unverified:
            print(f"    delegates_to (unverified -- stdlib/builtin/unindexed): {unverified}")
        known_real = called + instantiated
        if known_real:
            print(f"    delegates_to (graph ground truth from CALLS/INSTANTIATES): {known_real}")

    print(f"\nDone. {done} summarized, {skipped} already had a summary (skipped).")
    conn.close()


if __name__ == "__main__":
    main()
