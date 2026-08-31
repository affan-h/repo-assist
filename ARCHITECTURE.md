# repo-assist — Architecture Contract

**Read this before touching code. This document defines invariants Antigravity
must not violate. If a task requires violating one of these, stop and get
explicit sign-off first — don't let the agent quietly work around it.**

Grounded against: Sourcegraph Cody's code-graph architecture, Aider's
tree-sitter + PageRank repo map, code-graph-rag's unified multi-language
schema, and Arafat et al. 2025 ("Citation-Grounded Code Comprehension,"
arXiv:2512.12117) — the closest academic match to this system's design
(hybrid BM25+dense retrieval, import-graph expansion, mechanical citation
verification). Cite that paper in interviews; it validates this isn't a toy
architecture.

---

## 0. What must never break

These are the five invariants. Everything else in this doc explains *how*
to uphold them.

1. **No query for repo A ever returns evidence from repo B.** Every table,
   every retrieval call, every cache key is scoped by `repo_id`.
2. **Every claim the system makes must trace to a real `[file:start-end]`
   citation that mechanically overlaps a retrieved chunk.** No exceptions,
   no "the model said so."
3. **A partially-ingested repo is never queryable as if it were ready.**
   Status is explicit and visible at every stage.
4. **Existing behavior on `httpx` and `got` (the original two repos) does
   not regress.** They're your regression suite — if a "generalization"
   change breaks the numbers on these two, that's a bug, not a tradeoff.
5. **Every ingestion stage fails visibly and is resumable**, not silently
   stuck or silently wrong.

---

## 1. Extractor Contract (Phase 1)

Language dispatch is per-**file**, not per-repo: `.py` → Python
tree-sitter grammar, `.ts`/`.tsx` → TypeScript grammar. Both grammars must
emit the **same node/edge schema** below — language is a field, not a
fork in the pipeline (this is how code-graph-rag unifies 13 languages
under one graph; don't reinvent it more heavily than that).

### Symbol node (minimum required fields)
```
{
  id: str            # stable, unique within repo_id
  repo_id: str
  kind: str           # "function" | "method" | "class" | "property" | "module"
  language: str        # "python" | "typescript"
  name: str
  qualified_name: str    # e.g. "werkzeug.routing.MapAdapter.match"
  file_path: str        # relative to repo root
  start_line: int
  end_line: int
  source_text: str
}
```

### Edge (minimum required fields)
```
{
  source_id: str
  target_id: str
  edge_type: str    # "CALLS" | "IMPORTS" | "INSTANTIATES" | "EXTENDS"
  repo_id: str
}
```

### Rules
- **Don't pre-spec every language edge case.** The original project found
  its hardest bugs empirically (module-level instantiations invisible to
  the call resolver; property-assigned TS functions invisible to the
  extractor) — expect the same here. When Antigravity hits a new node
  shape on a new repo, it fixes the extractor and **adds one line to the
  bug log in §6**, it does not silently drop the symbol.
- **Exclusions are a static list, not a design phase.** Skip
  `node_modules/`, `.venv/`, `venv/`, `dist/`, `build/`, `__pycache__/`,
  `.git/`, anything matched by the repo's own `.gitignore`, and files
  over ~2000 lines (generated/vendored code smell — log and skip, don't
  hang on it).
- **Qualified names are the join key.** `resolve_calls.py` and
  `resolve_inheritance.py` must resolve against `qualified_name`, not
  `name` alone — this is what the getter/setter collision bug in the
  original project was about. Don't reintroduce it.

### Recommended addition: PageRank over the symbol graph
Production repo-intelligence tools (Aider's repo-map, in production use)
rank symbols by importance via PageRank over the CALLS/IMPORTS graph, not
just raw structural presence. This is cheap (~50 lines with `networkx`,
runs in milliseconds even on large graphs) and directly targets this
project's weakest eval category (`topology`, 33%) — that category is
fundamentally "which symbols matter architecturally," which is exactly
what PageRank answers and raw graph traversal doesn't. Add this as a
Phase 1 task: `rank_symbols.py`, output `pagerank_score` on each symbol
node, expose it to `router.py` as a tiebreaker/ranking signal.

---

## 2. Graph & Repo Isolation Contract (Phase 1–3)

The real design question is not "one SQLite file vs. many" — it's:
**can a query for repo A ever see repo B's data?** The invariant:

```
repo_id
   ↓
graph (symbols, edges — scoped by repo_id)
   ↓
retrieval (BM25 index, embeddings — scoped by repo_id)
   ↓
synthesis (router, synthesizer — repo_id passed explicitly, never inferred)
   ↓
verification (citation check — only against that repo_id's retrieved chunks)
```

- Every table gets a `repo_id` column. Every query in `query_tools.py`
  takes `repo_id` as an explicit, required parameter — never a global,
  never inferred from "the last ingested repo."
- If you go with one shared SQLite DB (recommended over per-repo files,
  since this is now a multi-repo web app): every index that matters for
  query performance should be `(repo_id, ...)`, not just `(...)`.
- `qualified_name` is unique **within** a `repo_id`, not globally. Two
  repos can both have `app.main.handler` — that's fine and expected.

---

## 3. Ingestion State Machine (Phase 2–3, the FastAPI wrapper)

```
QUEUED → CLONED → PARSED → GRAPH_BUILT → HISTORY_ATTACHED
       → RISK_SCORED → READY
                     ↘ FAILED (with a reason, at whichever stage it failed)
```

- Every stage writes its status **before** starting and **after**
  finishing, so a killed process leaves the repo visibly stuck at a named
  stage, not silently "in progress" forever. A simple `status` +
  `status_updated_at` + `error_message` column on the `repos` table
  covers this — no need for a heavier job framework at this scale.
- `GET /repos/{id}/status` must reflect this truthfully. A repo in any
  state other than `READY` must not be queryable — `POST
  /repos/{id}/ask` returns a clear 409/425-style error, not a partial or
  wrong answer.
- **Bounded history policy** (this is a real scope guard, not
  optional): fetch at most the most recent **300 PRs, 300 issues, and
  a fixed set of Discussion categories** per repo by default. Unbounded
  history fetching on a large repo can consume your entire "days"
  timeline on one bad test run. Make the bound a constant in one place,
  not scattered across scripts.
- **Graceful degradation, not pipeline failure**, when a repo lacks
  GitHub-specific history (no Discussions, no Issues tracker, squash-only
  PRs): log what was skipped and why, move to the next stage. The
  original project's own finding — that `got` vs `httpx` had 89% vs 35%
  PR-traceability for structural reasons — is your proof this is
  *expected variance*, not a bug to chase.
- Risk scoring (Phase 3) gets relabeled honestly in any UI/docs copy:
  it's a **heuristic** (churn × complexity), empirically validated only
  on `got` in the original project. Don't claim it's a general predictor.

---

## 4. Evidence & Citation Contract (the query/verifier layer)

This is the project's strongest asset — protect it above all else.

- Every synthesized answer must include citations in `[file:start-end]`
  format.
- Citation verification is **mechanical**: check that the cited range
  overlaps a chunk that was actually retrieved for this query, via
  interval arithmetic. Not a second LLM call, not "trust the model."
- **Add an auto-cite fallback** if not already robust: if the model's
  answer contains no valid self-citation, append the citation of the
  highest-ranked retrieved chunk rather than returning an uncited answer.
  The Arafat et al. paper found this is the difference between 100%
  citation coverage and as low as 22% for weaker models — cheap
  insurance, do this regardless of which LLM you end up calling.
- Closed-list citation: the model can only cite something that was
  actually enumerated in its prompt. Keep this — it's already in the
  original design and it's correct.
- When retrieval spans multiple files (expect this for most non-trivial
  questions — the comparable academic study found ~62% of queries need
  cross-file evidence), prefer packing strategies that spread across
  files over ones that dump many chunks from one file. This is what your
  existing import-graph expansion in `router.py` is for; keep it and
  don't let a "simplify for the demo" pass remove it.

---

## 5. What Antigravity is explicitly told, every session

Paste this at the start of any Antigravity task that touches Phase 1–3,
the DB schema, or the query layer:

> Do not refactor unrelated code. Preserve existing behavior for httpx
> and got — they are the regression suite. Every generalization must
> maintain backward compatibility with the existing graph/query schema
> unless the schema change is explicitly documented here and every
> downstream consumer (query_tools.py, router.py, synthesizer.py,
> grader.py) is updated in the same change. Do not touch
> graph_schema.py or table definitions without calling it out
> explicitly before making the edit.

---

## 6. Bug log (append here as Antigravity finds new failure modes)

Seed this with the originals so a fresh agent pass doesn't reintroduce
them:

- Module-level instantiations (`DEFAULT_LIMITS = Limits(...)`) were
  invisible to the call resolver — it only recognized calls inside a
  function/method body.
- Property-assigned TS functions (`got.extend = (...) => {...}`) were
  invisible to the TS symbol extractor — distinct tree-sitter node shape.
- `pr_cache` lazy-fetch existed in code but was never actually wired to
  trigger.
- Flat character-count truncation cut off answers mid-sentence; replaced
  with relevance-scored excerpt extraction.
- Citation validation rejected correct citations over bracket-formatting
  mismatch (`"[CODE#extend]"` vs `"CODE#extend"`).
- Plain-English questions without a backtick-quoted identifier had no
  reliable resolution path — fixed via full-text source search fallback.

*(New entries go below this line as you find them.)*

- **CLI_README.md's setup instructions are stale.** It lists
  `GROQ_API_KEY` and `CEREBRAS_API_KEY` as required — both providers are
  dead (Cerebras now requires payment, Groq's availability was never
  reconfirmed). Every model string in the codebase now points at Gemini.
  Actual requirement: `GEMINI_API_KEY` + `GITHUB_TOKEN` only. Fix the
  docs, don't chase the old env vars.
- **Same-vendor verification, not cross-vendor.** Correction to an
  earlier note here: `orchestrator.py` DOES force the verifier onto a
  different model than the synthesizer (a real diversity requirement,
  implemented). The actual weakness is narrower than "no independence"
  — both synthesis and verification now run on *Gemini* models (just
  different ones), not on genuinely different vendors as originally
  planned pre-Cerebras/Groq collapse. State it that way in interviews:
  "cross-model but same-vendor verification," not "no independence."
- **Phase 2 requires a local Ollama instance running `qwen2.5-coder:1.5b`**
  for symbol-relationship summarization. Not mentioned in CLI_README.md.
  Must be installed and the model pulled before Phase 2 will complete.
- **KNOWN UNRESOLVED — do not attempt as a quick task:** v2's
  `why`/`unanswerable_why` regression. The synthesizer's
  citation-sufficiency rule needs two separate thresholds (strict for
  catching hallucinated `unanswerable_why` cases, loose for genuine
  `why` questions with real-but-indirect evidence) but currently shares
  one rule, and two prior attempts to fix one broke the other. Treat
  this as a real, scoped diagnosis task on its own — likely needs the
  retrieval/graph/synthesis/verification failure-type breakdown from
  ARCHITECTURE.md §7 step 5, not a quick threshold tweak.
- **`topology`'s weak score is partly a grading artifact, not purely a
  bug.** Some multi-hop call chains require control-flow-sensitive
  return-type inference that was deliberately scoped out of Phase 1.
  The system correctly reports partial/unknown chains rather than
  guessing, but the benchmark's all-or-nothing grading scores this as a
  full failure even when the partial trace is correct. Worth knowing
  before "fixing" this category — some of the gap may be ungradeable
  with the current strict rubric, not fixable in the pipeline.
- **Free-tier Gemini quota is real and tight** (roughly 20–500
  requests/day depending on model). A full `--engine both` grader run
  can exhaust it mid-run. Use `--retry-failed` to resume without
  re-spending quota, and `--override-model` to switch models
  mid-recovery without editing source.
- **README's "Building the database from scratch" section never
  actually states the `git clone` step for httpx/got.** `build_full_graph.py`
  hardcodes `repos/httpx` and `repos/got` (relative to `src/`) and
  assumes they're already cloned there — the README jumps straight from
  env vars to "Phase 1 parses both repos" with no clone command shown.
  This is exactly the gap that matters most for the generalization goal:
  the eventual "any repo" pipeline needs an explicit, automated clone
  step keyed by `repo_id`/URL, not a manual pre-clone assumption. Treat
  auto-cloning as part of the Phase 1 generalization task, not a
  separate afterthought.
- **v1 (rule-based router, zero LLM in retrieval) is the shipped
  default and the stronger engine** (41.1% vs v2's 35.7% overall,
  87.5% vs 37.5% on the safety-critical `unanswerable_why` category).
  Use `--engine v1` as the baseline for all regression testing unless
  specifically testing v2. v2 exists and works but hasn't earned
  default status by the project's own pre-registered gate.
- **CONFIRMED, IMPORTANT FOR GENERALIZATION: every path in the pipeline
  is relative to `cwd`, not to the script's own location.**
  `build_full_graph.py`'s `Path("repos/httpx")` and
  `save_graph(cg, "data/code_graph.db")`, and `query_tools.py`'s
  `DB_PATH`, all assume the process is launched with `cwd == src/`.
  Confirmed by hitting this twice: once for `repos/` (had to symlink
  `src/repos -> ../repos`), once for `data/` (had to `mkdir src/data`).
  This is fine for CLI usage but **will break the moment ingestion runs
  as a FastAPI background task**, where cwd can't be assumed. Before
  the FastAPI wrapper stage (ARCHITECTURE.md §7 step 6), convert these
  to explicit paths derived from a config value (e.g.
  `Path(__file__).parent` or an env-configured data root), not
  bare relative `Path("repos/...")` / `Path("data/...")` literals. Do
  this as its own small, explicit task — don't let it get silently
  bundled into the Phase 1 language-dispatch generalization task.
- Verified real numbers from a from-scratch Phase 1 run: 864 symbols,
  406 CALLS edges, 42 INSTANTIATES, 65 EXTENDS — matches README exactly.
  This run is the checkpoint-phase1-verified baseline for regression
  comparison going forward.
- **FOUND AND FIXED: `fetch_got_releases.py` had an inconsistent
  `DB_PATH`.** Every other script uses `"data/code_graph.db"` (relative
  to `src/`); this one used `"../data/code_graph.db"`, a leftover from
  before the script was moved into `src/` (its own comment said "where
  this script now lives" but the path wasn't updated to match). Fixed
  to match the rest of the pipeline. This is the third relative-path
  bug hit in a row across three different scripts — reinforces the
  §"CONFIRMED, IMPORTANT FOR GENERALIZATION" note above: consolidate
  every scattered `DB_PATH = "..."` literal into one shared config
  constant. Good, small, low-risk first Antigravity task — purely
  mechanical, directly fixes a real recurring bug class, and is a safe
  warm-up before the riskier Phase 1 language-dispatch generalization.
- Phase 2 progress log (for regression reference): `mine_history.py` —
  1012 commit-rows (httpx), 1093 (got). Discussions indexed: httpx
  "Potential Issue" (431), got "Q&A" (141) — categories chosen as the
  closest analog to `why`-type ground truth per repo, since category
  names differ between the two repos' GitHub Discussion setups. Releases:
  got, 164 fetched via REST (httpx has no separate release-fetch step —
  presumably covered by build_docs_table.py or PR/changelog data instead,
  confirm this before assuming httpx needs an equivalent script).
- **THE DB_PATH SAGA — READ THIS BEFORE TOUCHING ANY PATH CONSTANT.**
  Five separate scripts each hardcode where `code_graph.db` lives,
  and they didn't agree:
  - `build_full_graph.py`, `mine_history.py`, `fetch_pr.py`,
    `summarize_symbols.py` → all use `"data/code_graph.db"`
    (i.e., `src/data/code_graph.db` when run from `src/`, which is how
    every phase script is meant to be run).
  - `fetch_got_releases.py`, `build_docs_table.py` originally had
    `"../data/code_graph.db"` (i.e. `repo-assist/data/`) — these were
    genuine bugs (leftover from before the scripts were moved into
    `src/`, per their own comments) and were correctly fixed to
    `"data/code_graph.db"` to match the majority.
  - `query_tools.py` (used by the actual `repo-assist` CLI/`cli.py`)
    ALSO originally had `"../data/code_graph.db"`. This one was
    initially assumed to be the same bug and nearly "fixed" backwards
    — but real evidence (a manual `cp` of the DB to the repo root
    happened to make the CLI work, which looked like confirmation of
    the wrong theory) caused genuine confusion for several exchanges
    before checking actual `SELECT COUNT(*) FROM summaries` row counts
    against both file locations settled it: `src/data/code_graph.db`
    was the live, growing file (fed by `summarize_symbols.py`,
    confirmed still running and incrementing), and the `../data/`
    copy was a frozen, stale duplicate from the manual `cp`. Fixed
    `query_tools.py` to `"data/code_graph.db"`, deleted the stale
    duplicate. **Lesson: when two files disagree, check actual content
    (row counts, checksums) — never infer which one is "real" from
    file timestamps or from "it happened to work when I did X."**
  - **Net result: `src/data/code_graph.db` is now the single canonical
    DB location, and all five scripts agree on `"data/code_graph.db"`
    relative to `src/`.** CLI_README.md's own text is now slightly
    stale (it describes the OLD `../data/code_graph.db` convention) —
    fix that doc as a trivial follow-up.
  - This whole saga is the strongest possible evidence for
    consolidating every `DB_PATH` literal into one shared constant
    (imported from a single `config.py` or similar) rather than
    six independently-typed string literals. Make this genuinely the
    first Antigravity task — it's small, mechanical, and every hour
    spent NOT doing this risks this exact multi-hour debugging saga
    recurring on a new repo's DB path during generalization.
- Verified: after all fixes, `repo-assist ask httpx "What does the
  Client class do?"` returns a real, correctly-cited answer
  (`Source: CODE#Client`) against the single canonical DB. This is the
  `checkpoint-full-pipeline-verified` baseline — the true regression
  reference point for everything from here forward.
- **Phase 1 generalized to per-file language dispatch (checkpoint-phase1-generalized).**
  All repo-name branching (`"httpx"` vs `"got"` hardcoded in
  extract_symbols.py, resolve_imports.py, resolve_calls.py,
  resolve_calls_typed.py/_ts.py, resolve_inheritance.py,
  build_full_graph.py) replaced with per-file extension dispatch
  (.py -> Python grammar, .ts/.tsx -> TS grammar) plus a `repo` string
  now passed as a parameter instead of hardcoded. Added TSX grammar
  support (wasn't present before). Centralized file exclusion
  (node_modules/, .venv/, dist/, build/, __pycache__/, .git/, test/,
  tests/, dot-dirs, .gitignore respect via `pathspec`, 5000-line cap)
  in one shared location in extract_symbols.py rather than duplicated
  per-script. Added rank_symbols.py: PageRank over CALLS/INSTANTIATES/
  EXTENDS via `networkx`, writes `pagerank_score` onto each symbol
  (graph_schema.py updated to add this field — a real, deliberate,
  narrated schema change, not silent). Verified: httpx/got numbers
  unchanged (864/406/42/65), both CLI regression checks pass (httpx
  Client question correctly cited; got retry-default question
  correctly abstained rather than hallucinating).
- **Phase 2 and 3 generalized to arbitrary repositories with bounded history policy (checkpoint-phase2-3-generalized).**
  - Consolidated bounded history policy constants in `src/config.py` (`MAX_PRS = 300`, `MAX_ISSUES = 300`, `MAX_DISCUSSIONS_PER_CATEGORY = 300`, `MAX_COMMITS_PER_FILE = 500`).
  - Added automatic full git repository cloning (`ensure_repo_cloned` in `build_full_graph.py`) to fetch complete git histories required by PyDriller without shallow truncation.
  - Implemented graceful degradation across all GitHub-dependent tools (`index_discussions.py`, `fetch_issue.py`, `fetch_pr.py`): repos without Discussions or Issues enabled log clear skip messages and exit cleanly rather than crashing the pipeline. Verified live against `psf/requests` where Discussions are disabled.
  - Generalize Phase 2/3 pipelines to dynamically discover all indexed repos from SQLite (`mine_history.py`, `build_docs_table.py`, `summarize_symbols.py`, `compute_churn.py`, `compute_complexity.py`, `compute_centrality.py`, `compute_risk_scores.py`).
  - Empirical verification against `psf/requests`: 22 files, 335 symbols, 75 imports, 128 CALLS edges, 11 INSTANTIATES edges, 37 EXTENDS edges, 2,400 commit-file rows, 18 docs chunks, 335 PageRank scores. Hit bounded commit cap on `src/requests/models.py` (500 commits).
  - **FINDING RESOLVED IN NEXT TASK:** `cli.py` has `choices=["httpx", "got"]` for the `repo` positional argument in `repo-assist ask`, preventing querying arbitrary third repos until the query/router layer is generalized in a future task.
- **Query / CLI layer generalized to arbitrary repositories (checkpoint-query-cli-generalized).**
  - `src/cli.py`: Removed hardcoded `choices=["httpx", "got"]`. Replaced with `get_ingested_repos()` dynamically querying `SELECT DISTINCT repo FROM files`. Non-ingested repo queries return clean, actionable error: `Error: repo 'foo' not found -- ingested repos are: got, httpx, requests`.
  - `src/router.py`: Generalized `search_source_code(repo, query)` from hardcoded per-repo subpaths to dynamic root resolution `repos/<repo>` with standard directory denial filters (`.git`, `node_modules`, `.venv`, `__pycache__`, `tests`).
  - `src/query_tools.py`: Generalized `get_source_snippet()` to resolve `repos/<repo>/<file_path>` for any repo layout. Added `resolve_github_owner_repo()` to infer GitHub `(owner, repo)` dynamically from the git remote origin for live PR/issue fallback. Audited all SQL queries to ensure 100% airtight `repo = ?` filtering and repo isolation.
  - `src/orchestrator.py`: Generalize `PLANNER_INSTRUCTIONS` prompt from two-repo mention to ingested repositories.
  - **All 5 Verifications Passed:**
    1. `repo-assist ask httpx "What does the Client class do?"` -> Correctly answered with `Source: CODE#Client`.
    2. `repo-assist ask got "Why does got default to 2 retries?"` -> Honest abstention without hallucination.
    3. `repo-assist ask requests "What does the Session class do?"` -> Correctly answered with `Source: CODE#Session`.
    4. `repo-assist ask requests "What does the Client class do?"` (Cross-contamination test) -> Correctly abstained, zero bleeding from `httpx` (`requests` has no `Client` class).
    5. `repo-assist ask nonexistent-repo "test"` -> Helpful error message without stack trace.

---

## 7. Sequencing (for the "days" timeline)

1. This contract (done — you're reading it)
2. Generalize Phase 1 against **one new real repo**, chosen now, not
   abstractly (pick something with decent PR/issue history and a mix of
   Python or TS — e.g. a mid-size FastAPI or Express project)
3. Generalize Phase 2/3 with the bounded-history policy + graceful
   degradation + visible failure states
4. **CLI milestone** (prove this before touching FastAPI or React):
   `ingest(url) → READY → ask(question) → answer + citations`, working
   end-to-end on the new repo, with `httpx`/`got` still passing
5. Diagnose the weak eval categories (`topology` 33%, `why` 36%) by
   failure type — retrieval miss, graph miss, synthesis miss, or
   verification false-reject — *before* changing any code for them
6. FastAPI wrapper around the now-proven pipeline, `repo_id` threaded
   everywhere, `BackgroundTasks` for ingestion (documented as a demo-scale
   choice, not a production claim)
7. React frontend — thin, last, lowest technical risk
8. Eval: 2 regression repos (httpx, got) + 2 unseen repos, same six
   question categories, same rubric, report failures honestly
