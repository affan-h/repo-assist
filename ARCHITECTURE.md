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
