# repo-assist

**Agentic RAG over GitHub repositories.** Ask what a codebase does, where something lives, how it works,
or *why* it was designed that way. Every claim in the answer cites the exact lines, commits, issues, PRs,
discussions or release notes it came from, and reasons are quoted verbatim from the maintainers. If the
evidence isn't there, it says so.

```text
$ repo-assist ask httpx "Why was sync support dropped in httpx 0.8?"
  -> search_history(query='sync support dropped 0.8')
     8 results
  -> read_thread(ref='issue#572')
     Supporting Sync. Done right. (chars 0-6000 of 7662)

  httpx 0.8 dropped the sync interface because bridging it onto the async core through Sync/Async
  subclasses was producing poorer code, and it failed in environments with top-level async support,
  such as Jupyter [1].

  > "Have Sync and Async variant classes both subclassing Base classes was leading to poorer, less
  >  comprehensible code."  (issue#572)

Sources:
  [1] issue#572  https://github.com/encode/httpx/issues/572
```

It indexes code in **15 languages with symbol extraction** (Python, JavaScript/TypeScript, Go, Java, Kotlin, Rust,
C, C++, C#, Ruby, PHP, Swift, Scala), plus any other language tree-sitter has a grammar for, as plain chunks.
History comes from git and GitHub. It runs on the Gemini free tier with local embeddings and one SQLite file.

## How it works

```text
                ┌──────────────────────────── ingest (once per repo, pinned commit) ────────────────────────────┐
 GitHub repo ──▶│ git clone ──▶ tree-sitter: symbol-level code chunks, symbol table, call sites (15 languages)      │
                │           ──▶ Markdown: heading-scoped doc chunks                                               │
                │           ──▶ git log: one chunk per commit message                                             │
                │ GraphQL  ──▶ issues, PRs, discussions (+30 comments), release notes created before the commit,     │
                │              also split into comment-sized passages for keyword search                           │
                │ bge-small (local ONNX) embeddings ──▶ SQLite: chunks · FTS5 (BM25) · vectors · symbols · calls     │
                └────────────────────────────────────────────────────────────────────────────────────────────────────┘

 question ──▶ Agent (pydantic-ai, Gemini) ── bounded tool loop ─────────────────────────────────────────────┐
                 search_code      hybrid BM25 + dense, fused with reciprocal rank fusion                    │
                 search_history   commits / issues / PRs / discussions / releases, down to the comment      │
                 read_thread      a thread, paged, with the threads it links to                             │
                 code_history     git log -L: lines of code -> commits that changed them -> linked PRs      │
                 find_symbol      definition + call sites                                                   │
                 read_file, grep, list_files   the pinned checkout                                           │
              every tool records what it showed the model ──▶ evidence ledger ─────────────────────────────┘
                                                                         │
              Answer {answer, citations[], rationale_quotes[], abstained} ──▶ grounding checks (in code):
                  · each citation must resolve to the ledger (code: inside lines the model actually read)
                  · a "why" answer must quote the stated reason verbatim from evidence it read
                  failures go back to the model as a retry ──▶ answer + quotes + GitHub permalinks
```

**Ingestion** (`src/repo_assist/ingest/`) clones a repository at a pinned commit and splits it along its own
structure. Each function and method becomes a chunk with its exact line range, classes get a header chunk, and
Markdown is split by heading. Grammars come from `tree-sitter-language-pack`. Languages without a symbol spec are
still indexed, as line windows. History comes from `git log` and the GitHub GraphQL API. Threads created after the
indexed commit are dropped, so history matches the code snapshot. Long threads are also split into comment-sized
passages. These are keyword-indexed only, so they don't slow embedding, and they let a search land on the comment
that contains the answer instead of the start of a 40-comment thread.

**Retrieval** (`retrieval.py`) runs BM25 (SQLite FTS5, with `camelCase`/`snake_case` identifiers split into words)
and exact cosine search, and fuses the two rankings with reciprocal rank fusion.

**The agent** (`agent.py`, `tools.py`) is one pydantic-ai agent with a hard step budget. For "why" questions it has
a path most retrieval systems lack. `code_history` runs `git log -L` on specific lines, so the agent can go from
code, to the commits that changed it, to the PRs and issues those commits reference, which is how a developer
answers "why is this code like this?". If the step budget runs out, the agent answers from the evidence it has
gathered instead of returning nothing.

**Grounding is enforced in code** (`grounding.py`). Every tool writes what it returned into an evidence ledger. The
final answer is a typed object that an output validator checks:

- every citation must resolve to the ledger, and a code citation like `httpx/_client.py:762-765` must lie inside
  lines the model was actually shown;
- an answer to a rationale question must include `rationale_quotes`: the maintainer's own words, copied verbatim
  from evidence the agent read (matching ignores whitespace, case and Markdown).

Failures go back to the model as a retry, with the problems and valid refs listed. The quote rule targets the
main failure found in evaluation: presenting a description of behaviour, or an announcement, as if it were the
reason. A reason that can't be quoted from a source is an inference, and the model is told to abstain.

**The baseline.** The evaluation compares the agent with classic single-shot RAG (one hybrid retrieval, one LLM
call, no tools) that shares the same index, answer schema and grounding checks, so the agent's extra cost has to
justify itself.

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| **One agent, not several** | An earlier version had a planner, three "specialist" agents and a verifier. Its own evaluation scored it *below* the simpler pipeline (35.7% vs 41.1%), and most specialists wrapped the same functions. One agent with good tools is easier to reason about and measure. | No parallel specialists. |
| **Grounding checks in code** | Prompts ask; validators enforce. Citations and rationale quotes are checked against what tools actually returned, so the model can't cite something it never read or put words in a maintainer's mouth. | Extra model retries when the model gets it wrong; "why" answers need a quotable source. |
| **History as first-class evidence** | "Why" questions are where code search alone fails: the reasons live in PRs, issues, release notes and commit messages. Comment-level passages and `code_history` make them reachable. | Needs a GitHub token; large repos take minutes to fetch. |
| **SQLite for everything** | A repository is a few thousand chunks, so exact cosine search over an in-memory numpy matrix takes about a millisecond, and FTS5 gives BM25 in the same file. One file, transactional re-indexing, no extra service. | Exact search won't scale to millions of chunks; then swap an ANN index (FAISS/HNSW, pgvector, Qdrant) in behind `dense_search`. |
| **Local embeddings (bge-small, ONNX)** | Gemini's free tier counts each embedded text as a request (100/min), which makes indexing thousands of chunks take hours. Larger local models were 3–10× slower on the CPU this was built on. Identifiers are handled by BM25 and grep. | 512-token context; weaker on code than code-specific models. |
| **Name-matched call sites, verified by reading** | A per-language type-resolved call graph is a large project of its own. Name matching plus `read_file` gets the agent to the right code. | Common names give false positives (the tool says so). |
| **pydantic-ai, model as a config string** | Typed outputs, output validators with retry, usage limits, and testable with scripted models. Changing provider is one environment variable. | Defaults tuned for Gemini 3.5 Flash-Lite. |

## Evaluation

```bash
python evals/run_eval.py --split test     # agent + baseline on every question; writes evals/results/test/report.md
```

**Questions** (`evals/questions.json`), in two splits:

- **test** (37, held out): written before the system ran on them; these give the headline numbers. 24 are "why"
  questions whose gold answers are the maintainers' own stated reasons: release-note "Why:" lines, PR descriptions,
  CHANGELOG entries and docs. They were mined with `evals/mine_rationale.py`, then checked and written up by hand.
  10 are code questions on a Go repository (go-chi/chi), and 3 are unanswerable, verified to have no documented
  rationale.
- **dev** (56): the original question bank on httpx (Python) and got (TypeScript), covering what/where/how/topology
  and why questions. It was used while building the system. Re-checking it against the pinned source and GitHub
  threads changed 22 of the 56 reference answers, three of them (W10, Y12, Y6) because the system cited evidence the
  manual curation had missed. Each question's `verification` field records what was checked.

**How answers are scored** (`evals/report.py`):

| Metric | How | |
|---|---|---|
| Correctness | answerable: share of the reference's key facts the answer states (0 if it declined or contradicts the reference); unanswerable: 1 if it declined | judge / exact |
| Correct declines, false declines | from the answer's `abstained` flag | exact |
| Gold evidence cited | a citation overlaps the gold evidence (line ranges or thread refs) | exact |
| Why: rationale from a gold source | the verbatim quote comes from a gold source | exact |
| Claims supported | each claim checked against only the evidence the answer cited | judge |
| Latency, LLM calls, tokens | from the run | exact |

Everything that can be checked mechanically is. The LLM judge (`evals/judge.py`) only decides which key facts an
answer states and whether its claims are supported, and it never sees answers that declined. Its reliability is
measured, not assumed: `evals/judge_audit.py` has a second, independently trained free model (Gemma 4, on the same
free API key) re-make every key-fact decision, and reports agreement, Cohen's kappa and every disagreement. Correctness carries a 95% bootstrap interval, and the agent is compared with the
baseline on the same questions (paired bootstrap difference, exact McNemar test). Every LLM call goes through a
record/replay cache (`evals/llm_cache.py`), so a run stopped by the free tier's daily quota resumes the next day
without repeating work.

### Retrieval ablation (no LLM)

Recall@5, i.e. whether a chunk overlapping the gold evidence is in the top 5 (`evals/retrieval_eval.py`):

| Split | Subset | BM25 | Dense | Hybrid (RRF) |
|---|---|---|---|---|
| dev (51) | code & docs | 0.74 | 0.61 | **0.79** |
| dev | history | 0.46 | **0.62** | **0.62** |
| test (34) | code & docs | 0.88 | 0.76 | **0.94** |
| test | history | **1.00** | 0.59 | 0.71 |

Hybrid is the most reliable retriever for code and docs on both splits, which is why it's used. History is mixed. On
dev, embeddings help. On test, keyword search alone found every gold thread, most likely because those questions
were mined from the maintainers' sentences and reuse their exact wording ("prefixUrl", "isStream"), which favours
BM25. The retriever was not retuned after seeing this, because that would be tuning on the test set.

### End-to-end results

<!-- RESULTS -->

## Quick start

You need a [Gemini API key](https://aistudio.google.com/apikey) (the free tier works) and, for issue/PR/discussion
history, a GitHub token (`export GITHUB_TOKEN=$(gh auth token)` works).

```bash
cp .env.example .env        # add GEMINI_API_KEY and GITHUB_TOKEN
```

### Docker

```bash
docker compose up --build   # http://localhost:8000
# index a repo from the UI, or:
docker compose exec app repo-assist ingest go-chi/chi
```

### Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm ci && npm run build && cd ..

repo-assist ingest encode/httpx --ref b5addb64f0161ff6bfe94c124ef76f6a1fba5254
repo-assist ask httpx "Where does httpx decode response text using the Content-Type charset?"
repo-assist serve                                    # web UI + API on http://127.0.0.1:8000
```

For frontend development, run `npm run dev` in `frontend/` (it proxies `/api` to port 8000).

### API

| Method | Path | |
|---|---|---|
| `GET` | `/api/repos` | List repositories with status and index stats |
| `POST` | `/api/repos` | `{"url": "owner/name", "ref": "optional"}`: index in the background (202) |
| `GET` | `/api/repos/{id}` | Status, current stage, stats |
| `POST` | `/api/repos/{id}/ask` | `{"question": "..."}`: JSON answer with citations, quotes, trace and usage |
| `POST` | `/api/repos/{id}/ask/stream` | Same, as server-sent events: `tool_call`, `tool_result`, `answer`, `done` |

Interactive docs are at `/docs`.

### Reproducing the evaluation

```bash
repo-assist ingest encode/httpx      --ref b5addb64f0161ff6bfe94c124ef76f6a1fba5254
repo-assist ingest sindresorhus/got  --ref 66ca4e025622e1dd71f4e4f13aa6e1b79d61b069
repo-assist ingest go-chi/chi        --ref v5.5.0
python evals/retrieval_eval.py --split test      # retrieval ablation, no LLM
python evals/run_eval.py --split test             # agent vs baseline; resumable, cached
python evals/judge_audit.py --split test          # second-judge agreement check
pytest                                            # offline unit/integration tests
```

On the free tier (500 Gemini Flash-Lite requests per day), a question costs about 10 requests (agent, baseline and
judge), so the test split fits in one day. Runs stop cleanly at the daily quota and resume from the cache.

## Project layout

```text
src/repo_assist/
  ingest/        clone, tree-sitter parsing (15 languages), docs, git + GitHub history, passages, pipeline
  store.py       SQLite schema (chunks, passages, FTS5, vectors, symbols, call sites)
  embeddings.py  local ONNX embeddings with a content-hash cache
  retrieval.py   BM25, dense, RRF fusion, passage-aware history search, symbols, grep, file reads
  grounding.py   answer schema; citation and rationale-quote checks
  tools.py       evidence ledger and agent tools
  agent.py       prompts, the agent and the evaluation baseline, run loop
  api.py         FastAPI (JSON + SSE) and static UI
  cli.py         repo-assist ingest | ask | repos | serve
frontend/        React + Vite UI
evals/           questions (dev/test), runner, report, judge + judge audit, retrieval ablation,
                 statistics, LLM cache, rationale miner, results (archive/ holds the v1 system's runs)
tests/           offline tests (fixture git repos, hash embedder, scripted models)
```

## Limitations

- **The judge grades its own model family.** Both agent and judge run on Gemini 3.5 Flash-Lite, because the free tier
  allows only 20 requests/day on stronger models. Deterministic metrics carry the main claims, and every judgement is
  saved for inspection. Manual review of failures found occasional misgrades in both directions.
- **Small evaluation sets.** 37 test questions give wide intervals, so read the bootstrap ranges, not the point
  estimates. Each configuration is run once.
- **Mined questions share wording with their sources**, which flatters keyword retrieval (see the retrieval ablation).
- **Indexing is slow on CPU.** Embedding is the bottleneck: a small library takes a few minutes, and httpx with its full
  history took about 20 minutes on a 2015 dual-core laptop the first time (re-indexing reuses cached embeddings).
- **History coverage is bounded.** Up to 30 comments per thread are fetched. Rationale kept outside GitHub (chat, blog
  posts, external gists) is invisible, and the system should abstain.
- **Call sites are matched by name, not resolved by type.**
- **Single-node.** Ingestion runs in one background worker inside the API process. Concurrent multi-user ingestion
  would need a job queue.

## License

[MIT](LICENSE)
