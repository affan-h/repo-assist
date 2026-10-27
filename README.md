# repo-assist

**Agentic RAG over GitHub repositories.** Ask what a codebase does, where something lives, how it works,
or *why* it was designed that way, and get an answer in which every claim cites the exact lines, commits,
issues, PRs, discussions or release notes it came from. If the evidence isn't there, it says so.

```text
$ repo-assist ask httpx "Why was sync support dropped in httpx 0.8?"
  -> search_history(query='sync support dropped 0.8')
     8 results
  -> read_thread(ref='PR#544')
     Drop sync (chars 0-1317 of 1317)
  -> read_thread(ref='issue#572')
     Supporting Sync. Done right. (chars 0-6000 of 7662)

  Sync support was dropped in 0.8 (PR #544) because bridging a sync API onto the async core
  through Sync/Async subclasses produced poorer, less comprehensible code [1][2], and it failed in
  environments that already run an event loop, such as Jupyter [1]. ...

Sources:
  [1] issue#572  https://github.com/encode/httpx/issues/572
  [2] PR#544     https://github.com/encode/httpx/pull/544
```

<!-- RESULTS -->

## How it works

```text
                ┌─────────────────────────── ingest (once per repo, pinned commit) ───────────────────────────┐
 GitHub repo ──▶│ git clone ──▶ tree-sitter: symbol-level code chunks, symbol table, call sites               │
                │           ──▶ Markdown: heading-scoped doc chunks                                          │
                │           ──▶ git log: one chunk per commit message                                        │
                │ GraphQL  ──▶ issues, PRs, discussions (+30 comments), release notes, created before the commit│
                │ bge-small (local ONNX) embeddings ──▶ SQLite: chunks · FTS5 (BM25) · vectors · symbols · calls │
                └───────────────────────────────────────────────────────────────────────────────────────────────┘

 question ──▶ Agent (pydantic-ai, Gemini) ── bounded tool loop ──────────────────────────────────────┐
                 search_code      hybrid BM25 + dense, fused with reciprocal rank fusion             │
                 search_history   same, over commits / issues / PRs / discussions / releases         │
                 find_symbol      definition + call sites from the symbol tables                    │
                 read_file, grep  the pinned checkout (tracked files only)                          │
                 read_thread      full thread, paged                                                │
              every tool records the evidence it showed the model ──▶ evidence ledger ─────────────┘
                                                                         │
              structured Answer {answer, citations[], abstained} ──▶ citation validator
                  rejects any citation not in the ledger (code citations must fall inside lines
                  the model actually saw) and makes the model retry ──▶ answer + GitHub permalinks
```

**Ingestion** (`src/repo_assist/ingest/`) clones a repository at a pinned commit and splits it along its
own structure. Each function and method becomes a chunk with its exact line range, classes get a header
chunk, and Markdown is split by heading with the heading trail kept as context. History comes from `git log`
and the GitHub GraphQL API. Threads created after the indexed commit are dropped, so the history matches the
code snapshot and the evaluation can't benefit from future knowledge.

**Retrieval** (`retrieval.py`) runs BM25 (SQLite FTS5, with `camelCase`/`snake_case` identifiers split into
words) and exact cosine search, then fuses the two rankings with reciprocal rank fusion. Lexical search finds
identifiers, dense search finds paraphrases, and fusion keeps the hits either one ranks highly (measured below).

**The agent** (`agent.py`) is one pydantic-ai agent with six tools and a hard step budget. Search results are
treated as pointers: the instructions push the model to read the code before relying on details, and to look in
history for "why" questions, where only a stated reason counts.

**Grounding is enforced in code, not only in the prompt.** Every tool writes what it returned into an evidence
ledger. The final answer is a typed object, and an output validator checks each citation against the ledger. A
code citation such as `httpx/_client.py:762-765` is accepted only if those lines fall inside a range the model
was actually shown. Anything else is rejected with a list of valid refs, and the model retries. Invented
citations can't reach the user.

**The single-shot baseline** uses the same index, schema and validator, with one hybrid retrieval and one LLM call
and no tools. It exists so the agent's extra cost has to justify itself in the evaluation.

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| **One agent, not several** | An earlier version of this project had a planner, three "specialist" agents and a verifier. Its own evaluation scored it *below* the simpler pipeline (35.7% vs 41.1%), with most specialists wrapping the same functions. One agent with good tools is easier to reason about and measure. | No parallel specialists; a single loop with a step budget. |
| **SQLite for everything** | A repository is a few thousand chunks, so exact cosine search over an in-memory numpy matrix takes about a millisecond, and FTS5 gives BM25 in the same file. One file, transactional re-indexing, no extra service. | Exact search won't scale to millions of chunks; at that point swap in an ANN index (FAISS/HNSW, pgvector, Qdrant) behind `dense_search`. |
| **Local embeddings (bge-small, ONNX)** | Gemini's free tier counts each embedded text as a request (100/min), which makes indexing thousands of chunks take hours. Larger local models were 3–10× slower on the CPU this was built on. Exact identifiers are handled by BM25 and grep, so dense search only needs to capture meaning. | 512-token context; weaker on code than code-specific models. |
| **Tools for exact lookup (grep, find_symbol, read_file)** | Code questions often hinge on an exact identifier, import or literal, which embeddings blur. This replaces the old project's 1,200 lines of call-graph resolvers with name-based call sites that the agent verifies by reading. | Call sites are matched by name, so common names produce false positives (the tool says so). |
| **History as evidence** | "Why" questions are where code search alone fails; the reasons live in issues, PRs, discussions and release notes. Fetching threads before the pinned commit keeps the evaluation honest. | Requires a GitHub token; large repos take minutes to fetch. |
| **pydantic-ai, model as a config string** | Typed outputs, output validators with retry, usage limits, and testable with scripted models. Changing provider is one environment variable. | Defaults tuned for Gemini 3.5 Flash-Lite. |

## Evaluation

`evals/` holds a 56-question set over two pinned repositories, **httpx** (Python) and **got** (TypeScript),
covering *what / how / where* questions, call-chain and import *topology* questions, *why* questions answered
from project history, and *unanswerable* questions where the right behaviour is to say no documented rationale
exists (9 of the 56). Three questions have false premises that the system should correct.

The set started from this project's original question bank, and every reference answer was re-checked
against the pinned source, docs and GitHub threads. **That audit changed 23 of the 56 questions.** Some answers were
wrong (e.g. "`None` disables a mount", a non-existent `source/as-stream/` directory, "six hooks" when there are
seven), some "why" answers cited reasons that appear nowhere in the project, and several "unanswerable" questions
turned out to be answerable from the CHANGELOG, from issue threads or from code comments. Two of those corrections
came *from the system*: in run 1 the agent cited evidence for W10 and Y12 that the manual curation had missed.
Each entry records what was checked, and what changed, in its `verification` field.

Two scripts, both reproducible from the pinned commits:

- `evals/retrieval_eval.py`: BM25 vs dense vs hybrid retrieval, measured against line-level gold evidence.
  No LLM involved.
- `evals/run_eval.py`: the agent vs the single-shot baseline, end to end. A stronger model than the agent grades
  each answer: **correctness** (coverage of key facts from the reference), **abstention** (did it decline
  when it should, and only then), and **groundedness** (each claim checked against *only the evidence the system
  cited*, so a correct answer from the model's prior knowledge still counts as ungrounded). It also records
  whether the gold evidence was retrieved and cited, latency, LLM calls, tool calls and tokens.

<!-- EVAL_DETAILS -->

## Quick start

You need a [Gemini API key](https://aistudio.google.com/apikey) (the free tier works) and, for issue/PR/discussion
history, a GitHub token.

```bash
cp .env.example .env        # add GEMINI_API_KEY and GITHUB_TOKEN
```

### Docker

```bash
docker compose up --build   # http://localhost:8000
# index a repo from the UI, or:
docker compose exec app repo-assist ingest pallets/itsdangerous
```

### Local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm ci && npm run build && cd ..

repo-assist ingest encode/httpx --ref b5addb64f0161ff6bfe94c124ef76f6a1fba5254
repo-assist ask httpx "Where does httpx decode response text using the Content-Type charset?"
repo-assist ask httpx "..." --mode single_shot      # the baseline, for comparison
repo-assist serve                                    # web UI + API on http://127.0.0.1:8000
```

For frontend development, run `npm run dev` in `frontend/` (it proxies `/api` to port 8000).

### API

| Method | Path | |
|---|---|---|
| `GET` | `/api/repos` | List repositories with status and index stats |
| `POST` | `/api/repos` | `{"url": "owner/name", "ref": "optional"}`: index in the background (202) |
| `GET` | `/api/repos/{id}` | Status, current stage, stats |
| `POST` | `/api/repos/{id}/ask` | `{"question": "...", "mode": "agent" \| "single_shot"}`: JSON answer with citations, trace and usage |
| `POST` | `/api/repos/{id}/ask/stream` | Same, as server-sent events: `tool_call`, `tool_result`, `answer`, `done` |

Interactive docs are at `/docs`.

### Reproducing the evaluation

```bash
repo-assist ingest encode/httpx      --ref b5addb64f0161ff6bfe94c124ef76f6a1fba5254
repo-assist ingest sindresorhus/got  --ref 66ca4e025622e1dd71f4e4f13aa6e1b79d61b069
python evals/retrieval_eval.py       # writes evals/results/retrieval.{json,md}
python evals/run_eval.py             # writes evals/results/runs/*.jsonl and summary.{json,md}; resumable
pytest                               # offline unit/integration tests
```

## Project layout

```text
src/repo_assist/
  ingest/        clone, tree-sitter parsing, doc chunking, git + GitHub history, pipeline
  store.py       SQLite schema (chunks, FTS5, vectors, symbols, call sites)
  embeddings.py  local ONNX embeddings with a content-hash cache
  retrieval.py   BM25, dense, RRF fusion, symbol lookup, grep, file reads
  agent.py       tools, evidence ledger, citation validator, agent + single-shot baseline
  api.py         FastAPI (JSON + SSE) and static UI
  cli.py         repo-assist ingest | ask | repos | serve
frontend/        React + Vite UI
evals/           question set, retrieval ablation, end-to-end eval, LLM judge, saved results
tests/           offline tests (fixture git repo, hash embedder, scripted models)
```

## Limitations

- **Indexing is slow on CPU.** Embedding is the bottleneck. A small library takes a few minutes; httpx with its
  full history (about 6,000 chunks) took around 20 minutes on a 2015 dual-core laptop. Re-indexing reuses cached
  embeddings.
- **History coverage is bounded.** Up to 30 comments per thread are fetched. Rationale buried deeper in very long
  threads, or kept outside GitHub (chat, external blog posts), is invisible, and the system will abstain.
- **Call sites are matched by name, not resolved by type.** Good enough for an agent that verifies by reading, but
  it is not a precise call graph.
- **LLM judging is imperfect.** The judge is a stronger model of the same family as the agent, and it can be lenient
  or strict at the margins. Per-question judgements are saved in `evals/results/runs/` for inspection. Model
  outputs are not deterministic, so re-runs vary by a few points.
- **Single-node.** Ingestion runs in one background worker inside the API process. Concurrent multi-user ingestion
  would need a job queue.

## License

[MIT](LICENSE)
