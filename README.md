# repo-assist

**Agentic RAG over GitHub repositories.** Ask what a codebase does, where something lives, how it works,
or *why* it was designed that way. Every claim in the answer cites the exact lines, commits, issues, PRs,
discussions or release notes it came from, and reasons are quoted verbatim from the maintainers. If the
evidence isn't there, it says so.

```text
$ repo-assist ask httpx "Why was sync support dropped in httpx 0.8?"
  -> search_history(query='sync support httpx 0.8')
     8 results
  -> read_thread(ref='PR#544')
     Drop sync (chars 0-1317 of 1317)
  -> read_thread(ref='issue#522')
     Discussion about dropping sync support. (chars 0-6000 of 30000)

  16.5s, 5 LLM calls, 3 tool calls, 21446 tokens

  In httpx 0.8, sync support was dropped so that the project could focus exclusively on providing an
  async client and simplify the codebase [issue#587, PR#544].

  According to the maintainers, maintaining both sync and async support resulted in awkward subclasses
  and abstractions (such as `BaseRequest`, `BaseResponse`, `AsyncRequest`, `AsyncResponse`, `Client`,
  `AsyncClient`, and separate dispatchers), complex bridging code, and differing type annotations [...]

  > "Subclassing in order to provide sync+async, leaading to BaseRequest, BaseResponse, AsyncRequest,
  >  AsyncResponse, Request, Response, Client, AsyncClient, Dispatcher, AsyncDispatcher just *doesn't*
  >  feel legible when getting to grips with any particular part of the codebase. [...]"  (PR#544)

Sources:
  [1] PR#544     https://github.com/encode/httpx/pull/544
  [2] issue#522  https://github.com/encode/httpx/issues/522
  [3] issue#587  https://github.com/encode/httpx/issues/587
```

(Real output, line-wrapped and shortened where marked `[...]`; the quote's typo is the maintainer's.)

It indexes code in **15 languages with symbol extraction** (Python, JavaScript/TypeScript, Go, Java, Kotlin, Rust,
C, C++, C#, Ruby, PHP, Swift, Scala), plus any other language tree-sitter has a grammar for, as plain chunks.
History comes from git and GitHub. It runs on the Gemini free tier with local embeddings and one SQLite file.

On 37 held-out questions it scores **75% vs 56%** for single-shot RAG over the same index (+19 points, 95% CI
+3 to +35), at about 5× the LLM calls. A second, independent judge model changes both numbers by about 4 points
but not the gap. [Results and failure analysis](#end-to-end-results).

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

- **test** (37, held out): written before the system ran on them; these give the headline numbers. 25 are "why"
  questions whose gold answers are the maintainers' own stated reasons: release-note "Why:" lines, PR descriptions,
  CHANGELOG entries, docs and issue comments. They were mined with `evals/mine_rationale.py`, then checked and
  written up by hand. 10 are code questions on a Go repository (go-chi/chi), and 2 are unanswerable, verified to
  have no documented rationale. One label was corrected after the run: "Why is the router called chi?" was written
  as unanswerable, but the agent answered it from a maintainer comment in chi issue #199 that the manual search had
  missed. It is now an answerable question (cY7, original label kept in its `verification` field). Before the
  correction the agent's correctness was 73% and the baseline's 59%.
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
| test (35) | code & docs | 0.88 | 0.76 | **0.94** |
| test | history | **0.94** | 0.56 | 0.67 |

Hybrid is the most reliable retriever for code and docs on both splits, which is why it's used. History is mixed. On
dev, embeddings help. On test, keyword search alone found 17 of the 18 gold threads, most likely because those
questions were mined from the maintainers' sentences and reuse their exact wording ("prefixUrl", "isStream"), which
favours BM25. The one test question that wasn't mined ("Why is the router called chi?") is missed by all three
retrievers when the raw question is the query. The agent still found it, because it writes its own search queries.
The retriever was not retuned after seeing this, because that would be tuning on the test set.

### End-to-end results

Test split, 37 held-out questions, one run per system on Gemini 3.5 Flash-Lite (full report:
[`evals/results/test/report.md`](evals/results/test/report.md); every answer, trace and judgement is in
`answers.jsonl` next to it). Brackets are 95% bootstrap intervals.

| Metric | Agent | Single-shot RAG baseline |
|---|---|---|
| **Correctness** | **75%** [62–88] | 56% [41–72] |
| Correctness, "why" questions (25) | 69% | 55% |
| Correctness, code questions on chi (10) | 87% | 50% |
| Correct declines (2 unanswerable) | 2 / 2 | 2 / 2 |
| False declines (35 answerable) | 17% | 43% |
| Claims supported by the cited evidence | 98% | 95% |
| LLM calls / tokens per question | 5.5 / 26k | 1.1 / 3.8k |
| Latency p50 / p95 | 7.6s / 19.7s | 2.0s / 5.8s |

**Agent vs baseline on the same questions:** +19 points [+3, +35]. The agent alone solved 9 questions and the
baseline alone solved 2 (exact McNemar p = 0.065). The agent is better, but with 37 questions the size of the gap is
uncertain. It costs about 5× the LLM calls and 7× the tokens. Most of the gain comes from questions the baseline
declines because one retrieval pass didn't surface the evidence. The agent's extra searches, `read_thread` and
`code_history` find it.

**Is the judge trustworthy?** `evals/judge_audit.py` had Gemma 4 re-make all 116 key-fact decisions:
[91% agreement, Cohen's kappa 0.51](evals/results/test/judge_audit.md). In 9 of the 11 disagreements the main judge
said "covered" and Gemma said "not covered", and the two spot-checked cases (cC7, hY7) side with Gemma, so the main
judge is somewhat lenient. Rescoring with Gemma's verdicts gives agent 71% and baseline 52%. The absolute numbers
move about 4 points, but the gap between the systems doesn't change.

**Where the agent fails** (8 questions scored 0, all read by hand):

- **Release notes it never opened (4).** Got's v10.0.0 release notes give the reason for gY2, gY3, gY5 and gY6. The
  agent found the PRs and commits that *made* each change. They describe what changed, not why, so it declined
  (gY3, gY6), ran out of steps (gY5), or quoted an older issue thread instead (gY2). History search ranks a long
  release note below the focused PR threads, and the agent has no way to ask for "the notes of the release that
  shipped this".
- **The source was one call away (2).** For cY1 it searched for "514" but never called `read_thread("PR#514")`. For
  hY4 it never searched the CHANGELOG, which the baseline's single retrieval did find. These are the baseline's only
  two wins.
- **Wrong reason or missing facts (2):** hY6 and cC2.

These were not fixed. They were found on the test split, and tuning against them would make the numbers above
meaningless. They are the next things to work on, measured on dev.

**Two metrics that look bad for the agent but aren't.** *Gold evidence cited* is 57% for both systems, and
*rationale quoted from a gold source* is 48% for the agent vs 87% for the baseline. The gold lists name the one
source a question was mined from. The agent often answers correctly from another real source: a release note
instead of the PR, a commit message, or the docs. cY3, cY4, gY7, gY8 and gY11 all scored 100% without citing the
gold source. The baseline only quotes when its single retrieval returned the gold source (15 answers) and otherwise
declines. The agent quotes in 21 answers. These are exact checks of *which* source was used, not of correctness.

The dev split hasn't been re-run on this version, to stay within the free quota. The v1 system's runs are kept
in `evals/results/archive/`.

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
  allows only 20 requests/day on stronger models. A second model (Gemma 4) agrees on 91% of key-fact decisions
  (kappa 0.51) and finds the main judge slightly lenient. That moves correctness about 4 points for both systems but
  not the gap between them. Every judgement is saved for inspection.
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
