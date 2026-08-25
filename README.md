# repo-assist

A codebase intelligence system that builds a genuine, queryable understanding of a codebase — going beyond "chat with your repo" RAG wrappers. It combines a real, parsed structural graph, mined commit/PR/discussion history, local doc/release-note indexing, and local LLM summarization, then answers questions through a citation-grounded synthesis layer that abstains honestly when nothing is actually documented.

Built and evaluated end-to-end against two pinned repositories — [`encode/httpx`](https://github.com/encode/httpx) (Python) and [`sindresorhus/got`](https://github.com/sindresorhus/got) (TypeScript) — on a 2015 MacBook Air (Intel i5, 8GB RAM, no GPU), using only free-tier APIs and local tooling. Zero budget was a hard constraint throughout.

The guiding discipline across the whole project: **verify every real number against real output before trusting it.** Nearly every component described below had at least one real bug found this way — that history is documented in-line rather than cleaned up after the fact, because it's genuinely informative about how the system reached its current state.

---

## What's actually in this repo

- **A local developer CLI** (`repo-assist ask <repo> "<question>"`) — the real way anyone uses this day to day. Full usage in [`CLI_README.md`](CLI_README.md).
- **Two full query engines**, `v1` and `v2` (see [Architecture](#architecture)), selectable via `--engine`.
- **A real, offline data pipeline** (`src/`) that builds the structural graph, mines history, indexes docs, generates local summaries, and scores risk — see [Building the database from scratch](#building-the-database-from-scratch).
- **A real evaluation harness** (`grader.py`) with a 56-question, hand-verified benchmark and an automated LLM-as-judge, used to make an honest, gated decision about which engine is the default.

---

## Quick start (you already have a populated database)

```bash
git clone https://github.com/affan-h/repo-assist.git
cd repo-assist/src
pip install -e .
export GEMINI_API_KEY="your-key-here"
repo-assist ask httpx "What does the Limits class control?"
```

If you don't have `data/code_graph.db` yet, see [Building the database from scratch](#building-the-database-from-scratch) — it's a real, one-time pipeline, not something that ships pre-built in this repo (the repo does not commit the database; it's built from live GitHub data and would go stale the moment it was checked in).

---

## Architecture

### v1 — rule-based router (the current default, and the reason why)

`router.py` dispatches each question to one of six fixed categories (`what` / `how` / `where` / `why` / `unanswerable_why` / `topology`), each of which calls a specific, tested combination of real tool functions from `query_tools.py`: symbol resolution, source snippets, call-chain tracing over the real structural graph, commit/PR/issue/discussion mining, doc search, release-note search. **Zero LLM involvement in retrieval** — only the final `synthesizer.py` step uses a model, and it's given a closed, numbered list of real evidence sources to cite from, so a hallucinated citation is structurally caught rather than merely discouraged by prompting.

This is why v1 is cheap, fast, and deterministic in what it retrieves.

### v2 — multi-agent orchestrator with semantic retrieval

Adds two genuinely new things on top of v1, without touching v1's code:

1. **Semantic/vector retrieval** (`embeddings.py`, `build_embeddings_index.py`) — every symbol summary, doc chunk, and cached PR body is embedded locally (Google's EmbeddingGemma-300M, run via raw ONNX Runtime — see [Known limitations](#known-limitations) for why not `sentence-transformers`) and searchable by cosine similarity, finding conceptually related content with no literal keyword overlap, which v1's IDF-based search structurally can't do.
2. **A real, bounded multi-agent orchestrator** (`orchestrator.py`, `agents/`) — a single-shot planner picks which of four specialists (`structural`, `history`, `docs_semantic`, `docs_keyword`) to invoke per question, from a **closed enum**, not free text; they run sequentially; results feed the **same, unmodified** v1 synthesizer; a verifier agent (forced onto a different model than synthesis, per a real diversity requirement) checks the draft for unsupported claims and can trigger exactly one retry, routed differently depending on whether the gap looks like a missing specialist or a synthesis error over evidence that was already present.

Full real design reasoning — including three rounds of adversarial review that caught and corrected several claims before implementation — lives in `repo-assist-v2-plan.md`.

### v1 vs v2 — real evaluation result

Both engines were run through the same real, 56-question, hand-verified benchmark, judged automatically by a separate LLM (to avoid self-grading bias), with a hard, pre-registered gate for whether v2 would become the new default:

| | v1 | v2 |
|---|---|---|
| Overall (56 questions) | **41.1%** | 35.7% |
| `unanswerable_why` (safety-critical: correctly declining when nothing is documented) | 87.5% | 37.5% (required floor: 65%) |

**v2 did not pass its own gate.** It genuinely does better on some categories (`what`: 80% vs 50%, `where`: 56% vs 33% — the semantic retrieval and multi-agent structure are real and working) but regressed hard on `why`/`unanswerable_why`. The diagnosed cause: a mid-project fix that tightened the synthesizer's citation-sufficiency rule (a cited source must *state a reason*, not just describe behavior — itself a real fix for a real hallucination bug found during evaluation) overcorrected and made v2 abstain on `why` questions v1 correctly answers.

**v1 remains the default.** `--engine v2` is real, fully functional, and left available for comparison and further work — it simply hasn't earned default status by this project's own stated criteria yet. See [Known limitations](#known-limitations) for what a fix would need.

---

## Building the database from scratch

This is the real, one-time offline pipeline. Run once per machine, from `src/`:

```bash
export GITHUB_TOKEN="..."          # personal access token, for PR/issue/discussion fetching
export GEMINI_API_KEY="..."        # for v1/v2 synthesis and verification
```

**Phase 1 — structural graph.** Parses both repos with tree-sitter, resolves symbols/imports/call edges, builds the graph, saves to `data/code_graph.db`. Produces 800+ real symbols, real CALLS/INSTANTIATES/EXTENDS edges, real cross-file import edges.

**Phase 2 — history and provenance.** Mines full local git history via PyDriller, extracts and links PR numbers from commit messages, lazily fetches PRs/issues from GitHub with local caching, bulk-indexes GitHub Discussions, and links PRs to the discussions that actually explain them. Then generates a local, private symbol-relationship summary for every real symbol using a small local Ollama model (`qwen2.5-coder:1.5b` — chosen specifically because it fits comfortably in 8GB RAM with no GPU), with post-generation verification against the real known-symbol table rather than trusting the model's relationship claims blind.

**Phase 3 — risk scoring.** Computes churn (real commit counts per file) and complexity (real branch-construct counts per symbol, excluding classes to avoid a known aggregation bug), combines them via percentile rank into a risk score, validated against real closed `bug`-labeled GitHub issues per file.

The full real build log for all three phases — every bug found, every design decision made and why, every number verified against actual output — is in `project_context.md`. It's meant to be pasted as-is into a fresh session if you ever need to resume or extend this pipeline; it's written to be sufficient context on its own.

**Then, for v2's additions:**

```bash
python3 build_embeddings_index.py    # one-time; downloads a ~1.2GB local model on first run
python3 compute_centrality.py        # optional; real PageRank over the structural graph
```

---

## Using the CLI

Full command reference, flags, and examples: **[`CLI_README.md`](CLI_README.md)**.

---

## Evaluating changes

```bash
python3 grader.py --questions phase0_questions.json --output results.json --engine v1
python3 grader.py --questions phase0_questions.json --output results.json --engine both
```

`--engine both` runs the full 56-question benchmark through both engines and automatically evaluates the regression gate described above. Free-tier LLM quotas are real and tight enough that a full run can be interrupted partway through — `--retry-failed` resumes without re-spending quota on questions that already scored:

```bash
python3 grader.py --questions phase0_questions.json --output results.json \
  --retry-failed results.json --engine v2 --override-model google:gemini-3.7-flash
```

`--override-model` swaps the live model constants for that run only, useful when a specific model's daily quota is exhausted and you want to finish on a different one with fresh quota — it never edits source files.

---

## Known limitations

Stated plainly rather than buried:

- **v2's `why`/`unanswerable_why` regression is diagnosed but not fixed.** The synthesizer's citation-sufficiency rule likely needs two separate thresholds — a stricter one for catching `unanswerable_why` hallucinations, a looser one for genuine `why` questions where real evidence exists but doesn't rise to "explicitly states a reason." Right now both categories share one rule, and tightening it to fix one broke the other.
- **v2's daily free-tier LLM quota is real and tight** — Gemini's free tier caps range roughly 20–500 requests/day depending on the specific model, and a full dual-engine benchmark run can exhaust it partway through. `--retry-failed` / `--override-model` exist specifically to recover from this.
- **`topology` questions asking for an exact, ordered multi-hop call chain** are the weakest category in both engines. The underlying call-graph mechanism is real and correct as far as it goes, but some legitimate chains require control-flow-sensitive return-type inference that was deliberately scoped out of Phase 1 (documented in `project_context.md`) — the system honestly reports "I don't know the full chain" past that real boundary rather than guessing, which the benchmark's strict all-or-nothing grading scores as 0 even when the partial trace it did produce is correct.
- **This tool is scoped to `httpx` and `got` only.** Extending it to an arbitrary repo requires re-running the full Phase 1–3 pipeline above against that repo first — real, non-trivial work, not a config flag. The v2 schema additions (`embeddings`, `centrality_scores`) are already keyed by `repo` and need no migration if this is pursued later.
- **v2's embeddings layer bypasses `sentence-transformers` entirely**, calling `onnxruntime` directly instead, due to a real, confirmed version incompatibility between current `optimum` (2.x) and what `sentence-transformers`'s own ONNX backend code still expects internally (pre-2.x import paths) as of when this was built. If a future `sentence-transformers` release fixes this, `embeddings.py`'s docstring has the full account of what to check before reverting.
- **Cerebras and Groq**, the LLM providers this project originally scoped around (see `repo-assist-v2-plan.md` §3.3/§3.8), stopped being usable partway through this project (Cerebras began requiring payment; Groq's real availability wasn't independently confirmed afterward). Every model string in this codebase now points at Gemini as a result — a real, documented departure from the original plan, not a silent one. `verifier_agent.py`'s docstring covers the resulting, weaker same-vendor provider-diversity guarantee this created for v2's verification step.

## Project history

Built across two long sessions, each producing a self-contained handoff document meant to be pasted as-is into a fresh session with zero prior context:

- **`project_context.md`** — v1's complete build: structural graph, history mining, local summarization, risk scoring. Documents every real bug found and the discipline of tracing each one to its actual root cause before fixing it, rather than accepting a plausible-looking fix.
- **`repo-assist-v2-plan.md`** — v2's design, reviewed adversarially across three passes before implementation began. Documents every correction made during that review (a dropped framework-popularity claim that didn't survive scrutiny, a corrected characterization of the verifier's real independence, a fixed ambiguous regression-gate definition) as real material, not something to quietly omit from the public-facing version.

The same discipline extended into implementation and evaluation: real bugs were found and fixed by tracing live output rather than guessing (evidence truncation cutting real answers short, a symbol-resolution gap, a semantic-search results key that was silently invisible to the synthesizer since day one, a docs-table coverage gap that made a real, correct answer structurally unfindable, a judge-prompt scope bug that misapplied one category's grading rule to another), and when the final, full regression-gate run came back honest but short of the bar, that result was kept rather than adjusted.
