# repo-assist

A codebase intelligence system that builds a genuine, queryable understanding of two real open-source repositories (`encode/httpx` and `sindresorhus/got`) — combining a real structural code graph, git/GitHub history mining, local LLM summarization, risk scoring, and a grounded question-answering layer. Ask it things like *"why does this library retry requests the way it does?"* or *"where does this function enforce that a closed client can't send requests?"* and get an answer backed by real, checkable evidence — or an honest "I don't know" when the evidence isn't there.

**This is not a general "chat with any repo" tool.** It's deliberately scoped to two pinned repositories, built and evaluated end-to-end against them. See [Scope and honesty](#scope-and-honesty) below for exactly what that means.

## Why this exists

Tools like Cursor, Copilot, and naive RAG-over-source-files chatbots guess at codebase understanding from embeddings and context windows. This project takes a different approach: build *real* structure first (an actual parsed symbol/call/import graph, not an LLM's guess), mine *real* history (actual commits, PRs, GitHub Discussions, GitHub Issues, release notes), and only then layer reasoning on top — with a verifier step that checks every claim against real, retrieved evidence rather than trusting a language model's word for it.

The guiding discipline throughout: **verify everything against real evidence before trusting it.** This wasn't a slogan — it caught real bugs repeatedly, in the code and in the project's own hand-written evaluation questions (see below).

## Architecture

```
Phase 1: Structural graph        Phase 2: History & provenance      Phase 3: Risk scoring
  extract_symbols.py               mine_history.py                    compute_churn.py
  resolve_imports.py               fetch_issue.py *                   compute_complexity.py
  resolve_calls.py                 fetch_discussion.py                compute_risk_scores.py
  resolve_calls_typed.py           index_discussions.py
  resolve_calls_typed_ts.py        link_pr_to_discussions.py
  resolve_inheritance.py           summarize_symbols.py
  graph_schema.py                  build_docs_table.py
  build_full_graph.py              fetch_got_releases.py
                    │                        │                              │
                    └────────────────────────┼──────────────────────────────┘
                                              ▼
                                   data/code_graph.db (single SQLite file)
                                              │
                                              ▼
                              Query/Verifier Layer (built this session)
                                query_tools.py   -- data access, all tables
                                router.py        -- rule-based planning per question category
                                synthesizer.py   -- grounded, citation-checked answer generation
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                                ▼
                         cli.py                          grader.py
                    (ask questions from                (Phase 0 evaluation
                     your terminal)                      harness, LLM-as-judge)
```

*`fetch_pr.py` is real, load-bearing, and referenced throughout, but isn't included in this curated file set — see [What's not in this repo](#whats-not-in-this-repo).

Everything lands in **one SQLite database**, `data/code_graph.db` — files, symbols, import/call/inheritance edges, commits, cached PRs/issues/discussions, bulk-indexed discussions, LLM-generated summaries, churn/complexity/risk scores, chunked documentation, and cached release notes. Thirteen-plus tables, one file, built incrementally across three phases before this session even began.

## What each phase actually does

### Phase 1 — Structural graph
Parses every Python file in httpx and every TypeScript file in got with tree-sitter, extracting real classes, functions, and methods (`extract_symbols.py`), resolves real cross-file imports (`resolve_imports.py`), builds a real call/instantiation graph via same-class direct-call detection plus local type inference (`resolve_calls.py`, `resolve_calls_typed.py`/`_ts.py`), and resolves real inheritance edges (`resolve_inheritance.py`). `build_full_graph.py` runs all of this as one pipeline against one shared graph object — a real bug (five separate scripts each silently overwriting each other's partial graph) is why that combined-pipeline script exists at all.

**Result:** 864 real symbols, 406 CALLS edges, 42 INSTANTIATES edges, 65 EXTENDS edges, all independently verified — including a real six-hop call chain traced end to end by hand.

### Phase 2 — History and provenance
Mines full commit history for every indexed file via PyDriller (`mine_history.py`), lazily fetches and caches individual PRs and Issues from GitHub's GraphQL API on demand (`fetch_issue.py`), bulk-indexes entire Discussion categories upfront (`index_discussions.py` — a pivot made after keyword search alone proved unreliable), links PRs to relevant discussions with a real IDF-weighted, title-aware scoring formula that took six real debugging rounds to get right (`link_pr_to_discussions.py`), and generates local, privacy-preserving symbol summaries with a small on-device model via Ollama (`summarize_symbols.py`).

**Real, load-bearing finding from this phase:** commit-to-PR traceability differs sharply by repo — 89% for httpx, only 35% for got — because got's maintainer pushes directly to the branch far more often. This isn't a tooling gap; it's real, and the whole system's "why" answers are expected to look different per repo because of it.

### Phase 3 — Risk / blast-radius scoring
Computes churn (commit frequency) and complexity (branch-construct density) per file, combines them into a percentile-rank risk score (`compute_churn.py`, `compute_complexity.py`, `compute_risk_scores.py`). Validated against real closed `bug`-labeled GitHub issues: high-risk got files averaged 44.7 real bug issues vs. 15.6 for low-risk files — a genuine, non-trivial correlation, not a coincidence.

### Query/Verifier Layer — built entirely this session
This was the one major unbuilt piece of the original plan. It's a three-stage pipeline:

1. **Router** (`router.py`) — rule-based dispatch by question category (`what`/`how`/`where`/`why`/`unanswerable_why`/`topology`), each with its own tool plan built from real, iterative debugging: symbol resolution with a fallback chain (exact match → fuzzy match → full-text source search), ordered call-chain tracing with explicit branch-point transparency (never silently guesses which of several real code branches is "the" answer), reverse-import lookup, and evidence caps tuned against real, observed token-budget failures on free-tier LLM APIs.
2. **Synthesizer** (`synthesizer.py`) — structured, schema-validated answer generation (PydanticAI) with a **closed-list citation mechanism**: the model can only cite a source that was actually retrieved and is explicitly enumerated in the prompt; a post-generation check catches and corrects any citation that doesn't match a real source, rather than trusting the model's self-report. Temperature is forced to 0 — real, observed model non-determinism (identical input producing a fabricated answer 1 run in 3) is a genuine reliability risk this mitigates, though doesn't eliminate.
3. **Grader** (`grader.py`) — the real Phase 0 evaluation harness, wired to the actual pipeline above (its original form assumed a different, LLM-tool-calling architecture and needed real adaptation, documented in the file itself). Uses a different model provider for judging than for generation, to avoid self-grading bias.

## Real results

56 hand-verified questions across both repos, six categories, LLM-as-judge grading (Gemini, a different provider than the Groq/Cerebras models being evaluated):

| Category | Score |
|---|---|
| `unanswerable_why` | **75%** — the safety-critical category: correctly declining to answer rather than fabricating |
| `what` | 56% |
| `how` | 56% |
| `where` | 44% |
| `why` | 36% |
| `topology` | 33% |
| **Overall** | **49%** |

This is an honest number, not a cherry-picked one — it started this session at a genuine 24% baseline and improved through a long sequence of real, root-caused bug fixes (documented below), not prompt tweaking or ground-truth manipulation. Every remaining gap has a specific, understood cause; none are unexplained mysteries.

## Real bugs found and fixed

A representative, non-exhaustive selection:

- **Phase 1:** module-level instantiations (e.g. `DEFAULT_LIMITS = Limits(...)`) were silently invisible to the call resolver — it only ever recognized calls inside a function/method body, never at true module scope. Fixed, graph rebuilt, verified.
- **Phase 1:** property-assigned functions (`got.extend = (...) => {...}`, a real, central public API method) were invisible to the TypeScript symbol extractor — a distinct tree-sitter node shape the original walker never handled. Fixed, graph rebuilt, verified against the real `got.extend()` questions in the eval set.
- **Query layer:** `pr_cache` had exactly one real row before this session — the lazy PR-fetch mechanism existed in Phase 2's code but was never actually wired into anything that would trigger it. Fixed.
- **Query layer:** a flat character-count cap on documentation and release-note content was silently truncating real answers mid-sentence — one confirmed case cut off got's `v9.0.0` retry rationale one paragraph before the actual explanatory sentence. Replaced with relevance-scored excerpt extraction that finds the actually-relevant window regardless of where it sits in a long document.
- **Query layer:** a citation-validation check was rejecting genuinely correct citations over a bracket-formatting mismatch (a model citing `"[CODE#extend]"` instead of `"CODE#extend"`), silently converting correct, well-grounded answers into forced "I don't know" abstentions. Fixed.
- **Query layer:** a systemic resolution gap meant any plain-English question without a backtick-quoted code identifier had no reliable path to real evidence at all — fixed by adding a full-text source-code search fallback, which alone recovered a meaningful cluster of previously-stuck questions in one fix.

## Real errors found in the project's own evaluation questions

The same "verify, don't trust" discipline was turned on the hand-written ground truth itself, and found real mistakes — corrected in `phase0_questions.json`, each with the correction documented inline:

- httpx's mount-matching **does** sort by specificity via a real `URLPattern.priority()` method — the original question claimed plain insertion order with no such mechanism. Confirmed by reading the actual source: the docstring literally says *"The priority allows URLPattern instances to be sortable, so that we can match from most specific to least specific."*
- httpx's response-decoding mechanism is a dedicated `TextDecoder` class, not a bare `.decode()` call as originally claimed.
- got's real hooks documentation lists **seven** hooks (`beforeCache` genuinely exists in the current docs), not six as the original question assumed — likely written against an older version of the library.
- A question about a got v10 "rename" turned out, on reading the real release notes in full, to describe a genuine *structural split* into two separate options with **no stated rationale anywhere in the text** — not a rename with a documented reason as originally claimed. The system's correct behavior on this question is to abstain, which it now does.

## Scope and honesty

**What this is:**
- A local developer tool. Your machine, your Ollama instance, your own free-tier API keys (Groq, Cerebras, GitHub, Gemini). No server, no hosting, no cost beyond what you already pay.
- Scoped to exactly two repositories (`httpx`, `got`), chosen deliberately for contrasting engineering cultures and both being GitHub-native with rich issue/PR/discussion history.
- A real, working, installable CLI (`repo-assist ask <repo> "<question>"`), not just a research script.

**What this is not:**
- Not a general "point it at any repo" tool. Extending it to a new repository means re-running Phases 1–3 against that repo — real, non-trivial work, not a config flag.
- Not free of dependency on things outside this project's control. httpx's GitHub Issues tracker was closed by its maintainer in February 2026 — confirmed directly against the live API — which permanently blocks a handful of `why` questions whose real answer lives in an Issue rather than a PR or Discussion. This is logged, not hidden.
- Not perfect. 49% is a real, defensible number for a system built and evaluated end-to-end in the time available, with every gap traced to a specific cause — not a claim of completeness.

## Setup and usage

```bash
cd src
pip install -e .

export GROQ_API_KEY="..."
export CEREBRAS_API_KEY="..."
export GITHUB_TOKEN="..."

repo-assist ask got "Why does got default to 2 retries?"
repo-assist ask httpx "Where does httpx decode response content according to charset?" --category where
```

Run `repo-assist` from `src/` specifically — every script in this project reads `data/code_graph.db` via a path relative to that directory. See `CLI_README.md` for full CLI details, and run `grader.py` to re-run the full 56-question evaluation yourself.

## What's not in this repo

This is a curated, core-pipeline file set — real, load-bearing files only. Left out deliberately: roughly 25 one-off diagnostic and verification scripts written during development (`diagnose_*.py`, `verify_*.py`, `inspect_*.py`), a near-duplicate early version of the symbol extractor (`step7_extract_symbols.py`, superseded by `extract_symbols.py`), and `fetch_pr.py` (structurally identical to the included `fetch_issue.py` — both are real, lazy GitHub GraphQL fetchers with local SQLite caching, referenced throughout the codebase and required to actually run it — add your own copy following `fetch_issue.py`'s pattern, or reconstruct it from the project history). None of this is hidden — it reflects normal, honest development history, just not included in a repo meant to be read and understood, not archaeology.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `qwen2.5-coder:1.5b` pulled, for Phase 2's summarization step
- Free API keys: [Groq](https://console.groq.com), [Cerebras](https://cloud.cerebras.ai), a [GitHub personal access token](https://github.com/settings/tokens), and [Gemini](https://aistudio.google.com/apikey) (for `grader.py`'s judge)
- `data/code_graph.db` — the built database. Not included in this repository (a real, large file with cached GitHub content); build it yourself by running `build_full_graph.py` followed by the Phase 2/3 scripts in order.
