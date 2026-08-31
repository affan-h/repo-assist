# repo-assist CLI

Ask real, grounded questions about the `httpx` and `got` codebases from your terminal.

## Setup

1. From the `repo-assist/` root directory (where `pyproject.toml` lives):
   ```bash
   pip install -e .
   ```
   This installs the `repo-assist` command into your current Python environment (works inside your existing venv).

2. Make sure your environment variables are set:
   ```bash
   export GEMINI_API_KEY="..."
   export GITHUB_TOKEN="..."
   ```
   (Note: synthesis, verification, and evaluation use Gemini models via `GEMINI_API_KEY`).

3. Run `repo-assist` from **`repo-assist/src/`** specifically -- the database path is canonically defined in `src/config.py` as `DB_PATH = "data/code_graph.db"` (relative to `src/`), which is the shared single source of truth used by all phase scripts and the CLI. Running it from outside `src/` will fail to resolve the relative database path.

## Usage

```bash
repo-assist ask got "Why does got default to 2 retries?"
repo-assist ask httpx "Where does httpx decode response content according to charset?"
repo-assist ask httpx "What does the Limits class control?" --category what
repo-assist ask got "Trace the call chain from got(url) to the Node.js http.request call" --category topology
```

Add `--verbose` / `-v` to see routing diagnostics (which category was used, how many tool calls ran, resolution confidence, and whether the fallback model was used):

```bash
repo-assist ask got "Why does got default to 2 retries?" --verbose
```

## Scope

This tool is scoped to the two repos this whole project was built and evaluated against: `httpx` and `got`. It is not (yet) a general "point at any repo" tool -- extending it to arbitrary repos would require re-running Phase 1-3 of the pipeline (structural graph extraction, history mining, summarization) against the new repo first. See `project_context.md` for why this scope was chosen and what it would take to widen it.

## Known limitations (see project_context.md and results*.json for the full real evaluation)

- Real, current eval score: ~49% on the 56-question Phase 0 benchmark, with the safety-critical "correctly declines to answer" behavior at 75%.
- `why` questions citing httpx GitHub Issues (not PRs/Discussions) cannot be answered -- httpx's Issues tracker was closed by the maintainer in Feb 2026; this is a confirmed, external, permanent constraint, not a bug in this tool.
- Multi-hop call-chain tracing (topology questions asking for an exact ordered path) is the weakest category -- the underlying mechanism works but doesn't always find the precise canonical path a human researcher would.
