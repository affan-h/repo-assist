# repo-assist

A codebase intelligence system that answers structural, historical, and architectural questions about codebases with grounded, verifiable citations. Instead of generic LLM search, it builds a local SQLite code graph (AST symbols, call edges, inheritance), mines git commit churn and GitHub discussions, indexes documentation, and synthesizes answers that cite concrete sources (`CODE#<symbol>`, `PR#<num>`, `DOCS#<section>`) or honestly abstain when evidence is missing.

Includes pre-indexed data for five open-source repositories (`httpx`, `got`, `requests`, `itsdangerous`, `bottle`), a React web interface, a REST API, and a terminal CLI.

---

## Quick Start with Docker (Recommended)

Docker Compose starts both the backend API and frontend UI with all pre-seeded repositories ready to query immediately.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Ollama](https://ollama.com) installed and running on your host machine (`ollama serve`)
  *(Symbol summarization during new repo ingestion runs a local 1.5B model on your host CPU)*
- A [Gemini API key](https://aistudio.google.com/apikey) (used for grounded synthesis and verification)
- A [GitHub Personal Access Token](https://github.com/settings/tokens) (classic token with standard public repo access, used for live ingestion)

### Run in 3 steps:

```bash
# 1. Pull the local summarization model on your host (one-time, ~1GB)
ollama pull qwen2.5-coder:1.5b

# 2. Configure credentials
cp .env.example .env
# Edit .env and fill in GEMINI_API_KEY and GITHUB_TOKEN

# 3. Start the containers
docker compose up --build
```

Once started:
- **Web UI**: [http://localhost:5173](http://localhost:5173) — select any repository to ask questions immediately, or paste a public GitHub URL to ingest a new repository live.
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop and completely remove persisted data volumes:
```bash
docker compose down -v
```

---

## Manual Setup (Without Docker)

If you prefer running directly on your host machine:

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- Git
- Ollama with `qwen2.5-coder:1.5b`

### 1. Backend

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies and repo-assist CLI
pip install -e .

# Set environment variables
cp .env.example .env
export GEMINI_API_KEY="your-gemini-key"
export GITHUB_TOKEN="your-github-token"

# Run the API server (from src/ so database paths resolve properly)
cd src
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) to access the interface.

---

## Usage Examples

### 1. CLI Usage

You can query any ingested repository directly from your terminal using `repo-assist ask`:

```bash
# General questions (auto-routed)
repo-assist ask httpx "What does the Client class do?"
repo-assist ask got "Why does got default to 2 retries?"
repo-assist ask httpx "Where does httpx decode response content according to charset?"

# Force a specific query category (what, how, where, why, topology)
repo-assist ask httpx "What does the Limits class control?" --category what
repo-assist ask got "Trace the call chain from got(url) to the Node.js http.request call" --category topology

# Select query engine (v1 default vs v2 multi-agent)
repo-assist ask httpx "What does AsyncClient do?" --engine v1
repo-assist ask httpx "What does AsyncClient do?" --engine v2

# Verbose mode: inspect routing decisions, retrieved sources, and citations
repo-assist ask got "Why does got default to 2 retries?" --verbose
```

> **Note on CLI Directory:** Run `repo-assist` from inside `src/` (or set `DATA_DIR` and `REPOS_DIR` in your environment) so the CLI resolves `data/code_graph.db` properly.

### 2. REST API Usage

The backend exposes FastAPI endpoints for programmatic access:

#### Query a repository
```bash
curl -X POST http://localhost:8000/repos/httpx/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Client class do?", "engine": "v1"}'
```
Response:
```json
{
  "repo_id": "httpx",
  "question": "What does the Client class do?",
  "engine": "v1",
  "category": "what",
  "answer": "The Client class represents an HTTP client in HTTPX that handles connection pooling, HTTP/2 support, redirects...",
  "citation_source_id": "CODE#Client",
  "abstained": false,
  "abstain_reason": null,
  "model_used": "google:gemini-3.5-flash-lite"
}
```

#### List all indexed repositories
```bash
curl http://localhost:8000/repos
```

#### Ingest a new repository
```bash
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/pallets/click.git"}'
```

#### Poll ingestion progress
```bash
curl http://localhost:8000/repos/click/status
```

---

## Architecture Overview

`repo-assist` combines deterministic static analysis with targeted multi-agent retrieval:

```text
               ┌──────────────────────────────────────────────┐
               │    User Query (Web UI, CLI, or REST API)    │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │              Query Orchestrator              │
               │  • v1: Deterministic category router         │
               │  • v2: Multi-agent planner + ONNX embeddings │
               └──────────────────────┬───────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     Code & History Graph      │               │     Semantic & Doc Indices    │
│ • Tree-sitter AST symbols     │               │ • Local BGE ONNX embeddings   │
│ • CALLS & EXTENDS hierarchies │               │ • Markdown doc chunks (BM25)  │
│ • Git commit churn & authors  │               │ • GitHub PR discussions &     │
│ • Local Ollama symbol summary │               │   release notes               │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └──────────────────────┬────────────────────────┘
                                       ▼
               ┌──────────────────────────────────────────────┐
               │         Grounded Synthesizer & Verifier       │
               │ • Closed-universe citations (CODE#, PR#)     │
               │ • Verifier checks draft against evidence     │
               │ • Explicit abstention on missing proof       │
               └──────────────────────────────────────────────┘
```

### Ingestion Pipeline
When a repository is added, it progresses through an 8-stage pipeline:
1. `QUEUED` → Scheduled in background worker.
2. `CLONED` → Git shallow clone to local storage.
3. `PARSED` → Tree-sitter AST extraction for symbols and imports.
4. `GRAPH_BUILT` → Call-graph edges, typed calls, inheritance hierarchies, and PageRank centrality.
5. `HISTORY_ATTACHED` → Git commit churn mined via PyDriller, PR metadata, and documentation chunks.
6. `SUMMARIZED` → Symbol-level purpose and delegation summaries generated via local Ollama.
7. `INDEXED` → Dense vector embeddings generated via ONNX runtime (`BAAI/bge-small-en-v1.5`).
8. `READY` → Open for queries.

---

## Known Limitations

- **Why-question coverage & closed issue trackers:** Answering "why" a design choice was made requires that the maintainers recorded the rationale in commit messages, PR descriptions, or discussions. If no rationale was ever documented, or if an issue tracker was closed (for instance, HTTPX closed its GitHub Issues tracker in early 2026), the system will honestly abstain rather than confabulate a justification.
- **Engine choice (`v1` vs `v2`):**
  - **`v1` (default)** uses a deterministic router that selects specialized retrieval tools based on query intent. It is fast, lightweight, and excels at precision and safe abstention.
  - **`v2`** adds dense vector embeddings and a multi-agent planner with specialist agents (`structural`, `history`, `docs`). While `v2` is better at broad conceptual searches across large documentation, its strict verification filters can cause it to abstain more conservatively on nuanced historical questions.
- **Local LLM ingestion speed:** Symbol summarization during new repository ingestion relies on `qwen2.5-coder:1.5b` running on the host via Ollama. On CPU without GPU acceleration, ingesting a repository with 20-50 files typically takes 1 to 4 minutes.
- **Multi-hop call topology boundaries:** Call chain tracing resolves static and typed call edges accurately. Deep dynamic chains involving dynamic metaprogramming or complex runtime monkey-patching cannot be fully resolved statically; in those cases, the system returns the verified segment of the chain rather than guessing the remainder.
- **Background task scale:** Ingestion runs as an in-process FastAPI `BackgroundTask` backed by SQLite in WAL mode. This is designed for single-node developer use; deploying as a high-concurrency shared service would require a distributed task queue (e.g. Celery / Redis).

---

## License

This project is licensed under the [MIT License](LICENSE).
