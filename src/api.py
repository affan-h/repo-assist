"""
FastAPI service wrapper for repo-assist (Phase 4 / Step 2).

Provides HTTP endpoints to ingest GitHub repositories asynchronously and ask grounded
questions against the ingested code graph, commit/PR/discussion history, docs, and risk scores.

Architecture & Scope:
- Scope: Demo-scale service for interview evaluation and portfolio demo, NOT production.
- Concurrency: Uses FastAPI BackgroundTasks for ingestion (no Celery/Redis/external queue).
- Storage: Single shared SQLite database (config.DB_PATH) with repo-scoped isolation.
- State Machine: Ingestion transitions through QUEUED -> CLONED -> PARSED -> GRAPH_BUILT
  -> HISTORY_ATTACHED -> RISK_SCORED -> READY (or FAILED with an error message).
- Invariant: Non-READY repositories reject queries with HTTP 409 Conflict per ARCHITECTURE.md §0.
"""

import os
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

from config import DB_PATH, REPOS_DIR
import graph_schema
from graph_schema import save_graph, init_db
from build_full_graph import ensure_repo_cloned
from resolve_imports import build_graph_with_imports
from resolve_calls import resolve_calls_for_file, CallResolutionStats
from resolve_calls_typed import process_python_file, TypedCallStats
from resolve_calls_typed_ts import process_typescript_file, TypedCallStatsTS
from resolve_inheritance import resolve_inheritance_for_file, InheritanceStats
from rank_symbols import compute_pagerank
from extract_symbols import find_source_files
from mine_history import init_commits_table, get_indexed_files, mine_file_history
from build_docs_table import discover_repo_docs, chunk_file
from index_discussions import index_repo_discussions
from compute_churn import init_churn_table, compute_for_repo as compute_churn_for_repo
from compute_complexity import init_complexity_table, compute_for_repo as compute_complexity_for_repo
from compute_centrality import compute_centrality
from compute_risk_scores import init_risk_table, compute_risk
from query_tools import resolve_github_owner_repo
import router
import synthesizer
import cli

app = FastAPI(
    title="repo-assist API",
    description="Grounded codebase intelligence service over structural graphs, history, and docs.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Database & State Helper
# ---------------------------------------------------------------------------

def update_repo_status(repo_id: str, status_val: str, error_message: Optional[str] = None, url: Optional[str] = None):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        if url:
            conn.execute(
                """INSERT INTO repos (repo_id, url, status, status_updated_at, error_message)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(repo_id) DO UPDATE SET
                     url = excluded.url,
                     status = excluded.status,
                     status_updated_at = excluded.status_updated_at,
                     error_message = excluded.error_message""",
                (repo_id, url, status_val, now, error_message),
            )
        else:
            conn.execute(
                """UPDATE repos SET status = ?, status_updated_at = ?, error_message = ? WHERE repo_id = ?""",
                (status_val, now, error_message, repo_id),
            )
        conn.commit()


def get_repo_record(repo_id: str) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM repos WHERE repo_id = ?", (repo_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Ingestion State Machine Worker (Runs in BackgroundTask)
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(repo_id: str, url: str):
    """
    Executes the full Phase 1-3 ingestion pipeline sequentially,
    writing state transitions at each boundary per ARCHITECTURE.md §3.
    """
    print(f"\n[Ingest Worker] Starting ingestion for '{repo_id}' from {url}")
    try:
        # 1. QUEUED -> CLONED
        print(f"[Ingest Worker: {repo_id}] Stage 1: Cloning repository...")
        name, repo_root = ensure_repo_cloned(url, REPOS_DIR)
        update_repo_status(repo_id, "CLONED")

        # 2. CLONED -> PARSED (Phase 1 Step 1: Files, symbols, imports)
        print(f"[Ingest Worker: {repo_id}] Stage 2: Parsing symbols and imports...")
        cg = build_graph_with_imports(repos={repo_id: repo_root})
        save_graph(cg, DB_PATH)
        update_repo_status(repo_id, "PARSED")

        # 3. PARSED -> GRAPH_BUILT (Phase 1 Step 2: Calls, typed calls, inheritance, PageRank)
        print(f"[Ingest Worker: {repo_id}] Stage 3: Resolving call graph and inheritance...")
        call_stats = CallResolutionStats()
        for f in find_source_files(repo_root):
            resolve_calls_for_file(f, repo_root, repo_id, cg, call_stats)

        py_files = find_source_files(repo_root, extensions=(".py",))
        if py_files:
            py_typed_stats = TypedCallStats()
            for py_file in py_files:
                process_python_file(py_file, repo_root, repo_id, cg, py_typed_stats)

        ts_files = find_source_files(repo_root, extensions=(".ts", ".tsx"))
        if ts_files:
            ts_typed_stats = TypedCallStatsTS()
            for ts_file in ts_files:
                process_typescript_file(ts_file, repo_root, repo_id, cg, ts_typed_stats)

        inherit_stats = InheritanceStats()
        for f in find_source_files(repo_root):
            resolve_inheritance_for_file(f, repo_root, repo_id, cg, inherit_stats)

        save_graph(cg, DB_PATH)
        compute_pagerank(DB_PATH, repo=repo_id)
        update_repo_status(repo_id, "GRAPH_BUILT")

        # 4. GRAPH_BUILT -> HISTORY_ATTACHED (Phase 2: Commits, docs, discussions)
        print(f"[Ingest Worker: {repo_id}] Stage 4: Mining commit history and documentation...")
        with sqlite3.connect(DB_PATH) as conn:
            init_commits_table(DB_PATH)
            files = get_indexed_files(DB_PATH, repo_id)
            for f in files:
                mine_file_history(str(repo_root), repo_id, f, conn)

            discovered_docs = discover_repo_docs(repo_id, repo_root)
            for r_name, p, stored_rel, level in discovered_docs:
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    chunks = chunk_file(text, level)
                    for i, (heading, content) in enumerate(chunks):
                        conn.execute(
                            "INSERT INTO docs (repo, file_path, heading, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                            (r_name, stored_rel, heading, i, content),
                        )
                except Exception as doc_err:
                    print(f"  [Docs error on {p}]: {doc_err}")
            conn.commit()

        # Graceful degradation on GitHub Discussions
        try:
            slug = resolve_github_owner_repo(repo_id)
            token = os.environ.get("GITHUB_TOKEN")
            if slug and token:
                owner, gh_repo = slug
                try:
                    index_repo_discussions(owner, gh_repo, token, DB_PATH)
                except Exception as disc_err:
                    print(f"  [Discussions skipped]: {disc_err}")
        except Exception:
            pass

        update_repo_status(repo_id, "HISTORY_ATTACHED")

        # 5. HISTORY_ATTACHED -> RISK_SCORED (Phase 3: Churn, complexity, centrality, risk)
        print(f"[Ingest Worker: {repo_id}] Stage 5: Computing risk metrics...")
        with sqlite3.connect(DB_PATH) as conn:
            init_churn_table(DB_PATH)
            compute_churn_for_repo(conn, repo_id)

            init_complexity_table(DB_PATH)
            compute_complexity_for_repo(conn, repo_id, str(repo_root))

            compute_centrality(DB_PATH, repo=repo_id)

            init_risk_table(DB_PATH)
            compute_risk(conn, repo_id)

        update_repo_status(repo_id, "RISK_SCORED")

        # 6. RISK_SCORED -> READY
        update_repo_status(repo_id, "READY")
        print(f"[Ingest Worker: {repo_id}] Ingestion complete -> Status: READY")

    except Exception as e:
        tb = traceback.format_exc()
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[Ingest Worker: {repo_id}] Ingestion FAILED at stage: {err_msg}\n{tb}")
        update_repo_status(repo_id, "FAILED", error_message=err_msg)


# ---------------------------------------------------------------------------
# API Request / Response Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str = Field(..., description="GitHub repository URL (e.g. 'https://github.com/bottlepy/bottle' or 'owner/repo')")


class IngestResponse(BaseModel):
    repo_id: str
    url: str
    status: str
    status_updated_at: str


class RepoStatusResponse(BaseModel):
    repo_id: str
    url: Optional[str]
    status: str
    status_updated_at: str
    error_message: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., description="Plain-English question about the repository")
    category: Optional[str] = Field(None, description="Optional question category override (what, where, how, why, topology)")
    engine: str = Field("v1", description="Engine to use ('v1' for router/synthesizer, 'v2' for orchestrator)")


class AskResponse(BaseModel):
    repo_id: str
    question: str
    engine: str
    category: Optional[str] = None
    answer: str
    citation_source_id: Optional[str] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    model_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "google:gemini-3.5-flash-lite"
FALLBACK_MODEL = "google:gemini-3.5-flash"


@app.on_event("startup")
def startup_event():
    init_db(DB_PATH)


@app.post("/repos", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_repo(req: IngestRequest, background_tasks: BackgroundTasks):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Repository URL must not be empty.")

    # Derive repo identifier
    if url.startswith("http://") or url.startswith("https://") or url.startswith("git@"):
        repo_id = url.rstrip("/").split("/")[-1].removesuffix(".git")
    elif "/" in url:
        repo_id = url.split("/")[-1]
    else:
        repo_id = url

    # Write initial QUEUED state
    now = datetime.now(timezone.utc).isoformat()
    update_repo_status(repo_id, "QUEUED", url=url)

    # Launch background task
    background_tasks.add_task(run_ingestion_pipeline, repo_id, url)

    return IngestResponse(
        repo_id=repo_id,
        url=url,
        status="QUEUED",
        status_updated_at=now,
    )


@app.get("/repos/{repo_id}/status", response_model=RepoStatusResponse)
def get_repo_status(repo_id: str):
    rec = get_repo_record(repo_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found.")
    return RepoStatusResponse(
        repo_id=rec["repo_id"],
        url=rec.get("url"),
        status=rec["status"],
        status_updated_at=rec["status_updated_at"],
        error_message=rec.get("error_message"),
    )


@app.get("/repos", response_model=List[RepoStatusResponse])
def list_repos():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM repos ORDER BY repo_id").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            result.append(
                RepoStatusResponse(
                    repo_id=d["repo_id"],
                    url=d.get("url"),
                    status=d["status"],
                    status_updated_at=d["status_updated_at"],
                    error_message=d.get("error_message"),
                )
            )
        return result


@app.post("/repos/{repo_id}/ask", response_model=AskResponse)
def ask_repo(repo_id: str, req: AskRequest):
    rec = get_repo_record(repo_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found. Ingest it first via POST /repos.")

    current_status = rec["status"]
    if current_status != "READY":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{repo_id}' is not ready for querying (current status: {current_status}). "
                   f"Ingestion must complete successfully before questions can be answered.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if req.engine == "v2":
        import orchestrator
        v2_res = orchestrator.run_query(repo_id, question)
        return AskResponse(
            repo_id=repo_id,
            question=question,
            engine="v2",
            answer=v2_res.answer,
            abstained=v2_res.abstained,
            abstain_reason=v2_res.abstain_reason,
        )

    # v1 Engine (default)
    cat = req.category or cli.guess_category(question)
    plan_res = router.plan_and_execute(repo_id, cat, question)
    tool_results = plan_res.get("tool_results", {})

    answer_obj, model_used = synthesizer.synthesize_with_fallback(
        PRIMARY_MODEL,
        question,
        repo_id,
        tool_results,
        fallback_model=FALLBACK_MODEL,
    )

    return AskResponse(
        repo_id=repo_id,
        question=question,
        engine="v1",
        category=cat,
        answer=answer_obj.answer,
        citation_source_id=answer_obj.citation_source_id,
        abstained=answer_obj.abstained,
        abstain_reason=answer_obj.abstain_reason,
        model_used=model_used,
    )
