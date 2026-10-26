"""HTTP API and static web app.

    GET  /api/repos                     list repositories
    POST /api/repos                     ingest {url, ref?} in the background (202)
    GET  /api/repos/{id}                status and stats
    POST /api/repos/{id}/ask            answer {question, mode} as JSON
    POST /api/repos/{id}/ask/stream     same, as server-sent events (tool calls live, then the answer)
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repo_assist import agent, ingest, store
from repo_assist.ingest.git import parse_repo_url

log = logging.getLogger(__name__)

# One ingestion at a time: it is CPU-bound (embedding) and writes a large transaction.
_ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

FRONTEND_DIR = Path(os.environ.get("REPO_ASSIST_FRONTEND_DIR",
                                   Path(__file__).resolve().parents[2] / "frontend" / "dist"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    with store.session() as conn:  # ingests interrupted by a restart will never finish
        conn.execute("""UPDATE repos SET status='failed', stage='Failed', error='Interrupted by a server restart'
                        WHERE status IN ('queued', 'ingesting')""")
    yield
    _ingest_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="repo-assist", version="1.0.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    url: str = Field(description="GitHub URL or owner/name", examples=["encode/httpx"])
    ref: str | None = Field(default=None, description="Commit, tag or branch (default: default branch)")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["agent", "single_shot"] = "agent"


def _repo_or_404(repo_id: str) -> dict:
    with store.session() as conn:
        repo = store.get_repo(conn, repo_id)
    if repo is None:
        raise HTTPException(404, f"Repository {repo_id!r} not found")
    return repo


def _ready_or_409(repo_id: str) -> dict:
    repo = _repo_or_404(repo_id)
    if not repo["stats"]:  # never indexed; a re-ingest keeps serving the previous index
        raise HTTPException(409, f"Repository {repo_id!r} is not indexed yet (status: {repo['status']})")
    return repo


def _run_ingest(repo_id: str, ref: str | None) -> None:
    try:
        ingest.ingest(repo_id, ref=ref)
    except Exception:  # status and error are recorded on the repo row
        log.exception("Ingestion of %s failed", repo_id)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/repos")
def list_repos() -> list[dict]:
    with store.session() as conn:
        return store.list_repos(conn)


@app.post("/api/repos", status_code=202)
def create_repo(req: IngestRequest) -> dict:
    try:
        parse_repo_url(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with store.session() as conn:
        busy = conn.execute("SELECT id FROM repos WHERE status IN ('queued', 'ingesting')").fetchone()
    repo_id = ingest.register_repo(req.url)
    _ingest_pool.submit(_run_ingest, repo_id, req.ref)
    repo = _repo_or_404(repo_id)
    if busy and busy["id"] != repo_id:
        repo["stage"] = f"Queued behind {busy['id']}"
    return repo


@app.get("/api/repos/{repo_id}")
def get_repo(repo_id: str) -> dict:
    return _repo_or_404(repo_id)


@app.post("/api/repos/{repo_id}/ask")
async def ask(repo_id: str, req: AskRequest) -> dict:
    _ready_or_409(repo_id)
    result = await agent.answer(repo_id, req.question, mode=req.mode)
    return result.to_dict()


@app.post("/api/repos/{repo_id}/ask/stream")
async def ask_stream(repo_id: str, req: AskRequest) -> StreamingResponse:
    _ready_or_409(repo_id)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def produce() -> None:
        try:
            await agent.answer(repo_id, req.question, mode=req.mode, on_event=queue.put_nowait)
        except Exception as exc:  # surface model/API failures to the client
            log.exception("Answering failed")
            queue.put_nowait({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            queue.put_nowait({"type": "done"})

    async def events():
        task = asyncio.create_task(produce())
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event["type"] == "done":
                    break
        finally:
            task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = FRONTEND_DIR / path
        if path and candidate.is_file() and FRONTEND_DIR in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "index.html")
