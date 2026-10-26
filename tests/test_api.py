import json

import pytest
from fastapi.testclient import TestClient

from repo_assist import api, store


@pytest.fixture
def client(data_dir):
    with TestClient(api.app) as c:
        yield c


def test_list_and_get_repo(ingested_repo, client):
    repos = client.get("/api/repos").json()
    assert [r["id"] for r in repos] == [ingested_repo]
    assert client.get(f"/api/repos/{ingested_repo}").json()["status"] == "ready"
    assert client.get("/api/repos/nope").status_code == 404


def test_create_repo_rejects_bad_urls(client):
    assert client.post("/api/repos", json={"url": "not a repo"}).status_code == 400


def test_ask_requires_indexed_repo(client):
    with store.session() as conn:
        store.upsert_repo(conn, "pending", "acme", "pending", "https://github.com/acme/pending")
    assert client.post("/api/repos/pending/ask", json={"question": "what is this?"}).status_code == 409
    assert client.post("/api/repos/missing/ask", json={"question": "what is this?"}).status_code == 404


def test_startup_marks_interrupted_ingests_failed(data_dir):
    with store.session() as conn:
        store.upsert_repo(conn, "stuck", "acme", "stuck", "https://github.com/acme/stuck")
        store.update_repo(conn, "stuck", status="ingesting")
    with TestClient(api.app) as c:
        repo = c.get("/api/repos/stuck").json()
    assert repo["status"] == "failed" and "restart" in repo["error"]


def test_stream_emits_tool_events_then_answer(ingested_repo, client, monkeypatch):
    async def fake_answer(repo_id, question, mode="agent", model=None, on_event=None):
        on_event({"type": "tool_call", "id": "1", "tool": "search_code", "args": {"query": question}})
        on_event({"type": "tool_result", "id": "1", "summary": "3 results", "refs": []})
        on_event({"type": "answer", "result": {"answer": "ok", "citations": []}})

    monkeypatch.setattr(api.agent, "answer", fake_answer)
    response = client.post(f"/api/repos/{ingested_repo}/ask/stream", json={"question": "how are timeouts set?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer", "done"]


def test_stream_reports_errors(ingested_repo, client, monkeypatch):
    async def failing_answer(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(api.agent, "answer", failing_answer)
    response = client.post(f"/api/repos/{ingested_repo}/ask/stream", json={"question": "anything?"})
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[0] == {"type": "error", "message": "RuntimeError: model unavailable"}
    assert events[-1]["type"] == "done"
