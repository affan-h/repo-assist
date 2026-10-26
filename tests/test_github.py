import json

import httpx

from repo_assist.ingest import github


def _node(number, created, title="Title", body="Body <!-- template -->", comments=()):
    return {"number": number, "title": title, "body": body, "createdAt": created, "mergedAt": None,
            "author": {"login": "alice"}, "answer": None,
            "comments": {"nodes": [{"body": c, "author": {"login": "bob"}} for c in comments]}}


def _page(nodes, has_next=False, cursor=None):
    return {"pageInfo": {"hasNextPage": has_next, "endCursor": cursor}, "nodes": nodes}


def test_fetch_threads_skips_future_threads_and_paginates(monkeypatch):
    pages = {
        ("issues", None): _page([_node(9, "2026-03-01T00:00:00Z"), _node(5, "2026-01-05T00:00:00Z",
                                 comments=["Because it is faster."])], True, "c1"),
        ("issues", "c1"): _page([_node(1, "2025-01-01T00:00:00Z")]),
        ("pullRequests", None): _page([_node(7, "2026-01-02T00:00:00Z", title="Add retries")]),
        ("discussions", None): _page([]),
        ("releases", None): _page([{"tagName": "v1.0.0", "name": "1.0", "description": "Why: simpler API.",
                                    "createdAt": "2025-06-01T00:00:00Z"}]),
    }
    requested_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        connection = next(c for c in ("issues", "pullRequests", "discussions", "releases") if f"{c}(first" in payload["query"])
        requested_sizes.append(payload["variables"]["pageSize"])
        page = pages[(connection, payload["variables"]["cursor"])]
        return httpx.Response(200, json={"data": {"repository": {connection: page}}})

    real_client = httpx.Client
    monkeypatch.setattr(github.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))

    chunks = github.fetch_thread_chunks("acme", "widget", "token", "2026-02-01T00:00:00Z", max_threads=100)
    refs = [c.ref for c in chunks]
    assert refs == ["release:v1.0.0", "issue#5", "issue#1", "PR#7"]   # issue#9 is after the cutoff

    issue = chunks[1]
    assert "template" not in issue.text                               # HTML comments stripped
    assert "--- comment by @bob ---\nBecause it is faster." in issue.text
    assert issue.embed_text.startswith("Title\nBody")                 # embeddings use title + body only
    assert "(closed without merge)" in chunks[3].text
    assert chunks[0].kind == "release" and "Why: simpler API." in chunks[0].text


def test_heavy_pages_are_retried_smaller(monkeypatch):
    sizes = []

    def handler(request):
        size = json.loads(request.content)["variables"]["pageSize"]
        sizes.append(size)
        if size > 10:
            return httpx.Response(502)
        return httpx.Response(200, json={"data": {"repository": {"issues": _page([_node(1, "2020-01-01T00:00:00Z")])}}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        chunks = github._fetch_connection(client, "a", "b", "issues", "2030-01-01T00:00:00Z", limit=10)
    assert sizes == [25, 12, 6] and [c.ref for c in chunks] == ["issue#1"]
