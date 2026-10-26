"""Fetch issues, pull requests, discussions (with comments) and releases via the GitHub GraphQL API.

Threads created after the indexed commit are skipped, so the history index
matches the code snapshot (and evaluation can't benefit from future knowledge).
"""

import re
import time
from collections.abc import Callable

import httpx

from repo_assist.store import Chunk

GRAPHQL_URL = "https://api.github.com/graphql"
MAX_THREAD_CHARS = 30_000
COMMENTS_PER_THREAD = 30

_COMMENTS = f"comments(first: {COMMENTS_PER_THREAD}) {{ nodes {{ body author {{ login }} }} }}"
_FIELDS = {
    "issues": f"number title body createdAt author {{ login }} {_COMMENTS}",
    "pullRequests": f"number title body createdAt mergedAt author {{ login }} {_COMMENTS}",
    "discussions": f"number title body createdAt author {{ login }} answer {{ body author {{ login }} }} {_COMMENTS}",
    "releases": "tagName name description createdAt",
}
_KIND = {"issues": "issue", "pullRequests": "pr", "discussions": "discussion", "releases": "release"}
_REF_PREFIX = {"issue": "issue#", "pr": "PR#", "discussion": "discussion#"}

_QUERY = """
query($owner: String!, $name: String!, $cursor: String, $pageSize: Int!) {
  repository(owner: $owner, name: $name) {
    %(connection)s(first: $pageSize, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { %(fields)s }
    }
  }
}
"""

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def thread_ref(kind: str, number: int) -> str:
    return f"{_REF_PREFIX[kind]}{number}"


def fetch_thread_chunks(owner: str, name: str, token: str, created_before: str, max_threads: int,
                        progress: Callable[[str], None] = lambda _: None) -> list[Chunk]:
    chunks: list[Chunk] = []
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        for connection in ("releases", "issues", "pullRequests", "discussions"):
            remaining = max_threads - len(chunks)
            if remaining <= 0:
                break
            kind_chunks = _fetch_connection(client, owner, name, connection, created_before, remaining)
            chunks.extend(kind_chunks)
            progress(f"Fetched {len(kind_chunks)} {connection}")
    return chunks


def _fetch_connection(client: httpx.Client, owner: str, name: str, connection: str,
                      created_before: str, limit: int) -> list[Chunk]:
    query = _QUERY % {"connection": connection, "fields": _FIELDS[connection]}
    chunks, cursor, page_size = [], None, 25
    while len(chunks) < limit:
        data = _graphql(client, query, {"owner": owner, "name": name, "cursor": cursor, "pageSize": page_size})
        if data is None:  # page too heavy for GitHub's timeout: retry it smaller
            if page_size <= 10:
                raise RuntimeError(f"GitHub GraphQL kept failing for {connection}")
            page_size //= 2
            continue
        page = data["repository"][connection]
        for node in page["nodes"]:
            if node["createdAt"] <= created_before:
                kind = _KIND[connection]
                chunks.append(_release_chunk(node) if kind == "release" else _thread_chunk(kind, node))
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return chunks[:limit]


def _graphql(client: httpx.Client, query: str, variables: dict) -> dict | None:
    for attempt in range(4):
        response = client.post(GRAPHQL_URL, json={"query": query, "variables": variables})
        if response.status_code in (502, 504):
            return None
        if response.status_code == 403 and "rate limit" in response.text.lower():
            time.sleep(30 * (attempt + 1))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            message = payload["errors"][0].get("message", "")
            if "timeout" in message.lower() or "something went wrong" in message.lower():
                return None
            raise RuntimeError(f"GitHub GraphQL error: {message}")
        return payload["data"]
    raise RuntimeError("GitHub GraphQL rate limit not recovering")


def _clean(text: str | None) -> str:
    return _HTML_COMMENT.sub("", text or "").strip()


def _thread_chunk(kind: str, node: dict) -> Chunk:
    author = (node.get("author") or {}).get("login", "ghost")
    title, body = node["title"], _clean(node["body"])
    status = ""
    if kind == "pr":
        status = " (merged)" if node.get("mergedAt") else " (closed without merge)"
    parts = [f"{title}\n{_REF_PREFIX[kind]}{node['number']} by @{author} on {node['createdAt'][:10]}{status}",
             body]
    answer = node.get("answer")
    if answer:
        parts.append(f"--- accepted answer by @{(answer.get('author') or {}).get('login', 'ghost')} ---\n"
                     f"{_clean(answer['body'])}")
    for comment in node["comments"]["nodes"]:
        commenter = (comment.get("author") or {}).get("login", "ghost")
        parts.append(f"--- comment by @{commenter} ---\n{_clean(comment['body'])}")
    text = "\n\n".join(p for p in parts if p)[:MAX_THREAD_CHARS]
    return Chunk(
        kind=kind, ref=thread_ref(kind, node["number"]), title=title, text=text,
        created_at=node["createdAt"], embed_text=f"{title}\n{body}"[:2000],
    )


def _release_chunk(node: dict) -> Chunk:
    tag, title = node["tagName"], node["name"] or node["tagName"]
    body = _clean(node["description"])
    return Chunk(
        kind="release", ref=f"release:{tag}", title=title, created_at=node["createdAt"],
        text=f"{title}\nRelease {tag} on {node['createdAt'][:10]}\n\n{body}"[:MAX_THREAD_CHARS],
        embed_text=f"Release {title}\n{body}"[:2000],
    )
