import pytest

from repo_assist import store
from repo_assist.ingest import code, docs, git
from tests.conftest import FIXTURE_FILES


@pytest.mark.parametrize("url", [
    "encode/httpx",
    "https://github.com/encode/httpx",
    "https://github.com/encode/httpx.git",
    "https://github.com/encode/httpx/",
    "git@github.com:encode/httpx.git",
])
def test_parse_repo_url(url):
    assert git.parse_repo_url(url) == ("encode", "httpx")


def test_parse_repo_url_rejects_other_hosts():
    with pytest.raises(ValueError):
        git.parse_repo_url("https://gitlab.com/a/b/c")


def test_python_symbols_and_chunks():
    parsed = code.parse_file("pkg/client.py", FIXTURE_FILES["pkg/client.py"].encode())
    symbols = {s.qualified_name: s for s in parsed.symbols}
    assert set(symbols) == {"Client", "Client.__init__", "Client.get", "Client.send",
                            "Client._transport_for_url", "build_request"}
    assert symbols["Client.get"].kind == "method"
    assert symbols["build_request"].kind == "function"

    by_symbol = {c.symbol: c for c in parsed.chunks}
    method = by_symbol["Client._transport_for_url"]
    assert method.text.lstrip().startswith("def _transport_for_url")
    assert method.ref == f"pkg/client.py:{method.start_line}-{method.end_line}"
    # Class header chunk stops before the first method and lists members for search.
    assert "def __init__" not in by_symbol["Client"].text
    assert "_transport_for_url" in by_symbol["Client"].extra_search_terms
    # Module-level code (imports, constants) is kept in its own chunk.
    assert any(c.symbol is None and "DEFAULT_TIMEOUT = 5.0" in c.text for c in parsed.chunks)


def test_call_sites_record_enclosing_symbol():
    parsed = code.parse_file("pkg/client.py", FIXTURE_FILES["pkg/client.py"].encode())
    calls = {(c.callee, c.caller) for c in parsed.calls}
    assert ("send", "Client.get") in calls
    assert ("build_request", "Client.get") in calls
    assert ("_transport_for_url", "Client.send") in calls


def test_typescript_symbols():
    parsed = code.parse_file("src/retry.ts", FIXTURE_FILES["src/retry.ts"].encode())
    kinds = {s.qualified_name: s.kind for s in parsed.symbols}
    assert kinds == {"RetryOptions": "interface", "calculateRetryDelay": "function",
                     "Retrier": "class", "Retrier.run": "method"}
    assert ("computeCap", "calculateRetryDelay") in {(c.callee, c.caller) for c in parsed.calls}


def test_long_definitions_are_windowed():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    parsed = code.parse_file("big.py", f"def big():\n{body}\n".encode())
    spans = [(c.start_line, c.end_line) for c in parsed.chunks]
    assert len(spans) > 1 and spans[0][0] == 1 and spans[-1][1] == 201
    assert all(b[0] <= a[1] for a, b in zip(spans, spans[1:]))  # windows overlap


def test_markdown_sections_carry_breadcrumbs():
    chunks = docs.chunk_document("docs/guide.md", FIXTURE_FILES["docs/guide.md"])
    titles = [c.title for c in chunks]
    assert "Guide > Timeouts" in titles
    timeout_chunk = chunks[titles.index("Guide > Timeouts")]
    assert "five seconds" in timeout_chunk.text
    assert timeout_chunk.ref == f"docs/guide.md:{timeout_chunk.start_line}-{timeout_chunk.end_line}"


def test_headings_inside_code_fences_are_ignored():
    text = "# Title\n\nSome introduction text that is long enough to stand alone as a section.\n" \
           "Even more text here to pass the minimum size for a section chunk.\n\n" \
           "```bash\n# not a heading\necho hi\n```\n"
    assert [c.title for c in docs.chunk_document("a.md", text)] == ["Title"]


def test_test_files_are_not_indexed():
    assert not code.is_indexable_source("tests/test_client.py")
    assert not code.is_indexable_source("source/core/index.test.ts")
    assert code.is_indexable_source("httpx/_client.py")
    assert not code.is_indexable_source("types/index.d.ts")


def test_ingest_end_to_end(ingested_repo):
    with store.session() as conn:
        repo = store.get_repo(conn, ingested_repo)
        assert repo["status"] == "ready"
        assert repo["stats"]["code"] > 0 and repo["stats"]["doc"] > 0 and repo["stats"]["commit"] == 1
        kinds = {r["kind"] for r in conn.execute("SELECT kind FROM chunks WHERE repo_id = ?", (ingested_repo,))}
        assert kinds == {"code", "doc", "commit"}
        # tests/ is excluded from the search index
        assert not conn.execute("SELECT 1 FROM chunks WHERE path LIKE 'tests/%'").fetchone()
        fts_rows = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        chunk_rows = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        assert fts_rows == chunk_rows


def test_reingest_replaces_index(ingested_repo):
    from repo_assist import ingest
    from tests.conftest import HashEmbedder

    with store.session() as conn:
        before = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    ingest.ingest(ingested_repo, embedder=HashEmbedder())
    with store.session() as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == before
        assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == before


def test_github_failure_degrades_to_code_and_commits(data_dir, fixture_repo, monkeypatch):
    import httpx

    from repo_assist import ingest
    from repo_assist.config import get_settings
    from repo_assist.ingest import github
    from tests.conftest import HashEmbedder

    monkeypatch.setenv("GITHUB_TOKEN", "invalid")
    get_settings.cache_clear()

    def unauthorized(*args, **kwargs):
        request = httpx.Request("POST", github.GRAPHQL_URL)
        raise httpx.HTTPStatusError("401 Bad credentials", request=request, response=httpx.Response(401, request=request))

    monkeypatch.setattr(github, "fetch_thread_chunks", unauthorized)
    monkeypatch.setattr(git, "clone", lambda owner, name, dest, ref=None: git.Checkout(
        path=fixture_repo, sha="0" * 40, date="2026-01-01T00:00:00+00:00"))
    repo_id = ingest.register_repo("acme/widget")
    stats = ingest.ingest(repo_id, embedder=HashEmbedder())
    assert stats["commit"] == 1 and stats["code"] > 0
    assert "GitHub history skipped" in stats["warnings"][0]
