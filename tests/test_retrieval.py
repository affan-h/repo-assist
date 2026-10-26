import pytest

from repo_assist import retrieval
from tests.conftest import HashEmbedder


def test_fts_query_drops_stopwords_and_splits_identifiers():
    assert retrieval.fts_query("What does `_transport_for_url` do?") == '"transport" OR "url"'
    assert retrieval.fts_query("how is beforeRetry called") == '"before" OR "retry" OR "called"'
    assert retrieval.fts_query("what is the") is None


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = dict(retrieval.reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]]))
    assert max(fused, key=fused.get) == 1          # rank 1 + rank 2 beats everything else
    assert fused[3] > fused[2] and fused[3] > fused[4]
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_method_by_description(ingested_repo, mode):
    hits = retrieval.search(ingested_repo, "transport for url", mode=mode, embedder=HashEmbedder())
    assert hits, mode
    assert hits[0].symbol == "Client._transport_for_url"
    assert hits[0].ref == f"pkg/client.py:{hits[0].start_line}-{hits[0].end_line}"


def test_search_respects_kinds(ingested_repo):
    history = retrieval.search(ingested_repo, "default timeout", kinds=retrieval.HISTORY_KINDS,
                               embedder=HashEmbedder())
    assert [h.kind for h in history] == ["commit"]
    assert "hung servers fail fast" in history[0].text
    code = retrieval.search(ingested_repo, "default timeout", embedder=HashEmbedder())
    assert all(h.kind in retrieval.CODE_KINDS for h in code)


def test_get_chunk_is_case_insensitive(ingested_repo):
    hit = retrieval.search(ingested_repo, "default timeout", kinds=("commit",), embedder=HashEmbedder())[0]
    assert retrieval.get_chunk(ingested_repo, hit.ref.upper()).id == hit.id


def test_find_definitions_prefers_qualified_match(ingested_repo):
    defs = retrieval.find_definitions(ingested_repo, "httpx.Client.get()")
    assert [(d["qualified_name"], d["path"]) for d in defs] == [("Client.get", "pkg/client.py")]
    assert retrieval.find_definitions(ingested_repo, "nonexistent") == []


def test_find_callers(ingested_repo):
    callers = retrieval.find_callers(ingested_repo, "Client.send")
    assert [c["caller"] for c in callers] == ["Client.get"]


def test_grep_and_read_lines(ingested_repo):
    matches, total = retrieval.grep(ingested_repo, r"DEFAULT_TIMEOUT")
    assert total == 2 and matches[0][0] == "pkg/client.py"
    # grep covers files excluded from the search index, such as tests
    assert retrieval.grep(ingested_repo, r"def test_get", path_glob="tests/*")[1] == 1

    lines, start, end, total_lines = retrieval.read_lines(ingested_repo, "pkg/client.py", 1, 3)
    assert (start, end) == (1, 3) and lines[0] == '"""HTTP client."""'
    _, _, end, _ = retrieval.read_lines(ingested_repo, "pkg/client.py", 20, 10_000)
    assert end == total_lines


def test_read_lines_rejects_untracked_paths(ingested_repo):
    with pytest.raises(KeyError):
        retrieval.read_lines(ingested_repo, "../../etc/passwd", 1, 5)
    with pytest.raises(KeyError):
        retrieval.read_lines(ingested_repo, ".git/config", 1, 5)
