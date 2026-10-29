"""The Qdrant backend: same results as exact search, and blue/green rebuilds that can't leave a broken index."""

import pytest

from repo_assist import ingest, retrieval, store, vectors
from tests.conftest import HashEmbedder

QUERIES = ["transport for url", "default timeout", "build a request", "run the tests"]


def _dense(repo_id, query, kinds=retrieval.CODE_KINDS + retrieval.HISTORY_KINDS):
    with store.session() as conn:
        return retrieval.dense_search(conn, repo_id, query, kinds, HashEmbedder(), limit=10)


@pytest.mark.parametrize("backend", ["qdrant"])
def test_qdrant_matches_exact_search(ingested_repo, backend):
    exact = vectors.ExactIndex()
    for query in QUERIES:
        with store.session() as conn:
            expected = exact.search(conn, ingested_repo, HashEmbedder().embed_query(query), retrieval.CODE_KINDS, 10)
        got = _dense(ingested_repo, query, retrieval.CODE_KINDS)
        assert [s for _, s in got] == pytest.approx([s for _, s in expected], abs=1e-4), query
        assert _by_score(got) == _by_score(expected), query  # same ids at each score; ties may swap order


def _by_score(results):
    groups = {}
    for chunk_id, score in results:
        groups.setdefault(round(score, 4), set()).add(chunk_id)
    groups.pop(min(groups), None)  # a tie at the cutoff can be filled by any of its members
    return groups


@pytest.mark.parametrize("backend", ["qdrant"])
def test_reingest_switches_the_alias_and_drops_the_old_collection(ingested_repo, backend):
    index = vectors.get_index()
    before = index._target(ingested_repo)
    ingest.ingest(ingested_repo, embedder=HashEmbedder())
    after = index._target(ingested_repo)
    assert before and after and after != before
    assert [c.name for c in index.client.get_collections().collections] == [after]
    assert _dense(ingested_repo, "transport for url")


@pytest.mark.parametrize("backend", ["qdrant"])
def test_failed_reingest_keeps_the_previous_index(ingested_repo, backend, monkeypatch):
    index = vectors.get_index()
    before = index._target(ingested_repo)
    real_update = store.update_repo

    def fail_when_marking_ready(conn, repo_id, **fields):
        if fields.get("status") == "ready":
            raise RuntimeError("disk full")
        real_update(conn, repo_id, **fields)
    monkeypatch.setattr(store, "update_repo", fail_when_marking_ready)

    with pytest.raises(RuntimeError):
        ingest.ingest(ingested_repo, embedder=HashEmbedder())
    assert index._target(ingested_repo) == before                    # alias untouched
    assert [c.name for c in index.client.get_collections().collections] == [before]  # new one discarded
    assert _dense(ingested_repo, "transport for url")                # still searchable


@pytest.mark.parametrize("backend", ["qdrant"])
def test_missing_collection_is_rebuilt_from_sqlite(ingested_repo, backend):
    index = vectors.get_index()
    index.client.delete_collection(index._target(ingested_repo))
    assert index._target(ingested_repo) is None
    assert _dense(ingested_repo, "transport for url")
    assert index._target(ingested_repo) is not None
