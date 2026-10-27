"""Passage-level history search, git-backed tools, and agent variant toolsets."""

import asyncio
import subprocess

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from repo_assist import agent, embeddings, retrieval, store
from repo_assist.config import get_settings
from repo_assist.ingest import github
from tests.conftest import HashEmbedder

DEEP_COMMENT = "We deliberately chose a five second default because hung connections block worker pools."
LONG_ISSUE = "Default timeouts\nissue#7 by @alice on 2025-01-01\n\nShould the client time out by default?\n\n" + \
    "\n\n".join(f"--- comment by @user{i} ---\n" + "Unrelated discussion about logging formats. " * 12
                for i in range(6)) + f"\n\n--- comment by @maintainer ---\n{DEEP_COMMENT}"


@pytest.fixture
def repo_with_history(data_dir, fixture_repo, monkeypatch):
    """Fixture repo with a long issue thread, a PR, and a second commit that links them."""
    from repo_assist import ingest
    from repo_assist.ingest import git

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    get_settings.cache_clear()
    threads = [
        store.Chunk(kind="issue", ref="issue#7", title="Default timeouts", text=LONG_ISSUE,
                    created_at="2025-01-01T00:00:00Z", embed_text="Default timeouts"),
        store.Chunk(kind="pr", ref="PR#12", title="Make timeout configurable", text="Make timeout configurable\n\nFixes #7.",
                    created_at="2025-01-02T00:00:00Z"),
    ]
    monkeypatch.setattr(github, "fetch_thread_chunks", lambda *a, **k: list(threads))

    def fake_clone(owner, name, dest, ref=None):
        import shutil
        if not dest.exists():
            shutil.copytree(fixture_repo, dest)
            path = dest / "pkg" / "client.py"
            path.write_text(path.read_text().replace("DEFAULT_TIMEOUT = 5.0", "DEFAULT_TIMEOUT = 5.0  # seconds"))
            env = {"GIT_AUTHOR_NAME": "Dev", "GIT_AUTHOR_EMAIL": "d@x", "GIT_COMMITTER_NAME": "Dev",
                   "GIT_COMMITTER_EMAIL": "d@x", "PATH": "/usr/bin:/bin:/usr/local/bin"}
            subprocess.run(["git", "-C", str(dest), "commit", "-qam", "Make timeout configurable (#12)\n\nFixes #7"],
                           check=True, env=env)
        sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        return git.Checkout(path=dest, sha=sha, date="2026-01-01T00:00:00+00:00")

    monkeypatch.setattr(git, "clone", fake_clone)
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    repo_id = ingest.register_repo("acme/widget")
    ingest.ingest(repo_id, embedder=HashEmbedder())
    return repo_id


def test_long_threads_get_keyword_passages(repo_with_history):
    with store.session() as conn:
        passages = conn.execute("SELECT ref, parent, embedding FROM chunks WHERE kind = 'passage'").fetchall()
    assert passages and all(p["parent"] == "issue#7" and p["embedding"] is None for p in passages)


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_history_search_lands_on_the_matching_comment(repo_with_history, mode):
    # In hybrid mode the thread can also rank via its whole-text embedding; the result must still
    # show the matching comment rather than the start of the thread.
    hits = retrieval.search_history(repo_with_history, "five second default hung connections worker pools",
                                    mode=mode)
    top = next(h for h in hits[:3] if h.ref == "issue#7")
    assert top.kind == "issue"
    assert DEEP_COMMENT in top.text and top.passage_offset > 0
    assert LONG_ISSUE[top.passage_offset:].lstrip().startswith("--- comment by @maintainer")


def _run(repo_id, mode, calls, question="What changed the timeout?"):
    seen = {}

    def respond(messages, info):
        seen.setdefault("tools", {t.name for t in info.function_tools})
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        seen["returns"] = [r.content for r in returns]
        nxt = next(calls, None)
        if nxt is not None:
            return ModelResponse(parts=[nxt])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer": "Unknown.", "citations": [], "abstained": True})])

    result = asyncio.run(agent.answer(repo_id, question, mode=mode, model=FunctionModel(respond)))
    return result, seen


def test_code_history_traces_lines_to_linked_threads(repo_with_history):
    with store.session() as conn:
        line = next(r["start_line"] for r in conn.execute(
            "SELECT start_line FROM chunks WHERE path = 'pkg/client.py' AND text LIKE '%DEFAULT_TIMEOUT = 5.0%'"))
    result, seen = _run(repo_with_history, "agent", iter([
        ToolCallPart("code_history", {"path": "pkg/client.py", "start_line": line, "end_line": line + 3}),
        ToolCallPart("read_thread", {"ref": "PR#12"}),
    ]))
    history = seen["returns"][0]
    assert "Make timeout configurable (#12)" in history and "linked: issue#7, PR#12" in history
    assert "Linked threads mentioned here: issue#7" in seen["returns"][1]
    assert any(ref.startswith("commit:") for ref in result.evidence)


def test_noindex_variant_only_has_checkout_and_git_tools(repo_with_history):
    result, seen = _run(repo_with_history, "agent_noindex", iter([
        ToolCallPart("list_files", {"path_glob": "pkg/*"}),
        ToolCallPart("search_commits", {"pattern": "timeout"}),
    ]))
    assert seen["tools"] == {"list_files", "grep", "read_file", "search_commits", "code_history"}
    assert seen["returns"][0] == "pkg/client.py"
    assert "Make timeout configurable (#12)" in seen["returns"][1] and "linked: #7, #12" in seen["returns"][1]


def test_bm25_variant_searches_without_embeddings(repo_with_history, monkeypatch):
    modes = []
    original = retrieval.search
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: (modes.append(k.get("mode")), original(*a, **k))[1])
    _run(repo_with_history, "agent_bm25", iter([ToolCallPart("search_code", {"query": "timeout"})]))
    assert modes == ["bm25"]
