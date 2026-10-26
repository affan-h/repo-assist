import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from repo_assist import agent, embeddings
from repo_assist.agent import Answer, Evidence, resolve_citation, validate_answer
from tests.conftest import HashEmbedder

EVIDENCE = {
    "pkg/client.py:10-20": Evidence("pkg/client.py:10-20", "code", "\n".join(f"line {i}" for i in range(10, 21)),
                                    path="pkg/client.py", start_line=10, end_line=20),
    "PR#12": Evidence("PR#12", "pr", "Add retries"),
    "commit:abcdef1234": Evidence("commit:abcdef1234", "commit", "Fix timeout"),
}


@pytest.mark.parametrize("ref, expected", [
    ("pkg/client.py:10-20", "pkg/client.py:10-20"),
    ("pkg/client.py:12-14", "pkg/client.py:12-14"),        # sub-range of a seen chunk
    ("pkg/client.py:15", "pkg/client.py:15-15"),            # single line
    ("pkg/client.py#L12-L14", "pkg/client.py:12-14"),       # GitHub-style anchor
    ("`pr#12`", "PR#12"),                                   # case and formatting
    ("commit:abcdef1", "commit:abcdef1234"),                # abbreviated sha
    ("pkg/client.py:5-12", None),                           # extends beyond what was seen
    ("pkg/client.py:30-31", None),
    ("pkg/other.py:10-20", None),
    ("PR#13", None),
    ("commit:abc", None),                                   # too short to be unambiguous
    ("pkg/client.py", None),                                # no line range
])
def test_resolve_citation(ref, expected):
    resolved = resolve_citation(ref, EVIDENCE)
    assert (resolved[0] if resolved else None) == expected


def test_validate_answer():
    ok = Answer(answer="It retries [1] on timeouts [2].", citations=["PR#12", "pkg/client.py:12-13"], abstained=False)
    assert validate_answer(ok, EVIDENCE) == []

    uncited = Answer(answer="It retries.", citations=[], abstained=False)
    assert "must cite" in validate_answer(uncited, EVIDENCE)[0]

    abstained = Answer(answer="No documented rationale was found.", citations=[], abstained=True)
    assert validate_answer(abstained, EVIDENCE) == []

    bad_marker = Answer(answer="See [3].", citations=["PR#12"], abstained=False)
    assert "markers [3]" in validate_answer(bad_marker, EVIDENCE)[0]

    invented = Answer(answer="See [1].", citations=["PR#99"], abstained=False)
    assert "'PR#99'" in validate_answer(invented, EVIDENCE)[0]


def _scripted_model(seen_retry: list):
    """Search, then cite an unseen ref (rejected), then cite the found chunk."""

    def respond(messages, info: AgentInfo) -> ModelResponse:
        output_tool = info.output_tools[0].name
        last_parts = messages[-1].parts
        retries = [p for p in last_parts if isinstance(p, ToolReturnPart) and p.outcome == "retried"]
        if retries:
            seen_retry.append(retries[0].content)
            found = next(p for m in messages for p in m.parts
                         if isinstance(p, ToolReturnPart) and p.tool_name == "search_code")
            ref = found.content.split("]")[0].lstrip("[")
            return ModelResponse(parts=[ToolCallPart(output_tool, {
                "answer": "It returns the mounted transport [1].", "citations": [ref], "abstained": False})])
        if any(isinstance(p, ToolReturnPart) for p in last_parts):
            return ModelResponse(parts=[ToolCallPart(output_tool, {
                "answer": "It returns a transport [1].", "citations": ["pkg/nowhere.py:1-2"], "abstained": False})])
        return ModelResponse(parts=[ToolCallPart("search_code", {"query": "transport for url"})])

    return FunctionModel(respond)


def test_agent_loop_enforces_grounded_citations(ingested_repo, monkeypatch):
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    events, seen_retry = [], []
    result = asyncio.run(agent.answer(ingested_repo, "What does _transport_for_url return?",
                                      model=_scripted_model(seen_retry), on_event=events.append))

    assert "pkg/nowhere.py:1-2" in seen_retry[0], "the invented citation should have been rejected"
    assert result.error is None and not result.abstained
    assert [c.tool for c in result.trace] == ["search_code"]
    citation = result.citations[0]
    assert citation.path == "pkg/client.py" and citation.snippet.lstrip().startswith("def _transport_for_url")
    assert citation.url.startswith("https://github.com/acme/widget/blob/")
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer"]
    assert events[0]["args"] == {"query": "transport for url"}


def test_tools_register_evidence(ingested_repo, monkeypatch):
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    calls = iter([
        ToolCallPart("find_symbol", {"name": "Client.send"}),
        ToolCallPart("grep", {"pattern": "DEFAULT_TIMEOUT ="}),
        ToolCallPart("read_file", {"path": "pkg/client.py", "start_line": 1, "end_line": 4}),
        ToolCallPart("search_history", {"query": "why default timeout"}),
    ])

    def respond(messages, info):
        nxt = next(calls, None)
        if nxt is not None:
            return ModelResponse(parts=[nxt])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer": "Timeouts default to 5s so hung servers fail fast [1][2].",
            "citations": ["pkg/client.py:4-4", "commit:" + _commit_sha()], "abstained": False})])

    def _commit_sha():
        return next(r for r in evidence_refs if r.startswith("commit:")).split(":")[1][:7]

    evidence_refs = []
    original_add = agent.Deps.add
    monkeypatch.setattr(agent.Deps, "add", lambda self, item: (evidence_refs.append(item.ref), original_add(self, item)))
    result = asyncio.run(agent.answer(ingested_repo, "Why is the default timeout 5s?", model=FunctionModel(respond)))

    assert [c.tool for c in result.trace] == ["find_symbol", "grep", "read_file", "search_history"]
    assert any(r.endswith(":4-4") for r in result.evidence)          # grep hit on DEFAULT_TIMEOUT
    assert [c.kind for c in result.citations] == ["code", "commit"]
    assert result.citations[1].url.startswith("https://github.com/acme/widget/commit/")


def test_single_shot_uses_retrieved_context_without_tools(ingested_repo, monkeypatch):
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    prompts = []

    def respond(messages, info):
        assert not info.function_tools
        prompts.append(messages[0].parts[-1].content)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer": "No documented rationale.", "citations": [], "abstained": True})])

    result = asyncio.run(agent.answer(ingested_repo, "transport for url", mode="single_shot",
                                      model=FunctionModel(respond)))
    assert "Retrieved evidence" in prompts[0] and "_transport_for_url" in prompts[0]
    assert result.abstained and result.trace == [] and result.evidence
