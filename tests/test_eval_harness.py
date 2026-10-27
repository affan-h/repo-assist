"""The evaluation harness end to end with scripted models (no API calls)."""

import asyncio
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from evals import common, run_eval
from repo_assist import embeddings
from tests.conftest import HashEmbedder

QUESTION = {
    "id": "Q1", "repo": "widget", "category": "where", "expected": "answer", "split": "test",
    "question": "Where is the transport chosen for a URL?",
    "reference_answer": "Client._transport_for_url in pkg/client.py.",
    "key_facts": ["Client._transport_for_url picks the transport", "It lives in pkg/client.py"],
    "gold_evidence": ["pkg/client.py:19-21"], "gold_mentions": ["_transport_for_url", "pkg/client.py"],
}


def _agent_model():
    def respond(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("find_symbol", {"name": "_transport_for_url"})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer": "`Client._transport_for_url` in pkg/client.py picks it [1].",
            "citations": ["pkg/client.py:19-21"], "abstained": False})])
    return FunctionModel(respond)


def _judge_model():
    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "key_facts": [{"fact": f, "covered": True} for f in QUESTION["key_facts"]],
            "claims": [{"claim": "The method picks the transport", "supported": True}],
            "declined": False, "contradicts_reference": False, "notes": "ok"})])
    return FunctionModel(respond)


def test_evaluate_summarise_render(ingested_repo, monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps({"questions": [QUESTION]}))
    monkeypatch.setattr(common, "DATASET", dataset)

    for mode in ("agent", "single_shot"):
        record = asyncio.run(run_eval.evaluate_one(QUESTION, mode, {"agent": _agent_model(), "judge": _judge_model()}))
        assert record["deterministic"]["abstained_as_expected"] is True
        out = run_eval.runs_dir("test")
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"{mode}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    agent_record = run_eval.load_runs("test", "agent")["Q1"]
    assert agent_record["deterministic"]["gold_in_citations"] is True
    assert agent_record["deterministic"]["mention_recall"] == 1.0
    assert [t["tool"] for t in agent_record["trace"]] == ["find_symbol"]

    summary = run_eval.summarise("test", common.load_questions(split="test"))
    assert summary["questions"] == 1
    assert summary["modes"]["agent"]["overall"]["correctness"]["mean"] == 1.0
    assert "agent vs single_shot" in summary["comparisons"]
    report = run_eval.render(summary)
    assert "Paired comparisons" in report and "| Answer correctness (judge) |" in report
