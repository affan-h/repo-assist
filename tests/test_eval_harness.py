"""The evaluation harness end to end with scripted models (no API calls)."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from evals import common, report, run_eval
from repo_assist import embeddings
from tests.conftest import HashEmbedder

ANSWERABLE = {
    "id": "Q1", "repo": "widget", "category": "where", "expected": "answer", "split": "test",
    "question": "Where is the transport chosen for a URL?",
    "reference_answer": "Client._transport_for_url in pkg/client.py.",
    "key_facts": ["Client._transport_for_url picks the transport", "It lives in pkg/client.py"],
    "gold_evidence": ["pkg/client.py:19-21"],
}
UNANSWERABLE = {
    "id": "Q2", "repo": "widget", "category": "unanswerable", "expected": "abstain", "split": "test",
    "question": "Why is the library called widget?", "reference_answer": "Not documented.",
    "key_facts": ["Says no rationale is documented"], "gold_evidence": [],
}


def _agent_model():
    def respond(messages, info):
        output = info.output_tools[0].name
        if "called widget" in str(messages[0]):
            return ModelResponse(parts=[ToolCallPart(output, {"answer": "Not documented.", "citations": [], "abstained": True})])
        if info.function_tools and len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("find_symbol", {"name": "_transport_for_url"})])
        ref = "pkg/client.py:19-21"
        return ModelResponse(parts=[ToolCallPart(output, {
            "answer": "`Client._transport_for_url` in pkg/client.py picks it [1].", "citations": [ref], "abstained": False})])
    return FunctionModel(respond)


def _judge_model():
    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "key_facts": [{"fact": f, "covered": True} for f in ANSWERABLE["key_facts"]],
            "claims": [{"claim": "The method picks the transport", "supported": True}],
            "contradicts_reference": False, "notes": "ok"})])
    return FunctionModel(respond)


@pytest.fixture
def eval_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps({"questions": [ANSWERABLE, UNANSWERABLE]}))
    monkeypatch.setattr(common, "DATASET", dataset)
    return tmp_path


def test_run_and_report(ingested_repo, monkeypatch, eval_dirs):
    monkeypatch.setattr(embeddings, "_embedder", HashEmbedder())
    models = {"agent": _agent_model(), "judge": _judge_model()}
    out = common.results_dir("test") / "answers.jsonl"
    out.parent.mkdir(parents=True)
    for question in (ANSWERABLE, UNANSWERABLE):
        for system in run_eval.SYSTEMS:
            record = asyncio.run(run_eval.answer_and_grade(question, system, models))
            with out.open("a") as f:
                f.write(json.dumps(record) + "\n")

    records = run_eval.load_answers("test")
    assert records[("Q2", "agent")]["judgement"] is None          # declines are not sent to the judge
    assert [t["tool"] for t in records[("Q1", "agent")]["trace"]] == ["find_symbol"]

    result = report.build_report("test")
    agent = result["systems"]["agent"]["overall"]
    assert result["questions"] == 2
    assert agent["correctness"]["mean"] == 1.0
    assert agent["correct_declines"] == 1.0 and agent["false_declines"] == 0.0
    assert agent["gold_cited"] == 1.0 and agent["claims_supported"] == 1.0
    assert result["comparison"]["correctness_diff"] == 0.0
    text = report.write_report("test")
    assert "| Correctness |" in text and (common.results_dir("test") / "report.json").exists()


@pytest.mark.parametrize("abstained, judgement, expected", [
    (True, None, 0.0),                                                        # declined an answerable question
    (False, {"key_facts": [{"fact": "a", "covered": True}, {"fact": "b", "covered": False}],
             "claims": [], "contradicts_reference": False}, 0.5),
    (False, {"key_facts": [{"fact": "a", "covered": True}], "claims": [], "contradicts_reference": True}, 0.0),
])
def test_score_answerable(abstained, judgement, expected):
    record = {"abstained": abstained, "judgement": judgement, "citations": [], "quotes": []}
    assert report.score(ANSWERABLE, record)["correct"] == expected


def test_score_unanswerable_uses_the_flag():
    answered = {"abstained": False, "judgement": {"key_facts": [], "claims": [], "contradicts_reference": False},
                "citations": [], "quotes": []}
    assert report.score(UNANSWERABLE, answered)["correct"] == 0.0
    assert report.score(UNANSWERABLE, {**answered, "abstained": True, "judgement": None})["correct"] == 1.0
