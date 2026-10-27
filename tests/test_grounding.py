import pytest

from repo_assist.grounding import Answer, Evidence, Quote, is_rationale_question, quote_is_supported, validate_answer

EVIDENCE = {
    "issue#572": Evidence("issue#572", "issue",
                          "The approach of bridging a sync interface onto an async one had issues:\n"
                          "* Having Sync and Async variant classes was leading to **poorer, less comprehensible** code."),
    "pkg/a.py:1-3": Evidence("pkg/a.py:1-3", "code", "x = 1\n# keep it simple\ny = 2", path="pkg/a.py",
                             start_line=1, end_line=3),
}


@pytest.mark.parametrize("quote, ok", [
    ("was leading to poorer, less comprehensible code", True),           # markdown emphasis ignored
    ("bridging a sync interface   onto an async one", True),             # whitespace normalised
    ("BRIDGING A SYNC INTERFACE ONTO AN ASYNC ONE", True),                # case-insensitive
    ("bridging a sync interface onto a sync one", False),                 # altered wording
    ("had issues", False),                                                # too short to carry a reason
])
def test_quotes_must_be_verbatim(quote, ok):
    assert quote_is_supported(Quote(ref="issue#572", quote=quote), EVIDENCE) is ok


def test_quote_must_come_from_the_cited_source():
    assert not quote_is_supported(Quote(ref="pkg/a.py:1-3", quote="was leading to poorer code"), EVIDENCE)
    assert not quote_is_supported(Quote(ref="issue#999", quote="was leading to poorer, less comprehensible code"), EVIDENCE)


@pytest.mark.parametrize("question, rationale", [
    ("Why was sync support dropped?", True),
    ("What was the reason for the split?", True),
    ("What has the maintainer said about the trade-offs?", True),
    ("Where is the retry logic?", False),
    ("How does pagination stop?", False),
])
def test_rationale_question_detection(question, rationale):
    assert is_rationale_question(question) is rationale


def test_why_answers_need_a_verified_quote():
    unquoted = Answer(answer="It produced messy code [1].", citations=["issue#572"], abstained=False)
    problems = validate_answer(unquoted, EVIDENCE, "Why was sync dropped?")
    assert any("rationale_quotes" in p for p in problems)

    quoted = unquoted.model_copy(update={"rationale_quotes": [
        Quote(ref="issue#572", quote="Having Sync and Async variant classes was leading to poorer, less comprehensible code")]})
    assert validate_answer(quoted, EVIDENCE, "Why was sync dropped?") == []

    invented = unquoted.model_copy(update={"rationale_quotes": [
        Quote(ref="issue#572", quote="Sync support was too slow for production users")]})
    assert any("not found verbatim" in p for p in validate_answer(invented, EVIDENCE, "Why was sync dropped?"))


def test_abstaining_or_non_why_answers_need_no_quote():
    abstained = Answer(answer="No documented rationale.", citations=[], abstained=True)
    assert validate_answer(abstained, EVIDENCE, "Why 5 seconds?") == []
    factual = Answer(answer="x is 1 [1].", citations=["pkg/a.py:1-1"], abstained=False)
    assert validate_answer(factual, EVIDENCE, "What is x?") == []
