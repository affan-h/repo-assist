"""LLM-as-judge: key-fact coverage, abstention, and claim-level groundedness.

The judge sees the reference answer and key facts (for correctness) and,
separately, only the evidence the system cited (for groundedness), so an answer
that is right from prior knowledge but unsupported by its citations is caught.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from repo_assist.config import get_settings


class FactCheck(BaseModel):
    fact: str
    covered: bool = Field(description="The answer states this fact or a clear equivalent.")


class ClaimCheck(BaseModel):
    claim: str
    supported: bool = Field(description="The cited evidence text explicitly supports this claim.")


class Judgement(BaseModel):
    key_facts: list[FactCheck] = Field(description="One entry per key fact, in the order given.")
    claims: list[ClaimCheck] = Field(description=(
        "Atomic factual claims the answer makes about the repository (skip hedges, restatements of the "
        "question and statements that something was not found)."))
    declined: bool = Field(description=(
        "The answer declines to give a reason/answer, e.g. says no documented rationale was found."))
    contradicts_reference: bool = Field(description="The answer asserts something the reference contradicts.")
    notes: str = Field(description="One or two sentences explaining the grading.")


INSTRUCTIONS = """\
You grade answers produced by a question-answering system over a code repository. Be strict and literal.

1. key_facts: for each key fact, mark covered=true only if the system answer states it (paraphrase is fine).
   If the expected behaviour is "abstain", the key facts describe what a good decline looks like.
2. claims: list the answer's atomic factual claims about the repository and mark each supported=true only
   if the CITED EVIDENCE (not the reference, not your own knowledge) explicitly supports it. Do not list
   statements that something was not found or is not documented; those are not claims about the repository.
3. declined: true if the answer says the evidence does not contain the answer/rationale.
4. contradicts_reference: true if the answer asserts something incompatible with the reference answer."""


_judge: Agent[None, Judgement] | None = None


def get_judge() -> Agent[None, Judgement]:
    global _judge
    if _judge is None:
        _judge = Agent(get_settings().judge_model, output_type=Judgement, instructions=INSTRUCTIONS, retries=2)
    return _judge


def build_prompt(question: dict, answer: str, abstained: bool, cited: list[tuple[str, str]]) -> str:
    facts = "\n".join(f"- {f}" for f in question["key_facts"])
    evidence = "\n\n".join(f"[{ref}]\n{text}" for ref, text in cited) or "(no evidence cited)"
    return (
        f"QUESTION:\n{question['question']}\n\n"
        f"EXPECTED BEHAVIOUR: {question['expected']}\n\n"
        f"REFERENCE ANSWER:\n{question['reference_answer']}\n\n"
        f"KEY FACTS:\n{facts}\n\n"
        f"SYSTEM ANSWER (abstained={abstained}):\n{answer}\n\n"
        f"CITED EVIDENCE:\n{evidence}"
    )


async def judge(question: dict, answer: str, abstained: bool, cited: list[tuple[str, str]]) -> Judgement:
    result = await get_judge().run(build_prompt(question, answer, abstained, cited))
    return result.output


def score(question: dict, judgement: Judgement, abstained: bool) -> dict:
    """Turn a judgement into per-question metrics."""
    declined = abstained or judgement.declined
    if question["expected"] == "abstain" and judgement.key_facts:
        # Key fact 1 of every abstain question is "says no documented rationale was found"; an answer that
        # says so while also describing related facts (e.g. what changed) is a correct decline.
        declined = declined or judgement.key_facts[0].covered
    facts = judgement.key_facts
    coverage = sum(f.covered for f in facts) / len(facts) if facts else 0.0
    claims = judgement.claims
    metrics = {
        "groundedness": sum(c.supported for c in claims) / len(claims) if claims else None,
        "unsupported_claims": [c.claim for c in claims if not c.supported],
    }
    if question["expected"] == "abstain":
        metrics.update(correct=1.0 if declined and not judgement.contradicts_reference else 0.0,
                       abstained_correctly=declined, false_abstention=False)
    else:
        correct = 0.0 if declined or judgement.contradicts_reference else coverage
        metrics.update(correct=correct, abstained_correctly=None, false_abstention=declined)
    return metrics
