"""LLM judge, used only for what can't be checked mechanically.

For an answer the system did not decline, the judge reports:
- which key facts from the reference the answer states (correctness), and
- for each factual claim in the answer, whether the evidence the answer *cited* supports it
  (groundedness: a correct claim the citations don't back is still ungrounded).

Whether the system declined is not judged: the answer carries an explicit `abstained`
flag, so abstention metrics are exact.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from repo_assist.config import get_settings


class FactCheck(BaseModel):
    fact: str
    covered: bool = Field(description="The answer states this fact or a clear equivalent.")


class ClaimCheck(BaseModel):
    claim: str
    supported: bool = Field(description="The cited evidence explicitly supports this claim.")


class Judgement(BaseModel):
    key_facts: list[FactCheck] = Field(description="One entry per key fact, in the order given.")
    claims: list[ClaimCheck] = Field(description=(
        "The answer's factual claims about the repository (skip hedges and restatements of the question)."))
    contradicts_reference: bool = Field(description="The answer asserts something the reference contradicts.")
    notes: str = Field(description="One or two sentences explaining the grading.")


INSTRUCTIONS = """\
You grade answers produced by a question-answering system over a code repository. Be strict and literal.

1. key_facts: for each key fact, covered=true only if the system answer states it (paraphrase is fine,
   vague gestures are not).
2. claims: list the answer's factual claims about the repository; supported=true only if the CITED EVIDENCE
   (not the reference, not your own knowledge) explicitly supports the claim.
3. contradicts_reference: true if the answer asserts something incompatible with the reference answer."""

JUDGE = Agent(None, output_type=Judgement, instructions=INSTRUCTIONS, retries=2)


def build_prompt(question: dict, answer: str, cited: list[tuple[str, str]]) -> str:
    facts = "\n".join(f"- {f}" for f in question["key_facts"])
    evidence = "\n\n".join(f"[{ref}]\n{text}" for ref, text in cited) or "(no evidence cited)"
    return (f"QUESTION:\n{question['question']}\n\nREFERENCE ANSWER:\n{question['reference_answer']}\n\n"
            f"KEY FACTS:\n{facts}\n\nSYSTEM ANSWER:\n{answer}\n\nCITED EVIDENCE:\n{evidence}")


async def judge(question: dict, answer: str, cited: list[tuple[str, str]], model=None) -> Judgement:
    result = await JUDGE.run(build_prompt(question, answer, cited), model=model or get_settings().judge_model)
    return result.output
