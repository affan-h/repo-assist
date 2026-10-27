"""Check the judge against a second, independent free model.

    python evals/judge_audit.py --split test

The main judge (judge.py) decides, for every answer that didn't decline, which of the
reference's key facts it states. Here a different model re-makes each of those
decisions with a deliberately simple prompt (one YES/NO line per fact), and the two are
compared: raw agreement and Cohen's kappa over individual key-fact decisions, plus a
list of every disagreement for inspection. High agreement between two independently
trained models doesn't prove the grades are right, but low agreement would show the
correctness numbers can't be trusted.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai import Agent  # noqa: E402

from evals.common import load_questions, results_dir  # noqa: E402
from evals.llm_cache import CachedModel  # noqa: E402
from evals.stats import cohens_kappa  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402

AUDIT_MODEL = os.environ.get("REPO_ASSIST_AUDIT_MODEL", "google:gemma-4-26b-a4b-it")

# No system instructions: Gemma models on the Gemini API don't accept them.
SECOND_JUDGE = Agent(None, output_type=str)

PROMPT = """Question: {question}

System answer:
\"\"\"{answer}\"\"\"

For each key fact below, does the system answer state it (paraphrase is fine, vague hints are not)?
Reply with exactly one line per fact, in order, each line only YES or NO.
{facts}"""


def parse_verdicts(text: str, n: int) -> list[bool] | None:
    verdicts = re.findall(r"\b(YES|NO)\b", text.upper())
    return [v == "YES" for v in verdicts] if len(verdicts) == n else None


async def second_opinion(question: dict, record: dict, model) -> list[bool] | None:
    facts = [f["fact"] for f in record["judgement"]["key_facts"]]
    prompt = PROMPT.format(question=question["question"], answer=record["answer"],
                           facts="\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, start=1)))
    result = await SECOND_JUDGE.run(prompt, model=model)
    return parse_verdicts(result.output, len(facts))


async def audit(split: str, model=None) -> dict:
    questions = {q["id"]: q for q in load_questions(split=split)}
    path = results_dir(split) / "answers.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    judged = [r for r in records if r.get("judgement") and r["id"] in questions and r["judgement"]["key_facts"]]
    model = model or CachedModel(AUDIT_MODEL, get_settings().data_dir / "llm_cache.db")

    first, second, disagreements, unparsed = [], [], [], 0
    for record in judged:
        verdicts = await second_opinion(questions[record["id"]], record, model)
        if verdicts is None:
            unparsed += 1
            continue
        for fact, verdict in zip(record["judgement"]["key_facts"], verdicts, strict=True):
            first.append(fact["covered"])
            second.append(verdict)
            if fact["covered"] != verdict:
                disagreements.append({"id": record["id"], "system": record["system"], "fact": fact["fact"],
                                      "judge": fact["covered"], "second_judge": verdict})
    result = {
        "judge_model": get_settings().judge_model, "second_judge_model": str(getattr(model, "model_name", model)),
        "answers": len(judged) - unparsed, "unparsed": unparsed, "decisions": len(first),
        "agreement": sum(a == b for a, b in zip(first, second)) / len(first) if first else None,
        "kappa": cohens_kappa(first, second) if first else None,
        "judge_covered_rate": sum(first) / len(first) if first else None,
        "second_judge_covered_rate": sum(second) / len(second) if second else None,
        "disagreements": disagreements,
    }
    return result


def render(result: dict) -> str:
    if not result["decisions"]:
        return "No judged answers to audit yet.\n"
    lines = [f"Judge `{result['judge_model']}` vs second judge `{result['second_judge_model']}` on "
             f"{result['decisions']} key-fact decisions from {result['answers']} answers: agreement "
             f"{result['agreement']:.0%}, Cohen's kappa {result['kappa']:.2f}. 'Covered' rate: judge "
             f"{result['judge_covered_rate']:.0%}, second judge {result['second_judge_covered_rate']:.0%}."
             + (f" ({result['unparsed']} answers skipped: second judge reply not parseable.)" if result["unparsed"] else ""),
             "", "| Question | System | Key fact | Judge | Second judge |", "|---|---|---|---|---|"]
    for d in result["disagreements"]:
        lines.append(f"| {d['id']} | {d['system']} | {d['fact']} | {'yes' if d['judge'] else 'no'} | "
                     f"{'yes' if d['second_judge'] else 'no'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    split = parser.parse_args().split
    result = asyncio.run(audit(split))
    (results_dir(split) / "judge_audit.json").write_text(json.dumps(result, indent=2))
    text = render(result)
    (results_dir(split) / "judge_audit.md").write_text(text)
    print(text)
