"""Measure how often the LLM judge agrees with an independent reviewer.

    python evals/judge_audit.py sample --split test --n 30   # writes judge_audit.json to fill in
    python evals/judge_audit.py score --split test           # agreement once `review` fields are filled

The sample is drawn at random (fixed seed) from judged answers. For each key fact the
reviewer records whether the answer states it, without looking at the judge's verdict
(the file keeps the judge's verdict in a separate field). Agreement is reported as raw
agreement and Cohen's kappa over individual key-fact decisions.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.common import load_questions, results_dir  # noqa: E402
from evals.stats import cohens_kappa  # noqa: E402

SEED = 11


def sample(split: str, n: int) -> Path:
    questions = {q["id"]: q for q in load_questions(split=split)}
    records = [json.loads(line) for line in (results_dir(split) / "answers.jsonl").read_text().splitlines()]
    judged = [r for r in records if r.get("judgement") and r["id"] in questions]
    picked = random.Random(SEED).sample(judged, min(n, len(judged)))
    items = [{
        "id": r["id"], "system": r["system"], "question": questions[r["id"]]["question"],
        "reference_answer": questions[r["id"]]["reference_answer"], "answer": r["answer"],
        "key_facts": [f["fact"] for f in r["judgement"]["key_facts"]],
        "judge": [f["covered"] for f in r["judgement"]["key_facts"]],
        "review": None,  # fill with one true/false per key fact
    } for r in picked]
    path = results_dir(split) / "judge_audit.json"
    path.write_text(json.dumps({"reviewer": None, "items": items}, indent=2))
    return path


def score(split: str) -> str:
    data = json.loads((results_dir(split) / "judge_audit.json").read_text())
    judge, review = [], []
    for item in data["items"]:
        if item["review"] is None:
            continue
        judge += item["judge"]
        review += item["review"]
    if not review:
        return "No reviewed items yet."
    agreement = sum(a == b for a, b in zip(judge, review)) / len(review)
    text = (f"Judge vs reviewer ({data['reviewer']}) on {len(review)} key-fact decisions from "
            f"{sum(i['review'] is not None for i in data['items'])} answers: agreement {agreement:.0%}, "
            f"Cohen's kappa {cohens_kappa(judge, review):.2f}. Judge said 'covered' {sum(judge) / len(judge):.0%} "
            f"of the time, the reviewer {sum(review) / len(review):.0%}.")
    (results_dir(split) / "judge_audit.md").write_text(text + "\n")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["sample", "score"])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    print(sample(args.split, args.n) if args.action == "sample" else score(args.split))
