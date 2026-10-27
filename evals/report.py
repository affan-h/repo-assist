"""Score saved answers and write the evaluation report for a split.

    python evals/report.py --split test

Metrics, per system, over the questions both systems answered:

  Correctness        answerable question: share of the reference's key facts the answer states
                     (0 if it declined or contradicts the reference; judged).
                     Unanswerable question: 1 if the system declined, else 0 (exact, from the flag).
  Correct declines   share of unanswerable questions the system declined (exact)
  False declines     share of answerable questions the system declined (exact)
  Gold evidence cited  the answer cites a source overlapping the gold evidence (exact)
  Rationale from gold source  "why" answers whose verbatim quote comes from a gold source (exact)
  Claims supported   share of the answer's claims backed by the evidence it cited (judged)

Correctness carries a 95% bootstrap interval, and the agent is compared to the baseline on
the same questions (paired bootstrap difference, exact McNemar test on solved/unsolved).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.common import any_match, gold_items, load_questions, results_dir  # noqa: E402
from evals.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402

SOLVED = 0.5  # a question counts as solved when at least half its key facts are stated


def score(question: dict, record: dict) -> dict:
    gold = gold_items(question)
    answerable = question["expected"] == "answer"
    judgement = record.get("judgement")
    if not answerable:
        correct = 1.0 if record["abstained"] else 0.0
    elif record["abstained"] or judgement is None:
        correct = 0.0
    elif judgement["contradicts_reference"]:
        correct = 0.0
    else:
        facts = judgement["key_facts"]
        correct = sum(f["covered"] for f in facts) / len(facts) if facts else 0.0
    claims = (judgement or {}).get("claims") or []
    quotes = [q["ref"] for q in record.get("quotes", [])]
    return {
        "correct": correct,
        "answerable": answerable,
        "declined": record["abstained"],
        "gold_cited": any_match(gold, record["citations"]) if gold and answerable else None,
        "rationale_from_gold": any_match(gold, quotes) if quotes and gold else None,
        "claims_supported": sum(c["supported"] for c in claims) / len(claims) if claims else None,
    }


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _ci(values) -> dict | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    low, high = bootstrap_ci(values)
    return {"mean": statistics.fmean(values), "low": low, "high": high, "n": len(values)}


def _percentile(values, q: float) -> float | None:
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else None


def aggregate(scored: list[tuple[dict, dict]]) -> dict:
    """`scored` is a list of (score, record) pairs."""
    s = [x for x, _ in scored]
    records = [r for _, r in scored]
    return {
        "n": len(s),
        "correctness": _ci([x["correct"] for x in s]),
        "correctness_answerable": _ci([x["correct"] for x in s if x["answerable"]]),
        "correct_declines": _mean([x["declined"] for x in s if not x["answerable"]]),
        "false_declines": _mean([x["declined"] for x in s if x["answerable"]]),
        "gold_cited": _mean([x["gold_cited"] for x in s]),
        "rationale_from_gold": _mean([x["rationale_from_gold"] for x in s]),
        "claims_supported": _mean([x["claims_supported"] for x in s]),
        "latency_p50": _percentile([r["latency_s"] for r in records], 0.5),
        "latency_p95": _percentile([r["latency_s"] for r in records], 0.95),
        "llm_calls": _mean([r["usage"].get("requests") for r in records]),
        "tokens": _mean([r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0) for r in records]),
        "errors": sum(1 for r in records if r["error"]),
    }


def build_report(split: str) -> dict:
    questions = {q["id"]: q for q in load_questions(split=split)}
    path = results_dir(split) / "answers.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    by_system: dict[str, dict[str, dict]] = {}
    for r in records:
        if r["id"] in questions:
            by_system.setdefault(r["system"], {})[r["id"]] = r  # later records win (re-runs)
    systems = [s for s in ("agent", "baseline") if s in by_system]
    ids = sorted(set(questions).intersection(*(set(by_system[s]) for s in systems))) if systems else []

    settings = get_settings()
    report = {"split": split, "questions": len(ids), "total_questions": len(questions),
              "agent_model": settings.agent_model, "judge_model": settings.judge_model, "systems": {}, "comparison": None}
    scores = {s: {i: score(questions[i], by_system[s][i]) for i in ids} for s in systems}
    for s in systems:
        pairs = [(scores[s][i], by_system[s][i]) for i in ids]
        categories = sorted({questions[i]["category"] for i in ids})
        report["systems"][s] = {
            "overall": aggregate(pairs),
            "by_category": {c: aggregate([p for p, i in zip(pairs, ids) if questions[i]["category"] == c]) for c in categories},
        }
    if len(systems) == 2 and ids:
        a = [scores["agent"][i]["correct"] for i in ids]
        b = [scores["baseline"][i]["correct"] for i in ids]
        diff, low, high = paired_bootstrap_diff(a, b)
        report["comparison"] = {"correctness_diff": diff, "low": low, "high": high,
                                **mcnemar_exact([x >= SOLVED for x in a], [x >= SOLVED for x in b])}
    report["per_question"] = {i: {s: scores[s][i]["correct"] for s in systems} for i in ids}
    return report


def _fmt(value, kind="pct") -> str:
    if value is None:
        return "–"
    if isinstance(value, dict):
        return f"{value['mean']:.0%} [{value['low']:.0%}–{value['high']:.0%}]"
    return {"pct": f"{value:.0%}", "s": f"{value:.1f}s", "num": f"{value:.1f}", "int": f"{value:,.0f}"}[kind]


def render(report: dict) -> str:
    systems = list(report["systems"])
    if not systems:
        return f"No answers yet for the {report['split']} split.\n"
    head = (f"**{report['split']} split**: {report['questions']} of {report['total_questions']} questions answered by "
            f"both systems. Agent model `{report['agent_model']}`, judge `{report['judge_model']}`. "
            f"Brackets: 95% bootstrap interval.\n")
    rows = [("Correctness", "correctness", "pct"), ("Correctness, answerable questions", "correctness_answerable", "pct"),
            ("Correct declines (unanswerable)", "correct_declines", "pct"),
            ("False declines (answerable)", "false_declines", "pct"),
            ("Gold evidence cited", "gold_cited", "pct"),
            ("Why: rationale quoted from a gold source", "rationale_from_gold", "pct"),
            ("Claims supported by cited evidence", "claims_supported", "pct"),
            ("Latency p50", "latency_p50", "s"), ("Latency p95", "latency_p95", "s"),
            ("LLM calls per question", "llm_calls", "num"), ("Tokens per question", "tokens", "int"),
            ("Runs with errors", "errors", "int")]
    out = [head, "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for label, key, kind in rows:
        out.append(f"| {label} | " + " | ".join(_fmt(report["systems"][s]["overall"][key], kind) for s in systems) + " |")
    c = report["comparison"]
    if c:
        out += ["", f"**Agent vs baseline, same questions:** correctness {c['correctness_diff']:+.0%} "
                    f"[{c['low']:+.0%}, {c['high']:+.0%}]; solved by the agent only: {c['a_only']}, by the baseline "
                    f"only: {c['b_only']} (exact McNemar p = {c['p_value']:.3f})."]
    categories = sorted({c for s in systems for c in report["systems"][s]["by_category"]})
    out += ["", "| Category | n | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for cat in categories:
        n = report["systems"][systems[0]]["by_category"][cat]["n"]
        cells = [_fmt((report["systems"][s]["by_category"][cat]["correctness"] or {}).get("mean")) for s in systems]
        out.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_report(split: str) -> str:
    report = build_report(split)
    out = results_dir(split)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    text = render(report)
    (out / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    print(write_report(parser.parse_args().split))
