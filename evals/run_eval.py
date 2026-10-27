"""End-to-end evaluation: agent vs single-shot RAG on the curated question set.

    python evals/run_eval.py                       # both modes, all questions (resumes)
    python evals/run_eval.py --modes agent --ids httpx-what-1
    python evals/run_eval.py --summary-only        # rebuild summary from saved runs

Each answer is graded by an LLM judge (see judge.py). Results are appended to
evals/results/runs/<mode>.jsonl as they complete, so an interrupted run (e.g. a
rate-limit wall) resumes where it stopped.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai.exceptions import ModelConnectionError, ModelHTTPError, ModelTimeoutError  # noqa: E402

from evals.common import RESULTS_DIR, any_match, gold_items, load_questions  # noqa: E402
from evals.judge import Judgement, judge, score  # noqa: E402
from repo_assist import agent  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402

RUNS_DIR = RESULTS_DIR / "runs"
MODES = ("single_shot", "agent")
MAX_ATTEMPTS = 8


class QuotaExhausted(RuntimeError):
    pass


async def with_rate_limit_retry(make_call):
    """Retry on 429/5xx, honouring the server's suggested delay. Stop on daily-quota exhaustion."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await make_call()
        except (ModelConnectionError, ModelTimeoutError) as exc:  # network blips
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"    [{type(exc).__name__}] retrying in {10 * attempt}s", flush=True)
            await asyncio.sleep(10 * attempt)
        except ModelHTTPError as exc:
            body = str(exc.body or exc)
            if exc.status_code == 429 and "PerDay" in body:
                raise QuotaExhausted(body[:300]) from exc
            if exc.status_code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise
            match = re.search(r"retry in ([\d.]+)s", body) or re.search(r"'retryDelay': '(\d+)s'", body)
            delay = float(match.group(1)) + 1 if match else min(15 * attempt, 90)
            print(f"    [{exc.status_code}] backing off {delay:.0f}s (attempt {attempt})", flush=True)
            await asyncio.sleep(delay)


def cited_evidence(result: agent.Result) -> list[tuple[str, str]]:
    cited = []
    for c in result.citations:
        if c.kind in ("code", "doc"):
            cited.append((c.ref, c.snippet))
        else:
            cited.append((c.ref, result.evidence[c.ref].text[:6000] if c.ref in result.evidence else c.snippet))
    return cited


async def evaluate_one(question: dict, mode: str) -> dict:
    result = await with_rate_limit_retry(lambda: agent.answer(question["repo"], question["question"], mode=mode))
    cited = cited_evidence(result)
    judgement = await with_rate_limit_retry(lambda: judge(question, result.answer, result.abstained, cited))
    gold = gold_items(question)
    record = {
        "id": question["id"], "repo": question["repo"], "category": question["category"],
        "expected": question["expected"], "mode": mode, "question": question["question"],
        "answer": result.answer, "abstained": result.abstained, "error": result.error,
        "citations": [c.ref for c in result.citations],
        "evidence_refs": list(result.evidence),
        "gold_in_evidence": any_match(gold, list(result.evidence)) if gold else None,
        "gold_in_citations": any_match(gold, [c.ref for c in result.citations]) if gold else None,
        "trace": [{"tool": t.tool, "args": t.args, "summary": t.summary} for t in result.trace],
        "usage": result.usage, "latency_s": result.latency_s,
        "cited_evidence": [{"ref": ref, "text": text} for ref, text in cited],
        "judgement": judgement.model_dump(),
    }
    record["metrics"] = score(question, judgement, result.abstained)
    return record


def load_runs(mode: str) -> dict[str, dict]:
    path = RUNS_DIR / f"{mode}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("judgement")}


async def run(modes: list[str], questions: list[dict], force: bool) -> None:
    """Run question by question, all modes per question, so that an interrupted run
    (e.g. a daily quota) still leaves every mode evaluated on the same questions."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = {mode: {} if force else load_runs(mode) for mode in modes}
    todo = [q for q in questions if any(q["id"] not in done[m] for m in modes)]
    print(f"{len(todo)} questions to run", flush=True)
    for i, question in enumerate(todo, start=1):
        for mode in modes:
            if question["id"] in done[mode]:
                continue
            started = time.monotonic()
            try:
                record = await evaluate_one(question, mode)
            except QuotaExhausted as exc:
                print(f"Daily quota exhausted; stopping. Re-run later to resume.\n{exc}")
                return
            except Exception as exc:  # e.g. a provider outage: skip now, retried on the next run
                print(f"[{i}/{len(todo)}] {question['id']:<5} {mode:<11} FAILED {type(exc).__name__}: {exc}"[:300], flush=True)
                continue
            done[mode][question["id"]] = record
            with (RUNS_DIR / f"{mode}.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            m = record["metrics"]
            grounded = m["groundedness"] if m["groundedness"] is None else round(m["groundedness"], 2)
            print(f"[{i}/{len(todo)}] {question['id']:<5} {mode:<11} correct={m['correct']:.2f} "
                  f"grounded={grounded} abstained={record['abstained']} tools={len(record['trace'])} "
                  f"{time.monotonic() - started:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _pct(values, q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def aggregate(records: list[dict]) -> dict:
    answerable = [r for r in records if r["expected"] == "answer"]
    unanswerable = [r for r in records if r["expected"] == "abstain"]
    return {
        "n": len(records),
        "correctness": _mean(r["metrics"]["correct"] for r in records),
        "correctness_answerable": _mean(r["metrics"]["correct"] for r in answerable),
        "abstention_accuracy": _mean(float(r["metrics"]["abstained_correctly"]) for r in unanswerable),
        "false_abstention_rate": _mean(float(r["metrics"]["false_abstention"]) for r in answerable),
        "groundedness": _mean(r["metrics"]["groundedness"] for r in records),
        "fully_grounded_rate": _mean(float(r["metrics"]["groundedness"] == 1.0) for r in records
                                     if r["metrics"]["groundedness"] is not None),
        "gold_evidence_retrieved": _mean(float(r["gold_in_evidence"]) for r in records if r["gold_in_evidence"] is not None),
        "gold_evidence_cited": _mean(float(r["gold_in_citations"]) for r in records if r["gold_in_citations"] is not None),
        "errors": sum(1 for r in records if r["error"]),
        "latency_p50_s": _pct([r["latency_s"] for r in records], 0.5),
        "latency_p95_s": _pct([r["latency_s"] for r in records], 0.95),
        "llm_calls_mean": _mean(r["usage"].get("requests") for r in records),
        "tool_calls_mean": _mean(len(r["trace"]) for r in records),
        "tokens_mean": _mean(r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0) for r in records),
    }


def summarise(questions: list[dict]) -> dict:
    # Only compare modes on questions every mode has completed.
    runs = {mode: load_runs(mode) for mode in MODES}
    by_id = {q["id"]: q for q in questions}
    for records in runs.values():  # re-score from saved judgements so rubric fixes apply retroactively
        for r in records.values():
            if r["id"] in by_id:
                q = by_id[r["id"]]
                r.update(category=q["category"], expected=q["expected"])
                r["metrics"] = score(q, Judgement(**r["judgement"]), r["abstained"])
    ids = {q["id"] for q in questions}.intersection(*(set(r) for r in runs.values() if r))
    summary = {"agent_model": get_settings().agent_model, "judge_model": get_settings().judge_model,
               "embedding_model": get_settings().embedding_model, "modes": {}}
    for mode in MODES:
        records = [r for r in runs[mode].values() if r["id"] in ids]
        if not records:
            continue
        categories = sorted({r["category"] for r in records})
        summary["modes"][mode] = {
            "overall": aggregate(records),
            "by_category": {c: aggregate([r for r in records if r["category"] == c]) for c in categories},
            "by_repo": {repo: aggregate([r for r in records if r["repo"] == repo])
                        for repo in sorted({r["repo"] for r in records})},
        }
    return summary


def _fmt(value, kind: str = "ratio") -> str:
    if value is None:
        return "–"
    if kind == "ratio":
        return f"{value:.0%}"
    if kind == "seconds":
        return f"{value:.1f}s"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def render(summary: dict) -> str:
    modes = list(summary["modes"])
    rows = [
        ("Questions", "n", "int"),
        ("Answer correctness (all)", "correctness", "ratio"),
        ("Correctness, answerable questions", "correctness_answerable", "ratio"),
        ("Abstention accuracy (unanswerable)", "abstention_accuracy", "ratio"),
        ("False abstention rate (answerable)", "false_abstention_rate", "ratio"),
        ("Groundedness (claims supported by citations)", "groundedness", "ratio"),
        ("Answers with every claim supported", "fully_grounded_rate", "ratio"),
        ("Gold evidence retrieved", "gold_evidence_retrieved", "ratio"),
        ("Gold evidence cited", "gold_evidence_cited", "ratio"),
        ("Latency p50", "latency_p50_s", "seconds"),
        ("Latency p95", "latency_p95_s", "seconds"),
        ("LLM calls / question", "llm_calls_mean", "num"),
        ("Tool calls / question", "tool_calls_mean", "num"),
        ("Tokens / question", "tokens_mean", "int"),
        ("Runs with errors", "errors", "int"),
    ]
    out = [f"Agent model: `{summary['agent_model']}` · judge: `{summary['judge_model']}` · "
           f"embeddings: `{summary['embedding_model']}`\n",
           "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for label, key, kind in rows:
        cells = []
        for m in modes:
            v = summary["modes"][m]["overall"][key]
            cells.append(f"{v:,.0f}" if kind == "int" and v is not None else _fmt(v, kind))
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    out += ["", "**Correctness by category**", "",
            "| Category | n | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    categories = sorted({c for m in modes for c in summary["modes"][m]["by_category"]})
    for c in categories:
        n = summary["modes"][modes[0]]["by_category"].get(c, {}).get("n", "")
        cells = [_fmt(summary["modes"][m]["by_category"].get(c, {}).get("correctness")) for m in modes]
        out.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repo")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--force", action="store_true", help="Re-run questions that already have results")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.repo, args.ids)
    # Alternate repos so a run cut short by a quota still covers both.
    by_repo = [[q for q in questions if q["repo"] == r] for r in dict.fromkeys(q["repo"] for q in questions)]
    questions = [q for group in zip_longest(*by_repo) for q in group if q is not None]
    if not args.summary_only:
        asyncio.run(run(args.modes, questions, args.force))
    summary = summarise(load_questions())
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.md").write_text(render(summary))
    print("\n" + render(summary))


if __name__ == "__main__":
    main()
