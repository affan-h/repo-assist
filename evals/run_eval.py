"""End-to-end evaluation of answering modes on the curated question set.

    python evals/run_eval.py --split test                          # baseline + agent on the test split
    python evals/run_eval.py --split test --modes agent agent_noindex
    python evals/run_eval.py --split dev --summary-only            # rebuild the report from saved runs

Methodology:
- Splits: `dev` questions were used while building the system; `test` questions were
  written before the final system ran on them and are the ones to report.
- Deterministic metrics come first (abstention flag, gold evidence retrieved/cited,
  required mentions, rationale source); an LLM judge only checks key-fact coverage and
  whether each claim is supported by the evidence the system cited.
- Every LLM call goes through a record/replay cache, so resumed or repeated runs are free
  and reproducible. Runs interleave modes per question so that a run cut short by a daily
  quota still compares modes on the same questions.
- Reports carry 95% bootstrap intervals and paired comparisons (exact McNemar test).
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
from evals.llm_cache import CachedModel  # noqa: E402
from evals.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff  # noqa: E402
from repo_assist import agent  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402

MODES = ("single_shot", "agent", "agent_bm25", "agent_noindex")
MAX_ATTEMPTS = 8
CORRECT_THRESHOLD = 0.5  # a question counts as answered correctly at >= half its key facts


class QuotaExhausted(RuntimeError):
    pass


async def with_retry(make_call):
    """Retry transient failures, honouring the server's suggested delay. Stop on daily-quota exhaustion."""
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


def deterministic_metrics(question: dict, result: agent.Result) -> dict:
    """Metrics computed without an LLM."""
    gold = gold_items(question)
    citations = [c.ref for c in result.citations]
    mentions = question.get("gold_mentions", [])
    answer = result.answer.lower()
    return {
        "abstained_as_expected": result.abstained == (question["expected"] == "abstain"),
        "gold_in_evidence": any_match(gold, list(result.evidence)) if gold else None,
        "gold_in_citations": any_match(gold, citations) if gold else None,
        "mention_recall": (sum(m.lower() in answer for m in mentions) / len(mentions)) if mentions else None,
        # For answered "why" questions: does a verbatim rationale quote come from a gold source?
        "rationale_from_gold": (any_match(gold, [q["ref"] for q in result.quotes]) if result.quotes and gold else None),
    }


async def evaluate_one(question: dict, mode: str, models: dict) -> dict:
    result = await with_retry(lambda: agent.answer(question["repo"], question["question"], mode=mode,
                                                   model=models["agent"]))
    cited = cited_evidence(result)
    judgement = await with_retry(lambda: judge(question, result.answer, result.abstained, cited, model=models["judge"]))
    return {
        "id": question["id"], "repo": question["repo"], "category": question["category"],
        "expected": question["expected"], "mode": mode, "question": question["question"],
        "answer": result.answer, "abstained": result.abstained, "error": result.error,
        "citations": [c.ref for c in result.citations], "quotes": result.quotes,
        "evidence_refs": list(result.evidence),
        "trace": [{"tool": t.tool, "args": t.args, "summary": t.summary} for t in result.trace],
        "usage": result.usage, "latency_s": result.latency_s,
        "cited_evidence": [{"ref": ref, "text": text} for ref, text in cited],
        "deterministic": deterministic_metrics(question, result),
        "judgement": judgement.model_dump(),
    }


def runs_dir(split: str) -> Path:
    return RESULTS_DIR / "v2" / split / "runs"


def load_runs(split: str, mode: str) -> dict[str, dict]:
    path = runs_dir(split) / f"{mode}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("judgement")}


async def run(split: str, modes: list[str], questions: list[dict], force: bool, use_cache: bool) -> None:
    """Question by question, all modes per question, so a quota cut leaves modes comparable."""
    settings = get_settings()
    cache_path = settings.data_dir / "llm_cache.db"
    models = {"agent": settings.agent_model, "judge": settings.judge_model}
    if use_cache:
        models = {k: CachedModel(v, cache_path) for k, v in models.items()}
    runs_dir(split).mkdir(parents=True, exist_ok=True)
    done = {mode: {} if force else load_runs(split, mode) for mode in modes}
    todo = [q for q in questions if any(q["id"] not in done[m] for m in modes)]
    print(f"{len(todo)} questions to run ({split} split, modes: {', '.join(modes)})", flush=True)
    for i, question in enumerate(todo, start=1):
        for mode in modes:
            if question["id"] in done[mode]:
                continue
            started = time.monotonic()
            try:
                record = await evaluate_one(question, mode, models)
            except QuotaExhausted as exc:
                print(f"Daily quota exhausted; stopping. Re-run later to resume.\n{exc}")
                return
            except Exception as exc:  # e.g. a provider outage: skip now, retried on the next run
                print(f"[{i}/{len(todo)}] {question['id']:<6} {mode:<13} FAILED {type(exc).__name__}: {exc}"[:300], flush=True)
                continue
            done[mode][question["id"]] = record
            with (runs_dir(split) / f"{mode}.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            m = score(question, Judgement(**record["judgement"]), record["abstained"])
            print(f"[{i}/{len(todo)}] {question['id']:<6} {mode:<13} correct={m['correct']:.2f} "
                  f"abstained={record['abstained']} tools={len(record['trace'])} "
                  f"{time.monotonic() - started:.0f}s", flush=True)
    if use_cache:
        print(f"LLM cache: {sum(m.hits for m in models.values())} hits, {sum(m.misses for m in models.values())} calls")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _pct(values, q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else None


def _with_ci(values: list[float]) -> dict | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    low, high = bootstrap_ci(values)
    return {"mean": statistics.fmean(values), "low": low, "high": high, "n": len(values)}


def aggregate(records: list[dict]) -> dict:
    answerable = [r for r in records if r["expected"] == "answer"]
    unanswerable = [r for r in records if r["expected"] == "abstain"]
    det = [r["deterministic"] for r in records]
    return {
        "n": len(records),
        "correctness": _with_ci([r["metrics"]["correct"] for r in records]),
        "correctness_answerable": _with_ci([r["metrics"]["correct"] for r in answerable]),
        "abstention_accuracy": _with_ci([float(r["metrics"]["abstained_correctly"]) for r in unanswerable]),
        "false_abstention_rate": _with_ci([float(r["metrics"]["false_abstention"]) for r in answerable]),
        "abstained_as_expected": _mean(float(d["abstained_as_expected"]) for d in det),
        "groundedness": _mean(r["metrics"]["groundedness"] for r in records),
        "gold_evidence_retrieved": _mean(float(d["gold_in_evidence"]) for d in det if d["gold_in_evidence"] is not None),
        "gold_evidence_cited": _mean(float(d["gold_in_citations"]) for d in det if d["gold_in_citations"] is not None),
        "mention_recall": _mean(d["mention_recall"] for d in det),
        "rationale_from_gold": _mean(float(d["rationale_from_gold"]) for d in det if d["rationale_from_gold"] is not None),
        "errors": sum(1 for r in records if r["error"]),
        "latency_p50_s": _pct([r["latency_s"] for r in records], 0.5),
        "latency_p95_s": _pct([r["latency_s"] for r in records], 0.95),
        "llm_calls_mean": _mean(r["usage"].get("requests") for r in records),
        "tool_calls_mean": _mean(len(r["trace"]) for r in records),
        "tokens_mean": _mean(r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0) for r in records),
    }


def summarise(split: str, questions: list[dict]) -> dict:
    by_id = {q["id"]: q for q in questions}
    runs = {mode: load_runs(split, mode) for mode in MODES}
    runs = {m: r for m, r in runs.items() if r}
    ids = set(by_id).intersection(*(set(r) for r in runs.values())) if runs else set()  # paired questions only
    for records in runs.values():  # re-score from saved judgements so rubric fixes apply retroactively
        for r in records.values():
            if r["id"] in by_id:
                q = by_id[r["id"]]
                r.update(category=q["category"], expected=q["expected"])
                r["metrics"] = score(q, Judgement(**r["judgement"]), r["abstained"])
    settings = get_settings()
    summary = {"split": split, "agent_model": settings.agent_model, "judge_model": settings.judge_model,
               "embedding_model": settings.embedding_model, "questions": len(ids), "modes": {}, "comparisons": {}}
    for mode, records in runs.items():
        selected = [records[i] for i in sorted(ids)]
        summary["modes"][mode] = {
            "overall": aggregate(selected),
            "by_category": {c: aggregate([r for r in selected if r["category"] == c])
                            for c in sorted({r["category"] for r in selected})},
        }
    if "agent" in runs:
        for other in runs:
            if other == "agent":
                continue
            a = [runs["agent"][i]["metrics"]["correct"] for i in sorted(ids)]
            b = [runs[other][i]["metrics"]["correct"] for i in sorted(ids)]
            diff, low, high = paired_bootstrap_diff(a, b)
            summary["comparisons"][f"agent vs {other}"] = {
                "correctness_diff": diff, "low": low, "high": high,
                **mcnemar_exact([x >= CORRECT_THRESHOLD for x in a], [x >= CORRECT_THRESHOLD for x in b]),
            }
    return summary


def _ci(stat: dict | None) -> str:
    if stat is None:
        return "–"
    return f"{stat['mean']:.0%} [{stat['low']:.0%}–{stat['high']:.0%}]"


def _plain(value, kind: str) -> str:
    if value is None:
        return "–"
    return {"ratio": f"{value:.0%}", "seconds": f"{value:.1f}s", "num": f"{value:.1f}", "int": f"{value:,.0f}"}[kind]


def render(summary: dict) -> str:
    modes = list(summary["modes"])
    out = [f"**{summary['split']} split**, {summary['questions']} questions answered by every mode. "
           f"Agent model `{summary['agent_model']}`, judge `{summary['judge_model']}`, embeddings "
           f"`{summary['embedding_model']}`. Brackets: 95% bootstrap interval.\n",
           "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    ci_rows = [("Answer correctness (judge)", "correctness"), ("… on answerable questions", "correctness_answerable"),
               ("Abstention accuracy (unanswerable)", "abstention_accuracy"),
               ("False abstention rate (answerable)", "false_abstention_rate")]
    for label, key in ci_rows:
        out.append(f"| {label} | " + " | ".join(_ci(summary["modes"][m]["overall"][key]) for m in modes) + " |")
    plain_rows = [("Abstained exactly when expected (flag)", "abstained_as_expected", "ratio"),
                  ("Gold evidence retrieved", "gold_evidence_retrieved", "ratio"),
                  ("Gold evidence cited", "gold_evidence_cited", "ratio"),
                  ("Required mentions present", "mention_recall", "ratio"),
                  ("Rationale quoted from a gold source", "rationale_from_gold", "ratio"),
                  ("Claims supported by cited evidence (judge)", "groundedness", "ratio"),
                  ("Latency p50", "latency_p50_s", "seconds"), ("Latency p95", "latency_p95_s", "seconds"),
                  ("LLM calls / question", "llm_calls_mean", "num"), ("Tool calls / question", "tool_calls_mean", "num"),
                  ("Tokens / question", "tokens_mean", "int"), ("Runs with errors", "errors", "int")]
    for label, key, kind in plain_rows:
        out.append(f"| {label} | " + " | ".join(_plain(summary["modes"][m]["overall"][key], kind) for m in modes) + " |")

    if summary["comparisons"]:
        out += ["", "**Paired comparisons** (same questions; correct = at least half the key facts)", "",
                "| Comparison | Correctness difference | Agent-only correct | Other-only correct | McNemar p |",
                "|---|---|---|---|---|"]
        for name, c in summary["comparisons"].items():
            out.append(f"| {name} | {c['correctness_diff']:+.0%} [{c['low']:+.0%}, {c['high']:+.0%}] | "
                       f"{c['a_only']} | {c['b_only']} | {c['p_value']:.3f} |")

    categories = sorted({c for m in modes for c in summary["modes"][m]["by_category"]})
    out += ["", "**Correctness by category**", "", "| Category | n | " + " | ".join(modes) + " |",
            "|---|---|" + "---|" * len(modes)]
    for c in categories:
        n = summary["modes"][modes[0]]["by_category"].get(c, {}).get("n", "")
        cells = [_plain((summary["modes"][m]["by_category"].get(c, {}).get("correctness") or {}).get("mean"), "ratio")
                 for m in modes]
        out.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["single_shot", "agent"])
    parser.add_argument("--repo")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--force", action="store_true", help="Re-run questions that already have results")
    parser.add_argument("--no-cache", action="store_true", help="Call the model even if a recorded response exists")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.repo, args.ids, split=args.split)
    # Alternate repos so a run cut short by a quota still covers all of them.
    by_repo = [[q for q in questions if q["repo"] == r] for r in dict.fromkeys(q["repo"] for q in questions)]
    questions = [q for group in zip_longest(*by_repo) for q in group if q is not None]
    if not args.summary_only:
        asyncio.run(run(args.split, args.modes, questions, args.force, not args.no_cache))
    summary = summarise(args.split, load_questions(split=args.split))
    out_dir = runs_dir(args.split).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(render(summary))
    print("\n" + render(summary))


if __name__ == "__main__":
    main()
