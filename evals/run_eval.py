"""Run the agent and the single-shot baseline on a question split, then write the report.

    python evals/run_eval.py --split test            # held-out questions (headline results)
    python evals/run_eval.py --split dev             # questions used during development
    python evals/run_eval.py --split test --ids cY1  # a few questions, e.g. as a smoke test

Each question is answered by both systems; answers that don't decline are graded by
the judge (judge.py). Raw records go to evals/results/<split>/answers.jsonl and the
report (report.py) is rebuilt at the end. Every LLM call goes through a record/replay
cache, so an interrupted run (e.g. at the free tier's daily quota) resumes where it
stopped without paying again for finished work.
"""

import argparse
import asyncio
import json
import sys
import time
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic_ai.exceptions import ModelConnectionError, ModelHTTPError, ModelTimeoutError  # noqa: E402

from evals.common import load_questions, results_dir  # noqa: E402
from evals.judge import judge  # noqa: E402
from evals.llm_cache import CachedModel  # noqa: E402
from evals.report import write_report  # noqa: E402
from repo_assist import agent  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402
from repo_assist.retry import is_daily_quota, retry_delay  # noqa: E402

SYSTEMS = ("agent", "baseline")
MAX_ATTEMPTS = 8


class QuotaExhausted(RuntimeError):
    pass


async def with_retry(make_call):
    """Retry transient failures, honouring the server's suggested delay. Stop on daily-quota exhaustion."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await make_call()
        except (ModelConnectionError, ModelTimeoutError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"    [{type(exc).__name__}] retrying in {10 * attempt}s", flush=True)
            await asyncio.sleep(10 * attempt)
        except ModelHTTPError as exc:
            if is_daily_quota(exc):
                raise QuotaExhausted(str(exc.body or exc)[:300]) from exc
            delay = retry_delay(exc, attempt)
            if delay is None or attempt == MAX_ATTEMPTS:
                raise
            print(f"    [{exc.status_code}] backing off {delay:.0f}s (attempt {attempt})", flush=True)
            await asyncio.sleep(delay)


def cited_evidence(result: agent.Result) -> list[tuple[str, str]]:
    """The text behind each citation, as the judge sees it."""
    cited = []
    for c in result.citations:
        if c.kind in ("code", "doc"):
            cited.append((c.ref, c.snippet))
        else:
            cited.append((c.ref, result.evidence[c.ref].text[:6000] if c.ref in result.evidence else c.snippet))
    return cited


async def answer_and_grade(question: dict, system: str, models: dict) -> dict:
    result = await with_retry(lambda: agent.answer(question["repo"], question["question"], mode=system,
                                                   model=models["agent"]))
    cited = cited_evidence(result)
    judgement = None
    if not result.abstained:  # declining is scored from the flag; nothing to judge
        judgement = (await with_retry(lambda: judge(question, result.answer, cited, model=models["judge"]))).model_dump()
    return {
        "id": question["id"], "system": system, "question": question["question"],
        "answer": result.answer, "abstained": result.abstained, "error": result.error,
        "citations": [c.ref for c in result.citations], "quotes": result.quotes,
        "evidence_refs": list(result.evidence), "cited_evidence": [{"ref": r, "text": t} for r, t in cited],
        "trace": [{"tool": t.tool, "args": t.args, "summary": t.summary} for t in result.trace],
        "usage": result.usage, "latency_s": result.latency_s, "judgement": judgement,
    }


def load_answers(split: str) -> dict[tuple[str, str], dict]:
    path = results_dir(split) / "answers.jsonl"
    if not path.exists():
        return {}
    return {(r["id"], r["system"]): r for r in map(json.loads, path.read_text().splitlines())}


async def run(split: str, questions: list[dict]) -> None:
    settings = get_settings()
    cache = settings.data_dir / "llm_cache.db"
    models = {"agent": CachedModel(settings.agent_model, cache), "judge": CachedModel(settings.judge_model, cache)}
    out = results_dir(split) / "answers.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_answers(split)
    todo = [(q, s) for q in questions for s in SYSTEMS if (q["id"], s) not in done]
    print(f"{len(todo)} answers to produce ({split} split)", flush=True)
    for i, (question, system) in enumerate(todo, start=1):
        started = time.monotonic()
        try:
            record = await answer_and_grade(question, system, models)
        except QuotaExhausted as exc:
            print(f"Daily quota exhausted; stopping. Run again later to resume.\n{exc}")
            break
        except Exception as exc:  # e.g. a provider outage: skipped now, retried on the next run
            print(f"[{i}/{len(todo)}] {question['id']:<6} {system:<8} FAILED {type(exc).__name__}: {exc}"[:300], flush=True)
            continue
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{i}/{len(todo)}] {question['id']:<6} {system:<8} abstained={record['abstained']} "
              f"tools={len(record['trace'])} {time.monotonic() - started:.0f}s", flush=True)
    calls = sum(m.misses for m in models.values())
    print(f"LLM calls: {calls} new, {sum(m.hits for m in models.values())} replayed from cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--repo")
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()

    questions = load_questions(args.repo, args.ids, split=args.split)
    # Alternate repos so a run cut short by the quota still covers all of them.
    by_repo = [[q for q in questions if q["repo"] == r] for r in dict.fromkeys(q["repo"] for q in questions)]
    questions = [q for group in zip_longest(*by_repo) for q in group if q is not None]
    asyncio.run(run(args.split, questions))
    print("\n" + write_report(args.split))


if __name__ == "__main__":
    main()
