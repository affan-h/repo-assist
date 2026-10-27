"""Retrieval ablation: BM25 vs dense vs hybrid (RRF) on the eval questions. No LLM calls.

For each question with gold evidence, the question text is used as the query
against the source/doc index (code gold) or the history index (history gold),
and we record the rank of the first chunk that overlaps the gold evidence.

    python evals/retrieval_eval.py --split test     # writes evals/results/test/retrieval.{json,md}
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.common import gold_items, load_questions, results_dir  # noqa: E402
from repo_assist import retrieval, store  # noqa: E402

MODES = ("bm25", "dense", "hybrid")
DEPTH = 20


def first_relevant_rank(question: dict, mode: str, conn) -> int | None:
    gold = gold_items(question)
    ranks = []
    for history, group in ((False, [g for g in gold if not g.is_history]), (True, [g for g in gold if g.is_history])):
        if not group:
            continue
        if history:  # same passage-aware search the agent's search_history tool uses
            hits = retrieval.search_history(question["repo"], question["question"], k=DEPTH, mode=mode, conn=conn)
        else:
            hits = retrieval.search(question["repo"], question["question"], retrieval.CODE_KINDS, k=DEPTH,
                                    mode=mode, conn=conn)
        for rank, hit in enumerate(hits, start=1):
            if any(g.matches(hit.ref) for g in group):
                ranks.append(rank)
                break
    return min(ranks) if ranks else None


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "questions": n,
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
        "mrr@10": statistics.fmean(1 / r if r is not None and r <= 10 else 0 for r in ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--repo")
    args = parser.parse_args()

    questions = [q for q in load_questions(args.repo, split=args.split) if q.get("gold_evidence")]
    rows = []
    with store.session() as conn:
        for q in questions:
            row = {"id": q["id"], "repo": q["repo"], "category": q["category"],
                   "history": any(g.is_history for g in gold_items(q))}
            for mode in MODES:
                row[mode] = first_relevant_rank(q, mode, conn)
            rows.append(row)
            print(f"{q['id']:<12} " + "  ".join(f"{m}={row[m]}" for m in MODES), flush=True)

    subsets = {"all": rows, "code_and_docs": [r for r in rows if not r["history"]],
               "history": [r for r in rows if r["history"]]}
    summary = {name: {m: summarise([r[m] for r in subset]) for m in MODES}
               for name, subset in subsets.items() if subset}
    out_dir = results_dir(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval.json").write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    (out_dir / "retrieval.md").write_text(render(summary))
    print("\n" + render(summary))


def render(summary: dict) -> str:
    lines = ["| Subset | Retriever | n | Recall@1 | Recall@5 | Recall@10 | MRR@10 |", "|---|---|---|---|---|---|---|"]
    for subset, by_mode in summary.items():
        for mode, s in by_mode.items():
            lines.append(f"| {subset} | {mode} | {s['questions']} | {s['recall@1']:.2f} | {s['recall@5']:.2f} | "
                         f"{s['recall@10']:.2f} | {s['mrr@10']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
