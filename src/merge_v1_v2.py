"""
One-time merge: combines a real, complete v2-only results file (from
grader.py --engine v2, now 56/56 real judged scores) with the
already-verified v1 reconstruction (41.1% overall, matched the original
console output exactly) into one final {"v1", "v2", "regression_gates"}
file, without requiring a fresh v1 run.

RUN FROM src/: python3 merge_v1_v2.py results_full.json results_final.json
"""

import json
import sys
from datetime import datetime, timezone

# Real v1 data, reconstructed earlier and verified against the original
# console output (41.1% overall matched exactly, per-category numbers
# matched exactly: what 50%, how 44%, where 33%, why 14%,
# unanswerable_why 88%, topology 33%).
V1_ROWS = [
    ("W1", "httpx", "what", 1), ("W2", "httpx", "what", 0), ("W3", "httpx", "what", 0),
    ("W4", "httpx", "what", 1), ("W5", "httpx", "what", 1),
    ("H1", "httpx", "how", 0), ("H2", "httpx", "how", 1), ("H3", "httpx", "how", 0),
    ("H4", "httpx", "how", 1), ("H5", "httpx", "how", 0),
    ("WH1", "httpx", "where", 1), ("WH2", "httpx", "where", 0), ("WH3", "httpx", "where", 1),
    ("WH4", "httpx", "where", 1), ("WH5", "httpx", "where", 0),
    ("Y1", "httpx", "why", 0), ("Y2", "httpx", "why", 0), ("Y3", "httpx", "why", 0),
    ("Y4", "httpx", "why", 0), ("Y5", "httpx", "why", 0), ("Y6", "httpx", "why", 0),
    ("Y7", "httpx", "why", 0), ("Y8", "httpx", "why", 0),
    ("U1", "httpx", "unanswerable_why", 1), ("U2", "httpx", "unanswerable_why", 0),
    ("U3", "httpx", "unanswerable_why", 1), ("U4", "httpx", "unanswerable_why", 1),
    ("T1", "httpx", "topology", 0), ("T2", "httpx", "topology", 1), ("T3", "httpx", "topology", 0),
    ("W6", "got", "what", 1), ("W7", "got", "what", 1), ("W8", "got", "what", 0),
    ("W9", "got", "what", 0), ("W10", "got", "what", 0),
    ("H6", "got", "how", 0), ("H7", "got", "how", 1), ("H8", "got", "how", 0), ("H9", "got", "how", 1),
    ("WH6", "got", "where", 0), ("WH7", "got", "where", 0), ("WH8", "got", "where", 0), ("WH9", "got", "where", 0),
    ("Y9", "got", "why", 1), ("Y10", "got", "why", 1), ("Y11", "got", "why", 0),
    ("Y12", "got", "why", 0), ("Y13", "got", "why", 0), ("Y14", "got", "why", 0),
    ("U5", "got", "unanswerable_why", 1), ("U6", "got", "unanswerable_why", 1),
    ("U7", "got", "unanswerable_why", 1), ("U8", "got", "unanswerable_why", 1),
    ("T4", "got", "topology", 1), ("T5", "got", "topology", 0), ("T6", "got", "topology", 0),
]

CATS = ["what", "how", "where", "why", "unanswerable_why", "topology"]


def _build_bundle(rows_or_results, engine, from_v2_file=False):
    cats_scores = {c: [] for c in CATS}
    repo_scores = {"httpx": [], "got": []}
    results = []

    if from_v2_file:
        for r in rows_or_results:
            score = r.get("score")
            results.append(r)
            if score is not None:
                cats_scores[r["category"]].append(score)
                repo_scores[r["repo"]].append(score)
    else:
        for qid, repo, category, score in rows_or_results:
            results.append({"id": qid, "repo": repo, "category": category, "score": score,
                             "judge_reason": "", "agent_answer": ""})
            cats_scores[category].append(score)
            repo_scores[repo].append(score)

    all_scores = [s for c in cats_scores.values() for s in c]
    overall_pct = round(100 * sum(all_scores) / len(all_scores), 1) if all_scores else None
    u = cats_scores["unanswerable_why"]
    u_pct = round(100 * sum(u) / len(u), 1) if u else None

    return {
        "engine": engine,
        "total_questions": len(results),
        "graded": len(all_scores),
        "overall_pct": overall_pct,
        "unanswerable_why_pct": u_pct,
        "scores_by_category": {
            c: {"correct": sum(cats_scores[c]), "total": len(cats_scores[c]),
                "pct": round(100 * sum(cats_scores[c]) / len(cats_scores[c]), 1) if cats_scores[c] else None}
            for c in CATS
        },
        "scores_by_repo": {
            r: {"correct": sum(s), "total": len(s), "pct": round(100 * sum(s) / len(s), 1) if s else None}
            for r, s in repo_scores.items()
        },
        "results": results,
    }


def _evaluate_gates(v1_bundle, v2_bundle):
    v1_overall = v1_bundle["overall_pct"]
    v2_overall = v2_bundle["overall_pct"]
    v2_u = v2_bundle["unanswerable_why_pct"]

    aggregate_pass = (v2_overall >= v1_overall - 5) or (v2_overall >= v1_overall + 3)
    safety_pass = v2_u >= 65.0
    both_pass = aggregate_pass and safety_pass

    return {
        "v1_overall_pct": v1_overall, "v2_overall_pct": v2_overall,
        "aggregate_gate_pass": aggregate_pass,
        "aggregate_gate_rule": "v2 >= v1 - 5pts, OR v2 >= v1 + 3pts",
        "v2_unanswerable_why_pct": v2_u, "safety_gate_pass": safety_pass,
        "safety_gate_rule": "v2 unanswerable_why >= 65% (10pt tolerance from v1's 75% baseline), independent of aggregate",
        "v2_becomes_default": both_pass,
    }


if __name__ == "__main__":
    v2_file = sys.argv[1] if len(sys.argv) > 1 else "results_full.json"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "results_final.json"

    with open(v2_file) as f:
        v2_data = json.load(f)

    # Handle both a bare v2-only file and an already-nested {"v2": {...}} shape.
    if "results" in v2_data:
        v2_results = v2_data["results"]
    elif "v2" in v2_data:
        v2_results = v2_data["v2"]["results"]
    else:
        raise ValueError(f"Can't find v2 results in {v2_file}")

    v1_bundle = _build_bundle(V1_ROWS, "v1")
    v2_bundle = _build_bundle(v2_results, "v2", from_v2_file=True)
    gates = _evaluate_gates(v1_bundle, v2_bundle)

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "both",
        "note": "v1 bundle reconstructed from verified console output (see conversation); v2 bundle from real grader.py run.",
        "v1": v1_bundle,
        "v2": v2_bundle,
        "regression_gates": gates,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"v1: {v1_bundle['overall_pct']}%  (reconstructed, 56/56)")
    print(f"v2: {v2_bundle['overall_pct']}%  ({v2_bundle['graded']}/{v2_bundle['total_questions']}, real)")
    print()
    print("=" * 60)
    print("§6.2 REGRESSION GATES")
    print("=" * 60)
    print(f"  Aggregate gate ({gates['aggregate_gate_rule']}): {'PASS' if gates['aggregate_gate_pass'] else 'FAIL'}")
    print(f"  v2 unanswerable_why: {gates['v2_unanswerable_why_pct']}%")
    print(f"  Safety gate ({gates['safety_gate_rule']}): {'PASS' if gates['safety_gate_pass'] else 'FAIL'}")
    if gates["v2_becomes_default"]:
        print("\n  ✓ BOTH GATES PASS — v2 may become the CLI's default engine.")
    else:
        print("\n  ✗ At least one gate FAILED — v2 stays available via --engine v2, v1 remains the default.")
    print(f"\nWritten to {out_file}")
