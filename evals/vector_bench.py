"""Exact search vs Qdrant's HNSW index: query latency and recall. No LLM calls.

    docker run -p 6333:6333 qdrant/qdrant:v1.19.2
    python evals/vector_bench.py --qdrant http://localhost:6333   # writes evals/results/vector_bench.{json,md}

Part 1, real vectors: every embedded chunk of the indexed repositories. Queries are
the evaluation questions plus chunks held out of the index. Recall@10 is the share
of the exact top 10 that a method returns (exact search defines the truth).

Part 2, scaling: random unit vectors at growing sizes, latency only. A matrix product
costs the same whatever the data, so timings are valid; recall on random vectors is
not representative of real embeddings and isn't reported.
"""

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from evals.common import RESULTS_DIR, load_questions  # noqa: E402
from repo_assist.config import get_settings  # noqa: E402
from repo_assist.embeddings import get_embedder  # noqa: E402

K = 10
HELD_OUT = 300
HNSW = models.HnswConfigDiff(m=16, ef_construct=100)
EF_VALUES = (16, 64, 128)
SCALES = (10_000, 50_000, 200_000)
SEED = 7


def real_vectors() -> np.ndarray:
    conn = sqlite3.connect(get_settings().db_path)
    rows = conn.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
    return np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])


def exact_top(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    scores = matrix @ query
    top = np.argpartition(-scores, K)[:K]
    return top[np.argsort(-scores[top])]


def timed(fn, queries) -> tuple[list, dict]:
    results, ms = [], []
    for q in queries:
        started = time.perf_counter()
        results.append(fn(q))
        ms.append((time.perf_counter() - started) * 1000)
    ms.sort()
    return results, {"p50_ms": statistics.median(ms), "p95_ms": ms[int(0.95 * (len(ms) - 1))]}


def recall(results: list, truth: list) -> float:
    return statistics.fmean(len(set(map(int, r)) & set(map(int, t))) / K for r, t in zip(results, truth, strict=True))


def build_collection(client: QdrantClient, name: str, matrix: np.ndarray) -> float:
    """Upload and wait until the HNSW graph covers every vector. Returns seconds taken."""
    if client.collection_exists(name):
        client.delete_collection(name)
    started = time.perf_counter()
    client.create_collection(
        name, vectors_config=models.VectorParams(size=matrix.shape[1], distance=models.Distance.DOT),
        hnsw_config=HNSW, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))  # always build HNSW
    client.upload_collection(name, vectors=matrix, ids=range(len(matrix)), batch_size=1024, wait=True)
    while True:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= len(matrix):
            return time.perf_counter() - started
        time.sleep(1)


def qdrant_search(client: QdrantClient, name: str, **params):
    def search(q):
        points = client.query_points(name, query=q.tolist(), limit=K, with_payload=False,
                                     search_params=models.SearchParams(**params)).points
        return [p.id for p in points]
    return search


def part_real(client: QdrantClient) -> dict:
    vectors = real_vectors()
    rng = np.random.default_rng(SEED)
    held = rng.choice(len(vectors), HELD_OUT, replace=False)
    corpus = np.delete(vectors, held, axis=0)
    embedder = get_embedder()
    questions = [np.asarray(embedder.embed_query(q["question"]), dtype=np.float32) for q in load_questions()]
    queries = questions + list(vectors[held])

    truth, numpy_lat = timed(lambda q: exact_top(corpus, q), queries)
    build_s = build_collection(client, "bench_real", corpus)
    rows = [{"method": "Exact, in-process (numpy)", "recall@10": 1.0, **numpy_lat}]
    results, lat = timed(qdrant_search(client, "bench_real", exact=True), queries)
    rows.append({"method": "Qdrant, exact (full scan)", "recall@10": recall(results, truth), **lat})
    for ef in EF_VALUES:
        results, lat = timed(qdrant_search(client, "bench_real", hnsw_ef=ef), queries)
        rows.append({"method": f"Qdrant, HNSW (ef={ef})", "recall@10": recall(results, truth), **lat})
    client.delete_collection("bench_real")
    return {"vectors": len(corpus), "dim": corpus.shape[1], "queries": len(queries),
            "questions": len(questions), "index_build_s": build_s, "rows": rows}


def part_scaling(client: QdrantClient, dim: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for n in SCALES:
        matrix = rng.standard_normal((n, dim), dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        noise = 0.01 * rng.standard_normal((100, dim), dtype=np.float32)  # float32, like real queries
        queries = list(matrix[rng.choice(n, 100, replace=False)] + noise)
        _, numpy_lat = timed(lambda q: exact_top(matrix, q), queries)
        build_s = build_collection(client, "bench_scale", matrix)
        _, hnsw_lat = timed(qdrant_search(client, "bench_scale", hnsw_ef=64), queries)
        client.delete_collection("bench_scale")
        out.append({"vectors": n, "numpy_p50_ms": numpy_lat["p50_ms"], "hnsw_p50_ms": hnsw_lat["p50_ms"],
                    "index_build_s": build_s})
        print(f"  {n:>7,} vectors: numpy {numpy_lat['p50_ms']:.1f} ms, HNSW {hnsw_lat['p50_ms']:.1f} ms", flush=True)
    return out


def render(result: dict) -> str:
    real = result["real"]
    lines = [f"**Real vectors:** {real['vectors']:,} embedded chunks ({real['dim']} dims) from the indexed repositories; "
             f"{real['queries']} queries ({real['questions']} evaluation questions, {HELD_OUT} held-out chunks). "
             f"HNSW m={HNSW.m}, ef_construct={HNSW.ef_construct}; index built in {real['index_build_s']:.0f}s.", "",
             "| Method | Recall@10 | Latency p50 | p95 |", "|---|---|---|---|"]
    for r in real["rows"]:
        lines.append(f"| {r['method']} | {r['recall@10']:.3f} | {r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms |")
    lines += ["", "**Scaling** (random unit vectors, same dimension; latency only):", "",
              "| Vectors | Exact, in-process p50 | Qdrant HNSW (ef=64) p50 | HNSW build |", "|---|---|---|---|"]
    for s in result["scaling"]:
        lines.append(f"| {s['vectors']:,} | {s['numpy_p50_ms']:.1f} ms | {s['hnsw_p50_ms']:.1f} ms | "
                     f"{s['index_build_s']:.0f}s |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant", default="http://localhost:6333")
    client = QdrantClient(url=parser.parse_args().qdrant, timeout=300)
    print("Part 1: real vectors", flush=True)
    real = part_real(client)
    print("Part 2: scaling", flush=True)
    result = {"real": real, "scaling": part_scaling(client, real["dim"])}
    (RESULTS_DIR / "vector_bench.json").write_text(json.dumps(result, indent=2))
    text = render(result)
    (RESULTS_DIR / "vector_bench.md").write_text(text)
    print(text)
