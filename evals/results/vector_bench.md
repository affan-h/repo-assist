**Real vectors:** 13,251 embedded chunks (384 dims) from the indexed repositories; 393 queries (93 evaluation questions, 300 held-out chunks). HNSW m=16, ef_construct=100; index built in 15s.

| Method | Recall@10 | Latency p50 | p95 |
|---|---|---|---|
| Exact, in-process (numpy) | 1.000 | 1.9 ms | 2.5 ms |
| Qdrant, exact (full scan) | 0.993 | 14.2 ms | 16.8 ms |
| Qdrant, HNSW (ef=16) | 0.940 | 10.6 ms | 12.6 ms |
| Qdrant, HNSW (ef=64) | 0.986 | 12.0 ms | 14.9 ms |
| Qdrant, HNSW (ef=128) | 0.990 | 11.2 ms | 13.2 ms |

**Scaling** (random unit vectors, same dimension; latency only):

| Vectors | Exact, in-process p50 | Qdrant HNSW (ef=64) p50 | HNSW build |
|---|---|---|---|
| 10,000 | 1.4 ms | 12.3 ms | 20s |
| 50,000 | 6.4 ms | 12.0 ms | 136s |
| 200,000 | 28.5 ms | 13.7 ms | 692s |
