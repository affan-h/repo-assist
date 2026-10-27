Agent model: `google:gemini-3.5-flash-lite` · judge: `google:gemini-3.5-flash-lite` · embeddings: `BAAI/bge-small-en-v1.5`

| Metric | single_shot | agent |
|---|---|---|
| Questions | 42 | 42 |
| Answer correctness (all) | 56% | 83% |
| Correctness, answerable questions | 51% | 81% |
| Abstention accuracy (unanswerable) | 100% | 100% |
| False abstention rate (answerable) | 37% | 5% |
| Groundedness (claims supported by citations) | 94% | 97% |
| Answers with every claim supported | 92% | 94% |
| Gold evidence retrieved | 79% | 90% |
| Gold evidence cited | 62% | 82% |
| Latency p50 | 4.2s | 18.6s |
| Latency p95 | 10.5s | 96.3s |
| LLM calls / question | 1.0 | 6.3 |
| Tool calls / question | 0.0 | 5.2 |
| Tokens / question | 4,190 | 35,244 |
| Runs with errors | 0 | 2 |

**Correctness by category**

| Category | n | single_shot | agent |
|---|---|---|---|
| how | 10 | 42% | 91% |
| topology | 4 | 0% | 70% |
| unanswerable | 3 | 100% | 100% |
| what | 9 | 74% | 89% |
| where | 6 | 61% | 100% |
| why | 10 | 60% | 60% |
