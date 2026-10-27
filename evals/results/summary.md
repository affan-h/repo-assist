Agent model: `google:gemini-3.5-flash-lite` · judge: `google:gemini-3.5-flash-lite` · embeddings: `BAAI/bge-small-en-v1.5`

| Metric | single_shot | agent |
|---|---|---|
| Questions | 51 | 51 |
| Answer correctness (all) | 55% | 80% |
| Correctness, answerable questions | 46% | 86% |
| Abstention accuracy (unanswerable) | 100% | 50% |
| False abstention rate (answerable) | 40% | 9% |
| Groundedness (claims supported by citations) | 95% | 96% |
| Answers with every claim supported | 92% | 89% |
| Gold evidence retrieved | 74% | 93% |
| Gold evidence cited | 59% | 91% |
| Latency p50 | 3.9s | 31.1s |
| Latency p95 | 13.0s | 147.2s |
| LLM calls / question | 1.0 | 7.1 |
| Tool calls / question | 0.0 | 5.9 |
| Tokens / question | 4,089 | 39,958 |
| Runs with errors | 0 | 3 |

**Correctness by category**

| Category | n | single_shot | agent |
|---|---|---|---|
| how | 10 | 42% | 86% |
| topology | 6 | 22% | 97% |
| unanswerable | 8 | 100% | 50% |
| what | 10 | 77% | 100% |
| where | 9 | 48% | 96% |
| why | 8 | 29% | 50% |
