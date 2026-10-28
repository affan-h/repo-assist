**test split**: 37 of 37 questions answered by both systems. Agent model `google:gemini-3.5-flash-lite`, judge `google:gemini-3.5-flash-lite`. Brackets: 95% bootstrap interval.

| Metric | agent | baseline |
|---|---|---|
| Correctness | 75% [62%–88%] | 56% [41%–72%] |
| Correctness, answerable questions | 74% [59%–87%] | 54% [38%–70%] |
| Correct declines (unanswerable) | 100% | 100% |
| False declines (answerable) | 17% | 43% |
| Gold evidence cited | 57% | 57% |
| Why: rationale quoted from a gold source | 48% | 87% |
| Claims supported by cited evidence | 98% | 95% |
| Latency p50 | 7.6s | 2.0s |
| Latency p95 | 19.7s | 5.8s |
| LLM calls per question | 5.5 | 1.1 |
| Tokens per question | 26,379 | 3,778 |
| Runs with errors | 1 | 0 |

**Agent vs baseline, same questions:** correctness +19% [+3%, +35%]; solved by the agent only: 9, by the baseline only: 2 (exact McNemar p = 0.065).

| Category | n | agent | baseline |
|---|---|---|---|
| how | 5 | 80% | 40% |
| topology | 1 | 100% | 0% |
| unanswerable | 2 | 100% | 100% |
| what | 3 | 89% | 67% |
| where | 1 | 100% | 100% |
| why | 25 | 69% | 55% |
