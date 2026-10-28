Judge `google:gemini-3.5-flash-lite` vs second judge `gemma-4-26b-a4b-it` on 116 key-fact decisions from 49 answers: agreement 91%, Cohen's kappa 0.51. 'Covered' rate: judge 92%, second judge 86%.

Correctness under each judge (same answers): agent 75% (judge) vs 71% (second judge); baseline 56% (judge) vs 52% (second judge).

| Question | System | Key fact | Judge | Second judge |
|---|---|---|---|---|
| gY2 | agent | The rename makes clear the option does not do any URL resolution | no | yes |
| hY2 | baseline | The flow is: create request, apply auth, follow redirects, send | yes | no |
| hY6 | agent | So the same auth class works with both sync and async clients | no | yes |
| cY6 | agent | They belong in a router enhancer outside the core library (chi.Router is an interface that can be wrapped) | yes | no |
| hY7 | agent | Use a single scoped or global client instead | yes | no |
| hY7 | baseline | Use a single scoped or global client instead | yes | no |
| cC1 | baseline | It returns an empty string if there is no routing context or no such key | yes | no |
| cC3 | agent | With returns a new inline mux sharing the same tree | yes | no |
| cC4 | agent | It resets the wildcard URLParam before calling the sub-router | yes | no |
| cC7 | agent | It sets a context deadline with context.WithTimeout | yes | no |
| cC7 | baseline | It sets a context deadline with context.WithTimeout | yes | no |
