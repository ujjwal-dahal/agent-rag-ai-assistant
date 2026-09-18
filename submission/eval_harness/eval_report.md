# Agentic Loop Evaluation Report

| Query | Task completion | Tool-call correctness | Trajectory length | Total tokens | Failure class |
|---|---:|---:|---:|---:|---|
| What is Retrieval-Augmented Generation? | True | True | 3 | 5527 | cascading_soft_failure |
| What is an embedding in this project? | True | True | 3 | 4206 | cascading_soft_failure |
| Which part of the system stores retrieved vectors? | True | True | 3 | 4392 | cascading_soft_failure |
| Calculate 125 * 8 + 10. | True | True | 3 | 4226 | cascading_soft_failure |
| What is the square root of 144? | True | True | 3 | 3796 | cascading_soft_failure |
| What is the history of the Roman Empire? | True | True | 3 | 4943 | cascading_soft_failure |
| Tell me about the model. | True | True | 1 | 2094 | none |
| Give me one short tip for writing clear prompts. | True | True | 3 | 3620 | cascading_soft_failure |
| What does Ollama provide in this application? | True | True | 3 | 5398 | cascading_soft_failure |
| **Aggregate** | **completion rate: 100.0%** | | **average trajectory: 2.78** | **total: 38202** | |

## Failure Injection

The failure injection test was not run. Execute `python eval_harness.py --inject-failure` to record the malformed-retrieval behavior.

Token totals use LLM usage metadata when available. When a provider does not return usage metadata, the harness uses a small character-based estimate.
