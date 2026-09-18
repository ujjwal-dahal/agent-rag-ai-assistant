# Agentic Loop Evaluation Report

| Query                                              |           Task completion | Tool-call correctness |            Trajectory length |  Total tokens | Failure class |
| -------------------------------------------------- | ------------------------: | --------------------: | ---------------------------: | ------------: | ------------- |
| What is Retrieval-Augmented Generation?            |                     False |                 False |                            0 |             9 | hard_failure  |
| What is an embedding in this project?              |                     False |                 False |                            0 |             9 | hard_failure  |
| Which part of the system stores retrieved vectors? |                     False |                 False |                            0 |            12 | hard_failure  |
| Calculate 125 \* 8 + 10.                           |                     False |                 False |                            0 |             5 | hard_failure  |
| What is the square root of 144?                    |                     False |                 False |                            0 |             7 | hard_failure  |
| What is the history of the Roman Empire?           |                     False |                 False |                            0 |            10 | hard_failure  |
| Tell me about the model.                           |                     False |                 False |                            0 |             6 | hard_failure  |
| Give me one short tip for writing clear prompts.   |                     False |                  True |                            0 |            12 | hard_failure  |
| What does Ollama provide in this application?      |                     False |                 False |                            0 |            11 | hard_failure  |
| **Aggregate**                                      | **completion rate: 0.0%** |                       | **average trajectory: 0.00** | **total: 81** |               |

## Failure Injection

With `--inject-failure`, the local run could not reach `verify_answer` because the configured Ollama endpoint was unavailable. The implementation guard for malformed or empty RAG evidence is documented in `failure_injection_report.md`.

Token totals use LLM usage metadata when available. When a provider does not return usage metadata, the harness uses a small character-based estimate.
