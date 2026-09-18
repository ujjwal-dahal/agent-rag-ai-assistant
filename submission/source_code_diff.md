# Source Code Diff Summary

This list covers every tracked or untracked project file currently changed since the Week 15 baseline commit (`0f75c1c`, `COMPLETED : TASK 2 completed`). The `submission/` packaging files are listed separately below.

| File path                       | Change                                                                                                                                                                        |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                     | Added agentic loop architecture, context engineering, single-agent rationale, evaluation harness, failure injection, token accounting, and tool/agent boundary documentation. |
| `app/api/main.py`               | Added the structured `POST /chat` endpoint that invokes the agentic graph.                                                                                                    |
| `app/graph/ollama_agent.py`     | Added `verify_answer`, structured verification state, bounded retry routing, RAG message clearing, iteration/token tracking, and `run_agent`.                                 |
| `app/schemas/response.py`       | Extended `AssistantResponse` with `verified` and `iterations_used`.                                                                                                           |
| `app/tools/rag_tool.py`         | Current worktree change to the RAG tool configuration; retained as a source change for traceability.                                                                          |
| `data/chroma_db/chroma.sqlite3` | Current Chroma database binary changed during local retrieval/evaluation activity.                                                                                            |
| `eval_harness.py`               | Added the plain-Python nine-query evaluation harness, token accounting, failure classes, and `--inject-failure`.                                                              |
| `eval_report.md`                | Added generated per-query evaluation results, aggregate metrics, and failure-injection notes.                                                                                 |

## Submission Packaging Files

| File path                                 | Change                                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------------------------- |
| `submission/README.md`                    | Copy of the current project README.                                                    |
| `submission/architecture_diagram.md`      | Self-contained full pipeline and single-agent verification-loop diagram.               |
| `submission/eval_harness/eval_harness.py` | Copy of the current evaluation harness source.                                         |
| `submission/eval_harness/eval_report.md`  | Copy of the current generated evaluation report.                                       |
| `submission/failure_injection_report.md`  | Standalone injection, expected behavior, observed limitation, and before/after report. |
| `submission/writeup.md`                   | Concise implementation-specific Week 16 write-up.                                      |
| `submission/source_code_diff.md`          | This file.                                                                             |
