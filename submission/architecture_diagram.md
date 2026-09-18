# Architecture Diagram

This submission uses one LangGraph agent boundary. Groq is the primary LLM configured by the project, with Ollama/qwen2.5:1.5b available through the existing application fallback path.

```text
USER
  |
  v
Streamlit UI or POST /chat
  |
  v
FastAPI
  |
  v
+----------------------------------------------------------------+
| SINGLE-AGENT LANGGRAPH BOUNDARY                               |
|                                                                |
|  START -> LLM answer/tool-selection node                       |
|              |                                                 |
|              +-- rag_search --> Chroma vector store            |
|              |                   |                             |
|              |                   +--> retrieved context         |
|              |                                                 |
|              +-- calculator --> calculation result              |
|              |                                                 |
|              +-- no tool --> draft answer                       |
|                                                                |
|  tool result -> record_tool_results -> LLM draft answer         |
|                                      |                         |
|                                      v                         |
|                                verify_answer                    |
|                                      |                         |
|                         +------------+------------+             |
|                         |                         |             |
|                    sufficient                insufficient       |
|                         |                         |             |
|                         v                         v             |
|                    FINAL JSON             iterations < 3?       |
|                                                   |             |
|                                      +------------+------------+ |
|                                      |                         | |
|                                     yes                        no|
|                                      |                         | |
|                                      v                         v |
|                           prepare_retry / clear RAG     FINAL JSON|
|                           chunks and keep summary       verified: false
|                                      |                 + note     |
|                                      +-------> LLM      |
|                                                                |
|  MAX_ITERATIONS = 3                                            |
+----------------------------------------------------------------+
  |
  v
{answer, source_used, tool_used, verified, iterations_used}
```

The verifier is a control-flow node, not a second agent. There is no multi-agent coordination path: the same LLM role produces an answer and then evaluates it with a verification prompt.
