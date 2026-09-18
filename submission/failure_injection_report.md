# Failure Injection Report

## Injection

The harness flag `--inject-failure` monkeypatches `rag_search` for the specific query `What is the history of the Roman Empire?`. The injected tool returns the malformed string:

```text
{malformed retrieval output
```

This simulates malformed or unusable retrieval output from the knowledge-base tool.

## Expected Agent Behavior

`verify_answer` has an explicit RAG guard for empty or unusable evidence. It should mark the result insufficient, provide a refined query, clear prior RAG `ToolMessage` chunks before retrying, and continue until the three-iteration cap. If verification still fails, `run_agent` returns `verified: false` and appends:

```text
Answer could not be fully verified against sources.
```

It should not confidently treat the malformed text as factual evidence.

## Observed Run

The command was run as requested:

```text
python eval_harness.py --inject-failure
```

The current local run could not reach the graph's `verify_answer` node because the configured Ollama endpoint was unavailable. The generated evaluation report therefore records the affected cases as `hard_failure` with no completed trajectory, rather than inventing a successful retry or hallucinated answer.

### Before

```json
{
  "query": "What is the history of the Roman Empire?",
  "tool_output": "{malformed retrieval output"
}
```

### After: expected with providers available

```json
{
  "answer": "Answer could not be fully verified against sources.",
  "source_used": true,
  "tool_used": "rag_search",
  "verified": false,
  "iterations_used": 3
}
```

The exact post-injection values above are the implementation's hard-cap behavior, not a fabricated live result. The actual live aggregate and failure classification are recorded in [`eval_report.md`](eval_harness/eval_report.md).
