# Week 16 Agentic Loop Write-up

## Context Engineering Technique

I implemented **clearing tool results** in `prepare_retry` in `app/graph/ollama_agent.py`. When verification rejects a RAG answer, the node removes previous raw RAG `ToolMessage` chunks and keeps only a short insufficiency summary such as the iteration number and missing evidence. This prevents duplicate or stale chunks from growing the LLM context on every retry while preserving the reason the next retrieval is needed.

## Agentic Pattern

This is a **SINGLE-AGENT LOOP**: one LangGraph graph uses one LLM role for answer/tool selection and a verification prompt in `verify_answer`. I did not split the task into multiple agents because the task is too small for separate context isolation, parallelization, or specialization to pay off. One graph avoids a sequential bottleneck and multi-agent coordination overhead.

## Evaluation Harness

`eval_harness.py` runs nine queries covering factual RAG, a retry-oriented RAG question, calculator calls, missing knowledge, ambiguity, and a no-tool request. It records task completion, tool-call correctness, trajectory length, failure class, and total tokens. The complete per-query table and aggregate row are in [`eval_report.md`](eval_harness/eval_report.md). The current recorded run has a `0.0%` completion rate, average trajectory `0.00`, and total tokens `81`, because Ollama was unavailable locally; these are the actual generated results.

## Skill vs. Agent

Verification could have been a fixed Skill/prompt template, but I made it a node because it must decide whether control flow loops or stops.

## Token and Cost Accounting

The `total_tokens` column in [`eval_report.md`](eval_harness/eval_report.md) uses provider usage metadata when available and a small estimate otherwise. There is no multi-agent baseline because the design is intentionally single-agent.

## Failure Injection Test

The `--inject-failure` mode replaces `rag_search` output with malformed text for the no-KB test query. The separate [`failure_injection_report.md`](failure_injection_report.md) documents the injection, expected `verify_answer` behavior, before/after output, and the observed limitation: the local provider outage prevented the verifier from running, so no successful retry is claimed.

## Tool vs. Agent Boundary

`rag_search` and `calculator` are bounded tool calls, not agents. Each performs one request/response operation and has no internal state, independent planning, or multi-turn reasoning. The LangGraph LLM decides when to call them, while `verify_answer` decides whether the overall answer is sufficient and whether another tool round is needed.
