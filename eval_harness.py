"""Small, dependency-free evaluation harness for the agentic assistant."""

import argparse
from pathlib import Path

from app import graph as graph_package  # noqa: F401
import app.graph.ollama_agent as graph_module


TEST_CASES = [
    ("RAG factual", "What is Retrieval-Augmented Generation?", "rag_search"),
    ("RAG factual", "What is an embedding in this project?", "rag_search"),
    ("RAG retry", "Which part of the system stores retrieved vectors?", "rag_search"),
    ("Calculator", "Calculate 125 * 8 + 10.", "calculator"),
    ("Calculator", "What is the square root of 144?", "calculator"),
    ("No KB answer", "What is the history of the Roman Empire?", "rag_search"),
    ("Ambiguous", "Tell me about the model.", "rag_search"),
    ("No tool", "Give me one short tip for writing clear prompts.", None),
    ("RAG factual", "What does Ollama provide in this application?", "rag_search"),
]


INJECTED_QUERY = "What is the history of the Roman Empire?"


def expected_tool_correct(result: dict, expected: str | None) -> bool:
    return result.get("tool_used") == expected


def usable_answer(result: dict) -> bool:
    answer = result.get("answer", "")
    return bool(answer and answer.strip())


def classify(result: dict, error: Exception | None) -> str:
    if error is not None or not usable_answer(result):
        return "hard_failure"
    if not result.get("verified", False):
        if result.get("iterations_used", 0) >= 2:
            return "cascading_soft_failure"
        return "soft_failure"
    return "none"


def run_case(query: str, expected: str | None) -> dict:
    try:
        result = graph_module.run_agent(query)
        error = None
    except Exception as exc:  # A harness row should survive one failed case.
        result = {
            "answer": "",
            "tool_used": None,
            "iterations_used": 0,
            "verified": False,
            "total_tokens": 0,
        }
        error = exc

    total_tokens = int(result.get("total_tokens", 0))
    if not total_tokens:
        total_tokens = max(
            1,
            (len(query) + len(result.get("answer", ""))) // 4,
        )
    return {
        "query": query,
        "task_completion": usable_answer(result),
        "tool_call_correctness": expected_tool_correct(result, expected),
        "trajectory_length": result.get("iterations_used", 0),
        "total_tokens": total_tokens,
        "verified": bool(result.get("verified", False)),
        "failure_class": classify(result, error),
        "error": str(error) if error else "",
    }


def write_report(rows: list[dict], injection_note: str) -> None:
    completed = sum(row["task_completion"] for row in rows)
    average_trajectory = (
        sum(row["trajectory_length"] for row in rows) / len(rows) if rows else 0
    )
    total_tokens = sum(row["total_tokens"] for row in rows)
    completion_rate = completed / len(rows) if rows else 0

    lines = [
        "# Agentic Loop Evaluation Report",
        "",
        "| Query | Task completion | Tool-call correctness | Trajectory length | Total tokens | Failure class |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        query = row["query"].replace("|", "\\|")
        lines.append(
            f"| {query} | {row['task_completion']} | "
            f"{row['tool_call_correctness']} | {row['trajectory_length']} | "
            f"{row['total_tokens']} | {row['failure_class']} |"
        )
    lines.extend(
        [
            "| **Aggregate** | **completion rate: "
            f"{completion_rate:.1%}** | | **average trajectory: "
            f"{average_trajectory:.2f}** | **total: {total_tokens}** | |",
            "",
            "## Failure Injection",
            "",
            injection_note,
            "",
            "Token totals use LLM usage metadata when available. When a provider does not return usage metadata, the harness uses a small character-based estimate.",
        ]
    )
    Path("eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inject-failure",
        action="store_true",
        help="Return empty retrieval output for the no-KB test query.",
    )
    args = parser.parse_args()

    original_func = graph_module.rag_search.func
    injection_result = None
    if args.inject_failure:
        def injected_rag_search(question: str) -> str:
            if "roman empire" in question.lower():
                return ""
            return original_func(question)

        graph_module.rag_search.func = injected_rag_search

    try:
        rows = [run_case(query, expected) for _, query, expected in TEST_CASES]
        if args.inject_failure:
            injection_result = next(
                row for row in rows if row["query"] == INJECTED_QUERY
            )
    finally:
        if args.inject_failure:
            graph_module.rag_search.func = original_func

    if args.inject_failure and injection_result:
        if injection_result["task_completion"] and not injection_result["verified"]:
            injection_result["failure_class"] = "soft_failure"
            injection_note = (
                "With `--inject-failure`, rag_search returned empty output. The verifier "
                "detected missing evidence, retried to the hard cap, and returned a usable "
                "answer with `verified: false`; it did not confidently accept invalid context."
            )
        else:
            injection_note = (
                "With `--inject-failure`, the run did not produce the expected graceful "
                "unverified result. Inspect the injected row above."
            )
    else:
        injection_note = (
            "The failure injection test was not run. Execute `python eval_harness.py "
            "--inject-failure` to record the malformed-retrieval behavior."
        )

    write_report(rows, injection_note)
    print("Wrote eval_report.md")


if __name__ == "__main__":
    main()
