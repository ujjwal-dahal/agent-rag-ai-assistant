"""Run real prompt experiments, trace the agent, and write comparison artifacts."""

import argparse
import json
import os
import statistics
from pathlib import Path

import mlflow

from app.config.prompt_versions import PROMPT_CONFIGS
from app.graph.ollama_agent import run_agent_versioned
from mlops.evidently_groq import GROQ_JUDGE_MODEL, run_regression_suite


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "mlruns_artifacts"
TEST_CASES = [
    {"id": "rag_definition", "query": "What is Retrieval-Augmented Generation?", "expected_tool": "rag_search"},
    {"id": "calculator", "query": "Calculate 125 * 8 + 10.", "expected_tool": "calculator"},
    {"id": "missing_knowledge", "query": "What is the history of the Roman Empire?", "expected_tool": "rag_search"},
    {"id": "no_tool", "query": "Give me one short tip for writing clear prompts.", "expected_tool": None},
]


def run_case(case: dict, version: str) -> dict:
    try:
        result = run_agent_versioned(case["query"], version)
        return {
            **case,
            "answer": result.get("answer", ""),
            "tool_used": result.get("tool_used"),
            "verified": bool(result.get("verified")),
            "iterations": result.get("iterations_used", 0),
            "tokens": result.get("total_tokens", 0),
            "termination_reason": result.get("termination_reason", "error"),
            "trace": result.get("trace", []),
            "error": "",
        }
    except Exception as exc:
        return {**case, "answer": "", "tool_used": None, "verified": False,
                "iterations": 0, "tokens": 0, "termination_reason": "error",
                "trace": [], "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--experiment", default="w17-agentic-assistant")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    comparison = []

    for version, config in PROMPT_CONFIGS.items():
        run_dir = ARTIFACT_ROOT / version
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = [run_case(case, version) for case in TEST_CASES]
        (run_dir / "traces.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (run_dir / "prompt.txt").write_text(config.prompt, encoding="utf-8")
        golden_by_id = {row["id"]: row for row in json.loads((ARTIFACT_ROOT / "golden_responses.json").read_text(encoding="utf-8"))} if (ARTIFACT_ROOT / "golden_responses.json").exists() else {}
        for row in rows:
            row["reference_answer"] = golden_by_id.get(row["id"], {}).get("answer", "")
        with mlflow.start_run(run_name=version, description=f"Trace-driven experiment for {version} judged by Groq {GROQ_JUDGE_MODEL}"):
            mlflow.log_params({"prompt_version": version, "top_k": config.top_k,
                               "chunk_size": config.chunk_size, "temperature": config.temperature,
                               "max_iterations": config.max_iterations})
            completed = sum(bool(row["answer"]) for row in rows)
            verified = sum(row["verified"] for row in rows)
            mlflow.log_metrics({"task_success_rate": completed / len(rows),
                                "verified_rate": verified / len(rows),
                                "avg_iterations": statistics.mean(row["iterations"] for row in rows),
                                "avg_tool_calls": statistics.mean(sum(step.get("kind") == "tool" for step in row["trace"]) for row in rows),
                                "avg_tokens": statistics.mean(row["tokens"] for row in rows)})
            regression = run_regression_suite(rows, version, run_dir)
            pct_tests_passed = regression["pct_tests_passed"]
            mlflow.log_artifacts(str(run_dir))
            mlflow.log_metric("pct_tests_passed", pct_tests_passed)
        comparison.append({"version": version, "task_success_rate": completed / len(rows),
                           "avg_iterations": statistics.mean(row["iterations"] for row in rows),
                           "avg_tokens": statistics.mean(row["tokens"] for row in rows),
                           "pct_tests_passed": pct_tests_passed})

    (ARTIFACT_ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    best = max(comparison, key=lambda row: (row["task_success_rate"], -row["avg_tokens"]))
    best_rows = [row for row in [run_case(case, best["version"]) for case in TEST_CASES] if row["answer"]]
    (ARTIFACT_ROOT / "golden_responses.json").write_text(json.dumps(best_rows, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()