"""Evidently LLM-judge integration backed by Groq."""

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import pandas as pd
from groq import Groq

from app.config.config import env_file
from evidently.legacy.features.llm_judge import BinaryClassificationPromptTemplate
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests.base_test import Test, TestResult, TestStatus
from evidently._pydantic_compat import Field
from evidently.llm.models import LLMMessage
from evidently.llm.utils.errors import LLMRateLimitError, LLMRequestError
from evidently.llm.utils.wrapper import LLMResult, LLMWrapper, llm_provider


GROQ_JUDGE_MODEL = "qwen/qwen3.8-27b"


@llm_provider("groq", None)
class GroqWrapper(LLMWrapper):
    """Adapt Groq chat completions to Evidently's LLMWrapper contract."""

    def __init__(self, model: str, options):
        self.model = model
        self.client = Groq(api_key=env_file.MODEL_API_KEY)

    async def complete(self, messages: list[LLMMessage], seed: int | None = None) -> LLMResult[str]:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=0,
                seed=seed,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "rate" in str(exc).lower():
                raise LLMRateLimitError(str(exc)) from exc
            raise LLMRequestError(f"Groq judge request failed: {exc}", original_error=exc) from exc
        content = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResult(
            content,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


CORRECTNESS_TEMPLATE = BinaryClassificationPromptTemplate(
    criteria=(
        "Does the candidate answer preserve the material information in the approved reference "
        "answer and avoid contradicting it? Mark incorrect if it invents a conflicting fact."
    ),
    target_category="correct",
    non_target_category="incorrect",
    include_reasoning=True,
)

TOOL_TEMPLATE = BinaryClassificationPromptTemplate(
    criteria=(
        "Did the assistant use the expected tool for this query? The expected tool may be "
        "calculator, rag_search, or none. Mark incorrect when the selected tool differs."
    ),
    target_category="correct",
    non_target_category="incorrect",
    include_reasoning=True,
)


def _judge(template: BinaryClassificationPromptTemplate, payload: str) -> dict:
    request = next(
        template.iterate_messages(
            pd.DataFrame([{"input": payload}]),
            {"input": "input"},
        )
    )
    wrapper = GroqWrapper(GROQ_JUDGE_MODEL, None)
    return wrapper.run_sync(request)


class GroqJudgeTest(Test):
    """A native Evidently Test whose cases are judged by Groq."""

    class Config:
        type_alias = "evidently:test:GroqJudgeTest"

    name: ClassVar[str] = "Groq LLM judge"
    group: ClassVar[str] = "LLM regression"
    kind: str = "reference_correctness"
    threshold: float = 0.75
    details: list[dict] = Field(default_factory=list)

    def __init__(self, kind: str, threshold: float = 0.75):
        super().__init__(kind=kind, threshold=threshold, details=[])
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "details", [])

    def groups(self):
        return {"test_group": "LLM regression", "test_type": self.kind}

    def check(self) -> TestResult:
        if self._context is None or self._context.data is None:
            return TestResult(
                name=self.name,
                description="No Evidently current dataset was provided.",
                status=TestStatus.ERROR,
                group=self.group,
                parameters=None,
            )
        template = CORRECTNESS_TEMPLATE if self.kind == "reference_correctness" else TOOL_TEMPLATE
        self.details.clear()
        for row in self._context.data.current_data.to_dict("records"):
            if self.kind == "reference_correctness":
                payload = (
                    f"Question: {row['query']}\nApproved reference answer: {row['reference_answer']}\n"
                    f"Candidate answer: {row['answer']}"
                )
            else:
                expected_tool = row["expected_tool"] if pd.notna(row["expected_tool"]) else "none"
                actual_tool = row["tool_used"] if pd.notna(row["tool_used"]) else "none"
                payload = (
                    f"Question: {row['query']}\nExpected tool: {expected_tool}\n"
                    f"Actual tool: {actual_tool}"
                )
            verdict = _judge(template, payload)
            self.details.append({"id": row["id"], "verdict": verdict})
        passed = sum(item["verdict"].get("category") == "correct" for item in self.details)
        pct = passed / len(self.details) if self.details else 0.0
        status = TestStatus.SUCCESS if pct >= self.threshold else TestStatus.FAIL
        return TestResult(
            name=self.name,
            description=f"{self.kind}: {passed}/{len(self.details)} cases passed ({pct:.1%}); threshold={self.threshold:.1%}",
            status=status,
            group=self.group,
            parameters=None,
        )


def run_regression_suite(rows: list[dict], version: str, output_dir: Path) -> dict:
    current = pd.DataFrame(rows)
    suite_tests = [
        GroqJudgeTest("reference_correctness"),
        GroqJudgeTest("tool_appropriateness"),
    ]
    suite = TestSuite(suite_tests, name=f"Groq regression {version}")
    suite.run(reference_data=None, current_data=current)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite.save_html(str(output_dir / "evidently_report.html"))
    summary = suite.as_dict()
    details = {test.kind: test.details for test in suite_tests}
    report = {"status": "passed" if bool(suite) else "failed", "version": version,
              "summary": summary["summary"], "details": details}
    (output_dir / "evidently_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = sum(len(test.details) for test in suite_tests)
    passed = sum(
        item["verdict"].get("category") == "correct"
        for test in suite_tests
        for item in test.details
    )
    report["pct_tests_passed"] = passed / total if total else 0.0
    return report