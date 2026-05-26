from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evals.run_eval import (
    AgentOutput,
    EvalCase,
    EvalResult,
    JudgeDecision,
    MockEvalAgent,
    compute_metrics,
    evaluate_cases,
    load_cases,
    write_case_details,
    write_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLICIT_CASES_PATH = PROJECT_ROOT / "evals" / "implicit_cases.jsonl"


class NoMCPAgent:
    registered_tools = ("multiply", "shell_exec")

    def run_case(self, case):
        raise AssertionError("MCP case should be skipped before agent execution")


class StaticAgent:
    def __init__(
        self,
        *,
        output: str,
        tools: tuple[str, ...] = (),
        registered_tools: tuple[str, ...] = (),
    ) -> None:
        self.output = output
        self.tools = tools
        self.registered_tools = registered_tools or tools

    def run_case(self, case):
        return AgentOutput(output=self.output, tools=self.tools)


class PassingJudge:
    provider = "mock"
    model = "judge-test"

    def __init__(self) -> None:
        self.calls = 0

    def judge_case(self, case, output, tools, semantic_reasons):
        self.calls += 1
        return JudgeDecision(True, 0.92, "语义满足 rubric")


class ExplodingJudge:
    provider = "mock"
    model = "judge-test"

    def judge_case(self, case, output, tools, semantic_reasons):
        raise RuntimeError("judge unavailable")


class EvalRunnerTest(unittest.TestCase):
    def test_implicit_cases_are_balanced_and_avoid_explicit_tool_prompts(self) -> None:
        cases = load_cases(IMPLICIT_CASES_PATH)
        categories = Counter(case.category for case in cases)
        banned_prompt_fragments = (
            "multiply",
            "current_time",
            "shell_exec",
            "web_search",
            "web_fetch",
            "save_user_profile",
            "create_cron_job",
            "list_cron_jobs",
            "delete_cron_job",
            "github_",
            "MCP",
            "mcp",
            "请调用",
            "请使用",
            "工具",
        )

        self.assertEqual(len(cases), 30)
        self.assertEqual(
            categories,
            Counter(
                {
                    "local_tool_use": 5,
                    "shell_safety": 5,
                    "memory": 5,
                    "cron": 5,
                    "mcp_tool_use": 5,
                    "dialogue_reasoning": 5,
                }
            ),
        )
        for case in cases:
            for fragment in banned_prompt_fragments:
                self.assertNotIn(fragment, case.prompt)

    def test_load_cases_from_jsonl_with_mcp_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_path = Path(directory) / "cases.jsonl"
            cases_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "case_1",
                                "category": "local_tool_use",
                                "prompt": "算 2*3",
                                "expected_contains": ["6"],
                                "expected_tool": "multiply",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "case_2",
                                "category": "mcp_tool_use",
                                "prompt": "查 GitHub 仓库",
                                "expected_contains": ["GitHub"],
                                "expected_tool_prefix": "github_",
                                "requires_mcp": True,
                                "skip_reason": "GitHub MCP 未配置",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_cases(cases_path)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].id, "case_1")
        self.assertEqual(cases[0].expected_contains, ("6",))
        self.assertEqual(cases[0].expected_tool, "multiply")
        self.assertEqual(cases[1].expected_tool_prefix, "github_")
        self.assertTrue(cases[1].requires_mcp)
        self.assertEqual(cases[1].skip_reason, "GitHub MCP 未配置")

    def test_load_cases_parses_judge_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_path = Path(directory) / "cases.jsonl"
            cases_path.write_text(
                json.dumps(
                    {
                        "id": "case_1",
                        "category": "dialogue_reasoning",
                        "prompt": "应该怎么做？",
                        "expected_contains": ["方案"],
                        "judge_rubric": "回答应说明先给方案。",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_cases(cases_path)

        self.assertEqual(cases[0].judge_rubric, "回答应说明先给方案。")

    def test_judge_can_rescue_semantic_rule_failure(self) -> None:
        case = EvalCase(
            id="semantic",
            category="dialogue_reasoning",
            prompt="只给方案时要不要改文件？",
            expected_contains=("不应该",),
            judge_rubric="回答应表达不要直接改文件。",
        )
        judge = PassingJudge()

        results = evaluate_cases(
            [case],
            StaticAgent(output="不，先给方案和风险。"),
            judge=judge,
        )

        self.assertEqual(results[0].status, "PASS")
        self.assertEqual(results[0].rule_status, "FAIL")
        self.assertEqual(results[0].judge_status, "PASS")
        self.assertEqual(results[0].judge_score, 0.92)
        self.assertEqual(results[0].reasons, ())
        self.assertEqual(judge.calls, 1)

    def test_judge_does_not_override_missing_tool(self) -> None:
        case = EvalCase(
            id="tool",
            category="local_tool_use",
            prompt="确认目录",
            expected_contains=("目录",),
            expected_tool="shell_exec",
            judge_rubric="回答应说明目录。",
        )
        judge = PassingJudge()

        results = evaluate_cases(
            [case],
            StaticAgent(output="当前目录是 /tmp", tools=()),
            judge=judge,
        )

        self.assertEqual(results[0].status, "FAIL")
        self.assertEqual(results[0].judge_status, None)
        self.assertEqual(judge.calls, 0)

    def test_judge_error_keeps_rule_failure(self) -> None:
        case = EvalCase(
            id="semantic",
            category="dialogue_reasoning",
            prompt="只给方案时要不要改文件？",
            expected_contains=("不应该",),
            judge_rubric="回答应表达不要直接改文件。",
        )

        results = evaluate_cases(
            [case],
            StaticAgent(output="不，先给方案和风险。"),
            judge=ExplodingJudge(),
        )

        self.assertEqual(results[0].status, "FAIL")
        self.assertEqual(results[0].judge_status, "ERROR")
        self.assertIn("judge error", results[0].reasons[-1])

    def test_compute_metrics_handles_pass_fail_and_skip(self) -> None:
        results = [
            EvalResult(
                case_id="a",
                category="local_tool_use",
                status="PASS",
                latency_seconds=0.1,
                output="ok",
            ),
            EvalResult(
                case_id="b",
                category="local_tool_use",
                status="FAIL",
                latency_seconds=0.2,
                output="bad",
            ),
            EvalResult(
                case_id="c",
                category="mcp_tool_use",
                status="SKIP",
                latency_seconds=0.0,
                output="",
                reasons=("GitHub MCP 未配置",),
            ),
            EvalResult(
                case_id="d",
                category="shell_safety",
                status="PASS",
                latency_seconds=0.4,
                output="拒绝",
            ),
        ]

        metrics = compute_metrics(
            results,
            registered_tools=("multiply", "shell_exec"),
            workspace_root="/tmp/eval-workspace",
            run_id="test-run",
        )

        self.assertEqual(metrics["run_id"], "test-run")
        self.assertEqual(metrics["workspace_root"], "/tmp/eval-workspace")
        self.assertEqual(metrics["registered_tools"], ["multiply", "shell_exec"])
        self.assertFalse(metrics["judge_enabled"])
        self.assertEqual(metrics["judge_cases"], 0)
        self.assertEqual(metrics["total_cases"], 4)
        self.assertEqual(metrics["evaluated_cases"], 3)
        self.assertEqual(metrics["skipped_cases"], 1)
        self.assertEqual(metrics["passed_cases"], 2)
        self.assertEqual(metrics["task_success_rate"], 0.6667)
        self.assertEqual(metrics["category_success_rate"]["local_tool_use"], 0.5)
        self.assertEqual(metrics["category_success_rate"]["mcp_tool_use"], 0.0)
        self.assertEqual(metrics["sandbox_safety_pass_rate"], 1.0)
        self.assertEqual(metrics["p50_latency_seconds"], 0.2)

    def test_mock_mode_outputs_and_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_path = Path(directory) / "cases.jsonl"
            output_path = Path(directory) / "metrics.json"
            details_path = Path(directory) / "cases.jsonl"
            cases_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "tool",
                                "category": "local_tool_use",
                                "prompt": "算 7*8",
                                "expected_contains": ["56"],
                                "expected_tool": "multiply",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "id": "mcp",
                                "category": "mcp_tool_use",
                                "prompt": "查 GitHub",
                                "expected_contains": ["GitHub"],
                                "expected_tool_prefix": "github_",
                                "requires_mcp": True,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cases = load_cases(cases_path)
            results = evaluate_cases(cases, MockEvalAgent())
            metrics = compute_metrics(results, registered_tools=MockEvalAgent().registered_tools)
            write_metrics(output_path, metrics)
            write_case_details(details_path, results)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            details = [
                json.loads(line)
                for line in details_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(saved["total_cases"], 2)
        self.assertEqual(saved["evaluated_cases"], 2)
        self.assertEqual(saved["passed_cases"], 2)
        self.assertEqual(saved["task_success_rate"], 1.0)
        self.assertEqual(details[0]["status"], "PASS")
        self.assertIn("rule_status", details[0])
        self.assertIn("judge_status", details[0])
        self.assertEqual(details[1]["tools"], ["github_get_repo"])

    def test_metrics_include_judge_counts(self) -> None:
        results = [
            EvalResult(
                case_id="a",
                category="dialogue_reasoning",
                status="PASS",
                latency_seconds=0.1,
                output="ok",
                rule_status="FAIL",
                judge_status="PASS",
                judge_score=0.9,
                judge_reason="ok",
                judge_model="judge-model",
            ),
            EvalResult(
                case_id="b",
                category="dialogue_reasoning",
                status="FAIL",
                latency_seconds=0.1,
                output="bad",
                rule_status="FAIL",
                judge_status="ERROR",
                judge_reason="timeout",
                judge_model="judge-model",
            ),
        ]

        metrics = compute_metrics(
            results,
            judge_enabled=True,
            judge_provider="openai",
            judge_model="judge-model",
        )

        self.assertTrue(metrics["judge_enabled"])
        self.assertEqual(metrics["judge_provider"], "openai")
        self.assertEqual(metrics["judge_model"], "judge-model")
        self.assertEqual(metrics["judge_cases"], 2)
        self.assertEqual(metrics["judge_passed"], 1)
        self.assertEqual(metrics["judge_errors"], 1)

    def test_mcp_case_is_skipped_when_required_tools_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_path = Path(directory) / "cases.jsonl"
            cases_path.write_text(
                json.dumps(
                    {
                        "id": "mcp",
                        "category": "mcp_tool_use",
                        "prompt": "查 GitHub",
                        "expected_contains": ["GitHub"],
                        "expected_tool_prefix": "github_",
                        "requires_mcp": True,
                        "skip_reason": "GitHub MCP 未配置",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            results = evaluate_cases(load_cases(cases_path), NoMCPAgent())

        self.assertEqual(results[0].status, "SKIP")
        self.assertEqual(results[0].reasons, ("GitHub MCP 未配置",))


if __name__ == "__main__":
    unittest.main()
