from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from AsyncClaw.agent import AgentLoop
from AsyncClaw.tools import ToolContext, build_tool_registry, shell_exec_tool


class StaticApprovalProvider:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.calls = []

    def approve(self, *, command: str, cwd: Path, reason: str | None = None) -> bool:
        self.calls.append({"command": command, "cwd": cwd, "reason": reason})
        return self.approved


class ShellExecLLM:
    def __init__(self, command: str) -> None:
        self.command = command
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_shell",
                                    "type": "function",
                                    "function": {
                                        "name": "shell_exec",
                                        "arguments": json.dumps({"command": self.command}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "完成",
                    }
                }
            ]
        }


class ShellExecToolTest(unittest.TestCase):
    def test_shell_exec_is_not_exposed_by_default(self) -> None:
        context = ToolContext(cwd=Path.cwd())

        tool_names = [
            tool["function"]["name"]
            for tool in build_tool_registry(context).to_openai_tools()
        ]

        self.assertNotIn("shell_exec", tool_names)

    def test_shell_exec_is_exposed_when_context_allows_it(self) -> None:
        context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)

        tool_names = [
            tool["function"]["name"]
            for tool in build_tool_registry(context).to_openai_tools()
        ]

        self.assertIn("shell_exec", tool_names)

    def test_agent_loop_passes_tool_context_to_shell_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(cwd=Path(directory), allow_shell_exec=True)
            llm = ShellExecLLM("pwd")
            agent = AgentLoop(
                llm,
                build_tool_registry(context),
                tool_context=context,
            )

            result = agent.run([{"role": "user", "content": "我在哪个目录？"}])

        shell_result = result.observations[0]["result"]
        self.assertEqual(result.output, "完成")
        self.assertEqual(shell_result["cwd"], str(Path(directory).resolve()))
        self.assertEqual(shell_result["stdout"].strip(), str(Path(directory).resolve()))

    def test_safe_command_executes_without_approval(self) -> None:
        context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)

        result = shell_exec_tool.call({"command": "printf hello"}, context)

        self.assertFalse(result["blocked"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "hello")

    def test_dangerous_command_is_blocked_before_subprocess(self) -> None:
        context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)

        with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
            result = shell_exec_tool.call({"command": "rm -rf /tmp/AsyncClaw-test"}, context)

        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "破坏性命令")
        run.assert_not_called()

    def test_unapproved_command_does_not_execute(self) -> None:
        approval = StaticApprovalProvider(approved=False)
        context = ToolContext(
            cwd=Path.cwd(),
            allow_shell_exec=True,
            approval_provider=approval,
        )

        with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
            result = shell_exec_tool.call({"command": "python -V"}, context)

        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "命令未通过审批")
        self.assertEqual(len(approval.calls), 1)
        run.assert_not_called()

    def test_timeout_returns_structured_result(self) -> None:
        approval = StaticApprovalProvider(approved=True)
        context = ToolContext(
            cwd=Path.cwd(),
            allow_shell_exec=True,
            shell_timeout_seconds=0.1,
            approval_provider=approval,
        )

        result = shell_exec_tool.call({"command": "sleep 2"}, context)

        self.assertFalse(result["blocked"])
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])

    def test_output_is_truncated(self) -> None:
        context = ToolContext(
            cwd=Path.cwd(),
            allow_shell_exec=True,
            shell_output_limit_bytes=5,
        )

        result = shell_exec_tool.call({"command": "printf 123456789"}, context)

        self.assertIn("[已截断到 5 字节]", result["stdout"])


if __name__ == "__main__":
    unittest.main()
