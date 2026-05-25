from __future__ import annotations

import json
import subprocess
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
            expected_cwd = Path(directory).resolve() / "workspace" / "office"
            llm = ShellExecLLM("pwd")
            agent = AgentLoop(
                llm,
                build_tool_registry(context),
                tool_context=context,
            )

            result = agent.run([{"role": "user", "content": "我在哪个目录？"}])

        shell_result = result.observations[0]["result"]
        self.assertEqual(result.output, "完成")
        self.assertEqual(shell_result["cwd"], str(expected_cwd))
        self.assertEqual(shell_result["stdout"].strip(), str(expected_cwd))

    def test_tool_context_creates_default_sandbox_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(cwd=Path(directory), allow_shell_exec=True)

            self.assertEqual(
                context.sandbox_root,
                Path(directory).resolve() / "workspace" / "office",
            )
            self.assertTrue(context.sandbox_root.is_dir())

    def test_safe_command_executes_without_approval_in_sandbox_root(self) -> None:
        approval = StaticApprovalProvider(approved=False)
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(
                cwd=Path(directory),
                allow_shell_exec=True,
                approval_provider=approval,
            )

            result = shell_exec_tool.call({"command": "pwd"}, context)

        self.assertFalse(result["blocked"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), str(context.sandbox_root))
        self.assertEqual(result["cwd"], str(context.sandbox_root))
        self.assertEqual(approval.calls, [])

    def test_exact_diagnostic_commands_are_safe(self) -> None:
        approval = StaticApprovalProvider(approved=False)
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(
                cwd=Path(directory),
                allow_shell_exec=True,
                approval_provider=approval,
            )
            commands = [
                "pwd",
                "ls",
                "python --version",
                "conda --version",
                "conda env list",
                "which python",
                "which conda",
                "pwd | wc -l",
            ]

            with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args="",
                    returncode=0,
                    stdout="ok",
                    stderr="",
                )
                results = [shell_exec_tool.call({"command": command}, context) for command in commands]

        self.assertTrue(all(not result["blocked"] for result in results))
        self.assertEqual(approval.calls, [])
        self.assertEqual(run.call_count, len(commands))

    def test_outside_paths_are_denied_before_subprocess(self) -> None:
        context = ToolContext(cwd=Path.cwd(), allow_shell_exec=True)

        denied_commands = [
            "rm -rf /tmp/AsyncClaw-test",
            "cat /etc/passwd",
            "ls ..",
            "ls ~/Desktop",
            "python --config=/tmp/config.py",
            "python -c 'print(1)'",
            "node -e 'console.log(1)'",
        ]

        with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
            results = [
                shell_exec_tool.call({"command": command}, context)
                for command in denied_commands
            ]

        self.assertTrue(all(result["blocked"] for result in results))
        run.assert_not_called()

    def test_symlink_escape_is_denied_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside_file = root / "outside.txt"
            outside_file.write_text("secret", encoding="utf-8")
            context = ToolContext(cwd=root, allow_shell_exec=True)
            (context.sandbox_root / "outside_link").symlink_to(outside_file)

            with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
                result = shell_exec_tool.call({"command": "cat outside_link"}, context)

        self.assertTrue(result["blocked"])
        run.assert_not_called()

    def test_unapproved_command_does_not_execute(self) -> None:
        approval = StaticApprovalProvider(approved=False)
        context = ToolContext(
            cwd=Path.cwd(),
            allow_shell_exec=True,
            approval_provider=approval,
        )

        with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
            result = shell_exec_tool.call({"command": "echo hello > output.txt"}, context)

        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "命令未通过审批")
        self.assertEqual(len(approval.calls), 1)
        run.assert_not_called()

    def test_approved_confirm_command_executes_in_sandbox_root(self) -> None:
        approval = StaticApprovalProvider(approved=True)
        with tempfile.TemporaryDirectory() as directory:
            context = ToolContext(
                cwd=Path(directory),
                allow_shell_exec=True,
                approval_provider=approval,
            )

            result = shell_exec_tool.call({"command": "echo hello > output.txt"}, context)

            output_path = context.sandbox_root / "output.txt"
            self.assertFalse(result["blocked"])
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["cwd"], str(context.sandbox_root))
            self.assertEqual(output_path.read_text(encoding="utf-8").strip(), "hello")
            self.assertEqual(len(approval.calls), 1)
            self.assertEqual(approval.calls[0]["cwd"], context.sandbox_root)

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

        with patch("AsyncClaw.tools.builtin.shell.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args="",
                returncode=0,
                stdout="123456789",
                stderr="",
            )
            result = shell_exec_tool.call({"command": "pwd"}, context)

        self.assertIn("[已截断到 5 字节]", result["stdout"])


if __name__ == "__main__":
    unittest.main()
