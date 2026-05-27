from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from AsyncClaw.agent.cron import CronJob
from AsyncClaw.channels import AgentRequest, AgentService
from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.cli.main import main
from AsyncClaw.config import LLMConfig
from AsyncClaw.channels.service import _resolve_env_file
from AsyncClaw.cli.agent import (
    EXIT_COMMANDS,
    _normalize_user_input,
    _patch_stdout_context,
    _read_user_text,
    _render_cron_start,
    _render_startup,
)


class SimpleLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "你好，我是 AsyncClaw",
                    }
                }
            ]
        }


class ToolCallingLLM:
    def __init__(self) -> None:
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
                                    "id": "call_multiply",
                                    "type": "function",
                                    "function": {
                                        "name": "multiply",
                                        "arguments": json.dumps({"a": 2, "b": 4}),
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
                        "content": "2 * 4 = 8",
                    }
                }
            ]
        }


class SaveProfileLLM:
    def __init__(self) -> None:
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
                                    "id": "call_save_profile",
                                    "type": "function",
                                    "function": {
                                        "name": "save_user_profile",
                                        "arguments": json.dumps(
                                            {"profile_markdown": "# 用户画像\n- 常用 pyclaw"}
                                        ),
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
                        "content": "记住了",
                    }
                }
            ]
        }


class AgentServiceTest(unittest.TestCase):
    def test_service_handles_text_from_configured_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "outside"
            project_root = Path(directory) / "project"
            cwd.mkdir()
            project_root.mkdir()
            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                service = AgentService(
                    cwd=cwd,
                    llm=SimpleLLM(),
                    config=LLMConfig(
                        api_key="test-key",
                        model="test-model",
                        base_url="https://example.test/v1",
                        provider="openai",
                        agent_max_steps=3,
                    ),
                )

                response = service.handle_text("你好")
                session_exists = service.workspace.session_path.exists()
                outside_session_exists = (cwd / "workspace" / "session").exists()

        self.assertEqual(response.output, "你好，我是 AsyncClaw")
        self.assertEqual(response.cwd, cwd.resolve())
        self.assertEqual(response.steps, 1)
        self.assertEqual(response.session_id, service.workspace.session_id)
        self.assertEqual(service.max_steps, 3)
        self.assertEqual(service.workspace.root, project_root.resolve() / "workspace")
        self.assertEqual(service.tool_context.cwd, cwd.resolve())
        self.assertEqual(service.tool_context.sandbox_root, service.workspace.root / "office")
        self.assertEqual(service.log_path, project_root.resolve() / "logs" / "events.jsonl")
        self.assertTrue(service.tool_context.allow_shell_exec)
        self.assertTrue(session_exists)
        self.assertFalse(outside_session_exists)

    def test_service_can_disable_shell_tool_for_non_cli_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "project"
            project_root.mkdir()
            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                service = AgentService(
                    cwd=directory,
                    llm=SimpleLLM(),
                    allow_shell_exec=False,
                )

            tool_names = [
                tool["function"]["name"]
                for tool in service.tools.to_openai_tools()
            ]

        self.assertNotIn("shell_exec", tool_names)
        self.assertFalse(service.tool_context.allow_shell_exec)

    def test_service_returns_tool_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "project"
            project_root.mkdir()
            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                response = AgentService(
                    cwd=directory,
                    llm=ToolCallingLLM(),
                ).handle(AgentRequest(text="2 乘以 4"))

        self.assertEqual(response.output, "2 * 4 = 8")
        self.assertEqual(response.observations[0]["name"], "multiply")
        self.assertEqual(response.observations[0]["result"], {"product": 8})

    def test_service_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "project"
            project_root.mkdir()
            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                service = AgentService(cwd=directory, llm=SimpleLLM())

                with self.assertRaisesRegex(ValueError, "输入不能为空"):
                    service.handle_text("   ")

    def test_service_can_use_explicit_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "outside"
            workspace_root = Path(directory) / "state"
            cwd.mkdir()
            service = AgentService(
                cwd=cwd,
                workspace_root=workspace_root,
                llm=SimpleLLM(),
            )

            response = service.handle_text("你好")
            session_exists = service.workspace.session_path.exists()
            outside_session_exists = (cwd / "workspace" / "session").exists()

        self.assertEqual(response.cwd, cwd.resolve())
        self.assertEqual(service.workspace.root, workspace_root.resolve())
        self.assertEqual(service.tool_context.sandbox_root, workspace_root.resolve() / "office")
        self.assertEqual(service.log_path, workspace_root.resolve() / "logs" / "events.jsonl")
        self.assertTrue(session_exists)
        self.assertFalse(outside_session_exists)

    def test_service_keeps_custom_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "custom-workspace")
            service = AgentService(
                cwd=directory,
                workspace_root=Path(directory) / "ignored",
                workspace=workspace,
                llm=SimpleLLM(),
            )

        self.assertIs(service.workspace, workspace)
        self.assertEqual(service.tool_context.sandbox_root, workspace.root / "office")

    def test_service_records_saved_profile_in_project_workspace_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "outside"
            project_root = Path(directory) / "project"
            cwd.mkdir()
            project_root.mkdir()
            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                service = AgentService(
                    cwd=cwd,
                    llm=SaveProfileLLM(),
                )

                response = service.handle_text("记住我常用 pyclaw")
                profile = service.workspace.load_user_profile()
                turns = service.workspace.load_session_turns()

        self.assertEqual(response.output, "记住了")
        self.assertEqual(profile, "# 用户画像\n- 常用 pyclaw")
        self.assertEqual(service.workspace.root, project_root.resolve() / "workspace")
        self.assertEqual([message["role"] for message in turns[-1]["messages"]], [
            "user",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(turns[-1]["messages"][2]["name"], "save_user_profile")


class EnvFileResolutionTest(unittest.TestCase):
    def test_default_env_file_prefers_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "run"
            project_root = Path(directory) / "project"
            cwd.mkdir()
            project_root.mkdir()
            cwd_env = cwd / ".env"
            project_env = project_root / ".env"
            cwd_env.write_text("OPENAI_API_KEY=cwd-key\n", encoding="utf-8")
            project_env.write_text("OPENAI_API_KEY=project-key\n", encoding="utf-8")

            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                resolved = _resolve_env_file(cwd, ".env")

        self.assertEqual(resolved, cwd_env)

    def test_default_env_file_falls_back_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "run"
            project_root = Path(directory) / "project"
            cwd.mkdir()
            project_root.mkdir()
            project_env = project_root / ".env"
            project_env.write_text("OPENAI_API_KEY=project-key\n", encoding="utf-8")

            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                resolved = _resolve_env_file(cwd, ".env")

        self.assertEqual(resolved, project_env)

    def test_explicit_relative_env_file_resolves_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "run"
            project_root = Path(directory) / "project"
            cwd.mkdir()
            project_root.mkdir()
            project_env = project_root / ".env.test"
            project_env.write_text("OPENAI_API_KEY=project-key\n", encoding="utf-8")

            with patch("AsyncClaw.channels.service._project_root", return_value=project_root):
                resolved = _resolve_env_file(cwd, ".env.test", explicit=True)

        self.assertEqual(resolved, cwd / ".env.test")

    def test_absolute_env_file_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env.any"

            resolved = _resolve_env_file(Path(directory), env_file, explicit=True)

        self.assertEqual(resolved, env_file)


class CliMainTest(unittest.TestCase):
    def test_agent_command_dispatches_to_rich_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("AsyncClaw.cli.main.run_agent_cli", return_value=0) as run_agent_cli:
                exit_code = main(["agent", "--cwd", directory, "--env-file", ".env.test", "--no-shell"])

        self.assertEqual(exit_code, 0)
        run_agent_cli.assert_called_once_with(
            cwd=Path(directory),
            env_file=".env.test",
            env_file_explicit=True,
            workspace_root=None,
            allow_shell_exec=False,
            allow_cron=True,
            cron_max_concurrent_jobs=2,
        )

    def test_agent_command_marks_default_env_file_as_implicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("AsyncClaw.cli.main.run_agent_cli", return_value=0) as run_agent_cli:
                exit_code = main(["agent", "--cwd", directory])

        self.assertEqual(exit_code, 0)
        run_agent_cli.assert_called_once_with(
            cwd=Path(directory),
            env_file=".env",
            env_file_explicit=False,
            workspace_root=None,
            allow_shell_exec=True,
            allow_cron=True,
            cron_max_concurrent_jobs=2,
        )

    def test_agent_command_dispatches_explicit_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace_root = Path(directory) / "state"
            with patch("AsyncClaw.cli.main.run_agent_cli", return_value=0) as run_agent_cli:
                exit_code = main(["agent", "--cwd", directory, "--workspace-root", str(workspace_root)])

        self.assertEqual(exit_code, 0)
        run_agent_cli.assert_called_once_with(
            cwd=Path(directory),
            env_file=".env",
            env_file_explicit=False,
            workspace_root=workspace_root,
            allow_shell_exec=True,
            allow_cron=True,
            cron_max_concurrent_jobs=2,
        )

    def test_agent_command_can_disable_cron(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("AsyncClaw.cli.main.run_agent_cli", return_value=0) as run_agent_cli:
                exit_code = main(["agent", "--cwd", directory, "--no-cron"])

        self.assertEqual(exit_code, 0)
        run_agent_cli.assert_called_once_with(
            cwd=Path(directory),
            env_file=".env",
            env_file_explicit=False,
            workspace_root=None,
            allow_shell_exec=True,
            allow_cron=False,
            cron_max_concurrent_jobs=2,
        )

    def test_agent_command_passes_cron_concurrency_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("AsyncClaw.cli.main.run_agent_cli", return_value=0) as run_agent_cli:
                exit_code = main(
                    ["agent", "--cwd", directory, "--cron-max-concurrent-jobs", "4"]
                )

        self.assertEqual(exit_code, 0)
        run_agent_cli.assert_called_once_with(
            cwd=Path(directory),
            env_file=".env",
            env_file_explicit=False,
            workspace_root=None,
            allow_shell_exec=True,
            allow_cron=True,
            cron_max_concurrent_jobs=4,
        )


class CliStartupRenderTest(unittest.TestCase):
    def test_startup_renders_wordmark_and_concise_status(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100)

        _render_startup(console, object())
        output = console.export_text(styles=False)

        self.assertIn("AsyncClaw", output)
        self.assertIn("Welcome to AsyncClaw", output)
        self.assertIn("Ready in dev mode.", output)
        self.assertIn("/exit", output)
        self.assertIn("local agent runtime", output)
        self.assertIn("╭", output)
        self.assertNotIn("workspace", output)
        self.assertNotIn("session", output)
        self.assertNotIn("cron_dir", output)


class CliInputTest(unittest.TestCase):
    def test_exit_commands_include_slash_exit_alias(self) -> None:
        self.assertIn("/exit", EXIT_COMMANDS)
        self.assertIn("exit", EXIT_COMMANDS)
        self.assertIn("quit", EXIT_COMMANDS)

    def test_normalize_user_input_applies_backspace(self) -> None:
        self.assertEqual(_normalize_user_input("hellp\x7fo"), "hello")
        self.assertEqual(_normalize_user_input("hellp\bo"), "hello")

    def test_normalize_user_input_strips_ansi_delete_and_arrows(self) -> None:
        self.assertEqual(_normalize_user_input("hello\x1b[3~"), "hello")
        self.assertEqual(_normalize_user_input("hello\x1b[D"), "hello")

    def test_normalize_user_input_applies_ctrl_u_line_clear(self) -> None:
        self.assertEqual(_normalize_user_input("draft\x15final"), "final")

    def test_read_user_text_uses_prompt_session_and_normalizes(self) -> None:
        class FakePromptSession:
            def __init__(self) -> None:
                self.prompts = []

            def prompt(self, prompt: str) -> str:
                self.prompts.append(prompt)
                return "hellp\x7fo\n"

        session = FakePromptSession()
        console = Console(file=io.StringIO(), record=True)

        text = _read_user_text(console, session=session)

        self.assertEqual(text, "hello")
        self.assertEqual(session.prompts, ["用户: "])

    def test_read_user_text_patches_stdout_in_raw_mode(self) -> None:
        class FakePromptSession:
            def prompt(self, prompt: str) -> str:
                return "hello"

        calls = []

        @contextmanager
        def fake_patch_stdout(**kwargs):
            calls.append(kwargs)
            yield

        console = Console(file=io.StringIO(), record=True)
        with patch("AsyncClaw.cli.agent.patch_stdout", fake_patch_stdout):
            text = _read_user_text(console, session=FakePromptSession())

        self.assertEqual(text, "hello")
        self.assertEqual(calls, [{"raw": True}])

    def test_read_user_text_internal_session_patches_stdout_in_raw_mode(self) -> None:
        class FakePromptSession:
            def prompt(self, prompt: str) -> str:
                return "hello"

        calls = []

        @contextmanager
        def fake_patch_stdout(**kwargs):
            calls.append(kwargs)
            yield

        console = Console(file=io.StringIO(), record=True)
        with patch("AsyncClaw.cli.agent.PromptSession", FakePromptSession):
            with patch("AsyncClaw.cli.agent.patch_stdout", fake_patch_stdout):
                text = _read_user_text(console)

        self.assertEqual(text, "hello")
        self.assertEqual(calls, [{"raw": True}])

    def test_patch_stdout_context_falls_back_when_raw_is_unsupported(self) -> None:
        calls = []

        def fake_patch_stdout(*args, **kwargs):
            calls.append(kwargs)
            if kwargs:
                raise TypeError("raw is not supported")
            return nullcontext()

        with patch("AsyncClaw.cli.agent.patch_stdout", fake_patch_stdout):
            with _patch_stdout_context():
                pass

        self.assertEqual(calls, [{"raw": True}, {}])

    def test_prompt_toolkit_non_raw_output_replaces_escape_character(self) -> None:
        from prompt_toolkit.output.vt100 import Vt100_Output

        class FakeStdout:
            def __init__(self) -> None:
                self.encoding = "utf-8"
                self.buffer = None

            def write(self, text: str) -> int:
                return len(text)

            def flush(self) -> None:
                pass

            def isatty(self) -> bool:
                return True

        output = Vt100_Output(FakeStdout(), get_size=lambda: None)
        output.write("\x1b[35m")

        self.assertIn("?[35m", "".join(output._buffer))


class CliCronRenderTest(unittest.TestCase):
    def test_cron_start_does_not_render_prompt(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100)
        job = CronJob(
            id="job-1",
            name="当前时间任务",
            prompt="输出当前时间",
            action="agent",
            schedule={"type": "every", "seconds": 10},
            enabled=True,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            last_run_at=None,
            next_run_at="2026-01-01T00:00:10+00:00",
            run_count=0,
            failure_count=0,
            last_error=None,
            running=False,
        )

        _render_cron_start(console, job)
        output = console.export_text(styles=False)

        self.assertIn("定时任务执行中", output)
        self.assertIn("当前时间任务", output)
        self.assertIn("正在执行", output)
        self.assertNotIn("输出当前时间", output)

    def test_prompt_toolkit_dependency_is_declared(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("prompt_toolkit>=3.0", pyproject)
        self.assertIn("prompt_toolkit>=3.0", requirements)


if __name__ == "__main__":
    unittest.main()
