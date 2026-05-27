from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from AsyncClaw import (
    AgentLoop,
    JsonlEventLogger,
    Tool,
    ToolContext,
    ToolRegistry,
    WorkspaceStore,
    current_time_tool,
    multiply_tool,
    shell_exec_tool,
)


class FakeLLM:
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
                                        "arguments": json.dumps({"a": 3, "b": 5}),
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
                        "content": "3 * 5 = 15",
                    }
                }
            ]
        }


class ErrorLLM:
    def create_chat_completion(self, **kwargs):
        raise RuntimeError("llm 调用失败")


class AsyncToolLLM:
    def __init__(self) -> None:
        self.requests = []

    async def create_chat_completion(self, **kwargs):
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
                                    "id": "call_async_echo",
                                    "type": "function",
                                    "function": {
                                        "name": "async_echo",
                                        "arguments": json.dumps({"text": "hello"}),
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
                        "content": "hello",
                    }
                }
            ]
        }


class UnknownToolLLM:
    def __init__(self, arguments: object | None = None, name: str = "missing_tool") -> None:
        self.arguments = "{}" if arguments is None else arguments
        self.name = name
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
                                    "id": "call_missing",
                                    "type": "function",
                                    "function": {
                                        "name": self.name,
                                        "arguments": self.arguments,
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
                        "content": "已处理错误",
                    }
                }
            ]
        }


class MemoryLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, event: str, data: dict[str, object]) -> None:
        self.records.append({"event": event, "data": data})


class BrokenLogger:
    def log(self, event: str, data: dict[str, object]) -> None:
        raise RuntimeError("日志失败")


class WorkspaceLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "已读取 workspace",
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
                                            {"profile_markdown": "# 用户画像\n- 喜欢 Python"}
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


class SummaryWorkspaceLLM:
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
                            "content": "合并后的近期摘要",
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "已读取压缩 workspace",
                    }
                }
            ]
        }


class LongToolResultLLM:
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
                                    "id": "call_long",
                                    "type": "function",
                                    "function": {
                                        "name": "long_result",
                                        "arguments": "{}",
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
                        "content": "已读取长结果",
                    }
                }
            ]
        }


class CountingToolRegistry(ToolRegistry):
    def __init__(self, tools: list[Tool]) -> None:
        super().__init__(tools)
        self.to_openai_tools_calls = 0

    def to_openai_tools(self) -> list[dict[str, object]]:
        self.to_openai_tools_calls += 1
        return super().to_openai_tools()


class AgentLoopTest(unittest.TestCase):
    def test_runs_tool_call_and_returns_final_answer(self) -> None:
        llm = FakeLLM()
        agent = AgentLoop(llm, ToolRegistry([multiply_tool]), logger=MemoryLogger())

        result = agent.run([{"role": "user", "content": "3 乘以 5 等于多少？"}])

        self.assertEqual(result.output, "3 * 5 = 15")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.observations[0]["name"], "multiply")
        self.assertEqual(result.observations[0]["result"], {"product": 15})

        first_request = llm.requests[0]
        self.assertEqual(first_request["tool_choice"], "auto")
        self.assertEqual(
            first_request["tools"][0]["function"]["name"],
            "multiply",
        )
        self.assertIn("parameters", first_request["tools"][0]["function"])

        second_request_messages = llm.requests[1]["messages"]
        self.assertEqual(second_request_messages[-1]["role"], "tool")
        self.assertEqual(second_request_messages[-1]["tool_call_id"], "call_multiply")
        self.assertEqual(json.loads(second_request_messages[-1]["content"]), {"product": 15})

    def test_reuses_cached_openai_tool_schema_across_steps(self) -> None:
        llm = FakeLLM()
        registry = CountingToolRegistry([multiply_tool])
        agent = AgentLoop(llm, registry, logger=MemoryLogger())

        agent.run([{"role": "user", "content": "3 乘以 5 等于多少？"}])

        self.assertEqual(registry.to_openai_tools_calls, 1)
        self.assertIs(llm.requests[0]["tools"], llm.requests[1]["tools"])

    def test_current_time_tool_returns_local_time_shape(self) -> None:
        result = ToolRegistry([current_time_tool]).call("current_time", {})

        self.assertEqual(set(result), {"timezone", "iso_time", "date", "time"})
        datetime.fromisoformat(result["iso_time"])
        datetime.fromisoformat(f"{result['date']}T{result['time']}")

    def test_arun_executes_async_tool_handler(self) -> None:
        async def async_echo(arguments):
            await asyncio.sleep(0)
            return {"echo": arguments["text"]}

        tool = Tool(
            name="async_echo",
            description="回显文本。",
            schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=async_echo,
        )
        result = asyncio.run(AgentLoop(AsyncToolLLM(), ToolRegistry([tool])).arun([
            {"role": "user", "content": "echo hello"}
        ]))

        self.assertEqual(result.output, "hello")
        self.assertEqual(result.observations[0]["result"], {"echo": "hello"})

    def test_unknown_tool_returns_observation_error(self) -> None:
        result = AgentLoop(UnknownToolLLM(), ToolRegistry([]), logger=MemoryLogger()).run([
            {"role": "user", "content": "调用不存在的工具"}
        ])

        self.assertEqual(result.output, "已处理错误")
        self.assertIn("未知工具", result.observations[0]["result"]["error"])

    def test_invalid_tool_arguments_return_observation_error(self) -> None:
        result = AgentLoop(UnknownToolLLM(arguments="[1, 2]"), ToolRegistry([])).run([
            {"role": "user", "content": "传错参数"}
        ])

        self.assertEqual(result.output, "已处理错误")
        self.assertEqual(result.observations[0]["name"], "missing_tool")
        self.assertIn("工具参数必须解析为 JSON 对象", result.observations[0]["result"]["error"])

    def test_logger_failure_does_not_stop_run(self) -> None:
        result = AgentLoop(
            FakeLLM(),
            ToolRegistry([multiply_tool]),
            logger=BrokenLogger(),
        ).run([{"role": "user", "content": "3 乘以 5 等于多少？"}])

        self.assertEqual(result.output, "3 * 5 = 15")

    def test_writes_jsonl_events_for_tool_run(self) -> None:
        llm = FakeLLM()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            agent = AgentLoop(
                llm,
                ToolRegistry([multiply_tool]),
                logger=JsonlEventLogger(log_path),
            )

            agent.run([{"role": "user", "content": "3 乘以 5 等于多少？"}])

            records = _read_jsonl(log_path)

        self.assertEqual(
            [record["event"] for record in records],
            [
                "user_message",
                "llm_request",
                "llm_response",
                "tool_call",
                "tool_result",
                "llm_request",
                "llm_response",
                "final_answer",
            ],
        )
        self.assertEqual(records[4]["data"]["result"], {"product": 15})
        self.assertEqual(records[-1]["data"]["content"], "3 * 5 = 15")

    def test_writes_error_event_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            agent = AgentLoop(
                ErrorLLM(),
                ToolRegistry([multiply_tool]),
                logger=JsonlEventLogger(log_path),
            )

            with self.assertRaises(RuntimeError):
                agent.run([{"role": "user", "content": "你好"}])

            records = _read_jsonl(log_path)

        self.assertEqual(records[-1]["event"], "error")
        self.assertEqual(records[-1]["data"]["type"], "RuntimeError")
        self.assertEqual(records[-1]["data"]["message"], "llm 调用失败")

    def test_workspace_injects_system_profile_and_recent_messages(self) -> None:
        llm = WorkspaceLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            workspace.save_user_profile("# 用户画像\n- 常用 pyclaw 环境")
            (workspace.skills_dir / "python-helper").mkdir()
            (workspace.skills_dir / "python-helper" / "SKILL.md").write_text(
                "---\n"
                "name: python-helper\n"
                "description: 处理 Python 代码时使用。\n"
                "---\n"
                "回答 Python 问题时先考虑标准库。",
                encoding="utf-8",
            )
            for index in range(12):
                workspace.append_session_turn([
                    {"role": "user", "content": f"old-{index}"},
                ])
            agent = AgentLoop(
                llm,
                ToolRegistry([]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            result = agent.run([{"role": "user", "content": "你好"}])
            request_messages = llm.requests[0]["messages"]
            session_records = _read_jsonl(workspace.session_path)
            history_records = _read_jsonl(workspace.user_inputs_path)

        self.assertEqual(result.output, "已读取 workspace")
        self.assertEqual(request_messages[0]["role"], "system")
        self.assertIn("常用 pyclaw 环境", request_messages[0]["content"])
        self.assertIn('name="python-helper"', request_messages[0]["content"])
        self.assertIn("处理 Python 代码时使用。", request_messages[0]["content"])
        self.assertNotIn("回答 Python 问题时先考虑标准库。", request_messages[0]["content"])
        self.assertEqual([message["content"] for message in request_messages[1:13]], [
            f"old-{index}" for index in range(12)
        ])
        self.assertEqual(request_messages[-1], {"role": "user", "content": "你好"})
        self.assertEqual(session_records[-1]["messages"][0]["content"], "你好")
        self.assertEqual(session_records[-1]["messages"][-1]["content"], "已读取 workspace")
        self.assertEqual(history_records[-1]["content"], "你好")

    def test_workspace_shell_prompt_guides_local_shell_use_when_available(self) -> None:
        llm = WorkspaceLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            agent = AgentLoop(
                llm,
                ToolRegistry([shell_exec_tool]),
                tool_context=ToolContext(cwd=Path(directory), allow_shell_exec=True),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            agent.run([{"role": "user", "content": "当前在哪个目录？"}])
            system_prompt = llm.requests[0]["messages"][0]["content"]

        self.assertIn("本地 shell 工具使用规则", system_prompt)
        self.assertIn("当前目录", system_prompt)
        self.assertIn("shell_exec", system_prompt)
        self.assertIn("不要在 shell_exec 可用时声称无法访问本地工作目录", system_prompt)

    def test_workspace_shell_prompt_is_not_injected_when_shell_tool_is_unavailable(self) -> None:
        llm = WorkspaceLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            agent = AgentLoop(
                llm,
                ToolRegistry([]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            agent.run([{"role": "user", "content": "当前在哪个目录？"}])
            system_prompt = llm.requests[0]["messages"][0]["content"]

        self.assertNotIn("本地 shell 工具使用规则", system_prompt)
        self.assertNotIn("不要在 shell_exec 可用时声称无法访问本地工作目录", system_prompt)

    def test_workspace_records_complete_tool_turn(self) -> None:
        llm = FakeLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            agent = AgentLoop(
                llm,
                ToolRegistry([multiply_tool]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            result = agent.run([{"role": "user", "content": "3 乘以 5 等于多少？"}])
            turns = workspace.load_session_turns()

        self.assertEqual(result.output, "3 * 5 = 15")
        self.assertEqual(len(turns), 1)
        messages = turns[0]["messages"]
        self.assertEqual([message["role"] for message in messages], [
            "user",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_multiply")
        self.assertEqual(messages[2]["tool_call_id"], "call_multiply")
        self.assertEqual(json.loads(messages[2]["content"]), {"product": 15})

    def test_current_turn_tool_result_is_not_truncated_before_final_answer(self) -> None:
        long_text = "x" * 5000

        def long_result(arguments):
            return {"text": long_text}

        tool = Tool(
            name="long_result",
            description="返回长文本。",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=long_result,
        )
        llm = LongToolResultLLM()
        agent = AgentLoop(llm, ToolRegistry([tool]), logger=MemoryLogger())

        result = agent.run([{"role": "user", "content": "读取长结果"}])
        tool_message = llm.requests[1]["messages"][-1]

        self.assertEqual(result.output, "已读取长结果")
        self.assertEqual(json.loads(tool_message["content"]), {"text": long_text})
        self.assertNotIn("历史工具结果已截断", tool_message["content"])

    def test_workspace_compacts_history_before_llm_request(self) -> None:
        llm = SummaryWorkspaceLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
                summary_threshold=3,
                recent_turn_limit=1,
            )
            workspace.save_short_term_summary("旧摘要")
            for index in range(3):
                workspace.append_session_turn([
                    {"role": "user", "content": f"old-user-{index}"},
                    {"role": "assistant", "content": f"old-assistant-{index}"},
                ])
            agent = AgentLoop(
                llm,
                ToolRegistry([]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            result = agent.run([{"role": "user", "content": "你好"}])
            summary_request = llm.requests[0]
            chat_request_messages = llm.requests[1]["messages"]
            turns = workspace.load_session_turns()

        self.assertEqual(result.output, "已读取压缩 workspace")
        self.assertNotIn("tools", summary_request)
        self.assertEqual(chat_request_messages[0]["role"], "system")
        self.assertEqual(chat_request_messages[1]["role"], "system")
        self.assertIn("合并后的近期摘要", chat_request_messages[1]["content"])
        self.assertEqual(chat_request_messages[2]["content"], "old-user-2")
        self.assertEqual(chat_request_messages[-1]["content"], "你好")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["messages"][0]["content"], "old-user-2")
        self.assertEqual(turns[1]["messages"][0]["content"], "你好")

    def test_workspace_save_user_profile_tool_overwrites_memory(self) -> None:
        llm = SaveProfileLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            agent = AgentLoop(
                llm,
                ToolRegistry([]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            result = agent.run([{"role": "user", "content": "我喜欢 Python"}])
            tool_names = [
                tool["function"]["name"]
                for tool in llm.requests[0]["tools"]
            ]
            profile = workspace.load_user_profile()

        self.assertEqual(result.output, "记住了")
        self.assertIn("save_user_profile", tool_names)
        self.assertEqual(profile, "# 用户画像\n- 喜欢 Python")

    def test_workspace_prompt_guides_model_to_decide_memory_use(self) -> None:
        llm = WorkspaceLLM()
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(
                root=Path(directory) / "workspace",
                session_id="session-a",
            )
            agent = AgentLoop(
                llm,
                ToolRegistry([]),
                workspace=workspace,
                logger=MemoryLogger(),
            )

            agent.run([
                {"role": "user", "content": "我喜欢打篮球， 玩游戏，我 是学计算机的，你呢？"}
            ])
            system_prompt = llm.requests[0]["messages"][0]["content"]

        self.assertIn("save_user_profile", system_prompt)
        self.assertIn("稳定偏好和兴趣", system_prompt)
        self.assertIn("稳定身份背景", system_prompt)
        self.assertIn("如果需要记录", system_prompt)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
