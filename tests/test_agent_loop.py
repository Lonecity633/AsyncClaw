from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from AsyncClaw import AgentLoop, JsonlEventLogger, Tool, ToolRegistry, current_time_tool, multiply_tool


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
