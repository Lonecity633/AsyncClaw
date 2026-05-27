"""ReAct 风格的智能体运行时。"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any

from AsyncClaw.agent.events import EventLogger
from AsyncClaw.agent.logger import JsonlEventLogger
from AsyncClaw.agent.messages import (
    build_observation,
    build_tool_message,
    normalize_assistant_message,
    normalize_tool_call,
)
from AsyncClaw.tools import ToolContext, ToolRegistry, create_save_user_profile_tool
from AsyncClaw.tools.executor import ToolExecutor
from AsyncClaw.agent.workspace import WorkspaceStore


SHORT_TERM_SUMMARY_SYSTEM_PROMPT = """你负责维护 AsyncClaw 的近期对话摘要。

请把旧摘要和新丢弃的完整回合合并成一份新的短期摘要。保留：
- 用户明确目标、偏好、约束和仍然有效的上下文。
- 已完成或未完成任务的当前状态。
- 工具调用结果中对后续对话有用的结论。

删除无价值寒暄、重复内容、失败但无后续意义的中间过程。输出 Markdown，内容要紧凑。"""

LOCAL_SHELL_TOOL_SYSTEM_PROMPT = """本地 shell 工具使用规则：
- 当用户明确输入 shell 命令（例如 pwd、ls）或询问当前目录、工作目录、沙箱内文件状态时，如果 shell_exec 可用，应调用 shell_exec 获取真实结果。
- 不要在 shell_exec 可用时声称无法访问本地工作目录或无法查看沙箱内文件。
- 根据 shell_exec 返回的 stdout、stderr、blocked、reason 和 exit_code 向用户解释结果。"""


@dataclass
class AgentResult:
    """`AgentLoop.run` 和 `AgentLoop.arun` 返回的结果。"""

    messages: list[dict[str, Any]]
    output: str | None
    steps: int
    observations: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    """使用 OpenAI 兼容工具运行推理-行动-观察循环。"""

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry,
        max_steps: int = 8,
        tool_choice: str | dict[str, Any] = "auto",
        logger: EventLogger | None = None,
        tool_context: ToolContext | None = None,
        tool_executor: ToolExecutor | None = None,
        workspace: WorkspaceStore | None = None,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.tool_choice = tool_choice
        self.logger = logger if logger is not None else JsonlEventLogger()
        self.tool_context = tool_context
        self.workspace = workspace
        self.system_prompt = system_prompt
        if self.workspace is not None and not self.tools.has("save_user_profile"):
            self.tools.register(create_save_user_profile_tool(self.workspace))
        self.tool_executor = tool_executor or ToolExecutor(tools)
        self._openai_tools = self.tools.to_openai_tools()

    def run(self, messages: list[dict[str, Any]]) -> AgentResult:
        """同步运行智能体。

        如果已经位于事件循环中，请使用 `arun()`。
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(messages))
        raise RuntimeError("AgentLoop.run() 不能在运行中的事件循环内调用；请使用 arun()")

    async def arun(self, messages: list[dict[str, Any]]) -> AgentResult:
        working_messages = [dict(message) for message in messages]
        observations: list[dict[str, Any]] = []
        await self._compact_workspace_if_needed()
        short_term_messages = self._load_short_term_messages()
        self._record_user_inputs(working_messages)
        await self._log_user_message(working_messages)

        try:
            for step in range(1, self.max_steps + 1):
                assistant_message = await self._reason(working_messages, short_term_messages)
                working_messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    output = assistant_message.get("content")
                    self._record_session_turn(working_messages)
                    await self._log("final_answer", {"content": output, "steps": step})
                    return AgentResult(
                        messages=working_messages,
                        output=output,
                        steps=step,
                        observations=observations,
                    )

                tool_messages, step_observations = await self._act(tool_calls)
                working_messages.extend(tool_messages)
                observations.extend(step_observations)

            raise RuntimeError(f"智能体循环超过最大步数 max_steps={self.max_steps}")
        except Exception as exc:
            await self._log("error", {"type": type(exc).__name__, "message": str(exc)})
            raise

    async def _reason(
        self,
        messages: list[dict[str, Any]],
        short_term_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tools = self._openai_tools
        llm_messages = self._build_llm_messages(messages, short_term_messages or [])
        request = {
            "messages": llm_messages,
            "tools": tools,
            "tool_choice": self.tool_choice,
        }
        await self._log("llm_request", request)
        response = self.llm.create_chat_completion(
            messages=request["messages"],
            tools=tools,
            tool_choice=self.tool_choice,
        )
        response = await _maybe_await(response)
        message = normalize_assistant_message(response)
        await self._log("llm_response", {"message": message})
        return message

    def _build_llm_messages(
        self,
        messages: list[dict[str, Any]],
        short_term_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.workspace is None:
            system_prompt = self._build_system_prompt(messages)
            if not system_prompt:
                return deepcopy(messages)
            llm_messages = [{"role": "system", "content": system_prompt}]
            llm_messages.extend(
                deepcopy([message for message in messages if message.get("role") != "system"])
            )
            return llm_messages

        system_prompt = self._build_system_prompt(messages)
        llm_messages = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(deepcopy(short_term_messages))
        llm_messages.extend(
            deepcopy([message for message in messages if message.get("role") != "system"])
        )
        return llm_messages

    def _build_system_prompt(self, messages: list[dict[str, Any]]) -> str:
        prompt_parts = []
        if self.system_prompt:
            prompt_parts.append(self.system_prompt)
        if self.tools.has("shell_exec"):
            prompt_parts.append(LOCAL_SHELL_TOOL_SYSTEM_PROMPT)
        prompt_parts.extend(
            str(message.get("content"))
            for message in messages
            if message.get("role") == "system" and message.get("content")
        )
        base_prompt = "\n\n".join(prompt_parts) if prompt_parts else None
        if self.workspace is None:
            return base_prompt or ""
        return self.workspace.build_system_prompt(base_prompt)

    def _load_short_term_messages(self) -> list[dict[str, Any]]:
        if self.workspace is None:
            return []
        return self.workspace.load_context_messages()

    def _record_user_inputs(self, messages: list[dict[str, Any]]) -> None:
        if self.workspace is None:
            return
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                self.workspace.append_user_input(content)

    def _record_session_turn(self, messages: list[dict[str, Any]]) -> None:
        if self.workspace is None:
            return
        self.workspace.append_session_turn(
            [message for message in messages if message.get("role") != "system"]
        )

    async def _compact_workspace_if_needed(self) -> None:
        if self.workspace is None:
            return
        try:
            result = await self.workspace.compact_session_if_needed(
                self._summarize_short_term_memory
            )
            if result.get("compacted"):
                await self._log("memory_compacted", result)
        except Exception as exc:
            await self._log(
                "memory_summary_error",
                {"type": type(exc).__name__, "message": str(exc)},
            )

    async def _summarize_short_term_memory(
        self,
        previous_summary: str,
        discarded_turns: list[dict[str, Any]],
    ) -> str:
        payload = {
            "previous_summary": previous_summary or "",
            "discarded_turns": [
                turn.get("messages") or []
                for turn in discarded_turns
            ],
        }
        messages = [
            {
                "role": "system",
                "content": SHORT_TERM_SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]
        await self._log(
            "memory_summary_request",
            {
                "previous_summary_bytes": len(previous_summary.encode("utf-8")),
                "discarded_turns": len(discarded_turns),
            },
        )
        response = self.llm.create_chat_completion(messages=messages)
        response = await _maybe_await(response)
        message = normalize_assistant_message(response)
        summary = message.get("content") or ""
        await self._log(
            "memory_summary_response",
            {"summary_bytes": len(summary.encode("utf-8"))},
        )
        return summary

    async def _act(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tool_messages = []
        observations = []
        for tool_call in tool_calls:
            tool_message, observation = await self._execute_tool_call(tool_call)
            tool_messages.append(tool_message)
            observations.append(observation)
        return tool_messages, observations

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            request = normalize_tool_call(tool_call)
        except Exception as exc:
            function = tool_call.get("function") or {}
            request = None
            tool_call_id = tool_call.get("id")
            tool_name = function.get("name")
            result = {"error": str(exc)}
            await self._log(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "arguments": {},
                    "raw_arguments": function.get("arguments", "{}"),
                },
            )
        else:
            tool_call_id = request.id
            tool_name = request.name
            await self._log(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "arguments": request.arguments,
                },
            )
            execution = await self.tool_executor.aexecute(
                tool_name,
                request.arguments,
                self.tool_context,
            )
            result = execution.result

        await self._log(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "result": result,
            },
        )
        message = build_tool_message(
            tool_call_id=tool_call_id,
            name=tool_name,
            result=result,
        )
        observation = build_observation(
            tool_call_id=tool_call_id,
            name=tool_name,
            result=result,
        )
        return message, observation

    async def _log_user_message(self, messages: list[dict[str, Any]]) -> None:
        for message in reversed(messages):
            if message.get("role") == "user":
                await self._log("user_message", {"content": message.get("content")})
                return

    async def _log(self, event: str, data: dict[str, Any]) -> None:
        try:
            result = self.logger.log(event, data)
            await _maybe_await(result)
        except Exception:
            pass


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value
