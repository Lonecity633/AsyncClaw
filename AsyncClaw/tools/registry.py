"""工具注册表与上下文感知的注册表构造器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.spec import Tool
from AsyncClaw.agent.workspace import WorkspaceStore


class ToolRegistry:
    """函数工具的内存注册表。"""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未知工具：{name}") from exc

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> Any:
        return self.get(name).call(arguments, context)

    async def acall(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> Any:
        return await self.get(name).acall(arguments, context)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]


def build_tool_registry(
    context: ToolContext | None = None,
    workspace: WorkspaceStore | None = None,
) -> ToolRegistry:
    """构造单次智能体运行可见的工具集合。"""

    from AsyncClaw.tools.builtin.math import multiply_tool
    from AsyncClaw.tools.builtin.memory import create_save_user_profile_tool
    from AsyncClaw.tools.builtin.shell import shell_exec_tool
    from AsyncClaw.tools.builtin.time import current_time_tool

    context = context or ToolContext(cwd=Path.cwd())
    tools = [multiply_tool, current_time_tool]
    if workspace is not None:
        tools.append(create_save_user_profile_tool(workspace))
    if context.allow_shell_exec:
        tools.append(shell_exec_tool)
    return ToolRegistry(tools)
