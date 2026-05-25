"""工具执行边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolExecution:
    """单次工具调用的结构化结果。"""

    name: str | None
    result: Any


class ToolExecutor:
    """执行已注册工具，并将失败转换为观察结果。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(
        self,
        name: str | None,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolExecution:
        if not name:
            return ToolExecution(name=name, result={"error": "缺少工具函数名称"})
        try:
            result = self.registry.call(name, arguments, context)
        except Exception as exc:
            result = {"error": str(exc)}
        return ToolExecution(name=name, result=result)

    async def aexecute(
        self,
        name: str | None,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolExecution:
        if not name:
            return ToolExecution(name=name, result={"error": "缺少工具函数名称"})
        try:
            result = await self.registry.acall(name, arguments, context)
        except Exception as exc:
            result = {"error": str(exc)}
        return ToolExecution(name=name, result=result)
