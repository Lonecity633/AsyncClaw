"""工具规格与调用辅助函数。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import Parameter, isawaitable, signature
from typing import Any, Callable

from AsyncClaw.tools.context import ToolContext

ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    """通过 OpenAI 兼容 schema 暴露的函数工具。"""

    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    def call(self, arguments: dict[str, Any], context: ToolContext | None = None) -> Any:
        result = _invoke_handler(self.handler, arguments, context)
        if isawaitable(result):
            return _run_awaitable(result)
        return result

    async def acall(
        self,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> Any:
        result = _invoke_handler(self.handler, arguments, context)
        if isawaitable(result):
            return await result
        return result


def _invoke_handler(
    handler: ToolHandler,
    arguments: dict[str, Any],
    context: ToolContext | None,
) -> Any:
    if _accepts_context(handler):
        return handler(arguments, context)
    return handler(arguments)


def _run_awaitable(value: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    if hasattr(value, "close"):
        value.close()
    raise RuntimeError("不能在运行中的事件循环内同步调用异步工具；请使用 acall()")


def _accepts_context(handler: ToolHandler) -> bool:
    try:
        parameters = signature(handler).parameters.values()
    except (TypeError, ValueError):
        return True

    positional = {
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    }
    count = 0
    for parameter in parameters:
        if parameter.kind == Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in positional:
            count += 1
    return count >= 2
