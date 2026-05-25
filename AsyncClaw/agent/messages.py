"""OpenAI 兼容消息的归一化辅助函数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    """模型请求的工具调用归一化结果。"""

    id: str | None
    name: str | None
    arguments: dict[str, Any]
    raw_arguments: Any = None


def normalize_assistant_message(response: Any) -> dict[str, Any]:
    """将 OpenAI SDK 对象或字典响应归一化为消息字典。"""

    choice = _get(_get(response, "choices")[0], "message")
    return _message_to_dict(choice)


def normalize_tool_call(tool_call: dict[str, Any]) -> ToolCallRequest:
    function = tool_call.get("function") or {}
    raw_arguments = function.get("arguments", "{}")
    return ToolCallRequest(
        id=tool_call.get("id"),
        name=function.get("name"),
        arguments=parse_tool_arguments(raw_arguments),
        raw_arguments=raw_arguments,
    )


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("工具参数必须解析为 JSON 对象")
        return parsed
    raise TypeError("工具参数必须是 JSON 字符串或字典")


def build_tool_message(
    *,
    tool_call_id: str | None,
    name: str | None,
    result: Any,
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def build_observation(
    *,
    tool_call_id: str | None,
    name: str | None,
    result: Any,
) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call_id,
        "name": name,
        "result": result,
    }


def _message_to_dict(message: Any) -> dict[str, Any]:
    content = _get(message, "content", None)
    tool_calls = _get(message, "tool_calls", None)

    normalized: dict[str, Any] = {
        "role": _get(message, "role", "assistant"),
        "content": content,
    }
    if tool_calls:
        normalized["tool_calls"] = [_tool_call_to_dict(call) for call in tool_calls]
    return normalized


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    function = _get(tool_call, "function")
    return {
        "id": _get(tool_call, "id"),
        "type": _get(tool_call, "type", "function"),
        "function": {
            "name": _get(function, "name"),
            "arguments": _get(function, "arguments", "{}"),
        },
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
