"""Normalize MCP tool call results for AsyncClaw observations."""

from __future__ import annotations

from typing import Any


def _normalize_tool_result(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        return {"error": _content_to_text(result.get("content")) or result}
    if "structuredContent" in result:
        return result["structuredContent"]
    if "content" in result:
        content = result["content"]
        if _is_single_text_content(content):
            return content[0]["text"]
        return content
    return result


def _is_single_text_content(content: Any) -> bool:
    return (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get("type") == "text"
        and "text" in content[0]
    )


def _content_to_text(content: Any) -> str | None:
    if _is_single_text_content(content):
        return str(content[0]["text"])
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts) if parts else None
    return None
