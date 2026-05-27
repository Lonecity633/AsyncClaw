"""MCP HTTP and SSE response decoding."""

from __future__ import annotations

import json
from typing import Any


def _decode_response(body: bytes, content_type: str) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type:
        return _decode_sse_response(text)
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise RuntimeError("MCP 响应必须是 JSON-RPC 对象")
    return decoded


def _decode_sse_response(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        decoded = json.loads(data)
        if isinstance(decoded, dict):
            return decoded
    raise RuntimeError("MCP SSE 响应缺少 JSON-RPC data")
