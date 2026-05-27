"""MCP tool name mapping helpers."""

from __future__ import annotations

import re

from AsyncClaw.config import MCPServerConfig


def _local_tool_name(server: MCPServerConfig, remote_name: str) -> str:
    name = _normalize_name(remote_name)
    prefix = server.tool_prefix
    if prefix is None:
        prefix = server.name
    if prefix is False or prefix == "":
        return name
    return f"{_normalize_name(str(prefix))}_{name}"


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return normalized or "mcp_tool"
