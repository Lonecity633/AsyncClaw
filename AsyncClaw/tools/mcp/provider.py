"""Tool provider that exposes configured MCP servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from AsyncClaw.config import MCPServerConfig
from AsyncClaw.tools.mcp.naming import _local_tool_name
from AsyncClaw.tools.mcp.result import _normalize_tool_result
from AsyncClaw.tools.spec import Tool


@dataclass
class MCPToolProvider:
    """Discover and expose tools from configured MCP servers."""

    servers: tuple[MCPServerConfig, ...]
    include_cron_tools: bool = True

    def list_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        for server in self.servers:
            if not server.enabled:
                continue
            if not self.include_cron_tools and not server.allow_cron:
                continue
            try:
                tools.extend(_tools_for_server(server))
            except Exception:
                continue
        return tools

    def close(self) -> None:
        return None


def _tools_for_server(server: MCPServerConfig) -> list[Tool]:
    if server.transport != "streamable-http":
        raise ValueError(f"暂不支持 MCP transport：{server.transport}")
    if not server.url:
        raise ValueError(f"MCP server {server.name} 缺少 url")

    client_class = _streamable_http_client_class()
    client = client_class(
        url=server.url,
        headers=server.headers,
        timeout_seconds=server.timeout_seconds,
    )
    client.initialize()
    remote_tools = client.list_tools()
    return [_convert_tool(server, client, remote_tool) for remote_tool in remote_tools]


def _convert_tool(
    server: MCPServerConfig,
    client: Any,
    remote_tool: dict[str, Any],
) -> Tool:
    remote_name = str(remote_tool.get("name") or "").strip()
    if not remote_name:
        raise ValueError(f"MCP server {server.name} 返回了缺少 name 的工具")

    local_name = _local_tool_name(server, remote_name)
    description = (
        remote_tool.get("description")
        or remote_tool.get("title")
        or f"MCP tool {remote_name} from {server.name}."
    )
    schema = remote_tool.get("inputSchema") or {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}

    def call(arguments: dict[str, Any]) -> Any:
        return _normalize_tool_result(client.call_tool(remote_name, arguments))

    return Tool(
        name=local_name,
        description=str(description),
        schema=schema,
        handler=call,
    )


def _streamable_http_client_class() -> type:
    # Keep patching `AsyncClaw.tools.mcp.StreamableHTTPMCPClient` effective for tests
    # and callers that treated the old single-file module as the public patch point.
    from AsyncClaw.tools import mcp as public_mcp

    return public_mcp.StreamableHTTPMCPClient
