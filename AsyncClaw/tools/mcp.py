"""MCP tool provider adapters."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from AsyncClaw.config import MCPServerConfig
from AsyncClaw.tools.spec import Tool

MCP_PROTOCOL_VERSION = "2025-06-18"


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


class StreamableHTTPMCPClient:
    """Small JSON-RPC client for MCP Streamable HTTP endpoints."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.session_id: str | None = None
        self._next_id = 1

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "AsyncClaw", "version": "0.1.0"},
            },
        )
        try:
            self.notify("notifications/initialized", {})
        except Exception:
            pass

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else None
            result = self.request("tools/list", params)
            page = result.get("tools") if isinstance(result, dict) else None
            if isinstance(page, list):
                tools.extend(tool for tool in page if isinstance(tool, dict))
            cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            name_header=name,
        )

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        name_header: str | None = None,
    ) -> Any:
        request_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = self._post(payload, method=method, name_header=name_header)
        if "error" in response:
            error = response["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"MCP {method} 调用失败：{message}")
        return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._post(payload, method=method, expect_response=False)

    def _post(
        self,
        payload: dict[str, Any],
        *,
        method: str,
        name_header: str | None = None,
        expect_response: bool = True,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": method,
            **self.headers,
        }
        if name_header:
            headers["Mcp-Name"] = name_header
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        request = urllib.request.Request(
            self.url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MCP HTTP 连接失败：{exc.reason}") from exc

        if not expect_response or not body:
            return {}
        return _decode_response(body, content_type)


def _tools_for_server(server: MCPServerConfig) -> list[Tool]:
    if server.transport != "streamable-http":
        raise ValueError(f"暂不支持 MCP transport：{server.transport}")
    if not server.url:
        raise ValueError(f"MCP server {server.name} 缺少 url")

    client = StreamableHTTPMCPClient(
        url=server.url,
        headers=server.headers,
        timeout_seconds=server.timeout_seconds,
    )
    client.initialize()
    remote_tools = client.list_tools()
    return [_convert_tool(server, client, remote_tool) for remote_tool in remote_tools]


def _convert_tool(
    server: MCPServerConfig,
    client: StreamableHTTPMCPClient,
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
