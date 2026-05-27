"""Streamable HTTP JSON-RPC client for MCP servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from AsyncClaw.tools.mcp.constants import MCP_PROTOCOL_VERSION
from AsyncClaw.tools.mcp.response import _decode_response


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
