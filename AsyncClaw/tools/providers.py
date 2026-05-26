"""Composable tool providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.config import MCPConfig
from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.mcp import MCPToolProvider
from AsyncClaw.tools.registry import ToolRegistry, build_tool_registry
from AsyncClaw.tools.spec import Tool


class ToolProvider(Protocol):
    """A source that can provide AsyncClaw tools."""

    def list_tools(self) -> list[Tool]:
        """Return tools exposed by this provider."""

    def close(self) -> None:
        """Release provider resources."""


@dataclass
class LocalToolProvider:
    """Expose AsyncClaw's built-in local tools."""

    context: ToolContext | None = None
    workspace: WorkspaceStore | None = None

    def list_tools(self) -> list[Tool]:
        context = self.context or ToolContext(cwd=Path.cwd())
        return build_tool_registry(context, workspace=self.workspace).tools()

    def close(self) -> None:
        return None


def build_tool_registry_from_providers(
    *,
    context: ToolContext | None = None,
    workspace: WorkspaceStore | None = None,
    mcp_config: MCPConfig | None = None,
    include_cron_tools: bool = True,
    providers: list[ToolProvider] | None = None,
) -> ToolRegistry:
    """Build a registry from local tools plus optional external providers."""

    registry = ToolRegistry()
    selected_providers: list[ToolProvider] = providers or [
        LocalToolProvider(context=context, workspace=workspace)
    ]
    if mcp_config is not None and mcp_config.servers:
        selected_providers.append(
            MCPToolProvider(
                servers=mcp_config.servers,
                include_cron_tools=include_cron_tools,
            )
        )

    for provider in selected_providers:
        for tool in provider.list_tools():
            if registry.has(tool.name):
                continue
            registry.register(tool)
    return registry
