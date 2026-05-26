"""工具注册与内置工具。"""

from AsyncClaw.tools.approval import ApprovalProvider, CliApprovalProvider
from AsyncClaw.tools.builtin.cron import create_cron_tools
from AsyncClaw.tools.builtin.math import multiply_tool
from AsyncClaw.tools.builtin.memory import create_save_user_profile_tool
from AsyncClaw.tools.builtin.shell import shell_exec_tool
from AsyncClaw.tools.builtin.time import current_time_tool
from AsyncClaw.tools.builtin.web import web_fetch_tool, web_search_tool
from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.executor import ToolExecution, ToolExecutor
from AsyncClaw.tools.mcp import MCPToolProvider, StreamableHTTPMCPClient
from AsyncClaw.tools.providers import (
    LocalToolProvider,
    ToolProvider,
    build_tool_registry_from_providers,
)
from AsyncClaw.tools.registry import build_tool_registry
from AsyncClaw.tools.registry import ToolRegistry
from AsyncClaw.tools.safety import SafetyDecision, check_shell_command, resolve_sandbox_path
from AsyncClaw.tools.spec import Tool, ToolHandler

__all__ = [
    "ApprovalProvider",
    "CliApprovalProvider",
    "SafetyDecision",
    "Tool",
    "ToolContext",
    "ToolExecution",
    "ToolExecutor",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "build_tool_registry",
    "build_tool_registry_from_providers",
    "check_shell_command",
    "create_cron_tools",
    "create_save_user_profile_tool",
    "current_time_tool",
    "LocalToolProvider",
    "MCPToolProvider",
    "multiply_tool",
    "resolve_sandbox_path",
    "shell_exec_tool",
    "StreamableHTTPMCPClient",
    "web_fetch_tool",
    "web_search_tool",
]
