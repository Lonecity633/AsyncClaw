"""工具注册与内置工具。"""

from AsyncClaw.tools.approval import ApprovalProvider, CliApprovalProvider
from AsyncClaw.tools.builtin.math import multiply_tool
from AsyncClaw.tools.builtin.shell import shell_exec_tool
from AsyncClaw.tools.builtin.time import current_time_tool
from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.executor import ToolExecution, ToolExecutor
from AsyncClaw.tools.registry import build_tool_registry
from AsyncClaw.tools.registry import ToolRegistry
from AsyncClaw.tools.safety import SafetyDecision, check_shell_command
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
    "ToolRegistry",
    "build_tool_registry",
    "check_shell_command",
    "current_time_tool",
    "multiply_tool",
    "shell_exec_tool",
]
