"""工具基础类型的兼容导入。"""

from AsyncClaw.tools.registry import ToolRegistry
from AsyncClaw.tools.spec import Tool, ToolHandler

__all__ = ["Tool", "ToolHandler", "ToolRegistry"]
