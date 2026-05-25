"""由上下文控制暴露的 shell 执行工具。"""

from __future__ import annotations

import subprocess
from typing import Any

from AsyncClaw.tools.approval import CliApprovalProvider
from AsyncClaw.tools.context import ToolContext
from AsyncClaw.tools.safety import check_shell_command
from AsyncClaw.tools.spec import Tool


def _shell_exec(arguments: dict[str, Any], context: ToolContext | None) -> dict[str, Any]:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return _blocked_result(command="", context=context, reason="命令必须是非空字符串")
    command = command.strip()

    if context is None:
        return _blocked_result(command=command, context=context, reason="缺少工具上下文")
    if not context.allow_shell_exec:
        return _blocked_result(command=command, context=context, reason="当前上下文不允许 shell_exec")

    decision = check_shell_command(command, context.sandbox_root)
    if decision.action == "deny":
        return _blocked_result(command=command, context=context, reason=decision.reason)

    approved = decision.action == "safe"
    if decision.action == "confirm":
        provider = context.approval_provider or CliApprovalProvider()
        approved = provider.approve(
            command=command,
            cwd=context.sandbox_root,
            reason=decision.reason,
        )
        if not approved:
            return _blocked_result(
                command=command,
                context=context,
                reason="命令未通过审批",
                approved=False,
            )

    return _run_shell_command(command, context=context, approved=approved)


def _run_shell_command(
    command: str,
    *,
    context: ToolContext,
    approved: bool,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=context.sandbox_root,
            capture_output=True,
            text=True,
            timeout=context.shell_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(context.sandbox_root),
            "exit_code": None,
            "stdout": _limit_text(_coerce_output(exc.stdout), context.shell_output_limit_bytes),
            "stderr": _limit_text(_coerce_output(exc.stderr), context.shell_output_limit_bytes),
            "timed_out": True,
            "approved": approved,
            "blocked": False,
            "reason": f"命令执行超过 {context.shell_timeout_seconds} 秒后超时",
        }

    return {
        "command": command,
        "cwd": str(context.sandbox_root),
        "exit_code": completed.returncode,
        "stdout": _limit_text(completed.stdout, context.shell_output_limit_bytes),
        "stderr": _limit_text(completed.stderr, context.shell_output_limit_bytes),
        "timed_out": False,
        "approved": approved,
        "blocked": False,
        "reason": None,
    }


def _blocked_result(
    *,
    command: str,
    context: ToolContext | None,
    reason: str | None,
    approved: bool = False,
) -> dict[str, Any]:
    return {
        "command": command,
        "cwd": str(context.sandbox_root) if context else None,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "approved": approved,
        "blocked": True,
        "reason": reason,
    }


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _limit_text(text: str, limit: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    truncated = encoded[:limit].decode("utf-8", errors="replace")
    return f"{truncated}\n[已截断到 {limit} 字节]"


shell_exec_tool = Tool(
    name="shell_exec",
    description="在当前工具上下文中通过安全检查后执行 shell 命令。",
    schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令。",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    handler=_shell_exec,
)
