"""本地工具执行的安全策略辅助函数。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DecisionAction = Literal["allow", "needs_approval", "deny"]


@dataclass(frozen=True)
class SafetyDecision:
    action: DecisionAction
    reason: str | None = None


_SAFE_COMMANDS = {
    "cat",
    "echo",
    "find",
    "grep",
    "head",
    "ls",
    "printf",
    "pwd",
    "rg",
    "sed",
    "tail",
    "wc",
}

_DENY_PATTERNS = [
    (
        re.compile(
            r"(^|[;&|]\s*)(sudo\s+)?([\w./-]+/)?"
            r"(rm|rmdir|mkfs|shutdown|reboot|halt|poweroff|dd)\b"
        ),
        "破坏性命令",
    ),
    (
        re.compile(r"(^|[;&|]\s*)(sudo\s+)?([\w./-]+/)?git\s+reset\b"),
        "破坏性 git reset",
    ),
    (re.compile(r"\bchmod\s+-R\s+777\b"), "不安全的递归 chmod"),
    (re.compile(r">\s*/(bin|etc|sbin|usr|var|System|Library)\b"), "写入受保护路径"),
]


def check_shell_command(command: str) -> SafetyDecision:
    for pattern, reason in _DENY_PATTERNS:
        if pattern.search(command):
            return SafetyDecision("deny", reason)

    command_names = _extract_command_names(command)
    if command_names and all(name in _SAFE_COMMANDS for name in command_names):
        return SafetyDecision("allow")

    return SafetyDecision("needs_approval", "命令不在安全白名单内")


def _extract_command_names(command: str) -> list[str]:
    normalized = re.sub(r"\s*(\|\||&&|[;|])\s*", r" \1 ", command)
    tokens = normalized.split()
    names: list[str] = []
    expect_command = True
    for token in tokens:
        if token in {";", "|", "&&", "||"}:
            expect_command = True
            continue
        if expect_command:
            if "=" in token and not token.startswith(("/", "./")):
                continue
            names.append(token.split("/")[-1])
            expect_command = False
    return names
