"""单次运行的工具上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from AsyncClaw.tools.approval import ApprovalProvider


@dataclass(frozen=True)
class ToolContext:
    """单次智能体运行的能力与执行边界。"""

    cwd: Path
    allow_shell_exec: bool = False
    approval_mode: Literal["dangerous_only"] = "dangerous_only"
    shell_timeout_seconds: int = 10
    shell_output_limit_bytes: int = 16 * 1024
    approval_provider: ApprovalProvider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())
