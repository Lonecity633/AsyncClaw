"""需要运行时确认的工具审批提供者。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ApprovalProvider(Protocol):
    """询问一个潜在风险工具操作是否可以继续。"""

    def approve(self, *, command: str, cwd: Path, reason: str | None = None) -> bool:
        ...


class CliApprovalProvider:
    """本地 CLI 的交互式审批提供者。"""

    def approve(self, *, command: str, cwd: Path, reason: str | None = None) -> bool:
        print("shell_exec 需要审批。")
        print(f"工作目录：{cwd}")
        print(f"命令：{command}")
        if reason:
            print(f"原因：{reason}")
        answer = input("是否执行该命令？[是/否] ").strip().lower()
        return answer in {"y", "yes", "是", "确认", "执行"}
