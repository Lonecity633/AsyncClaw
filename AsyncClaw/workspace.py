"""Workspace-backed session, history, and memory storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any


DEFAULT_SYSTEM_PROMPT = """你是 AsyncClaw 智能体。

你有一个长期记忆工具 save_user_profile，用来维护当前用户画像。每次回复前，先判断用户新消息是否包含值得长期保存的信息。

需要记录到 memory 的信息：
- 稳定身份背景：专业、职业、长期角色、所在地、常用语言。
- 稳定偏好和兴趣：喜欢/不喜欢的活动、技术栈、工具、交流风格。
- 长期目标和计划：学习方向、项目方向、职业目标。
- 常用环境和约束：常用 conda 环境、操作系统、项目路径、技术限制。
- 用户明确要求“记住”“以后都按这个来”“我的习惯是”等信息。

不要记录到 memory 的信息：
- 一次性问题、临时命令、当前任务进度、短期排错状态。
- 密码、token、API key、身份证号、银行卡等敏感凭据。
- 模糊、玩笑、反问或你不能确定为长期事实的内容。

如果需要记录：
1. 读取当前用户画像。
2. 合并新事实，去重并保留已有有用信息。
3. 调用 save_user_profile，传入更新后的完整 Markdown 用户画像。
4. 工具调用后再正常回复用户。

如果不需要记录，不要调用 save_user_profile。"""


@dataclass(frozen=True)
class WorkspaceStore:
    """Stores short-term sessions, user input history, and long-term profile."""

    root: Path | str = Path("workspace")
    session_id: str | None = None
    short_term_limit: int = 10

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "session_id", self.session_id or _new_session_id())
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_dir(self) -> Path:
        return self.root / "session"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def session_path(self) -> Path:
        return self.session_dir / f"{self.session_id}.jsonl"

    @property
    def user_inputs_path(self) -> Path:
        return self.history_dir / "user_inputs.jsonl"

    @property
    def user_profile_path(self) -> Path:
        return self.memory_dir / "user_profile.md"

    def load_recent_messages(self) -> list[dict[str, Any]]:
        records = self._read_jsonl(self.session_path)
        messages = [
            {"role": record["role"], "content": record.get("content")}
            for record in records
            if record.get("role") in {"user", "assistant"}
        ]
        return messages[-self.short_term_limit :]

    def append_session_message(self, role: str, content: Any) -> None:
        if role not in {"user", "assistant"}:
            return
        self._append_jsonl(
            self.session_path,
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "role": role,
                "content": content,
            },
        )

    def append_user_input(self, content: Any) -> None:
        self._append_jsonl(
            self.user_inputs_path,
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "content": content,
            },
        )

    def load_user_profile(self) -> str:
        if not self.user_profile_path.exists():
            return ""
        return self.user_profile_path.read_text(encoding="utf-8")

    def save_user_profile(self, profile_markdown: str) -> dict[str, Any]:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.user_profile_path.write_text(profile_markdown, encoding="utf-8")
        return {
            "path": str(self.user_profile_path),
            "bytes": len(profile_markdown.encode("utf-8")),
            "saved": True,
        }

    def build_system_prompt(self, base_prompt: str | None = None) -> str:
        prompt = base_prompt or DEFAULT_SYSTEM_PROMPT
        profile = self.load_user_profile().strip()
        if not profile:
            profile = "（暂无用户画像）"
        return f"{prompt}\n\n当前用户画像：\n{profile}"

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str))
            file.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
