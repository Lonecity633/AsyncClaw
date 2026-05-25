"""Workspace-backed session, history, and memory storage."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable


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
    summary_threshold: int = 40
    recent_turn_limit: int = 10

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
    def short_term_summary_path(self) -> Path:
        return self.session_dir / f"{self.session_id}.summary.md"

    @property
    def user_inputs_path(self) -> Path:
        return self.history_dir / "user_inputs.jsonl"

    @property
    def user_profile_path(self) -> Path:
        return self.memory_dir / "user_profile.md"

    def load_session_turns(self) -> list[dict[str, Any]]:
        records = self._read_jsonl(self.session_path)
        turns: list[dict[str, Any]] = []
        for record in records:
            messages = record.get("messages")
            if isinstance(messages, list):
                turns.append(
                    {
                        "timestamp": record.get("timestamp") or _utc_now(),
                        "session_id": record.get("session_id") or self.session_id,
                        "messages": _copy_messages(messages),
                    }
                )
        return turns

    def load_context_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        summary = self.load_short_term_summary().strip()
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"近期对话摘要：\n{summary}",
                }
            )
        for turn in self.load_session_turns():
            messages.extend(_copy_messages(turn.get("messages") or []))
        return messages

    def append_session_turn(self, messages: list[dict[str, Any]]) -> None:
        session_messages = [
            message
            for message in _copy_messages(messages)
            if message.get("role") in {"user", "assistant", "tool"}
        ]
        if not session_messages:
            return
        self._append_jsonl(
            self.session_path,
            {
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "messages": session_messages,
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

    def load_short_term_summary(self) -> str:
        if not self.short_term_summary_path.exists():
            return ""
        return self.short_term_summary_path.read_text(encoding="utf-8")

    def save_short_term_summary(self, summary: str) -> dict[str, Any]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.short_term_summary_path.write_text(summary, encoding="utf-8")
        return {
            "path": str(self.short_term_summary_path),
            "bytes": len(summary.encode("utf-8")),
            "saved": True,
        }

    async def compact_session_if_needed(
        self,
        summarizer: Callable[[str, list[dict[str, Any]]], Any],
    ) -> dict[str, Any]:
        turns = self.load_session_turns()
        if len(turns) < self.summary_threshold:
            return {
                "compacted": False,
                "reason": "below_threshold",
                "turn_count": len(turns),
            }

        discard_count = max(0, len(turns) - self.recent_turn_limit)
        if discard_count == 0:
            return {
                "compacted": False,
                "reason": "nothing_to_discard",
                "turn_count": len(turns),
            }

        discarded_turns = turns[:discard_count]
        kept_turns = turns[discard_count:]
        summary = summarizer(self.load_short_term_summary(), discarded_turns)
        if isawaitable(summary):
            summary = await summary
        if not isinstance(summary, str):
            raise TypeError("短期摘要生成器必须返回字符串")

        self.save_short_term_summary(summary.strip())
        self._write_jsonl(self.session_path, kept_turns)
        return {
            "compacted": True,
            "discarded_turns": len(discarded_turns),
            "kept_turns": len(kept_turns),
        }

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

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
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


def _copy_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [deepcopy(message) for message in messages if isinstance(message, dict)]
