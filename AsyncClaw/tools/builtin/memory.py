"""Workspace memory tools."""

from __future__ import annotations

from typing import Any

from AsyncClaw.tools.spec import Tool
from AsyncClaw.agent.workspace import WorkspaceStore


def create_save_user_profile_tool(workspace: WorkspaceStore) -> Tool:
    """Create a tool that overwrites the workspace user profile."""

    def _save_user_profile(arguments: dict[str, Any]) -> dict[str, Any]:
        profile = arguments.get("profile_markdown")
        if not isinstance(profile, str):
            return {"saved": False, "error": "profile_markdown 必须是字符串"}
        return workspace.save_user_profile(profile)

    return Tool(
        name="save_user_profile",
        description="覆盖写入 workspace/memory/user_profile.md，用于保存完整的长期用户画像。",
        schema={
            "type": "object",
            "properties": {
                "profile_markdown": {
                    "type": "string",
                    "description": "更新后的完整 Markdown 用户画像。",
                }
            },
            "required": ["profile_markdown"],
            "additionalProperties": False,
        },
        handler=_save_user_profile,
    )
