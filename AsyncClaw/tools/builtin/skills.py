"""Workspace skill loading tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.tools.spec import Tool


MAX_SKILL_RESOURCE_BYTES = 64 * 1024
SENSITIVE_RESOURCE_NAMES = {
    ".env",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}


def create_load_skill_tool(workspace: WorkspaceStore) -> Tool:
    """Create a tool that progressively loads workspace skill content."""

    def _load_skill(arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return {"loaded": False, "error": "name 必须是非空字符串"}
        name = name.strip()

        skill = next((item for item in workspace.load_skills() if item.name == name), None)
        if skill is None:
            return {"loaded": False, "name": name, "error": "未知 skill"}

        resource_path = arguments.get("resource_path")
        if resource_path is None or resource_path == "":
            return {
                "loaded": True,
                "name": skill.name,
                "description": skill.description,
                "path": str(skill.path),
                "content": skill.body,
            }
        if not isinstance(resource_path, str):
            return {"loaded": False, "name": skill.name, "error": "resource_path 必须是字符串"}
        return _load_skill_resource(skill.base_dir, skill.name, resource_path)

    return Tool(
        name="load_skill",
        description=(
            "按需加载 workspace skill 的完整说明或 skill 目录内的文本资源。"
            "当用户任务匹配 available_skills 中某个 description 时，"
            "先用 name 调用本工具读取 SKILL.md 正文；"
            "如正文要求读取 references/ 下的资料，再传入 resource_path。"
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要加载的 skill name。",
                },
                "resource_path": {
                    "type": "string",
                    "description": "可选。skill 目录内的相对文本资源路径。",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_load_skill,
    )


def _load_skill_resource(base_dir: Path, skill_name: str, resource_path: str) -> dict[str, Any]:
    raw_path = Path(resource_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "resource_path 必须是 skill 目录内的相对路径",
        }
    if raw_path.name in SENSITIVE_RESOURCE_NAMES:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "拒绝读取敏感文件名",
        }

    base = base_dir.resolve()
    target = (base / raw_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "resource_path 越界",
        }
    if not target.is_file():
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "资源不存在",
        }

    try:
        payload = target.read_bytes()
    except OSError as exc:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": f"读取失败: {exc}",
        }
    if len(payload) > MAX_SKILL_RESOURCE_BYTES:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "资源超过大小限制",
        }
    if b"\x00" in payload:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "拒绝读取二进制资源",
        }
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "loaded": False,
            "name": skill_name,
            "resource_path": resource_path,
            "error": "资源必须是 UTF-8 文本",
        }
    return {
        "loaded": True,
        "name": skill_name,
        "resource_path": resource_path,
        "path": str(target),
        "content": content,
    }
