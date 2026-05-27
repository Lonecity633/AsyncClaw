"""Workspace skill discovery and progressive prompt rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path


SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Skill:
    """A workspace skill described by SKILL.md frontmatter."""

    name: str
    description: str
    path: Path
    base_dir: Path
    body: str
    valid: bool = True
    error: str | None = None


def load_workspace_skills(
    skills_dir: Path | str,
    *,
    include_invalid: bool = False,
) -> list[Skill]:
    """Load one-level workspace skills from ``<skills_dir>/<name>/SKILL.md``."""

    root = Path(skills_dir)
    if not root.exists() or not root.is_dir():
        return []

    skills: list[Skill] = []
    skill_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for skill_dir in skill_dirs:
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.is_file():
            continue
        skill = load_skill_file(skill_path)
        if skill.valid or include_invalid:
            skills.append(skill)
    return sorted(skills, key=lambda skill: skill.name)


def load_skill_file(path: Path | str) -> Skill:
    """Parse a single SKILL.md file."""

    skill_path = Path(path).resolve()
    base_dir = skill_path.parent.resolve()
    fallback_name = base_dir.name
    try:
        text = skill_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return _invalid_skill(fallback_name, skill_path, base_dir, f"读取失败: {exc}")

    parsed = _parse_frontmatter(text)
    if parsed is None:
        return _invalid_skill(fallback_name, skill_path, base_dir, "缺少 frontmatter")
    metadata, body = parsed

    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    body = body.strip()
    if not name:
        return _invalid_skill(fallback_name, skill_path, base_dir, "缺少 name")
    if not SKILL_NAME_RE.fullmatch(name):
        return _invalid_skill(name, skill_path, base_dir, "name 必须是 kebab-case")
    if name != fallback_name:
        return _invalid_skill(name, skill_path, base_dir, "name 必须与 skill 目录名一致")
    if not description:
        return _invalid_skill(name, skill_path, base_dir, "缺少 description")
    if not body:
        return _invalid_skill(name, skill_path, base_dir, "正文不能为空")

    return Skill(
        name=name,
        description=description,
        path=skill_path,
        base_dir=base_dir,
        body=body,
    )


def build_skills_catalog(skills: list[Skill]) -> str:
    """Render a compact skills catalog for the system prompt."""

    valid_skills = [skill for skill in skills if skill.valid]
    if not valid_skills:
        return ""
    lines = ["<available_skills>"]
    for skill in valid_skills:
        lines.append(
            f'  <skill name="{escape(skill.name)}" '
            f'description="{escape(skill.description)}" />'
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str] | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return None

    metadata: dict[str, str] = {}
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = _strip_quotes(value.strip())
    body = "\n".join(lines[end_index + 1 :])
    return metadata, body


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _invalid_skill(name: str, path: Path, base_dir: Path, error: str) -> Skill:
    return Skill(
        name=name,
        description="",
        path=path,
        base_dir=base_dir,
        body="",
        valid=False,
        error=error,
    )
