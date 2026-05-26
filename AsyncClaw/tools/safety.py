"""本地工具执行的软沙箱安全策略辅助函数。"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DecisionAction = Literal["safe", "confirm", "deny"]


@dataclass(frozen=True)
class SafetyDecision:
    action: DecisionAction
    reason: str | None = None


_SHELL_OPERATORS = {"|", ";", "&&", "||", "&"}
_REDIRECT_OPERATORS = {"<", ">", ">>", "<<"}
_SENSITIVE_NAMES = {
    ".env",
    ".netrc",
    "authorized_keys",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "passwd",
    "secret",
    "secrets",
}
_CONFIRM_COMMANDS = {
    "chmod",
    "chown",
    "conda",
    "cp",
    "curl",
    "git",
    "mkdir",
    "mv",
    "npm",
    "pip",
    "pip3",
    "pnpm",
    "python",
    "python3",
    "rm",
    "rmdir",
    "sleep",
    "touch",
    "uv",
    "wget",
    "yarn",
}
_DENY_COMMANDS = {
    "dd",
    "halt",
    "mkfs",
    "node",
    "poweroff",
    "reboot",
    "shutdown",
    "sudo",
}


def check_shell_command(
    command: str,
    sandbox_root: Path | str | None = None,
) -> SafetyDecision:
    """对 shell 命令做 safe/confirm/deny 分级。"""

    sandbox_root = _coerce_sandbox_root(sandbox_root)
    if not command.strip():
        return SafetyDecision("deny", "命令必须是非空字符串")
    if "$(" in command or "`" in command:
        return SafetyDecision("deny", "命令包含绕过沙箱的 shell 展开")

    try:
        tokens = _tokenize(command)
    except ValueError as exc:
        return SafetyDecision("deny", str(exc))

    path_decision = _check_tokens_for_blocked_paths(tokens, sandbox_root)
    if path_decision is not None:
        return path_decision

    if any(token in {";", "&&", "||", "&"} for token in tokens):
        return SafetyDecision("deny", "复杂 shell 控制语法不允许")

    segments = _split_pipeline(tokens)
    if segments is None:
        return SafetyDecision("deny", "管道语法无效")

    bypass = _check_bypass_commands(segments)
    if bypass is not None:
        return bypass

    if _all_segments_are_safe(segments):
        return SafetyDecision("safe", "只读环境诊断命令")

    if any(token in _REDIRECT_OPERATORS for token in tokens):
        return SafetyDecision("confirm", "命令包含重定向，可能读写文件")

    if _has_confirm_command(segments):
        return SafetyDecision("confirm", "命令需要用户确认")

    return SafetyDecision("confirm", "命令不在安全白名单内")


def resolve_sandbox_path(path: str | Path, sandbox_root: Path | str) -> Path:
    """解析沙箱内路径，越界或敏感路径会抛出 ValueError。"""

    root = _coerce_sandbox_root(sandbox_root)
    raw_path = Path(path)
    text = str(path)
    if raw_path.is_absolute():
        raise ValueError("禁止访问绝对路径")
    if text == "~" or text.startswith("~/") or text.startswith("~" + "/"):
        raise ValueError("禁止访问用户主目录路径")
    if any(part == ".." for part in raw_path.parts):
        raise ValueError("禁止使用 .. 访问沙箱外路径")
    if _contains_sensitive_name(raw_path.parts):
        raise ValueError("禁止访问敏感文件")

    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("路径逃出 sandbox_root") from exc
    return resolved


def _coerce_sandbox_root(sandbox_root: Path | str | None) -> Path:
    root = Path(sandbox_root) if sandbox_root is not None else Path.cwd() / "workspace" / "office"
    return root.resolve()


def _tokenize(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
    lexer.whitespace_split = True
    return list(lexer)


def _split_pipeline(tokens: list[str]) -> list[list[str]] | None:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "|":
            if not segments[-1]:
                return None
            segments.append([])
            continue
        segments[-1].append(token)
    if not segments[-1]:
        return None
    return segments


def _check_bypass_commands(segments: list[list[str]]) -> SafetyDecision | None:
    for segment in segments:
        command = _command_name(segment)
        if command in _DENY_COMMANDS:
            return SafetyDecision("deny", f"命令 {command} 不允许在软沙箱中执行")
        if command in {"python", "python3"} and "-c" in segment[1:]:
            return SafetyDecision("deny", "禁止使用 python -c 绕过沙箱")
        if command == "node" and "-e" in segment[1:]:
            return SafetyDecision("deny", "禁止使用 node -e 绕过沙箱")
    return None


def _all_segments_are_safe(segments: list[list[str]]) -> bool:
    if not segments:
        return False
    if len(segments) == 1:
        return _is_safe_diagnostic(segments[0], allow_wc=False)
    return all(_is_safe_diagnostic(segment, allow_wc=True) for segment in segments)


def _is_safe_diagnostic(segment: list[str], *, allow_wc: bool) -> bool:
    if segment == ["pwd"]:
        return True
    if segment == ["ls"]:
        return True
    if segment == ["python", "--version"]:
        return True
    if segment == ["conda", "--version"]:
        return True
    if segment == ["conda", "env", "list"]:
        return True
    if segment == ["conda", "info", "--envs"]:
        return True
    if segment in (["which", "python"], ["which", "conda"]):
        return True
    if allow_wc and segment == ["wc", "-l"]:
        return True
    return False


def _has_confirm_command(segments: list[list[str]]) -> bool:
    return any(_command_name(segment) in _CONFIRM_COMMANDS for segment in segments)


def _command_name(segment: list[str]) -> str | None:
    if not segment:
        return None
    return Path(segment[0]).name


def _check_tokens_for_blocked_paths(
    tokens: list[str],
    sandbox_root: Path,
) -> SafetyDecision | None:
    for index, token in enumerate(tokens):
        if token in _SHELL_OPERATORS or token in _REDIRECT_OPERATORS:
            continue
        if _is_url(token):
            continue
        if _looks_like_fd(token, tokens, index):
            continue
        forbidden_path = _check_forbidden_path_syntax(token)
        if forbidden_path is not None:
            return forbidden_path
        if _is_option(token) and not _looks_like_path(token):
            continue
        if _is_command_position(tokens, index):
            command_path = _check_command_path(token, sandbox_root)
            if command_path is not None:
                return command_path
            continue
        if _is_redirection_target(tokens, index) or not _is_option(token):
            try:
                resolve_sandbox_path(token, sandbox_root)
            except ValueError as exc:
                return SafetyDecision("deny", str(exc))
        elif _contains_sensitive_name((token,)):
            return SafetyDecision("deny", "禁止访问敏感文件")
    return None


def _check_command_path(token: str, sandbox_root: Path) -> SafetyDecision | None:
    if token.startswith("/") or token.startswith("~"):
        return SafetyDecision("deny", "禁止通过路径执行沙箱外命令")
    if "/" not in token:
        return None
    try:
        resolve_sandbox_path(token, sandbox_root)
    except ValueError as exc:
        return SafetyDecision("deny", str(exc))
    return None


def _check_forbidden_path_syntax(token: str) -> SafetyDecision | None:
    if token.startswith("/") or "=/" in token:
        return SafetyDecision("deny", "禁止访问绝对路径")
    if token.startswith("~") or "=~" in token:
        return SafetyDecision("deny", "禁止访问用户主目录路径")
    if (
        token == ".."
        or token.startswith("../")
        or token.endswith("/..")
        or "/../" in token
        or "=../" in token
    ):
        return SafetyDecision("deny", "禁止使用 .. 访问沙箱外路径")
    return None


def _looks_like_path(token: str) -> bool:
    return (
        "/" in token
        or token.startswith(".")
        or token.startswith("~")
        or token in {"..", "."}
    )


def _is_command_position(tokens: list[str], index: int) -> bool:
    return index == 0 or tokens[index - 1] == "|"


def _is_redirection_target(tokens: list[str], index: int) -> bool:
    return index > 0 and tokens[index - 1] in _REDIRECT_OPERATORS


def _is_option(token: str) -> bool:
    return token.startswith("-")


def _is_url(token: str) -> bool:
    return token.startswith(("http://", "https://"))


def _looks_like_fd(token: str, tokens: list[str], index: int) -> bool:
    return token.isdigit() and index + 1 < len(tokens) and tokens[index + 1] in _REDIRECT_OPERATORS


def _contains_sensitive_name(parts: tuple[str, ...] | list[str]) -> bool:
    normalized = {part.lower() for part in parts}
    return bool(normalized & _SENSITIVE_NAMES)
