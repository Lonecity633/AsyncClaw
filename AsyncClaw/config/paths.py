"""Shared path resolution helpers for runtime configuration."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for an editable AsyncClaw checkout."""

    return Path(__file__).resolve().parents[2]


def resolve_env_file(
    cwd: str | Path,
    path: str | Path = ".env",
    *,
    explicit: bool = False,
    root: str | Path | None = None,
) -> Path:
    """Resolve the dotenv file using CLI-compatible precedence rules."""

    cwd_path = Path(cwd)
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    candidate = cwd_path / resolved
    if explicit or resolved != Path(".env"):
        return candidate

    if candidate.exists():
        return candidate

    fallback = Path(root) / ".env" if root is not None else project_root() / ".env"
    if fallback.exists():
        return fallback

    return candidate


def resolve_workspace_root(
    cwd: str | Path,
    workspace_root: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve the state workspace root."""

    if workspace_root is None:
        base = Path(root) if root is not None else project_root()
        return (base / "workspace").resolve()
    resolved = Path(workspace_root)
    if resolved.is_absolute():
        return resolved.resolve()
    return (Path(cwd).resolve() / resolved).resolve()


def resolve_log_path(
    workspace_root: str | Path | None,
    resolved_workspace_root: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve the JSONL event log path."""

    if workspace_root is None:
        base = Path(root) if root is not None else project_root()
        return (base / "logs" / "events.jsonl").resolve()
    return (Path(resolved_workspace_root) / "logs" / "events.jsonl").resolve()


def resolve_dotenv_relative_path(path: str | Path, env_path: str | Path) -> Path:
    """Resolve a path declared inside a dotenv file relative to that file."""

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    base_dir = Path(env_path).parent
    if base_dir == Path("."):
        base_dir = Path.cwd()
    return base_dir / resolved
