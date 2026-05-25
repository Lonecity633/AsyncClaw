"""Rich-powered interactive agent CLI."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from AsyncClaw.channels.service import AgentService


EXIT_COMMANDS = {"exit", "quit"}
ANSI_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]")

try:
    import readline  # noqa: F401
except ImportError:
    readline = None


def run_agent_cli(
    *,
    cwd: str | Path | None = None,
    env_file: str | Path = ".env",
    env_file_explicit: bool = False,
    workspace_root: str | Path | None = None,
    allow_shell_exec: bool = True,
    console: Console | None = None,
) -> int:
    """Run the interactive CLI agent."""

    console = console or Console()
    service = AgentService(
        cwd=cwd,
        env_file=env_file,
        env_file_explicit=env_file_explicit,
        workspace_root=workspace_root,
        allow_shell_exec=allow_shell_exec,
    )
    _render_startup(console, service)

    while True:
        try:
            user_text = _read_user_text(console)
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print("[yellow]已退出 AsyncClaw。[/]")
            return 0

        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            console.print("[yellow]已退出 AsyncClaw。[/]")
            return 0

        try:
            with console.status("[bold green]AsyncClaw 思考中...[/]", spinner="dots"):
                response = service.handle_text(user_text)
        except Exception as exc:
            console.print(
                Panel(
                    Text(str(exc), style="red"),
                    title="错误",
                    border_style="red",
                )
            )
            continue

        _render_response(console, response.output)


def _render_startup(console: Console, service: AgentService) -> None:
    config = service.config
    provider = config.provider if config is not None else "custom"
    model = config.model if config is not None else "custom"
    shell_status = "enabled" if service.tool_context.allow_shell_exec else "disabled"
    body = (
        f"[bold]cwd[/]: {service.cwd}\n"
        f"[bold]workspace[/]: {service.workspace.root}\n"
        f"[bold]session[/]: {service.workspace.session_id}\n"
        f"[bold]provider[/]: {provider}\n"
        f"[bold]model[/]: {model}\n"
        f"[bold]env[/]: {service.env_file_path}\n"
        f"[bold]shell_exec[/]: {shell_status}\n\n"
        "输入 [bold]exit[/] 或 [bold]quit[/] 退出。"
    )
    console.print(
        Panel.fit(
            body,
            title="[bold cyan]AsyncClaw Agent[/]",
            border_style="cyan",
        )
    )


def _render_response(console: Console, output: str | None) -> None:
    console.print(Rule("[bold green]助手[/]", style="green"))
    console.print(Markdown(output or "（无输出）"))
    console.print()


def _read_user_text(console: Console) -> str:
    raw_text = Prompt.ask("[bold cyan]用户[/]", console=console)
    return _normalize_user_input(raw_text).strip()


def _normalize_user_input(raw_text: str) -> str:
    text = ANSI_SEQUENCE_RE.sub("", raw_text)
    chars: list[str] = []
    for char in text:
        if char in {"\b", "\x7f"}:
            if chars:
                chars.pop()
            continue
        if char == "\x15":
            chars.clear()
            continue
        if char.isprintable() or char.isspace():
            chars.append(char)
    return "".join(chars)
