"""Rich-powered interactive agent CLI."""

from __future__ import annotations

from contextlib import nullcontext
import re
import threading
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from AsyncClaw.agent.cron import CronJob
from AsyncClaw.channels.service import AgentService
from AsyncClaw.cli.logo import render_pixel_logo

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
except ImportError:  # pragma: no cover - dependency is declared for installed CLI use.
    PromptSession = None
    patch_stdout = None


EXIT_COMMANDS = {"/exit", "exit", "quit"}
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
    allow_cron: bool = True,
    cron_max_concurrent_jobs: int = 2,
    console: Console | None = None,
) -> int:
    """Run the interactive CLI agent."""

    console = console or Console()
    render_lock = threading.RLock()
    prompt_session = PromptSession() if PromptSession is not None else None
    service = AgentService(
        cwd=cwd,
        env_file=env_file,
        env_file_explicit=env_file_explicit,
        workspace_root=workspace_root,
        allow_shell_exec=allow_shell_exec,
        allow_cron=allow_cron,
        cron_max_concurrent_jobs=cron_max_concurrent_jobs,
        on_cron_job_start=lambda job: _render_cron_start(console, job, lock=render_lock),
        on_cron_job_result=lambda result: _render_cron_result(console, result, lock=render_lock),
    )
    _render_startup(console, service, lock=render_lock)

    try:
        while True:
            try:
                user_text = _read_user_text(console, session=prompt_session)
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
                with _render_context(render_lock):
                    console.print(
                        Panel(
                            Text(str(exc), style="red"),
                            title="错误",
                            border_style="red",
                        )
                    )
                continue

            _render_response(console, response.output, lock=render_lock)
    finally:
        service.stop_cron()


def _render_startup(
    console: Console,
    service: AgentService,
    *,
    lock: threading.RLock | None = None,
) -> None:
    with _render_context(lock):
        console.print(render_pixel_logo(), overflow="ignore", no_wrap=True)
        console.print("[bold]Welcome to [purple]AsyncClaw[/][/]")
        console.print("[cyan]Ready in dev mode.[/]")
        console.print(
            "[dim]Type a command to begin. Use[/] [purple]/exit[/] [dim]to quit.[/]"
        )
        console.print()


def _render_response(
    console: Console,
    output: str | None,
    *,
    lock: threading.RLock | None = None,
) -> None:
    with _render_context(lock):
        console.print(Rule("[bold green]助手[/]", style="green"))
        console.print(Markdown(output or "（无输出）"))
        console.print()


def _render_cron_start(
    console: Console,
    job: CronJob,
    *,
    lock: threading.RLock | None = None,
) -> None:
    with _render_context(lock):
        console.print()
        console.print(
            Panel(
                Text("正在执行..."),
                title=f"定时任务执行中: {job.name}",
                border_style="magenta",
            )
        )
        console.print()


def _render_cron_result(
    console: Console,
    result: dict,
    *,
    lock: threading.RLock | None = None,
) -> None:
    name = result.get("name") or "定时任务"
    with _render_context(lock):
        if result.get("success"):
            console.print(Rule(f"[bold magenta]定时任务结果: {name}[/]", style="magenta"))
            console.print(Markdown(result.get("output") or "（无输出）"))
            console.print()
            return

        console.print(
            Panel(
                Text(str(result.get("error") or "未知错误"), style="red"),
                title=f"定时任务失败: {name}",
                border_style="red",
            )
        )
        console.print()


def _render_context(lock: threading.RLock | None):
    return lock if lock is not None else nullcontext()


def _patch_stdout_context():
    if patch_stdout is None:
        return nullcontext()
    try:
        return patch_stdout(raw=True)
    except TypeError:
        return patch_stdout()


def _read_user_text(
    console: Console,
    *,
    session: Any | None = None,
) -> str:
    if session is not None:
        if patch_stdout is not None:
            with _patch_stdout_context():
                raw_text = session.prompt("用户: ")
        else:
            raw_text = session.prompt("用户: ")
        return _normalize_user_input(raw_text).strip()
    if PromptSession is not None and patch_stdout is not None:
        prompt_session = PromptSession()
        with _patch_stdout_context():
            raw_text = prompt_session.prompt("用户: ")
        return _normalize_user_input(raw_text).strip()
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
