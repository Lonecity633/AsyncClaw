"""Rich-powered interactive agent CLI."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import re
import threading
from pathlib import Path
from typing import Any, Iterator

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
APPROVAL_ACCEPTED = {"y", "yes", "是", "确认", "执行"}
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
    approval_provider = CliShellApprovalProvider(
        console=console,
        session=prompt_session,
        lock=render_lock,
    )
    service = AgentService(
        cwd=cwd,
        env_file=env_file,
        env_file_explicit=env_file_explicit,
        workspace_root=workspace_root,
        allow_shell_exec=allow_shell_exec,
        approval_provider=approval_provider,
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
                with console.status("[bold green]AsyncClaw 思考中...[/]", spinner="dots") as status:
                    approval_provider.status = status
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
            finally:
                approval_provider.status = None

            _render_response(console, response.output, lock=render_lock)
    finally:
        service.stop_cron()


class CliShellApprovalProvider:
    """Interactive shell approval prompt for the Rich CLI."""

    def __init__(
        self,
        *,
        console: Console,
        session: Any | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self.console = console
        self.session = session
        self.lock = lock
        self.status: Any | None = None

    def approve(self, *, command: str, cwd: Path, reason: str | None = None) -> bool:
        with self._paused_status():
            with _render_context(self.lock):
                self.console.print()
                self.console.print("[bold yellow]shell_exec 需要审批。[/]")
                self.console.print(f"工作目录：{cwd}")
                self.console.print(f"命令：{command}")
                if reason:
                    self.console.print(f"原因：{reason}")
                try:
                    answer = _read_approval_text(self.console, session=self.session)
                except (EOFError, KeyboardInterrupt):
                    self.console.print("[yellow]已取消执行。[/]")
                    return False
        return answer.strip().lower() in APPROVAL_ACCEPTED

    @contextmanager
    def _paused_status(self) -> Iterator[None]:
        status = self.status
        if status is None:
            yield
            return

        status.stop()
        try:
            yield
        finally:
            status.start()


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
        return _read_session_text(session, "用户: ")
    if PromptSession is not None and patch_stdout is not None:
        prompt_session = PromptSession()
        return _read_session_text(prompt_session, "用户: ")
    raw_text = Prompt.ask("[bold cyan]用户[/]", console=console)
    return _normalize_user_input(raw_text).strip()


def _read_approval_text(
    console: Console,
    *,
    session: Any | None = None,
) -> str:
    if session is not None:
        return _read_session_text(session, "是否执行该命令？[是/否] ")
    if PromptSession is not None and patch_stdout is not None:
        prompt_session = PromptSession()
        return _read_session_text(prompt_session, "是否执行该命令？[是/否] ")
    raw_text = console.input("是否执行该命令？[是/否] ")
    return _normalize_user_input(raw_text).strip()


def _read_session_text(session: Any, prompt: str) -> str:
    if patch_stdout is not None:
        with _patch_stdout_context():
            raw_text = session.prompt(prompt)
    else:
        raw_text = session.prompt(prompt)
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
