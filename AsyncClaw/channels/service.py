"""Transport-neutral agent service construction and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from AsyncClaw.agent.llm import create_openai_llm
from AsyncClaw.agent.logger import JsonlEventLogger
from AsyncClaw.agent.runtime import AgentLoop
from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.channels.base import AgentRequest, AgentResponse
from AsyncClaw.config import LLMConfig, load_llm_config
from AsyncClaw.tools import ToolContext, ToolRegistry, build_tool_registry


class AgentService:
    """Build and run an AsyncClaw agent without depending on a CLI."""

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        env_file: str | Path = ".env",
        env_file_explicit: bool = False,
        workspace_root: str | Path | None = None,
        config: LLMConfig | None = None,
        llm: Any | None = None,
        max_steps: int | None = None,
        allow_shell_exec: bool = True,
        workspace: WorkspaceStore | None = None,
        tool_context: ToolContext | None = None,
        tools: ToolRegistry | None = None,
        logger: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.config = config
        self.workspace_root = _resolve_workspace_root(self.cwd, workspace_root)
        self.log_path = _resolve_log_path(workspace_root, self.workspace_root)
        self.env_file_path = _resolve_env_file(
            self.cwd,
            env_file,
            explicit=env_file_explicit,
        )

        if llm is None:
            self.config = self.config or load_llm_config(env_file=self.env_file_path)
            llm = create_openai_llm(self.config)

        self.workspace = workspace or WorkspaceStore(root=self.workspace_root)
        self.tool_context = tool_context or ToolContext(
            cwd=self.cwd,
            sandbox_root=self.workspace.root / "office",
            allow_shell_exec=allow_shell_exec,
        )
        self.tools = tools or build_tool_registry(self.tool_context, workspace=self.workspace)
        self.logger = logger or JsonlEventLogger(self.log_path)
        self.max_steps = max_steps or (self.config.agent_max_steps if self.config else 8)
        self.agent = AgentLoop(
            llm=llm,
            tools=self.tools,
            max_steps=self.max_steps,
            logger=self.logger,
            tool_context=self.tool_context,
            workspace=self.workspace,
            system_prompt=system_prompt,
        )

    def handle(self, request: AgentRequest) -> AgentResponse:
        """Run one text request through the agent."""

        text = request.text.strip()
        if not text:
            raise ValueError("输入不能为空")

        result = self.agent.run([{"role": "user", "content": text}])
        return AgentResponse(
            output=result.output,
            session_id=self.workspace.session_id,
            cwd=self.cwd,
            steps=result.steps,
            messages=result.messages,
            observations=result.observations,
        )

    def handle_text(self, text: str) -> AgentResponse:
        """Convenience wrapper for simple text channels."""

        return self.handle(AgentRequest(text=text))


def _resolve_env_file(cwd: Path, path: str | Path, *, explicit: bool = False) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    cwd_path = cwd / resolved
    if explicit or resolved != Path(".env"):
        return cwd_path

    if cwd_path.exists():
        return cwd_path

    project_env = _project_root() / ".env"
    if project_env.exists():
        return project_env

    return cwd_path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_workspace_root(cwd: Path, workspace_root: str | Path | None = None) -> Path:
    if workspace_root is None:
        return (_project_root() / "workspace").resolve()
    resolved = Path(workspace_root)
    if resolved.is_absolute():
        return resolved.resolve()
    return (cwd / resolved).resolve()


def _resolve_log_path(workspace_root: str | Path | None, resolved_workspace_root: Path) -> Path:
    if workspace_root is None:
        return (_project_root() / "logs" / "events.jsonl").resolve()
    return (resolved_workspace_root / "logs" / "events.jsonl").resolve()
