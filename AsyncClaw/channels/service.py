"""Transport-neutral agent service construction and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from AsyncClaw.agent.cron import CronJob, CronService, CronStore
from AsyncClaw.agent.llm import create_openai_llm
from AsyncClaw.agent.logger import JsonlEventLogger
from AsyncClaw.agent.runtime import AgentLoop
from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.channels.base import AgentRequest, AgentResponse
from AsyncClaw.config import (
    LLMConfig,
    MCPConfig,
    load_llm_config,
    load_mcp_config,
    resolve_env_file,
    resolve_log_path,
    resolve_workspace_root,
)
from AsyncClaw.tools import (
    ApprovalProvider,
    ToolContext,
    ToolRegistry,
    build_tool_registry_from_providers,
)


CRON_SYSTEM_PROMPT = """这是定时任务的一次触发，不是创建新的循环。

- 调度器已经负责按 schedule 反复触发；本次只执行一次任务。
- 不要使用 watch、sleep、while True、循环脚本或常驻命令。
- 如果需要获取当前信息，请根据可用工具自动决定是否调用工具。
- 最终只输出本次任务的执行结果，不要复述这些调度器约束。"""


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
        approval_provider: ApprovalProvider | None = None,
        allow_cron: bool = False,
        cron_interval_seconds: float = 1.0,
        cron_max_concurrent_jobs: int = 2,
        on_cron_job_start: Callable[[CronJob], None] | None = None,
        on_cron_job_result: Callable[[dict[str, Any]], None] | None = None,
        workspace: WorkspaceStore | None = None,
        tool_context: ToolContext | None = None,
        tools: ToolRegistry | None = None,
        mcp_config: MCPConfig | None = None,
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

        self.llm = llm
        self.workspace = workspace or WorkspaceStore(root=self.workspace_root)
        self.tool_context = tool_context or ToolContext(
            cwd=self.cwd,
            sandbox_root=self.workspace.root / "office",
            allow_shell_exec=allow_shell_exec,
            approval_provider=approval_provider,
        )
        if mcp_config is not None:
            self.mcp_config = mcp_config
        elif tools is not None:
            self.mcp_config = MCPConfig()
        else:
            self.mcp_config = load_mcp_config(env_file=self.env_file_path)
        self.tools = tools or build_tool_registry_from_providers(
            context=self.tool_context,
            workspace=self.workspace,
            mcp_config=self.mcp_config,
        )
        self.logger = logger or JsonlEventLogger(self.log_path)
        self.max_steps = max_steps or (self.config.agent_max_steps if self.config else 8)
        self.cron_store = CronStore(self.workspace)
        self.cron_service: CronService | None = None
        self.on_cron_job_start = on_cron_job_start
        self.on_cron_job_result = on_cron_job_result
        self.system_prompt = system_prompt
        self.agent = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            max_steps=self.max_steps,
            logger=self.logger,
            tool_context=self.tool_context,
            workspace=self.workspace,
            system_prompt=system_prompt,
        )
        if allow_cron:
            self.start_cron(
                interval_seconds=cron_interval_seconds,
                max_concurrent_jobs=cron_max_concurrent_jobs,
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

    def handle_cron_text(self, job: CronJob) -> AgentResponse:
        """Run one cron job as an isolated agent turn."""

        text = job.prompt.strip()
        if not text:
            raise ValueError("定时任务 prompt 不能为空")

        cron_workspace = WorkspaceStore(root=self.workspace.root)
        cron_tool_context = ToolContext(
            cwd=self.tool_context.cwd,
            sandbox_root=self.tool_context.sandbox_root,
            allow_shell_exec=self.tool_context.allow_shell_exec,
            approval_mode="never",
            shell_timeout_seconds=self.tool_context.shell_timeout_seconds,
            shell_output_limit_bytes=self.tool_context.shell_output_limit_bytes,
            approval_provider=None,
        )
        cron_tools = build_tool_registry_from_providers(
            context=cron_tool_context,
            workspace=cron_workspace,
            mcp_config=self.mcp_config,
            include_cron_tools=False,
        )
        cron_system_prompt = (
            f"{self.system_prompt}\n\n{CRON_SYSTEM_PROMPT}"
            if self.system_prompt
            else CRON_SYSTEM_PROMPT
        )
        cron_agent = AgentLoop(
            llm=self.llm,
            tools=cron_tools,
            max_steps=self.max_steps,
            logger=self.logger,
            tool_context=cron_tool_context,
            workspace=cron_workspace,
            system_prompt=cron_system_prompt,
        )
        result = cron_agent.run([{"role": "user", "content": text}])
        return AgentResponse(
            output=result.output,
            session_id=cron_workspace.session_id,
            cwd=self.cwd,
            steps=result.steps,
            messages=result.messages,
            observations=result.observations,
        )

    def start_cron(
        self,
        *,
        interval_seconds: float = 1.0,
        max_concurrent_jobs: int = 2,
    ) -> None:
        """Start the background cron heartbeat service."""

        if self.cron_service is None:
            self.cron_service = CronService(
                service=self,
                store=self.cron_store,
                interval_seconds=interval_seconds,
                max_concurrent_jobs=max_concurrent_jobs,
                logger=self.logger,
                on_job_start=self.on_cron_job_start,
                on_job_result=self.on_cron_job_result,
            )
        self.cron_service.start()

    def stop_cron(self) -> None:
        """Stop the background cron heartbeat service if it is running."""

        if self.cron_service is not None:
            self.cron_service.stop()


def _resolve_env_file(cwd: Path, path: str | Path, *, explicit: bool = False) -> Path:
    return resolve_env_file(cwd, path, explicit=explicit, root=_project_root())


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_workspace_root(cwd: Path, workspace_root: str | Path | None = None) -> Path:
    return resolve_workspace_root(cwd, workspace_root, root=_project_root())


def _resolve_log_path(workspace_root: str | Path | None, resolved_workspace_root: Path) -> Path:
    return resolve_log_path(
        workspace_root,
        resolved_workspace_root,
        root=_project_root(),
    )
