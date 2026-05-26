"""Cron job management tools."""

from __future__ import annotations

from typing import Any

from AsyncClaw.agent.cron import CronStore
from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.tools.spec import Tool


def create_cron_tools(workspace: WorkspaceStore) -> list[Tool]:
    """Create tools that manage workspace/cron/jobs.json."""

    store = CronStore(workspace)

    def create_cron_job(arguments: dict[str, Any]) -> dict[str, Any]:
        job = store.create_job(
            name=arguments.get("name", ""),
            prompt=arguments.get("prompt", ""),
            schedule=arguments.get("schedule", {}),
            action=arguments.get("action") or "agent",
        )
        return {"created": True, "job": job.to_dict()}

    def list_cron_jobs(arguments: dict[str, Any]) -> dict[str, Any]:
        jobs = store.list_jobs()
        return {"jobs": [job.to_dict() for job in jobs]}

    def delete_cron_job(arguments: dict[str, Any]) -> dict[str, Any]:
        job_id = arguments.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("id 必须是非空字符串")
        deleted = store.delete_job(job_id.strip())
        return {"deleted": deleted, "id": job_id.strip()}

    return [
        Tool(
            name="create_cron_job",
            description=(
                "在 workspace/cron/jobs.json 中创建一个定时任务。"
                "任务到期时会把 prompt 交给智能体执行；"
                "智能体会根据任务内容和可用工具自动决定是否调用工具。"
                "schedule 支持 {type:'at', run_at:'UTC ISO 时间'} "
                "或 {type:'every', seconds:正整数}。"
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "定时任务名称。",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "任务到期时发送给智能体执行的用户消息。",
                    },
                    "schedule": {
                        "type": "object",
                        "description": "任务调度配置，支持 at 或 every。",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["at", "every"],
                            },
                            "run_at": {
                                "type": "string",
                                "description": "at 任务的 UTC ISO 时间。",
                            },
                            "seconds": {
                                "type": "integer",
                                "description": "every 任务的间隔秒数。",
                            },
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "prompt", "schedule"],
                "additionalProperties": False,
            },
            handler=create_cron_job,
        ),
        Tool(
            name="list_cron_jobs",
            description="列出 workspace/cron/jobs.json 中的所有定时任务。",
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=list_cron_jobs,
        ),
        Tool(
            name="delete_cron_job",
            description="从 workspace/cron/jobs.json 中按 id 删除定时任务。",
            schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "要删除的定时任务 id。",
                    }
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            handler=delete_cron_job,
        ),
    ]
