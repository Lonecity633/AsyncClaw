from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from AsyncClaw.agent.cron import CronService, CronStore
from AsyncClaw.agent.workspace import WorkspaceStore
from AsyncClaw.channels.service import AgentService
from AsyncClaw.tools import ToolContext, build_tool_registry


class FakeAgentService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def handle_text(self, text: str):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("任务失败")
        return type("Response", (), {"output": f"done: {text}"})()


class CurrentTimeToolLLM:
    def __init__(self) -> None:
        self.requests = []

    def create_chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_current_time",
                                    "type": "function",
                                    "function": {
                                        "name": "current_time",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "已通过 current_time 输出当前时间",
                    }
                }
            ]
        }


class CronStoreTest(unittest.TestCase):
    def test_workspace_creates_cron_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")

            self.assertTrue(workspace.cron_dir.is_dir())
            self.assertEqual(workspace.cron_jobs_path, workspace.cron_dir / "jobs.json")

    def test_create_and_reload_every_job(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")
            store = CronStore(workspace)

            job = store.create_job(
                name="提醒",
                prompt="提醒我喝水",
                schedule={"type": "every", "seconds": 60},
                now=now,
            )
            reloaded = CronStore(workspace).list_jobs()[0]

        self.assertEqual(reloaded.id, job.id)
        self.assertEqual(reloaded.name, "提醒")
        self.assertEqual(reloaded.prompt, "提醒我喝水")
        self.assertEqual(reloaded.action, "notify")
        self.assertEqual(reloaded.schedule, {"type": "every", "seconds": 60})
        self.assertEqual(
            reloaded.next_run_at,
            (now + timedelta(seconds=60)).isoformat(),
        )

    def test_due_jobs_only_returns_enabled_due_not_running_jobs(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            due = store.create_job(
                name="due",
                prompt="执行",
                schedule={"type": "at", "run_at": (now - timedelta(seconds=1)).isoformat()},
                now=now,
            )
            future = store.create_job(
                name="future",
                prompt="稍后",
                schedule={"type": "at", "run_at": (now + timedelta(seconds=60)).isoformat()},
                now=now,
            )
            running = store.create_job(
                name="running",
                prompt="运行中",
                schedule={"type": "at", "run_at": (now - timedelta(seconds=1)).isoformat()},
                now=now,
            )
            store.mark_running(running.id, True)

            due_ids = {job.id for job in store.due_jobs(now)}

        self.assertIn(due.id, due_ids)
        self.assertNotIn(future.id, due_ids)
        self.assertNotIn(running.id, due_ids)

    def test_delete_job_persists(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="delete-me",
                prompt="删除",
                schedule={"type": "every", "seconds": 30},
                now=now,
            )

            deleted = store.delete_job(job.id)

        self.assertTrue(deleted)
        self.assertEqual(store.list_jobs(), [])

    def test_reset_running_jobs_recovers_stale_locks(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recovered_at = now + timedelta(minutes=1)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="running",
                prompt="恢复",
                schedule={"type": "every", "seconds": 30},
                now=now,
            )
            store.mark_running(job.id, True)

            recovered = store.reset_running_jobs(now=recovered_at)
            updated = store.list_jobs()[0]

        self.assertEqual([job.id for job in recovered], [job.id])
        self.assertFalse(updated.running)
        self.assertEqual(updated.updated_at, recovered_at.isoformat())
        self.assertIn("恢复", updated.last_error or "")

    def test_invalid_schedule_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))

            with self.assertRaisesRegex(ValueError, "schedule.type"):
                store.create_job(name="bad", prompt="bad", schedule={"type": "cron"})

    def test_invalid_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))

            with self.assertRaisesRegex(ValueError, "action"):
                store.create_job(
                    name="bad",
                    prompt="bad",
                    schedule={"type": "every", "seconds": 60},
                    action="unknown",
                )


class CronServiceTest(unittest.TestCase):
    def test_tick_runs_legacy_notify_job_through_agent(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="once",
                prompt="提醒一次",
                schedule={"type": "at", "run_at": now.isoformat()},
                now=now,
            )
            service = FakeAgentService()
            cron = CronService(service=service, store=store, interval_seconds=999)

            results = cron.tick(now)
            updated = store.list_jobs()[0]

        self.assertEqual(len(service.calls), 1)
        self.assertIn("原始任务：提醒一次", service.calls[0])
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["action"], "notify")
        self.assertIn("done:", results[0]["output"])
        self.assertNotEqual(results[0]["output"], "提醒一次")
        self.assertEqual(results[0]["id"], job.id)
        self.assertFalse(updated.enabled)
        self.assertIsNone(updated.next_run_at)
        self.assertEqual(updated.run_count, 1)
        self.assertFalse(updated.running)

    def test_tick_executes_agent_job_and_disables_at_job(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="once",
                prompt="执行一次",
                schedule={"type": "at", "run_at": now.isoformat()},
                action="agent",
                now=now,
            )
            service = FakeAgentService()
            cron = CronService(service=service, store=store, interval_seconds=999)

            results = cron.tick(now)
            updated = store.list_jobs()[0]

        self.assertEqual(len(service.calls), 1)
        self.assertIn("原始任务：执行一次", service.calls[0])
        self.assertIn("不要使用 watch", service.calls[0])
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["action"], "agent")
        self.assertEqual(results[0]["id"], job.id)
        self.assertFalse(updated.enabled)
        self.assertIsNone(updated.next_run_at)
        self.assertEqual(updated.run_count, 1)
        self.assertFalse(updated.running)

    def test_tick_records_failure_and_keeps_every_job_scheduled(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="repeat",
                prompt="会失败",
                schedule={"type": "every", "seconds": 60},
                action="agent",
                now=now - timedelta(seconds=60),
            )
            service = FakeAgentService(fail=True)
            cron = CronService(service=service, store=store, interval_seconds=999)

            results = cron.tick(now)
            updated = store.list_jobs()[0]

        self.assertFalse(results[0]["success"])
        self.assertEqual(results[0]["id"], job.id)
        self.assertEqual(results[0]["action"], "agent")
        self.assertEqual(updated.failure_count, 1)
        self.assertIn("任务失败", updated.last_error or "")
        self.assertTrue(updated.enabled)
        self.assertEqual(updated.next_run_at, (now + timedelta(seconds=60)).isoformat())
        self.assertFalse(updated.running)

    def test_start_recovers_stale_running_jobs(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="stale",
                prompt="执行",
                schedule={
                    "type": "at",
                    "run_at": (now + timedelta(hours=1)).isoformat(),
                },
                now=now,
            )
            store.mark_running(job.id, True)
            cron = CronService(
                service=FakeAgentService(),
                store=store,
                interval_seconds=999,
            )

            cron.start()
            cron.stop()
            updated = store.list_jobs()[0]

        self.assertFalse(updated.running)
        self.assertIn("恢复", updated.last_error or "")

    def test_tick_invokes_callbacks(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            store = CronStore(WorkspaceStore(root=Path(directory) / "workspace"))
            job = store.create_job(
                name="callback",
                prompt="提醒",
                schedule={"type": "at", "run_at": now.isoformat()},
                now=now,
            )
            starts = []
            results = []
            cron = CronService(
                service=FakeAgentService(),
                store=store,
                interval_seconds=999,
                on_job_start=starts.append,
                on_job_result=results.append,
            )

            cron.tick(now)

        self.assertEqual([started.id for started in starts], [job.id])
        self.assertEqual([result["id"] for result in results], [job.id])
        self.assertEqual(results[0]["action"], "notify")
        self.assertTrue(results[0]["success"])
        self.assertEqual(len(cron.service.calls), 1)

    def test_current_time_cron_job_uses_agent_tool_loop(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")
            llm = CurrentTimeToolLLM()
            service = AgentService(
                cwd=directory,
                workspace=workspace,
                llm=llm,
                allow_shell_exec=False,
            )
            job = service.cron_store.create_job(
                name="time",
                prompt="输出当前时间",
                schedule={"type": "at", "run_at": now.isoformat()},
                action="agent",
                now=now,
            )
            cron = CronService(service=service, store=service.cron_store, interval_seconds=999)

            results = cron.tick(now)
            updated = service.cron_store.list_jobs()[0]

        first_request_tools = [
            tool["function"]["name"]
            for tool in llm.requests[0]["tools"]
        ]
        self.assertIn("current_time", first_request_tools)
        self.assertEqual(len(llm.requests), 2)
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["id"], job.id)
        self.assertEqual(results[0]["output"], "已通过 current_time 输出当前时间")
        self.assertNotEqual(results[0]["output"], "输出当前时间")
        self.assertEqual(updated.run_count, 1)


class CronToolTest(unittest.TestCase):
    def test_cron_tools_create_list_and_delete_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")
            registry = build_tool_registry(ToolContext(cwd=Path(directory)), workspace=workspace)

            created = registry.call(
                "create_cron_job",
                {
                    "name": "提醒",
                    "prompt": "提醒我休息",
                    "schedule": {"type": "every", "seconds": 120},
                },
            )
            listed = registry.call("list_cron_jobs", {})
            deleted = registry.call(
                "delete_cron_job",
                {"id": created["job"]["id"]},
            )
            payload = json.loads(workspace.cron_jobs_path.read_text(encoding="utf-8"))

        self.assertTrue(created["created"])
        self.assertEqual(listed["jobs"][0]["name"], "提醒")
        self.assertEqual(listed["jobs"][0]["action"], "agent")
        self.assertTrue(deleted["deleted"])
        self.assertEqual(payload["jobs"], [])

    def test_cron_tool_defaults_to_agent_without_prompt_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")
            registry = build_tool_registry(ToolContext(cwd=Path(directory)), workspace=workspace)

            created = registry.call(
                "create_cron_job",
                {
                    "name": "当前时间",
                    "prompt": "每十秒输出当前时间",
                    "schedule": {"type": "every", "seconds": 10},
                },
            )

        self.assertEqual(created["job"]["action"], "agent")

    def test_cron_tool_schema_does_not_expose_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceStore(root=Path(directory) / "workspace")
            registry = build_tool_registry(ToolContext(cwd=Path(directory)), workspace=workspace)

            cron_tool = next(
                tool
                for tool in registry.to_openai_tools()
                if tool["function"]["name"] == "create_cron_job"
            )

        properties = cron_tool["function"]["parameters"]["properties"]
        self.assertNotIn("action", properties)


if __name__ == "__main__":
    unittest.main()
