"""Workspace-backed cron job storage and heartbeat runner."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, ClassVar
from uuid import uuid4

from AsyncClaw.agent.workspace import WorkspaceStore


@dataclass
class CronJob:
    """A scheduled agent task persisted in workspace/cron/jobs.json."""

    id: str
    name: str
    prompt: str
    action: str
    schedule: dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str
    last_run_at: str | None
    next_run_at: str | None
    run_count: int
    failure_count: int
    last_error: str | None
    running: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronJob":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            prompt=str(data.get("prompt") or ""),
            action=_normalize_action(data.get("action")),
            schedule=dict(data.get("schedule") or {}),
            enabled=bool(data.get("enabled", True)),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            last_run_at=data.get("last_run_at"),
            next_run_at=data.get("next_run_at"),
            run_count=int(data.get("run_count") or 0),
            failure_count=int(data.get("failure_count") or 0),
            last_error=data.get("last_error"),
            running=bool(data.get("running", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CronStore:
    """JSON-backed store for scheduled jobs."""

    version = 1
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _locks: ClassVar[dict[str, threading.RLock]] = {}

    def __init__(self, workspace: WorkspaceStore | str | Path):
        if isinstance(workspace, WorkspaceStore):
            self.path = workspace.cron_jobs_path
        else:
            self.path = Path(workspace)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self._lock_for_path(self.path)

    @classmethod
    def _lock_for_path(cls, path: Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._locks_guard:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._locks[key] = lock
            return lock

    def list_jobs(self) -> list[CronJob]:
        with self._lock:
            payload = self._read_payload()
            return [
                CronJob.from_dict(job)
                for job in payload.get("jobs", [])
                if isinstance(job, dict)
            ]

    def create_job(
        self,
        *,
        name: str,
        prompt: str,
        schedule: dict[str, Any],
        action: str = "notify",
        now: datetime | None = None,
    ) -> CronJob:
        if not isinstance(name, str):
            raise TypeError("任务名称必须是字符串")
        if not isinstance(prompt, str):
            raise TypeError("任务 prompt 必须是字符串")
        name = name.strip()
        prompt = prompt.strip()
        if not name:
            raise ValueError("任务名称不能为空")
        if not prompt:
            raise ValueError("任务 prompt 不能为空")

        normalized_schedule = _normalize_schedule(schedule)
        normalized_action = _normalize_action(action)
        timestamp = _format_datetime(now or _utc_datetime())
        next_run_at = _initial_next_run_at(normalized_schedule, now or _utc_datetime())
        job = CronJob(
            id=uuid4().hex,
            name=name,
            prompt=prompt,
            action=normalized_action,
            schedule=normalized_schedule,
            enabled=True,
            created_at=timestamp,
            updated_at=timestamp,
            last_run_at=None,
            next_run_at=next_run_at,
            run_count=0,
            failure_count=0,
            last_error=None,
            running=False,
        )
        with self._lock:
            jobs = self.list_jobs()
            jobs.append(job)
            self._write_jobs(jobs)
        return job

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            jobs = self.list_jobs()
            kept = [job for job in jobs if job.id != job_id]
            deleted = len(kept) != len(jobs)
            if deleted:
                self._write_jobs(kept)
            return deleted

    def clear_jobs(self) -> int:
        with self._lock:
            jobs = self.list_jobs()
            deleted_count = len(jobs)
            if deleted_count:
                self._write_jobs([])
            return deleted_count

    def due_jobs(self, now: datetime | None = None) -> list[CronJob]:
        with self._lock:
            now = now or _utc_datetime()
            due: list[CronJob] = []
            for job in self.list_jobs():
                if _is_due(job, now):
                    due.append(job)
            return due

    def claim_due_jobs(
        self,
        now: datetime | None = None,
        *,
        limit: int | None = None,
    ) -> list[CronJob]:
        """Atomically mark due jobs as running and return the claimed jobs."""

        if limit is not None and limit <= 0:
            return []
        with self._lock:
            now = now or _utc_datetime()
            jobs = self.list_jobs()
            claimed: list[CronJob] = []
            timestamp = _format_datetime(now)
            for job in jobs:
                if limit is not None and len(claimed) >= limit:
                    break
                if not _is_due(job, now):
                    continue
                job.running = True
                job.updated_at = timestamp
                claimed.append(job)
            if claimed:
                self._write_jobs(jobs)
            return claimed

    def mark_running(self, job_id: str, running: bool = True) -> CronJob | None:
        with self._lock:
            now = _utc_now()
            jobs = self.list_jobs()
            updated: CronJob | None = None
            for job in jobs:
                if job.id == job_id:
                    job.running = running
                    job.updated_at = now
                    updated = job
                    break
            self._write_jobs(jobs)
            return updated

    def reset_running_jobs(
        self,
        *,
        error: str = "任务在上次运行中断后恢复为待执行状态",
        now: datetime | None = None,
    ) -> list[CronJob]:
        with self._lock:
            timestamp = _format_datetime(now or _utc_datetime())
            jobs = self.list_jobs()
            updated: list[CronJob] = []
            for job in jobs:
                if job.running:
                    job.running = False
                    job.updated_at = timestamp
                    job.last_error = error
                    updated.append(job)
            if updated:
                self._write_jobs(jobs)
            return updated

    def mark_success(self, job_id: str, now: datetime | None = None) -> CronJob | None:
        with self._lock:
            now = now or _utc_datetime()
            timestamp = _format_datetime(now)
            jobs = self.list_jobs()
            updated: CronJob | None = None
            for job in jobs:
                if job.id == job_id:
                    job.running = False
                    job.last_run_at = timestamp
                    job.updated_at = timestamp
                    job.run_count += 1
                    job.last_error = None
                    job.next_run_at = _next_after_run(job.schedule, now)
                    if job.schedule.get("type") == "at":
                        job.enabled = False
                    updated = job
                    break
            self._write_jobs(jobs)
            return updated

    def mark_failure(
        self,
        job_id: str,
        error: str,
        now: datetime | None = None,
    ) -> CronJob | None:
        with self._lock:
            now = now or _utc_datetime()
            timestamp = _format_datetime(now)
            jobs = self.list_jobs()
            updated: CronJob | None = None
            for job in jobs:
                if job.id == job_id:
                    job.running = False
                    job.last_run_at = timestamp
                    job.updated_at = timestamp
                    job.failure_count += 1
                    job.last_error = error
                    job.next_run_at = _next_after_run(job.schedule, now)
                    if job.schedule.get("type") == "at":
                        job.enabled = False
                    updated = job
                    break
            self._write_jobs(jobs)
            return updated

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": self.version, "jobs": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"cron jobs 文件不是有效 JSON: {self.path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("cron jobs 文件顶层必须是 JSON 对象")
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            raise ValueError("cron jobs 文件 jobs 字段必须是列表")
        return payload

    def _write_jobs(self, jobs: list[CronJob]) -> None:
        payload = {
            "version": self.version,
            "jobs": [job.to_dict() for job in jobs],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class CronService:
    """Heartbeat loop that dispatches due cron jobs through AgentService."""

    def __init__(
        self,
        *,
        service: Any,
        store: CronStore,
        interval_seconds: float = 1.0,
        max_concurrent_jobs: int = 2,
        logger: Any | None = None,
        on_job_start: Callable[[CronJob], None] | None = None,
        on_job_result: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs 必须是正整数")
        self.service = service
        self.store = store
        self.interval_seconds = interval_seconds
        self.max_concurrent_jobs = max_concurrent_jobs
        self.logger = logger
        self.on_job_start = on_job_start
        self.on_job_result = on_job_result
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._workers_lock = threading.RLock()
        self._workers: set[threading.Thread] = set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        recovered = self.store.reset_running_jobs()
        if recovered:
            self._log(
                "cron_jobs_recovered",
                {"jobs": [job.to_dict() for job in recovered]},
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="asyncclaw-cron",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._join_workers(timeout)

    def tick(
        self,
        now: datetime | None = None,
        *,
        wait: bool = True,
    ) -> list[dict[str, Any]]:
        self._prune_workers()
        capacity = self._available_capacity()
        jobs = self.store.claim_due_jobs(now, limit=capacity)
        if wait:
            return self._run_jobs_and_wait(jobs, now=now)
        self._start_workers(jobs, now=now)
        return []

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick(wait=False)
            except Exception as exc:
                self._log(
                    "cron_tick_error",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
            self._stop.wait(self.interval_seconds)

    def _run_job(self, job: CronJob, now: datetime | None = None) -> dict[str, Any]:
        self._log("cron_job_start", {"id": job.id, "name": job.name})
        self._notify_job_start(job)

        try:
            if hasattr(self.service, "handle_cron_text"):
                response = self.service.handle_cron_text(job)
            else:
                response = self.service.handle_text(_build_agent_job_prompt(job))
        except Exception as exc:
            error = str(exc)
            updated = self.store.mark_failure(job.id, error, now=now)
            result = {
                "id": job.id,
                "name": job.name,
                "action": job.action,
                "success": False,
                "error": error,
                "job": updated.to_dict() if updated else None,
            }
            self._log("cron_job_error", result)
            self._notify_job_result(result)
            return result

        updated = self.store.mark_success(job.id, now=now)
        result = {
            "id": job.id,
            "name": job.name,
            "action": job.action,
            "success": True,
            "output": response.output,
            "job": updated.to_dict() if updated else None,
        }
        self._log("cron_job_result", result)
        self._notify_job_result(result)
        return result

    def _run_jobs_and_wait(
        self,
        jobs: list[CronJob],
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if not jobs:
            return []
        if len(jobs) == 1:
            return [self._run_job(jobs[0], now=now)]

        results: list[dict[str, Any]] = []
        results_lock = threading.Lock()
        threads = [
            threading.Thread(
                target=self._run_job_for_results,
                args=(job, results, results_lock, now),
                name=f"asyncclaw-cron-sync-{job.id[:8]}",
                daemon=True,
            )
            for job in jobs
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def _run_job_for_results(
        self,
        job: CronJob,
        results: list[dict[str, Any]],
        results_lock: threading.Lock,
        now: datetime | None,
    ) -> None:
        result = self._run_job(job, now=now)
        with results_lock:
            results.append(result)

    def _start_workers(
        self,
        jobs: list[CronJob],
        *,
        now: datetime | None = None,
    ) -> None:
        for job in jobs:
            thread = threading.Thread(
                target=self._run_worker,
                args=(job, now),
                name=f"asyncclaw-cron-job-{job.id[:8]}",
                daemon=True,
            )
            with self._workers_lock:
                self._workers.add(thread)
            thread.start()

    def _run_worker(self, job: CronJob, now: datetime | None) -> None:
        thread = threading.current_thread()
        try:
            self._run_job(job, now=now)
        finally:
            with self._workers_lock:
                self._workers.discard(thread)

    def _available_capacity(self) -> int:
        with self._workers_lock:
            active = sum(1 for worker in self._workers if worker.is_alive())
        return max(0, self.max_concurrent_jobs - active)

    def _prune_workers(self) -> None:
        with self._workers_lock:
            self._workers = {worker for worker in self._workers if worker.is_alive()}

    def _join_workers(self, timeout: float | None) -> None:
        with self._workers_lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=timeout)
        self._prune_workers()

    def _notify_job_start(self, job: CronJob) -> None:
        if self.on_job_start is None:
            return
        try:
            self.on_job_start(job)
        except Exception as exc:
            self._log(
                "cron_job_start_notify_error",
                {"type": type(exc).__name__, "message": str(exc), "id": job.id},
            )

    def _notify_job_result(self, result: dict[str, Any]) -> None:
        if self.on_job_result is None:
            return
        try:
            self.on_job_result(result)
        except Exception as exc:
            self._log(
                "cron_job_result_notify_error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "id": result.get("id"),
                },
            )

    def _log(self, event: str, data: dict[str, Any]) -> None:
        if self.logger is None:
            return
        try:
            self.logger.log(event, data)
        except Exception:
            pass


def _normalize_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schedule, dict):
        raise TypeError("schedule 必须是 JSON 对象")
    schedule_type = schedule.get("type")
    if schedule_type == "at":
        run_at = schedule.get("run_at")
        if not isinstance(run_at, str) or not run_at.strip():
            raise ValueError("at 任务必须提供 run_at")
        return {"type": "at", "run_at": _format_datetime(_parse_datetime(run_at))}
    if schedule_type == "every":
        seconds = schedule.get("seconds")
        if not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("every 任务 seconds 必须是正整数")
        return {"type": "every", "seconds": seconds}
    raise ValueError("schedule.type 仅支持 at 或 every")


def _is_due(job: CronJob, now: datetime) -> bool:
    if not job.enabled or job.running or not job.next_run_at:
        return False
    next_run_at = _parse_datetime(job.next_run_at)
    return next_run_at <= now


def _build_agent_job_prompt(job: CronJob) -> str:
    return (
        "这是定时任务的一次触发，不是创建新的循环。\n"
        "- 调度器已经负责按 schedule 反复触发；本次只执行一次任务。\n"
        "- 不要使用 watch、sleep、while True、循环脚本或常驻命令。\n"
        "- 如果需要获取当前信息，请根据可用工具自动决定是否调用工具。\n\n"
        f"原始任务：{job.prompt}"
    )


def _normalize_action(action: Any) -> str:
    if action is None or action == "":
        return "notify"
    if action in {"notify", "agent"}:
        return str(action)
    raise ValueError("action 仅支持 notify 或 agent")


def _initial_next_run_at(schedule: dict[str, Any], now: datetime) -> str:
    if schedule["type"] == "at":
        return schedule["run_at"]
    return _format_datetime(now + timedelta(seconds=schedule["seconds"]))


def _next_after_run(schedule: dict[str, Any], now: datetime) -> str | None:
    if schedule["type"] == "at":
        return None
    return _format_datetime(now + timedelta(seconds=schedule["seconds"]))


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now() -> str:
    return _format_datetime(_utc_datetime())
