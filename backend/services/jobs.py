import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.config import JOB_HISTORY_LIMIT, JOB_RETENTION_SECONDS
from backend.services.storage import discard

JobWorker = Callable[["Job"], None]

TERMINAL_STATES = {"completed", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    result: Any = None
    exports: dict[str, Path] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False
    process: subprocess.Popen | None = None
    events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None

    def append_log(self, line: str) -> None:
        self.logs.append(line)
        self.logs = self.logs[-500:]
        self.updated_at = time.time()
        self.emit("log", {"line": line})

    def emit(self, event_type: str = "status", payload: dict[str, Any] | None = None) -> None:
        self.updated_at = time.time()
        self.events.put({"type": event_type, "job": self.snapshot(), "payload": payload or {}})

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "logs": self.logs,
            "result": self.result,
            "exports": {key: f"/api/exports/{self.id}/{key}" for key in self.exports},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cancel_requested": self.cancel_requested,
        }

    def release(self) -> None:
        """Delete this job's on-disk exports. Called when the job is evicted."""
        discard(*self.exports.values())
        self.exports = {}


class JobRegistry:
    """In-memory job store with bounded history.

    Jobs used to accumulate forever: every finished job kept its result, logs and
    export files alive for the lifetime of the process, so a busy server grew
    without limit. Terminal jobs are now evicted once they exceed either the age
    or the count budget, and their export files are deleted with them.
    """

    def __init__(
        self,
        retention_seconds: float = JOB_RETENTION_SECONDS,
        history_limit: int = JOB_HISTORY_LIMIT,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._retention_seconds = retention_seconds
        self._history_limit = history_limit

    def create(self, kind: str, worker: JobWorker) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self.prune()
        job.thread = threading.Thread(target=self._run, args=(job, worker), daemon=True)
        job.thread.start()
        return job

    def put(self, job: Job) -> Job:
        """Store a pre-built job (the synchronous /api/transcribe 'latest' slot).

        Any job already under that id is released first so its export files do
        not linger once they are unreachable.
        """
        with self._lock:
            previous = self._jobs.get(job.id)
            self._jobs[job.id] = job
        if previous is not None and previous is not job:
            previous.release()
        self.prune()
        return job

    def _run(self, job: Job, worker: JobWorker) -> None:
        job.status = "running"
        job.message = "Running"
        job.emit()
        try:
            worker(job)
            if job.status not in {"cancelled", "failed"}:
                job.status = "completed"
                job.progress = 1.0
                job.message = "Completed"
        except Exception as exc:
            job.status = "failed"
            job.message = str(exc)
            job.append_log(f"Error: {exc}")
        finally:
            job.emit("done")

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if not job:
            return None
        job.cancel_requested = True
        job.message = "Cancellation requested"
        if job.process and job.process.poll() is None:
            job.process.terminate()
        job.emit()
        return job

    def active_summary(self) -> list[dict[str, Any]]:
        return [
            job.snapshot()
            for job in list(self._jobs.values())
            if job.status in {"queued", "running"}
        ]

    def prune(self) -> int:
        """Evict expired / surplus terminal jobs. Returns how many were removed."""
        cutoff = time.time() - self._retention_seconds
        with self._lock:
            finished = [
                job
                for job in self._jobs.values()
                # "latest" is the well-known slot for the sync endpoint; it is
                # replaced in place rather than aged out.
                if job.status in TERMINAL_STATES and job.id != "latest"
            ]
            expired = [job for job in finished if job.updated_at < cutoff]
            survivors = sorted(
                (job for job in finished if job.updated_at >= cutoff),
                key=lambda job: job.updated_at,
                reverse=True,
            )
            surplus = survivors[self._history_limit :]
            doomed = expired + surplus
            for job in doomed:
                self._jobs.pop(job.id, None)

        for job in doomed:
            job.release()
        return len(doomed)

    def stats(self) -> dict[str, int]:
        jobs = list(self._jobs.values())
        return {
            "total": len(jobs),
            "active": sum(1 for job in jobs if job.status in {"queued", "running"}),
            "retention_seconds": int(self._retention_seconds),
            "history_limit": self._history_limit,
        }


job_registry = JobRegistry()
