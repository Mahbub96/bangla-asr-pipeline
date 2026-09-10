import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


JobWorker = Callable[["Job"], None]


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


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, worker: JobWorker) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        job.thread = threading.Thread(target=self._run, args=(job, worker), daemon=True)
        job.thread.start()
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
            for job in self._jobs.values()
            if job.status in {"queued", "running"}
        ]


job_registry = JobRegistry()

