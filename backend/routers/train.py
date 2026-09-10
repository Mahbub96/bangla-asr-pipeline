import os
import subprocess

from fastapi import APIRouter, HTTPException

from backend.config import ROOT_DIR
from backend.schemas import JobResponse, TrainingRequest
from backend.services.jobs import Job, job_registry
from backend.services.training import build_training_args, relative_command

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/jobs", response_model=JobResponse)
def create_training_job(request: TrainingRequest):
    args = build_training_args(request.model_dump())

    def work(job: Job) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        job.result = {"command": relative_command(args)}
        job.append_log(relative_command(args))
        job.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
            env=env,
        )
        assert job.process.stdout is not None
        for line in iter(job.process.stdout.readline, ""):
            if job.cancel_requested:
                job.status = "cancelled"
                job.message = "Cancelled"
                job.process.terminate()
                return
            job.append_log(line.rstrip())
        code = job.process.wait()
        if code != 0 and not job.cancel_requested:
            raise RuntimeError(f"Training exited with code {code}")
        if job.cancel_requested:
            job.status = "cancelled"
            job.message = "Cancelled"

    try:
        job = job_registry.create("train", work)
        return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/command")
def generate_training_command(request: TrainingRequest):
    args = build_training_args(request.model_dump())
    return {"command": relative_command(args)}

