import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import ROOT_DIR
from backend.schemas import JobResponse, TrainingRequest
from backend.services.exports import write_temp_export
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
            return

        exports: dict[str, Path] = {}
        comparison_path = (ROOT_DIR / request.comparison_output).resolve()
        if comparison_path.is_file():
            report = json.loads(comparison_path.read_text(encoding="utf-8"))
            job.result = report | {"command": relative_command(args)}
            exports["comparison_json"] = write_temp_export(
                comparison_path.read_text(encoding="utf-8"), ".json"
            )
            markdown = comparison_path.with_suffix(".md")
            if markdown.is_file():
                exports["comparison_md"] = write_temp_export(markdown.read_text(encoding="utf-8"), ".md")
            for key in ["analysis_json", "chart_data_json", "old_confusion_csv", "new_confusion_csv"]:
                path_value = report.get(key)
                if path_value and Path(path_value).is_file():
                    suffix = Path(path_value).suffix or ".txt"
                    exports[key] = write_temp_export(Path(path_value).read_text(encoding="utf-8"), suffix)
        output_dir = (ROOT_DIR / request.output_dir).resolve()
        metrics_dir = output_dir / "metrics"
        for name, filename in {"training_log_jsonl": "trainer_log.jsonl", "training_log_csv": "trainer_log.csv"}.items():
            path = metrics_dir / filename
            if path.is_file():
                exports[name] = write_temp_export(path.read_text(encoding="utf-8"), path.suffix)
        if exports:
            job.exports = exports

    try:
        job = job_registry.create("train", work)
        return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/command")
def generate_training_command(request: TrainingRequest):
    args = build_training_args(request.model_dump())
    return {"command": relative_command(args)}

