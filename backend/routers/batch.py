from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import ROOT_DIR
from backend.routers.transcribe import run_transcription, save_upload
from backend.schemas import DirectoryBatchRequest, JobResponse, TranscriptionOptions
from backend.services.audio import allowed_audio_path
from backend.services.exports import create_table_exports
from backend.services.jobs import Job, job_registry

router = APIRouter(prefix="/api/batch", tags=["batch"])


def batch_worker(paths: list[Path], options: TranscriptionOptions):
    def work(job: Job) -> None:
        rows = []
        for index, path in enumerate(paths, 1):
            if job.cancel_requested:
                job.status = "cancelled"
                job.message = "Cancelled"
                return
            job.message = f"Processing {path.name}"
            job.progress = (index - 1) / max(1, len(paths))
            job.emit()
            try:
                result, _ = run_transcription(path, options)
                rows.append(
                    {
                        "file": path.name,
                        "language": result["language"],
                        "confidence": result["language_probability"],
                        "duration_sec": result["duration_sec"],
                        "speed_factor": result["speed_factor"],
                        "text": result["text"],
                    }
                )
                job.append_log(f"{index}/{len(paths)} {path.name}: {result['text'][:120]}")
            except Exception as exc:
                rows.append({"file": path.name, "language": "Error", "text": str(exc)})
                job.append_log(f"{index}/{len(paths)} {path.name}: ERROR {exc}")
            job.result = rows
            job.progress = index / max(1, len(paths))
            job.exports = create_table_exports(rows, "batch")
            job.emit()

    return work


@router.post("/jobs", response_model=JobResponse)
async def create_batch_job(
    files: list[UploadFile] | None = File(None),
    directory_path: str = Form(""),
    model_name: str = Form("large-v3-turbo"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
    initial_prompt: str = Form(""),
    vad_filter: bool = Form(True),
):
    options = TranscriptionOptions(
        model_name=model_name,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        initial_prompt=initial_prompt or None,
        vad_filter=vad_filter,
    )
    paths: list[Path] = []
    if files:
        for file in files:
            uploaded = await save_upload(file)
            if allowed_audio_path(uploaded):
                paths.append(uploaded)
    elif directory_path:
        directory = Path(directory_path)
        if not directory.is_absolute():
            directory = ROOT_DIR / directory
        if not directory.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory not found: {directory_path}")
        paths = [path for path in sorted(directory.rglob("*")) if path.is_file() and allowed_audio_path(path)]

    if not paths:
        raise HTTPException(status_code=400, detail="No audio files found")

    job = job_registry.create("batch", batch_worker(paths, options))
    return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")


@router.post("/directory/jobs", response_model=JobResponse)
def create_directory_batch_job(request: DirectoryBatchRequest):
    directory = Path(request.directory_path)
    if not directory.is_absolute():
        directory = ROOT_DIR / directory
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory_path}")
    paths = [path for path in sorted(directory.rglob("*")) if path.is_file() and allowed_audio_path(path)]
    if not paths:
        raise HTTPException(status_code=400, detail="No audio files found")
    job = job_registry.create("batch", batch_worker(paths, request))
    return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")

