from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import MAX_BATCH_FILES, resolve_within_allowed_roots
from backend.routers.transcribe import run_transcription, save_upload
from backend.schemas import DirectoryBatchRequest, JobResponse, TranscriptionOptions
from backend.services.audio import allowed_audio_path
from backend.services.exports import create_table_exports
from backend.services.jobs import Job, job_registry
from backend.services.storage import discard

router = APIRouter(prefix="/api/batch", tags=["batch"])


def _resolve_directory(raw: str) -> Path:
    """Resolve a caller-supplied directory, refusing paths outside the data roots."""
    try:
        directory = resolve_within_allowed_roots(raw)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {raw}")
    return directory


def _collect_audio(directory: Path) -> list[Path]:
    paths = [path for path in sorted(directory.rglob("*")) if path.is_file() and allowed_audio_path(path)]
    if not paths:
        raise HTTPException(status_code=400, detail="No audio files found")
    if len(paths) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch of {len(paths)} files exceeds the limit of {MAX_BATCH_FILES}",
        )
    return paths


def batch_worker(
    paths: list[Path],
    options: TranscriptionOptions,
    cleanup: bool = False,
    display_names: dict[Path, str] | None = None,
):
    """cleanup=True when `paths` are uploaded temp files this job owns."""
    names = display_names or {}

    def work(job: Job) -> None:
        rows = []
        try:
            for index, path in enumerate(paths, 1):
                label = names.get(path, path.name)
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.message = "Cancelled"
                    return
                job.message = f"Processing {label}"
                job.progress = (index - 1) / max(1, len(paths))
                job.emit()
                try:
                    result, _ = run_transcription(
                        path, options, create_exports=False, display_name=label
                    )
                    quality = result.get("quality", {})
                    rows.append(
                        {
                            "file": label,
                            "language": result["language"],
                            "confidence": result["language_probability"],
                            "profile": quality.get("profile", options.profile),
                            "repetition_score": quality.get("repetition_score"),
                            "warnings": " | ".join(quality.get("warnings", [])),
                            "duration_sec": result["duration_sec"],
                            "speed_factor": result["speed_factor"],
                            "text": result["text"],
                        }
                    )
                    job.append_log(f"{index}/{len(paths)} {label}: {result['text'][:120]}")
                except Exception as exc:
                    rows.append({"file": label, "language": "Error", "text": str(exc)})
                    job.append_log(f"{index}/{len(paths)} {label}: ERROR {exc}")
                job.result = rows
                job.progress = index / max(1, len(paths))
                # Replace the previous export pair rather than orphaning it.
                stale = job.exports
                job.exports = create_table_exports(rows, "batch")
                discard(*stale.values())
                job.emit()
        finally:
            if cleanup:
                discard(*paths)

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
    profile: str = Form("auto"),
    chunk_length: int = Form(30),
    vad_aggressiveness: str = Form("medium"),
    condition_on_previous_text: bool = Form(True),
    repetition_guard: bool = Form(True),
    hotwords: str = Form(""),
    output_script: str = Form("native"),
):
    options = TranscriptionOptions(
        model_name=model_name,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        initial_prompt=initial_prompt or None,
        vad_filter=vad_filter,
        profile=profile,
        chunk_length=chunk_length,
        vad_aggressiveness=vad_aggressiveness,
        condition_on_previous_text=condition_on_previous_text,
        repetition_guard=repetition_guard,
        hotwords=hotwords or None,
        output_script=output_script,
    )
    paths: list[Path] = []
    display_names: dict[Path, str] = {}
    uploaded = False
    if files:
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Batch of {len(files)} files exceeds the limit of {MAX_BATCH_FILES}",
            )
        uploaded = True
        try:
            for file in files:
                saved = await save_upload(file)
                if allowed_audio_path(saved):
                    paths.append(saved)
                    # Report the client's filename, not the scratch name.
                    display_names[saved] = Path(file.filename or saved.name).name
                else:
                    discard(saved)
        except Exception:
            discard(*paths)
            raise
    elif directory_path:
        paths = _collect_audio(_resolve_directory(directory_path))

    if not paths:
        raise HTTPException(status_code=400, detail="No audio files found")

    job = job_registry.create(
        "batch", batch_worker(paths, options, cleanup=uploaded, display_names=display_names)
    )
    return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")


@router.post("/directory/jobs", response_model=JobResponse)
def create_directory_batch_job(request: DirectoryBatchRequest):
    paths = _collect_audio(_resolve_directory(request.directory_path))
    job = job_registry.create("batch", batch_worker(paths, request))
    return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")
