import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.config import MAX_UPLOAD_BYTES
from backend.schemas import TranscriptionOptions
from backend.services.audio import sanitize_audio_input
from backend.services.exports import create_transcription_exports
from backend.services.jobs import Job, job_registry
from backend.services.model_manager import model_manager
from backend.services.quality import postprocess_result, resolve_transcription_quality
from backend.services.storage import discard, new_temp_path

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


def run_transcription(
    audio_path: Path,
    options: TranscriptionOptions,
    create_exports: bool = True,
    display_name: str | None = None,
) -> tuple[dict, dict]:
    """Transcribe one file.

    ``create_exports=False`` for bulk callers (batch, evaluate) that only read
    ``result``: generating txt/srt/vtt/json per file would write four scratch
    files per item that nothing ever downloads.

    ``display_name`` is what the client should see in ``result["file"]``; without
    it an upload would echo back the server's scratch filename.
    """
    clean_path = sanitize_audio_input(audio_path)
    try:
        quality = resolve_transcription_quality(
            profile=options.profile,
            language=options.language,
            beam_size=options.beam_size,
            temperature=options.temperature,
            vad_filter=options.vad_filter,
            vad_aggressiveness=options.vad_aggressiveness,
            condition_on_previous_text=options.condition_on_previous_text,
            chunk_length=options.chunk_length,
            initial_prompt=options.initial_prompt,
            hotwords=options.hotwords,
        )
        result = model_manager.transcribe(
            clean_path,
            model_name=options.model_name,
            device=options.device,
            compute_type=options.compute_type,
            language=options.language,
            **quality.transcribe_kwargs,
        )
        result = postprocess_result(result, quality.profile, options.repetition_guard, options.output_script)
        # Don't leak server-side scratch paths to API clients.
        result["file"] = display_name or Path(audio_path).name
        exports = create_transcription_exports(result) if create_exports else {}
        return result, exports
    finally:
        # ffmpeg output is a separate temp file; the caller owns audio_path.
        if clean_path != Path(audio_path):
            discard(clean_path)


async def save_upload(upload: UploadFile) -> Path:
    """Stream an upload to managed scratch space, enforcing the size cap."""
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    target = new_temp_path(suffix)
    written = 0
    try:
        with open(target, "wb") as out:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)
    except Exception:
        discard(target)
        raise
    if written == 0:
        discard(target)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return target


@router.post("")
async def transcribe_upload(
    audio: UploadFile = File(...),
    model_name: str = Form("large-v3-turbo"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
    initial_prompt: str = Form(""),
    vad_filter: bool = Form(True),
    device: str = Form("auto"),
    compute_type: str = Form("auto"),
    profile: str = Form("auto"),
    chunk_length: int = Form(30),
    vad_aggressiveness: str = Form("medium"),
    condition_on_previous_text: bool = Form(True),
    repetition_guard: bool = Form(True),
    hotwords: str = Form(""),
    output_script: str = Form("native"),
):
    try:
        options = TranscriptionOptions(
            model_name=model_name,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=initial_prompt or None,
            vad_filter=vad_filter,
            device=device,
            compute_type=compute_type,
            profile=profile,
            chunk_length=chunk_length,
            vad_aggressiveness=vad_aggressiveness,
            condition_on_previous_text=condition_on_previous_text,
            repetition_guard=repetition_guard,
            hotwords=hotwords or None,
            output_script=output_script,
        )
        path = await save_upload(audio)
        try:
            result, exports = run_transcription(
                path, options, display_name=Path(audio.filename or "audio.wav").name
            )
        finally:
            discard(path)
        transient_job = Job(
            id="latest",
            kind="transcribe",
            status="completed",
            progress=1.0,
            message="Completed",
            result=result,
            exports=exports,
        )
        payload = transient_job.snapshot()
        payload["result"] = result
        payload["exports"] = {key: f"/api/exports/latest/{key}" for key in exports}
        # Replaces any previous "latest" and deletes its export files.
        job_registry.put(transient_job)
        return payload
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
