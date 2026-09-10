import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas import TranscriptionOptions
from backend.services.audio import sanitize_audio_input
from backend.services.exports import create_transcription_exports
from backend.services.jobs import Job
from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


def run_transcription(audio_path: Path, options: TranscriptionOptions) -> tuple[dict, dict]:
    clean_path = sanitize_audio_input(audio_path)
    result = model_manager.transcribe(
        clean_path,
        model_name=options.model_name,
        device=options.device,
        compute_type=options.compute_type,
        language=options.language,
        beam_size=options.beam_size,
        initial_prompt=options.initial_prompt,
        vad_filter=options.vad_filter,
        temperature=options.temperature,
    )
    exports = create_transcription_exports(result)
    return result, exports


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    target = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    target.close()
    with open(target.name, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return Path(target.name)


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
        )
        path = await save_upload(audio)
        result, exports = run_transcription(path, options)
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
        from backend.services.jobs import job_registry

        job_registry._jobs["latest"] = transient_job
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
