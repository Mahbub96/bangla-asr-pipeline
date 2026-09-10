import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.schemas import TranscriptionOptions
from backend.services.audio import sanitize_audio_input
from backend.services.exports import create_transcription_exports
from backend.services.jobs import Job
from backend.services.model_manager import model_manager
from backend.services.quality import postprocess_result, resolve_transcription_quality

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


def run_transcription(audio_path: Path, options: TranscriptionOptions) -> tuple[dict, dict]:
    clean_path = sanitize_audio_input(audio_path)
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
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
