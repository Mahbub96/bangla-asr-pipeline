import shutil
import tempfile
from pathlib import Path

import jiwer
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import ROOT_DIR
from backend.routers.transcribe import run_transcription
from backend.schemas import EvaluateRequest, JobResponse
from backend.services.exports import create_table_exports
from backend.services.jobs import Job, job_registry

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


def evaluate_worker(request: EvaluateRequest):
    def work(job: Job) -> None:
        csv_path = Path(request.metadata_csv_path)
        audio_base = Path(request.audio_dir)
        if not csv_path.is_absolute():
            csv_path = ROOT_DIR / csv_path
        if not audio_base.is_absolute():
            audio_base = ROOT_DIR / audio_base
        df = pd.read_csv(csv_path)
        audio_col = next((c for c in ["audio_path", "audio", "file_name", "filename", "path"] if c in df.columns), None)
        text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text", "transcript"] if c in df.columns), None)
        if not audio_col or not text_col:
            raise ValueError(f"CSV must include audio and text columns. Found: {list(df.columns)}")

        rows = []
        for index, row in df.iterrows():
            if job.cancel_requested:
                job.status = "cancelled"
                job.message = "Cancelled"
                return
            rel_audio = str(row[audio_col]).strip()
            ref = str(row[text_col]).strip()
            audio_path = audio_base / rel_audio
            if not audio_path.is_file() and Path(rel_audio).is_file():
                audio_path = Path(rel_audio)
            if not audio_path.is_file():
                job.append_log(f"Missing audio: {rel_audio}")
                continue
            result, _ = run_transcription(audio_path, request)
            hyp = result["text"].strip()
            rows.append(
                {
                    "file": rel_audio,
                    "ground_truth": ref,
                    "prediction": hyp,
                    "wer": jiwer.wer(ref, hyp) if ref else 1.0,
                    "cer": jiwer.cer(ref, hyp) if ref else 1.0,
                }
            )
            refs = [item["ground_truth"] for item in rows]
            hyps = [item["prediction"] for item in rows]
            job.result = {
                "rows": rows,
                "overall_wer": jiwer.wer(refs, hyps) if rows else None,
                "overall_cer": jiwer.cer(refs, hyps) if rows else None,
            }
            job.progress = (index + 1) / max(1, len(df))
            job.exports = create_table_exports(rows, "evaluation")
            job.emit()

    return work


@router.post("/jobs", response_model=JobResponse)
async def create_evaluate_job(
    csv_file: UploadFile | None = File(None),
    metadata_csv_path: str = Form("data/test/metadata.csv"),
    audio_dir: str = Form("data/test/audio"),
    model_name: str = Form("large-v3-turbo"),
    language: str = Form("auto"),
):
    if csv_file:
        target = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        target.close()
        with open(target.name, "wb") as out:
            shutil.copyfileobj(csv_file.file, out)
        metadata_csv_path = target.name
    request = EvaluateRequest(metadata_csv_path=metadata_csv_path, audio_dir=audio_dir, model_name=model_name, language=language)
    if not Path(request.metadata_csv_path).is_absolute() and not (ROOT_DIR / request.metadata_csv_path).is_file():
        raise HTTPException(status_code=400, detail=f"CSV not found: {request.metadata_csv_path}")
    job = job_registry.create("evaluate", evaluate_worker(request))
    return JobResponse(job_id=job.id, status_url=f"/api/jobs/{job.id}", events_url=f"/api/jobs/{job.id}/events")

