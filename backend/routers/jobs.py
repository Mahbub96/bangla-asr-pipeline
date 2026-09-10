import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from backend.services.jobs import job_registry

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = job_registry.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
def stream_job_events(job_id: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        yield f"data: {json.dumps({'type': 'snapshot', 'job': job.snapshot()}, ensure_ascii=False)}\n\n"
        while True:
            if job.status in {"completed", "failed", "cancelled"} and job.events.empty():
                yield f"data: {json.dumps({'type': 'done', 'job': job.snapshot()}, ensure_ascii=False)}\n\n"
                break
            try:
                event = job.events.get(timeout=1)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:
                yield f": keepalive {time.time()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/exports/{job_id}/{export_format}")
def download_export(job_id: str, export_format: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.exports.get(export_format)
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path, filename=f"{job.kind}_{job.id}.{export_format}")

