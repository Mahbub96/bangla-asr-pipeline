import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.services.jobs import TERMINAL_STATES, job_registry

router = APIRouter(prefix="/api", tags=["jobs"])

# Emitted while a job is quiet so proxies and load balancers keep the
# connection open instead of timing it out.
KEEPALIVE_SECONDS = 15.0


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
async def stream_job_events(job_id: str, request: Request):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        yield f"data: {json.dumps({'type': 'snapshot', 'job': job.snapshot()}, ensure_ascii=False)}\n\n"
        last_keepalive = time.monotonic()
        while True:
            # Without this the generator loops forever on a job that never
            # reaches a terminal state, leaking a worker thread per dead client.
            if await request.is_disconnected():
                return
            if job.status in TERMINAL_STATES and job.events.empty():
                yield f"data: {json.dumps({'type': 'done', 'job': job.snapshot()}, ensure_ascii=False)}\n\n"
                return
            try:
                # get_nowait + sleep keeps the event loop free; a blocking
                # queue.get() inside async would stall the whole server.
                event = job.events.get_nowait()
            except Exception:
                now = time.monotonic()
                if now - last_keepalive >= KEEPALIVE_SECONDS:
                    last_keepalive = now
                    yield f": keepalive {now}\n\n"
                await asyncio.sleep(0.25)
                continue
            last_keepalive = time.monotonic()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/exports/{job_id}/{export_format}")
def download_export(job_id: str, export_format: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.exports.get(export_format)
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(path, filename=f"{job.kind}_{job.id}.{export_format}")
