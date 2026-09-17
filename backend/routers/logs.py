from fastapi import APIRouter, Query

from backend.services.jobs import job_registry
from backend.services.log_buffer import log_buffer

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/runtime")
def runtime_logs(limit: int = Query(default=300, ge=1, le=1000)):
    return {"logs": log_buffer.tail(limit)}


@router.get("/jobs/active")
def active_job_logs():
    return {"jobs": job_registry.active_summary()}
