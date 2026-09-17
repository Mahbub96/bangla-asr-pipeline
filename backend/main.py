import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from backend.config import CORS_ORIGINS, FRONTEND_DIST, JOB_RETENTION_SECONDS
from backend.routers import batch, diagnostics, evaluate, jobs, logs, train, transcribe
from backend.services.jobs import job_registry
from backend.services.log_buffer import log_buffer
from backend.services.storage import sweep_orphans

logging.basicConfig(
    level=os.getenv("ASR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("asr")
logging.getLogger().addHandler(log_buffer)

JANITOR_INTERVAL_SECONDS = 900


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Background janitor: bounds job history and scratch-disk usage.

    Without it a long-lived container accumulates finished jobs and their export
    files indefinitely.
    """

    async def janitor() -> None:
        while True:
            try:
                await asyncio.sleep(JANITOR_INTERVAL_SECONDS)
                evicted = job_registry.prune()
                # Grace period well past the retention window so files belonging
                # to a live job are never swept out from under it.
                swept = sweep_orphans(JOB_RETENTION_SECONDS + 3600)
                if evicted or swept:
                    logger.info("janitor: evicted %d job(s), removed %d orphan file(s)", evicted, swept)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("janitor pass failed")

    task = asyncio.create_task(janitor())
    logger.info("ASR API ready (job retention %ds)", JOB_RETENTION_SECONDS)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Bangla & English ASR Studio API", version="1.0.0", lifespan=lifespan)

# ASR_CORS_ORIGINS="*" (the container default) allows any origin. Credentials
# cannot be combined with a wildcard origin list, so they are disabled in that
# mode — the API is token-less anyway.
_allow_any_origin = "*" in CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if _allow_any_origin else CORS_ORIGINS,
    allow_origin_regex=r"^https?://.*" if _allow_any_origin else None,
    allow_credentials=not _allow_any_origin,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(batch.router)
app.include_router(evaluate.router)
app.include_router(train.router)
app.include_router(diagnostics.router)
app.include_router(jobs.router)
app.include_router(logs.router)


@app.get("/api/health")
def health():
    return {"ok": True, "jobs": job_registry.stats()}


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        # The SPA fallback must not swallow unknown API routes: returning
        # index.html for /api/typo hides the error behind a 200 HTML page.
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(FRONTEND_DIST / "index.html")
