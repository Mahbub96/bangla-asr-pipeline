from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from backend.config import CORS_ORIGINS, FRONTEND_DIST
from backend.routers import batch, diagnostics, evaluate, jobs, train, transcribe

app = FastAPI(title="Bangla & English ASR Studio API", version="1.0.0")

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


@app.get("/api/health")
def health():
    return {"ok": True}


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str = ""):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(index)

