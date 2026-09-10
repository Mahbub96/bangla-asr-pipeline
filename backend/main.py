from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from backend.config import FRONTEND_DIST
from backend.routers import batch, diagnostics, evaluate, jobs, train, transcribe

app = FastAPI(title="Bangla & English ASR Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
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

