import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    """Allow container deployments to relocate data dirs without code changes."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


SCRIPTS_DIR = _path_from_env("ASR_SCRIPTS_DIR", ROOT_DIR / "scripts")
MODELS_DIR = _path_from_env("ASR_MODELS_DIR", ROOT_DIR / "models")
DATA_DIR = _path_from_env("ASR_DATA_DIR", ROOT_DIR / "data")
FRONTEND_DIST = _path_from_env("ASR_FRONTEND_DIST", ROOT_DIR / "frontend" / "dist")
TMP_DIR = _path_from_env("ASR_TMP_DIR", ROOT_DIR / ".tmp" / "api")

# Comma-separated list of allowed origins, or "*" to allow any.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ASR_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8080,http://localhost:8080",
    ).split(",")
    if origin.strip()
]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
