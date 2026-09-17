import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    """Allow container deployments to relocate data dirs without code changes."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


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

# Server-side filesystem access (batch-by-directory, evaluation CSVs) is confined
# to these roots. Without this, an API caller could point the server at any path
# on the host, e.g. /etc, and enumerate it.
ALLOWED_DATA_ROOTS = [
    path.resolve()
    for path in {
        DATA_DIR,
        ROOT_DIR / "data",
        *(
            Path(entry).expanduser().resolve()
            for entry in os.getenv("ASR_EXTRA_DATA_ROOTS", "").split(":")
            if entry.strip()
        ),
    }
]

# Job retention. Finished jobs hold their exports on disk until evicted, so these
# two values bound both memory and scratch-disk growth.
JOB_RETENTION_SECONDS = _int_from_env("ASR_JOB_RETENTION_SECONDS", 6 * 60 * 60)
JOB_HISTORY_LIMIT = _int_from_env("ASR_JOB_HISTORY_LIMIT", 200)

# Reject oversized uploads before they are streamed to disk (bytes).
MAX_UPLOAD_BYTES = _int_from_env("ASR_MAX_UPLOAD_BYTES", 512 * 1024 * 1024)
MAX_BATCH_FILES = _int_from_env("ASR_MAX_BATCH_FILES", 200)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


def resolve_within_allowed_roots(raw_path: str) -> Path:
    """Resolve a caller-supplied path, refusing anything outside the data roots.

    Relative paths are interpreted against ROOT_DIR, matching the previous
    behaviour; the difference is that the resolved result must now sit inside an
    allowed root, which also defeats ``../`` traversal and symlink escapes.
    """
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    resolved = candidate.resolve()

    for root in ALLOWED_DATA_ROOTS:
        if resolved == root or root in resolved.parents:
            return resolved

    allowed = ", ".join(str(root) for root in ALLOWED_DATA_ROOTS)
    raise PermissionError(f"Path '{raw_path}' is outside the permitted data directories ({allowed})")
