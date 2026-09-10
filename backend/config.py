from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
TMP_DIR = ROOT_DIR / ".tmp" / "api"

TMP_DIR.mkdir(parents=True, exist_ok=True)

