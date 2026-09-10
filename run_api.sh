#!/usr/bin/env bash
# Launch only the FastAPI backend for the Bangla & English ASR Studio.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Virtual environment not found at .venv"
    exit 1
fi

"$PYTHON_EXEC" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
