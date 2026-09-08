#!/usr/bin/env bash
# Launch the Bangla & English ASR Gradio Web Interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Virtual environment not found at .venv"
    exit 1
fi

echo "Starting Bangla & English ASR Web UI on http://localhost:7860 ..."
"$PYTHON_EXEC" app.py
