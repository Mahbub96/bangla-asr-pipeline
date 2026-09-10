#!/usr/bin/env bash
# Launch the Bangla & English ASR React + FastAPI Studio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Virtual environment not found at .venv"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install --prefix "$SCRIPT_DIR/frontend"
fi

echo "Starting Bangla & English ASR Studio..."
echo "Backend API : http://127.0.0.1:8000"
echo "Frontend UI : http://127.0.0.1:5173"
echo "Microphone access works from localhost / 127.0.0.1."

"$PYTHON_EXEC" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
    kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

npm run dev --prefix "$SCRIPT_DIR/frontend" -- "$@"
