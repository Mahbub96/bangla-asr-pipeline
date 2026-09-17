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

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5173}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

echo "Starting Bangla & English ASR Studio..."
echo "Backend API : http://127.0.0.1:${BACKEND_PORT}  | Network: http://192.168.50.140:${BACKEND_PORT}"
echo "Frontend UI : http://127.0.0.1:${PORT}  | Network: http://192.168.50.140:${PORT}"
echo "Note: Modern browsers require localhost or HTTPS for microphone access."

"$PYTHON_EXEC" -m uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

cleanup() {
    kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

npm run dev --prefix "$SCRIPT_DIR/frontend" -- --host "$HOST" --port "$PORT" "$@"
