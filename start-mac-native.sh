#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export COPYFILE_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="${HF_HOME:-$PWD/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models}"
export ASR_HOST="${ASR_HOST:-0.0.0.0}"
export ASR_PORT="${ASR_PORT:-8000}"
export VITE_API_BASE="${VITE_API_BASE:-http://localhost:${ASR_PORT}}"

mkdir -p logs
find . -name '._*' -type f -delete
find .venv -name '._*' -type f -delete 2>/dev/null || true

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run ./setup-mac-mps.sh first." >&2
  exit 1
fi
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm ci)
fi

BACKEND_LOG="logs/mac-native-backend.log"
FRONTEND_LOG="logs/mac-native-frontend.log"

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then kill "${BACKEND_PID}" 2>/dev/null || true; fi
  if [ -n "${FRONTEND_PID:-}" ]; then kill "${FRONTEND_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "Starting native backend on http://localhost:${ASR_PORT}"
.venv/bin/python -m uvicorn backend.main:app --host "${ASR_HOST}" --port "${ASR_PORT}" 2>&1 | tee "${BACKEND_LOG}" &
BACKEND_PID=$!

echo "Starting native frontend on http://localhost:5173"
(cd frontend && npm run dev 2>&1 | tee "../${FRONTEND_LOG}") &
FRONTEND_PID=$!

echo "Backend log:  ${BACKEND_LOG}"
echo "Frontend log: ${FRONTEND_LOG}"
echo "Open: http://localhost:5173"
echo "Stop: Ctrl+C"

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
