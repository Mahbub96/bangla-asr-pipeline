#!/usr/bin/env bash
# One-click launcher for the Bangla & English ASR React + FastAPI Studio.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "          Bangla & English ASR Studio Launcher              "
echo "============================================================"
echo "Frontend UI : http://127.0.0.1:5173"
echo "Backend API : http://127.0.0.1:8000"
echo "============================================================"
echo ""

exec "$SCRIPT_DIR/run_ui.sh" "$@"

