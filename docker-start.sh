#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-platform Docker launcher for the Bangla & English ASR Studio.
# Works on Linux, macOS and Windows (Git Bash / WSL).
#
#   ./docker-start.sh              production stack (UI on :8080)
#   ./docker-start.sh dev          dev stack, hot reload (UI on :5173)
#   ./docker-start.sh gpu          production stack on NVIDIA CUDA
#   ./docker-start.sh down         stop everything
#   ./docker-start.sh logs         follow logs
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! docker compose version >/dev/null 2>&1; then
    echo "Error: 'docker compose' (v2) is required. Install Docker Desktop or the compose plugin." >&2
    exit 1
fi

MODE="${1:-prod}"
shift || true

case "$MODE" in
    prod|up|"")
        FILES=(-f docker-compose.yml)
        UI_URL="http://localhost:${FRONTEND_PORT:-8080}"
        ;;
    dev)
        FILES=(-f docker-compose.yml -f docker-compose.dev.yml)
        UI_URL="http://localhost:${DEV_FRONTEND_PORT:-5173}"
        ;;
    gpu)
        FILES=(-f docker-compose.yml -f docker-compose.gpu.yml)
        UI_URL="http://localhost:${FRONTEND_PORT:-8080}"
        ;;
    down)
        docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.gpu.yml down "$@"
        exit 0
        ;;
    logs)
        docker compose -f docker-compose.yml logs -f "$@"
        exit 0
        ;;
    *)
        echo "Usage: $0 [prod|dev|gpu|down|logs]" >&2
        exit 1
        ;;
esac

# Bind-mounted dirs must exist before compose creates them as root.
mkdir -p models data checkpoints

echo "============================================================"
echo "     Bangla & English ASR Studio  —  Docker (${MODE})"
echo "============================================================"
echo "UI  : ${UI_URL}"
echo "API : http://localhost:${BACKEND_PORT:-8000}"
echo "Note: browsers only grant microphone access on localhost or HTTPS."
echo "============================================================"

docker compose "${FILES[@]}" up -d --build "$@"

echo
docker compose "${FILES[@]}" ps
echo
echo "Follow logs:  docker compose ${FILES[*]} logs -f"
echo "Stop:         ./docker-start.sh down"
