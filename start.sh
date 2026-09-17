#!/usr/bin/env bash
set -euo pipefail

# Prevent macOS from creating AppleDouble files while Docker archives context.
export COPYFILE_DISABLE=1
export DOCKER_BUILDKIT=1

mkdir -p logs
find . -name '._*' -type f -delete

LOG_FILE="logs/docker-up.log"
echo "Starting Bangla ASR stack (CPU/macOS-compatible)..."
echo "Live log file: ${LOG_FILE}"
echo "Watch from another terminal: tail -f ${LOG_FILE}"

docker compose --progress plain -f docker-compose.yml up --build 2>&1 | tee "${LOG_FILE}"
