#!/usr/bin/env bash
set -euo pipefail

# Prevent macOS from creating AppleDouble files while Docker archives context.
export COPYFILE_DISABLE=1
export DOCKER_BUILDKIT=1

mkdir -p logs

# External macOS volumes can create ._* AppleDouble sidecars. Docker may fail
# while reading their xattrs before it can ignore them, so remove them first.
find . -name '._*' -type f -delete

LOG_FILE="logs/docker-gpu-up.log"
echo "Starting Bangla ASR GPU stack..."
echo "Live log file: ${LOG_FILE}"
echo "Watch from another terminal: tail -f ${LOG_FILE}"

docker compose --progress plain -f docker-compose.yml -f docker-compose.gpu.yml up --build 2>&1 | tee "${LOG_FILE}"
