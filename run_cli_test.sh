#!/usr/bin/env bash
# Quick CLI test runner for transcribing audio files without starting the web app.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python virtual environment not found at .venv"
    echo "Please run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

DEFAULT_MODEL="tiny"
if ls models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*/model.bin 1> /dev/null 2>&1; then
    DEFAULT_MODEL="large-v3-turbo"
elif ls models/models--Systran--faster-whisper-large-v3-turbo/snapshots/*/model.bin 1> /dev/null 2>&1; then
    DEFAULT_MODEL="large-v3-turbo"
fi

TARGET="${1:-data/test/audio}"
MODEL="${2:-$DEFAULT_MODEL}"
LANG_OPT="${3:-auto}"

echo "============================================================"
echo "          Bangla & English ASR CLI Test Runner              "
echo "============================================================"
echo "Target Input   : $TARGET"
echo "Whisper Model  : $MODEL"
echo "Language Mode  : $LANG_OPT"
echo "============================================================"
echo ""

"$PYTHON_EXEC" scripts/transcribe.py "$TARGET" \
    --model "$MODEL" \
    --language "$LANG_OPT" \
    --output "test_transcriptions.json"

echo ""
echo "Transcriptions completed! Saved details to: test_transcriptions.json"
