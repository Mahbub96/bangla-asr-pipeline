#!/usr/bin/env python3
"""
Pre-download Whisper models to the local 'models/' directory for offline or fast execution.
"""

import argparse
import sys
from pathlib import Path

def download_whisper_model(model_name="large-v3-turbo", models_dir="models"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: 'faster-whisper' is not installed yet. Activate .venv and run pip install faster-whisper.", file=sys.stderr)
        sys.exit(1)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/verifying Whisper model: '{model_name}'")
    print(f"Destination cache directory: {models_path.resolve()}")
    print("This may take a few minutes depending on your internet connection...\n")

    # Initializing WhisperModel triggers the download and verification
    # Using 'cpu' and 'int8' as safe defaults for loading check
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(models_path)
    )

    print(f"\n✓ Successfully downloaded and verified '{model_name}' in {models_path.resolve()}!")

def main():
    parser = argparse.ArgumentParser(description="Download Whisper models locally.")
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        choices=["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"],
        help="Whisper model to download (default: large-v3-turbo)."
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Target directory to store model weights (default: ./models)."
    )

    args = parser.parse_args()
    download_whisper_model(args.model, args.models_dir)

if __name__ == "__main__":
    main()
