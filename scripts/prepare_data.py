#!/usr/bin/env python3
"""
Data preparation and directory initialization tool for Bangla/English ASR.
Sets up folder structure, template CSVs, and audio validation.
"""

import argparse
import csv
import os
from pathlib import Path

def init_directories(base_dir="."):
    """Creates the standard directory structure for speech datasets."""
    base = Path(base_dir)
    
    dirs = [
        base / "data" / "train" / "audio",
        base / "data" / "val" / "audio",
        base / "data" / "test" / "audio",
        base / "models"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {d}")

    # Create template metadata.csv for train, val, test if not present
    templates = [
        (base / "data" / "train" / "metadata.csv", [
            {"audio_path": "sample1.wav", "sentence": "আমি বাংলায় গান গাই।"},
            {"audio_path": "sample2.wav", "sentence": "Hello, how are you today?"}
        ]),
        (base / "data" / "val" / "metadata.csv", [
            {"audio_path": "sample_val1.wav", "sentence": "আজকে আবহাওয়া খুব ভালো।"}
        ]),
        (base / "data" / "test" / "metadata.csv", [
            {"audio_path": "sample_test1.wav", "sentence": "আমি ভাত খাই।"},
            {"audio_path": "sample_test2.wav", "sentence": "This is a test audio for recognition."}
        ]),
    ]

    for csv_file, sample_rows in templates:
        if not csv_file.exists():
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["audio_path", "sentence"])
                writer.writeheader()
                for row in sample_rows:
                    writer.writerow(row)
            print(f"Created template metadata: {csv_file}")
        else:
            print(f"Existing file found, keeping: {csv_file}")

    print("\nDataset directories initialized successfully!")

def validate_dataset(csv_path, audio_dir):
    """Checks metadata consistency and verifies that all audio files exist."""
    csv_p = Path(csv_path)
    audio_p = Path(audio_dir)

    if not csv_p.exists():
        print(f"Error: CSV file '{csv_p}' not found.")
        return

    import pandas as pd
    df = pd.read_csv(csv_p)
    print(f"\nValidating {csv_p} (Total rows: {len(df)})...")

    missing = 0
    for idx, row in df.iterrows():
        audio_name = str(row["audio_path"]).strip()
        full_audio = audio_p / audio_name
        if not full_audio.exists():
            print(f"  Missing audio file: {full_audio}")
            missing += 1

    if missing == 0:
        print("✓ All audio files referenced in metadata exist!")
    else:
        print(f"✗ Warning: {missing} audio file(s) referenced in metadata are missing.")

def main():
    parser = argparse.ArgumentParser(description="Initialize directories and validate speech dataset.")
    parser.add_argument("--init", action="store_true", help="Initialize data/train, data/val, data/test directories and templates.")
    parser.add_argument("--validate", action="store_true", help="Validate audio files referenced in a metadata CSV.")
    parser.add_argument("--csv", default="data/test/metadata.csv", help="Path to metadata CSV (for validation).")
    parser.add_argument("--audio_dir", default="data/test/audio", help="Path to audio directory (for validation).")

    args = parser.parse_args()

    if args.init or (not args.init and not args.validate):
        init_directories()

    if args.validate:
        validate_dataset(args.csv, args.audio_dir)

if __name__ == "__main__":
    main()
