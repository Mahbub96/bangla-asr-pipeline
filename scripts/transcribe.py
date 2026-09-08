#!/usr/bin/env python3
"""
Transcription script using Whisper (faster-whisper).
Optimized for CPU/GPU execution with automatic language detection or explicit Bangla ('bn') / English ('en').
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

def get_transcriber(model_size="large-v3-turbo", device=None, compute_type=None, download_root=None):
    """Load faster-whisper model with optimal settings."""
    from faster_whisper import WhisperModel

    if device is None:
        # Default to CPU unless CUDA is explicitly functional
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model '{model_size}' on {device.upper()} (compute_type: {compute_type})...")
    start_time = time.time()
    
    model = WhisperModel(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
        download_root=download_root
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.\n")
    return model

def transcribe_file(model, audio_path, language=None, beam_size=5):
    """Transcribe a single audio file."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # If language is 'auto', pass None to let whisper auto-detect
    lang = None if (language in [None, "auto"]) else language

    start_time = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=lang,
        vad_filter=True, # filter out non-speech silence
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    duration = info.duration

    collected_segments = []
    full_text_list = []

    for seg in segments:
        collected_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        })
        full_text_list.append(seg.text.strip())

    full_text = " ".join(full_text_list).strip()
    elapsed = time.time() - start_time

    return {
        "file": str(audio_path),
        "duration_sec": round(duration, 2),
        "transcription_time_sec": round(elapsed, 2),
        "speed_factor": round(duration / elapsed, 2) if elapsed > 0 else 0,
        "language": detected_lang,
        "language_probability": round(lang_prob, 4),
        "text": full_text,
        "segments": collected_segments
    }

def process_directory(model, dir_path, language=None, output_file=None):
    """Batch transcribe all audio files in a directory."""
    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = [p for p in Path(dir_path).rglob("*") if p.suffix.lower() in valid_exts]

    if not audio_files:
        print(f"No audio files found in {dir_path}")
        return []

    print(f"Found {len(audio_files)} audio files in {dir_path}. Beginning transcription...\n")
    results = []

    for idx, audio_file in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] Processing: {audio_file.name}")
        try:
            res = transcribe_file(model, audio_file, language=language)
            results.append(res)
            print(f"  → Lang: {res['language']} ({res['language_probability']:.2%}) | Duration: {res['duration_sec']}s")
            print(f"  → Text: {res['text']}\n")
        except Exception as e:
            print(f"  [ERROR] Failed to transcribe {audio_file.name}: {e}\n")

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Transcribe Bangla/English audio using Whisper.")
    parser.add_argument("input", help="Path to an audio file or directory containing audio files.")
    parser.add_argument("--model", default="large-v3-turbo", help="Model size or path (default: large-v3-turbo). Options: large-v3, large-v3-turbo, medium, small, base, tiny.")
    parser.add_argument("--language", default="auto", choices=["auto", "bn", "en"], help="Target language code: 'bn' (Bangla), 'en' (English), or 'auto'.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Compute device ('cpu' or 'cuda'). Auto-detected if omitted.")
    parser.add_argument("--compute_type", default=None, help="Quantization type: 'int8', 'float32', 'float16'. Default is 'int8' for CPU.")
    parser.add_argument("--models_dir", default="models", help="Directory where model weights are stored/cached.")
    parser.add_argument("--output", default=None, help="Optional JSON file to save transcription results.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Initialize model
    model = get_transcriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.models_dir
    )

    if input_path.is_file():
        result = transcribe_file(model, input_path, language=args.language)
        print("=" * 60)
        print(f"File: {result['file']}")
        print(f"Detected Language: {result['language']} ({result['language_probability']:.2%})")
        print(f"Audio Duration: {result['duration_sec']}s | Transcribe Time: {result['transcription_time_sec']}s (Speed: {result['speed_factor']}x)")
        print("-" * 60)
        print("Transcription:")
        print(result['text'])
        print("=" * 60)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.output}")

    elif input_path.is_dir():
        process_directory(model, input_path, language=args.language, output_file=args.output)

if __name__ == "__main__":
    main()
