#!/usr/bin/env python3
"""
Evaluation script to benchmark Whisper on a test/validation dataset.
Calculates Word Error Rate (WER) and Character Error Rate (CER) against ground truth transcripts.
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from transcribe import get_transcriber, transcribe_file

try:
    import jiwer
except ImportError:
    print("jiwer is required for evaluation. Run: pip install jiwer", file=sys.stderr)
    sys.exit(1)

def compute_metrics(ground_truth_list, hypothesis_list):
    """Compute WER and CER between ground truths and model predictions."""
    wer = jiwer.wer(ground_truth_list, hypothesis_list)
    cer = jiwer.cer(ground_truth_list, hypothesis_list)
    return wer, cer

def evaluate_dataset(metadata_csv, audio_dir, model_name="large-v3-turbo", language="bn", device=None, compute_type=None, models_dir="models", output_csv="evaluation_results.csv"):
    metadata_path = Path(metadata_csv)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Standardize column names
    audio_col = None
    for candidate in ["audio_path", "audio", "file_name", "filename", "path", "id"]:
        if candidate in df.columns:
            audio_col = candidate
            break

    text_col = None
    for candidate in ["sentence", "transcription", "ground_truth", "text", "transcript"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if not audio_col or not text_col:
        raise ValueError(
            f"CSV must contain audio path and text columns. Detected columns: {list(df.columns)}. "
            f"Expected audio column like 'audio_path'/'file_name' and text column like 'sentence'/'text'."
        )

    print(f"Loaded dataset: {len(df)} samples from {metadata_csv}")
    print(f"Audio column: '{audio_col}', Text column: '{text_col}'")

    model = get_transcriber(
        model_size=model_name,
        device=device,
        compute_type=compute_type,
        download_root=models_dir
    )

    results = []
    audio_base = Path(audio_dir) if audio_dir else metadata_path.parent

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        rel_audio = str(row[audio_col]).strip()
        ref_text = str(row[text_col]).strip()

        audio_full_path = audio_base / rel_audio
        if not audio_full_path.is_file():
            # Try searching relative to CSV directory or direct path
            if Path(rel_audio).is_file():
                audio_full_path = Path(rel_audio)
            else:
                print(f"Warning: Missing audio file {audio_full_path}, skipping.")
                continue

        try:
            pred = transcribe_file(model, audio_full_path, language=language)
            hyp_text = pred["text"].strip()
            
            # Per-sample metrics
            sample_wer = jiwer.wer(ref_text, hyp_text) if ref_text else 1.0
            sample_cer = jiwer.cer(ref_text, hyp_text) if ref_text else 1.0

            results.append({
                "audio_file": rel_audio,
                "duration_sec": pred["duration_sec"],
                "ground_truth": ref_text,
                "prediction": hyp_text,
                "detected_language": pred["language"],
                "wer": round(sample_wer, 4),
                "cer": round(sample_cer, 4)
            })
        except Exception as e:
            print(f"Error evaluating {rel_audio}: {e}")

    if not results:
        print("No samples were successfully evaluated.")
        return

    results_df = pd.DataFrame(results)
    
    # Overall corpus metrics
    all_refs = results_df["ground_truth"].tolist()
    all_hyps = results_df["prediction"].tolist()
    overall_wer, overall_cer = compute_metrics(all_refs, all_hyps)

    print("\n" + "=" * 50)
    print("                 EVALUATION RESULTS               ")
    print("=" * 50)
    print(f"Total Samples Evaluated: {len(results_df)}")
    print(f"Overall Word Error Rate (WER)     : {overall_wer:.2%}")
    print(f"Overall Character Error Rate (CER): {overall_cer:.2%}")
    print("=" * 50)

    out_p = Path(output_csv)
    results_df.to_csv(out_p, index=False, encoding="utf-8")
    print(f"\nDetailed predictions and sample-level errors saved to: {out_p.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on test/validation dataset.")
    parser.add_argument("--metadata", default="data/test/metadata.csv", help="Path to metadata CSV file.")
    parser.add_argument("--audio_dir", default="data/test/audio", help="Path to directory containing audio files.")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model (large-v3-turbo, large-v3, medium, etc.)")
    parser.add_argument("--language", default="bn", choices=["bn", "en", "auto"], help="Language code (default: 'bn' for Bangla).")
    parser.add_argument("--device", default=None, help="Device: 'cpu' or 'cuda'.")
    parser.add_argument("--models_dir", default="models", help="Directory where model weights are stored.")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV path for results.")

    args = parser.parse_args()
    evaluate_dataset(
        metadata_csv=args.metadata,
        audio_dir=args.audio_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        models_dir=args.models_dir,
        output_csv=args.output
    )

if __name__ == "__main__":
    main()
