#!/usr/bin/env python3
"""
Evaluation script to benchmark Whisper on a test/validation dataset.
Calculates Word Error Rate (WER) and Character Error Rate (CER) against ground truth transcripts.
"""

import argparse
import glob
import os
import sys
import tempfile
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

def load_audio_array(audio_path: Path, target_sr: int = 16000):
    import librosa
    import soundfile as sf

    array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return array, sr

def get_transformers_transcriber(model_name, language="bn", device=None):
    import json
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_path = Path(model_name)
    adapter_config = model_path / "adapter_config.json"
    processor = WhisperProcessor.from_pretrained(model_name, language="bengali" if language == "bn" else None, task="transcribe")

    if adapter_config.is_file():
        from peft import PeftModel

        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"LoRA adapter config missing base_model_name_or_path: {adapter_config}")
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, model_name)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(device)
    model.eval()
    print(f"Loading Transformers Whisper model '{model_name}' on {device.upper()}...")
    return processor, model, device

def transcribe_file_transformers(processor, model, device, audio_path: Path, language="bn"):
    import torch

    array, sr = load_audio_array(audio_path)
    inputs = processor(array, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = None
    if language != "auto":
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="bengali" if language == "bn" else "english",
            task="transcribe",
        )
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return {
        "text": text,
        "duration_sec": round(len(array) / sr, 2),
        "language": language,
    }

def resolve_parquet_files(patterns: str) -> list[str]:
    files: list[str] = []
    for pattern in [item.strip() for item in patterns.split(",") if item.strip()]:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif Path(pattern).is_file():
            files.append(pattern)
    deduped = list(dict.fromkeys(files))
    if not deduped:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return deduped

def iter_parquet_rows(parquet_patterns: str, max_samples: int | None = None):
    """Yield rows from ASR parquet shards containing audio.bytes + transcription."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for --parquet evaluation. Install requirements_gpu.txt.") from exc

    emitted = 0
    for parquet_path in resolve_parquet_files(parquet_patterns):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                audio = row.get("audio") or {}
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                ref_text = next((row.get(c) for c in ["sentence", "transcription", "ground_truth", "text", "transcript"] if row.get(c)), None)
                if not audio_bytes or not ref_text:
                    continue
                name = Path(audio.get("path") or row.get("file_path") or f"sample_{emitted:08d}.wav").name
                yield {
                    "audio_bytes": audio_bytes,
                    "audio_file": name,
                    "ground_truth": str(ref_text).strip(),
                    "source_path": str(parquet_path),
                }
                emitted += 1
                if max_samples and emitted >= max_samples:
                    return

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

def evaluate_parquet_dataset(parquet_patterns, model_name="large-v3-turbo", language="bn", device=None, compute_type=None, models_dir="models", output_csv="evaluation_results.csv", max_samples=None, engine="faster-whisper"):
    rows_iter = iter_parquet_rows(parquet_patterns, max_samples=max_samples)
    processor = None
    resolved_device = None
    if engine == "transformers":
        processor, model, resolved_device = get_transformers_transcriber(model_name, language=language, device=device)
    else:
        model = get_transcriber(
            model_size=model_name,
            device=device,
            compute_type=compute_type,
            download_root=models_dir
        )

    results = []
    with tempfile.TemporaryDirectory(prefix="asr_parquet_eval_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for index, row in enumerate(tqdm(rows_iter, desc="Evaluating parquet"), 1):
            audio_path = tmpdir_path / row["audio_file"]
            audio_path.write_bytes(row["audio_bytes"])
            ref_text = row["ground_truth"]
            try:
                if engine == "transformers":
                    assert processor is not None and resolved_device is not None
                    pred = transcribe_file_transformers(processor, model, resolved_device, audio_path, language=language)
                else:
                    pred = transcribe_file(model, audio_path, language=language)
                hyp_text = pred["text"].strip()
                results.append({
                    "audio_file": row["audio_file"],
                    "source_path": row["source_path"],
                    "duration_sec": pred["duration_sec"],
                    "ground_truth": ref_text,
                    "prediction": hyp_text,
                    "detected_language": pred["language"],
                    "wer": round(jiwer.wer(ref_text, hyp_text) if ref_text else 1.0, 4),
                    "cer": round(jiwer.cer(ref_text, hyp_text) if ref_text else 1.0, 4),
                })
            except Exception as e:
                print(f"Error evaluating {row['audio_file']}: {e}")
            finally:
                audio_path.unlink(missing_ok=True)

    if not results:
        print("No samples were successfully evaluated.")
        return

    results_df = pd.DataFrame(results)
    overall_wer, overall_cer = compute_metrics(results_df["ground_truth"].tolist(), results_df["prediction"].tolist())
    print("\n" + "=" * 50)
    print("              PARQUET EVALUATION RESULTS          ")
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
    parser.add_argument("--parquet", default=None, help="Comma-separated parquet files/globs with embedded audio bytes. Avoids permanent WAV extraction.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick parquet evaluation smoke tests.")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model (large-v3-turbo, large-v3, medium, etc.)")
    parser.add_argument("--language", default="bn", choices=["bn", "en", "auto"], help="Language code (default: 'bn' for Bangla).")
    parser.add_argument("--device", default=None, help="Device: 'cpu' or 'cuda'.")
    parser.add_argument("--compute_type", default=None, help="Quantization type, e.g. int8, float32, float16.")
    parser.add_argument("--models_dir", default="models", help="Directory where model weights are stored.")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV path for results.")
    parser.add_argument("--engine", default="faster-whisper", choices=["faster-whisper", "transformers"], help="Transcription backend for evaluation.")

    args = parser.parse_args()
    if args.parquet:
        evaluate_parquet_dataset(
            parquet_patterns=args.parquet,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            models_dir=args.models_dir,
            output_csv=args.output,
            max_samples=args.max_samples,
            engine=args.engine,
        )
    else:
        evaluate_dataset(
            metadata_csv=args.metadata,
            audio_dir=args.audio_dir,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            models_dir=args.models_dir,
            output_csv=args.output
        )

if __name__ == "__main__":
    main()
