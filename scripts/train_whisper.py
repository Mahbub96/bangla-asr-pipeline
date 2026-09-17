#!/usr/bin/env python3
"""
Whisper Fine-Tuning Script for Bangla/English ASR.
Designed for execution on a large compute machine (GPU / Multi-GPU).
Supports full fine-tuning or parameter-efficient LoRA / QLoRA.
"""

import argparse
import csv
import glob
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import IterableDataset

try:
    import evaluate
    from datasets import Audio, Dataset, load_dataset
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
except ImportError:
    # Notice for local execution
    class TrainerCallback:  # type: ignore[no-redef]
        pass

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss at padded tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous steps, cut it off
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class StructuredMetricsCallback(TrainerCallback):
    """Persist trainer metrics as JSONL and CSV for later visualization."""

    def __init__(self, metrics_dir: str | Path):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.metrics_dir / "trainer_log.jsonl"
        self.csv_path = self.metrics_dir / "trainer_log.csv"
        self.rows: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {
            "step": state.global_step,
            "epoch": float(state.epoch or 0),
            **{key: value for key, value in logs.items() if isinstance(value, (int, float, str, bool))},
        }
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fieldnames = sorted({key for item in self.rows for key in item})
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

def load_data_from_csv(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    audio_col = next(c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns)
    text_col = next(c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns)

    audio_base = Path(audio_dir) if audio_dir else Path(csv_path).parent
    df["audio"] = df[audio_col].apply(lambda x: str(audio_base / x) if not Path(x).is_file() else str(x))
    df["sentence"] = df[text_col]

    return Dataset.from_pandas(df[["audio", "sentence"]])

def resolve_parquet_files(patterns: str) -> list[str]:
    """Resolve comma-separated parquet globs into a stable file list."""
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

def normalize_dataset_columns(dataset):
    """Standardize dataset columns to the trainer's expected audio/sentence pair."""
    columns = list(getattr(dataset, "column_names", None) or [])
    if not columns and hasattr(dataset, "features") and dataset.features:
        columns = list(dataset.features.keys())
    if "audio" not in columns:
        raise ValueError(f"Parquet dataset must contain an 'audio' column. Found: {columns}")
    text_col = next((c for c in ["sentence", "transcription", "ground_truth", "text", "transcript"] if c in columns), None)
    if not text_col:
        raise ValueError(f"Parquet dataset must contain a transcript/text column. Found: {columns}")
    if text_col != "sentence":
        dataset = dataset.rename_column(text_col, "sentence")
    # Hugging Face can infer this struct as Audio and auto-decode through
    # torchcodec. Keep decoding disabled so we read embedded WAV bytes with
    # soundfile instead; this is more stable inside Docker and avoids writing
    # permanent extracted WAV files.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset

def decode_audio_for_features(audio: Any) -> dict[str, Any]:
    """Decode path-based or Parquet embedded audio into a 16 kHz mono array."""
    import librosa
    import soundfile as sf

    target_sr = 16000
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        array = audio["array"]
        sr = audio["sampling_rate"]
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    else:
        array, sr = librosa.load(str(audio), sr=None, mono=True)
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return {"array": array, "sampling_rate": sr}

def load_data_from_parquet(parquet_patterns: str, streaming: bool = False):
    """Load Parquet ASR shards with embedded audio bytes, no WAV extraction required."""
    files = resolve_parquet_files(parquet_patterns)
    print(f"Loading {len(files)} parquet file(s): {files[0]}" + (f" ... {files[-1]}" if len(files) > 1 else ""))
    if streaming:
        return files
    dataset = load_dataset("parquet", data_files=files, split="train", streaming=streaming)
    return normalize_dataset_columns(dataset)

def iter_parquet_examples(files: list[str]):
    import pyarrow.parquet as pq

    for parquet_path in files:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                audio = row.get("audio") or {}
                sentence = next((row.get(c) for c in ["sentence", "transcription", "ground_truth", "text", "transcript"] if row.get(c)), None)
                if audio and sentence:
                    yield {"audio": audio, "sentence": str(sentence).strip()}

def inspect_parquet_patterns(name: str, patterns: str) -> None:
    files = resolve_parquet_files(patterns)
    sample = next(iter_parquet_examples(files))
    audio = decode_audio_for_features(sample["audio"])
    print(f"{name}: files={len(files)}, rows=streaming/unknown, columns=['audio', 'sentence']")
    print(f"{name} sample keys={list(sample.keys())}, sentence={sample['sentence'][:120]}")
    print(f"{name} audio sampling_rate={audio.get('sampling_rate')}, array_len={len(audio.get('array', []))}")

class StreamingParquetSpeechDataset(IterableDataset):
    def __init__(self, files: list[str], feature_extractor: Any, tokenizer: Any):
        self.files = files
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __iter__(self):
        for row in iter_parquet_examples(self.files):
            audio = decode_audio_for_features(row["audio"])
            yield {
                "input_features": self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0],
                "labels": self.tokenizer(row["sentence"]).input_ids,
            }

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Bangla/English speech data.")
    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", default="openai/whisper-large-v3-turbo", help="Base model checkpoint.")
    parser.add_argument("--language", default="bengali", help="Language name for Whisper tokenizer ('bengali' or 'english').")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task ('transcribe' or 'translate').")
    
    # Datasets
    parser.add_argument("--train_csv", default="data/train/metadata.csv", help="Training metadata CSV.")
    parser.add_argument("--train_audio", default="data/train/audio", help="Training audio folder.")
    parser.add_argument("--val_csv", default="data/val/metadata.csv", help="Validation metadata CSV.")
    parser.add_argument("--val_audio", default="data/val/audio", help="Validation audio folder.")
    parser.add_argument("--train_parquet", default=None, help="Comma-separated parquet files/globs for training. Uses embedded audio bytes directly.")
    parser.add_argument("--val_parquet", default=None, help="Comma-separated parquet files/globs for validation. Uses embedded audio bytes directly.")
    parser.add_argument("--streaming_parquet", action="store_true", help="Stream parquet shards instead of materializing them. Requires --max_steps > 0 for training.")
    parser.add_argument("--dry_run_data", action="store_true", help="Load and inspect datasets, then exit before model loading/training.")
    parser.add_argument("--output_dir", default="./checkpoints/whisper_bangla_lora", help="Directory to save fine-tuned checkpoints.")
    parser.add_argument("--metrics_dir", default=None, help="Directory for structured JSONL/CSV training metrics. Default: output_dir/metrics.")
    parser.add_argument("--num_proc", type=int, default=2, help="Number of processes for feature extraction.")

    # LoRA / QLoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA parameter-efficient fine-tuning.")
    parser.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA with BitsAndBytes.")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension rank.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", help="Comma-separated target module names.")

    # Batching & Compute
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Per-device eval batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing.")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Use FP16 mixed precision on GPU.")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use BF16 mixed precision on Ampere/Ada GPU.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="DataLoader workers count.")

    # Optimizer & Scheduler
    parser.add_argument("--optim", default="adamw_torch", help="Optimizer ('adamw_torch', 'adamw_bnb_8bit', 'adafactor').")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", default="linear", help="LR scheduler type.")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient clipping norm.")

    # Duration & Checkpointing
    parser.add_argument("--num_epochs", type=int, default=5, help="Total training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides num_epochs.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency in steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency in steps.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging frequency in steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to retain.")
    parser.add_argument("--metric_for_best_model", default="wer", help="Metric for selecting best model ('wer', 'cer', 'loss').")
    parser.add_argument("--report_to", default="tensorboard", help="Dashboard logger ('tensorboard', 'none', 'wandb').")

    # Decoding during Evaluation
    parser.add_argument("--generation_max_length", type=int, default=225, help="Max token length for generation during eval.")
    parser.add_argument("--generation_num_beams", type=int, default=1, help="Beam count during evaluation generation.")

    args = parser.parse_args()
    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    if args.streaming_parquet and args.max_steps <= 0:
        raise ValueError("--streaming_parquet requires --max_steps > 0 because streaming datasets do not expose a fixed length.")

    print("Loading datasets...")
    train_dataset = load_data_from_parquet(args.train_parquet, args.streaming_parquet) if args.train_parquet else load_data_from_csv(args.train_csv, args.train_audio)
    val_dataset = load_data_from_parquet(args.val_parquet, args.streaming_parquet) if args.val_parquet else load_data_from_csv(args.val_csv, args.val_audio)
    if args.dry_run_data:
        if args.streaming_parquet:
            inspect_parquet_patterns("train", args.train_parquet)
            inspect_parquet_patterns("validation", args.val_parquet)
            return
        def describe(name, dataset):
            columns = list(getattr(dataset, "column_names", None) or getattr(dataset, "features", {}).keys())
            length = "streaming/unknown"
            try:
                length = len(dataset)
            except Exception:
                pass
            print(f"{name}: rows={length}, columns={columns}")
            sample = next(iter(dataset))
            print(f"{name} sample keys={list(sample.keys())}, sentence={str(sample.get('sentence', ''))[:120]}")
            audio = decode_audio_for_features(sample.get("audio"))
            print(f"{name} audio sampling_rate={audio.get('sampling_rate')}, array_len={len(audio.get('array', []))}")
        describe("train", train_dataset)
        describe("validation", val_dataset)
        return

    print(f"Loading processor and model: {args.model_name_or_path}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)

    if args.streaming_parquet:
        train_dataset = StreamingParquetSpeechDataset(train_dataset, feature_extractor, tokenizer)
        val_dataset = StreamingParquetSpeechDataset(val_dataset, feature_extractor, tokenizer)

    is_cuda = torch.cuda.is_available()
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if args.use_qlora and not is_cuda:
        print("⚠️ Warning: QLoRA (4-bit bitsandbytes) requires an NVIDIA CUDA GPU. Falling back to standard LoRA on current platform.")
        args.use_qlora = False
        args.use_lora = True

    load_kwargs = {}
    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
            )
            load_kwargs["device_map"] = "auto"
        except ImportError:
            print("Warning: bitsandbytes not found, falling back to standard loading.")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, **load_kwargs)

    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    model.generate = torch.no_grad()(model.generate)

    if args.use_lora or args.use_qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        print(f"Applying LoRA configuration (r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules})...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    def prepare_dataset(batch):
        audio = decode_audio_for_features(batch["audio"])
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    if not args.streaming_parquet:
        print("Preprocessing datasets (extracting Mel features & tokenizing)...")
        train_columns = list(getattr(train_dataset, "column_names", None) or getattr(train_dataset, "features", {}).keys())
        val_columns = list(getattr(val_dataset, "column_names", None) or getattr(val_dataset, "features", {}).keys())
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_columns, num_proc=args.num_proc)
        val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_columns, num_proc=args.num_proc)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        exact_match = sum(1 for pred_text, ref_text in zip(pred_str, label_str) if pred_text.strip() == ref_text.strip()) / max(len(label_str), 1)
        char_accuracy = max(0.0, 1.0 - (cer / 100.0))
        return {"wer": wer, "cer": cer, "exact_match": exact_match, "char_accuracy": char_accuracy}

    import inspect
    eval_strat_key = "eval_strategy" if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters else "evaluation_strategy"
    eval_kwargs = {eval_strat_key: "steps"}

    use_fp16 = args.fp16 and is_cuda
    use_bf16 = args.bf16 and (is_cuda or is_mps)

    if args.fp16 and not is_cuda:
        print("⚠️ Notice: FP16 is only supported on NVIDIA CUDA GPUs. Falling back to FP32.")

    if is_mps:
        print("ℹ️ Device: Apple Silicon Metal Performance Shaders (MPS) active.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=False,
        report_to=[args.report_to] if args.report_to != "none" else [],
        **eval_kwargs
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[StructuredMetricsCallback(args.metrics_dir or Path(args.output_dir) / "metrics")],
    )

    accelerator = "CUDA GPU" if is_cuda else ("Apple Silicon MPS" if is_mps else "CPU")
    print(f"\nStarting Training on {accelerator}...")
    trainer.train()

    print(f"\nTraining complete. Saving best model checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
