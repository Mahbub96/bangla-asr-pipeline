#!/usr/bin/env python3
"""
Whisper Fine-Tuning Script for Bangla/English ASR.
Designed for execution on a large compute machine (GPU / Multi-GPU).
Supports full fine-tuning or parameter-efficient LoRA / QLoRA.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path
import pandas as pd
import torch

try:
    import evaluate
    from datasets import Dataset, Audio
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError:
    # Notice for local execution
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

def load_data_from_csv(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    audio_col = next(c for c in ["audio_path", "audio", "file_name", "path"] if c in df.columns)
    text_col = next(c for c in ["sentence", "transcription", "ground_truth", "text"] if c in df.columns)

    audio_base = Path(audio_dir) if audio_dir else Path(csv_path).parent
    df["audio"] = df[audio_col].apply(lambda x: str(audio_base / x) if not Path(x).is_file() else str(x))
    df["sentence"] = df[text_col]

    dataset = Dataset.from_pandas(df[["audio", "sentence"]])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Bangla/English speech data.")
    parser.add_argument("--model_name_or_path", default="openai/whisper-large-v3-turbo", help="Base model checkpoint.")
    parser.add_argument("--train_csv", default="data/train/metadata.csv", help="Training metadata CSV.")
    parser.add_argument("--train_audio", default="data/train/audio", help="Training audio folder.")
    parser.add_argument("--val_csv", default="data/val/metadata.csv", help="Validation metadata CSV.")
    parser.add_argument("--val_audio", default="data/val/audio", help="Validation audio folder.")
    parser.add_argument("--output_dir", default="./checkpoints/whisper_bangla_lora", help="Directory to save fine-tuned checkpoints.")
    parser.add_argument("--language", default="bengali", help="Language name for Whisper tokenizer ('bengali' or 'english').")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA parameter-efficient fine-tuning (recommended for single GPU).")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Total training epochs.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation frequency in steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Checkpoint save frequency in steps.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging frequency in steps.")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Use FP16 mixed precision on GPU.")

    args = parser.parse_args()

    print(f"Loading processor and model: {args.model_name_or_path}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    model.generate = torch.no_grad()(model.generate)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        print("Applying LoRA configuration to Whisper...")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = load_data_from_csv(args.train_csv, args.train_audio)
    val_dataset = load_data_from_csv(args.val_csv, args.val_audio)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Preprocessing datasets (extracting Mel features & tokenizing)...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=2)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=2)

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
        return {"wer": wer, "cer": cer}

    import inspect
    eval_strat_key = "eval_strategy" if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters else "evaluation_strategy"
    eval_kwargs = {eval_strat_key: "steps"}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        fp16=args.fp16,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
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
    )

    print("\nStarting Training on GPU...")
    trainer.train()

    print(f"\nTraining complete. Saving best model checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
