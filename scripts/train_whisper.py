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
    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", default="openai/whisper-large-v3-turbo", help="Base model checkpoint.")
    parser.add_argument("--language", default="bengali", help="Language name for Whisper tokenizer ('bengali' or 'english').")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task ('transcribe' or 'translate').")
    
    # Datasets
    parser.add_argument("--train_csv", default="data/train/metadata.csv", help="Training metadata CSV.")
    parser.add_argument("--train_audio", default="data/train/audio", help="Training audio folder.")
    parser.add_argument("--val_csv", default="data/val/metadata.csv", help="Validation metadata CSV.")
    parser.add_argument("--val_audio", default="data/val/audio", help="Validation audio folder.")
    parser.add_argument("--output_dir", default="./checkpoints/whisper_bangla_lora", help="Directory to save fine-tuned checkpoints.")
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

    print(f"Loading processor and model: {args.model_name_or_path}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)

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

    print("Loading datasets...")
    train_dataset = load_data_from_csv(args.train_csv, args.train_audio)
    val_dataset = load_data_from_csv(args.val_csv, args.val_audio)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Preprocessing datasets (extracting Mel features & tokenizing)...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=args.num_proc)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=args.num_proc)

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
        fp16=args.fp16,
        bf16=args.bf16,
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
    )

    print("\nStarting Training on GPU...")
    trainer.train()

    print(f"\nTraining complete. Saving best model checkpoint to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
