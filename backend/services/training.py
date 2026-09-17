import sys
from pathlib import Path
from typing import Any

from backend.config import ROOT_DIR, SCRIPTS_DIR


def build_raw_training_args(config: dict[str, Any]) -> list[str]:
    def value(name: str, default: Any) -> Any:
        current = config.get(name, default)
        return default if current in (None, "") else current

    args = [
        sys.executable,
        str(SCRIPTS_DIR / "train_whisper.py"),
        "--model_name_or_path",
        str(value("model_name_or_path", "openai/whisper-large-v3-turbo")),
        "--language",
        str(value("language", "bengali")),
        "--task",
        str(value("task", "transcribe")),
        "--train_csv",
        str(value("train_csv", "data/train/metadata.csv")),
        "--train_audio",
        str(value("train_audio", "data/train/audio")),
        "--val_csv",
        str(value("val_csv", "data/val/metadata.csv")),
        "--val_audio",
        str(value("val_audio", "data/val/audio")),
        "--output_dir",
        str(value("output_dir", "./checkpoints/whisper_bangla_lora")),
        "--num_proc",
        str(int(value("num_proc", 2))),
        "--batch_size",
        str(int(value("batch_size", 1))),
        "--eval_batch_size",
        str(int(value("eval_batch_size", 1))),
        "--gradient_accumulation_steps",
        str(int(value("gradient_accumulation_steps", 8))),
        "--learning_rate",
        str(value("learning_rate", "1e-4")),
        "--optim",
        str(value("optim", "adamw_torch")),
        "--lr_scheduler_type",
        str(value("lr_scheduler_type", "linear")),
        "--warmup_steps",
        str(int(value("warmup_steps", 50))),
        "--weight_decay",
        str(float(value("weight_decay", 0.01))),
        "--max_grad_norm",
        str(float(value("max_grad_norm", 1.0))),
        "--num_epochs",
        str(int(value("num_epochs", 1))),
        "--max_steps",
        str(int(value("max_steps", 10))),
        "--eval_steps",
        str(int(value("eval_steps", 10))),
        "--save_steps",
        str(int(value("save_steps", 10))),
        "--logging_steps",
        str(int(value("logging_steps", 1))),
        "--save_total_limit",
        str(int(value("save_total_limit", 2))),
        "--metric_for_best_model",
        str(value("metric_for_best_model", "wer")),
        "--report_to",
        str(value("report_to", "tensorboard")),
        "--generation_max_length",
        str(int(value("generation_max_length", 225))),
        "--generation_num_beams",
        str(int(value("generation_num_beams", 1))),
        "--dataloader_num_workers",
        str(int(value("dataloader_num_workers", 0))),
    ]

    train_parquet = str(value("train_parquet", "")).strip()
    if train_parquet:
        args.extend(["--train_parquet", train_parquet])

    val_parquet = str(value("val_parquet", "")).strip()
    if val_parquet:
        args.extend(["--val_parquet", val_parquet])

    if config.get("streaming_parquet", False):
        args.append("--streaming_parquet")

    if config.get("dry_run_data", False):
        args.append("--dry_run_data")

    mode = str(value("finetune_mode", "lora")).lower()
    if "qlora" in mode or config.get("use_qlora"):
        args.append("--use_qlora")
    elif "lora" in mode or config.get("use_lora", True):
        args.append("--use_lora")

    if "--use_lora" in args or "--use_qlora" in args:
        args.extend(
            [
                "--lora_r",
                str(int(value("lora_r", 16))),
                "--lora_alpha",
                str(int(value("lora_alpha", 32))),
                "--lora_dropout",
                str(float(value("lora_dropout", 0.05))),
                "--lora_target_modules",
                str(value("lora_target_modules", "q_proj,v_proj")),
            ]
        )

    precision = str(value("precision", "fp32")).lower()
    if "bf16" in precision:
        args.append("--bf16")
    elif "fp16" in precision:
        args.append("--fp16")

    if config.get("gradient_checkpointing", True):
        args.append("--gradient_checkpointing")

    return args


def build_training_args(config: dict[str, Any]) -> list[str]:
    raw_args = build_raw_training_args(config)
    if config.get("dry_run_data", False) or not config.get("guarded_training", True):
        return raw_args

    def value(name: str, default: Any) -> Any:
        current = config.get(name, default)
        return default if current in (None, "") else current

    args = [
        sys.executable,
        str(SCRIPTS_DIR / "guarded_train.py"),
        "--backup_source",
        str(value("backup_source", "models")),
        "--backup_dest",
        str(value("backup_dest", "backups/models")),
        "--backup_name",
        str(value("backup_name", "current")),
        "--test_parquet",
        str(value("test_parquet", "data/sources/subakko/hf/Data/test-*.parquet")),
        "--baseline_output",
        str(value("baseline_output", "checkpoints/baseline-old.csv")),
        "--after_output",
        str(value("after_output", "checkpoints/after-new.csv")),
        "--comparison_output",
        str(value("comparison_output", "checkpoints/model_comparison.json")),
        "--eval_model_before",
        str(value("eval_model_before", "openai/whisper-large-v3-turbo")),
        "--eval_engine_before",
        str(value("eval_engine_before", "transformers")),
        "--eval_engine_after",
        str(value("eval_engine_after", "transformers")),
        "--models_dir",
        str(value("models_dir", "models")),
    ]
    eval_model_after = str(value("eval_model_after", "")).strip()
    if eval_model_after:
        args.extend(["--eval_model_after", eval_model_after])
    eval_max_samples = value("eval_max_samples", 20)
    if eval_max_samples not in (None, ""):
        args.extend(["--eval_max_samples", str(int(eval_max_samples))])
    args.append("--")
    args.extend(raw_args)
    return args


def relative_command(args: list[str]) -> str:
    rendered = []
    for item in args:
        try:
            path = Path(item)
            if path.is_absolute() and path.is_relative_to(ROOT_DIR):
                item = str(path.relative_to(ROOT_DIR))
        except Exception:
            pass
        rendered.append(f'"{item}"' if " " in item else item)
    return " ".join(rendered)

