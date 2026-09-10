import sys
from pathlib import Path
from typing import Any

from backend.config import ROOT_DIR, SCRIPTS_DIR


def build_training_args(config: dict[str, Any]) -> list[str]:
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
        str(int(value("batch_size", 8))),
        "--eval_batch_size",
        str(int(value("eval_batch_size", 8))),
        "--gradient_accumulation_steps",
        str(int(value("gradient_accumulation_steps", 2))),
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
        str(int(value("num_epochs", 5))),
        "--max_steps",
        str(int(value("max_steps", -1))),
        "--eval_steps",
        str(int(value("eval_steps", 200))),
        "--save_steps",
        str(int(value("save_steps", 200))),
        "--logging_steps",
        str(int(value("logging_steps", 25))),
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
        str(int(value("dataloader_num_workers", 2))),
    ]

    mode = str(value("finetune_mode", "lora")).lower()
    if "qlora" in mode or config.get("use_qlora"):
        args.append("--use_qlora")
    elif "lora" in mode or config.get("use_lora", True):
        args.append("--use_lora")

    if "--use_lora" in args or "--use_qlora" in args:
        args.extend(
            [
                "--lora_r",
                str(int(value("lora_r", 32))),
                "--lora_alpha",
                str(int(value("lora_alpha", 64))),
                "--lora_dropout",
                str(float(value("lora_dropout", 0.05))),
                "--lora_target_modules",
                str(value("lora_target_modules", "q_proj,v_proj")),
            ]
        )

    precision = str(value("precision", "fp16")).lower()
    if "bf16" in precision:
        args.append("--bf16")
    elif "fp16" in precision:
        args.append("--fp16")

    if config.get("gradient_checkpointing", True):
        args.append("--gradient_checkpointing")

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

