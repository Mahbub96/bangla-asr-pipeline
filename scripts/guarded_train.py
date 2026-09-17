#!/usr/bin/env python3
"""Run ASR training with a single backup and before/after evaluation.

The guarded flow is intentionally conservative:
1. keep exactly one backup under the configured backup root,
2. evaluate the current model on a fixed test set,
3. run the training command,
4. evaluate the trained output on the same fixed test set,
5. write a comparison report for manual model selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from model_guard import attach_evaluation_metrics, compare, copy_backup, metrics


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def default_after_model(train_command: list[str], fallback: str) -> str:
    for index, item in enumerate(train_command):
        if item == "--output_dir" and index + 1 < len(train_command):
            return train_command[index + 1]
    return fallback


def build_eval_command(
    *,
    parquet: str,
    model: str,
    output: Path,
    language: str,
    engine: str,
    device: str | None,
    compute_type: str | None,
    models_dir: str,
    max_samples: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--parquet",
        parquet,
        "--model",
        model,
        "--language",
        language,
        "--models_dir",
        models_dir,
        "--output",
        str(output),
        "--engine",
        engine,
    ]
    if device:
        command.extend(["--device", device])
    if compute_type:
        command.extend(["--compute_type", compute_type])
    if max_samples:
        command.extend(["--max_samples", str(max_samples)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a Whisper training run with backup and WER/CER comparison.")
    parser.add_argument("--backup_source", default="models", help="Current model/checkpoint directory to back up before training.")
    parser.add_argument("--backup_dest", default="backups/models", help="Backup root. Existing backups inside it are removed.")
    parser.add_argument("--backup_name", default="current", help="Single backup folder name under --backup_dest.")
    parser.add_argument("--test_parquet", default="data/sources/subakko/hf/Data/test-*.parquet", help="Fixed held-out Parquet test set for before/after scoring.")
    parser.add_argument("--baseline_output", default="checkpoints/baseline-old.csv", help="Old-model evaluation CSV.")
    parser.add_argument("--after_output", default="checkpoints/after-new.csv", help="New-model evaluation CSV.")
    parser.add_argument("--comparison_output", default="checkpoints/model_comparison.json", help="Old-vs-new comparison JSON path.")
    parser.add_argument("--eval_model_before", default="large-v3-turbo", help="Model name/path used for baseline scoring.")
    parser.add_argument("--eval_model_after", default=None, help="Model name/path used after training. Defaults to train --output_dir.")
    parser.add_argument("--eval_language", default="bn", choices=["bn", "en", "auto"], help="Evaluation language code.")
    parser.add_argument("--eval_engine_before", default="faster-whisper", choices=["faster-whisper", "transformers"], help="Evaluation engine for baseline model.")
    parser.add_argument("--eval_engine_after", default="transformers", choices=["faster-whisper", "transformers"], help="Evaluation engine for trained model/checkpoint.")
    parser.add_argument("--eval_device", default=None, help="Evaluation device, e.g. cpu or cuda.")
    parser.add_argument("--eval_compute_type", default=None, help="faster-whisper compute type, e.g. int8/float16.")
    parser.add_argument("--models_dir", default="models", help="Model cache directory for evaluation.")
    parser.add_argument("--eval_max_samples", type=int, default=None, help="Optional fixed sample cap for quicker before/after scoring.")
    parser.add_argument("train_command", nargs=argparse.REMAINDER, help="Training command after --, usually: -- python scripts/train_whisper.py ...")

    args = parser.parse_args()
    train_command = list(args.train_command)
    if train_command and train_command[0] == "--":
        train_command = train_command[1:]
    if not train_command:
        parser.error("training command is required after --")

    backup_dir = copy_backup(
        Path(args.backup_source),
        Path(args.backup_dest),
        args.backup_name,
        hash_files=False,
        single=True,
    )
    print(f"Single backup ready: {backup_dir.resolve()}", flush=True)

    baseline_output = Path(args.baseline_output)
    after_output = Path(args.after_output)
    comparison_output = Path(args.comparison_output)

    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    after_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    print("Evaluating current model before training...", flush=True)
    run_command(
        build_eval_command(
            parquet=args.test_parquet,
            model=args.eval_model_before,
            output=baseline_output,
            language=args.eval_language,
            engine=args.eval_engine_before,
            device=args.eval_device,
            compute_type=args.eval_compute_type,
            models_dir=args.models_dir,
            max_samples=args.eval_max_samples,
        )
    )
    baseline_metrics = metrics(baseline_output)
    attach_evaluation_metrics(backup_dir, baseline_metrics, baseline_output)
    print("Baseline score saved into backup_manifest.json:", flush=True)
    print(json.dumps(baseline_metrics, ensure_ascii=False, indent=2), flush=True)

    print("Starting training...", flush=True)
    run_command(train_command)

    after_model = args.eval_model_after or default_after_model(train_command, "checkpoints/whisper_bangla_lora")
    print("Evaluating trained model on the same test set...", flush=True)
    run_command(
        build_eval_command(
            parquet=args.test_parquet,
            model=after_model,
            output=after_output,
            language=args.eval_language,
            engine=args.eval_engine_after,
            device=args.eval_device,
            compute_type=args.eval_compute_type,
            models_dir=args.models_dir,
            max_samples=args.eval_max_samples,
        )
    )

    report = compare(baseline_output, after_output, comparison_output)
    print("Manual decision report ready.", flush=True)
    print(f"Verdict: {report['verdict']}", flush=True)
    print(f"Comparison JSON: {comparison_output.resolve()}", flush=True)
    print(f"Comparison Markdown: {comparison_output.with_suffix('.md').resolve()}", flush=True)
    print("Do not replace the current model automatically; review WER/CER and sample outputs first.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
