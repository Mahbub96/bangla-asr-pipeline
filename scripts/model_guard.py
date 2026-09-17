#!/usr/bin/env python3
"""Backup and compare ASR models around a training run.

This script keeps training controlled: snapshot the current model/checkpoint,
store a baseline evaluation result, then compare the trained model against the
same evaluation set after training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(path: Path, *, hash_files: bool = False) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = file_path.stat()
        total_bytes += stat.st_size
        item: dict[str, Any] = {
            "path": str(file_path.relative_to(path)),
            "bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
        if hash_files:
            item["sha256"] = sha256_file(file_path)
        files.append(item)
    return {
        "source": str(path.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def copy_backup(source: Path, destination_root: Path, name: str | None, hash_files: bool) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source model/checkpoint does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source}")

    backup_name = name or f"{source.name}-{utc_stamp()}"
    backup_dir = destination_root / backup_name
    if backup_dir.exists():
        raise FileExistsError(f"Backup destination already exists: {backup_dir}")

    destination_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, backup_dir)

    manifest = tree_manifest(backup_dir, hash_files=hash_files)
    manifest.update(
        {
            "created_at_utc": utc_stamp(),
            "original_source": str(source),
            "backup_dir": str(backup_dir.resolve()),
            "hash_files": hash_files,
        }
    )
    (backup_dir / "backup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return backup_dir


def read_eval_csv(path: Path) -> tuple[list[str], list[str], list[float], list[float]]:
    refs: list[str] = []
    hyps: list[str] = []
    wers: list[float] = []
    cers: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"ground_truth", "prediction", "wer", "cer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            refs.append((row.get("ground_truth") or "").strip())
            hyps.append((row.get("prediction") or "").strip())
            wers.append(float(row.get("wer") or 0.0))
            cers.append(float(row.get("cer") or 0.0))
    if not refs:
        raise ValueError(f"No evaluated rows found in {path}")
    return refs, hyps, wers, cers


def metrics(path: Path) -> dict[str, Any]:
    refs, hyps, wers, cers = read_eval_csv(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "samples": len(refs),
        "mean_sample_wer": sum(wers) / len(wers),
        "mean_sample_cer": sum(cers) / len(cers),
    }
    try:
        import jiwer

        result["corpus_wer"] = float(jiwer.wer(refs, hyps))
        result["corpus_cer"] = float(jiwer.cer(refs, hyps))
    except Exception as exc:  # keep comparison useful even without jiwer
        result["corpus_metric_error"] = str(exc)
    return result


def compare(old_csv: Path, new_csv: Path, output: Path) -> dict[str, Any]:
    old = metrics(old_csv)
    new = metrics(new_csv)
    if old["samples"] != new["samples"]:
        raise ValueError(
            f"Evaluation sample count mismatch: old={old['samples']} new={new['samples']}. "
            "Use the exact same test CSV/audio set for both models."
        )

    old_wer = old.get("corpus_wer", old["mean_sample_wer"])
    new_wer = new.get("corpus_wer", new["mean_sample_wer"])
    old_cer = old.get("corpus_cer", old["mean_sample_cer"])
    new_cer = new.get("corpus_cer", new["mean_sample_cer"])

    wer_delta = new_wer - old_wer
    cer_delta = new_cer - old_cer
    if new_wer <= old_wer and new_cer <= old_cer and (new_wer < old_wer or new_cer < old_cer):
        verdict = "new_better"
    elif new_wer >= old_wer and new_cer >= old_cer and (new_wer > old_wer or new_cer > old_cer):
        verdict = "old_better"
    else:
        verdict = "gray_area_mixed_metrics"

    report = {
        "created_at_utc": utc_stamp(),
        "old": old,
        "new": new,
        "delta": {
            "wer": wer_delta,
            "cer": cer_delta,
            "wer_percent_points": wer_delta * 100,
            "cer_percent_points": cer_delta * 100,
        },
        "verdict": verdict,
        "rule": "Lower WER and CER are better. If one improves and the other regresses, keep it in gray_area_mixed_metrics for human review.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = output.with_suffix(".md")
    md.write_text(render_markdown(report), encoding="utf-8")
    return report


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    old = report["old"]
    new = report["new"]
    delta = report["delta"]
    old_wer = old.get("corpus_wer", old["mean_sample_wer"])
    new_wer = new.get("corpus_wer", new["mean_sample_wer"])
    old_cer = old.get("corpus_cer", old["mean_sample_cer"])
    new_cer = new.get("corpus_cer", new["mean_sample_cer"])
    return "\n".join(
        [
            "# Model Comparison Report",
            "",
            f"Verdict: **{report['verdict']}**",
            "",
            "| Metric | Old model | New model | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| WER | {pct(old_wer)} | {pct(new_wer)} | {delta['wer_percent_points']:+.2f} pp |",
            f"| CER | {pct(old_cer)} | {pct(new_cer)} | {delta['cer_percent_points']:+.2f} pp |",
            f"| Samples | {old['samples']} | {new['samples']} | 0 |",
            "",
            report["rule"],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup and compare ASR model evaluations.")
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup", help="Copy a model/checkpoint directory and write a manifest.")
    backup.add_argument("--source", required=True, type=Path, help="Model/checkpoint directory to backup.")
    backup.add_argument("--dest", default=Path("backups/models"), type=Path, help="Backup root directory.")
    backup.add_argument("--name", default=None, help="Backup folder name. Default: source-timestamp.")
    backup.add_argument("--hash", action="store_true", help="Also compute sha256 per file; slower for large models.")

    cmp_parser = sub.add_parser("compare", help="Compare old/new evaluation CSVs from scripts/evaluate.py.")
    cmp_parser.add_argument("--old", required=True, type=Path, help="Baseline/old evaluation CSV.")
    cmp_parser.add_argument("--new", required=True, type=Path, help="New trained model evaluation CSV.")
    cmp_parser.add_argument("--output", default=Path("reports/model_comparison.json"), type=Path)

    args = parser.parse_args()
    if args.command == "backup":
        backup_dir = copy_backup(args.source, args.dest, args.name, args.hash)
        print(f"Backup created: {backup_dir.resolve()}")
        print(f"Manifest: {(backup_dir / 'backup_manifest.json').resolve()}")
        return 0
    if args.command == "compare":
        report = compare(args.old, args.new, args.output)
        print(f"Verdict: {report['verdict']}")
        print(f"JSON report: {args.output.resolve()}")
        print(f"Markdown report: {args.output.with_suffix('.md').resolve()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
