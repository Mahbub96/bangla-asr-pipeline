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
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def is_macos_sidecar(path: Path) -> bool:
    """Return True for AppleDouble/resource-fork files created on macOS volumes."""
    return any(part.startswith("._") for part in path.parts)


def remove_path_tolerant(path: Path) -> None:
    """Remove a file/tree while tolerating disappearing macOS sidecars.

    AppleDouble files on external macOS volumes can vanish between directory
    listing and unlink during shutil.rmtree. Missing files are harmless for our
    single-backup replacement; real permission errors still surface.
    """

    def onerror(function: Any, item: str, exc_info: tuple[type[BaseException], BaseException, Any]) -> None:
        if isinstance(exc_info[1], FileNotFoundError):
            return
        raise exc_info[1]

    if path.is_dir():
        shutil.rmtree(path, onerror=onerror)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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
    for file_path in sorted(path.rglob("*")):
        relative_path = file_path.relative_to(path)
        if is_macos_sidecar(relative_path):
            continue
        try:
            if not file_path.is_file():
                continue
            stat = file_path.stat()
        except OSError:
            # External macOS volumes can expose unreadable AppleDouble xattr
            # sidecars. They are not real model data, so do not let them abort
            # backup manifest creation.
            if is_macos_sidecar(relative_path):
                continue
            raise
        total_bytes += stat.st_size
        item: dict[str, Any] = {
            "path": str(relative_path),
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


def copy_backup(source: Path, destination_root: Path, name: str | None, hash_files: bool, single: bool = False) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source model/checkpoint does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source}")

    backup_name = name or f"{source.name}-{utc_stamp()}"
    backup_dir = destination_root / backup_name

    destination_root.mkdir(parents=True, exist_ok=True)
    if single:
        for existing in destination_root.iterdir():
            remove_path_tolerant(existing)
    elif backup_dir.exists():
        raise FileExistsError(f"Backup destination already exists: {backup_dir}")
    shutil.copytree(
        source,
        backup_dir,
        ignore=lambda _directory, names: [name for name in names if name.startswith("._")],
    )

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


def attach_evaluation_metrics(backup_dir: Path, evaluation: dict[str, Any], evaluation_csv: Path) -> None:
    """Attach baseline WER/CER metrics to a backup manifest.

    The backup is useful only if we know how the backed-up model performed on
    the fixed test set. Keep that score next to the one retained backup so a
    later manual decision has the exact baseline evidence.
    """
    manifest_path = backup_dir / "backup_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Backup manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_evaluation"] = {
        "created_at_utc": utc_stamp(),
        "csv": str(evaluation_csv.resolve()),
        "metrics": evaluation,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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


def confusion_matrix(path: Path, *, max_items: int = 40) -> dict[str, Any]:
    refs, hyps, _, _ = read_eval_csv(path)
    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    exact = 0
    total_ref_tokens = 0
    total_pred_tokens = 0

    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_ref_tokens += len(ref_tokens)
        total_pred_tokens += len(hyp_tokens)
        matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                exact += i2 - i1
            elif tag == "replace":
                paired = min(i2 - i1, j2 - j1)
                for offset in range(paired):
                    substitutions[(ref_tokens[i1 + offset], hyp_tokens[j1 + offset])] += 1
                for token in ref_tokens[i1 + paired : i2]:
                    deletions[token] += 1
                for token in hyp_tokens[j1 + paired : j2]:
                    insertions[token] += 1
            elif tag == "delete":
                for token in ref_tokens[i1:i2]:
                    deletions[token] += 1
            elif tag == "insert":
                for token in hyp_tokens[j1:j2]:
                    insertions[token] += 1

    top_substitutions = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in substitutions.most_common(max_items)
    ]
    top_deletions = [{"expected": token, "count": count} for token, count in deletions.most_common(max_items)]
    top_insertions = [{"predicted": token, "count": count} for token, count in insertions.most_common(max_items)]
    return {
        "token_totals": {
            "reference": total_ref_tokens,
            "prediction": total_pred_tokens,
            "exact": exact,
            "substitutions": sum(substitutions.values()),
            "deletions": sum(deletions.values()),
            "insertions": sum(insertions.values()),
        },
        "top_substitutions": top_substitutions,
        "top_deletions": top_deletions,
        "top_insertions": top_insertions,
    }


def write_confusion_csv(matrix: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["type", "expected", "predicted", "count"])
        writer.writeheader()
        for row in matrix["top_substitutions"]:
            writer.writerow({"type": "substitution", **row})
        for row in matrix["top_deletions"]:
            writer.writerow({"type": "deletion", "expected": row["expected"], "predicted": "", "count": row["count"]})
        for row in matrix["top_insertions"]:
            writer.writerow({"type": "insertion", "expected": "", "predicted": row["predicted"], "count": row["count"]})


def write_analysis_artifacts(report: dict[str, Any], old_csv: Path, new_csv: Path, output: Path) -> dict[str, Any]:
    old_matrix = confusion_matrix(old_csv)
    new_matrix = confusion_matrix(new_csv)
    analysis = {
        "old_confusion_matrix": old_matrix,
        "new_confusion_matrix": new_matrix,
        "charts": {
            "wer_cer_bar": [
                {"model": "old", "wer": report["old"].get("corpus_wer", report["old"]["mean_sample_wer"]), "cer": report["old"].get("corpus_cer", report["old"]["mean_sample_cer"])},
                {"model": "new", "wer": report["new"].get("corpus_wer", report["new"]["mean_sample_wer"]), "cer": report["new"].get("corpus_cer", report["new"]["mean_sample_cer"])},
            ],
            "old_error_pie": old_matrix["token_totals"],
            "new_error_pie": new_matrix["token_totals"],
        },
    }
    analysis_path = output.with_name(f"{output.stem}_analysis.json")
    chart_path = output.with_name(f"{output.stem}_chart_data.json")
    old_confusion_path = output.with_name(f"{output.stem}_old_confusion.csv")
    new_confusion_path = output.with_name(f"{output.stem}_new_confusion.csv")
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_path.write_text(json.dumps(analysis["charts"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_confusion_csv(old_matrix, old_confusion_path)
    write_confusion_csv(new_matrix, new_confusion_path)
    return {
        "analysis_json": str(analysis_path.resolve()),
        "chart_data_json": str(chart_path.resolve()),
        "old_confusion_csv": str(old_confusion_path.resolve()),
        "new_confusion_csv": str(new_confusion_path.resolve()),
        "analysis": analysis,
    }


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
    report.update(write_analysis_artifacts(report, old_csv, new_csv, output))

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
    backup.add_argument("--single", action="store_true", help="Keep only this backup under --dest by deleting older backups first.")
    backup.add_argument("--metrics", default=None, type=Path, help="Optional evaluation CSV to store WER/CER in backup_manifest.json.")

    cmp_parser = sub.add_parser("compare", help="Compare old/new evaluation CSVs from scripts/evaluate.py.")
    cmp_parser.add_argument("--old", required=True, type=Path, help="Baseline/old evaluation CSV.")
    cmp_parser.add_argument("--new", required=True, type=Path, help="New trained model evaluation CSV.")
    cmp_parser.add_argument("--output", default=Path("reports/model_comparison.json"), type=Path)

    args = parser.parse_args()
    if args.command == "backup":
        backup_dir = copy_backup(args.source, args.dest, args.name, args.hash, args.single)
        if args.metrics:
            attach_evaluation_metrics(backup_dir, metrics(args.metrics), args.metrics)
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
