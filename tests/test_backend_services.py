from pathlib import Path
import csv
import json

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import EvaluateRequest
from backend.services.exports import generate_srt, generate_vtt
from backend.services.jobs import Job
from backend.services.quality import postprocess_result, resolve_profile, resolve_transcription_quality, transliterate_bangla_to_banglish
from backend.services.training import build_training_args, relative_command
from scripts.model_guard import attach_evaluation_metrics, compare, copy_backup, metrics


def test_subtitle_exports_include_timestamps_and_text():
    segments = [{"start": 1.25, "end": 3.5, "text": "আমি বাংলায় কথা বলি।"}]

    assert "00:00:01,250 --> 00:00:03,500" in generate_srt(segments)
    assert "00:00:01.250 --> 00:00:03.500" in generate_vtt(segments)
    assert "আমি বাংলায় কথা বলি।" in generate_srt(segments)


def test_job_snapshot_exposes_export_links():
    job = Job(id="abc", kind="batch", status="completed", exports={"csv": Path("/tmp/out.csv")})

    snapshot = job.snapshot()

    assert snapshot["exports"]["csv"] == "/api/exports/abc/csv"
    assert snapshot["status"] == "completed"


def test_runtime_logs_endpoint_exposes_backend_logs():
    client = TestClient(app)

    response = client.get("/api/logs/runtime?limit=5")

    assert response.status_code == 200
    assert "logs" in response.json()


def test_active_job_logs_endpoint_exposes_active_jobs():
    client = TestClient(app)

    response = client.get("/api/logs/jobs/active")

    assert response.status_code == 200
    assert "jobs" in response.json()


def test_training_command_uses_lora_defaults():
    args = build_training_args({})
    command = relative_command(args)

    assert "scripts/guarded_train.py" in command
    assert "scripts/train_whisper.py" in command
    assert "--test_parquet data/sources/subakko/hf/Data/test-*.parquet" in command
    assert "--comparison_output checkpoints/model_comparison.json" in command
    assert "--use_lora" in args
    assert "--lora_r" in args
    assert "--max_steps 2000" in command
    assert "--model_name_or_path openai/whisper-large-v3-turbo" in command


def test_dry_run_training_command_skips_guard():
    args = build_training_args({"dry_run_data": True})
    command = relative_command(args)

    assert "scripts/guarded_train.py" not in command
    assert "scripts/train_whisper.py" in command
    assert "--dry_run_data" in args


def test_single_backup_replaces_old_backup_and_stores_score(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"old-model")
    backup_root = tmp_path / "backups"
    stale = backup_root / "stale"
    stale.mkdir(parents=True)
    (stale / "old.txt").write_text("remove me", encoding="utf-8")

    backup_dir = copy_backup(source, backup_root, "current", hash_files=False, single=True)

    assert backup_dir == backup_root / "current"
    assert not stale.exists()
    assert (backup_dir / "weights.bin").is_file()

    eval_csv = tmp_path / "baseline.csv"
    with eval_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ground_truth", "prediction", "wer", "cer"])
        writer.writeheader()
        writer.writerow({"ground_truth": "আমি বাংলা বলি", "prediction": "আমি বাংলা বলি", "wer": "0", "cer": "0"})
    baseline = metrics(eval_csv)
    attach_evaluation_metrics(backup_dir, baseline, eval_csv)

    manifest = json.loads((backup_dir / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline_evaluation"]["metrics"]["samples"] == 1
    assert manifest["baseline_evaluation"]["metrics"]["mean_sample_wer"] == 0


def test_model_comparison_writes_analysis_and_confusion_artifacts(tmp_path):
    old_csv = tmp_path / "old.csv"
    new_csv = tmp_path / "new.csv"
    rows = [
        {"ground_truth": "আমি বাংলা বলি", "prediction": "আমি বাংলা বলি", "wer": "0", "cer": "0"},
        {"ground_truth": "সে ভাত খায়", "prediction": "সে গান খায়", "wer": "0.3333", "cer": "0.25"},
    ]
    for path in [old_csv, new_csv]:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["ground_truth", "prediction", "wer", "cer"])
            writer.writeheader()
            writer.writerows(rows)

    report = compare(old_csv, new_csv, tmp_path / "comparison.json")

    assert Path(report["analysis_json"]).is_file()
    assert Path(report["chart_data_json"]).is_file()
    assert Path(report["old_confusion_csv"]).is_file()
    assert Path(report["new_confusion_csv"]).is_file()
    assert report["analysis"]["old_confusion_matrix"]["top_substitutions"][0]["expected"] == "ভাত"


def test_bangla_profile_is_selected_and_tunes_decoder_options():
    quality = resolve_transcription_quality(
        profile="auto",
        language="bn",
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        vad_aggressiveness="medium",
        condition_on_previous_text=True,
        chunk_length=30,
        initial_prompt=None,
        hotwords=None,
    )

    assert resolve_profile("auto", "bn") == "bangla_high_accuracy"
    assert quality.profile == "bangla_high_accuracy"
    assert quality.transcribe_kwargs["beam_size"] == 8
    assert quality.transcribe_kwargs["condition_on_previous_text"] is False
    assert quality.transcribe_kwargs["no_repeat_ngram_size"] == 4


def test_postprocess_flags_low_confidence_and_trims_repetition():
    result = {
        "language": "bn (Bangla)",
        "language_probability": 0.47,
        "text": "আমারে আমারে আমারে আমারে আমারে",
        "segments": [{"start": 1.0, "end": 3.0, "text": "আমারে আমারে আমারে আমারে আমারে"}],
    }

    processed = postprocess_result(result, "bangla_high_accuracy", repetition_guard=True)

    assert processed["text"] == "আমারে আমারে আমারে"
    assert processed["quality"]["low_confidence"] is True
    assert processed["quality"]["suspicious_segment_count"] == 1
    assert processed["segments"][0]["suspicious"] is True


def test_bangla_can_be_returned_as_banglish():
    assert transliterate_bangla_to_banglish("আমি ভালো আছি") == "ami valo achi"
    processed = postprocess_result(
        {
            "language": "bn (Bangla)",
            "language_probability": 0.9,
            "text": "আমি ভালো আছি",
            "segments": [{"start": 0, "end": 1, "text": "আমি ভালো আছি"}],
        },
        "bangla_high_accuracy",
        output_script="banglish",
    )

    assert processed["text"] == "ami valo achi"
    assert processed["native_text"] == "আমি ভালো আছি"
    assert processed["segments"][0]["native_text"] == "আমি ভালো আছি"
    assert processed["quality"]["output_script"] == "banglish"


def test_evaluate_request_accepts_accuracy_options():
    request = EvaluateRequest(
        profile="bangla_high_accuracy",
        chunk_length=20,
        vad_aggressiveness="high",
        condition_on_previous_text=False,
        repetition_guard=True,
        hotwords="সংসদ কৃষক সার",
    )

    assert request.profile == "bangla_high_accuracy"
    assert request.chunk_length == 20
    assert request.condition_on_previous_text is False


def test_transcribe_endpoint_accepts_accuracy_fields(monkeypatch, tmp_path):
    captured = {}

    def fake_run_transcription(audio_path, options, create_exports=True, display_name=None):
        captured["options"] = options
        return (
            {
                "file": display_name or str(audio_path),
                "duration_sec": 1,
                "transcription_time_sec": 1,
                "speed_factor": 1,
                "language": "bn (Bangla)",
                "language_probability": 0.7,
                "text": "আমি বাংলা বলি",
                "segments": [],
                "quality": {"profile": "bangla_high_accuracy", "warnings": []},
            },
            {"txt": tmp_path / "result.txt"},
        )

    monkeypatch.setattr("backend.routers.transcribe.run_transcription", fake_run_transcription)
    client = TestClient(app)
    response = client.post(
        "/api/transcribe",
        files={"audio": ("sample.wav", b"fake audio", "audio/wav")},
        data={
            "language": "bn",
            "profile": "bangla_high_accuracy",
            "chunk_length": "20",
            "vad_aggressiveness": "high",
            "condition_on_previous_text": "false",
            "repetition_guard": "true",
            "hotwords": "সংসদ কৃষক সার",
            "output_script": "banglish",
        },
    )

    assert response.status_code == 200
    assert captured["options"].profile == "bangla_high_accuracy"
    assert captured["options"].chunk_length == 20
    assert captured["options"].vad_aggressiveness == "high"
    assert captured["options"].condition_on_previous_text is False
    assert captured["options"].output_script == "banglish"
