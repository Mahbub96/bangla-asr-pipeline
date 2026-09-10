from pathlib import Path

from backend.services.exports import generate_srt, generate_vtt
from backend.services.jobs import Job
from backend.services.training import build_training_args, relative_command


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


def test_training_command_uses_lora_defaults():
    args = build_training_args({})
    command = relative_command(args)

    assert "scripts/train_whisper.py" in command
    assert "--use_lora" in args
    assert "--lora_r" in args
    assert "--model_name_or_path openai/whisper-large-v3-turbo" in command

