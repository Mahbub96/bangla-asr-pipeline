from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import EvaluateRequest
from backend.services.exports import generate_srt, generate_vtt
from backend.services.jobs import Job
from backend.services.quality import postprocess_result, resolve_profile, resolve_transcription_quality
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

    def fake_run_transcription(audio_path, options):
        captured["options"] = options
        return (
            {
                "file": str(audio_path),
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
        },
    )

    assert response.status_code == 200
    assert captured["options"].profile == "bangla_high_accuracy"
    assert captured["options"].chunk_length == 20
    assert captured["options"].vad_aggressiveness == "high"
    assert captured["options"].condition_on_previous_text is False
