import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def format_timestamp(seconds: float, sep: str) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def generate_srt(segments: list[dict[str, Any]]) -> str:
    lines = []
    for index, segment in enumerate(segments, 1):
        start = format_timestamp(float(segment.get("start", segment.get("Start (s)", 0.0))), ",")
        end = format_timestamp(float(segment.get("end", segment.get("End (s)", 0.0))), ",")
        text = str(segment.get("text", segment.get("Transcription", ""))).strip()
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def generate_vtt(segments: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT\n"]
    for segment in segments:
        start = format_timestamp(float(segment.get("start", segment.get("Start (s)", 0.0))), ".")
        end = format_timestamp(float(segment.get("end", segment.get("End (s)", 0.0))), ".")
        text = str(segment.get("text", segment.get("Transcription", ""))).strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def write_temp_export(content: str, suffix: str) -> Path:
    target = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    target.write(content)
    target.close()
    return Path(target.name)


def create_transcription_exports(result: dict[str, Any]) -> dict[str, Path]:
    segments = result.get("segments", [])
    return {
        "txt": write_temp_export(str(result.get("text", "")), ".txt"),
        "srt": write_temp_export(generate_srt(segments), ".srt"),
        "vtt": write_temp_export(generate_vtt(segments), ".vtt"),
        "json": write_temp_export(json.dumps(result, ensure_ascii=False, indent=2), ".json"),
    }


def create_table_exports(rows: list[dict[str, Any]], stem: str = "results") -> dict[str, Path]:
    df = pd.DataFrame(rows)
    return {
        "csv": write_temp_export(df.to_csv(index=False), f"_{stem}.csv"),
        "json": write_temp_export(json.dumps(rows, ensure_ascii=False, indent=2), f"_{stem}.json"),
    }

