import shutil
import subprocess
import tempfile
from pathlib import Path


def sanitize_audio_input(audio_path: str | Path) -> Path:
    """Convert browser/container audio to a clean 16kHz mono WAV when ffmpeg is available."""
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    if not shutil.which("ffmpeg"):
        return source

    target = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
    target.close()
    target_path = Path(target.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    target_path.unlink(missing_ok=True)
    return source


def allowed_audio_path(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".mp4"}

