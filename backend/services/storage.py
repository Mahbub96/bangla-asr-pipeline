"""Temporary-file lifecycle for the ASR API.

Every upload, ffmpeg-normalised WAV and generated export used to be written with
``tempfile.NamedTemporaryFile(delete=False)`` into the system ``/tmp`` and never
removed, so a long-running container slowly filled its writable layer. All
scratch files now live under ``TMP_DIR`` (a mounted volume in Docker) and are
either deleted as soon as the request finishes or reaped once their owning job
is evicted.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config import TMP_DIR

UPLOAD_DIR = TMP_DIR / "uploads"
EXPORT_DIR = TMP_DIR / "exports"

for _directory in (UPLOAD_DIR, EXPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def new_temp_path(suffix: str = "", directory: Path | None = None) -> Path:
    """Create an empty file in a managed scratch directory and return its path."""
    target_dir = directory or UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=suffix, dir=target_dir, delete=False)
    handle.close()
    return Path(handle.name)


def write_temp_text(content: str, suffix: str, directory: Path | None = None) -> Path:
    path = new_temp_path(suffix, directory or EXPORT_DIR)
    path.write_text(content, encoding="utf-8")
    return path


def discard(*paths: Path | str | None) -> None:
    """Best-effort delete; never raises, so cleanup can live in a finally block."""
    for path in paths:
        if not path:
            continue
        try:
            candidate = Path(path)
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def temp_lifetime(*paths: Path | str | None) -> Iterator[None]:
    try:
        yield
    finally:
        discard(*paths)


def sweep_orphans(max_age_seconds: float) -> int:
    """Delete scratch files older than ``max_age_seconds``.

    Safety net for files whose owning request died mid-flight (client abort,
    worker crash) and therefore never reached their cleanup path.
    """
    cutoff = time.time() - max_age_seconds
    removed = 0
    for directory in (UPLOAD_DIR, EXPORT_DIR):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    return removed


def usage_bytes() -> int:
    total = 0
    for directory in (UPLOAD_DIR, EXPORT_DIR):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total
