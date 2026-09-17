"""Regression tests for the production-hardening behaviour.

Each test here pins a defect that was live in the deployed API:
path traversal via caller-supplied directories, unbounded job history,
leaked scratch files, missing upload limits, and the SPA fallback
masking unknown API routes.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import resolve_within_allowed_roots
from backend.main import app
from backend.services import storage
from backend.services.jobs import Job, JobRegistry


# --------------------------------------------------------------- path safety
def test_resolve_rejects_absolute_path_outside_data_roots():
    with pytest.raises(PermissionError):
        resolve_within_allowed_roots("/etc")


def test_resolve_rejects_dotdot_traversal():
    with pytest.raises(PermissionError):
        resolve_within_allowed_roots("data/../../../../etc/passwd")


def test_resolve_accepts_path_inside_data_root():
    resolved = resolve_within_allowed_roots("data/test")
    assert resolved.name == "test"
    assert "data" in resolved.parts


def test_batch_directory_endpoint_refuses_system_path():
    client = TestClient(app)
    response = client.post("/api/batch/directory/jobs", json={"directory_path": "/etc"})
    assert response.status_code == 403
    assert "outside the permitted data directories" in response.json()["detail"]


def test_evaluate_endpoint_refuses_system_path():
    client = TestClient(app)
    response = client.post(
        "/api/evaluate/jobs",
        data={"metadata_csv_path": "/etc/passwd", "audio_dir": "/etc"},
    )
    assert response.status_code in {400, 403}


# ------------------------------------------------------------ job retention
def _finished_job(registry: JobRegistry, job_id: str, age_seconds: float, export: Path | None = None) -> Job:
    job = Job(id=job_id, kind="batch", status="completed")
    job.updated_at = time.time() - age_seconds
    if export is not None:
        job.exports = {"csv": export}
    registry._jobs[job_id] = job
    return job


def test_prune_evicts_jobs_past_retention_window(tmp_path):
    registry = JobRegistry(retention_seconds=60, history_limit=100)
    export = tmp_path / "old.csv"
    export.write_text("a,b\n")
    _finished_job(registry, "old", age_seconds=600, export=export)
    _finished_job(registry, "fresh", age_seconds=1)

    removed = registry.prune()

    assert removed == 1
    assert registry.get("old") is None
    assert registry.get("fresh") is not None
    assert not export.exists(), "evicted job must delete its export files"


def test_prune_enforces_history_limit_keeping_newest():
    registry = JobRegistry(retention_seconds=10_000, history_limit=2)
    for index in range(5):
        _finished_job(registry, f"job{index}", age_seconds=index)

    registry.prune()

    assert registry.stats()["total"] == 2
    assert registry.get("job0") is not None, "newest job must survive"
    assert registry.get("job4") is None, "oldest job must be evicted"


def test_prune_never_evicts_the_latest_slot():
    registry = JobRegistry(retention_seconds=1, history_limit=1)
    _finished_job(registry, "latest", age_seconds=99_999)

    registry.prune()

    assert registry.get("latest") is not None


def test_put_releases_exports_of_the_replaced_job(tmp_path):
    registry = JobRegistry()
    old_export = tmp_path / "old.txt"
    old_export.write_text("old")
    first = Job(id="latest", kind="transcribe", status="completed", exports={"txt": old_export})
    registry.put(first)

    new_export = tmp_path / "new.txt"
    new_export.write_text("new")
    registry.put(Job(id="latest", kind="transcribe", status="completed", exports={"txt": new_export}))

    assert not old_export.exists(), "replaced job's export must be deleted"
    assert new_export.exists()


# -------------------------------------------------------------- scratch disk
def test_scratch_files_are_created_inside_the_managed_dir():
    path = storage.new_temp_path(".wav")
    try:
        assert path.parent == storage.UPLOAD_DIR
    finally:
        storage.discard(path)
    assert not path.exists()


def test_discard_is_safe_on_missing_and_none_paths(tmp_path):
    storage.discard(None, tmp_path / "does-not-exist.txt")  # must not raise


def test_sweep_orphans_removes_only_stale_files():
    stale = storage.new_temp_path(".wav")
    fresh = storage.new_temp_path(".wav")
    import os

    old_time = time.time() - 10_000
    os.utime(stale, (old_time, old_time))
    try:
        removed = storage.sweep_orphans(max_age_seconds=3600)
        assert removed >= 1
        assert not stale.exists()
        assert fresh.exists(), "recent scratch file must survive the sweep"
    finally:
        storage.discard(stale, fresh)


# ------------------------------------------------------------------ HTTP API
def test_bulk_callers_do_not_create_per_file_exports(monkeypatch):
    """Batch/evaluate read only `result`; per-file exports would be pure litter."""
    from backend.routers import transcribe as transcribe_router

    fake_result = {
        "language": "en",
        "language_probability": 0.9,
        "text": "hello",
        "segments": [],
        "duration_sec": 1.0,
        "speed_factor": 1.0,
        "quality": {},
    }
    monkeypatch.setattr(transcribe_router, "sanitize_audio_input", lambda path: Path(path))
    monkeypatch.setattr(
        transcribe_router.model_manager, "transcribe", lambda *a, **k: dict(fake_result)
    )
    monkeypatch.setattr(
        transcribe_router, "postprocess_result", lambda result, *a, **k: result
    )

    before = len(list(storage.EXPORT_DIR.iterdir()))
    _, exports = transcribe_router.run_transcription(
        Path("dummy.wav"),
        transcribe_router.TranscriptionOptions(model_name="tiny", language="en"),
        create_exports=False,
    )
    after = len(list(storage.EXPORT_DIR.iterdir()))

    assert exports == {}
    assert after == before, "create_exports=False must not write scratch files"


def test_upload_result_reports_client_filename_not_scratch_name(monkeypatch, tmp_path):
    """result['file'] must never expose the server's temp filename."""
    from backend.routers import transcribe as transcribe_router

    def fake_run(audio_path, options, create_exports=True, display_name=None):
        return (
            {
                "file": display_name or Path(audio_path).name,
                "duration_sec": 1,
                "transcription_time_sec": 1,
                "speed_factor": 1,
                "language": "en",
                "language_probability": 0.9,
                "text": "hi",
                "segments": [],
                "quality": {},
            },
            {},
        )

    monkeypatch.setattr(transcribe_router, "run_transcription", fake_run)
    client = TestClient(app)
    response = client.post(
        "/api/transcribe", files={"audio": ("my recording.wav", b"12345", "audio/wav")}
    )

    assert response.status_code == 200
    assert response.json()["result"]["file"] == "my recording.wav"


def test_unknown_api_route_returns_404_not_spa_html():
    client = TestClient(app)
    response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404
    assert "text/html" not in response.headers.get("content-type", "")


def test_empty_upload_is_rejected():
    client = TestClient(app)
    response = client.post("/api/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_health_reports_job_stats():
    client = TestClient(app)
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert "total" in payload["jobs"]
    assert "retention_seconds" in payload["jobs"]
