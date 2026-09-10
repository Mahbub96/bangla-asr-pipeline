import platform
import shutil
import sys
from importlib import metadata

import torch
from fastapi import APIRouter

from backend.config import MODELS_DIR
from backend.services.jobs import job_registry
from backend.services.model_manager import model_manager

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.get("")
def diagnostics():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        compute = f"NVIDIA CUDA GPU: {device} ({memory:.1f} GB VRAM)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        compute = f"Apple Silicon Metal (MPS): {platform.machine()}"
    else:
        compute = f"CPU: Multi-Core ({platform.machine()})"

    packages = {}
    for package in ["fastapi", "uvicorn", "faster_whisper", "transformers", "datasets", "peft", "accelerate", "jiwer"]:
        try:
            __import__(package)
            packages[package] = "ok"
        except Exception:
            packages[package] = "missing"

    disk_models = []
    if MODELS_DIR.exists():
        for directory in sorted(MODELS_DIR.glob("models--*")):
            size_mb = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) / (1024 * 1024)
            disk_models.append({"name": directory.name.replace("models--", ""), "size_mb": round(size_mb, 1)})

    return {
        "compute": compute,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "fastapi": metadata.version("fastapi"),
        "uvicorn": metadata.version("uvicorn"),
        "ffmpeg": shutil.which("ffmpeg"),
        "packages": packages,
        "loaded_models": model_manager.cache_state(),
        "disk_models": disk_models,
        "active_jobs": job_registry.active_summary(),
    }
