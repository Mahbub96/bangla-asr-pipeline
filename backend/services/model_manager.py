import sys
import threading
from pathlib import Path
from typing import Any

import torch

from backend.config import MODELS_DIR, SCRIPTS_DIR

sys.path.insert(0, str(SCRIPTS_DIR))
from transcribe import get_transcriber, transcribe_file  # noqa: E402


class ModelManager:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()

    def resolve_device(self, device: str | None = None) -> str:
        if device and device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_compute_type(self, device: str, compute_type: str | None = None) -> str:
        if compute_type and compute_type != "auto":
            return compute_type
        return "float16" if device == "cuda" else "int8"

    def get(self, model_name: str, device: str | None = None, compute_type: str | None = None) -> Any:
        resolved_device = self.resolve_device(device)
        resolved_compute = self.resolve_compute_type(resolved_device, compute_type)
        key = (model_name, resolved_device, resolved_compute)
        with self._lock:
            if key not in self._models:
                self._models[key] = get_transcriber(
                    model_size=model_name,
                    device=resolved_device,
                    compute_type=resolved_compute,
                    download_root=str(MODELS_DIR),
                )
            return self._models[key]

    def transcribe(self, audio_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        model = self.get(kwargs.pop("model_name"), kwargs.pop("device", None), kwargs.pop("compute_type", None))
        return transcribe_file(model, audio_path, **kwargs)

    def cache_state(self) -> list[dict[str, str]]:
        return [
            {"model": model, "device": device, "compute_type": compute_type}
            for model, device, compute_type in self._models.keys()
        ]


model_manager = ModelManager()

