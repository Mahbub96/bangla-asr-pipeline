from typing import Any, Literal

from pydantic import BaseModel, Field


class TranscriptionOptions(BaseModel):
    model_name: str = "large-v3-turbo"
    language: Literal["auto", "bn", "en"] = "auto"
    beam_size: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    initial_prompt: str | None = None
    vad_filter: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"


class DirectoryBatchRequest(TranscriptionOptions):
    directory_path: str = "data/test/audio"


class EvaluateRequest(TranscriptionOptions):
    metadata_csv_path: str = "data/test/metadata.csv"
    audio_dir: str = "data/test/audio"


class TrainingRequest(BaseModel):
    model_name_or_path: str = "openai/whisper-large-v3-turbo"
    language: str = "bengali"
    task: Literal["transcribe", "translate"] = "transcribe"
    train_csv: str = "data/train/metadata.csv"
    train_audio: str = "data/train/audio"
    val_csv: str = "data/val/metadata.csv"
    val_audio: str = "data/val/audio"
    output_dir: str = "./checkpoints/whisper_bangla_lora"
    num_proc: int = 2
    finetune_mode: str = "lora"
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    precision: str = "fp16"
    dataloader_num_workers: int = 2
    optim: str = "adamw_torch"
    learning_rate: str = "1e-4"
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    max_steps: int = -1
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 25
    save_total_limit: int = 2
    metric_for_best_model: str = "wer"
    report_to: str = "tensorboard"
    generation_max_length: int = 225
    generation_num_beams: int = 1


class JobResponse(BaseModel):
    job_id: str
    status_url: str
    events_url: str


class JobSnapshot(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    logs: list[str]
    result: Any = None
    exports: dict[str, str]
    created_at: float
    updated_at: float
    cancel_requested: bool

