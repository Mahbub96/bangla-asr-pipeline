import type { TrainingConfig } from "../types/asr";

export const ASR_MODELS = ["large-v3-turbo", "tiny"];

export const ASR_LANGUAGES = [
  { label: "Auto Bangla / English", value: "auto" },
  { label: "Bangla", value: "bn" },
  { label: "English", value: "en" }
];

export const ACCURACY_PROFILES = [
  { label: "Auto profile", value: "auto" },
  { label: "Balanced", value: "balanced" },
  { label: "Bangla High Accuracy", value: "bangla_high_accuracy" },
  { label: "English Fast", value: "english_fast" }
];

export const VAD_AGGRESSIVENESS = [
  { label: "Off", value: "off" },
  { label: "Low", value: "low" },
  { label: "Medium", value: "medium" },
  { label: "High", value: "high" }
];

export const DEFAULT_BATCH_DIRECTORY = "data/test/audio";
export const DEFAULT_METADATA_CSV = "data/test/metadata.csv";
export const DEFAULT_AUDIO_DIRECTORY = "data/test/audio";

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  model_name_or_path: "openai/whisper-large-v3-turbo",
  language: "bengali",
  task: "transcribe",
  train_csv: "data/train/metadata.csv",
  train_audio: "data/train/audio",
  val_csv: "data/val/metadata.csv",
  val_audio: "data/val/audio",
  output_dir: "./checkpoints/whisper_bangla_lora",
  batch_size: 8,
  eval_batch_size: 8,
  gradient_accumulation_steps: 2,
  learning_rate: "1e-4",
  num_epochs: 5,
  finetune_mode: "lora",
  precision: "fp16"
};

export function formatDuration(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);
  return `${mins}:${secs.toString().padStart(2, "0")}.${tenths}`;
}
