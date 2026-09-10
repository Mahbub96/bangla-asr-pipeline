import type { MicState } from "../lib/media";

export type ExportMap = Record<string, string>;

export type Segment = {
  start: number;
  end: number;
  text: string;
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
  compression_ratio?: number | null;
  repetition_score?: number;
  suspicious?: boolean;
};

export type TranscriptionResult = {
  file: string;
  duration_sec: number;
  transcription_time_sec: number;
  speed_factor: number;
  language: string;
  language_probability: number;
  text: string;
  segments: Segment[];
  quality?: {
    profile: string;
    language_probability: number;
    low_confidence: boolean;
    repetition_score: number;
    suspicious_segment_count: number;
    warnings: string[];
    suggested_retry: string;
    applies_bangla_cleanup: boolean;
  };
};

export type TranscriptionPayload = {
  result?: TranscriptionResult;
  exports: ExportMap;
};

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type JobSnapshot = {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  message: string;
  logs: string[];
  result: any;
  exports: ExportMap;
  created_at: number;
  updated_at: number;
  cancel_requested: boolean;
};

export type JobResponse = {
  job_id: string;
  status_url: string;
  events_url: string;
};

export type DiagnosticsPayload = {
  compute: string;
  os: string;
  python: string;
  ffmpeg?: string;
  loaded_models: unknown[];
  disk_models: Array<{ name: string; size_mb: number }>;
};

export type TrainingConfig = {
  model_name_or_path: string;
  language: string;
  task: string;
  train_csv: string;
  train_audio: string;
  val_csv: string;
  val_audio: string;
  output_dir: string;
  batch_size: number;
  eval_batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: string;
  num_epochs: number;
  finetune_mode: string;
  precision: string;
};

export type StudioTab = "live" | "batch" | "benchmark" | "training" | "diagnostics";
export type OutputView = "transcript" | "segments" | "raw";

export type RecorderState = {
  micState: MicState;
  file: File | null;
  recorded: Blob | null;
  previewUrl: string;
  previewMimeType: string;
  recordedDuration: number;
  recordingSeconds: number;
  inputLevel: number;
  waveform: number[];
};
