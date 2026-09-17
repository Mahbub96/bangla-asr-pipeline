# Bangla & English ASR Studio

Bangla & English ASR Studio is a local-first speech-to-text application and Whisper fine-tuning pipeline. It gives operators and developers one place to transcribe audio, batch-process folders, measure ASR accuracy, prepare training data, launch training, and compare old vs new models with evidence.

## What problem does it solve?

Generic Whisper models work well, but production Bangla/Banglish speech has practical problems:

- Bangla audio may be misdetected as another Indic language.
- Mixed Bangla/English speech needs controlled decoding and repeatable evaluation.
- Training without a baseline creates a gray area: you may not know whether the new model is actually better.
- Large datasets and model files need durable local storage, resumable download, validation, and clear folder structure.
- Operators need a UI for transcription/batch/evaluation, while engineers still need CLI scripts for automation and GPU servers.

This project solves those by combining:

- **React operator UI** for microphone/file transcription, batch jobs, benchmark jobs, diagnostics, and training controls.
- **FastAPI backend** for model loading, audio normalization, job tracking, exports, evaluation, and training subprocesses.
- **CLI scripts** for direct transcription, evaluation, model download, dataset preparation, controlled training, and old-vs-new model comparison.
- **Docker Compose stack** for reproducible local/dev/prod runs.

## Main features

- Browser microphone transcription and file upload fallback.
- Single-file and directory transcription.
- Batch audio jobs with live status and CSV/JSON exports.
- Benchmarking with WER/CER against ground-truth metadata.
- Bangla/English decoding profiles and post-processing guards.
- Whisper model cache under `models/` so downloads survive rebuilds.
- Dataset layout for `train`, `val`, and `test` splits.
- GPU fine-tuning with LoRA/QLoRA options.
- Controlled model training workflow: backup old model, evaluate old model, train new model, evaluate new model, compare results.
- Docker CPU, Docker dev hot reload, and NVIDIA GPU override.

## Repository layout

```text
.
├── backend/                    # FastAPI app, routers, services, schemas
│   └── Dockerfile              # Backend image; CPU by default, CUDA via build args
├── frontend/                   # Vite + React operator UI
│   ├── Dockerfile              # Dev/prod frontend image
│   └── nginx.conf.template     # Production SPA + /api reverse proxy
├── scripts/
│   ├── transcribe.py           # CLI transcription for one file or a directory
│   ├── evaluate.py             # WER/CER benchmark runner
│   ├── prepare_data.py         # Dataset folder init and metadata validation
│   ├── download_model.py       # Pre-download Whisper/faster-whisper models
│   ├── train_whisper.py        # Whisper fine-tuning script
│   ├── model_guard.py          # Backup + old/new model comparison guard
│   ├── download_subakko_segmented.py # Resumable segmented SUBAK.KO downloader
│   └── run_train_gpu.sh        # Example GPU training launcher
├── data/
│   ├── sources/                # Raw downloaded datasets
│   ├── processed/              # Derived/extracted audio and data
│   ├── manifests/              # Master and generated manifests
│   ├── train/audio/            # Training audio files
│   ├── train/metadata.csv      # Training metadata: audio_path,sentence
│   ├── val/audio/              # Validation audio files
│   ├── val/metadata.csv        # Validation metadata
│   ├── test/audio/             # Held-out test audio files
│   └── test/metadata.csv       # Held-out benchmark metadata
├── models/                     # Model cache; ignored by git
├── checkpoints/                # Fine-tuned checkpoints; ignored by git
├── backups/                    # Manual/model backups; ignored if created locally
├── reports/                    # Evaluation/comparison outputs
├── docker-compose.yml          # Production CPU stack
├── docker-compose.dev.yml      # Dev stack with hot reload
├── docker-compose.gpu.yml      # NVIDIA CUDA override
├── docker-start.sh             # Docker launcher
├── start.sh                    # Local dev launcher for UI + API
├── run_api.sh                  # Local backend only
├── run_ui.sh                   # Local frontend + backend
├── requirements.txt            # Local/API/inference dependencies
├── requirements_gpu.txt        # Training/GPU dependencies
└── .env.example                # Docker/runtime configuration template
```

## Requirements

### For Docker usage

- Docker Desktop or Docker Engine
- Docker Compose v2: `docker compose version`
- Enough disk space for datasets and models

### For local development without Docker

- Python 3.10+
- Node.js 20+ recommended
- FFmpeg
- `npm`
- Optional NVIDIA CUDA stack for GPU training

macOS FFmpeg install:

```bash
brew install ffmpeg
```

Ubuntu/Debian FFmpeg install:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Quick start with Docker

Docker is the easiest way to run the full stack.

```bash
cp .env.example .env
./docker-start.sh
```

Open:

- UI: `http://localhost:8080`
- API: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`

Follow logs:

```bash
docker compose logs -f backend
```

Stop everything:

```bash
./docker-start.sh down
```

### Docker dev mode with hot reload

```bash
cp .env.example .env
./docker-start.sh dev
```

Open:

- UI: `http://localhost:5173`
- API: `http://localhost:8000`

### Docker GPU mode

Use this only on Linux/WSL2 with NVIDIA Container Toolkit. macOS Docker Desktop cannot pass through NVIDIA GPU.

Verify GPU access first:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Then run:

```bash
./docker-start.sh gpu
```

For training endpoints inside Docker, set this in `.env` before building:

```env
INSTALL_TRAINING=true
DATA_MOUNT_MODE=rw
```

Then rebuild:

```bash
./docker-start.sh gpu
```

## Local development setup

Use this if you want to modify backend/frontend code directly on the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
npm install --prefix frontend
```

Start backend + frontend:

```bash
./start.sh
```

Or start them separately:

```bash
./run_api.sh
npm run dev --prefix frontend
```

Open:

- Local UI: `http://127.0.0.1:5173`
- Local API: `http://127.0.0.1:8000`

Important browser note: microphone access works on `localhost` / `127.0.0.1` or HTTPS. If you open the UI through a LAN IP, most browsers disable microphone capture and you should use file upload instead.

## Environment configuration

Copy `.env.example` to `.env` for Docker runs:

```bash
cp .env.example .env
```

Common settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FRONTEND_PORT` | Production UI port | `8080` |
| `BACKEND_PORT` | FastAPI port | `8000` |
| `DEV_FRONTEND_PORT` | Vite dev UI port | `5173` |
| `ASR_CORS_ORIGINS` | Allowed browser origins | `*` |
| `DATA_MOUNT_MODE` | Container dataset mount mode: `ro` or `rw` | `ro` |
| `BACKEND_MEMORY_LIMIT` | Backend container memory cap | `8g` |
| `OMP_NUM_THREADS` | CPU thread count | `4` |
| `TORCH_INDEX_URL` | PyTorch wheel index | CPU index |
| `INSTALL_TRAINING` | Install training dependencies in Docker image | `false` |

## Dataset format

Each split uses the same simple CSV format:

```csv
audio_path,sentence
audio_001.wav,আমি বাংলায় কথা বলতে ভালোবাসি।
audio_002.wav,Automatic speech recognition is ready.
```

Rules:

- `audio_path` is relative to that split's `audio/` folder unless an absolute path is provided.
- `sentence` is the ground-truth transcript.
- Keep `test` fixed and untouched for fair old-vs-new model comparison.
- Use `train` for fitting, `val` for training-time evaluation, and `test` for final benchmark.

Initialize folder structure:

```bash
python3 scripts/prepare_data.py --init
```

Validate a split:

```bash
python3 scripts/prepare_data.py \
  --validate \
  --csv data/test/metadata.csv \
  --audio_dir data/test/audio
```

## Large dataset preparation

Raw downloads should stay under `data/sources/<dataset>/`. Derived/extracted audio should go under `data/processed/` or the trainer-facing `data/train`, `data/val`, and `data/test` folders.

For SUBAK.KO segmented Hugging Face download:

```bash
python3 scripts/download_subakko_segmented.py \
  --split all \
  --segments 8 \
  --files 2 \
  2>&1 | tee data/sources/subakko/download.log
```

Human-readable live download log:

```bash
tail -f data/sources/subakko/download.log | grep --line-buffered -E 'START|DONE|FAIL|retry|complete'
```

Meaning:

- `START`: a shard started
- `DONE`: a shard completed
- `retry`: a part failed but is retrying
- `FAIL`: a shard failed after retries
- `SUBAK.KO download complete`: all selected files completed

Note: Hugging Face ASR datasets may be Parquet files with embedded audio, not plain `.wav` files. Download completion does not always mean training-ready audio exists. Extraction/conversion may still be required.

## Model download

Pre-download the default faster-whisper model cache:

```bash
python3 scripts/download_model.py --model large-v3-turbo --models_dir models
```

Supported common names include:

- `large-v3-turbo`
- `large-v3`
- `medium`
- `small`
- `base`
- `tiny`

## CLI transcription

Single file:

```bash
python3 scripts/transcribe.py path/to/audio.wav --language auto
```

Force Bangla:

```bash
python3 scripts/transcribe.py path/to/audio.wav --language bn
```

Directory with JSON output:

```bash
python3 scripts/transcribe.py data/test/audio --language auto --output reports/transcriptions.json
```

Useful options:

```bash
python3 scripts/transcribe.py path/to/audio.wav \
  --model large-v3-turbo \
  --language bn \
  --beam_size 5 \
  --vad_aggressiveness medium \
  --profile bangla_high_accuracy
```

## Evaluation / efficiency check

Evaluate one model on the fixed test set:

```bash
python3 scripts/evaluate.py \
  --metadata data/test/metadata.csv \
  --audio_dir data/test/audio \
  --model large-v3-turbo \
  --language bn \
  --output reports/evaluation_results.csv
```

Outputs:

- Total evaluated samples
- Overall WER
- Overall CER
- Per-sample predictions and errors in the output CSV

Lower WER/CER is better.

## Controlled training workflow: backup → baseline → train → compare

Before any real training run, do this sequence so the project does not enter an uncontrolled gray area.

### 1. Backup current model/checkpoint

If the current model is the local model cache:

```bash
python3 scripts/model_guard.py backup \
  --source models \
  --dest backups/models \
  --name before-training-$(date +%Y%m%d-%H%M%S)
```

If the current model is a checkpoint:

```bash
python3 scripts/model_guard.py backup \
  --source checkpoints/current_model \
  --dest backups/models \
  --name current-checkpoint-before-training-$(date +%Y%m%d-%H%M%S)
```

The backup command writes `backup_manifest.json` inside the backup folder.

### 2. Evaluate the old model on the fixed test set

```bash
python3 scripts/evaluate.py \
  --metadata data/test/metadata.csv \
  --audio_dir data/test/audio \
  --model large-v3-turbo \
  --language bn \
  --output reports/baseline-old.csv
```

### 3. Train the new model

Install GPU/training requirements on the training machine:

```bash
pip install -r requirements_gpu.txt
```

Run LoRA training:

```bash
python3 scripts/train_whisper.py \
  --model_name_or_path openai/whisper-large-v3-turbo \
  --train_csv data/train/metadata.csv \
  --train_audio data/train/audio \
  --val_csv data/val/metadata.csv \
  --val_audio data/val/audio \
  --output_dir checkpoints/whisper_bangla_lora \
  --language bengali \
  --task transcribe \
  --use_lora \
  --batch_size 8 \
  --eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --eval_steps 200 \
  --save_steps 200 \
  --metric_for_best_model wer
```

Or use the example launcher:

```bash
bash scripts/run_train_gpu.sh
```

### 4. Evaluate the new trained model on the exact same test set

```bash
python3 scripts/evaluate.py \
  --metadata data/test/metadata.csv \
  --audio_dir data/test/audio \
  --model checkpoints/whisper_bangla_lora \
  --language bn \
  --output reports/after-new.csv
```

### 5. Compare old vs new

```bash
python3 scripts/model_guard.py compare \
  --old reports/baseline-old.csv \
  --new reports/after-new.csv \
  --output reports/model_comparison.json
```

This writes:

- `reports/model_comparison.json`
- `reports/model_comparison.md`

Verdicts:

| Verdict | Meaning |
| --- | --- |
| `new_better` | New model improved WER/CER overall. |
| `old_better` | Old model is safer; new model regressed. |
| `gray_area_mixed_metrics` | One metric improved and another regressed; needs human review before replacing the old model. |

## LoRA training notes

LoRA fine-tuning trains small adapter layers instead of all Whisper weights.

Benefits:

- Lower VRAM/RAM requirement than full fine-tuning.
- Smaller checkpoint size.
- Faster experimentation.
- Original base model remains reusable.

Common defaults:

| Parameter | Default | Notes |
| --- | ---: | --- |
| `lora_r` | `32` | Adapter rank |
| `lora_alpha` | `64` | Adapter scaling |
| `lora_dropout` | `0.05` | Regularization |
| `lora_target_modules` | `q_proj,v_proj` | Whisper attention projection modules |
| `metric_for_best_model` | `wer` | Lower is better |

## API and UI workflow

After the app is running, use the UI sections:

- **Live Mic & Audio**: record or upload one file and transcribe.
- **Batch Audio**: process many files with job progress.
- **Benchmark**: run WER/CER evaluation against metadata CSV.
- **Training**: configure fine-tuning parameters and stream logs.
- **Diagnostics**: check package, hardware, and runtime status.

Backend API routes are under `/api/*`. Health check:

```bash
curl http://localhost:8000/api/health
```

## Production and security notes

- The API has no built-in authentication. Put it behind a trusted network, VPN, or authenticating reverse proxy before exposing it.
- Keep `models/`, `checkpoints/`, `data/`, `backups/`, and `reports/` out of git unless intentionally publishing small examples.
- Container logs are capped in `docker-compose.yml` to avoid unbounded disk growth.
- Uploaded scratch files are stored in the container temp volume and cleaned by backend retention logic.
- Batch/evaluate paths are restricted to `./data` and configured extra roots to reduce path traversal risk.

## Common commands

```bash
# Docker production
./docker-start.sh

# Docker dev hot reload
./docker-start.sh dev

# Docker logs
./docker-start.sh logs

# Stop Docker stack
./docker-start.sh down

# Local backend only
./run_api.sh

# Local UI + backend
./start.sh

# Validate test dataset
python3 scripts/prepare_data.py --validate --csv data/test/metadata.csv --audio_dir data/test/audio

# Transcribe a file
python3 scripts/transcribe.py path/to/audio.wav --language bn

# Evaluate fixed test set
python3 scripts/evaluate.py --metadata data/test/metadata.csv --audio_dir data/test/audio --language bn

# Compare old/new evaluation outputs
python3 scripts/model_guard.py compare --old reports/baseline-old.csv --new reports/after-new.csv --output reports/model_comparison.json
```

## Troubleshooting

### Docker project is not running

```bash
docker compose ps
./docker-start.sh logs
```

### Port already in use

Change `.env`:

```env
FRONTEND_PORT=8081
BACKEND_PORT=8001
DEV_FRONTEND_PORT=5174
```

Then restart Docker.

### Microphone does not work on LAN IP

Use `http://localhost:5173` or `http://127.0.0.1:5173`, or serve the UI through HTTPS. Browsers block microphone access on normal LAN HTTP origins.

### Training dependencies missing

For local/GPU training:

```bash
pip install -r requirements_gpu.txt
```

For Docker training, set:

```env
INSTALL_TRAINING=true
DATA_MOUNT_MODE=rw
```

Then rebuild the image.

### Download is running but no `.wav` files appear

Some ASR datasets download as Parquet shards. Parquet can contain embedded audio bytes and transcripts. You still need an extraction/conversion step before the data becomes trainer-ready audio files.

## License

This project is released under the Apache License 2.0.
