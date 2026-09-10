# Bangla & English Automatic Speech Recognition (ASR) Studio

An end-to-end Automatic Speech Recognition (ASR) studio and scalable fine-tuning pipeline tailored for **Bangla (বাংলা)** and **English** speech, powered by OpenAI's Whisper (`large-v3-turbo` / `large-v3`).

Designed with a local-first full-stack architecture:
1. **React Operator Studio:** A Vite + TypeScript frontend for microphone capture, file upload, batch jobs, benchmark runs, training logs, diagnostics, and exports.
2. **FastAPI ASR Backend:** A modular API owns model caching, audio normalization, transcription, background jobs, SSE progress streams, export files, and training subprocess control.
3. **CLI Compatibility:** Existing scripts remain available for direct inference, evaluation, data setup, model download, and remote GPU training.

---

## 🌟 Key Features & Highlights

- **🎙️ Browser Microphone Studio**: Reliable record-then-transcribe microphone capture with explicit browser permission states and file upload fallback.
- **📄 Multi-Format Subtitle & Text Export**: One-click download in `.txt`, `.srt` (SubRip), `.vtt` (WebVTT), and `.json` formats with precise timestamps.
- **📂 Batch Audio Transcription**: Multi-file upload or server-directory jobs with live status, logs, and CSV/JSON export.
- **📊 Benchmark Suite**: Automated Word Error Rate (WER) and Character Error Rate (CER) jobs against ground-truth CSVs.
- **🏋️ Batched Audio Training Studio (36+ Parameters)**:
  - Complete GUI exposing every Whisper training parameter (LoRA rank/alpha/dropout, target modules, QLoRA 4-bit, gradient checkpointing, mixed precision, LR schedulers, warmup, evaluation steps, beam search decoding).
  - **Dynamic Model-Adaptive Presets**: Changing the base model (`large-v3-turbo`, `large-v3`, `medium`, `small`, `base`, `tiny`) automatically tunes optimal batch sizes, gradient accumulation, precision, and LoRA ranks.
  - **Live Subprocess Console**: SSE terminal output streaming loss, WER/CER, and checkpoints with an instant abort button.
  - **Dataset Integrity Verifier**: Pre-flight validation of audio existence, sample rates, and transcripts.
  - **CLI Command Generator**: Generates equivalent copy-paste commands ready for remote headless cloud clusters (RunPod, Lambda Labs, AWS).
- **🖥️ System Diagnostics**: Real-time GPU detection, CUDA VRAM monitor, and package dependency health checks.
- **📱 Task-First React Layout**: Compact operator panels, responsive tables, clear progress states, and local workstation defaults.

---

## 📁 Repository Structure

```text
Bangla ASR/
├── backend/                 # FastAPI app, routers, and ASR service layer
├── frontend/                # Vite React TypeScript operator UI
├── start.sh                 # One-click launcher for frontend + backend
├── run_ui.sh                # Launches React + FastAPI dev stack
├── run_api.sh               # Launches FastAPI backend only
├── run_cli_test.sh          # Quick CLI test runner script
├── data/
│   ├── train/               # Training dataset
│   │   ├── audio/           # Audio files (.wav, .mp3, .flac)
│   │   └── metadata.csv     # [audio_path, sentence]
│   ├── val/                 # Validation dataset
│   │   ├── audio/
│   │   └── metadata.csv
│   └── test/                # Testing dataset
│       ├── audio/
│       └── metadata.csv
├── models/                  # Checkpoints cache (ignored by git)
├── scripts/
│   ├── transcribe.py        # Inference script (single file or batch folder)
│   ├── evaluate.py          # WER & CER benchmark tool
│   ├── prepare_data.py      # Directory setup and dataset validator
│   ├── create_test_audio.py # Generates test audio samples
│   ├── download_model.py    # Model weight downloader
│   ├── train_whisper.py     # Scalable GPU training script (36+ CLI options)
│   └── run_train_gpu.sh     # Shell launcher for remote GPU execution
├── requirements.txt         # Dependencies for local testing & Web UI
├── requirements_gpu.txt     # Dependencies for GPU training cluster
├── .gitignore
└── README.md
```

---

## 📊 Dataset Format

All datasets (`train`, `val`, `test`) follow a standardized CSV schema with audio files in the corresponding `audio/` directory:

### `metadata.csv` Schema:
```csv
audio_path,sentence
audio_001.wav,আমি বাংলায় কথা বলতে ভালোবাসি।
audio_002.wav,Automatic speech recognition is now ready.
```

* **`audio_path`**: File name inside the dataset's `audio/` folder.
* **`sentence`**: Ground truth transcript (Bengali script or English text).

### Validate Your Dataset
Verify audio file integrity and check for missing files referenced in the CSV:
```bash
python scripts/prepare_data.py --validate --csv data/test/metadata.csv --audio_dir data/test/audio
```

---

## 🚀 Quick Start (Local Machine)

### 1. System Requirements & Setup

#### On Ubuntu / Debian:
```bash
# Install system audio codec
sudo apt update && sudo apt install -y ffmpeg

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### On macOS (Apple Silicon M1 / M2 / M3 / M4):
```bash
# Install system audio codec via Homebrew
brew install ffmpeg

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the React + FastAPI Studio
Launch the local browser dashboard:
```bash
./start.sh
```
Then open **http://127.0.0.1:5173** in your browser.

The backend API runs at **http://127.0.0.1:8000**. Microphone access works from `localhost` / `127.0.0.1`; do not open the app through `0.0.0.0`.

Available sections:
- **🎯 Live Mic & Audio**:
  - Browser records audio with `MediaRecorder`, then sends the finished clip to FastAPI.
  - Cross-browser audio normalization converts WebM, OGG, MP4/AAC, MP3, and WAV streams to clean 16kHz WAV via FFmpeg.
- **📂 Batch Audio**: Folder or multi-file audio batch processing with live streaming tables.
- **📊 Benchmark (WER / CER)**: Error rate calculation and dataset validation against ground truth.
- **🏋️ Batched Training**: Full training hyperparameter suite (36+ parameters) with live console logs.
- **🖥️ Diagnostics**: Hardware acceleration and environment health check.

### 3. CLI Test Runner
Run transcription directly across the test audio folder:
```bash
./run_cli_test.sh
```
*(Optional arguments: `./run_cli_test.sh <target_path> <model_name> <language>`)*

---

### 4. Download Whisper Model
Pre-download model weights locally (defaults to `large-v3-turbo`):
```bash
python scripts/download_model.py --model large-v3-turbo
```
*(Options: `large-v3-turbo`, `large-v3`, `medium`, `small`, `base`, `tiny`)*

### 5. CLI Audio Transcription
Transcribe a single audio file with auto-detection or language specification:
```bash
# Auto-detect language (Bangla or English)
python scripts/transcribe.py path/to/audio.wav --language auto

# Specify Bangla ('bn')
python scripts/transcribe.py path/to/audio.wav --language bn
```

Transcribe an entire directory and export to JSON:
```bash
python scripts/transcribe.py data/test/audio/ --language auto --output results.json
```

### 6. Evaluate Test Set (WER & CER)
Compute Word Error Rate (WER) and Character Error Rate (CER) across your test split:
```bash
python scripts/evaluate.py --metadata data/test/metadata.csv --audio_dir data/test/audio --language auto
```
Outputs:
* Global corpus **WER** and **CER**
* Sample-by-sample analysis saved to `evaluation_results.csv`

---

## ⚡ Training & Fine-Tuning (GPU Cluster & Cloud)

When training on a server or GPU cluster (e.g. RTX 3090/4090, A10G, A100, H100):

### 1. Install GPU Requirements
```bash
pip install -r requirements_gpu.txt
```

### 2. Launch Training via GUI or CLI

#### Option A: Web UI Studio (Recommended)
Navigate to the **🏋️ Batched Training** tab, select your base model to auto-adapt parameters, and click **🚀 Launch Batched Training**. You can inspect live step loss and metrics in the streaming terminal or click **🛑 Abort Training** at any time.

#### Option B: Automated Shell Launcher
```bash
bash scripts/run_train_gpu.sh
```

#### Option C: Full CLI Customization
```bash
python scripts/train_whisper.py \
    --model_name_or_path "openai/whisper-large-v3-turbo" \
    --train_csv "data/train/metadata.csv" \
    --train_audio "data/train/audio" \
    --val_csv "data/val/metadata.csv" \
    --val_audio "data/val/audio" \
    --output_dir "./checkpoints/whisper_bangla_lora" \
    --language "bengali" \
    --task "transcribe" \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,v_proj" \
    --batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --gradient_checkpointing \
    --optim "adamw_torch" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "linear" \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_epochs 5 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --metric_for_best_model "wer"
```

---

## 🧬 Understanding LoRA (Low-Rank Adaptation)

This studio leverages **LoRA (Low-Rank Adaptation)** for fine-tuning Whisper without modifying the massive 800M frozen base model weights:

- **Mathematical Efficiency**: Instead of updating full $d \times k$ matrices ($1024 \times 1024 = 1,048,576$ parameters), LoRA injects two low-rank matrices $A$ and $B$ of rank $r=32$ ($2 \times 32 \times 1024 = 65,536$ parameters), training **< 2%** of the model.
- **Memory Footprint**: Reduces training VRAM/RAM from 24 GB down to **~6.5–8 GB**, enabling fine-tuning on consumer GPUs and Apple Silicon Unified Memory.
- **Compact Checkpoints**: Adapter weights are only **~30 MB** each (compared to ~3.1 GB for full models), conserving disk space.
- **Zero Catastrophic Forgetting**: Preserves Whisper's pre-trained multilingual foundation while adapting specifically to Bangla phonetics.
- **Zero Inference Overhead**: Adapter weights can be merged directly into the base weights ($W = W_0 + BA$) for production serving.

### Key LoRA Hyperparameters
- **Rank (`lora_r = 32`)**: Bottleneck dimension; 32 provides the sweet spot for speech acoustic feature adaptation.
- **Alpha (`lora_alpha = 64`)**: Scaling multiplier ($2 \times r$) determining adapter strength.
- **Dropout (`lora_dropout = 0.05`)**: Prevents overfitting on smaller training datasets.
- **Target Modules (`q_proj,v_proj`)**: Whisper's attention projection layers.

---

## 💻 Hardware & Platform Benchmarks

| Workload | Ubuntu (NVIDIA RTX 3090/4090) | Apple Silicon (Mac mini M4, 16 GB) | Linux / Intel CPU (8+ Cores) |
| :--- | :--- | :--- | :--- |
| **Whisper `turbo` Inference** | CUDA Float16 (Instant) | Apple Accelerate INT8 (20×–30× real-time) | CTranslate2 INT8 (8×–12× real-time) |
| **LoRA Fine-Tuning** | Full FP16 / BF16 (~7 GB VRAM) | Native Metal MPS (~7.5 GB RAM) | CPU Mini-Batches (Testing only) |
| **Audio Codec** | `/usr/bin/ffmpeg` via `apt` | `/opt/homebrew/bin/ffmpeg` via `brew` | System `ffmpeg` |
| **Recommended Mode** | CUDA + LoRA / Full | Metal MPS + LoRA (`batch_size=4`) | INT8 Inference & Benchmarks |

---

## 📄 License

This project is open source and available under the Apache License 2.0.
