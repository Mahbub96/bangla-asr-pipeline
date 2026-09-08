# Bangla & English Automatic Speech Recognition (ASR)

An end-to-end Automatic Speech Recognition (ASR) pipeline tailored for **Bangla (বাংলা)** and **English** speech, powered by OpenAI's Whisper (`large-v3-turbo` / `large-v3`).

Designed with a dual-workflow architecture:
1. **Local Machine:** Fast CPU-optimized testing, transcription, dataset preparation, and WER/CER evaluation via `faster-whisper` (CTranslate2 INT8).
2. **GPU Training Cluster:** Scalable fine-tuning pipeline using Hugging Face `transformers`, `accelerate`, and Parameter-Efficient Fine-Tuning (`peft` / LoRA).

---

## 📁 Repository Structure

```text
Bangla ASR/
├── app.py                   # Gradio Web UI (microphone, file upload, benchmarks)
├── run_ui.sh                # Shell launcher for Web UI
├── start.sh                 # Quick CLI test runner script
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
│   ├── train_whisper.py     # GPU training / LoRA fine-tuning script
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

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install local testing dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web UI (Recommended)
You can launch the browser-based interface to record speech live, upload files, and view timestamps:
```bash
./run_ui.sh
```
Then open **http://localhost:7860** in your browser.

### 3. One-Click CLI Test Runner
Run transcription directly across the test audio folder:
```bash
./start.sh
```
*(Optional arguments: `./start.sh <target_path> <model_name> <language>`)*

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

## ⚡ Training on a Large Compute Machine (GPU)

When ready to train on a server or GPU cluster (e.g. RTX 3090/4090, A10G, A100):

### 1. Install GPU Requirements
```bash
pip install -r requirements_gpu.txt
```

### 2. Launch Training
Run the training script with LoRA (Parameter-Efficient Fine-Tuning):
```bash
bash scripts/run_train_gpu.sh
```

Or customize hyperparameters directly:
```bash
python scripts/train_whisper.py \
    --model_name_or_path "openai/whisper-large-v3-turbo" \
    --train_csv "data/train/metadata.csv" \
    --train_audio "data/train/audio" \
    --val_csv "data/val/metadata.csv" \
    --val_audio "data/val/audio" \
    --output_dir "./checkpoints/whisper_bangla" \
    --language "bengali" \
    --use_lora \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_epochs 5
```

---

## ⚙️ Hardware Recommendations
* **Local Machine:** Multi-core CPU (8+ threads), 8GB+ RAM. Runs INT8 inference fast.
* **Large Compute / Training:** Single or multi-GPU with ≥16GB VRAM (24GB+ recommended for full `large-v3-turbo` fine-tuning).
