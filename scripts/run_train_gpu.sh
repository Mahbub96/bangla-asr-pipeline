#!/usr/bin/env bash
# Training launcher script for running on large GPU machine / cluster.

set -e

echo "=== Bangla & English Whisper Fine-Tuning Setup ==="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "Detected GPU hardware:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Ensure NVIDIA CUDA drivers are loaded."
fi

# Optional: Install GPU requirements if needed
# pip install -r requirements_gpu.txt

# Run Whisper fine-tuning with LoRA
python3 scripts/train_whisper.py \
    --model_name_or_path "openai/whisper-large-v3-turbo" \
    --train_csv "data/train/metadata.csv" \
    --train_audio "data/train/audio" \
    --val_csv "data/val/metadata.csv" \
    --val_audio "data/val/audio" \
    --output_dir "./checkpoints/whisper_bangla_turbo" \
    --language "bengali" \
    --use_lora \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_epochs 5

echo "=== Training Finished ==="
