#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export COPYFILE_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="${HF_HOME:-$PWD/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models}"

mkdir -p logs checkpoints/whisper_bangla_lora
find . -name '._*' -type f -delete

MAX_STEPS="${MAX_STEPS:-10}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LOG_FILE="logs/mac-mps-training.log"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run ./setup-mac-mps.sh first." >&2
  exit 1
fi

.venv/bin/python - <<'PY'
import torch
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
PY

echo "Starting native Mac MPS guarded training..."
echo "Log: ${LOG_FILE}"
echo "Follow: tail -f ${LOG_FILE}"

.venv/bin/python scripts/guarded_train.py \
  --backup_source models \
  --backup_dest backups/models \
  --backup_name current \
  --test_parquet 'data/sources/subakko/hf/Data/test-*.parquet' \
  --baseline_output checkpoints/baseline-old.csv \
  --after_output checkpoints/after-new.csv \
  --comparison_output checkpoints/model_comparison.json \
  --eval_model_before openai/whisper-large-v3-turbo \
  --eval_engine_before transformers \
  --eval_engine_after transformers \
  --models_dir models \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  -- \
  .venv/bin/python scripts/train_whisper.py \
    --model_name_or_path openai/whisper-large-v3-turbo \
    --language bengali \
    --task transcribe \
    --train_parquet 'data/sources/subakko/hf/Data/train-*.parquet' \
    --val_parquet 'data/sources/subakko/hf/Data/validation-*.parquet' \
    --streaming_parquet \
    --output_dir checkpoints/whisper_bangla_lora \
    --metrics_dir checkpoints/whisper_bangla_lora/metrics \
    --num_proc 2 \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 1e-4 \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --num_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --eval_steps 10 \
    --save_steps 10 \
    --logging_steps 1 \
    --save_total_limit 2 \
    --metric_for_best_model wer \
    --report_to tensorboard \
    --generation_max_length 225 \
    --generation_num_beams 1 \
    --dataloader_num_workers 0 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,v_proj \
    --gradient_checkpointing \
  2>&1 | tee "${LOG_FILE}"
