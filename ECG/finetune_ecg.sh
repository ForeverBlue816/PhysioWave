#!/bin/bash

# ============================================================================
# ECG Fine-tuning Script for PhysioWave
# ============================================================================
# Single-label ECG classification (default: PTB-XL, 5 diagnostic superclasses)
# on top of a pretrained ECG encoder.
#
# Usage:
#   1. Build the labelled HDF5 files:
#        python ECG/ptbxl_finetune.py --out-dir $SCRATCH/bio/ecg/ptbxl
#   2. Run from the repository root:  bash ECG/finetune_ecg.sh
#
# Paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/ecg/<task>   checkpoints/outputs $PW_FAST_ROOT/runs
#
# Comments sit on their own lines on purpose: a `#` after a trailing `\` does
# not comment out anything, it truncates the command at that point.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

# Number of GPUs to use for distributed training
NUM_GPUS="${NUM_GPUS:-4}"

# Downstream task; labelled HDF5 files live at $SCRATCH/bio/ecg/<TASK>/
TASK="${TASK:-ptbxl}"
DATA_DIR="${DATA_DIR:-${PW_DATA_ECG}/${TASK}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.h5}"

# Pretrained ECG model checkpoint
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${PW_CKPT_ROOT}/pretrain_ecg/best_model.pth}"

# Output directory for fine-tuning results
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_ecg_${TASK}}"

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

# The checkpoint is optional: finetune.py falls back to random init, which is
# how you smoke-test the pipeline before any encoder has been pretrained.
if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
    echo "WARNING: no checkpoint at ${PRETRAINED_MODEL}; training from scratch." >&2
    PRETRAINED_ARG=()
else
    PRETRAINED_ARG=(--pretrained_path "${PRETRAINED_MODEL}")
fi
mkdir -p "${OUTPUT_DIR}"

echo "ECG fine-tuning: task=${TASK} data=${DATA_DIR} out=${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# The architecture block must match the pretrained model exactly; the wavelet
# families, kernel size and patch size are the ones ECG/pretrain_ecg.sh used.
# ---------------------------------------------------------------------------
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels 12 \
  --max_level 3 \
  --wave_kernel_size 24 \
  --wavelet_names db4 db6 sym4 coif2 \
  --use_separate_channel \
  --patch_size 64 \
  --embed_dim 384 \
  --depth 8 \
  --num_heads 12 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  --use_pos_embed \
  --pos_embed_type 2d \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-2e-4}" \
  --min_lr 1e-6 \
  --weight_decay 1e-3 \
  --grad_clip 1.0 \
  --use_amp \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --scheduler cosine \
  --warmup_epochs 5 \
  --num_classes "${NUM_CLASSES:-5}" \
  --pooling mean \
  --head_hidden_dim 1024 \
  --head_dropout 0.1 \
  --select_by "${SELECT_BY:-loss}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}"

echo "ECG fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"
echo "Training metrics: ${OUTPUT_DIR}/training_metrics.json"
