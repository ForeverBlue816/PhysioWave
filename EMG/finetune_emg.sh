#!/bin/bash

# ============================================================================
# EMG Fine-tuning Script for PhysioWave
# ============================================================================
# Supervised fine-tuning on a downstream sEMG task (default: EPN-612 hand
# gestures, 6 classes) on top of a pretrained EMG encoder.
#
# Usage:
#   1. Build the labelled HDF5 files:
#        python EMG/epn_finetune.py --out-dir $SCRATCH/bio/emg/epn612
#   2. Run from the repository root:  bash EMG/finetune_emg.sh
#
# Paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/emg/<task>   checkpoints/outputs $PW_FAST_ROOT/runs
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

TASK="${TASK:-epn612}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EMG}/${TASK}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.h5}"

# Pretrained EMG model checkpoint
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${PW_CKPT_ROOT}/pretrain_emg/best_model.pth}"

# Output directory for fine-tuning results
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_emg_${TASK}}"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1

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

echo "EMG fine-tuning: task=${TASK} data=${DATA_DIR} out=${OUTPUT_DIR}"

# warmup_epochs >= epochs makes WarmupCosineSchedule spend the entire run
# ramping up, so the cosine decay never happens and the result says nothing
# about the learning rate that was requested.
if [[ "${WARMUP_EPOCHS:-5}" -ge "${EPOCHS:-5}" ]]; then
    _w=$(( ${EPOCHS:-5} / 10 )); [[ "${_w}" -lt 1 ]] && _w=0
    echo "WARNING: WARMUP_EPOCHS=${WARMUP_EPOCHS:-5} >= EPOCHS=${EPOCHS:-5}; " \
         "using ${_w} instead (the cosine decay would otherwise never start)." >&2
    WARMUP_EPOCHS="${_w}"
fi


# ---------------------------------------------------------------------------
# The architecture block must match EMG/pretrain_emg.sh exactly.
# ---------------------------------------------------------------------------
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels "${IN_CHANNELS:-8}" \
  --max_level "${MAX_LEVEL:-3}" \
  --wave_kernel_size "${WAVE_KERNEL_SIZE:-16}" \
  --wavelet_names sym4 sym5 db6 coif3 bior4.4 \
  --use_separate_channel \
  --patch_size "${PATCH_SIZE:-64}" \
  --embed_dim "${EMBED_DIM:-256}" \
  --depth "${DEPTH:-6}" \
  --num_heads "${NUM_HEADS:-8}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --dropout "${DROPOUT:-0.1}" \
  --norm "${NORM:-layernorm}" \
  --ffn "${FFN:-mlp}" \
  ${QK_NORM:+--qk_norm} \
  --wave_init_mode "${WAVE_INIT_MODE:-interp}" \
  --scale_fold "${SCALE_FOLD:-none}" \
  ${FOLD_PATCH_LEN:+--fold_patch_len "${FOLD_PATCH_LEN}"} \
  --fold_synthesis "${FOLD_SYNTHESIS:-0}" \
  ${FOLD_SYNTHESIS_NORM:+--fold_synthesis_norm} \
  ${FOLD_SHARE_CHANNELS:+--fold_share_channels} \
  ${FOLD_SHRINKAGE:+--fold_shrinkage} \
  --fold_scale_dropout "${FOLD_SCALE_DROPOUT:-0.0}" \
  --fold_gamma "${FOLD_GAMMA:-0.1}" \
  --fold_kl "${FOLD_KL:-0.0}" \
  --use_pos_embed \
  --pos_embed_type 2d \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-5}" \
  --lr "${LR:-2e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-3}" \
  --grad_clip "${GRAD_CLIP:-1.0}" \
  --use_amp \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --scheduler "${SCHEDULER:-cosine}" \
  --warmup_epochs "${WARMUP_EPOCHS:-5}" \
  --num_classes "${NUM_CLASSES:-6}" \
  --pooling mean \
  --head_dropout "${HEAD_DROPOUT:-0.1}" \
  --head_hidden_dim "${HEAD_HIDDEN_DIM:-512}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.1}" \
  --select_by "${SELECT_BY:-loss}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA:-}

echo "Fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"
