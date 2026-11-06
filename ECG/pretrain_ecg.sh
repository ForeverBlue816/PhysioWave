#!/bin/bash

# ============================================================================
# ECG Pretraining Script for PhysioWave
# ============================================================================
# Launches distributed pretraining for 12-lead ECG through the legacy
# top-level pretrain.py (ECG and EMG use it; EEG goes through
# physiowave.train.pretrain_main instead -- see EEG/pretrain_eeg.sh).
#
# Usage:
#   1. Build the HDF5 corpus:  python ECG/mimic_pretrain.py --out-dir $SCRATCH/bio/ecg/mimic_iv_ecg
#   2. Run from the repository root:  bash ECG/pretrain_ecg.sh
#
# Paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/ecg/<dataset>/*.h5   checkpoints $PW_FAST_ROOT/runs
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

# Corpus directory and the train/val split written by ECG/mimic_pretrain.py
DATASET="${DATASET:-mimic_iv_ecg}"
DATA_DIR="${DATA_DIR:-${PW_DATA_ECG}/${DATASET}}"

collect() {
    # Comma-separated list of HDF5 files matching a glob, or empty.
    local pattern="$1" out=""
    local f
    for f in ${pattern}; do
        [[ -e "${f}" ]] || continue
        out="${out:+${out},}${f}"
    done
    printf '%s' "${out}"
}

TRAIN_FILES="${TRAIN_FILES:-$(collect "${DATA_DIR}/*train*.h5")}"
VAL_FILES="${VAL_FILES:-$(collect "${DATA_DIR}/*val*.h5")}"

if [[ -z "${TRAIN_FILES}" ]]; then
    echo "ERROR: no *train*.h5 under ${DATA_DIR}." >&2
    echo "       Build it with ECG/mimic_pretrain.py, or set TRAIN_FILES/DATA_DIR." >&2
    exit 1
fi

# Output directory for checkpoints and logs
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_ecg}"
mkdir -p "${OUTPUT_DIR}"

echo "ECG pretraining: data=${DATA_DIR} out=${OUTPUT_DIR} gpus=${NUM_GPUS}"

# ---------------------------------------------------------------------------
# Architecture: 12-lead ECG, 3 wavelet levels, kernel 24, patch 64.
# Masking is frequency-guided at 70%, which suits the sharp QRS morphology.
# ---------------------------------------------------------------------------
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" pretrain.py \
  --train_files "${TRAIN_FILES}" \
  ${VAL_FILES:+--val_files "${VAL_FILES}"} \
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
  --max_length 2048 \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-2e-5}" \
  --weight_decay 1e-3 \
  --grad_accumulation_steps 2 \
  --grad_clip 1.0 \
  --use_amp \
  --scheduler cosine \
  --warmup_epochs 5 \
  --mask_ratio 0.7 \
  --masking_strategy frequency_guided \
  --importance_ratio 0.7 \
  --save_freq 10 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"

echo "ECG pretraining completed. Results saved to ${OUTPUT_DIR}"
