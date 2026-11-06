#!/bin/bash

# ============================================================================
# EMG Pretraining Script for PhysioWave
# ============================================================================
# Launches distributed pretraining for limb/skeletal sEMG through the legacy
# top-level pretrain.py.  Facial EMG must not enter this corpus -- it has
# different generators, bandwidth and artefact structure.
#
# Usage:
#   1. Build the HDF5 corpus:  python EMG/db6_pretrain.py --out-dir $SCRATCH/bio/emg/ninapro_db6
#   2. Run from the repository root:  bash EMG/pretrain_emg.sh
#
# Paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/emg/<dataset>/*.h5   checkpoints $PW_FAST_ROOT/runs
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

DATASET="${DATASET:-ninapro_db6}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EMG}/${DATASET}}"

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
    echo "       Build it with EMG/db6_pretrain.py, or set TRAIN_FILES/DATA_DIR." >&2
    exit 1
fi

# Output directory for checkpoints and logs
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_emg}"
mkdir -p "${OUTPUT_DIR}"

echo "EMG pretraining: data=${DATA_DIR} out=${OUTPUT_DIR} gpus=${NUM_GPUS}"

# ---------------------------------------------------------------------------
# Architecture: 8 electrodes, kernel 16 and shorter-support wavelet families
# than ECG/EEG -- sEMG bursts are broadband and brief, so long kernels smear
# onset timing.  Masking at 60%, slightly lower than ECG.
# ---------------------------------------------------------------------------
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" pretrain.py \
  --train_files "${TRAIN_FILES}" \
  ${VAL_FILES:+--val_files "${VAL_FILES}"} \
  --in_channels "${IN_CHANNELS:-8}" \
  --max_level 3 \
  --wave_kernel_size 16 \
  --wavelet_names sym4 sym5 db6 coif3 bior4.4 \
  --use_separate_channel \
  --patch_size "${PATCH_SIZE:-64}" \
  --embed_dim 256 \
  --depth 6 \
  --num_heads 8 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  --use_pos_embed \
  --pos_embed_type 2d \
  --max_length 2048 \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --epochs "${EPOCHS:-30}" \
  --lr "${LR:-1e-4}" \
  --weight_decay 0.01 \
  --grad_accumulation_steps 2 \
  --grad_clip 1.0 \
  --use_amp \
  --scheduler cosine \
  --warmup_epochs 10 \
  --mask_ratio 0.6 \
  --masking_strategy frequency_guided \
  --importance_ratio 0.6 \
  --save_freq 10 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"

echo "EMG pretraining completed. Results saved to ${OUTPUT_DIR}"
echo "Use the best_model.pth from ${OUTPUT_DIR} for downstream fine-tuning tasks"
