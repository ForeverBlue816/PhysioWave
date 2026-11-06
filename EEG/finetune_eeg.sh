#!/bin/bash

# ============================================================================
# EEG Fine-tuning Script for PhysioWave
# ============================================================================
# Supervised fine-tuning on a downstream EEG task (default: TUAB, normal vs
# abnormal) using the shared finetune.py entry point, the same one ECG and EMG
# use. The HDF5 files come from EEG/tuab_finetune.py.
#
# Usage:
#   1. Build the labelled HDF5 files into $SCRATCH/bio/eeg/tuab:
#        python EEG/tuab_finetune.py --root /data/tuh_abnormal/v3.0.0 \
#               --out-dir $SCRATCH/bio/eeg/tuab
#   2. Point PRETRAINED_MODEL at a pretrained EEG checkpoint
#   3. Adjust num_classes for your task (TUAB=2, TUSL=4, BCI IV-2a=4, TUAR=5)
#   4. Run from the repository root:  bash EEG/finetune_eeg.sh
#
# On CINECA Leonardo ($FAST is set) paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/eeg/<task>  |  checkpoints/outputs $PW_FAST_ROOT/runs
#
# Note on checkpoints: EEG/pretrain_eeg.sh writes an extension-format
# checkpoint (physiowave.models.checkpoint), which finetune.py does not read.
# Use a legacy-format EEG checkpoint here, or run the extension's own
# evaluation instead:  bash scripts/run_tpami.sh eval --suite eeg
#
# Comments sit on their own lines on purpose: a `#` after a trailing `\` does
# not comment out anything, it breaks the continuation.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Number of GPUs to use for distributed training
NUM_GPUS="${NUM_GPUS:-4}"

# Downstream task; the labelled HDF5 files live at <data>/<TASK>/
TASK="${TASK:-tuab}"

# Storage layout comes from scripts/cineca_env.sh.
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/${TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_eeg_${TASK}}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${PW_CKPT_ROOT}/pretrain_eeg/best_model.pth}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/${TASK}_train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/${TASK}_val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/${TASK}_test.h5}"

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f} (build it with EEG/tuab_finetune.py)" >&2; exit 1; }
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

echo "EEG fine-tuning: task=${TASK} data=${DATA_DIR} out=${OUTPUT_DIR}"

# Launch distributed fine-tuning for EEG
#
# Architecture block must match the pretrained model:
#   in_channels 19   -> the 10-20 montage written by EEG/tuab_finetune.py
#   patch_size 32    -> 8 s window at 256 Hz; smaller patches than ECG because
#                       EEG rhythms of interest sit below 45 Hz
#   wavelet_names    -> longer-support families track EEG rhythms better than
#                       the short kernels used for EMG bursts
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels 19 \
  --max_level 4 \
  --wave_kernel_size 24 \
  --wavelet_names db4 db6 sym4 coif3 \
  --use_separate_channel \
  --patch_size 32 \
  --embed_dim 384 \
  --depth 8 \
  --num_heads 12 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  --use_pos_embed \
  --pos_embed_type 2d \
  --batch_size 16 \
  --epochs 20 \
  --lr 2e-4 \
  --min_lr 1e-6 \
  --weight_decay 1e-3 \
  --grad_clip 1.0 \
  --use_amp \
  --num_workers 8 \
  --world_size ${NUM_GPUS} \
  --scheduler cosine \
  --warmup_epochs 5 \
  --num_classes 2 \
  --pooling mean \
  --head_hidden_dim 1024 \
  --head_dropout 0.1 \
  --select_by "${SELECT_BY:-loss}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}"

echo "EEG fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"
echo "Training metrics: ${OUTPUT_DIR}/training_metrics.json"
