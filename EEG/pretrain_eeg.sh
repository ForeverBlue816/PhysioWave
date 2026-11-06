#!/bin/bash

# ============================================================================
# EEG Pretraining Script for PhysioWave
# ============================================================================
# EEG pretraining runs through the extension pipeline
# (physiowave.train.pretrain_main), not the legacy pretrain.py used by ECG/EMG.
# EEG needs what the legacy path has no notion of: a montage with electrode
# coordinates, a reference-invariant spatial branch, and reference-consistency
# in the SSL objective. All of that lives in configs/pretrain/eeg.yaml.
#
# Usage:
#   1. Build the HDF5 corpus:  python EEG/tueg_pretrain.py --dataset tueg --root ...
#   2. Run from the repository root:  bash EEG/pretrain_eeg.sh
#
# On CINECA Leonardo ($FAST is set) the paths come from scripts/cineca_env.sh:
#   data $SCRATCH/bio/eeg/<dataset_id>  |  checkpoints/caches $PW_FAST_ROOT
# For a multi-node run use scripts/slurm/cineca_pretrain.sbatch instead; this
# script is the single-node / interactive-session entry point.
#
# Comments here sit on their own lines on purpose: a `#` after a trailing `\`
# does not comment out anything, it breaks the continuation.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Number of GPUs to use for distributed training
NUM_GPUS="${NUM_GPUS:-4}"

# Corpora to pretrain on. TUEG requires a signed data use agreement; nothing is
# downloaded automatically.
DATASETS="${DATASETS:-tueg,siena}"

# Mixture weights over the corpora, normalised internally (corpus size does not
# decide its share). Must line up with DATASETS.
WEIGHTS="${WEIGHTS:-[0.7,0.3]}"

# ---------------------------------------------------------------------------
# Storage layout. On Leonardo everything is resolved by scripts/cineca_env.sh;
# elsewhere it falls back to repository-relative directories.
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_eeg}"
mkdir -p "${OUTPUT_DIR}"

# One `data.roots.<id>` per corpus, each at <DATA_DIR>/<id>/*.h5.
# An empty corpus makes physiowave fall back to the synthetic smoke dataset, so
# a long run would quietly produce smoke-test numbers. Fail instead.
ROOT_ARGS=()
IFS=',' read -r -a DATASET_IDS <<< "${DATASETS}"
for id in "${DATASET_IDS[@]}"; do
    found=0
    for f in "${DATA_DIR}/${id}"/*.h5; do
        [[ -e "${f}" ]] && { found=1; break; }
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "ERROR: no *.h5 under ${DATA_DIR}/${id}." >&2
        echo "       Build it with EEG/tueg_pretrain.py, or set DATASETS/DATA_DIR." >&2
        exit 1
    fi
    ROOT_ARGS+=("data.roots.${id}=${DATA_DIR}/${id}")
done

# ---------------------------------------------------------------------------
# Everything below is a config override. The defaults come from
# configs/pretrain/eeg.yaml (which itself pulls in base + model/wast_tare):
#   - 10-10 64ch montage, 256 Hz
#   - TARE channel compression, SSL spatial branch enabled
#   - masked-raw + wavelet + reference-consistency + query-specialisation + covariance
# Override any of them with additional `key=value` pairs in EXTRA.
# ---------------------------------------------------------------------------
EXTRA="${EXTRA:-}"

OVERRIDES=(
  "data.datasets=[${DATASETS}]"
  "${ROOT_ARGS[@]}"
  "data.weights=${WEIGHTS}"
  "data.manifest_dir=${PW_MANIFEST_DIR}"

  # Preprocessing is applied at load time, so one cached corpus serves several
  # filter settings. notch_freq is 60.0 for US recordings (TUH), 50.0 in Europe.
  "data.preprocess.target_sampling_rate=256.0"
  "data.preprocess.notch_freq=60.0"
  "data.preprocess.bandpass=[0.5,45.0]"
  "data.preprocess.normalize=zscore"
  "data.preprocess.clip_sigma=20.0"
  "data.preprocess.cache_dir=${PW_PREP_CACHE}"
  "model.spatial.ssl.cache_dir=${PW_SSL_CACHE}"

  # Training
  "train.epochs=${EPOCHS:-50}"
  "train.batch_size=${BATCH_SIZE:-16}"
  "train.num_workers=${NUM_WORKERS:-8}"
)

echo "EEG pretraining: datasets=${DATASETS} data=${DATA_DIR} out=${OUTPUT_DIR}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  "${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" \
    -m physiowave.train.pretrain_main \
    --config pretrain/eeg \
    --output-dir "${OUTPUT_DIR}" \
    --resume auto \
    --set "${OVERRIDES[@]}" ${EXTRA}
else
  python -m physiowave.train.pretrain_main \
    --config pretrain/eeg \
    --output-dir "${OUTPUT_DIR}" \
    --resume auto \
    --set "${OVERRIDES[@]}" ${EXTRA}
fi

echo "EEG pretraining completed. Results saved to ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/best.pth"
