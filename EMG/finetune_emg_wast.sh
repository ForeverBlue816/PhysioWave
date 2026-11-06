#!/bin/bash

# ============================================================================
# EMG Fine-tuning through the TPAMI extension (WAST + TARE + factorized backbone)
# ============================================================================
# The counterpart to EMG/finetune_emg.sh, which drives the legacy
# BERTWaveletTransformer. Same data, same protocol, different architecture --
# that is what makes the two rows of the comparison table comparable.
#
# Token accounting on DB6 (14 channels, 512-sample window, patch 64, level 5):
#   legacy         (J+1)*C*S = 6*14*8 = 672 tokens
#   WAST           C*S       =   14*8 = 112
#   WAST + TARE    K*S       =    8*8 =  64      <- this script
#
# TARE also makes the token count independent of the channel count, so DB5's
# 16 channels and DB6's 14 produce the same 64 tokens.
#
# Usage:
#   TASK=db6 NUM_CLASSES=8 bash EMG/finetune_emg_wast.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

NUM_GPUS="${NUM_GPUS:-4}"
TASK="${TASK:-db6}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EMG}/${TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_wast_${TASK}}"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1

for f in train val test; do
    [[ -f "${DATA_DIR}/${f}.h5" ]] || {
        echo "ERROR: missing ${DATA_DIR}/${f}.h5" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

# The pretrained encoder is optional; without one this trains from scratch,
# which is the comparison point for the legacy from-scratch numbers.
PRETRAINED="${PRETRAINED:-}"
PRETRAINED_ARG=()
[[ -n "${PRETRAINED}" && -f "${PRETRAINED}" ]] && PRETRAINED_ARG=(--pretrained "${PRETRAINED}")

echo "EMG fine-tuning (WAST+TARE): task=${TASK} data=${DATA_DIR} out=${OUTPUT_DIR}"

"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" \
  -m physiowave.train.finetune_main \
  --config "${CONFIG:-pretrain/semg}" \
  --data-dir "${DATA_DIR}" \
  --num-classes "${NUM_CLASSES:-8}" \
  --output-dir "${OUTPUT_DIR}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --epochs "${EPOCHS:-60}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --warmup-epochs "${WARMUP_EPOCHS:-5}" \
  --label-smoothing "${LABEL_SMOOTHING:-0.1}" \
  --set model.backbone.dropout="${DROPOUT:-0.3}" \
       model.head_dropout="${HEAD_DROPOUT:-0.3}" \
       model.backbone.depth="${DEPTH:-4}" \
       model.embed_dim="${EMBED_DIM:-256}" \
       model.wast.embed_dim="${EMBED_DIM:-256}" \
       model.pooling="${POOLING:-mean}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --precision "${PRECISION:-bf16}" \
  --select-by "${SELECT_BY:-balanced_acc}" \
  --patience "${PATIENCE:-0}" \
  --min-delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --progress "${PROGRESS:-auto}" \
  ${EXTRA:-}

echo "Done. Results in ${OUTPUT_DIR} (results.json, history.json, best.pth)"
