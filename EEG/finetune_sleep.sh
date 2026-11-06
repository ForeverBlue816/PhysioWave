#!/bin/bash

# ============================================================================
# Sleep-EDF 5-class sleep staging
# ============================================================================
# Trains the folded wavelet transformer on Sleep-EDFx Sleep Cassette, on the
# preprocessing EEGPT uses, so the result sits next to their published row.
#
# Usage:
#   1. pip install braindecode
#   2. python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf
#   3. bash EEG/finetune_sleep.sh
#
# The comparison, stated up front so it is not read as more than it is:
#
#   EEGPT (NeurIPS'24)   BalAcc 0.6917 +- 0.0069   Kappa 0.6857 +- 0.0019
#
# That number is a *pretrained* encoder, frozen, with a four-layer transformer
# decoder head trained on top, scored on the validation fold it also selects
# on, averaged over ten folds. This script trains from scratch with every
# parameter free and -- under `--split holdout` -- scores a test set that
# nothing selected on. Beating it from scratch would be a real result;
# not beating it is the expected starting point and is the argument for
# pretraining, not against the architecture.
#
# Memory: after the need_weights fix in CrossScaleCAFFN, the whole model's
# fp32 activations at T=3000 are 0.66 GiB at batch 16, 1.27 at 32 and 2.51 at
# 64, against a 64 GiB card. Before it, the cross-scale attention alone wanted
# 17.2 GiB per decomposition level at batch 64 and the run died in step 1.
# BATCH_SIZE=64 is safe; 32 is the default only because more steps per epoch
# suits the 20-epoch budget.
#
# Architecture notes:
#   in_channels 2    Fpz-Cz and Pz-Oz, the two EEG derivations Sleep-EDF has
#   patch_size 50    0.5 s at 100 Hz, the timescale of a sleep spindle
#                    (0.5-2 s) and a K-complex (0.5-1.5 s). 3000 samples give
#                    60 time patches, so the folded model sees 2 x 60 = 120
#                    tokens where the unfolded one would see 480.
#   wave_init pad    every wavelet here is at most 16 taps and is zero-padded
#                    to the kernel rather than stretched. Stretching moves the
#                    half-band cutoff -- sym4 fitted to 16 taps by
#                    interpolation cuts at 0.203*pi instead of 0.5*pi and
#                    fails |H_lo|^2+|H_hi|^2 = 2 by 1.999 out of 2 -- so the
#                    "wavelet initialisation" would not be a filter bank.
#                    coif3 and bior4.4 are absent for that reason: coif3 is 18
#                    taps and does not fit, bior4.4 is biorthogonal and is not
#                    power-complementary by construction.
#
# Comments sit on their own lines on purpose: a `#` after a trailing `\` does
# not comment out anything, it breaks the continuation.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

NUM_GPUS="${NUM_GPUS:-4}"
TASK="${TASK:-sleep_edf}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/${TASK}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.h5}"

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_${TASK}}"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    [[ -f "${f}" ]] || {
        echo "ERROR: missing ${f}" >&2
        echo "       build it with: python EEG/sleep_edf_finetune.py --out-dir ${DATA_DIR}" >&2
        exit 1
    }
done

PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
if [[ -n "${PRETRAINED_MODEL}" && -f "${PRETRAINED_MODEL}" ]]; then
    PRETRAINED_ARG=(--pretrained_path "${PRETRAINED_MODEL}")
else
    [[ -n "${PRETRAINED_MODEL}" ]] && \
        echo "WARNING: no checkpoint at ${PRETRAINED_MODEL}; training from scratch." >&2
    PRETRAINED_ARG=()
fi
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Channel embedding. Off by default: none/none reproduces the run that existed
# before the feature, bit for bit, and every ablation row is measured against it.
#
# These are VALUES, passed straight through, not `${VAR:+--flag}` presence
# tests. That idiom is wrong for anything whose "off" value is a character:
# FOO=0 is a non-empty string, so `${FOO:+--foo}` turns the flag ON when the
# variable says 0.
# ---------------------------------------------------------------------------
# id/token -- variant C1 of the channel ablation, the EEGPT-style learned name
# embedding added to each patch token. This is the settled configuration, so it
# is the default here rather than something a caller has to remember. The
# ablation runner still sets both explicitly for every variant it scores,
# including C0's none/none, so this default does not reach that sweep.
#
# The split must carry channel metadata, which finetune.py requires as soon as
# the encoding is not 'none'. Writing it needs mne at --stage split time, so a
# split assembled in an environment without mne carries none and will now fail
# where it used to train. CHANNEL_ENCODING=none CHANNEL_INJECTION=none is the
# way back; rebuilding the split under $HOME/pwprep is the fix.
CHANNEL_ENCODING="${CHANNEL_ENCODING:-id}"
CHANNEL_INJECTION="${CHANNEL_INJECTION:-token}"
CHANNEL_EMBED_DIM="${CHANNEL_EMBED_DIM:-64}"
CHANNEL_FOLD_GATE_INIT="${CHANNEL_FOLD_GATE_INIT:-0.0}"
CHANNEL_TOKEN_GATE_INIT="${CHANNEL_TOKEN_GATE_INIT:-0.0}"

# ---------------------------------------------------------------------------
# The transformer block: all three switches ON.
#
#   norm=rmsnorm  ffn=swiglu  qk_norm=true
#
# They are independent -- `norm` and `ffn` were already passed as values above,
# and each stays its own ablation row -- but the block this model IS has all
# three. qk_norm puts an RMSNorm on q and k per head before attention
# (transformer_modules.py). It was off here only because nothing ever turned it
# on, not because anything chose it.
#
# NOTE: this differs from runs recorded before this change, which had
# qk_norm=False. Numbers from either side are not directly comparable; within
# one ablation every variant goes through this script and so agrees.
#
# A real boolean, not `${QK_NORM:+--qk_norm}`. That idiom is a PRESENCE test:
# QK_NORM=0 is a non-empty string and used to turn the flag ON, and the only
# way to say "off" was to unset the variable -- which stopped being a way to
# say anything the moment the default became on.
# ---------------------------------------------------------------------------
QK_NORM="${QK_NORM:-1}"
case "${QK_NORM}" in
    0|false|False|FALSE|no|off|"") QK_NORM_ARG=() ;;
    *)                             QK_NORM_ARG=(--qk_norm) ;;
esac

echo "Sleep-EDF fine-tuning: data=${DATA_DIR} out=${OUTPUT_DIR}"
echo "  channel: encoding=${CHANNEL_ENCODING} injection=${CHANNEL_INJECTION}" \
     "dim=${CHANNEL_EMBED_DIM} gates=(${CHANNEL_FOLD_GATE_INIT},${CHANNEL_TOKEN_GATE_INIT})" \
     "| block: norm=${NORM:-rmsnorm} ffn=${FFN:-swiglu}" \
     "qk_norm=$([[ ${#QK_NORM_ARG[@]} -gt 0 ]] && echo True || echo False)"

if [[ "${WARMUP_EPOCHS:-3}" -ge "${EPOCHS:-20}" ]]; then
    _w=$(( ${EPOCHS:-20} / 10 )); [[ "${_w}" -lt 1 ]] && _w=0
    echo "WARNING: WARMUP_EPOCHS >= EPOCHS; using ${_w} (the cosine decay would never start)." >&2
    WARMUP_EPOCHS="${_w}"
fi

# shellcheck disable=SC2086
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels "${IN_CHANNELS:-2}" \
  --max_level "${MAX_LEVEL:-3}" \
  --wave_kernel_size "${WAVE_KERNEL_SIZE:-16}" \
  --wavelet_names ${WAVELET_NAMES:-sym4 sym5 db6 sym8 db8} \
  --wave_init_mode "${WAVE_INIT_MODE:-pad}" \
  --use_separate_channel \
  --patch_size "${PATCH_SIZE:-50}" \
  --embed_dim "${EMBED_DIM:-384}" \
  --depth "${DEPTH:-6}" \
  --num_heads "${NUM_HEADS:-6}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --dropout "${DROPOUT:-0.1}" \
  --norm "${NORM:-rmsnorm}" \
  --ffn "${FFN:-swiglu}" \
  ${QK_NORM_ARG[@]+"${QK_NORM_ARG[@]}"} \
  --scale_fold "${SCALE_FOLD:-dynamic}" \
  ${FOLD_PATCH_LEN:+--fold_patch_len "${FOLD_PATCH_LEN}"} \
  --fold_synthesis "${FOLD_SYNTHESIS:-3}" \
  ${FOLD_SYNTHESIS_NORM:+--fold_synthesis_norm} \
  ${FOLD_SHARE_CHANNELS:+--fold_share_channels} \
  ${FOLD_SHRINKAGE:+--fold_shrinkage} \
  --fold_scale_dropout "${FOLD_SCALE_DROPOUT:-0.0}" \
  --fold_gamma "${FOLD_GAMMA:-0.1}" \
  --fold_kl "${FOLD_KL:-1e-3}" \
  --use_pos_embed \
  --pos_embed_type 2d \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-3e-4}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-1e-2}" \
  --grad_clip "${GRAD_CLIP:-1.0}" \
  --use_amp \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --scheduler "${SCHEDULER:-cosine}" \
  --warmup_epochs "${WARMUP_EPOCHS:-3}" \
  --num_classes "${NUM_CLASSES:-5}" \
  --pooling mean \
  --head_dropout "${HEAD_DROPOUT:-0.1}" \
  --head_hidden_dim "${HEAD_HIDDEN_DIM:-512}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.1}" \
  --channel_encoding "${CHANNEL_ENCODING}" \
  --channel_injection "${CHANNEL_INJECTION}" \
  --channel_embed_dim "${CHANNEL_EMBED_DIM}" \
  --channel_fold_gate_init "${CHANNEL_FOLD_GATE_INIT}" \
  --channel_token_gate_init "${CHANNEL_TOKEN_GATE_INIT}" \
  --select_by "${SELECT_BY:-balanced_acc}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA:-}

echo "Done. Results in ${OUTPUT_DIR}"
echo "  test_results.json   the number to report"
echo "  best_model.pth      inspect with scripts/inspect_checkpoint.py"
