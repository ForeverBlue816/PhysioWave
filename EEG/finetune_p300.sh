#!/bin/bash

# ============================================================================
# PhysioP300 binary P300 detection, leave-one-subject-out
# ============================================================================
# Trains the folded wavelet transformer on PhysioNet ERP-BCI, on the
# preprocessing EEGPT uses, so the result sits next to their published row.
#
# Usage:
#   1. source scripts/cineca_env.sh               # first: exports PW_DATA_EEG
#      source $HOME/pwprep/bin/activate           # second: the preparation env
#      python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci
#      python EEG/physio_p300_finetune.py --edf-dir $PW_DATA_EEG/erpbci \
#          --out-dir $PW_DATA_EEG/p300_f0 --fold 0
#   2. deactivate                                 # back to the training env
#      DATA_DIR=$PW_DATA_EEG/p300_f0 bash EEG/finetune_p300.sh
#
# The comparison, stated up front so it is not read as more than it is:
#
#   EEGPT (NeurIPS'24)  BalAcc 0.6502+-0.0063  Kappa 0.2999+-0.0139
#                       AUROC  0.7168+-0.0051
#
# That is a *pretrained* encoder, frozen, with a linear probe on top, scored on
# the held-out subject it also selects on -- their LOSO loop builds only
# train_dataset and valid_dataset and passes `callbacks = [lr_monitor]`, so
# there is no test set and no checkpoint callback. This script trains from
# scratch with every parameter free and scores a subject that selected nothing.
# Beating it from scratch would be a real result; not beating it is the
# expected starting point and is the argument for pretraining, not against the
# architecture.
#
# Nine folds, not one. A single held-out subject is 240 epochs per run times
# ~20 runs, and between-subject variance on ERP tasks is the dominant term --
# EEGPT's own +-0.0139 on kappa is across folds. One fold is a pilot, not a
# number. `bash EEG/finetune_p300.sh` runs FOLD; loop it 0..8 and average.
#
# Architecture notes:
#   in_channels 64   the converter's default montage: every EEG electrode
#                    erpbci records. It MUST match the split -- finetune.py
#                    checks it against the file's channel metadata and refuses
#                    a mismatch -- so a split built with --channels 62 or 58
#                    needs the matching IN_CHANNELS here.
#   patch_size 64    250 ms at 256 Hz, matching their d=64. The window is 512
#                    samples, so 8 time patches; the fold collapses the 4
#                    wavelet scales back onto 62 rows, giving 62 x 8 = 496
#                    tokens. WITHOUT the fold it would be (3+1) x 62 = 248 rows
#                    and 1984 tokens, which is the argument for scale_fold on
#                    a 62-channel montage rather than a 2-channel one.
#   class_weight     balanced. A Donchin speller flashes 6 rows and 6 columns
#                    and 2 contain the target, so the positive rate is exactly
#                    1/6. Unweighted, argmax at 0.5 collapses onto the majority
#                    and balanced accuracy sits at 0.5 while plain accuracy
#                    reports 0.833.
#   select_by auroc  what EEGPT monitors for binary tasks (their Appendix D),
#                    and the metric least disturbed by the class imbalance.
#   epochs 15        was 30. Sleep-EDF peaked at epoch 4 twice -- once out of
#                    20 and once out of 10, under different warmup and LR
#                    schedules -- and P300's training fold is 17.6k windows
#                    against Sleep-EDF's 97k, so it has less to fit and
#                    saturates no later. warmup is 2, not 3: at 15 epochs a
#                    3-epoch warmup is a fifth of the budget.
#
# Fold 0 baseline at the previous defaults (30 epochs, 58 channels), for
# anything compared against it later:
#
#   AUROC 0.7463   BalAcc 0.6604   Kappa 0.2837
#
# That row is 58 electrodes. A 62-channel split is a different input and is not
# a drop-in comparison with it -- to reproduce the row exactly, build the split
# with --channels 58 and pass IN_CHANNELS=58.
#
# against EEGPT's 0.7168 / 0.6502 / 0.2999. Ahead on the two threshold-free
# metrics and behind on kappa, which is the signature of a good ranking read at
# the wrong operating point rather than of a worse model.
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
FOLD="${FOLD:-0}"
TASK="${TASK:-p300}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/${TASK}_f${FOLD}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.h5}"

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_${TASK}_f${FOLD}}"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    [[ -f "${f}" ]] || {
        echo "ERROR: missing ${f}" >&2
        echo "       build it with:" >&2
        echo "         python EEG/physio_p300_finetune.py \\" >&2
        echo "             --edf-dir \$PW_DATA_EEG/erpbci --out-dir ${DATA_DIR} --fold ${FOLD}" >&2
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
# Channel embedding. Off by default: none/none is the run that existed before
# the feature, and every ablation row is measured against it.
#
# This montage is MONOPOLAR -- each channel is one electrode against a common
# reference, not a derivation. The `signed` encoder sees both endpoint indices
# pointing at the same electrode, so its direction term is exactly zero and what
# remains is the electrode's position, marked as monopolar. Nothing here claims
# a direction that the montage does not have.
#
# These are VALUES, passed straight through, not `${VAR:+--flag}` presence
# tests: FOO=0 is a non-empty string and that idiom would read it as "on".
# ---------------------------------------------------------------------------
# id/token -- variant C1 of the channel ablation, the EEGPT-style learned name
# embedding added to each patch token. This is the settled configuration, so it
# is the default here rather than something a caller has to remember. The
# ablation runner still sets both explicitly for every variant it scores,
# including C0's none/none, so this default does not reach that sweep.
#
# The split must carry channel metadata, which finetune.py requires as soon as
# the encoding is not 'none'. EEG/physio_p300_finetune.py --stage split writes
# it; a split built by something that did not will now fail where it used to
# train. CHANNEL_ENCODING=none CHANNEL_INJECTION=none is the way back.
CHANNEL_ENCODING="${CHANNEL_ENCODING:-id}"
CHANNEL_INJECTION="${CHANNEL_INJECTION:-token}"
CHANNEL_EMBED_DIM="${CHANNEL_EMBED_DIM:-64}"
CHANNEL_FOLD_GATE_INIT="${CHANNEL_FOLD_GATE_INIT:-0.0}"
CHANNEL_TOKEN_GATE_INIT="${CHANNEL_TOKEN_GATE_INIT:-0.0}"

# The three block switches, all on, matching EEG/finetune_sleep.sh. A real
# boolean: `${QK_NORM:+--qk_norm}` was a presence test in which QK_NORM=0 meant
# on, and unsetting was the only way to say off.
#
# NOTE: runs recorded before this change had qk_norm=False and are not directly
# comparable. Within one ablation every variant goes through this script.
QK_NORM="${QK_NORM:-1}"
case "${QK_NORM}" in
    0|false|False|FALSE|no|off|"") QK_NORM_ARG=() ;;
    *)                             QK_NORM_ARG=(--qk_norm) ;;
esac

echo "PhysioP300 fine-tuning: fold=${FOLD} data=${DATA_DIR} out=${OUTPUT_DIR}"
echo "  channel: encoding=${CHANNEL_ENCODING} injection=${CHANNEL_INJECTION}" \
     "dim=${CHANNEL_EMBED_DIM} gates=(${CHANNEL_FOLD_GATE_INIT},${CHANNEL_TOKEN_GATE_INIT})"
echo "  block: norm=${NORM:-rmsnorm} ffn=${FFN:-swiglu}" \
     "qk_norm=$([[ ${#QK_NORM_ARG[@]} -gt 0 ]] && echo True || echo False)"

if [[ "${WARMUP_EPOCHS:-2}" -ge "${EPOCHS:-15}" ]]; then
    _w=$(( ${EPOCHS:-15} / 10 )); [[ "${_w}" -lt 1 ]] && _w=0
    echo "WARNING: WARMUP_EPOCHS >= EPOCHS; using ${_w} (the cosine decay would never start)." >&2
    WARMUP_EPOCHS="${_w}"
fi

# shellcheck disable=SC2086
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels "${IN_CHANNELS:-64}" \
  --max_level "${MAX_LEVEL:-3}" \
  --wave_kernel_size "${WAVE_KERNEL_SIZE:-16}" \
  --wavelet_names ${WAVELET_NAMES:-sym4 sym5 db6 sym8 db8} \
  --wave_init_mode "${WAVE_INIT_MODE:-pad}" \
  --use_separate_channel \
  --patch_size "${PATCH_SIZE:-64}" \
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
  --epochs "${EPOCHS:-15}" \
  --lr "${LR:-3e-4}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-1e-2}" \
  --grad_clip "${GRAD_CLIP:-1.0}" \
  --use_amp \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --scheduler "${SCHEDULER:-cosine}" \
  --warmup_epochs "${WARMUP_EPOCHS:-2}" \
  --num_classes "${NUM_CLASSES:-2}" \
  --class_weight "${CLASS_WEIGHT:-balanced}" \
  --pooling mean \
  --head_dropout "${HEAD_DROPOUT:-0.1}" \
  --head_hidden_dim "${HEAD_HIDDEN_DIM:-512}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.1}" \
  --select_by "${SELECT_BY:-auroc}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --channel_encoding "${CHANNEL_ENCODING}" \
  --channel_injection "${CHANNEL_INJECTION}" \
  --channel_embed_dim "${CHANNEL_EMBED_DIM}" \
  --channel_fold_gate_init "${CHANNEL_FOLD_GATE_INIT}" \
  --channel_token_gate_init "${CHANNEL_TOKEN_GATE_INIT}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA:-}

echo "Done. Results in ${OUTPUT_DIR}"
echo "  test_results.json   the number to report"
echo "  best_model.pth      inspect with scripts/inspect_checkpoint.py"
