#!/bin/bash
# ============================================================================
# EEGPT's downstream benchmarks on the C1 pretrained encoder:
# BCIC-IV-2a, BCIC-IV-2b, KaggleERN.
#
#   TASK=bcic2a MODE=ft PRETRAINED=~/hf_release/eeg_c1_encoder.pth \
#       bash EEG/finetune_eegpt_bench.sh
#
# TASK    bcic2a | bcic2b | kaggleern
# MODE    ft       every parameter trains, from the pretrained encoder
#         probe    what pretraining supplied is frozen (freeze_scope
#                  pretrained): the montage's own fresh frontend and patcher,
#                  the pool and the head train -- the closest to EEGPT's
#                  frozen-encoder probe these montages allow
#         scratch  the same architecture, from random initialisation -- the
#                  control. Without it "ft" is a number, not a result.
# LR      overrides the config's lr; with LR set the output directory gains
#         _lr<LR>, so two learning rates of one mode do not share a directory.
# FOLD    which fold's split (default 0). One fold, not an average -- see the
#         note on protocol below.
# TAG     optional suffix on the output directory, to keep runs of different
#         encoders apart: TAG=best, TAG=final.
# PREP    pretrain (default) or eegpt -- BCIC only; see eegpt_bench_common.py.
#         PREP=eegpt adds "eegptprep" to the tag, so it never overwrites the
#         default run.
#
# PRETRAINED is an exported encoder (scripts/export_eeg_pretrained_encoder.py
# with no --route, or eeg_c1_encoder.pth from the release). A pretraining
# checkpoint -- best.pth, latest.pth -- is accepted too and exported first.
#
# ALL THREE TASKS AT ONCE: bash scripts/run_eegpt_bench.sh -- it downloads,
# then submits. This script is one run of one task.
#
# The split is built on first use from RAW_DIR, with the PRETRAINING
# preprocessing (EEG/eegpt_bench_common.py), into a directory named by the
# preprocessing version, so a batch job needs nothing prepared but the
# download -- which is the login node's job; compute nodes have no internet.
#
# PROTOCOL, stated so the number is not read as more than it is. EEGPT's rows
# are a mean over folds (nine LOSO folds on BCIC, four on KaggleERN) of a
# frozen-encoder probe scored on the subjects it also validated on. Here one
# fold is run with every parameter fine-tuned, its held-out subjects are a TEST
# set nothing selects on, and validation is a separate set of subjects. The
# comparison is therefore not like-for-like, and pessimistic for this model on
# the evaluation side.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"
# shellcheck disable=SC1091
source "$(pwd)/EEG/eegpt_bench_lib.sh"

TASK="${TASK:?set TASK=bcic2a|bcic2b|kaggleern}"
MODE="${MODE:-ft}"
FOLD="${FOLD:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

NCLS="$(eegpt_num_classes "${TASK}")" || exit 1
case "${MODE}" in
    ft|probe|scratch) ;;
    *) echo "ERROR: MODE must be ft, probe or scratch, not '${MODE}'" >&2; exit 1 ;;
esac

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1

RAW_DIR="${RAW_DIR:-$(eegpt_raw_dir "${TASK}")}"
DATA_DIR="${DATA_DIR:-$(eegpt_split_dir "${TASK}" "${FOLD}")}"
# TAG names WHICH encoder, so two of them -- best.pth and latest.pth, say --
# do not write into the same directory: TAG=final -> bcic2a_f0_ft_final.
RUN_TAG="$(eegpt_tag "${TASK}")"
[[ -n "${LR:-}" ]] && RUN_TAG="lr${LR}${RUN_TAG:+_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/eegpt_bench/${TASK}_f${FOLD}_${MODE}${RUN_TAG:+_${RUN_TAG}}}"
# pw_check_output_dir wants the parent to exist, and on a first run
# $PW_CKPT_ROOT/eegpt_bench does not. Checking the parent first keeps what the
# guard is for -- an unset PW_CKPT_ROOT puts the parent at /eegpt_bench, whose
# own parent is / -- and only then is the parent created.
pw_check_output_dir "$(dirname "${OUTPUT_DIR}")" || exit 1
mkdir -p "$(dirname "${OUTPUT_DIR}")"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1
mkdir -p "${OUTPUT_DIR}"
# finetune_main reuses an existing directory without clearing it. A rerun that
# dies before its first checkpoint would then leave the PREVIOUS run's
# results.json in place for the collector to report as this one's -- and the
# previous run here may well be the one with EEGPT-style preprocessing.
rm -f "${OUTPUT_DIR}/results.json" "${OUTPUT_DIR}/best.pth" \
      "${OUTPUT_DIR}/history.json" "${OUTPUT_DIR}/encoder.pth"

# --- the split ---------------------------------------------------------------- #
eegpt_build_split "${TASK}" "${FOLD}" "${DATA_DIR}" "${RAW_DIR}" || exit 1

# --- the encoder -------------------------------------------------------------- #
SET_ARGS=()
EXTRA_ARGS=()
if [[ "${MODE}" != "scratch" ]]; then
    PRETRAINED="${PRETRAINED:?MODE=${MODE} needs PRETRAINED=<encoder or checkpoint>}"
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: no file at ${PRETRAINED}" >&2; exit 1; }
    # An exported encoder carries route_id (None when routeless); a pretraining
    # checkpoint carries an optimizer and four frontends instead. The second is
    # exported here rather than handed to load_pretrained, which would take the
    # transformer and silently skip what it did not recognise.
    kind="$(python - "${PRETRAINED}" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("encoder" if "route_id" in ck else "checkpoint" if "model" in ck else "unknown")
PY
)"
    case "${kind}" in
        encoder)    ENCODER="${PRETRAINED}" ;;
        checkpoint) ENCODER="${OUTPUT_DIR}/encoder.pth"
                    echo "exporting the encoder from ${PRETRAINED}"
                    python scripts/export_eeg_pretrained_encoder.py \
                        --checkpoint "${PRETRAINED}" --output "${ENCODER}" ;;
        *) echo "ERROR: ${PRETRAINED} is neither an exported encoder nor a" \
                "pretraining checkpoint" >&2; exit 1 ;;
    esac
    SET_ARGS+=("model.eeg_c1.pretrained=${ENCODER}")
    # probe: freeze what pretraining SUPPLIED. These montages sit on no route,
    # so their frontend is built fresh; freezing it too would freeze random
    # filters and understate the probe.
    if [[ "${MODE}" == "probe" ]]; then
        EXTRA_ARGS+=(--freeze-encoder)
        SET_ARGS+=("model.eeg_c1.freeze_scope=${FREEZE_SCOPE:-pretrained}")
    fi
fi
if [[ -n "${LR:-}" ]]; then
    EXTRA_ARGS+=(--lr "${LR}")
fi
[[ -n "${EPOCHS:-}" ]]     && EXTRA_ARGS+=(--epochs "${EPOCHS}")
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
# SET: config overrides, space-separated, e.g. SET="model.eeg_c1.patch_samples=128".
# EXTRA: raw finetune_main flags, e.g. EXTRA="--patience 20".
# shellcheck disable=SC2206
[[ -n "${SET:-}" ]]   && SET_ARGS+=(${SET})
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && EXTRA_ARGS+=(${EXTRA})

echo "============================================================"
echo "  EEGPT benchmark  ${TASK}  fold ${FOLD}  mode ${MODE}"
echo "  data     ${DATA_DIR}  (preprocessing $(basename "${DATA_DIR}" | sed 's/.*_f[0-9]*_//'))"
echo "  encoder  ${ENCODER:-<none: random initialisation>}"
echo "  output   ${OUTPUT_DIR}"
echo "============================================================"

CMD=(-m physiowave.train.finetune_main
     --config "finetune/eeg_c1_${TASK}"
     --data-dir "${DATA_DIR}" --num-classes "${NCLS}"
     --output-dir "${OUTPUT_DIR}"
     --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-42}"
     --progress "${PROGRESS:-auto}"
     ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
[[ ${#SET_ARGS[@]} -gt 0 ]] && CMD+=(--set "${SET_ARGS[@]}")

# One GPU needs no process group; a rendezvous on a single device only adds a
# way to hang.
if [[ "${NUM_GPUS}" -le 1 ]]; then
    python "${CMD[@]}"
else
    "${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" "${CMD[@]}"
fi

echo "Done. ${OUTPUT_DIR}/results.json  (the 'test' block is the number to report)"
