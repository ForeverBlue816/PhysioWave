#!/bin/bash
# ============================================================================
# One ECG C1 downstream run: fine-tune, probe, or the from-scratch control.
#
#   TASK=ptbxl    MODE=ft      bash ECG/finetune_ecg_c1.sh
#   TASK=cpsc2018 MODE=probe   bash ECG/finetune_ecg_c1.sh
#   TASK=chapman  MODE=scratch bash ECG/finetune_ecg_c1.sh
#
# MODE
#   ft       the pretrained encoder, every weight trained      (the main row)
#   probe    the pretrained encoder frozen, only the head trained
#            (what the representation alone is worth)
#   scratch  the same model, no weights loaded                 (the control)
#
# TASK: ptbxl (paper: accuracy), ptbxl_super, cpsc2018, chapman (paper: F1)
#
# The split comes from ECG/ecg_downstream_prep.py; build it once with
# scripts/slurm/cineca_ecg_downstream.sbatch (which does it if missing) or
# by hand. A raw pretraining checkpoint is exported here to the L12_500
# encoder -- frontend, patcher, channel encoder and transformer -- which a
# 12-lead 500 Hz 10 s window loads whole.
#
# ENVIRONMENT
#   PRETRAINED   pretraining checkpoint or exported encoder
#                ($PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth)
#   DATA_DIR     the split   ($ECG_ROOT/downstream/c1/$TASK)
#   OUTPUT_DIR   ($PW_CKPT_ROOT/ecg_downstream/${TASK}_${MODE}[_$TAG])
#   TAG          suffix for a variant (an lr, a seed)
#   LR EPOCHS BATCH_SIZE SEED   override the config's train: block
#   SET          extra --set key=value pairs, space separated
#   NUM_GPUS     1 (these sets are small; one GPU per run, several in parallel)
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(pwd)/scripts/cineca_env.sh"

TASK="${TASK:?set TASK=ptbxl|ptbxl_super|cpsc2018|chapman}"
MODE="${MODE:-ft}"
NUM_GPUS="${NUM_GPUS:-1}"
case "${TASK}" in ptbxl|ptbxl_super|cpsc2018|chapman) ;;
    *) echo "ERROR: unknown TASK '${TASK}'" >&2; exit 1 ;; esac
case "${MODE}" in ft|probe|scratch) ;;
    *) echo "ERROR: MODE must be ft, probe or scratch, not '${MODE}'" >&2; exit 1 ;; esac
[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1

ECG_ROOT="${ECG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/ecg}"
DATA_DIR="${DATA_DIR:-${ECG_ROOT}/downstream/c1/${TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/ecg_downstream/${TASK}_${MODE}${TAG:+_${TAG}}}"
if [[ ! -f "${DATA_DIR}/split.json" ]]; then
    echo "ERROR: no split at ${DATA_DIR}. Build it first:" >&2
    echo "  python ECG/ecg_downstream_prep.py --task ${TASK} \\" >&2
    echo "      --raw-dir ${ECG_ROOT}/downstream/raw/<set> --out-dir ${DATA_DIR} --jobs 16" >&2
    echo "  (scripts/slurm/cineca_ecg_downstream.sbatch does it when missing)" >&2
    exit 1
fi
NCLS="$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['classes']))" \
        "${DATA_DIR}/split.json")"

pw_check_output_dir "$(dirname "${OUTPUT_DIR}")" || exit 1
mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/results.json" "${OUTPUT_DIR}/best.pth" "${OUTPUT_DIR}/history.json"

SET_ARGS=()
EXTRA_ARGS=()
if [[ "${MODE}" != "scratch" ]]; then
    PRETRAINED="${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_ecg_c1_moe/best.pth}"
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: no file at ${PRETRAINED}" >&2; exit 1; }
    kind="$(python - "${PRETRAINED}" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("encoder" if "route_id" in ck else "checkpoint" if "model" in ck else "unknown")
PY
)"
    case "${kind}" in
        encoder)    ENCODER="${PRETRAINED}" ;;
        checkpoint) ENCODER="${OUTPUT_DIR}/encoder_L12_500.pth"
                    python scripts/export_eeg_pretrained_encoder.py \
                        --checkpoint "${PRETRAINED}" --route L12_500 --output "${ENCODER}" ;;
        *) echo "ERROR: ${PRETRAINED} is neither an encoder nor a checkpoint" >&2; exit 1 ;;
    esac
    SET_ARGS+=("model.ecg_c1.pretrained=${ENCODER}")
    [[ "${MODE}" == "probe" ]] && EXTRA_ARGS+=(--freeze-encoder)
fi
[[ -n "${LR:-}" ]]         && EXTRA_ARGS+=(--lr "${LR}")
[[ -n "${EPOCHS:-}" ]]     && EXTRA_ARGS+=(--epochs "${EPOCHS}")
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
[[ -n "${SET:-}" ]]        && SET_ARGS+=(${SET})
[[ -n "${EXTRA:-}" ]]      && EXTRA_ARGS+=(${EXTRA})

echo "============================================================"
echo "  ECG C1 downstream  ${TASK}  mode ${MODE}  (${NCLS} classes)"
echo "  data     ${DATA_DIR}"
echo "  encoder  ${ENCODER:-<none: random initialisation>}"
echo "  output   ${OUTPUT_DIR}"
echo "============================================================"

CMD=(-m physiowave.train.finetune_main
     --config "finetune/ecg_c1_${TASK}"
     --data-dir "${DATA_DIR}" --num-classes "${NCLS}"
     --output-dir "${OUTPUT_DIR}"
     --num-workers "${NUM_WORKERS:-4}" --seed "${SEED:-42}"
     --progress "${PROGRESS:-auto}"
     ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
[[ ${#SET_ARGS[@]} -gt 0 ]] && CMD+=(--set "${SET_ARGS[@]}")

if [[ "${NUM_GPUS}" -le 1 ]]; then
    python "${CMD[@]}"
else
    "${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" "${CMD[@]}"
fi
echo "Done. ${OUTPUT_DIR}/results.json (its 'test' block is the number to report)"
