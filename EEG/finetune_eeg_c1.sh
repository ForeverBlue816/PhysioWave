#!/bin/bash
# ============================================================================
# One EEG C1 downstream run on Sleep-EDF or PhysioNet P300: fine-tune, probe,
# or the from-scratch control (physiowave/train/finetune_main.py).
#
#   TASK=sleep_edf   MODE=ft      bash EEG/finetune_eeg_c1.sh
#   TASK=physio_p300 MODE=probe   LR=1e-3 bash EEG/finetune_eeg_c1.sh
#   TASK=sleep_edf   MODE=scratch bash EEG/finetune_eeg_c1.sh
#
#   ft       the pretrained encoder, every parameter trains, layer-wise lr
#            decay 0.75 over the Transformer (the recipe of the sEMG runs)
#   probe    what pretraining supplied is frozen (freeze_scope pretrained);
#            the montage's own fresh frontend, the pool and the head train
#   scratch  the same model, no weights loaded                 (the control)
#
#   TASK         sleep_edf (5 stages, Fpz-Cz + Pz-Oz at 100 Hz, 30 s epochs,
#                configs/finetune/eeg_c1_sleep.yaml) | physio_p300 (target vs
#                non-target, configs/finetune/eeg_c1_p300.yaml)
#   DATA_DIR     the split: $PW_DATA_EEG/sleep_edf | $PW_DATA_EEG/p300_f0
#                (built by EEG/sleep_edf_finetune.py / EEG/physio_p300_finetune.py;
#                one subject-disjoint split, not folds)
#   PRETRAINED   pretraining checkpoint or exported encoder
#                ($PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth)
#   ROUTE        place the montage on a pretraining route (E64_256 for P300):
#                the pretrained wavelet frontend and patcher load as well, and a
#                probe freezes the encoder end to end, as EEGPT's does. Unset,
#                the montage gets its own fresh frontend (the configs' default:
#                only the Transformer transfers). Empty route slots stay zero.
#   CLASS_WEIGHT none | balanced (cross-entropy weighted N / (K n_c))
#   ENC_LR_SCALE ft only: the pretrained parameters at this multiple of LR,
#                the fresh frontend and head at LR (e.g. LR=1e-3 ENC_LR_SCALE=0.1)
#   LR, EPOCHS, LAYER_DECAY, SEED, SET, EXTRA, TAG, OUTPUT_DIR
#   OUTPUT_DIR   $PW_CKPT_ROOT/eeg_downstream/<task>_<mode>[_<tag>]
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

TASK="${TASK:?set TASK=sleep_edf|physio_p300}"
MODE="${MODE:-ft}"
NUM_GPUS="${NUM_GPUS:-1}"
case "${TASK}" in
    sleep_edf)   CONFIG=finetune/eeg_c1_sleep; NCLS=5; RATE=100
                 DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/sleep_edf}"
                 BUILD="python EEG/sleep_edf_finetune.py --out-dir ${PW_DATA_EEG}/sleep_edf --split holdout" ;;
    physio_p300) CONFIG=finetune/eeg_c1_p300; NCLS=2; RATE=256
                 DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/p300_f0}"
                 BUILD="python EEG/download_p300.py --dest ${PW_DATA_EEG}/erpbci && python EEG/physio_p300_finetune.py --edf-dir ${PW_DATA_EEG}/erpbci --out-dir ${PW_DATA_EEG}/p300_f0 --fold 0" ;;
    *) echo "ERROR: TASK must be sleep_edf or physio_p300, not '${TASK}'" >&2; exit 1 ;;
esac
case "${MODE}" in ft|probe|scratch) ;;
    *) echo "ERROR: MODE must be ft, probe or scratch, not '${MODE}'" >&2; exit 1 ;; esac
[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1

for s in train val test; do
    [[ -f "${DATA_DIR}/${s}.h5" ]] || {
        echo "ERROR: no ${DATA_DIR}/${s}.h5. Build the split first (needs mne; the" >&2
        echo "       download needs a login node):" >&2
        echo "         ${BUILD}" >&2
        exit 1; }
done

OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/eeg_downstream}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/${TASK}_${MODE}${TAG:+_${TAG}}}"
mkdir -p "${OUTPUT_DIR}"
# finetune_main reuses a directory without clearing it: a rerun that dies early
# must not leave the previous run's results.json to be reported as this one's.
rm -f "${OUTPUT_DIR}/results.json" "${OUTPUT_DIR}/best.pth" "${OUTPUT_DIR}/history.json"

SET_ARGS=()
EXTRA_ARGS=()
# A split built before the converters wrote the sampling rate unconditionally
# carries none, and the model refuses to guess. The converters resample to one
# fixed rate per task (sleep_edf_finetune.py FS = 100, physio_p300_finetune.py
# FS_OUT = 256), so state it when the file does not.
if ! python - "${DATA_DIR}/train.h5" <<'PY'
import sys, h5py
sys.exit(0 if h5py.File(sys.argv[1], "r").attrs.get("sampling_rate") else 1)
PY
then
    echo "NOTE: ${DATA_DIR}/train.h5 has no sampling_rate attribute; using ${RATE} Hz (the converter's rate)"
    SET_ARGS+=("model.eeg_c1.sampling_rate=${RATE}")
fi
if [[ -n "${ROUTE:-}" ]]; then
    patch="$(python -c "import sys; from physiowave.eeg_c1.routes import ROUTES
r = ROUTES.get(sys.argv[1]); print(r.patch_t if r else '')" "${ROUTE}")"
    [[ -n "${patch}" ]] || { echo "ERROR: ROUTE=${ROUTE} is not a route (physiowave/eeg_c1/routes.py)" >&2; exit 1; }
    SET_ARGS+=("model.eeg_c1.route_id=${ROUTE}" "model.eeg_c1.patch_samples=${patch}")
fi
[[ -n "${CLASS_WEIGHT:-}" ]] && EXTRA_ARGS+=(--class-weight "${CLASS_WEIGHT}")
if [[ "${MODE}" != "scratch" ]]; then
    PRETRAINED="${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: no file at ${PRETRAINED}" >&2; exit 1; }
    kind="$(python - "${PRETRAINED}" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("encoder" if "route_id" in ck else "checkpoint" if "model" in ck else "unknown")
PY
)"
    case "${kind}" in
        encoder)    ENCODER="${PRETRAINED}"
                    # an export made without --route carries no frontend: on a
                    # route it would load the Transformer alone, silently
                    if [[ -n "${ROUTE:-}" ]] && ! python - "${PRETRAINED}" "${ROUTE}" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if ck.get("route_id") == sys.argv[2] else 1)
PY
                    then
                        echo "ERROR: ${PRETRAINED} was not exported for route ${ROUTE}; pass the pretraining checkpoint instead" >&2
                        exit 1
                    fi ;;
        checkpoint) ENCODER="${OUTPUT_DIR}/encoder.pth"
                    python scripts/export_eeg_pretrained_encoder.py \
                        --checkpoint "${PRETRAINED}" --output "${ENCODER}" \
                        ${ROUTE:+--route "${ROUTE}"} ;;
        *) echo "ERROR: ${PRETRAINED} is neither an encoder nor a checkpoint" >&2; exit 1 ;;
    esac
    SET_ARGS+=("model.eeg_c1.pretrained=${ENCODER}")
    # probe: freeze what pretraining SUPPLIED. These montages sit on no route,
    # so their frontend is built fresh; freezing it too (the configs' default
    # scope, encoder) would freeze random filters and understate the probe.
    if [[ "${MODE}" == "probe" ]]; then
        EXTRA_ARGS+=(--freeze-encoder)
        SET_ARGS+=("model.eeg_c1.freeze_scope=${FREEZE_SCOPE:-pretrained}")
    fi
    if [[ "${MODE}" == "ft" ]]; then
        EXTRA_ARGS+=(--layer-decay "${LAYER_DECAY:-0.75}")
        [[ -n "${ENC_LR_SCALE:-}" ]] && EXTRA_ARGS+=(--encoder-lr-scale "${ENC_LR_SCALE}")
    fi
fi
[[ -n "${LR:-}" ]]     && EXTRA_ARGS+=(--lr "${LR}")
[[ -n "${EPOCHS:-}" ]] && EXTRA_ARGS+=(--epochs "${EPOCHS}")
# shellcheck disable=SC2206
[[ -n "${SET:-}" ]]    && SET_ARGS+=(${SET})
# EXTRA: raw finetune_main flags, e.g. EXTRA="--patience 5" or "--device cpu".
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]]  && EXTRA_ARGS+=(${EXTRA})

echo "============================================================"
echo "  EEG C1 downstream  ${TASK}  mode ${MODE}  lr ${LR:-<config>}  epochs ${EPOCHS:-<config>}"
echo "  route    ${ROUTE:-<own frontend>}  class weight ${CLASS_WEIGHT:-none}  seed ${SEED:-42}"
echo "  data     ${DATA_DIR}"
echo "  encoder  ${ENCODER:-<none: random initialisation>}"
echo "  output   ${OUTPUT_DIR}"
echo "============================================================"
CMD=(-m physiowave.train.finetune_main
     --config "${CONFIG}"
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
