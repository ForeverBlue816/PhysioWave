#!/bin/bash
# ============================================================================
# SeizeIT2, behind-the-ear EEG alone: C1 against EEGPT and LaBraM under one
# protocol (configs/fusion/seizeit2_eeg.yaml, physiowave/fusion/baselines.py).
# One node, four GPUs; GPU i takes runs i, i+4, ... Each run trains, then is
# scored event-level on the test patients' continuous recordings on the same
# GPU (Fusion/seizeit2_events.py). Ends with both tables.
#
#   bash scripts/download_eeg_baselines.sh            # once, on the login node
#   sbatch scripts/slurm/cineca_seizeit2_eeg.sbatch
#
#   RUNS      "c1:frozen c1:finetune eegpt:frozen eegpt:finetune labram:frozen
#              labram:finetune c1native:frozen c1native:finetune"
#             also luna_base / luna_large / luna_huge (LUNA, NeurIPS 2025), e.g.
#             RUNS="luna_large:frozen luna_large:finetune luna_huge:frozen luna_huge:finetune"
#             encoder:mode[:lr]. c1 = C1 behind the same channel adapter as the
#             baselines (on its E19_256 route); c1native = C1 on the two raw
#             channel names with new embedding rows, as in the fusion runs;
#             c1siena = C1 after the Siena continuation (C1_SIENA_CKPT). An lr
#             overrides train.lr for that run, e.g. c1siena:finetune:5e-5.
#   TAG       directory suffix (default sze): $OUT_ROOT/eeg_<encoder>_<mode>[_lr<lr>]_<TAG>
#   EEG_CKPT / EEGPT_CKPT / LABRAM_CKPT / LUNA_DIR   the checkpoints
#   C1_SIENA_CKPT  $PW_CKPT_ROOT/pretrain_eeg_c1_siena/latest.pth
#   STEP      seconds between windows in the event evaluation (1)
#   SET       extra --set overrides, space separated
# A finished run (results.json, events_cross.json) is skipped; a cut-off one
# resumes from latest.pth, so resubmitting the same command finishes the job.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_sz2_eeg
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

FUSION_ROOT="${FUSION_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/fusion}"
DATA="${DATA:-${FUSION_ROOT}/c1/seizeit2}"
RAW="${RAW:-${FUSION_ROOT}/raw/seizeit2}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/fusion}"
TAG="${TAG:-sze}"
STEP="${STEP:-1}"
RUNS="${RUNS:-c1:frozen c1:finetune eegpt:frozen eegpt:finetune labram:frozen labram:finetune c1native:frozen c1native:finetune}"
EEG_CKPT="${EEG_CKPT:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
EEGPT_CKPT="${EEGPT_CKPT:-${PW_CKPT_ROOT}/baselines/eegpt/eegpt_mcae_58chs_4s_large4E.safetensors}"
LABRAM_CKPT="${LABRAM_CKPT:-${PW_CKPT_ROOT}/baselines/labram/labram-base.pth}"
LUNA_DIR="${LUNA_DIR:-${PW_CKPT_ROOT}/baselines/luna}"
C1_SIENA_CKPT="${C1_SIENA_CKPT:-${PW_CKPT_ROOT}/pretrain_eeg_c1_siena/latest.pth}"
N_GPUS=4
read -r -a RUN_LIST <<< "${RUNS}"

[[ -f "${DATA}/train.h5" ]] || { echo "ERROR: ${DATA}/train.h5 missing -- run cineca_fusion_prep.sbatch"; exit 1; }
grep -q '"official"' "${DATA}/split.json" 2>/dev/null \
    || echo "WARNING: ${DATA}/split.json is not the official split"
need() { [[ -f "$2" ]] || { echo "ERROR: no $1 checkpoint at $2 (bash scripts/download_eeg_baselines.sh)"; exit 1; }; }
for spec in ${RUNS}; do
    e="${spec%%:*}"
    case "${e}" in
        c1|c1native) need C1 "${EEG_CKPT}" ;;
        c1siena) need "C1 (Siena continuation)" "${C1_SIENA_CKPT}" ;;
        eegpt) need EEGPT "${EEGPT_CKPT}" ;;
        labram) need LaBraM "${LABRAM_CKPT}" ;;
        luna_base|luna_large|luna_huge) need LUNA "${LUNA_DIR}/LUNA_${e#luna_}.safetensors" ;;
        *) echo "ERROR: unknown encoder ${e} in RUNS"; exit 1 ;;
    esac
done
mkdir -p "${OUT_ROOT}"
echo "runs    ${RUNS}"
echo "data    ${DATA}"
echo "c1      ${EEG_CKPT}"; echo "eegpt   ${EEGPT_CKPT}"; echo "labram  ${LABRAM_CKPT}"
echo "luna    ${LUNA_DIR}"
echo "c1siena ${C1_SIENA_CKPT}"
echo "output  ${OUT_ROOT}/eeg_<encoder>_<mode>_${TAG}"

worker() {
    local gpu="$1" i spec enc mode out log rc=0
    local -a sets
    for ((i = gpu; i < ${#RUN_LIST[@]}; i += N_GPUS)); do
        spec="${RUN_LIST[$i]}"; IFS=: read -r enc mode lr <<< "${spec}"
        out="${OUT_ROOT}/eeg_${enc}_${mode}${lr:+_lr${lr}}_${TAG}"; log="${out}.log"; mkdir -p "${out}"
        case "${enc}" in
            c1)       sets=("encoders.eeg=${EEG_CKPT}" "encoders.eeg_arch=c1") ;;
            c1siena)  sets=("encoders.eeg=${C1_SIENA_CKPT}" "encoders.eeg_arch=c1") ;;
            c1native) sets=("encoders.eeg=${EEG_CKPT}" "encoders.eeg_arch=c1" "model.eeg_adapter=null") ;;
            eegpt)    sets=("encoders.eeg=${EEGPT_CKPT}" "encoders.eeg_arch=eegpt") ;;
            labram)   sets=("encoders.eeg=${LABRAM_CKPT}" "encoders.eeg_arch=labram") ;;
            luna_base|luna_large|luna_huge)
                      sets=("encoders.eeg=${LUNA_DIR}/LUNA_${enc#luna_}.safetensors" "encoders.eeg_arch=${enc}") ;;
            *) echo "[gpu ${gpu}] ${spec}: unknown encoder"; rc=1; continue ;;
        esac
        [[ -n "${lr}" ]] && sets+=("train.lr=${lr}")
        if [[ -f "${out}/results.json" ]]; then
            echo "[gpu ${gpu}] ${spec}: trained, skipped"
        else
            echo "[gpu ${gpu}] ${spec}: train $(date +%H:%M) -> ${log}"
            # shellcheck disable=SC2086
            if ! CUDA_VISIBLE_DEVICES="${gpu}" python -m physiowave.fusion.train \
                    --config fusion/seizeit2_eeg --output-dir "${out}" \
                    --set "data.tasks.seizeit2.dir=${DATA}" "${sets[@]}" \
                          "model.encoder_mode=${mode}" ${SET:-} >> "${log}" 2>&1; then
                echo "[gpu ${gpu}] ${spec}: training FAILED -- see ${log}"; rc=1; continue
            fi
        fi
        if [[ -f "${out}/events_cross.json" ]]; then
            echo "[gpu ${gpu}] ${spec}: events done, skipped"; continue
        fi
        echo "[gpu ${gpu}] ${spec}: events $(date +%H:%M)"
        if CUDA_VISIBLE_DEVICES="${gpu}" python Fusion/seizeit2_events.py --run-dir "${out}" \
                --raw-dir "${RAW}" --step "${STEP}" --gate cross > "${out}/events.log" 2>&1; then
            echo "[gpu ${gpu}] ${spec}: done $(date +%H:%M)"
        else
            echo "[gpu ${gpu}] ${spec}: events FAILED -- see ${out}/events.log"; rc=1
        fi
    done
    return ${rc}
}

pids=()
for ((g = 0; g < N_GPUS && g < ${#RUN_LIST[@]}; g++)); do worker "${g}" & pids+=($!); done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done

mkdir -p "${OUT_ROOT}/eeg_${TAG}_table"
for d in "${OUT_ROOT}"/eeg_*_"${TAG}"; do
    [[ -d "${d}" ]] && ln -sfn "${d}" "${OUT_ROOT}/eeg_${TAG}_table/$(basename "${d}")"
done
python scripts/collect_fusion.py --root "${OUT_ROOT}/eeg_${TAG}_table" \
    | tee "${OUT_ROOT}/seizeit2_eeg_${TAG}_windows.txt"
python scripts/collect_seizeit2_events.py --root "${OUT_ROOT}/eeg_${TAG}_table" --condition complete \
    --markdown "${OUT_ROOT}/seizeit2_eeg_${TAG}_events.md" | tee "${OUT_ROOT}/seizeit2_eeg_${TAG}_events.txt"
exit ${rc}
