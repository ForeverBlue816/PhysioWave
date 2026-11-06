#!/bin/bash
# ============================================================================
# sEMG C1 downstream: every task and mode in ONE job, one node, four GPUs.
#
#   sbatch scripts/slurm/cineca_emg_downstream.sbatch
#
# GPU i takes tasks i, i+4, ... of TASKS (epn612, grabmyo, ninapro_db5,
# epn612_xuser, grabmyo_xsubj): it builds a task's split if it is missing
# (EMG/emg_downstream_prep.py, CPU, no network -- the raw data comes from
# scripts/download_emg_downstream.sh on a login node), then runs MODES on it
# one after the other:
#   ft       pretrained encoder, fine-tuned          (the row to report)
#   probe    pretrained encoder frozen, head (+ the montage's own frontend)
#   scratch  same model, random initialisation       (the control)
# and ends with the table of every finished run (scripts/collect_emg_downstream.py).
#
# A run whose results.json has a test block is skipped: resubmit to finish
# what the walltime cut off. FORCE=1 reruns everything.
#
#   PRETRAINED  $PW_CKPT_ROOT/pretrain_emg_c1_moe/best.pth
#   TASKS       "epn612 grabmyo ninapro_db5 epn612_xuser grabmyo_xsubj"
#               (ninapro_db2, uci_emg and uci_emg_xsess run when named:
#               TASKS="uci_emg uci_emg_xsess")
#   MODES       "ft ft@3e-4 probe@1e-3 scratch"  (MODE or MODE@LR; per mode the
#               table reports the run with the best VALIDATION score)
#   TAG, LR, EPOCHS, SEED, SET   passed to EMG/finetune_emg_c1.sh
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_emg_downstream
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

TASKS="${TASKS:-epn612 grabmyo ninapro_db5 epn612_xuser grabmyo_xsubj}"
MODES="${MODES:-ft ft@3e-4 probe@1e-3 scratch}"
read -r -a TASK_LIST <<< "${TASKS}"
read -r -a MODE_LIST <<< "${MODES}"
N_GPUS="${N_GPUS:-4}"
EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"
OUT_ROOT="${PW_CKPT_ROOT}/emg_downstream"
export EMG_ROOT NUM_GPUS=1 PROGRESS="${PROGRESS:-log}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
PREP_JOBS=$(( ${SLURM_CPUS_PER_TASK:-32} / N_GPUS - 1 ))
[[ ${PREP_JOBS} -ge 1 ]] || PREP_JOBS=1
mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "  sEMG C1 downstream  (job ${SLURM_JOB_ID:-?}, 1 node, ${N_GPUS} GPUs)"
echo "  tasks  ${TASKS}"
echo "  modes  ${MODES}"
echo "  encoder ${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_emg_c1_moe/best.pth}"
echo "  output ${OUT_ROOT}"
echo "============================================================"

set_for() {      # the download a task reads
    case "$1" in
        epn612|epn612_xuser)   echo epn612 ;;
        grabmyo|grabmyo_xsubj) echo grabmyo ;;
        ninapro_db2)           echo ninapro_db2 ;;
        ninapro_db5)           echo ninapro_db5 ;;
        uci_emg|uci_emg_xsess) echo uci_emg ;;
        *)                     return 1 ;;
    esac
}
raw_for() { local s; s="$(set_for "$1")" || return 1; echo "${EMG_ROOT}/downstream/raw/${s}"; }

done_already() {   # results.json with a test block
    python - "$1/results.json" <<'PY' 2>/dev/null
import json, sys
sys.exit(0 if "test" in json.load(open(sys.argv[1])) else 1)
PY
}

# One GPU's work: its tasks, each split then each mode, in order.
worker() {
    local gpu="$1" t task raw data mode out log rc=0
    for ((t = gpu; t < ${#TASK_LIST[@]}; t += N_GPUS)); do
        task="${TASK_LIST[$t]}"
        raw="$(raw_for "${task}")" || { echo "[gpu ${gpu}] unknown task ${task}"; rc=1; continue; }
        data="${EMG_ROOT}/downstream/c1/${task}"
        if [[ ! -f "${data}/split.json" ]]; then
            if [[ ! -f "${raw}/.complete" ]]; then
                echo "[gpu ${gpu}] ${task}: ${raw} not downloaded -- on a login node:" \
                     "bash scripts/download_emg_downstream.sh $(set_for "${task}")"
                rc=1; continue
            fi
            echo "[gpu ${gpu}] ${task}: building the split ($(date +%H:%M))"
            python EMG/emg_downstream_prep.py --task "${task}" --raw-dir "${raw}" \
                --out-dir "${data}" --jobs "${PREP_JOBS}" \
                > "${OUT_ROOT}/${task}_prep.log" 2>&1 \
                || { echo "[gpu ${gpu}] ${task}: split FAILED, see ${task}_prep.log"; rc=1; continue; }
        fi
        for token in "${MODE_LIST[@]}"; do
            mode="${token%%@*}"
            lr=""; [[ "${token}" == *@* ]] && lr="${token#*@}"
            tag="${lr:+lr${lr}}${TAG:+${lr:+_}${TAG}}"
            out="${OUT_ROOT}/${task}_${mode}${tag:+_${tag}}"
            if [[ "${FORCE:-0}" != "1" ]] && done_already "${out}"; then
                echo "[gpu ${gpu}] ${task} ${token}: already done, skipped"
                continue
            fi
            log="${out}.log"
            echo "[gpu ${gpu}] ${task} ${token}: start $(date +%H:%M) -> ${log}"
            if CUDA_VISIBLE_DEVICES="${gpu}" TASK="${task}" MODE="${mode}" \
                    LR="${lr:-${LR:-}}" DATA_DIR="${data}" OUTPUT_DIR="${out}" \
                    bash EMG/finetune_emg_c1.sh > "${log}" 2>&1; then
                echo "[gpu ${gpu}] ${task} ${token}: done $(date +%H:%M)"
            else
                echo "[gpu ${gpu}] ${task} ${token}: FAILED -- see ${log}"; rc=1
            fi
        done
    done
    return ${rc}
}

pids=()
for ((g = 0; g < N_GPUS && g < ${#TASK_LIST[@]}; g++)); do
    worker "${g}" &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done

echo ""
python scripts/collect_emg_downstream.py --root "${OUT_ROOT}" \
    --json "${OUT_ROOT}/summary.json" --markdown "${OUT_ROOT}/summary.md" \
    | tee "${OUT_ROOT}/summary.txt" || true
exit ${rc}
