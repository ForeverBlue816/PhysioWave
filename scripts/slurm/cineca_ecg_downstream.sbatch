#!/bin/bash
# ============================================================================
# ECG C1 downstream: every task and mode in ONE job, one node, four GPUs.
#
#   sbatch scripts/slurm/cineca_ecg_downstream.sbatch
#
# GPU i takes task i of TASKS (ptbxl, cpsc2018, chapman, ptbxl_super): it
# builds that task's split if it is missing (ECG/ecg_downstream_prep.py,
# CPU, no network -- the raw data comes from scripts/download_ecg_downstream.sh
# on a login node), then runs MODES one after the other on that GPU:
#   ft       pretrained encoder, fine-tuned          (the row to report)
#   probe    pretrained encoder frozen, head only
#   scratch  same model, random initialisation       (the control)
# and the job ends by printing the table of every finished run
# (scripts/collect_ecg_downstream.py), into $PW_CKPT_ROOT/ecg_downstream/.
#
# A run whose results.json already has a test block is skipped, so if the
# walltime cuts the job short, resubmitting the same command finishes the
# rest. FORCE=1 reruns everything.
#
# Budget: the largest task (Chapman, ~28 k training records) is on the order
# of half an hour per fine-tuning run on one GPU, a probe a fraction of that,
# and every config stops early after 10 epochs without improvement; the
# splits take minutes to tens of minutes of CPU. 6 h leaves a wide margin.
#
#   PRETRAINED  $PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth
#   TASKS       "ptbxl cpsc2018 chapman ptbxl_super"  (more than 4: GPU i also
#               takes tasks i+4, i+8, ...)
#   MODES       "ft ft@3e-5 ft@1e-5 probe@1e-3 scratch". A token is MODE or
#               MODE@LR: MODE@LR runs with that learning rate into
#               <task>_<mode>_lr<LR>, plain MODE with the config's (1e-4).
#               Several fine-tuning rates are run because the first PTB-XL
#               run peaked at epoch 2-3 at 1e-4 (too fast for a pretrained
#               encoder) and the probe was still improving at epoch 29 (too
#               slow for a head alone). The collector reports, per mode, the
#               run with the best VALIDATION score -- never the best test.
#   TAG, LR, EPOCHS, SEED, SET   passed to ECG/finetune_ecg_c1.sh
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_ecg_downstream
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

TASKS="${TASKS:-ptbxl cpsc2018 chapman ptbxl_super}"
MODES="${MODES:-ft ft@3e-5 ft@1e-5 probe@1e-3 scratch}"
read -r -a TASK_LIST <<< "${TASKS}"
read -r -a MODE_LIST <<< "${MODES}"
N_GPUS="${N_GPUS:-4}"
ECG_ROOT="${ECG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/ecg}"
OUT_ROOT="${PW_CKPT_ROOT}/ecg_downstream"
export ECG_ROOT NUM_GPUS=1 PROGRESS="${PROGRESS:-log}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
PREP_JOBS=$(( ${SLURM_CPUS_PER_TASK:-32} / N_GPUS - 1 ))
[[ ${PREP_JOBS} -ge 1 ]] || PREP_JOBS=1
mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "  ECG C1 downstream  (job ${SLURM_JOB_ID:-?}, 1 node, ${N_GPUS} GPUs)"
echo "  tasks  ${TASKS}"
echo "  modes  ${MODES}"
echo "  encoder ${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_ecg_c1_moe/best.pth}"
echo "  output ${OUT_ROOT}"
echo "============================================================"

raw_for() {
    case "$1" in
        ptbxl|ptbxl_super) echo "${ECG_ROOT}/downstream/raw/ptbxl" ;;
        cpsc2018)          echo "${ECG_ROOT}/downstream/raw/cpsc2018" ;;
        chapman)           echo "${ECG_ROOT}/downstream/raw/chapman" ;;
        *)                 return 1 ;;
    esac
}

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
        data="${ECG_ROOT}/downstream/c1/${task}"
        if [[ ! -f "${data}/split.json" ]]; then
            if [[ ! -f "${raw}/.complete" ]]; then
                echo "[gpu ${gpu}] ${task}: ${raw} not downloaded -- on a login node:" \
                     "bash scripts/download_ecg_downstream.sh ${task%_super}"
                rc=1; continue
            fi
            echo "[gpu ${gpu}] ${task}: building the split ($(date +%H:%M))"
            python ECG/ecg_downstream_prep.py --task "${task}" --raw-dir "${raw}" \
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
                    bash ECG/finetune_ecg_c1.sh > "${log}" 2>&1; then
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
python scripts/collect_ecg_downstream.py --root "${OUT_ROOT}" \
    --json "${OUT_ROOT}/summary.json" --markdown "${OUT_ROOT}/summary.md" \
    | tee "${OUT_ROOT}/summary.txt" || true
exit ${rc}
