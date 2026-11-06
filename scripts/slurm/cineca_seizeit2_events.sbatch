#!/bin/bash
# ============================================================================
# Event-level SeizeIT2 evaluation of trained fusion runs, on the test patients'
# continuous recordings (Fusion/seizeit2_events.py). One node, four GPUs: GPU i
# takes run directories i, i+4, ...
#
#   sbatch --export=ALL,RUN_DIRS="ralf_frozen_szo concat_frozen_szo ..." \
#          scripts/slurm/cineca_seizeit2_events.sbatch
#
#   RUN_DIRS   run directories under $PW_CKPT_ROOT/fusion (or absolute paths);
#              default every *_szo directory there with a best.pth
#   STEP       seconds between windows (1, the reference grid)
#   GATE       cross (the reference code) | both | none
#   SCORE_ONLY 1: no inference, re-score every run's cached predictions (after a
#              change to the scoring), all runs at once on the node's CPUs:
#              sbatch --time=3:00:00 --export=ALL,SCORE_ONLY=1,RUN_DIRS="..." <this>
# Each writes <run>/predictions/*.npz and <run>/events_<gate>.json; a run's
# predictions are reused on a resubmit, so a cut-off job continues.
# The table: python scripts/collect_seizeit2_events.py --root $PW_CKPT_ROOT/fusion
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_sz2_events
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

FUSION_ROOT="${FUSION_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/fusion}"
RAW="${RAW:-${FUSION_ROOT}/raw/seizeit2}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/fusion}"
STEP="${STEP:-1}"
GATE="${GATE:-cross}"
SCORE_ONLY="${SCORE_ONLY:-0}"
N_GPUS=4
if [[ -z "${RUN_DIRS:-}" ]]; then
    RUN_DIRS="$(cd "${OUT_ROOT}" && ls -d *_szo*/ 2>/dev/null | tr -d '/' | tr '\n' ' ')"
fi
read -r -a DIRS <<< "${RUN_DIRS}"
[[ ${#DIRS[@]} -gt 0 ]] || { echo "ERROR: no run directories"; exit 1; }
echo "runs ${DIRS[*]}"; echo "raw  ${RAW}"; echo "step ${STEP}  gate ${GATE}  score-only ${SCORE_ONLY}"
EXTRA=()
if [[ "${SCORE_ONLY}" == "1" ]]; then
    EXTRA=(--score-only); N_GPUS=${#DIRS[@]}      # CPU work: every run in parallel
fi

worker() {
    local gpu="$1" i d rc=0
    for ((i = gpu; i < ${#DIRS[@]}; i += N_GPUS)); do
        d="${DIRS[$i]}"; [[ "${d}" == /* ]] || d="${OUT_ROOT}/${d}"
        if [[ "${SCORE_ONLY}" == "1" ]]; then
            [[ -d "${d}/predictions" ]] || { echo "[${gpu}] ${d}: no predictions, skipped"; continue; }
        else
            [[ -f "${d}/best.pth" ]] || { echo "[gpu ${gpu}] ${d}: no best.pth, skipped"; continue; }
        fi
        echo "[gpu ${gpu}] $(basename "${d}"): start $(date +%H:%M)"
        if CUDA_VISIBLE_DEVICES="$((gpu % 4))" python Fusion/seizeit2_events.py --run-dir "${d}" \
                --raw-dir "${RAW}" --step "${STEP}" --gate "${GATE}" "${EXTRA[@]}" \
                >> "${d}/events.log" 2>&1; then
            echo "[gpu ${gpu}] $(basename "${d}"): done $(date +%H:%M)"
        else
            echo "[gpu ${gpu}] $(basename "${d}"): FAILED -- see ${d}/events.log"; rc=1
        fi
    done
    return ${rc}
}
pids=()
for ((g = 0; g < N_GPUS && g < ${#DIRS[@]}; g++)); do worker "${g}" & pids+=($!); done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
python scripts/collect_seizeit2_events.py --root "${OUT_ROOT}" --gate "${GATE}" \
    --markdown "${OUT_ROOT}/seizeit2_events.md" | tee "${OUT_ROOT}/seizeit2_events.txt"
exit ${rc}
