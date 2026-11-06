#!/bin/bash
# =============================================================================
# Hyper-parameter sweep for one EEGPT benchmark task, on one node.
#
#   sbatch --export=ALL,TASK=bcic2a,PRETRAINED=<encoder> \
#          scripts/slurm/cineca_eegpt_sweep.sbatch
#
# or all three tasks at once: SWEEP=1 bash scripts/run_eegpt_bench.sh
#
# Runs every configuration in EEG/eegpt_sweep_grid.sh for ft AND scratch --
# 6 x 2 = 12 runs -- four at a time, one per GPU. Each run lands in
# <task>_f<fold>_<mode>_sw_<config>; scripts/collect_eegpt_bench.py then picks,
# for each of ft and scratch separately, the configuration with the best
# VALIDATION score and reports that one's test result. The test subjects never
# choose a configuration.
#
# CONFIGS restricts the grid (CONFIGS="base enc01"), MODES the modes.
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240000
#SBATCH --job-name=physiowave_eegpt_sweep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source EEG/eegpt_bench_lib.sh
# shellcheck disable=SC1091
source EEG/eegpt_sweep_grid.sh

TASK="${TASK:?set TASK=bcic2a|bcic2b|kaggleern}"
FOLD="${FOLD:-0}"
MODES="${MODES:-ft scratch}"
CONFIGS="${CONFIGS:-$(eegpt_sweep_names)}"
N_GPUS="${N_GPUS:-4}"
export TASK FOLD NUM_GPUS=1 PROGRESS="${PROGRESS:-auto}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

runs=()
for c in ${CONFIGS}; do
    eegpt_sweep_env "${c}" >/dev/null || exit 1          # validate the name
    for m in ${MODES}; do runs+=("${c}:${m}"); done
done

echo "============================================================"
echo "  EEGPT sweep ${TASK} fold ${FOLD}: ${#runs[@]} runs, ${N_GPUS} at a time  (job ${SLURM_JOB_ID:-?})"
echo "  configs: ${CONFIGS}"
echo "  modes:   ${MODES}"
echo "  encoder: ${PRETRAINED:-<none>}"
pw_print_layout
echo "============================================================"

# The split once, before any run starts.
DATA_DIR="${DATA_DIR:-$(eegpt_split_dir "${TASK}" "${FOLD}")}"
eegpt_build_split "${TASK}" "${FOLD}" "${DATA_DIR}" \
    "${RAW_DIR:-$(eegpt_raw_dir "${TASK}")}" \
    || { echo "ERROR: building the split failed" >&2; exit 1; }
export DATA_DIR

# Worker w takes runs w, w+N, w+2N, ... on GPU w. A fixed stride rather than a
# queue: it needs no locking and no `wait -n`, which older bash lacks.
worker() {
    local w="$1" k c m log rc=0
    for (( k = w; k < ${#runs[@]}; k += N_GPUS )); do
        c="${runs[$k]%%:*}"; m="${runs[$k]##*:}"
        log="${PW_CKPT_ROOT}/eegpt_bench/${TASK}_f${FOLD}_${m}_sw_${c}.log"
        mkdir -p "$(dirname "${log}")"
        echo "  [gpu ${w}] ${c} ${m}  -> ${log}"
        if ( eegpt_sweep_env "${c}" && export MODE="${m}" CUDA_VISIBLE_DEVICES="${w}" \
               && bash EEG/finetune_eegpt_bench.sh ) > "${log}" 2>&1; then
            echo "  [gpu ${w}] ${c} ${m}: done"
        else
            echo "  [gpu ${w}] ${c} ${m}: FAILED -- see ${log}" >&2
            rc=1
        fi
    done
    return ${rc}
}

pids=()
for (( w = 0; w < N_GPUS && w < ${#runs[@]}; w++ )); do
    worker "${w}" &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done

echo ""
python scripts/collect_eegpt_bench.py --root "${PW_CKPT_ROOT}/eegpt_bench" \
    --json "${PW_CKPT_ROOT}/eegpt_bench/summary.json" \
    | tee "${PW_CKPT_ROOT}/eegpt_bench/summary.txt" || true
exit ${rc}
