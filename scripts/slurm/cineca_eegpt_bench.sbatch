#!/bin/bash
# =============================================================================
# EEGPT benchmarks (BCIC-IV-2a, BCIC-IV-2b, KaggleERN) on the C1 encoder.
#
#   sbatch --export=ALL,TASK=bcic2a,PRETRAINED=$HOME/hf_release/eeg_c1_encoder.pth \
#          scripts/slurm/cineca_eegpt_bench.sbatch
#
# Runs MODES side by side, one GPU each, on one node: ft (the pretrained
# encoder, every parameter fine-tuned) and scratch (the same model from random
# initialisation -- the control). The difference between the two is the
# measurement; either alone is only a number. probe (what pretraining supplied
# frozen) runs when named; a token MODE@LR sets that run's lr, e.g.
#   MODES="probe@1e-3 probe@2.5e-4"
#
# All three tasks:
#
#   for t in bcic2a bcic2b kaggleern; do
#     sbatch --export=ALL,TASK=$t,PRETRAINED=$HOME/hf_release/eeg_c1_encoder.pth \
#            scripts/slurm/cineca_eegpt_bench.sbatch
#   done
#
# Two encoders side by side (best vs last epoch): add TAG=best / TAG=final so
# their runs land in different directories.
#
# The data must be DOWNLOADED first, on the login node -- compute nodes have no
# internet. Converting it into a split happens here, on first use. See
# EEG/finetune_eegpt_bench.sh.
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240000
#SBATCH --job-name=physiowave_eegpt_bench
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

TASK="${TASK:?set TASK=bcic2a|bcic2b|kaggleern}"
FOLD="${FOLD:-0}"
MODES="${MODES:-ft scratch}"
read -r -a MODE_LIST <<< "${MODES}"
export TASK FOLD NUM_GPUS=1 PROGRESS="${PROGRESS:-auto}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================================"
echo "  EEGPT benchmark ${TASK} fold ${FOLD}, modes: ${MODES}  (job ${SLURM_JOB_ID:-?})"
echo "  encoder ${PRETRAINED:-<none>}   preprocessing ${PREP:-pretrain} (BCIC; ERN always pretrain)"
pw_print_layout
echo "============================================================"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/EEG/eegpt_bench_lib.sh"

# The split is built ONCE, before the modes start, rather than by each of them
# at the same moment into the same directory.
DATA_DIR="${DATA_DIR:-$(eegpt_split_dir "${TASK}" "${FOLD}")}"
eegpt_build_split "${TASK}" "${FOLD}" "${DATA_DIR}" \
    "${RAW_DIR:-$(eegpt_raw_dir "${TASK}")}" \
    || { echo "ERROR: building the split failed" >&2; exit 1; }
export DATA_DIR

pids=()
for i in "${!MODE_LIST[@]}"; do
    token="${MODE_LIST[$i]}"; mode="${token%%@*}"
    lr="${LR:-}"; [[ "${token}" == *@* ]] && lr="${token#*@}"
    run_tag="$(eegpt_tag "${TASK}")"
    [[ -n "${lr}" ]] && run_tag="lr${lr}${run_tag:+_${run_tag}}"
    log="${PW_CKPT_ROOT}/eegpt_bench/${TASK}_f${FOLD}_${mode}${run_tag:+_${run_tag}}.log"
    mkdir -p "$(dirname "${log}")"
    echo "  starting ${token} on GPU ${i}  -> ${log}"
    CUDA_VISIBLE_DEVICES="${i}" MODE="${mode}" LR="${lr}" \
        bash EEG/finetune_eegpt_bench.sh > "${log}" 2>&1 &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  ${MODE_LIST[$i]}: done"
    else
        echo "  ${MODE_LIST[$i]}: FAILED -- see its log" >&2
        rc=1
    fi
done

# The table as it stands now, also written to summary.txt: each task's job
# rewrites it when it finishes, so the last one to finish leaves all three.
echo ""
python scripts/collect_eegpt_bench.py --root "${PW_CKPT_ROOT}/eegpt_bench" \
    --json "${PW_CKPT_ROOT}/eegpt_bench/summary.json" \
    | tee "${PW_CKPT_ROOT}/eegpt_bench/summary.txt" || true
exit ${rc}
