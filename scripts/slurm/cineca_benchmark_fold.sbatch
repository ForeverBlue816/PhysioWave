#!/bin/bash
# ============================================================================
# Benchmark ScaleFold's compute saving on one GPU and collect the figure.
#
#   sbatch scripts/slurm/cineca_benchmark_fold.sbatch
#
# then, on a login node:
#   git add docs/runs/fold_efficiency && git commit -m "fold_efficiency: figure" && git push
#
# scripts/benchmark_scale_fold.py times the EEG C1 encoder with the fold and
# with the unfolded bands, per route, at the training batch sizes. One GPU,
# a few minutes. The node should be otherwise idle, so the job asks for a
# whole GPU and nothing shares it.
#
#   RUN_DIR    $PW_CKPT_ROOT/pretrain_eeg_c1_moe
#   NAME       docs/runs/<NAME>                        (fold_efficiency)
#   ITERS      timed forward passes per measurement (20)
#   EXTRA      more arguments for the script, e.g. "--batch-sizes E128_512=12"
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60000
#SBATCH --job-name=physiowave_fold_bench
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"

RUN_DIR="${RUN_DIR:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe}"
NAME="${NAME:-fold_efficiency}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

[[ -f "${RUN_DIR}/best.pth" ]] || { echo "ERROR: no ${RUN_DIR}/best.pth" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python scripts/benchmark_scale_fold.py --run-dir "${RUN_DIR}" \
    --format "${FORMAT:-svg,pdf}" --iters "${ITERS:-20}" --threads "${OMP_NUM_THREADS}" \
    ${EXTRA:-} || exit 1

DEST="docs/runs/${NAME}"
mkdir -p "${DEST}/figures" "${DEST}/figure_metadata"
cp "${RUN_DIR}"/figures/fig_paper_fold_efficiency.* "${DEST}/figures/"
cp "${RUN_DIR}"/figure_metadata/fig_paper_fold_efficiency.json "${DEST}/figure_metadata/"
cp "${RUN_DIR}/fold_efficiency.json" "${DEST}/"
echo "collected into ${DEST}; commit and push it from a login node"
