#!/bin/bash
# ============================================================================
# Draw the scale-fold-by-sleep-stage figure from a fine-tuned Sleep-EDF run
# and collect it into the repository.
#
#   sbatch scripts/slurm/cineca_visualize_sleep_fold.sbatch
#   sbatch --export=ALL,RUN=sleep_edf_ft_lr3e-5 scripts/slurm/cineca_visualize_sleep_fold.sbatch
#
# then, on a login node:
#   git add docs/runs/sleep_edf_fold && git commit -m "sleep_edf_fold: figure" && git push
#
# Only the wavelet frontend runs -- alpha is decided before the Transformer.
# On ONE GPU that is under a minute for the test split; on CPU it is not: the
# cross-scale FFNs run at the full 3000-sample length, ~90 ms a window on eight
# threads, which is most of an hour for ~35k windows.
#
#   RUN        the downstream run under $PW_CKPT_ROOT/eeg_downstream (sleep_edf_ft_lr1e-4)
#   DATA_DIR   the Sleep-EDF split ($PW_DATA_EEG/sleep_edf)
#   NAME       docs/runs/<NAME>                         (sleep_edf_fold)
#   FORMAT     svg,pdf
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --job-name=physiowave_sleep_fold
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"

RUN="${RUN:-sleep_edf_ft_lr1e-4}"
RUN_DIR="${RUN_DIR:-${PW_CKPT_ROOT}/eeg_downstream/${RUN}}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/sleep_edf}"
NAME="${NAME:-sleep_edf_fold}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Unbuffered, so `tail -f` on the .out shows progress as it happens; redirected
# to a file, Python otherwise holds its few lines until the job exits.
export PYTHONUNBUFFERED=1

[[ -f "${RUN_DIR}/best.pth" ]] || { echo "ERROR: no ${RUN_DIR}/best.pth" >&2; exit 1; }
[[ -f "${DATA_DIR}/test.h5" ]] || { echo "ERROR: no ${DATA_DIR}/test.h5" >&2; exit 1; }

python scripts/visualize_sleep_fold.py --run-dir "${RUN_DIR}" --data-dir "${DATA_DIR}" \
    --format "${FORMAT:-svg,pdf}" --threads "${OMP_NUM_THREADS}" || exit 1

DEST="docs/runs/${NAME}"
mkdir -p "${DEST}/figures" "${DEST}/figure_metadata"
cp "${RUN_DIR}"/figures/fig_paper_fold_by_stage.* "${DEST}/figures/"
cp "${RUN_DIR}"/figure_metadata/fig_paper_fold_by_stage.json "${DEST}/figure_metadata/"
echo "collected into ${DEST}; commit and push it from a login node"
