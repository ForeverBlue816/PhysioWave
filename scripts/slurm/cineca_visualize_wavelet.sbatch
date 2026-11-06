#!/bin/bash
# ============================================================================
# The learned wavelet front ends of the three C1 encoders: one figure and a
# report, collected into the repository.
#
#   sbatch scripts/slurm/cineca_visualize_wavelet.sbatch
#
# then, on a login node:
#   git add figures/fig_wavelet.* docs/runs/fig_wavelet.md docs/runs/fig_wavelet \
#     && git commit -m "fig_wavelet: learned front ends" && git push
#
# One GPU: the level-2/3 selector weights need the full decomposition, which
# is slow on CPU for the 128-channel routes. A few minutes.
#
#   EEG_CKPT, ECG_CKPT, EMG_CKPT   defaults below (the sEMG run's epoch-38 file)
#   WINDOWS                        windows per corpus for the selector (2000)
#   DEEP_WINDOWS                   of those, through the full decomposition (256)
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --job-name=physiowave_wavelet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"

EEG_CKPT="${EEG_CKPT:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
ECG_CKPT="${ECG_CKPT:-${PW_CKPT_ROOT}/pretrain_ecg_c1_moe/best.pth}"
EMG_CKPT="${EMG_CKPT:-${PW_CKPT_ROOT}/pretrain_emg_c1_moe/epoch38.pth}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for f in "${EEG_CKPT}" "${ECG_CKPT}" "${EMG_CKPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: no checkpoint at ${f}; this directory holds:" >&2
        ls -1 "$(dirname "${f}")"/*.pth >&2 2>/dev/null
        exit 1
    fi
done

OUT="${PW_CKPT_ROOT}/wavelet_frontend"
mkdir -p "${OUT}"
python scripts/visualize_wavelet_frontend.py \
    --eeg "${EEG_CKPT}" --ecg "${ECG_CKPT}" --emg "${EMG_CKPT}" \
    --windows "${WINDOWS:-2000}" --deep-windows "${DEEP_WINDOWS:-256}" \
    --out-dir "${OUT}" --report docs/runs/fig_wavelet.md \
    --format pdf,png,svg --threads "${OMP_NUM_THREADS}" || exit 1

mkdir -p figures docs/runs/fig_wavelet/figure_metadata docs/runs/fig_wavelet/figure_data
cp "${OUT}"/figures/fig_wavelet.pdf "${OUT}"/figures/fig_wavelet.png \
   "${OUT}"/figures/fig_wavelet.svg figures/
cp "${OUT}"/figure_metadata/fig_wavelet.json docs/runs/fig_wavelet/figure_metadata/
# The plotted arrays, so the figure can be redrawn anywhere without the
# checkpoints:  python scripts/visualize_wavelet_frontend.py --from-saved docs/runs/fig_wavelet
cp "${OUT}"/figure_data/fig_wavelet.npz docs/runs/fig_wavelet/figure_data/
echo "collected: figures/fig_wavelet.{pdf,png,svg}, docs/runs/fig_wavelet.md, docs/runs/fig_wavelet/"
