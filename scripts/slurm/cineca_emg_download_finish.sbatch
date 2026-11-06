#!/bin/bash
# ============================================================================
# The sEMG download's second stage, on a compute node: everything that needs
# CPU and no network.
#
#   STAGE=fetch nohup bash scripts/download_emg_pretrain_corpora.sh all > ~/emg_dl.log 2>&1 &   # login node
#   sbatch scripts/slurm/cineca_emg_download_finish.sbatch                                     # then this
#
# A login node kills a process that uses a lot of CPU, and md5ing CEMHSEY's
# 332 GB, unpacking emg2qwerty's 308 GB tar.gz and GESTURE's deflate64 zips is
# exactly that. Compute nodes have no internet, which this stage does not
# need: it checks each download is whole (fetch_ranged.py --check-only),
# verifies and unpacks it, and marks the corpus complete. A corpus whose bytes
# are not all there yet is reported and left for another STAGE=fetch run;
# resubmitting this afterwards picks up only what is still unfinished.
#
# No GPU. One core does the gzip and md5 work; eight leave room for Lustre.
# emg2qwerty's unpack is the long part (single-threaded gzip unless pigz is
# installed), hence the 8 h.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --job-name=physiowave_emg_dl_finish
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"
PREP_VENV="${PREP_VENV:-${HOME}/pw}"
if [[ -f "${PREP_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PREP_VENV}/bin/activate"
fi
export EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"
export STAGE=finish

echo "sEMG download, finishing stage, job ${SLURM_JOB_ID:-?} on $(hostname)"
bash scripts/download_emg_pretrain_corpora.sh "${TARGET:-all}"
_rc=$?
echo ""
bash scripts/download_emg_pretrain_corpora.sh status
exit ${_rc}
