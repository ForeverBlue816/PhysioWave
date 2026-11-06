#!/bin/bash
# ============================================================================
# sEMG corpus preprocessing, as a SLURM array. One dataset per submission.
#
#   INSPECT first, on a login node, and read the amplitude column:
#     DATASET=hyser INSPECT=20 bash EMG/preprocess_emg_corpus.sh
#
#   then, sized to the corpus:
#     sbatch --export=ALL,DATASET=emg2pose     --array=0-3 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=emg2qwerty   --array=0-3 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=hyser        --array=0-1 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=cemhsey_8x8  --array=0-3 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=cemhsey_5x13 --array=0-3 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=putemg       --array=0-0 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#
# WHERE THOSE SIZES COME FROM. The work is reading: emg2pose's tar is 463 GB,
# emg2qwerty ~340 GB and CEMHSEY 332 GB of zipped MATLAB files, each of which
# both CEMHSEY corpora read in full. Units are ~130 (emg2pose), ~280
# (emg2qwerty), ~210 (Hyser), ~400 (each CEMHSEY corpus) and ~90 (putEMG); on
# real records a CEMHSEY trial takes ~2 s to inflate, read and filter on one
# core and an emg2pose file well under one. Four tasks of 15 workers leave the
# 4 h walltime several times what that implies -- and a task that runs out
# resumes where it stopped.
#
# TASK 0 LISTS THE CORPUS and the others wait for its record_list.txt, so the
# tree is walked once. For the largest corpora that walk is the slow part;
# running
#   DATASET=emg2pose LIST_ONLY=1 bash EMG/preprocess_emg_corpus.sh
# on a login node first takes it out of the array's walltime (emg2pose's
# listing walks the 463 GB tar header by header; CEMHSEY's reads 13 zip
# directories).
#
# RESUMABLE. A unit of records is done when its part file exists, so a task
# that hits the walltime is resubmitted with the SAME --array and picks up at
# the first unit without one. Keep RECORDS_PER_UNIT the same across attempts.
#
# AFTER THE ARRAY, merge every dataset that has finished:
#   python scripts/build_eeg_c1_manifest.py --modality emg \
#       --corpus-root $EMG_ROOT/emg_c1_corpus --allow-missing --check-shards --jobs 16
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --job-name=physiowave_emg_prep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

DATASET="${DATASET:-}"
if [[ -z "${DATASET}" ]]; then
    echo "ERROR: name the corpus:" >&2
    echo "  sbatch --export=ALL,DATASET=emg2pose --array=0-3 \\" >&2
    echo "         scripts/slurm/cineca_emg_corpus_preprocess.sbatch" >&2
    exit 1
fi
export DATASET

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"
# The training venv plus wfdb (pip install wfdb once, for Hyser). Not pwprep:
# the shared preprocessing module imports torch, and $HOME/pw is the
# environment known to have it. PREP_VENV overrides.
PREP_VENV="${PREP_VENV:-${HOME}/pw}"
if [[ -f "${PREP_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PREP_VENV}/bin/activate"
fi

# One worker per core leaves nothing for h5py's gzip and the zip reads; the
# units are independent, so this is the whole parallelism of a task.
# Hyser: six, not fifteen. A unit is 64 files of 256 channels, split into four
# 64-channel grids; for the 25 s recordings that is ~3.3 GB of kept windows
# per worker, doubled while its shards are written, and fifteen of them ran
# the 120 GB node out of memory (both tasks, after 7 minutes). Six fit.
if [[ "${DATASET}" == "hyser" ]]; then
    export JOBS="${JOBS:-6}"
fi
export JOBS="${JOBS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 1 ))}"
export OMP_NUM_THREADS=1
export EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"

echo "task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} of ${DATASET}, ${JOBS} worker(s)"
bash EMG/preprocess_emg_corpus.sh
