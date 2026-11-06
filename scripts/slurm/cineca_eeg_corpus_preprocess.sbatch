#!/bin/bash
# ============================================================================
# FACED / TDBRAIN / PhysioNetMI / M3CV / HBN / HGD preprocessing, as a SLURM
# array. One dataset per submission; DATASET says which.
#
#   INSPECT first, on a login node, and READ THE REPORT:
#     DATASET=hbn INSPECT=60 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh
#
#   then, sized to the corpus:
#     sbatch --export=ALL,DATASET=physionet_mi --array=0-3  scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=faced        --array=0-7  scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=m3cv         --array=0-7  scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=hgd          --array=0-3  scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=tdbrain      --array=0-15 scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=hbn          --array=0-63 scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=siena        --array=0-3  scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
#
# THE ARRAY LENGTH IS A GUESS UNTIL YOU MEASURE IT. --inspect prints a per-file
# cost projection from the headers it reads; divide by the tasks you want and
# check it against --time below. HBN is ~1.9 TB of 128-channel 500 Hz recordings
# and is the one that actually needs 64 tasks; PhysioNetMI is 3.4 GB and would
# spend more time queueing for 64 than processing on 4.
#
# TDBRAIN AND HBN ARE BEHIND DATA USE AGREEMENTS. Nothing here downloads them.
#
# Everything else -- billing, memory, cores, resume, the N_SHARDS rule -- is as
# in cineca_tueg_preprocess.sbatch, and for the same reasons; see that file's
# header for why each is what it is. The one difference is --mem: these corpora
# hold shorter recordings than TUEG, but HBN's 128 channels at 500 Hz in float64
# make a 30-minute cap ~920 MB per worker before filtering, so the headroom
# stays.
#
# AFTER THE ARRAY, merge every dataset that has finished:
#   python scripts/build_eeg_c1_manifest.py --corpus-root $DATA_ROOT \
#       --datasets tueg,faced,tdbrain,physionet_mi,m3cv,hbn,hgd \
#       --allow-missing --check-shards --jobs 16
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --job-name=physiowave_eeg_prep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

DATASET="${DATASET:-}"
case "${DATASET}" in
    faced|tdbrain|physionet_mi|m3cv|hbn|hgd|siena) ;;
    tueg) echo "ERROR: TUEG uses cineca_tueg_preprocess.sbatch." >&2; exit 1 ;;
    "")  echo "ERROR: name the corpus:" >&2
         echo "  sbatch --export=ALL,DATASET=hbn --array=0-63 \\" >&2
         echo "         scripts/slurm/cineca_eeg_corpus_preprocess.sbatch" >&2
         exit 1 ;;
    *)   echo "ERROR: unknown DATASET '${DATASET}'." >&2; exit 1 ;;
esac
export DATASET

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"

# The preparation venv, not the training one: this step needs mne and needs no
# CUDA. PW_VARS_ONLY inside preprocess_eeg_corpus.sh keeps cineca_env.sh from
# replacing it.
if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
else
    echo "ERROR: no preparation venv at ${PWPREP:-${HOME}/pwprep}." >&2
    echo "       It is the one with mne; training's venv does not have it." >&2
    exit 1
fi

# One thread per worker process, not per task: numpy would otherwise start
# SLURM_CPUS_PER_TASK threads inside each of JOBS processes and oversubscribe
# the node by that product.
export JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# THE SHARD COUNT IS NOT THE NUMBER OF TASKS IN THIS ARRAY. It is the number
# the corpus was partitioned into, and a resubmission of two failed tasks out of
# sixteen has to use sixteen, not two.
#
# Deriving it from the array bounds works for --array=0-15 and silently breaks
# for --array=4,11: MIN=4 and MAX=11 gave 8, so task 4 processed shard 4-of-8 --
# a different set of subjects, the union of shards 4 and 12 of the original
# sixteen -- and task 11 got shard 11-of-8, which is not a shard, and died on
# the argument check. Shard 11 of the original sixteen was then processed by
# nobody.
#
# So: default to MAX+1, which is right for the --array=0-N form, and refuse to
# guess for anything else. Resubmitting a subset REQUIRES saying what the
# partition was:
#
#   sbatch --export=ALL,N_SHARDS=16 --array=4,11 \
#          scripts/slurm/cineca_tueg_preprocess.sbatch
_contiguous_from_zero=0
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" -eq 0 ]] && \
   [[ $(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 )) \
      -eq "${SLURM_ARRAY_TASK_COUNT:-0}" ]]; then
    _contiguous_from_zero=1
fi
if [[ -z "${N_SHARDS:-}" ]]; then
    if [[ "${_contiguous_from_zero}" -eq 1 ]]; then
        N_SHARDS=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))
    else
        echo "ERROR: this array is not --array=0-N, so the shard count cannot" >&2
        echo "       be read off its bounds. Task ids ${SLURM_ARRAY_TASK_MIN:-?}" >&2
        echo "       to ${SLURM_ARRAY_TASK_MAX:-?}, ${SLURM_ARRAY_TASK_COUNT:-?} task(s)." >&2
        echo "" >&2
        echo "       Say what the corpus was partitioned into:" >&2
        echo "         sbatch --export=ALL,N_SHARDS=16 --array=${SLURM_ARRAY_TASK_MIN:-4},... \\" >&2
        echo "                scripts/slurm/cineca_eeg_corpus_preprocess.sbatch" >&2
        echo "" >&2
        echo "       Guessing here would put a task on a different set of" >&2
        echo "       subjects than the run it is meant to be finishing." >&2
        exit 1
    fi
fi
if [[ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${N_SHARDS}" ]]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is out of range for ${N_SHARDS}" >&2
    echo "       shard(s). N_SHARDS must be the partition the corpus was" >&2
    echo "       split into, not the number of tasks resubmitted." >&2
    exit 1
fi
export SHARD="${SLURM_ARRAY_TASK_ID:-0}/${N_SHARDS}"

echo "============================================================"
echo "  ${DATASET} preprocessing  (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
echo "  shard ${SHARD}   cpus ${SLURM_CPUS_PER_TASK:-?}   workers ${JOBS}"
echo "  array ${SLURM_ARRAY_TASK_MIN:-?}-${SLURM_ARRAY_TASK_MAX:-?} (${SLURM_ARRAY_TASK_COUNT:-?} task(s))   N_SHARDS=${N_SHARDS}"
echo "  partition ${SLURM_JOB_PARTITION:-?}"
echo "  python $(python -c 'import sys;print(sys.prefix)')"
echo "============================================================"

bash EEG/preprocess_eeg_corpus.sh
_rc=$?

echo ""
echo "============================================================"
echo "  task ${SLURM_ARRAY_TASK_ID:-?} exit ${_rc}"
echo "  When every task has finished:"
echo "    python scripts/build_eeg_c1_manifest.py \\"
echo "        --corpus-root \${DATA_ROOT:-\${PW_DATA_EEG}/eeg_c1_corpus} \\"
echo "        --datasets \${DATASET} --allow-missing --check-shards"
echo "============================================================"
exit "${_rc}"
