#!/bin/bash
# ============================================================================
# TUEG preprocessing, as a SLURM array.
#
#   INSPECT first, on a login node, and read the report:
#     INSPECT=200 bash EEG/preprocess_tueg.sh
#
#   then:
#     sbatch --array=0-63 scripts/slurm/cineca_tueg_preprocess.sbatch
#
# The array index IS the shard index and --array's length is the shard count,
# so the two cannot drift apart. 64 tasks is a starting point; TUEG's per-file
# cost is dominated by reading and resampling, so tasks scale nearly linearly.
#
# NO GPU IS REQUESTED, on the partition everything else here uses. CINECA bills
# in CPU-hours and counts a GPU as eight of them, so a task asking for 8 cores
# and no GPU costs 8 CPU-hours an hour -- a quarter of what a 4-GPU training
# job on the same partition costs, and nothing is spent on accelerators this
# job would not touch. It decodes EDF, filters and resamples; there is no
# tensor in it.
#
# If the scheduler refuses a Booster job with no GPU, add --gres=gpu:1: it
# still bills as 8 CPU-hours an hour and simply leaves the card idle.
#
# dcgp_usr_prod, Leonardo's CPU partition, is the better home for this if the
# project has an allocation there -- check with `saldo -b` and pass
# --partition=dcgp_usr_prod. Not the default, because a partition the account
# cannot use is a submission that fails at 3am.
#
# MEMORY. The first run of this array was killed by the OOM killer on every one
# of its sixteen tasks -- OUT_OF_MEMORY, exit 0:125, after 7 to 27 minutes --
# because get_data() returns float64 and TUEG holds multi-hour recordings, so
# one worker could be holding twenty gigabytes while seven others did the same.
# --max-recording-minutes now caps what any worker holds at thirty minutes,
# which is about two gigabytes at the worst rate and channel count in this
# corpus. 120 GB for eight workers is the headroom on top of that; the node has
# 494 GB, so this is proportionate to the eight cores of thirty-two.
#
# Watch for it: `sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS -X`.
# A task that dies this way writes no manifest, so its shards are on disk and
# invisible to the merge until a resumed run finishes the summary.
#
# CORES ARE USED, not merely held. The filtering and resampling are
# single-threaded, so without --jobs a task decodes on one core and idles the
# rest. JOBS defaults to the cores the task was given.
#
# RESUMABLE. A shard whose HDF5 exists is reused, so resubmitting the same array
# after a walltime finishes what is left. Sharding is by subject, so tasks never
# touch each other's files and the train/val side of a subject is the same in
# every task.
#
# AFTER THE ARRAY, merge -- and only then is the corpus usable:
#   python scripts/build_eeg_c1_manifest.py --corpus-root $DATA_ROOT \
#       --datasets tueg --allow-missing --check-shards
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --job-name=physiowave_tueg_prep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"

# The preparation venv, not the training one: this step needs mne and needs no
# CUDA. PW_VARS_ONLY inside preprocess_tueg.sh keeps cineca_env.sh from
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
        echo "                scripts/slurm/cineca_tueg_preprocess.sbatch" >&2
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
echo "  TUEG preprocessing  (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
echo "  shard ${SHARD}   cpus ${SLURM_CPUS_PER_TASK:-?}   workers ${JOBS}"
echo "  array ${SLURM_ARRAY_TASK_MIN:-?}-${SLURM_ARRAY_TASK_MAX:-?} (${SLURM_ARRAY_TASK_COUNT:-?} task(s))   N_SHARDS=${N_SHARDS}"
echo "  partition ${SLURM_JOB_PARTITION:-?}"
echo "  python $(python -c 'import sys;print(sys.prefix)')"
echo "============================================================"

bash EEG/preprocess_tueg.sh
_rc=$?

echo ""
echo "============================================================"
echo "  task ${SLURM_ARRAY_TASK_ID:-?} exit ${_rc}"
echo "  When every task has finished:"
echo "    python scripts/build_eeg_c1_manifest.py \\"
echo "        --corpus-root \${DATA_ROOT:-\${PW_DATA_EEG}/eeg_c1_corpus} \\"
echo "        --datasets tueg --allow-missing --check-shards"
echo "============================================================"
exit "${_rc}"
