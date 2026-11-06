#!/bin/bash
# =============================================================================
# Sleep-EDF channel-embedding ablation, as a batch job.
#
#   sbatch scripts/slurm/cineca_sleep_channel_ablation.sbatch
#
# Overrides go through --export, same as the other sbatch scripts here:
#
#   sbatch --export=ALL,VARIANTS=C0,C4,SEEDS=42,FOLDS=0,EPOCHS=2 \
#          scripts/slurm/cineca_sleep_channel_ablation.sbatch
#
# Note the commas: --export takes a comma-separated list, and VARIANTS=C0,C4 is
# parsed as VARIANTS=C0 followed by a bare C4. Quote the whole assignment when a
# value contains commas:
#
#   sbatch --export=ALL,"VARIANTS=C0,C1,C2,C3,C4,C5","SEEDS=42,43,44" \
#          scripts/slurm/cineca_sleep_channel_ablation.sbatch
#
# RESUMABLE. SWEEP_ROOT does not carry the job id, and the runner treats a run
# as finished only when it wrote test_results.json. So a job that hits the
# walltime mid-sweep loses at most the run it was in, and resubmitting the same
# command picks up where it stopped. That is the intended way to work through
# the full matrix: submit, let it run out, submit again.
#
# COST. Six variants x three seeds x one fold is eighteen runs, on the order of
# six hours at the Sleep-EDF defaults on four GPUs. Ten folds is ten times that,
# which is several jobs. --time below is a walltime, not a budget: raise it and
# you wait longer in the queue, so resubmitting is usually faster than asking
# for twenty-four hours up front.
#
# The decode cache is NOT rebuilt. This job builds each fold's HDF5 from the
# existing $PW_DATA_EEG/sleep_edf/cache, which is file assembly, and that step
# needs the channel metadata writer -- so it needs mne. If mne is missing the
# split is built without metadata and every variant except C0 refuses to start;
# build the splits first under $HOME/pwprep if that happens.
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=physiowave_sleep_channel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"

# sbatch defaults to --export=ALL, so this job inherits the submitting shell's
# environment -- including a PW_VENV left pinned to $HOME/pwprep by
# `PW_VARS_ONLY=1 source scripts/cineca_env.sh`. That pin would send training to
# the preparation venv. Honour a pin that was passed deliberately on the command
# line, drop one that was merely inherited.
if [[ -n "${PW_VENV:-}" && "${PW_VENV}" != "${VENV:-${HOME}/pw}" \
      && "${PW_ALLOW_INHERITED_VENV:-0}" != "1" ]]; then
    echo "NOTE: ignoring inherited PW_VENV=${PW_VENV}; using ${VENV:-${HOME}/pw}." >&2
    echo "      PW_ALLOW_INHERITED_VENV=1 to keep it." >&2
    unset PW_VENV
fi

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

# Fail in seconds on the wrong interpreter, rather than after the splits are
# built and once per run.
pw_require_python_deps || exit 1

export NUM_GPUS="${NUM_GPUS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / NUM_GPUS))}"

export VARIANTS="${VARIANTS:-C0,C1,C2,C3,C4,C5}"
export SEEDS="${SEEDS:-42,43,44}"
export FOLDS="${FOLDS:-0}"
export DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/sleep_edf_channel}"
export CACHE_DIR="${CACHE_DIR:-${PW_DATA_EEG}/sleep_edf/cache}"
export SWEEP_ROOT="${SWEEP_ROOT:-${PW_CKPT_ROOT}/sleep_channel_ablation}"

echo "============================================================"
echo "  Sleep-EDF channel-embedding ablation  (job ${SLURM_JOB_ID})"
echo "  variants=${VARIANTS}  seeds=${SEEDS}  folds=${FOLDS}"
echo "  epochs=${EPOCHS:-<script default>}  gpus=${NUM_GPUS}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
pw_print_layout
echo "  SWEEP_ROOT=${SWEEP_ROOT}   (resubmit to resume)"
echo "============================================================"

bash EEG/run_sleep_channel_ablation.sh
_rc=$?

echo ""
echo "============================================================"
echo "  Summarise:  python scripts/collect_channel_ablation.py ${SWEEP_ROOT}"
echo "  Resume:     sbatch scripts/slurm/cineca_sleep_channel_ablation.sbatch"
echo "============================================================"
exit "${_rc}"
