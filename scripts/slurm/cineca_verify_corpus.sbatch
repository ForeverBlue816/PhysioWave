#!/bin/bash
# ============================================================================
# Read every shard of the EEG C1 corpus and rebuild the merged manifests from
# the ones that survive.
#
#   sbatch scripts/slurm/cineca_verify_corpus.sbatch                 # report only
#   sbatch --export=ALL,DROP=1 scripts/slurm/cineca_verify_corpus.sbatch
#
# WHY THIS EXISTS. `f["data"].shape` is answered from the HDF5 object header,
# which is written before any chunk, so a preprocessing task killed part-way
# through a write leaves a shard that every metadata check passes and that
# raises
#
#     OSError: Can't synchronously read data (inflate() failed)
#
# the first time training reads one of its windows -- taking all sixteen ranks
# down with it, hours in. The only check that can tell such a shard from a good
# one is inflating every chunk, which is what --check-level full does and what
# this job is for.
#
# COST. ~95,000 shards, ~1.5 TB compressed. gzip level 4 inflates at roughly
# 200 MB/s per core, so 32 workers is CPU-bound at well under an hour; the wall
# clock below is generous for a cold filesystem cache.
#
# DROP=1 writes manifests without the bad shards and lists them in
# merged/unreadable_shards.jsonl. Leave it unset first and READ THE REPORT: a
# handful of unreadable shards is a preprocessing task to re-run, but hundreds
# means something happened to the filesystem and dropping them silently loses
# a slice of the corpus.
#
# The bad shards can be rebuilt instead of dropped -- delete them and re-run
# the preprocessing array with --resume, which now re-derives any shard whose
# data does not inflate rather than counting it as done.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120000
#SBATCH --job-name=physiowave_verify_corpus
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1

# h5py and numpy only -- the preparation venv has them and needs no CUDA.
if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
elif [[ -f "${PW_VENV:-${HOME}/pw}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PW_VENV:-${HOME}/pw}/bin/activate"
else
    echo "ERROR: no venv at ${PWPREP:-${HOME}/pwprep} or ${PW_VENV:-${HOME}/pw}." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATA_ROOT="${DATA_ROOT:-${EEG_ROOT:-${WORK}/eeg}/eeg_c1_corpus}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-32}}"
LEVEL="${CHECK_LEVEL:-full}"

DROP_ARG=()
[[ "${DROP:-0}" == "1" ]] && DROP_ARG=(--drop-unreadable)

echo "============================================================"
echo "  corpus verification  (job ${SLURM_JOB_ID})"
echo "  DATA_ROOT   ${DATA_ROOT}"
echo "  level       ${LEVEL}   workers ${JOBS}"
echo "  drop        ${DROP:-0}"
echo "============================================================"

python scripts/build_eeg_c1_manifest.py \
    --corpus-root "${DATA_ROOT}" \
    --datasets tueg,faced,tdbrain,physionet_mi,m3cv,hbn,hgd \
    --allow-missing --check-shards --check-level "${LEVEL}" \
    --jobs "${JOBS}" "${DROP_ARG[@]}"
rc=$?

echo ""
if [[ ${rc} -eq 0 ]]; then
    echo "corpus verified; manifests in ${DATA_ROOT}/merged"
else
    echo "unreadable shards found. Which ones:"
    echo "  cat ${DATA_ROOT}/merged/unreadable_shards.jsonl"
    echo "  cat ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err     # the first eight"
    echo ""
    echo "A handful means a preprocessing task was killed part-way through."
    echo "Delete those shards and re-run the preprocessing array with --resume,"
    echo "which now rebuilds any shard whose data does not inflate; or, if the"
    echo "windows are not worth another pass, drop them and merge without them:"
    echo "  sbatch --export=ALL,DROP=1 scripts/slurm/cineca_verify_corpus.sbatch"
fi
exit ${rc}
