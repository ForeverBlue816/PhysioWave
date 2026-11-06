#!/bin/bash
# ============================================================================
# Unpack a large (and possibly encrypted) archive, as a batch job.
#
#   sbatch --export=ALL,ARCHIVE=/path/TDBRAIN.zip,DEST=/path/TDBRAIN/raw,PWENV=TDBRAIN_PW \
#          scripts/slurm/cineca_unpack_archive.sbatch
#
# Pass the password through the environment, never on the command line:
#
#   TDBRAIN_PW='...' sbatch --export=ALL,ARCHIVE=...,DEST=...,PWENV=TDBRAIN_PW \
#          scripts/slurm/cineca_unpack_archive.sbatch
#
# WHY THIS IS NOT A LOGIN-NODE COMMAND. Python's ZipCrypto decryption is a pure
# Python loop -- a few MB/s -- so TDBRAIN's encrypted 14.5 GB takes over an hour
# of solid CPU. A login node kills that, which is exactly what "Killed" at 4.5%
# was. As a job it gets a walltime and a log instead.
#
# RESUMABLE. unpack_archive.py skips entries already on disk at their recorded
# size and redoes the rest, so a killed run resumes by resubmitting rather than
# starting from zero. A file cut short mid-write has the wrong size and is
# redone, so resume cannot leave a truncated recording behind.
#
# FASTER ALTERNATIVE, if the site has p7zip: `7z` decrypts in C rather than in
# Python and is many times quicker.
#
#     7z x /path/TDBRAIN.zip -o/path/TDBRAIN/raw -p'<password>'
#
# It is not the default here because 7z is not always installed, and because
# unpack_archive.py checks the archive is neither truncated nor a bomb and
# refuses path traversal -- checks that matter more on an archive whose whole
# point is to be extracted somewhere important.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --job-name=physiowave_unpack
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"

if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
fi

ARCHIVE="${ARCHIVE:-}"
DEST="${DEST:-}"
PWENV="${PWENV:-}"

if [[ -z "${ARCHIVE}" || -z "${DEST}" ]]; then
    echo "ERROR: set ARCHIVE and DEST." >&2
    echo "  sbatch --export=ALL,ARCHIVE=/path/x.zip,DEST=/path/out \\" >&2
    echo "         scripts/slurm/cineca_unpack_archive.sbatch" >&2
    exit 1
fi
if [[ ! -f "${ARCHIVE}" ]]; then
    echo "ERROR: no such archive: ${ARCHIVE}" >&2
    exit 1
fi

ARGS=("${ARCHIVE}" --extract-to "${DEST}")
if [[ -n "${PWENV}" ]]; then
    if [[ -z "${!PWENV:-}" ]]; then
        echo "ERROR: PWENV=${PWENV} but \$${PWENV} is empty in the job." >&2
        echo "       Pass it through at submission time:" >&2
        echo "         ${PWENV}='...' sbatch --export=ALL,...,PWENV=${PWENV} ..." >&2
        exit 1
    fi
    ARGS+=(--password-env "${PWENV}")
fi

mkdir -p "${DEST}"
echo "============================================================"
echo "  unpack   job ${SLURM_JOB_ID:-?}"
echo "  archive  ${ARCHIVE}  ($(du -h "${ARCHIVE}" | cut -f1))"
echo "  dest     ${DEST}"
echo "  password $([[ -n "${PWENV}" ]] && echo "from \$${PWENV}" || echo "none")"
echo "============================================================"

python scripts/unpack_archive.py "${ARGS[@]}"
_rc=$?

echo ""
echo "============================================================"
echo "  exit ${_rc}   $(du -sh "${DEST}" 2>/dev/null | cut -f1) in ${DEST}"
if [[ "${_rc}" -ne 0 ]]; then
    echo "  Resubmit the same command to resume: entries already on disk at"
    echo "  their recorded size are skipped."
fi
echo "============================================================"
exit "${_rc}"
