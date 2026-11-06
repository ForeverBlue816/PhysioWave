#!/bin/bash
# ============================================================================
#
# !! COMPUTE NODES ON THIS SYSTEM HAVE NO ROUTE TO THE PUBLIC INTERNET. !!
# An aws transfer submitted here fails with "Could not connect to the endpoint
# URL" and leaves an empty directory. Downloads must run on a LOGIN NODE:
#
#     nohup bash scripts/download_hbn_login.sh > ~/hbn.log 2>&1 &
#
# That script is built around the login node's process limits -- low
# concurrency, resumable, and it reruns after being killed. This sbatch is kept
# only for a site where compute nodes can reach S3.
#
# HBN download, as a batch job.
#
#   sbatch --export=ALL,RELEASES=R3          scripts/slurm/cineca_hbn_download.sbatch
#   sbatch --export=ALL,RELEASES="R1 R2 R4"  scripts/slurm/cineca_hbn_download.sbatch
#   sbatch --export=ALL,RELEASES=all         scripts/slurm/cineca_hbn_download.sbatch
#
# WHY THIS IS NOT A LOGIN-NODE COMMAND. R1-R11 is 1.875 TB and a single release
# is 91-245 GB. A transfer that runs for hours on a login node is the kind of
# process a site kills, and losing it at 80% costs the whole afternoon. As a
# job it has a walltime, a log, and a resume.
#
# RESUMABLE, because it uses `aws s3 sync`: a file already present with a
# matching size is skipped. Resubmit the same command after a walltime and it
# picks up where it stopped. Whole releases already complete cost one listing.
#
# NO GPU, and eight cores is plenty -- this is network and filesystem, not
# compute. It bills as 8 CPU-hours an hour, the same as the preprocessing array.
#
# DISK FIRST. Check there is room before starting, because a transfer that fills
# the filesystem takes the rest of the project down with it:
#   cinecaquota ; df -h $EEG_ROOT
#
# R1-R9 sizes, GB:  103 120 140 230 224 91 245 157 185
# (only nine releases are mirrored to fcp-indi; the table's R10/R11 are not)
#
# START WITH R3 (140 GB). It is enough to prove the adapter reads this corpus,
# and every one of the five datasets before it needed a fix that only showed up
# on real data.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --job-name=physiowave_hbn_dl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"

if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
fi

export EEG_ROOT="${EEG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/eeg}"
# A compute node has room for more parallelism than a login node, which is
# where the conservative default comes from.
export AWS_CONCURRENCY="${AWS_CONCURRENCY:-8}"
RELEASES="${RELEASES:-R3}"

if ! aws --version >/dev/null 2>&1; then
    echo "ERROR: the AWS CLI does not run. HBN is S3-only." >&2
    echo "       If 'aws' is on PATH but dies on a print statement, it is the" >&2
    echo "       abandoned Python-2 package of that name, not awscli:" >&2
    echo "         pip uninstall -y aws && pip install awscli" >&2
    exit 1
fi

if [[ "${RELEASES}" == "all" ]]; then
    RELEASES="R1 R2 R3 R4 R5 R6 R7 R8 R9"
fi

echo "============================================================"
echo "  HBN download   job ${SLURM_JOB_ID:-?}"
echo "  releases  ${RELEASES}"
echo "  into      ${EEG_ROOT}/HBN/raw"
echo "  aws       $(aws --version 2>&1)"
echo "  free      $(df -h "${EEG_ROOT}" 2>/dev/null | tail -1)"
echo "============================================================"

_fail=0
for r in ${RELEASES}; do
    echo ""
    echo "### ${r}  $(date '+%H:%M:%S')"
    bash scripts/download_eeg_pretrain_corpora.sh hbn "${r}" || _fail=$((_fail + 1))
done

echo ""
echo "============================================================"
echo "  ${_fail} release(s) reported an error"
if [[ "${_fail}" -gt 0 ]]; then
    echo ""
    # Single quotes: inside double quotes bash runs backticks, and this line
    # had `aws s3 sync` in it -- so printing the explanation executed the
    # command it was describing.
    echo '  A release that downloaded nothing counts as an error here even when'
    echo '  aws exited 0 -- "aws s3 sync" against a prefix that does not exist'
    echo '  does exactly that, and leaves a directory indistinguishable from a'
    echo '  finished one.' 
fi
echo "  total $(du -sh "${EEG_ROOT}/HBN/raw" 2>/dev/null | cut -f1)"
echo ""
echo "  NEXT, and do not skip it -- every corpus before this one needed a fix"
echo "  that only appeared on real data:"
echo "    bash scripts/download_eeg_pretrain_corpora.sh verify-hbn"
echo "    DATASET=hbn INSPECT=60 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh"
echo "============================================================"
exit "${_fail}"
