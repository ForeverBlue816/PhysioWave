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
# Fetch every pretraining corpus that is not already on disk, in one job.
#
#   sbatch scripts/slurm/cineca_fetch_corpora.sbatch
#   sbatch --export=ALL,DATASETS="hbn" scripts/slurm/cineca_fetch_corpora.sbatch
#   TDBRAIN_PW='...' sbatch --export=ALL scripts/slurm/cineca_fetch_corpora.sbatch
#
# IDEMPOTENT BY DESIGN. Every dataset is checked before it is fetched and
# skipped if the recordings are already there, so this is safe to resubmit and
# safe to run while other datasets are already preprocessing. HBN resumes
# release by release; the NEMAR zips are not re-downloaded once unpacked.
#
# RESUBMIT TO CONTINUE. HBN alone is 1.875 TB, which needs a sustained 22 MB/s
# to fit in this walltime and will not always get it. The job reports exactly
# what is still missing and resubmitting picks up from there.
#
# TDBRAIN needs its password from the Data Use Agreement email, passed through
# the environment so it never appears in `ps` or in the job record:
#   TDBRAIN_PW='...' sbatch --export=ALL scripts/slurm/cineca_fetch_corpora.sbatch
# Without it, TDBRAIN's zip is downloaded-if-present but left unpacked, and the
# job says so rather than failing everything else.
#
# NOT INCLUDED: TUEG, which is already preprocessed, and any corpus behind a
# login this cannot perform. TDBRAIN's zip itself must be fetched by hand --
# accepting its data use agreement is not something a batch job does on
# someone's behalf.
#
# ENVIRONMENT:
#   DATASETS   space-separated subset (default: all of them)
#   EEG_ROOT   corpus root (default: the project scratch path)
#   TDBRAIN_PW password for TDBRAIN's archive, if it needs unpacking
#   HBN_RELEASES  which HBN releases (default: all, R1-R11)
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --job-name=physiowave_fetch
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
DATASETS="${DATASETS:-physionet_mi faced m3cv hgd tdbrain hbn}"
HBN_RELEASES="${HBN_RELEASES:-all}"

DL=scripts/download_eeg_pretrain_corpora.sh

# --------------------------------------------------------------------------- #
# What counts as "already there"
# --------------------------------------------------------------------------- #
# By RECORDING COUNT, not by directory existence. `bash ... layout` creates all
# six raw directories before anything is downloaded, and an aws sync against a
# bad prefix leaves one behind too, so a present directory proves nothing.
raw_dir() {
    case "$1" in
        physionet_mi) echo "${EEG_ROOT}/PhysioNetMI/raw" ;;
        faced)        echo "${EEG_ROOT}/FACED/raw" ;;
        m3cv)         echo "${EEG_ROOT}/M3CV/raw" ;;
        hgd)          echo "${EEG_ROOT}/HGD/raw" ;;
        tdbrain)      echo "${EEG_ROOT}/TDBRAIN/raw" ;;
        hbn)          echo "${EEG_ROOT}/HBN/raw" ;;
    esac
}

count_recordings() {
    find -L "$1" -type f \( -iname '*.edf' -o -iname '*.bdf' -o -iname '*.set' \
        -o -iname '*.fif' -o -iname '*.cnt' -o -iname '*.vhdr' \) 2>/dev/null \
        | wc -l | tr -d ' '
}

# Roughly what each corpus should hold once complete. Deliberately loose: this
# distinguishes "nothing there" and "a fraction there" from "done", not an exact
# inventory. verify-hbn does the exact comparison for the one that needs it.
expected_min() {
    case "$1" in
        physionet_mi) echo 1500 ;;
        faced)        echo 100 ;;
        m3cv)         echo 2000 ;;
        hgd)          echo 50 ;;
        tdbrain)      echo 1000 ;;
        # HBN is checked per release, not by this number -- see
        # hbn_missing_releases. A whole-corpus threshold cannot work here: R3
        # alone holds 1,812 recordings, so any total that R3 clears would call
        # a corpus missing ten of its eleven releases "already present".
        hbn)          echo 999999 ;;
    esac
}

#: Smallest release (R6, 91 GB) holds well over a thousand recordings at the
#: ~13 files/GB this corpus runs at, so a release under this floor is either
#: absent or barely started.
HBN_MIN_PER_RELEASE=200

hbn_missing_releases() {
    local want="$1" out="" r n
    [[ "${want}" == "all" ]] && want="R1 R2 R3 R4 R5 R6 R7 R8 R9"
    for r in ${want}; do
        n=$(count_recordings "${EEG_ROOT}/HBN/raw/${r}")
        if [[ "${n}" -lt "${HBN_MIN_PER_RELEASE}" ]]; then
            out="${out} ${r}"
        else
            echo "    ${r}: ${n} recording file(s) -- present" >&2
        fi
    done
    printf '%s' "${out# }"
}

echo "============================================================"
echo "  fetch corpora   job ${SLURM_JOB_ID:-?}"
echo "  root      ${EEG_ROOT}"
echo "  datasets  ${DATASETS}"
echo "  python    $(python -c 'import sys;print(sys.prefix)')"
echo "  aws       $(aws --version 2>&1 | head -1)"
echo "  free      $(df -h "${EEG_ROOT}" 2>/dev/null | tail -1)"
echo "============================================================"

# The AWS CLI is needed for HBN only, and the abandoned PyPI package named `aws`
# installs to the same bin/aws, so being on PATH proves nothing.
if [[ " ${DATASETS} " == *" hbn "* ]] && ! aws --version >/dev/null 2>&1; then
    echo "" >&2
    echo "WARNING: the AWS CLI does not run, so HBN will be skipped." >&2
    echo "         pip uninstall -y aws && pip install awscli" >&2
    echo "         Everything else continues." >&2
    DATASETS="${DATASETS//hbn/}"
    _hbn_no_aws=1
fi

# Status is accumulated as "id=text" lines rather than an associative array:
# bash 3.2 has no `declare -A`, and a summary that only works on the cluster is
# a summary nobody has ever seen run.
STATUS_LINES=""
_hbn_no_aws="${_hbn_no_aws:-0}"
set_status() { STATUS_LINES="${STATUS_LINES}$1=$2"$'\n'; }
get_status() {
    local v
    v=$(printf '%s' "${STATUS_LINES}" | awk -F= -v k="$1" \
        '$1==k {sub(/^[^=]*=/, ""); print; exit}')
    printf '%s' "${v:-not requested}"
}

[[ "${_hbn_no_aws}" -eq 1 ]] && \
    STATUS_LINES="hbn=SKIPPED -- the AWS CLI does not run"$'\n'

_t0=$SECONDS

for ds in ${DATASETS}; do
    dir="$(raw_dir "${ds}")"
    have=$(count_recordings "${dir}")
    want=$(expected_min "${ds}")

    echo ""
    echo "############################################################"
    echo "### ${ds}   have ${have} recording file(s), expect >= ${want}   $(date '+%H:%M:%S')"
    echo "############################################################"

    if [[ "${ds}" != "hbn" ]] && [[ "${have}" -ge "${want}" ]]; then
        echo "    already present -- skipping"
        set_status "${ds}" "skipped (${have} files)"
        continue
    fi

    case "${ds}" in
    tdbrain)
        # The archive must already be here: accepting the DUA is not something
        # a batch job does for someone.
        zip=$(find "${dir}" -maxdepth 1 -iname '*.zip' 2>/dev/null | head -1)
        if [[ -z "${zip}" ]]; then
            echo "    no TDBRAIN archive under ${dir}."
            echo "    It is behind an ORCID login and a data use agreement;"
            echo "    fetch it by hand, then resubmit this job."
            set_status "${ds}" "MISSING ARCHIVE"
            continue
        fi
        if [[ -z "${TDBRAIN_PW:-}" ]]; then
            echo "    archive is here but \$TDBRAIN_PW is not set, so it cannot"
            echo "    be unpacked. Resubmit with:"
            echo "      TDBRAIN_PW='...' sbatch --export=ALL $0"
            set_status "${ds}" "NEEDS PASSWORD"
            continue
        fi
        echo "    unpacking ${zip}"
        python scripts/unpack_archive.py "${zip}" --extract-to "${dir}" \
            --password-env TDBRAIN_PW
        rc=$?
        ;;
    hbn)
        echo "    checking releases individually"
        _todo=$(hbn_missing_releases "${HBN_RELEASES}")
        if [[ -z "${_todo}" ]]; then
            echo "    every requested release is present"
            set_status hbn "done (${have} files, all releases)"
            continue
        fi
        echo "    still to fetch:${_todo:+ }${_todo}"
        rc=0
        for _r in ${_todo}; do
            bash "${DL}" hbn "${_r}" || rc=1
        done
        ;;
    *)
        bash "${DL}" "${ds}"
        rc=$?
        ;;
    esac

    now=$(count_recordings "${dir}")
    if [[ "${ds}" == "hbn" ]]; then
        _left=$(hbn_missing_releases "${HBN_RELEASES}" 2>/dev/null)
        if [[ -z "${_left}" ]]; then
            set_status hbn "done (${now} files, all releases)"
        else
            set_status hbn "PARTIAL -- still missing:${_left:+ }${_left}"
        fi
        echo "    -> $(get_status "${ds}")"
        continue
    fi
    if [[ "${rc}" -eq 0 ]] && [[ "${now}" -ge "${want}" ]]; then
        set_status "${ds}" "done (${now} files)"
    elif [[ "${now}" -gt "${have}" ]]; then
        set_status "${ds}" "PARTIAL ${now}/${want} (exit ${rc}) -- resubmit to continue"
    else
        set_status "${ds}" "FAILED (exit ${rc}, still ${now} files)"
    fi
    echo "    -> $(get_status "${ds}")"
done

# --------------------------------------------------------------------------- #
echo ""
echo "============================================================"
echo "  FETCH SUMMARY after $(( (SECONDS - _t0) / 60 )) minute(s)"
echo "============================================================"
_incomplete=0
for ds in physionet_mi faced m3cv hgd tdbrain hbn; do
    st="$(get_status "${ds}")"
    printf "  %-14s %s\n" "${ds}" "${st}"
    [[ "${st}" == PARTIAL* || "${st}" == FAILED* || "${st}" == MISSING* \
       || "${st}" == NEEDS* ]] && _incomplete=$((_incomplete + 1))
done
echo ""
echo "  total on disk: $(du -sh "${EEG_ROOT}" 2>/dev/null | cut -f1)"

if [[ "${_incomplete}" -gt 0 ]]; then
    echo ""
    echo "  ${_incomplete} corpus/corpora incomplete. This job is idempotent --"
    echo "  resubmitting skips what is done and continues the rest."
fi

echo ""
echo "  NEXT, per corpus, and do not skip it. Every dataset so far needed a fix"
echo "  that only showed up on real data:"
echo "    DATASET=<id> INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh"
echo "  and for HBN specifically, the exact comparison against the bucket:"
echo "    bash scripts/download_eeg_pretrain_corpora.sh verify-hbn"
echo "============================================================"
exit "${_incomplete}"
