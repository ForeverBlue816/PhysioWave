#!/bin/bash
# ============================================================================
# HBN download -- READ THIS BEFORE RUNNING IT.
#
# This moves a lot of data through a login node. One transfer here was
# SIGKILLed partway (R5, at 164 of 224 GB), so a limit of some kind is enforced.
#
# An earlier version treated that kill as an obstacle and looped to restart
# after each one. That was wrong independently of what the limit is for: a
# process killed by the node is being stopped on purpose, and a loop that
# restarts it is working around a control rather than reporting it. It STOPS on
# a signal now.
#
# The situation this was written for is real -- compute nodes here have no route
# to the internet, so no batch job can download -- and the answer is to use a
# host meant for transfers. CINECA has Leonardo Datamovers; most sites have
# something equivalent.
#
# WHAT TO DO INSTEAD, in order:
#   1. Ask support for the data-transfer host or service for your project.
#   2. If a login node is genuinely the sanctioned route, ask what size and
#      duration are acceptable and stay inside it -- MAX_GB_PER_RUN below.
#   3. Transfer to a machine that is allowed to, then move it in.
#
#   bash scripts/download_hbn_login.sh R10 R11
#
# ENVIRONMENT:
#   EEG_ROOT         corpus root
#   AWS_CONCURRENCY  transfers in flight (default 3)
#   MAX_GB_PER_RUN   stop after roughly this much in one invocation (default
#                    50). A ceiling you can point at in a support request,
#                    rather than an open-ended transfer.
#   I_HAVE_CHECKED_THE_POLICY=1   required to run at all.
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EEG_ROOT="${EEG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/eeg}"
export EEG_ROOT
export AWS_CONCURRENCY="${AWS_CONCURRENCY:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"

RELEASES="$*"
# Nine, not eleven: only R1-R9 are mirrored to fcp-indi.
[[ -z "${RELEASES}" ]] && RELEASES="R1 R2 R3 R4 R5 R6 R7 R8 R9"
MAX_GB_PER_RUN="${MAX_GB_PER_RUN:-50}"

if [[ "${I_HAVE_CHECKED_THE_POLICY:-0}" != "1" ]]; then
    cat >&2 <<'MSG'
NOTE: this moves tens to hundreds of gigabytes through a login node.

The only hard evidence about this system is that a transfer WAS SIGKILLed
partway through (R5, at 164 of 224 GB), so some limit is enforced there. If
your site has a data-transfer host, it is the better route; CINECA's
Leonardo Datamovers are one.

Set I_HAVE_CHECKED_THE_POLICY=1 to proceed:

    I_HAVE_CHECKED_THE_POLICY=1 MAX_GB_PER_RUN=50 bash "$0" R10 R11

MSG
    exit 1
fi

if ! aws --version >/dev/null 2>&1; then
    echo "ERROR: the AWS CLI does not run. HBN is S3-only." >&2
    echo "  source \$HOME/pwprep/bin/activate && pip install awscli" >&2
    exit 1
fi

#: Set by s3_bytes when it fails, so the caller can say WHY rather than only
#: that it could not read the size. 2>/dev/null here was the same mistake as the
#: listing probe: it made "the release does not exist" and "the network was
#: down" indistinguishable, and they need opposite responses.
S3_ERR=""

s3_bytes() {
    local out err rc
    err=$(mktemp "${TMPDIR:-/tmp}/pw_s3sz.XXXXXX" 2>/dev/null) || err=/dev/null
    out=$(aws s3 ls --recursive --summarize --no-sign-request \
        "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_$1/" 2>"${err}")
    rc=$?
    S3_ERR=""
    if [[ "${rc}" -ne 0 ]]; then
        S3_ERR="aws exit ${rc}: $(head -2 "${err}" 2>/dev/null | tr '\n' ' ')"
    elif [[ -z "${out}" ]]; then
        S3_ERR="the prefix listed nothing -- cmi_bids_$1 may not exist"
    fi
    [[ "${err}" != /dev/null ]] && rm -f "${err}"
    printf '%s' "$(printf '%s' "${out}" | awk '/Total Size/ {print $3}')"
}

#: Which release directories the bucket actually holds. Printed when one cannot
#: be read, because "R11 is missing" and "R11 is called something else" look the
#: same from inside a loop over names someone wrote down.
list_releases() {
    aws s3 ls "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/" --no-sign-request \
        2>/dev/null | awk '{print $NF}' | tr -d '/' | grep -i '^cmi_bids' || true
}

_STAT_FLAG=""
if stat -c%s "${BASH_SOURCE[0]}" >/dev/null 2>&1; then _STAT_FLAG="-c%s"
elif stat -f%z "${BASH_SOURCE[0]}" >/dev/null 2>&1; then _STAT_FLAG="-f%z"
fi

local_bytes() {
    [[ -z "${_STAT_FLAG}" ]] && { echo 0; return; }
    find -L "$1" -type f -print0 2>/dev/null \
        | xargs -0 stat "${_STAT_FLAG}" 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

human() { awk -v b="$1" 'BEGIN {printf "%.1f GB", b/1073741824}'; }

echo "============================================================"
echo "  HBN via login node   $(date '+%F %H:%M:%S')"
echo "  releases    ${RELEASES}"
echo "  into        ${EEG_ROOT}/HBN/raw"
echo "  concurrency ${AWS_CONCURRENCY}   max rounds ${MAX_ROUNDS}"
echo "  pid         $$"
echo "============================================================"

_failed=""
for r in ${RELEASES}; do
    dest="${EEG_ROOT}/HBN/raw/${r}"
    mkdir -p "${dest}"

    want=$(s3_bytes "${r}")
    if [[ -z "${want}" || "${want}" -eq 0 ]]; then
        echo ""
        echo "### ${r}: cannot read the remote size -- skipping for now"
        [[ -n "${S3_ERR}" ]] && echo "    ${S3_ERR}"
        if [[ "${S3_ERR}" == *"may not exist"* ]]; then
            echo "    Release directories the bucket actually holds:"
            list_releases | sed 's/^/      /'
        fi
        _failed="${_failed} ${r}"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "### ${r}   target $(human "${want}")   $(date '+%H:%M:%S')"
    echo "############################################################"

    round=1
    start_bytes=$(local_bytes "${dest}")
    while [[ "${round}" -le "${MAX_ROUNDS}" ]]; do
        have=$(local_bytes "${dest}")
        if [[ "${have}" -ge "${want}" ]]; then
            echo "    complete: $(human "${have}") / $(human "${want}")"
            break
        fi
        pct=$(awk -v a="${have}" -v b="${want}" 'BEGIN {printf "%.1f", 100*a/b}')
        echo "    round ${round}: $(human "${have}") / $(human "${want}") (${pct}%)  $(date '+%H:%M:%S')"

        bash scripts/download_eeg_pretrain_corpora.sh hbn "${r}"
        rc=$?

        after=$(local_bytes "${dest}")

        # A signal is the SITE STOPPING YOU. Looping past it is circumventing a
        # resource control, which is how an account gets suspended. Stop, and
        # say what to ask for.
        if [[ "${rc}" -eq 137 || "${rc}" -eq 143 || "${rc}" -eq 130 ]]; then
            echo "" >&2
            echo "    KILLED (signal, exit ${rc}). STOPPING." >&2
            echo "    A login node killing this is the site enforcing a limit," >&2
            echo "    not a transient error. Do not restart it in a loop." >&2
            echo "    Ask your support desk for the data-transfer route before" >&2
            echo "    continuing. $(human "${after}") is on disk and is intact." >&2
            _failed="${_failed} ${r}"
            exit "${rc}"
        fi

        # A self-imposed ceiling, so one invocation is something you can
        # describe to a support desk rather than an open-ended transfer.
        moved=$(( (after - start_bytes) / 1073741824 ))
        if [[ "${moved}" -ge "${MAX_GB_PER_RUN}" ]]; then
            echo "" 
            echo "    reached MAX_GB_PER_RUN=${MAX_GB_PER_RUN} GB this run; stopping."
            echo "    Rerun later to continue; everything on disk is kept."
            exit 0
        fi

        if [[ "${after}" -le "${have}" ]] && [[ "${rc}" -ne 0 ]]; then
            echo "    no progress this round (exit ${rc}); pausing 60s" >&2
            sleep 60
        fi
        round=$(( round + 1 ))
    done

    have=$(local_bytes "${dest}")
    if [[ "${have}" -ge "${want}" ]]; then
        echo "    ${r} DONE"
    else
        echo "    ${r} still short after ${MAX_ROUNDS} round(s)" >&2
        _failed="${_failed} ${r}"
    fi
done

echo ""
echo "============================================================"
echo "  finished $(date '+%F %H:%M:%S')"
for r in ${RELEASES}; do
    d="${EEG_ROOT}/HBN/raw/${r}"
    printf "  %-4s %12s\n" "${r}" "$(human "$(local_bytes "${d}")")"
done
if [[ -n "${_failed}" ]]; then
    echo ""
    echo "  incomplete:${_failed}"
    echo "  Rerun -- everything already on disk is skipped:"
    echo "    nohup bash scripts/download_hbn_login.sh${_failed} > ~/hbn.log 2>&1 &"
else
    echo ""
    echo "  every release complete. Confirm against the bucket:"
    echo "    bash scripts/download_eeg_pretrain_corpora.sh verify-hbn"
fi
echo "============================================================"
