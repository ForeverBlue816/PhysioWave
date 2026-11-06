#!/bin/bash
# ============================================================================
# Fetch the ECG downstream benchmarks (PTB-XL, CPSC 2018, Chapman-Shaoxing).
#
#   source $HOME/pw/bin/activate
#   bash scripts/download_ecg_downstream.sh status
#   bash scripts/download_ecg_downstream.sh all
#   bash scripts/download_ecg_downstream.sh ptbxl | cpsc2018 | chapman
#
# RUN IT ON A LOGIN NODE (compute nodes have no internet). ~10 GB in all, and
# every step is resumable: a finished set carries .complete and is skipped, a
# rerun fetches only what is missing.
#
#   ptbxl     PTB-XL 1.0.3, records500 + the csv tables, from PhysioNet's open
#             S3 bucket (2.6 GB, minutes). Serves TASK=ptbxl and ptbxl_super.
#   chapman   PhysioNet ecg-arrhythmia 1.0.0 (Chapman-Shaoxing + Ningbo,
#             45,152 records), open S3 (5.5 GB, minutes).
#   cpsc2018  CPSC 2018 (6,877) + CPSC-Extra (3,453), 1.3 GB, from PhysioNet's
#             own uploads to Kaggle (needs the Kaggle CLI and a token, as for
#             MIMIC-IV-ECG). CPSC_SOURCE=physionet takes Challenge 2021's copy
#             over PhysioNet's HTTP server instead, ~20-40 KB/s a connection.
#
# LAYOUT -- what ECG/ecg_downstream_prep.py and the sbatch expect:
#   $ECG_ROOT/downstream/raw/ptbxl/{ptbxl_database.csv, scp_statements.csv, records500/}
#   $ECG_ROOT/downstream/raw/chapman/{WFDBRecords/, ConditionNames_SNOMED-CT.csv}
#   $ECG_ROOT/downstream/raw/cpsc2018/{Training_WFDB/, Training_2/}  (Kaggle)
#                                    or {cpsc_2018/g*, cpsc_2018_extra/g*} (PhysioNet)
#   $ECG_ROOT/downstream/c1/<task>/   written by the preprocessing
#
# None of these is in the ECG C1 pretraining corpus; preprocessing refuses
# them by name (physiowave/ecg_c1/routes.py, DOWNSTREAM_ONLY).
#
# LICENCES: PTB-XL CC BY 4.0; ecg-arrhythmia CC BY 4.0; Challenge 2021 data
# CC BY 4.0 (as PhysioNet lists them).
#
# ENVIRONMENT
#   ECG_ROOT   ($SCRATCH/bio/ecg)
#   JOBS       parallel downloads (16)
# ============================================================================

set -uo pipefail

ECG_ROOT="${ECG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/ecg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-16}"
SETS="ptbxl chapman cpsc2018"
CHALLENGE="${CHALLENGE:-https://physionet.org/files/challenge-2021/1.0.3/training}"

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

raw_dir() { echo "${ECG_ROOT}/downstream/raw/$1"; }
is_done() { [[ -f "$(raw_dir "$1")/.complete" ]]; }

get_ptbxl() {
    local d; d="$(raw_dir ptbxl)"
    "${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket physionet-open \
        --prefix ptb-xl/1.0.3/ --dest "${d}" \
        --include '(records500/.*\.(hea|dat)|\.csv|LICENSE\.txt)$' \
        --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}/records500" -name '*.hea' | wc -l)
    echo "    ${n} records (expected 21,799)"
    [[ ${n} -ge 21799 && -f "${d}/ptbxl_database.csv" ]]
}

get_chapman() {
    local d; d="$(raw_dir chapman)"
    "${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket physionet-open \
        --prefix ecg-arrhythmia/1.0.0/ --dest "${d}" \
        --include '(WFDBRecords/.*\.(hea|mat)|\.csv|LICENSE\.txt)$' \
        --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}/WFDBRecords" -name '*.hea' | wc -l)
    echo "    ${n} records (expected 45,152)"
    [[ ${n} -ge 45000 ]]
}

#: CPSC 2018 from PhysioNet's own uploads to Kaggle (CC0) unless
#: CPSC_SOURCE=physionet: Challenge 2020's Training_WFDB (the 6,877 CPSC 2018
#: training records) and Training_2 (the 3,453 of CPSC-Extra). PhysioNet's
#: HTTP server is the slow alternative (~20-40 KB/s a connection).
CPSC_SOURCE="${CPSC_SOURCE:-kaggle}"
CPSC_KAGGLE="${CPSC_KAGGLE:-physionet/china-physiological-signal-challenge-in-2018 physionet/china-12lead-ecg-challenge-database}"

kaggle_cli() {
    local kaggle="${KAGGLE_BIN:-${HOME}/kaggleenv/bin/kaggle}"
    if [[ ! -x "${kaggle}" ]]; then
        say "installing the kaggle CLI into ${kaggle%/bin/kaggle}" >&2
        python3 -m venv "${kaggle%/bin/kaggle}" && \
            env -u PYTHONPATH "${kaggle%/kaggle}/pip" install -q -U pip kaggle >&2 \
            || { warn "could not install the kaggle CLI"; return 1; }
    fi
    if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -s "${HOME}/.kaggle/access_token" \
          && ! -f "${HOME}/.kaggle/kaggle.json" && ! -f "${HOME}/.config/kaggle/kaggle.json" ]]; then
        warn "no Kaggle credentials (https://www.kaggle.com/settings/api -> Generate New Token)"
        return 1
    fi
    echo "${kaggle}"
}

get_cpsc2018() {
    local d part a b ds kaggle
    d="$(raw_dir cpsc2018)"
    if [[ "${CPSC_SOURCE}" == "kaggle" ]]; then
        kaggle="$(kaggle_cli)" || return 1
        for ds in ${CPSC_KAGGLE}; do
            say "Kaggle ${ds}"
            env -u PYTHONPATH "${kaggle}" datasets download -d "${ds}" -p "${d}" \
                --unzip || { warn "kaggle download of ${ds} failed"; return 1; }
        done
        a=$(find "${d}" -path '*Training_WFDB*' -name 'A*.hea' | wc -l)
        b=$(find "${d}" -path '*Training_2*' -name 'Q*.hea' | wc -l)
    else
        for part in cpsc_2018 cpsc_2018_extra; do
            say "PhysioNet HTTP ${CHALLENGE}/${part}/ (${JOBS} connections)"
            "${PYTHON}" "${HERE}/fetch_physionet_http.py" \
                --base "${CHALLENGE}/${part}/" --dest "${d}/${part}" \
                --jobs "${JOBS}" || return 1
        done
        a=$(find "${d}/cpsc_2018" -name '*.hea' | wc -l)
        b=$(find "${d}/cpsc_2018_extra" -name '*.hea' | wc -l)
    fi
    echo "    ${a} CPSC 2018 + ${b} CPSC-Extra records (expected 6,877 + 3,453)"
    # Which form the labels take is a property of the release; say it.
    grep -h -m1 '^#Dx\|^# Dx' $(find "${d}" -name 'A0001.hea' | head -1) 2>/dev/null \
        | sed 's/^/    label line of A0001: /'
    [[ ${a} -ge 6877 && ${b} -ge 3453 ]]
}

LOCK_HELD=""
release_lock() { [[ -n "${LOCK_HELD}" ]] && rm -rf "${LOCK_HELD}"; LOCK_HELD=""; }
trap release_lock EXIT
trap 'release_lock; exit 143' TERM INT

run_one() {
    local ds="$1" rc d
    d="$(raw_dir "${ds}")"
    if is_done "${ds}"; then
        say "${ds}: complete ($(cat "${d}/.complete")) -- skipping"; return 0
    fi
    mkdir -p "${d}"
    if ! mkdir "${d}/.downloading" 2>/dev/null; then
        warn "${ds}: already being downloaded -- skipping. If nothing is: rm -r ${d}/.downloading"
        return 3
    fi
    LOCK_HELD="${d}/.downloading"
    say "${ds} -> ${d}"
    "get_${ds}"; rc=$?
    release_lock
    if [[ ${rc} -eq 0 ]]; then
        date -u +%FT%TZ > "${d}/.complete"; say "${ds}: complete"
    elif [[ ${rc} -eq 137 ]]; then
        warn "${ds}: the downloader was killed (the login node's limit); rerun later"
    else
        warn "${ds}: not complete; rerun to continue"
    fi
    return ${rc}
}

status() {
    printf '%-10s %-10s %8s  %s\n' set state size path
    local ds d state
    for ds in ${SETS}; do
        d="$(raw_dir "${ds}")"
        if is_done "${ds}"; then state=complete
        elif [[ -d "${d}/.downloading" ]]; then state=running
        elif [[ -d "${d}" ]]; then state=partial
        else state=absent; fi
        printf '%-10s %-10s %8s  %s\n' "${ds}" "${state}" \
            "$(du -sh "${d}" 2>/dev/null | cut -f1)" "${d}"
    done
}

case "${1:-}" in
    status) status ;;
    all)
        failed=()
        for ds in ${SETS}; do run_one "${ds}" || failed+=("${ds}"); done
        echo; status
        [[ ${#failed[@]} -eq 0 ]] || { warn "incomplete: ${failed[*]} -- rerun"; exit 1; } ;;
    ptbxl|chapman|cpsc2018) run_one "$1" ;;
    *) echo "usage: bash $0 status|all|ptbxl|chapman|cpsc2018" >&2; exit 1 ;;
esac
