#!/bin/bash
# ============================================================================
# Fetch the sEMG downstream benchmarks: EMG-EPN-612, GRABMyo, NinaPro DB2/DB5,
# the UCI EMG-for-gestures set.
#
#   source $HOME/pw/bin/activate
#   bash scripts/download_emg_downstream.sh status
#   bash scripts/download_emg_downstream.sh all
#   bash scripts/download_emg_downstream.sh epn612 | grabmyo | ninapro_db2 | ninapro_db5 | uci_emg
#
# RUN IT ON A LOGIN NODE (compute nodes have no internet). ~35 GB in all;
# every step is resumable: a finished set carries .complete and is skipped,
# a rerun fetches only what is missing.
#
#   epn612       EMG-EPN-612, Zenodo 4421500: one 5.5 GB zip, kept zipped
#                (EMG/emg_downstream_prep.py reads the user JSON files out of
#                it), fetched in parallel ranged pieces. CC BY 4.0.
#   grabmyo      GRABMyo 1.1.0 from PhysioNet's open S3 bucket: the three
#                sessions' WFDB records (~10 GB). CC BY 4.0.
#   ninapro_db2  NinaPro DB2 from ninapro.hevs.ch: DB2_s1.zip .. DB2_s40.zip
#                (~19 GB), kept zipped; the .mat files are read out of them.
#                Cite Atzori et al., Sci. Data 2014.
#   ninapro_db5  NinaPro DB5 from Zenodo 1000116: s1.zip .. s10.zip (~200 MB),
#                kept zipped. Cite Pizzolato et al., PLoS ONE 2017.
#   uci_emg      UCI 481 "EMG data for gestures": one 18 MB zip, unpacked to
#                36 subject folders of two tab-separated recordings. CC BY 4.0;
#                cite Lobov et al., Sensors 2018.
#
# LAYOUT (what the preprocessing and scripts/slurm/cineca_emg_downstream.sbatch
# expect):
#   $EMG_ROOT/downstream/raw/epn612/EMG-EPN612 Dataset.zip
#   $EMG_ROOT/downstream/raw/grabmyo/Session{1,2,3}/session*_participant*/...
#   $EMG_ROOT/downstream/raw/ninapro_db2/DB2_s{1..40}.zip
#   $EMG_ROOT/downstream/raw/ninapro_db5/s{1..10}.zip
#   $EMG_ROOT/downstream/raw/uci_emg/EMG_data_for_gestures-master/{01..36}/{1,2}_raw_data_*.txt
#   $EMG_ROOT/downstream/c1/<task>/        written by the preprocessing
#
# None of these is in the sEMG pretraining corpus; the pretraining
# preprocessing refuses them by name (physiowave/emg_c1/routes.py).
#
# ENVIRONMENT
#   EMG_ROOT   ($SCRATCH/bio/emg)
#   JOBS       parallel connections (16)
# ============================================================================

set -uo pipefail

EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-16}"
SETS="epn612 grabmyo ninapro_db5 ninapro_db2 uci_emg"
EPN_URL="${EPN_URL:-https://zenodo.org/api/records/4421500/files/EMG-EPN612%20Dataset.zip/content}"
DB2_BASE="${DB2_BASE:-https://ninapro.hevs.ch/files/DB2_Preproc}"
DB5_BASE="${DB5_BASE:-https://zenodo.org/api/records/1000116/files}"
UCI_URL="${UCI_URL:-https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip}"

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
raw_dir() { echo "${EMG_ROOT}/downstream/raw/$1"; }
is_done() { [[ -f "$(raw_dir "$1")/.complete" ]]; }

get_epn612() {
    local d zip; d="$(raw_dir epn612)"; zip="${d}/EMG-EPN612 Dataset.zip"
    mkdir -p "${d}"
    say "Zenodo ${EPN_URL} (5.5 GB, ${JOBS} connections)"
    "${PYTHON}" "${HERE}/fetch_ranged.py" --url "${EPN_URL}" --dest "${zip}" \
        --jobs "${JOBS}" || return 1
    "${PYTHON}" - "${zip}" <<'PY'
import re, sys, zipfile
n = sum(1 for m in zipfile.ZipFile(sys.argv[1]).namelist()
        if re.search(r"(training|testing)JSON/user\d+/user\d+\.json$", m))
print(f"    {n} user files in the zip (expected 612)")
sys.exit(0 if n >= 612 else 1)
PY
}

get_grabmyo() {
    local d; d="$(raw_dir grabmyo)"
    "${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket physionet-open \
        --prefix grabmyo/1.1.0/ --dest "${d}" \
        --include '(Session[123]/.*\.(hea|dat)|MotionSequence\.txt|LICENSE\.txt)$' \
        --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}" -name 'session*_gesture*_trial*.hea' | wc -l)
    echo "    ${n} trials (expected 43 x 3 x 17 x 7 = 15,351)"
    [[ ${n} -ge 15000 ]]
}

get_ninapro_db2() {
    local d list s; d="$(raw_dir ninapro_db2)"; list="${d}/.files.tsv"
    mkdir -p "${d}"
    : > "${list}"
    for s in $(seq 1 40); do
        printf '%s\t%s\n' "${DB2_BASE}/DB2_s${s}.zip" "${d}/DB2_s${s}.zip" >> "${list}"
    done
    say "${DB2_BASE}/DB2_s{1..40}.zip (~19 GB, ${JOBS} connections)"
    "${PYTHON}" "${HERE}/fetch_ranged.py" --list "${list}" --jobs "${JOBS}" || return 1
    "${PYTHON}" - "${d}" <<'PY'
import glob, os, re, sys, zipfile
ok = 0
for z in sorted(glob.glob(os.path.join(sys.argv[1], "DB2_s*.zip"))):
    names = zipfile.ZipFile(z).namelist()
    ok += sum(1 for n in names if re.search(r"S\d+_E[123]_A1\.mat$", n)) == 3
print(f"    {ok} subject zips with all three exercises (expected 40)")
sys.exit(0 if ok == 40 else 1)
PY
}

get_ninapro_db5() {
    local d list s; d="$(raw_dir ninapro_db5)"; list="${d}/.files.tsv"
    mkdir -p "${d}"
    : > "${list}"
    for s in $(seq 1 10); do
        printf '%s\t%s\n' "${DB5_BASE}/s${s}.zip/content" "${d}/s${s}.zip" >> "${list}"
    done
    say "${DB5_BASE}/s{1..10}.zip (~200 MB, ${JOBS} connections)"
    "${PYTHON}" "${HERE}/fetch_ranged.py" --list "${list}" --jobs "${JOBS}" || return 1
    "${PYTHON}" - "${d}" <<'PY'
import glob, os, re, sys, zipfile
ok = 0
for z in sorted(glob.glob(os.path.join(sys.argv[1], "s*.zip"))):
    names = zipfile.ZipFile(z).namelist()
    ok += sum(1 for n in names if re.search(r"S\d+_E[123]_A1\.mat$", n)) == 3
print(f"    {ok} subject zips with all three exercises (expected 10)")
sys.exit(0 if ok == 10 else 1)
PY
}

get_uci_emg() {
    local d zip; d="$(raw_dir uci_emg)"; zip="${d}/emg_data_for_gestures.zip"
    mkdir -p "${d}"
    say "UCI 481 ${UCI_URL} (18 MB)"
    curl -fsSL --retry 3 -o "${zip}" "${UCI_URL}" || return 1
    "${PYTHON}" - "${zip}" "${d}" <<'PY'
import os, re, sys, zipfile
z, dest = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
members = [m for m in z.namelist() if not (m.startswith("/") or ".." in m.split("/"))]
z.extractall(dest, members)
n = sum(1 for m in members if re.search(r"/\d\d/\d_raw_data_.*\.txt$", m))
print(f"    {n} recordings (expected 36 subjects x 2 series = 72)")
sys.exit(0 if n >= 72 else 1)
PY
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
    printf '%-12s %-10s %8s  %s\n' set state size path
    local ds d state
    for ds in ${SETS}; do
        d="$(raw_dir "${ds}")"
        if is_done "${ds}"; then state=complete
        elif [[ -d "${d}/.downloading" ]]; then state=running
        elif [[ -d "${d}" ]]; then state=partial
        else state=absent; fi
        printf '%-12s %-10s %8s  %s\n' "${ds}" "${state}" \
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
    epn612|grabmyo|ninapro_db2|ninapro_db5|uci_emg) run_one "$1" ;;
    *) echo "usage: bash $0 status|all|epn612|grabmyo|ninapro_db5|ninapro_db2|uci_emg" >&2; exit 1 ;;
esac
