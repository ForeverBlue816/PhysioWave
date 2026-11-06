#!/bin/bash
# ============================================================================
# Fetch the multimodal benchmarks: MPDB (driving behaviour) and SeizeIT2.
#
#   source $HOME/pw/bin/activate
#   bash scripts/download_fusion_data.sh status
#   bash scripts/download_fusion_data.sh mpdb_raw            # 23 GB, the recommended MPDB source
#   bash scripts/download_fusion_data.sh mpdb                # 3 GB, the authors' epochs
#   bash scripts/download_fusion_data.sh seizeit2            # 97.5 GB
#   SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' bash scripts/download_fusion_data.sh seizeit2   # a subset first
#
# RUN IT ON A LOGIN NODE (compute nodes have no internet). Every step is
# resumable: a finished set carries .complete and is skipped; a rerun fetches
# only what is missing.
#
#   mpdb_raw  The RAW release, figshare 22193119 (CC BY 4.0, 22.9 GB in two
#             .rar): per subject four continuous EEGLAB .set files per device
#             (EEG&ECG, EMG, GSR) with the simulator markers and their times.
#             What Fusion/mpdb_raw_prep.py reads -- the recommended source.
#   mpdb      The authors' PREPROCESSED release, figshare 22192831 (CC BY 4.0,
#             3.15 GB in two .rar archives): per subject EEG_n.set, ECG_n.set,
#             EMG_n.set, GSR_n.set -- EEGLAB epochs of the ten simulator
#             events. Extracting needs unrar, 7z or bsdtar; if the login node
#             has none, the script says how to put rarlab's unrar in $HOME/bin.
#   seizeit2  OpenNeuro ds005873 (CC0), BIDS: per subject and run the EEG, ECG
#             and EMG EDFs and the events.tsv, from the open S3 bucket. The
#             movement EDFs are not fetched (not used). 125 subjects, 97.5 GB.
#
# LAYOUT (what Fusion/*_prep.py and scripts/slurm/cineca_fusion_prep.sbatch expect):
#   $FUSION_ROOT/raw/mpdb_raw/**/{n}_{brake,throttle,turn,change}.set
#   $FUSION_ROOT/raw/mpdb/**/{EEG,ECG,EMG}_{n}.set
#   $FUSION_ROOT/raw/seizeit2/sub-XXX/ses-01/{eeg,ecg,emg}/...
#   $FUSION_ROOT/c1/{mpdb,seizeit2}/{train,val,test}.h5     written by the preprocessing
#
# ENVIRONMENT
#   FUSION_ROOT  ($SCRATCH/bio/fusion)
#   JOBS         parallel connections (16)
#   SUBJECTS     seizeit2: a regex over sub-XXX; default every subject
# ============================================================================

set -uo pipefail

FUSION_ROOT="${FUSION_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/fusion}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    # a bare login shell: the project environment's python (or any python3)
    # shellcheck disable=SC1091
    PYTHON="$(set +u; source "${HERE}/cineca_env.sh" >/dev/null 2>&1; command -v python python3 | head -1)"
    [[ -n "${PYTHON}" ]] || { echo "ERROR: no python found; source scripts/cineca_env.sh first" >&2; exit 1; }
    echo "    python: ${PYTHON}"
fi
JOBS="${JOBS:-16}"
SETS="mpdb_raw mpdb seizeit2"
MPDB_FILES="https://ndownloader.figshare.com/files/39443269|preprocessed_data.rar
https://ndownloader.figshare.com/files/42478284|preprocessed_additional.rar"
MPDB_RAW_FILES="https://ndownloader.figshare.com/files/39444118|Rawdata.rar
https://ndownloader.figshare.com/files/42391515|Rawdata_additional.rar"

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
raw_dir() { echo "${FUSION_ROOT}/raw/$1"; }
is_done() { [[ -f "$(raw_dir "$1")/.complete" ]]; }

unrar_cmd() {
    if command -v unrar >/dev/null 2>&1; then echo "unrar x -o+ -idq"
    elif command -v 7z >/dev/null 2>&1; then echo "7z x -y -bso0"
    elif command -v 7za >/dev/null 2>&1; then echo "7za x -y -bso0"
    elif command -v bsdtar >/dev/null 2>&1; then echo "bsdtar -xf"
    else return 1; fi
}

get_mpdb() {
    local d list; d="$(raw_dir mpdb)"; list="${d}/.files.tsv"
    mkdir -p "${d}"
    : > "${list}"
    while IFS='|' read -r url name; do
        [[ -n "${url}" ]] && printf '%s\t%s\n' "${url}" "${d}/${name}" >> "${list}"
    done <<< "${MPDB_FILES}"
    say "figshare 22192831: two archives, 3.15 GB, ${JOBS} connections"
    "${PYTHON}" "${HERE}/fetch_ranged.py" --list "${list}" --jobs "${JOBS}" || return 1
    local un; un="$(unrar_cmd)" || {
        warn "no unrar / 7z / bsdtar on this node. Without root:"
        warn "  cd \$HOME && curl -LO https://www.rarlab.com/rar/rarlinux-x64-712.tar.gz && tar xzf rarlinux-x64-712.tar.gz"
        warn "  mkdir -p \$HOME/bin && cp rar/unrar \$HOME/bin/ && export PATH=\$HOME/bin:\$PATH"
        warn "then rerun this script; the archives are already here."
        return 1
    }
    local a
    for a in "${d}"/*.rar; do
        say "extracting $(basename "${a}") with ${un%% *}"
        (cd "${d}" && ${un} "${a}") || return 1
    done
    local n; n=$(find "${d}" -iname 'EEG_*.set' | wc -l)
    echo "    ${n} EEG_n.set files (expected 35), $(find "${d}" -iname 'EMG_*.set' | wc -l) EMG, $(find "${d}" -iname 'ECG_*.set' | wc -l) ECG"
    [[ ${n} -ge 30 ]]
}

get_mpdb_raw() {
    local d list; d="$(raw_dir mpdb_raw)"; list="${d}/.files.tsv"
    mkdir -p "${d}"
    : > "${list}"
    while IFS='|' read -r url name; do
        [[ -n "${url}" ]] && printf '%s\t%s\n' "${url}" "${d}/${name}" >> "${list}"
    done <<< "${MPDB_RAW_FILES}"
    say "figshare 22193119: two archives, 22.9 GB, ${JOBS} connections"
    "${PYTHON}" "${HERE}/fetch_ranged.py" --list "${list}" --jobs "${JOBS}" || return 1
    local un; un="$(unrar_cmd)" || { warn "no unrar / 7z / bsdtar (see mpdb above)"; return 1; }
    local a
    for a in "${d}"/*.rar; do
        say "extracting $(basename "${a}") with ${un%% *}"
        (cd "${d}" && ${un} "${a}") || return 1
    done
    local n; n=$(find "${d}" -iname '*_brake.set' | wc -l)
    echo "    ${n} subjects with a brake experiment (expected 35); .set files: $(find "${d}" -name '*.set' | wc -l)"
    [[ ${n} -ge 30 ]]
}

get_seizeit2() {
    local d inc; d="$(raw_dir seizeit2)"
    mkdir -p "${d}"
    # matched against the full key: ds005873/sub-001/ses-01/eeg/..._eeg.edf
    inc='(/(eeg|ecg|emg)/[^/]+\.(edf|tsv|json)$|^ds005873/[^/]+\.(json|tsv|txt)$|^ds005873/(README|CHANGES)$)'
    if [[ -n "${SUBJECTS:-}" ]]; then
        inc="(/(${SUBJECTS})/.*/(eeg|ecg|emg)/[^/]+\.(edf|tsv|json)$|^ds005873/[^/]+\.(json|tsv|txt)$)"
        say "subjects matching ${SUBJECTS}"
    fi
    say "s3://openneuro.org/ds005873/ -> ${d} (${JOBS} connections)"
    "${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket openneuro.org --prefix ds005873/ \
        --dest "${d}" --include "${inc}" --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}" -name '*_eeg.edf' | wc -l)
    echo "    ${n} EEG runs over $(find "${d}" -maxdepth 1 -name 'sub-*' | wc -l) subjects"
    if [[ -n "${SUBJECTS:-}" ]]; then
        echo "    (a subset: .complete is not written; rerun without SUBJECTS for the whole set)"
        return 2
    fi
    [[ ${n} -ge 1000 ]]
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
    elif [[ ${rc} -eq 2 ]]; then
        say "${ds}: subset fetched"
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
    all) for ds in ${SETS}; do run_one "${ds}"; done; echo; status ;;
    mpdb_raw|mpdb|seizeit2) run_one "$1" ;;
    *) echo "usage: bash $0 status|all|mpdb_raw|mpdb|seizeit2" >&2; exit 1 ;;
esac
