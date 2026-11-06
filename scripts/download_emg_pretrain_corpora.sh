#!/bin/bash
# ============================================================================
# Fetch the sEMG C1 pretraining corpora into one layout under $EMG_ROOT.
#
#   bash scripts/download_emg_pretrain_corpora.sh status      # what is there
#   bash scripts/download_emg_pretrain_corpora.sh all         # all five
#   bash scripts/download_emg_pretrain_corpora.sh hyser       # just one
#
#   nohup bash scripts/download_emg_pretrain_corpora.sh all > ~/emg_download.log 2>&1 &
#
# `all` fetches the five corpora, smallest first. Every step is resumable and
# idempotent: a finished corpus carries a .complete marker and is skipped, a
# partial download continues from where it stopped, and rerunning `all` after
# an interruption is the way to continue it.
#
# TWO STAGES, because compute nodes have no route to the internet and login
# nodes kill processes that use a lot of CPU (the first full run here had its
# downloader SIGKILLed three times):
#
#   STAGE=fetch   on a LOGIN NODE: bytes only. No md5, no unpacking, nothing
#                 marked complete except putEMG and Hyser, which need neither.
#   STAGE=finish  on a COMPUTE NODE, no network: checks every file is whole,
#                 md5s CEMHSEY's zips, unpacks GESTURE and emg2qwerty, reads
#                 emg2pose's tar headers, and marks each corpus complete --
#                 scripts/slurm/cineca_emg_download_finish.sbatch.
#   (unset)       both, for a machine that is allowed to do both.
#
#   source $HOME/pw/bin/activate      # Python >= 3.7 for the helpers
#   STAGE=fetch nohup bash scripts/download_emg_pretrain_corpora.sh all > ~/emg_dl.log 2>&1 &
#   sbatch scripts/slurm/cineca_emg_download_finish.sbatch      # once fetched
#
# A login node killing the downloader is a limit being enforced, and nothing
# here loops to get around it: the run stops and says so. Rerunning it later
# is your call; if kills keep coming, CINECA's data-transfer service is the
# route meant for this volume. The big single files are fetched by
# scripts/fetch_ranged.py in 256 MB pieces written straight into place, so a
# rerun fetches only the pieces that are missing.
#
# LAYOUT -- what EMG/preprocess_emg_corpus.sh expects:
#
#   $EMG_ROOT/emg2pose/raw/emg2pose_dataset.tar   read in place, not unpacked
#   $EMG_ROOT/emg2qwerty/raw/emg2qwerty-data-2021-08/*.hdf5   unpacked
#   $EMG_ROOT/Hyser/raw/<subset>_dataset/subjectNN_sessionS/*_raw_*.{hea,dat}
#   $EMG_ROOT/CEMHSEY/raw/GRASP_S{1..13}.zip      read in place (deflate)
#   $EMG_ROOT/CEMHSEY/raw/GESTURE_S{1..6}/...     unpacked (deflate64)
#   $EMG_ROOT/putEMG/raw/Data-HDF5/*.hdf5
#   $EMG_ROOT/emg_c1_corpus/<dataset>/            written by preprocessing
#
# SIZES (from the servers, 2026-10-06), and what stays on disk:
#   putEMG       30.9 GB, 712 HDF5 files over WebDAV       (optional corpus)
#   Hyser        76 GB: the *_raw_* records of hd-semg 2.0.0 from open S3
#                (the *_preprocess_* copies and force channels are skipped)
#   CEMHSEY      332 GB in 19 Zenodo zips (~0.6 MB/s a connection, so many
#                at once); GESTURE's 83 GB are unpacked and their zips removed
#   emg2qwerty   308 GB tar.gz -> ~340 GB of HDF5; the tar.gz is removed
#                after it unpacks (KEEP_ZIPS=1 keeps it), so ~650 GB at peak
#   emg2pose     463 GB tar, kept as it is
#   about 1.25 TB in all once done; `all` checks free space before it starts.
#
# UNPACKING emg2qwerty takes an hour or more of one core. If the login node
# kills it, run that one step where nothing will: the download is already
# there, so no internet is needed --
#   sbatch -A <account> -p lrd_all_serial --time 4:00:00 --mem 8G \
#          --wrap "bash scripts/download_emg_pretrain_corpora.sh emg2qwerty"
#
# NOT HERE, ON PURPOSE: NinaPro DB5 and EPN-612 are the downstream benchmarks
# and are never pretrained on; NinaPro DB6/7/8 are not used.
#
# LICENCES, as each source states them (2026-10-06): emg2pose CC BY-NC-SA 4.0;
# emg2qwerty CC BY-NC-SA 4.0 by its LICENSE file (its README says CC BY-NC
# 4.0); CEMHSEY CC BY 4.0 on Zenodo; Hyser ODC-By 1.0; putEMG CC BY-NC 4.0.
# Non-commercial terms carry over to the checkpoints trained on them.
#
# ENVIRONMENT:
#   EMG_ROOT     download root ($SCRATCH/bio/emg)
#   JOBS         connections at once for one big file (16)
#   CEMHSEY_JOBS connections at once over CEMHSEY's zips (24)
#   STAGE        fetch | finish | (unset: both) -- see above
#   KEEP_ZIPS    1: keep emg2qwerty's tar.gz and CEMHSEY's GESTURE zips
#   SKIP_SPACE_CHECK  1: do not refuse to start when free space looks short
# ============================================================================

set -uo pipefail

EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-16}"
CEMHSEY_JOBS="${CEMHSEY_JOBS:-24}"
STAGE="${STAGE:-}"
case "${STAGE}" in ""|fetch|finish) ;; *) echo "ERROR: STAGE=fetch|finish" >&2; exit 1 ;; esac
#: get_* return this when the bytes are here and the rest is STAGE=finish's.
FETCHED=10

OPEN_DATASETS="putemg hyser cemhsey emg2qwerty emg2pose"

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
EMG2POSE_URL="${EMG2POSE_URL:-https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar}"
EMG2POSE_BYTES="${EMG2POSE_BYTES:-462824048640}"
EMG2QWERTY_URL="${EMG2QWERTY_URL:-https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz}"
EMG2QWERTY_BYTES="${EMG2QWERTY_BYTES:-308382645571}"
# Two Zenodo records: GRASP S1-S10, then GRASP S11-S13, GESTURE S1-S6 and the
# authors' MATLAB functions. File names, sizes and md5s come from the API.
CEMHSEY_RECORDS="${CEMHSEY_RECORDS:-15077957 15070187}"
# putEMG's public share. The project's own downloader builds URLs that no
# longer resolve; the share's WebDAV endpoint serves every file, with ranges.
PUTEMG_SHARE="${PUTEMG_SHARE:-45NY5snj0U4tgQz}"
PUTEMG_DAV="${PUTEMG_DAV:-https://chmura.put.poznan.pl/public.php/webdav}"

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
    echo "targets: status all layout ${OPEN_DATASETS}"
}

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

raw_dir() {
    case "$1" in
        emg2pose)   echo "${EMG_ROOT}/emg2pose/raw" ;;
        emg2qwerty) echo "${EMG_ROOT}/emg2qwerty/raw" ;;
        hyser)      echo "${EMG_ROOT}/Hyser/raw" ;;
        cemhsey)    echo "${EMG_ROOT}/CEMHSEY/raw" ;;
        putemg)     echo "${EMG_ROOT}/putEMG/raw" ;;
        *) die "unknown dataset $1" ;;
    esac
}

#: GB each corpus needs at its peak while this script works on it.
need_gb() {
    case "$1" in
        putemg)     echo 32 ;;
        hyser)      echo 78 ;;
        cemhsey)    echo 350 ;;
        emg2qwerty) echo 660 ;;
        emg2pose)   echo 465 ;;
        *)          echo 0 ;;
    esac
}

is_done()   { [[ -f "$(raw_dir "$1")/.complete" ]]; }
mark_done() { date -u +%FT%TZ > "$(raw_dir "$1")/.complete"; }

free_gb() {
    local d="$1"
    while [[ ! -d "${d}" ]]; do d="$(dirname "${d}")"; done
    df -Pk "${d}" | awk 'NR==2 {printf "%d", $4 / 1048576}'
}

check_space() {
    local want="$1" have
    have="$(free_gb "${EMG_ROOT}")"
    if [[ "${have}" -lt "${want}" ]]; then
        if [[ "${SKIP_SPACE_CHECK:-0}" == "1" ]]; then
            warn "need ~${want} GB under ${EMG_ROOT}, ${have} GB free; continuing (SKIP_SPACE_CHECK=1)"
        else
            die "need ~${want} GB under ${EMG_ROOT} and only ${have} GB is free. Free some, or SKIP_SPACE_CHECK=1 if a quota rather than df is what limits you."
        fi
    fi
}

local_size() {
    [[ -f "$1" ]] || { echo 0; return; }
    if stat -c%s "$1" >/dev/null 2>&1; then stat -c%s "$1"; else stat -f%z "$1"; fi
}

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"; fi
}

ranged() {
    local rc
    if [[ "${STAGE}" == "finish" ]]; then
        # No network here: whole files pass, anything else is for STAGE=fetch.
        "${PYTHON}" "${HERE}/fetch_ranged.py" --check-only "$@"
        rc=$?
        [[ ${rc} -eq 0 ]] || warn "not all downloaded; finish it on a login node: STAGE=fetch bash $0 all"
        return ${rc}
    fi
    "${PYTHON}" "${HERE}/fetch_ranged.py" "$@"
    rc=$?
    if [[ ${rc} -eq 137 ]]; then
        warn "the downloader was killed (SIGKILL) -- on a login node that is the node's"
        warn "  process limit. The pieces already written are kept; see the header."
    fi
    return ${rc}
}

#: In STAGE=fetch, stop here: the bytes are in, the rest needs a compute node.
fetched_only() {
    if [[ "${STAGE}" == "fetch" ]]; then
        echo "    downloaded; $1 is left to: sbatch scripts/slurm/cineca_emg_download_finish.sbatch"
        return 0
    fi
    return 1
}

#: putEMG and Hyser need nothing after the download, and in STAGE=finish
#: nothing can be downloaded.
needs_network() {
    if [[ "${STAGE}" == "finish" ]]; then
        warn "$1 is not downloaded yet, and this stage has no network: STAGE=fetch on a login node"
        return 0
    fi
    return 1
}

# --------------------------------------------------------------------------- #
# One function per corpus. Each returns 0 only when the corpus is complete.
# --------------------------------------------------------------------------- #

get_emg2pose() {
    local d; d="$(raw_dir emg2pose)"
    local tar="${d}/emg2pose_dataset.tar"
    mkdir -p "${d}"
    say "${EMG2POSE_URL} (463 GB, ${JOBS} connections)"
    ranged --url "${EMG2POSE_URL}" --dest "${tar}" --size "${EMG2POSE_BYTES}" \
           --jobs "${JOBS}" || return 1
    fetched_only "reading the tar's headers" && return ${FETCHED}
    # The size is necessary, not sufficient: the tar's headers must read back.
    # An uncompressed tar is walked header to header by seeking, so this is
    # ~25 k small reads, not a pass over 463 GB.
    "${PYTHON}" - "${tar}" <<'PYEOF' || return 1
import sys, tarfile
n = 0
with tarfile.open(sys.argv[1], "r:") as tf:
    for m in tf:
        n += m.isfile() and m.name.endswith(".hdf5")
print(f"    {n:,} HDF5 file(s) in the tar (expected 25,253)")
sys.exit(0 if n >= 25000 else 1)
PYEOF
    echo "    kept as a tar: EMG/preprocess_emg_corpus.py reads each file out of it"
}

get_emg2qwerty() {
    local d; d="$(raw_dir emg2qwerty)"
    local gz="${d}/emg2qwerty-data-2021-08.tar.gz"
    local tree="${d}/emg2qwerty-data-2021-08"
    local n
    mkdir -p "${d}"
    n=$(find "${tree}" -name '*.hdf5' 2>/dev/null | wc -l)
    if [[ -f "${d}/.unpacked" && ${n} -ge 1100 ]]; then
        echo "    ${n} session file(s) already unpacked"
        return 0
    fi
    say "${EMG2QWERTY_URL} (308 GB, ${JOBS} connections)"
    ranged --url "${EMG2QWERTY_URL}" --dest "${gz}" --size "${EMG2QWERTY_BYTES}" \
           --jobs "${JOBS}" || return 1
    fetched_only "unpacking (an hour or more of CPU)" && return ${FETCHED}
    say "unpacking ${gz##*/} (an hour or more)"
    if command -v pigz >/dev/null 2>&1; then
        tar -I pigz -xf "${gz}" -C "${d}" || return 1
    else
        tar -xzf "${gz}" -C "${d}" || return 1
    fi
    n=$(find "${tree}" -name '*.hdf5' | wc -l)
    echo "    ${n} session file(s) (expected ~1,135)"
    [[ ${n} -ge 1100 ]] || return 1
    date -u +%FT%TZ > "${d}/.unpacked"
    [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "${gz}"
}

get_hyser() {
    local d; d="$(raw_dir hyser)"
    needs_network hyser && return 1
    say "S3 s3://physionet-open/hd-semg/2.0.0/, the *_raw_* records"
    "${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket physionet-open \
        --prefix hd-semg/2.0.0/ --dest "${d}" \
        --include '(_raw_.*\.(hea|dat)|readme\.txt|LICENSE\.txt)$' \
        --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}" -name '*_raw_*.hea' | wc -l)
    echo "    ${n} raw record(s) (expected 13,154)"
    [[ ${n} -ge 13000 ]]
}

get_cemhsey() {
    local d; d="$(raw_dir cemhsey)"
    local list="${d}/.files.tsv" want="${d}/.md5.tsv"
    mkdir -p "${d}"
    say "Zenodo records ${CEMHSEY_RECORDS} (19 zips, ${CEMHSEY_JOBS} connections)"
    # name, bytes, md5, url -- from Zenodo's API, so a re-upload with new
    # sizes or checksums is noticed rather than half-matched. Written to a
    # temporary file and moved into place, so a failed request (Zenodo answers
    # 503 at times) leaves the last good listing, which is used instead -- and
    # is all STAGE=finish, without a network, has.
    if [[ "${STAGE}" != "finish" ]]; then
        if "${PYTHON}" - "${d}" ${CEMHSEY_RECORDS} > "${want}.tmp" <<'PYEOF'; then
import json, sys, time, urllib.error, urllib.request
d = sys.argv[1]
for rec in sys.argv[2:]:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}",
                                        timeout=60) as r:
                meta = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 5:
                raise
            print(f"    Zenodo record {rec}: {exc}; retrying", file=sys.stderr)
            time.sleep(30 * (attempt + 1))
    for f in meta["files"]:
        print(f"{f['key']}\t{f['size']}\t{f['checksum'].split(':', 1)[1]}\t"
              f"{f['links']['self']}")
PYEOF
            mv "${want}.tmp" "${want}"
        else
            rm -f "${want}.tmp"
            [[ -s "${want}" ]] || { warn "Zenodo's API did not answer and there is no earlier listing"; return 1; }
            warn "Zenodo's API did not answer; using the listing from the last run"
        fi
    fi
    [[ -s "${want}" ]] || { warn "no CEMHSEY listing (${want}); run STAGE=fetch on a login node first"; return 1; }
    local name bytes md5 url
    : > "${list}"
    while IFS=$'\t' read -r name bytes md5 url; do
        # An unpacked GESTURE zip has served its purpose and is gone.
        [[ -f "${d}/${name%.zip}/.unpacked" ]] && continue
        printf '%s\t%s\t%s\n' "${url}" "${d}/${name}" "${bytes}" >> "${list}"
    done < "${want}"
    if [[ -s "${list}" ]]; then
        ranged --list "${list}" --jobs "${CEMHSEY_JOBS}" || return 1
    fi
    fetched_only "md5 and unpacking GESTURE" && return ${FETCHED}
    local bad=0
    while IFS=$'\t' read -r name bytes md5 url; do
        local zip="${d}/${name}"
        [[ -f "${d}/${name%.zip}/.unpacked" ]] && continue
        if [[ ! -f "${zip}.md5ok" ]]; then
            echo "    md5 ${name} ..."
            if [[ "$(md5_of "${zip}")" == "${md5}" ]]; then
                echo "${md5}" > "${zip}.md5ok"
            else
                warn "${name}: md5 differs from Zenodo's; removing it to fetch again"
                rm -f "${zip}" "${zip}.chunks"
                bad=1
                continue
            fi
        fi
        # GESTURE is deflate64, which Python's zipfile cannot inflate: unpack
        # it into a directory named after the zip, which preprocessing reads
        # in its place. GRASP is plain deflate and stays zipped.
        if [[ "${name}" == GESTURE_*.zip ]]; then
            unpack_deflate64 "${zip}" "${d}/${name%.zip}" || return 1
        fi
    done < "${want}"
    [[ ${bad} -eq 0 ]] || { warn "rerun to fetch the zip(s) that failed md5"; return 1; }
    local n; n=$(find "${d}" -name '*.mat' -path '*GESTURE_*' | wc -l)
    echo "    13 GRASP zips read in place; ${n} GESTURE trial file(s) unpacked"
}

unpack_deflate64() {   # zip dir
    local zip="$1" dir="$2" n_zip n_dir
    [[ -f "${dir}/.unpacked" ]] && return 0
    say "unpacking ${zip##*/} (deflate64)"
    mkdir -p "${dir}"
    # inflate64 first: it checks every member's CRC, and it does not depend on
    # how the node's unzip was built (a Leonardo compute node's could not do
    # deflate64, and it had no 7z). pip install inflate64 in the venv.
    if "${PYTHON}" -c "import inflate64" 2>/dev/null; then
        "${PYTHON}" "${HERE}/unzip_deflate64.py" "${zip}" --dest "${dir}" >/dev/null || return 1
    elif unzip -qo "${zip}" -d "${dir}"; then
        :
    elif command -v 7z >/dev/null 2>&1 && 7z x -y -o"${dir}" "${zip}" >/dev/null; then
        :
    else
        warn "${zip##*/}: could not unpack deflate64. In the venv:"
        warn "  pip install inflate64     (then resubmit this job)"
        return 1
    fi
    n_zip=$("${PYTHON}" -c "import sys, zipfile; print(sum(n.endswith('.mat') for n in zipfile.ZipFile(sys.argv[1]).namelist()))" "${zip}")
    n_dir=$(find "${dir}" -name '*.mat' | wc -l)
    if [[ ${n_zip} -eq 0 || ${n_dir} -ne ${n_zip} ]]; then
        warn "${zip##*/}: ${n_dir} .mat unpacked, the zip lists ${n_zip}"
        return 1
    fi
    date -u +%FT%TZ > "${dir}/.unpacked"
    [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "${zip}" "${zip}.md5ok"
}

get_putemg() {
    local d; d="$(raw_dir putemg)"
    needs_network putemg && return 1
    local list="${d}/.files.tsv"
    mkdir -p "${d}/Data-HDF5"
    say "${PUTEMG_DAV}/Data-HDF5/ (712 files)"
    # One PROPFIND names every file and its size.
    "${PYTHON}" - "${PUTEMG_DAV}" "${PUTEMG_SHARE}" "${d}" > "${list}" <<'PYEOF' || return 1
import base64, sys, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
dav, share, d = sys.argv[1:4]
req = urllib.request.Request(
    f"{dav}/Data-HDF5/", method="PROPFIND",
    headers={"Depth": "1",
             "Authorization": "Basic " + base64.b64encode(f"{share}:".encode()).decode()})
with urllib.request.urlopen(req, timeout=120) as r:
    root = ET.fromstring(r.read())
ns = {"d": "DAV:"}
host = urllib.parse.urlsplit(dav)
n = 0
for resp in root.findall("d:response", ns):
    href = urllib.parse.unquote(resp.findtext("d:href", namespaces=ns))
    size = resp.findtext(".//d:getcontentlength", namespaces=ns)
    name = href.rstrip("/").rsplit("/", 1)[-1]
    if not name.endswith(".hdf5") or not size:
        continue
    url = f"{host.scheme}://{share}:@{host.netloc}{host.path}/Data-HDF5/{urllib.parse.quote(name)}"
    print(f"{url}\t{d}/Data-HDF5/{name}\t{size}")
    n += 1
sys.exit(0 if n else 1)
PYEOF
    echo "    $(wc -l < "${list}") file(s) listed"
    ranged --list "${list}" --jobs "${JOBS}" || return 1
    local n; n=$(find "${d}/Data-HDF5" -name 'emg_*.hdf5' | wc -l)
    echo "    ${n} record(s) (expected 712)"
    [[ ${n} -ge 700 ]]
}

# ONE DOWNLOADER PER CORPUS, across login nodes: two processes writing the
# same file interleave their bytes into a file of the right size and the wrong
# content. Each corpus is claimed with an atomic mkdir on the shared
# filesystem; a holder killed with SIGKILL leaves the claim behind, and the
# message says how to clear it.
LOCK_HELD=""
release_lock() {
    [[ -n "${LOCK_HELD}" ]] && rm -rf "${LOCK_HELD}"
    LOCK_HELD=""
}
trap release_lock EXIT
trap 'release_lock; exit 143' TERM INT

claim() {
    local lock; lock="$(raw_dir "$1")/.downloading"
    if ! mkdir "${lock}" 2>/dev/null; then
        warn "$1: already being downloaded by $(cat "${lock}/owner" 2>/dev/null || echo 'another process') -- skipping."
        warn "  If that process is gone (killed with -9, node rebooted): rm -r ${lock}"
        return 1
    fi
    echo "$(hostname) pid $$, since $(date '+%F %T')" > "${lock}/owner"
    LOCK_HELD="${lock}"
}

run_one() {
    local ds="$1" rc
    if is_done "${ds}"; then
        say "${ds}: complete ($(cat "$(raw_dir "${ds}")/.complete")) -- skipping"
        return 0
    fi
    mkdir -p "$(raw_dir "${ds}")"
    claim "${ds}" || return 3
    say "${ds} -> $(raw_dir "${ds}")"
    "get_${ds}"
    rc=$?
    release_lock
    if [[ ${rc} -eq 0 ]]; then
        mark_done "${ds}"
        say "${ds}: complete"
    elif [[ ${rc} -eq ${FETCHED} ]]; then
        say "${ds}: downloaded, not yet verified (STAGE=finish does that)"
        rc=0
    else
        warn "${ds}: not complete; rerun to continue"
    fi
    return ${rc}
}

status() {
    printf '%-12s %-10s %10s  %s\n' dataset state on-disk path
    local ds d state size
    for ds in ${OPEN_DATASETS}; do
        d="$(raw_dir "${ds}")"
        if is_done "${ds}"; then state=complete
        elif [[ -d "${d}/.downloading" ]]; then state=running
        elif [[ -d "${d}" ]] && [[ -n "$(ls -A "${d}" 2>/dev/null)" ]]; then state=partial
        else state=absent; fi
        size="$(du -sh "${d}" 2>/dev/null | cut -f1)"
        printf '%-12s %-10s %10s  %s\n' "${ds}" "${state}" "${size:--}" "${d}"
    done
    echo "free under ${EMG_ROOT}: $(free_gb "${EMG_ROOT}") GB"
}

main() {
    local target="${1:-}"
    case "${target}" in
        ""|-h|--help|help) usage; return 0 ;;
        status) status ;;
        layout)
            local ds
            for ds in ${OPEN_DATASETS}; do mkdir -p "$(raw_dir "${ds}")"; done
            status ;;
        all)
            local ds want=0 failed=()
            for ds in ${OPEN_DATASETS}; do
                is_done "${ds}" || want=$(( want + $(need_gb "${ds}") ))
            done
            mkdir -p "${EMG_ROOT}"
            check_space "${want}"
            say "fetching into ${EMG_ROOT}: ${OPEN_DATASETS} (~${want} GB at most)"
            for ds in ${OPEN_DATASETS}; do
                run_one "${ds}" || failed+=("${ds}")
            done
            echo ""
            status
            echo ""
            if [[ ${#failed[@]} -gt 0 ]]; then
                warn "incomplete: ${failed[*]} -- rerun \`bash $0 all\` to continue"
                return 1
            fi ;;
        emg2pose|emg2qwerty|hyser|cemhsey|putemg)
            mkdir -p "${EMG_ROOT}"
            is_done "${target}" || check_space "$(need_gb "${target}")"
            run_one "${target}" ;;
        *) usage; die "unknown target '${target}'" ;;
    esac
}

# Sourced (by a test), it only defines the functions above.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
