#!/bin/bash
# ============================================================================
# Fetch the six remaining C1 pretraining corpora into one layout.
#
#   bash scripts/download_eeg_pretrain_corpora.sh layout
#   bash scripts/download_eeg_pretrain_corpora.sh physionet_mi
#   bash scripts/download_eeg_pretrain_corpora.sh hbn R3
#   bash scripts/download_eeg_pretrain_corpora.sh hbn all
#
# ONE DATASET PER INVOCATION, NAMED EXPLICITLY. There is no "download
# everything": these total roughly 2 TB, most of it HBN, and a command that
# starts two terabytes of transfer because it was run without arguments is a
# command that gets run without arguments.
#
# RUN THIS ON A LOGIN NODE. The compute nodes here have no route to the public
# internet -- an aws or curl transfer from one fails with "Could not connect to
# the endpoint URL". For HBN, whose releases are large enough that a login node
# will kill the transfer partway, use scripts/download_hbn_login.sh, which
# reruns after being killed and resumes.
#
# NOTHING HERE NEEDS npm. FACED, M3CV and HGD come straight from
# data.nemar.org over HTTPS, resumably; nemar-cli is a node package that adds
# nothing, and a login node without npm is the normal case.
#
# TDBRAIN IS NOT DOWNLOADED HERE and cannot be. It needs an ORCID login and an
# accepted data use agreement, and accepting an agreement is not something a
# script does on your behalf. `tdbrain` prints what to do instead.
#
# HBN is on open S3, but the "Not for Commercial Use" releases are excluded
# from the main model -- R1-R11 are the standard ones and are what `all` fetches.
#
# SIZES, so nothing starts a transfer that will not fit:
#   PhysioNetMI  1.9 GB zip / 3.4 GB unpacked
#   FACED        22.7 GB          M3CV  ~30 GB          HGD  25.1 GB
#   TDBRAIN      reserve 130 GB unpacked
#   HBN          ~1.49 TB for R1-R9 (103,120,140,230,224,91,245,157,185).
#                Only nine releases are mirrored to fcp-indi; R10 and R11 in
#                the published table are not there.
#
# AFTER DOWNLOADING, inspect before processing. Every adapter has been checked
# against synthetic trees with these corpora's conventions, not against your
# copy:
#   DATASET=<id> INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh
# ============================================================================

set -uo pipefail

EEG_ROOT="${EEG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/eeg}"

usage() {
    sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo "datasets: layout physionet_mi faced m3cv hgd hbn tdbrain"
    echo "checks:   verify-hbn [releases]   probe-hbn [releases]"
}

need() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "ERROR: $1 is not on PATH. $2" >&2
    return 1
}

# `aws` on PATH is not the same as a working AWS CLI. PyPI carries an abandoned
# Python-2 package literally named `aws`, and `pip install aws` puts its
# entry point at the same bin/aws an awscli install would use -- so it is found,
# it is executed, and it dies on a print statement from 2012. Run it before
# trusting it.
# GNU stat takes -c%s and BSD stat takes -f%z. Picking once, up front, rather
# than per file: a size check that silently returns 0 on the wrong platform
# would report every release as INCOMPLETE 0.0%.
_STAT_FLAG=""
_pick_stat() {
    [[ -n "${_STAT_FLAG}" ]] && return 0
    if stat -c%s "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
        _STAT_FLAG="-c%s"
    elif stat -f%z "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
        _STAT_FLAG="-f%z"
    else
        echo "ERROR: cannot determine file sizes with stat on this system." >&2
        return 1
    fi
}

tree_bytes() {
    _pick_stat || return 1
    find -L "$1" -type f -print0 2>/dev/null \
        | xargs -0 stat "${_STAT_FLAG}" 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

#: What `all` expands to. Listing the parent prefix on 2026-08-27 returned
#: cmi_bids_R1 .. cmi_bids_R9 and cmi_bids_NC, with no R10 or R11 -- and those
#: two would sort between R1 and R2, so they were not merely cut off.
#:
#: HBN's own release table DOES list R10 and R11 at the same S3 URIs, 295
#: subjects each. The two disagree. A parent listing is not proof a prefix is
#: absent, so a release named explicitly is still attempted and the direct
#: probe in fetch_one gives the real answer; only `all` is limited to what was
#: actually seen, because expanding it to a name that 404s is how hours went
#: into retrying nothing.
#:
#: cmi_bids_NC is the "Not for Commercial Use" release and is deliberately NOT
#: here: the main model uses the standard releases only.
HBN_RELEASES_ALL="R1 R2 R3 R4 R5 R6 R7 R8 R9"
HBN_RELEASES_DOCUMENTED="R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11"

have_working_aws() {
    command -v aws >/dev/null 2>&1 || return 1
    aws --version >/dev/null 2>&1
}

aws_repair_hint() {
    cat >&2 <<'MSG'

  The `aws` on PATH is the abandoned Python-2 package of that name, not the
  AWS CLI. They install to the same bin/aws. To replace it:

      pip uninstall -y aws
      pip install awscli
      aws --version          # should print aws-cli/1.x or /2.x

MSG
}

layout() {
    mkdir -p "${EEG_ROOT}"/{FACED,TDBRAIN,PhysioNetMI,M3CV,HBN,HGD}/{raw,processed,manifests}
    echo "layout under ${EEG_ROOT}:"
    ls -d "${EEG_ROOT}"/*/ 2>/dev/null
    echo ""
    echo "'processed' and 'manifests' are there for per-corpus scratch. The"
    echo "pipeline itself writes shards and manifests into"
    echo "  \${DATA_ROOT:-${EEG_ROOT}}/eeg_c1_corpus/<dataset>/"
    echo "which is where TUEG already is and where the merge step reads from."
}

case "${1:-}" in

layout) layout ;;

physionet_mi)
    # Open access, no agreement. 64 channels at 160 Hz, EDF+.
    dest="${EEG_ROOT}/PhysioNetMI/raw"; mkdir -p "${dest}"
    if have_working_aws; then
        echo "==> S3 (faster than the HTTP mirror)"
        aws s3 sync --no-sign-request \
            s3://physionet-open/eegmmidb/1.0.0/ "${dest}/eegmmidb-1.0.0"
    else
        command -v aws >/dev/null 2>&1 && aws_repair_hint
        need wget "Install it, or repair the AWS CLI as above." || exit 1
        echo "==> HTTP mirror (wget; the AWS CLI is unusable here)"
        ( cd "${dest}" && wget -r -N -c -np -nH --cut-dirs=3 \
            https://physionet.org/files/eegmmidb/1.0.0/ )
    fi
    echo ""
    echo "downloaded $(find "${dest}" -name '*.edf' 2>/dev/null | wc -l) .edf file(s)"
    echo "expected 1526 (109 subjects x 14 runs)"
    ;;

faced|m3cv|hgd)
    # NEMAR serves one zip per published version straight over HTTPS:
    #   https://data.nemar.org/<accession>/<version>.zip
    # nemar-cli is a node package and adds nothing here, so it is not used --
    # a login node with no npm is the normal case, not a broken one.
    #
    # VERSIONS ARE PINNED, and verified against nemar.org on 2026-08-26. A
    # dataset that gets republished moves to a new version and this URL 404s
    # rather than silently fetching something else; NEMAR_VERSION overrides.
    case "$1" in
        faced) acc=nm000112; dir=FACED; ver=v1.1.3; approx="22.7 GB" ;;
        m3cv)  acc=nm000166; dir=M3CV;  ver=v1.0.0; approx="21.1 GB" ;;
        hgd)   acc=nm000172; dir=HGD;   ver=v1.0.3; approx="25.1 GB" ;;
    esac
    ver="${NEMAR_VERSION:-${ver}}"
    # NEMAR_BASE_URL exists so this branch can be exercised without pulling
    # 21 GB; it is not something a real run sets.
    url="${NEMAR_BASE_URL:-https://data.nemar.org}/${acc}/${ver}.zip"
    dest="${EEG_ROOT}/${dir}/raw"; mkdir -p "${dest}"
    zip="${dest}/${dir}_${acc}_${ver}.zip"

    echo "==> ${dir} ${acc} ${ver}  (~${approx})"
    echo "    ${url}"
    echo "    -> ${zip}"
    echo ""

    # An archive already on disk and sound is not re-fetched. `curl -C -` on a
    # COMPLETE file is an error in some versions rather than a no-op, so a
    # second run of this script would otherwise fail on the one thing that
    # already succeeded -- and re-running is exactly what someone does after an
    # interrupted 22 GB transfer.
    _have_it=0
    if [[ -s "${zip}" ]]; then
        echo "==> ${zip} exists; checking whether it is complete"
        if python "$(dirname "${BASH_SOURCE[0]}")/unpack_archive.py" "${zip}" \
                >/dev/null 2>&1; then
            echo "    complete and sound -- skipping the download"
            _have_it=1
        else
            echo "    incomplete or unreadable -- resuming the download"
        fi
    fi

    if [[ "${_have_it}" -eq 0 ]]; then
        # -C - resumes a partial file. At this size on a shared login node an
        # interrupted transfer is likely, and restarting 22 GB from zero because
        # the flag was missing is the kind of thing that costs an afternoon.
        if command -v curl >/dev/null 2>&1; then
            curl -fL -C - --retry 5 --retry-delay 10 \
                 --connect-timeout 20 -o "${zip}" "${url}"
            _rc=$?
            # 33: server refused the range. Start over rather than leave a file
            # that is half of one transfer and half of another.
            if [[ "${_rc}" -eq 33 ]]; then
                echo "    server refused a ranged request; restarting whole" >&2
                curl -fL --retry 5 --retry-delay 10 \
                     --connect-timeout 20 -o "${zip}" "${url}"
                _rc=$?
            fi
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "${zip}" "${url}"
            _rc=$?
        else
            echo "ERROR: neither curl nor wget is available." >&2
            exit 1
        fi
        if [[ "${_rc}" -ne 0 ]]; then
            echo "" >&2
            echo "ERROR: download failed (exit ${_rc})." >&2
            echo "  If it was a 404, the published version has moved. Check" >&2
            echo "    https://nemar.org/dataset/${acc}" >&2
            echo "  and re-run with the version it names:" >&2
            echo "    NEMAR_VERSION=v1.2.3 bash $0 $1" >&2
            exit 1
        fi
    fi

    echo ""
    echo "==> checking the archive before unpacking"
    # NOT unzip: Info-ZIP rejects these as possible zip bombs, and the
    # environment variable it suggests would force a truncated download open
    # just as readily as a sound one.
    python "$(dirname "${BASH_SOURCE[0]}")/unpack_archive.py" "${zip}" \
        --extract-to "${dest}" || exit 1

    if [[ "$1" == "faced" ]]; then
        echo ""
        echo "FACED note: 92 subjects are 1000 Hz and 31 are 250 Hz, in this"
        echo "same release. Both are accepted -- the 250 Hz ones are the real"
        echo "acquisition, and every shard written from one records"
        echo "upsampled_from_hz in its provenance. A rate that is NEITHER is"
        echo "refused, because that is the preprocessed derivative."
    fi
    if [[ "$1" == "m3cv" ]]; then
        echo ""
        echo "M3CV note: this release is already cleaned (band-pass, 49-51 Hz"
        echo "notch, ICA, 1000 -> 250 Hz). The registry marks it"
        echo "notch_already_applied, so no second notch is run. Do not describe"
        echo "it as raw 1000 Hz data."
    fi
    ;;

hbn)
    if ! have_working_aws; then
        if command -v aws >/dev/null 2>&1; then
            echo "ERROR: the AWS CLI on PATH does not run." >&2
            aws_repair_hint
        else
            echo "ERROR: aws is not on PATH." >&2
            echo "" >&2
            # Overwhelmingly the reason: the prep venv is not active. Everything
            # in this pipeline that reads EEG lives in it, and a shell that has
            # lost it fails on whichever tool is reached for first.
            if [[ -z "${VIRTUAL_ENV:-}" ]]; then
                echo "  No virtualenv is active. The preparation venv is where" >&2
                echo "  mne and the AWS CLI live:" >&2
                echo "    source \$HOME/pwprep/bin/activate" >&2
                echo "  (your prompt should read (pwprep) again)" >&2
            else
                echo "  Active venv: ${VIRTUAL_ENV}" >&2
                echo "    pip install awscli" >&2
            fi
        fi
        echo "" >&2
        echo "  HBN is S3-only -- there is no HTTP mirror to fall back to." >&2
        exit 1
    fi
    rel="${2:-}"
    if [[ -z "${rel}" ]]; then
        echo "ERROR: name a release, or 'all' for R1-R11 (1.875 TB)." >&2
        echo "  bash $0 hbn R3      # 140 GB, enough to validate the adapter" >&2
        echo "  bash $0 hbn all     # every standard release" >&2
        exit 1
    fi
    fetch_one() {
        local r="$1"
        local dest="${EEG_ROOT}/HBN/raw/${r}"
        # TRAILING SLASH, deliberately. Without it "cmi_bids_R1" is a prefix
        # that also matches cmi_bids_R10 and cmi_bids_R11.
        local src="s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${r}/"

        echo "==> HBN ${r} -> ${dest}"

        # Does the prefix exist at all? `aws s3 sync` against one that does not
        # EXITS 0 HAVING DONE NOTHING, so a wrong bucket path and a completed
        # download are indistinguishable by exit code -- and mkdir has already
        # left a directory that looks exactly like a finished one.
        # 2>&1 here was a bug: it folded aws's ERROR TEXT into the variable, so
        # a failed listing came back non-empty and read as "the prefix exists".
        # Stderr is kept separate and the exit status is what decides.
        local remote lserr lsrc
        lserr=$(mktemp "${TMPDIR:-/tmp}/pw_awsls.XXXXXX" 2>/dev/null) || lserr=/dev/null
        local ls_try=1
        while :; do
            remote=$(aws s3 ls "${src}" --no-sign-request 2>"${lserr}")
            lsrc=$?
            [[ "${lsrc}" -eq 0 && -n "${remote}" ]] && break
            grep -qi "could not connect\|endpoint\|timed out\|connection" \
                "${lserr}" 2>/dev/null || break
            [[ "${ls_try}" -ge 3 ]] && break
            echo "    listing failed to connect; retry ${ls_try}/3 in 15s" >&2
            sleep 15
            ls_try=$(( ls_try + 1 ))
        done
        if [[ "${lsrc}" -ne 0 ]] || [[ -z "${remote}" ]]; then
            echo "    ERROR: cannot list ${src} (aws exit ${lsrc})." >&2
            if [[ -s "${lserr}" ]]; then
                echo "    aws said:" >&2
                sed 's/^/      /' "${lserr}" >&2
            elif [[ "${lsrc}" -eq 0 ]]; then
                echo "    The listing was empty: the release name is wrong or" >&2
                echo "    the bucket layout moved." >&2
            fi
            if grep -qi "could not connect\|endpoint" "${lserr}" 2>/dev/null; then
                echo "" >&2
                echo "    This is a NETWORK failure, not a wrong path: the node" >&2
                echo "    had no route to S3. The release is fine; resubmit and" >&2
                echo "    it will likely land on a node that can reach it." >&2
            fi
            [[ "${lserr}" != /dev/null ]] && rm -f "${lserr}"
            return 1
        fi
        [[ "${lserr}" != /dev/null ]] && rm -f "${lserr}"

        mkdir -p "${dest}"

        # THROTTLE. The CLI defaults to ten concurrent transfers with a large
        # queue, which on 100-245 GB of multi-megabyte files is enough resident
        # memory for a login node to SIGKILL the process -- exit 137, partway
        # through, looking like a corrupt download rather than a policy limit.
        # Written to a temp config rather than `aws configure set`, which would
        # edit ~/.aws/config permanently and change every other aws command the
        # user runs.
        # $TMPDIR is set per job on some schedulers and is not always writable
        # by the time a job body runs. Falling back rather than failing the
        # whole release: throttling is an optimisation, and losing it is not a
        # reason to skip a 200 GB download.
        local awscfg=""
        local d
        for d in "${TMPDIR:-}" /tmp "${dest}"; do
            [[ -z "${d}" || ! -d "${d}" || ! -w "${d}" ]] && continue
            awscfg=$(mktemp "${d}/pw_aws_cfg.XXXXXX" 2>/dev/null) && break
            awscfg=""
        done
        if [[ -z "${awscfg}" ]]; then
            echo "    NOTE: no writable temp dir for the aws config; running" >&2
            echo "          at the CLI's default concurrency." >&2
        fi
        local prev_cfg="${AWS_CONFIG_FILE:-}"
        if [[ -n "${awscfg}" ]]; then
            cat > "${awscfg}" <<CFG
[default]
retry_mode = standard
max_attempts = ${AWS_MAX_ATTEMPTS:-10}
s3 =
    max_concurrent_requests = ${AWS_CONCURRENCY:-4}
    max_queue_size = 100
    multipart_chunksize = 8MB
    multipart_threshold = 64MB
CFG
            export AWS_CONFIG_FILE="${awscfg}"
        fi

        # sync, not `cp --recursive`. At 100-245 GB per release a transfer WILL
        # be interrupted, and cp restarts every file from the beginning; sync
        # skips what is already there with a matching size. That is the
        # difference between resuming a release and re-fetching it.
        # "Could not connect to the endpoint URL" is what killed R4, R10 and
        # R11 while seven other releases downloaded fine from other nodes: a
        # compute node with no route to S3 at that moment. botocore's own
        # retries do not cover it -- it gives up and returns -- so the sync is
        # retried here, with backoff, and the transfer resumes from whatever
        # landed before the connection went.
        local rc=1 attempt=1
        local tries="${AWS_SYNC_TRIES:-4}"
        while [[ "${attempt}" -le "${tries}" ]]; do
            [[ "${attempt}" -gt 1 ]] && \
                echo "    attempt ${attempt}/${tries} (resuming)"
            aws s3 sync "${src}" "${dest}" --no-sign-request
            rc=$?
            [[ "${rc}" -eq 0 ]] && break
            # 130/137/143 are signals -- Ctrl-C or the scheduler. Retrying
            # something that was deliberately killed just gets killed again.
            if [[ "${rc}" -eq 130 || "${rc}" -eq 137 || "${rc}" -eq 143 ]]; then
                break
            fi
            if [[ "${attempt}" -lt "${tries}" ]]; then
                local wait=$(( attempt * 30 ))
                echo "    exit ${rc}; retrying in ${wait}s" >&2
                sleep "${wait}"
            fi
            attempt=$(( attempt + 1 ))
        done
        if [[ "${rc}" -ne 0 ]]; then
            echo "    aws s3 sync exited ${rc} after ${attempt} attempt(s)." >&2
            echo "    Its message is above, or in the job's .err under sbatch." >&2
        fi

        [[ -n "${awscfg}" ]] && rm -f "${awscfg}"
        if [[ -n "${prev_cfg}" ]]; then export AWS_CONFIG_FILE="${prev_cfg}"
        else unset AWS_CONFIG_FILE; fi

        # 137 is SIGKILL. On a login node that is the site's limit, not a
        # problem with the data -- and sync is resumable, so the answer is to
        # rerun it somewhere it is allowed to finish.
        if [[ "${rc}" -eq 137 ]] || [[ "${rc}" -eq 143 ]]; then
            echo "" >&2
            echo "    KILLED (exit ${rc}) -- a signal, not a download error." >&2
            echo "    A login node will not let a transfer this size finish." >&2
            echo "    What is on disk is intact and sync resumes from it:" >&2
            echo "      sbatch --export=ALL,RELEASES=${r} \\" >&2
            echo "             scripts/slurm/cineca_hbn_download.sbatch" >&2
        fi

        local n
        n=$(find -L "${dest}" -type f \( -iname '*.set' -o -iname '*.fdt' \
            -o -iname '*.edf' -o -iname '*.mff' \) 2>/dev/null | wc -l)
        echo "    ${r}: exit ${rc}, $(du -sh "${dest}" 2>/dev/null | cut -f1)"\
             "in ${n} recording file(s)"

        # An empty destination after a "successful" sync is a failure, whatever
        # the exit code said.
        if [[ "${n}" -eq 0 ]]; then
            echo "    ERROR: ${r} downloaded 0 recording files. Treating the" >&2
            echo "           sync as failed rather than reporting success on" >&2
            echo "           an empty directory." >&2
            return 1
        fi
        return "${rc}"
    }
    _hbn_rc=0
    if [[ "${rel}" == "all" ]]; then
        for _r in ${HBN_RELEASES_ALL}; do fetch_one "${_r}" || _hbn_rc=$?; done
    else
        case " ${HBN_RELEASES_ALL} " in
            *" ${rel} "*) ;;
            *) echo "NOTE: ${rel} was not in the parent listing on 2026-08-27," >&2
               echo "      which showed ${HBN_RELEASES_ALL} and NC. HBN's own" >&2
               echo "      release table does list R10 and R11 at these URIs," >&2
               echo "      so this is attempted anyway -- the direct probe" >&2
               echo "      below is what actually settles it." >&2 ;;
        esac
        fetch_one "${rel}" || _hbn_rc=$?
    fi
    [[ "${_hbn_rc:-0}" -eq 0 ]] || exit "${_hbn_rc}"
    echo ""
    echo "HBN note: files report 129 EEG channels. The adapter removes the"
    echo "NAMED vertex reference to reach 128 and writes what it removed into"
    echo "every shard's provenance. It refuses to drop a row by position, so a"
    echo "release whose reference is labelled differently fails loudly rather"
    echo "than quietly deleting whichever channel came last."
    ;;

probe-hbn)
    # Does a given prefix hold anything? The parent listing and HBN's release
    # table disagree about R10 and R11, and this is the test that decides:
    # listing the prefix ITSELF rather than looking for it in its parent.
    if ! have_working_aws; then
        echo "ERROR: the AWS CLI does not run." >&2; aws_repair_hint; exit 1
    fi
    rels="${2:-${HBN_RELEASES_DOCUMENTED}}"
    echo "listing each prefix directly (not via the parent):"
    echo ""
    for r in ${rels}; do
        pfx="s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${r}/"
        n=$(aws s3 ls "${pfx}" --no-sign-request 2>&1 | head -5)
        if [[ -z "${n}" ]]; then
            printf "  %-4s EMPTY -- nothing under %s\n" "${r}" "${pfx}"
        elif echo "${n}" | grep -qi "error\|could not"; then
            printf "  %-4s ERROR: %s\n" "${r}" "$(echo "${n}" | head -1)"
        else
            printf "  %-4s exists (%s...)\n" "${r}" \
                "$(echo "${n}" | head -1 | awk '{print $NF}')"
        fi
    done
    echo ""
    echo "A prefix that lists objects here is fetchable regardless of whether"
    echo "it appeared in the parent listing."
    ;;

verify-hbn)
    # "It finished quickly" is the symptom worth checking. A sync that failed
    # early, or one that only created the directory tree, leaves exactly the
    # layout a complete download leaves -- eleven directories named R1..R11 --
    # and nothing downstream notices until preprocessing reports a corpus a
    # tenth the expected size.
    #
    # So: compare against S3 itself, per release, in objects and in bytes.
    if ! have_working_aws; then
        echo "ERROR: the AWS CLI does not run; cannot check against S3." >&2
        aws_repair_hint
        exit 1
    fi
    rels="${2:-all}"
    [[ "${rels}" == "all" ]] && rels="${HBN_RELEASES_ALL}"

    printf "%-5s %12s %12s %14s %14s  %s\n" \
        release "local objs" "s3 objs" "local bytes" "s3 bytes" verdict
    _bad=0
    for r in ${rels}; do
        dest="${EEG_ROOT}/HBN/raw/${r}"
        if [[ ! -d "${dest}" ]]; then
            printf "%-5s %12s %12s %14s %14s  %s\n" "${r}" - - - - "ABSENT"
            _bad=$((_bad + 1)); continue
        fi
        l_objs=$(find -L "${dest}" -type f 2>/dev/null | wc -l | tr -d ' ')
        l_bytes=$(tree_bytes "${dest}")
        summary=$(aws s3 ls --recursive --summarize --no-sign-request \
            "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${r}/" 2>/dev/null \
            | tail -3)
        s_objs=$(echo "${summary}" | awk '/Total Objects/ {print $3}')
        s_bytes=$(echo "${summary}" | awk '/Total Size/ {print $3}')
        s_objs="${s_objs:-0}"; s_bytes="${s_bytes:-0}"

        if [[ "${s_objs}" -eq 0 ]]; then
            verdict="S3 LISTING FAILED"
            _bad=$((_bad + 1))
        elif [[ "${l_objs}" -eq "${s_objs}" ]] && [[ "${l_bytes}" -eq "${s_bytes}" ]]; then
            verdict="complete"
        else
            pct=$(awk -v a="${l_bytes}" -v b="${s_bytes}" \
                  'BEGIN {printf "%.1f", (b ? 100*a/b : 0)}')
            verdict="INCOMPLETE ${pct}%"
            _bad=$((_bad + 1))
        fi
        printf "%-5s %12s %12s %14s %14s  %s\n" \
            "${r}" "${l_objs}" "${s_objs}" "${l_bytes}" "${s_bytes}" "${verdict}"
    done

    echo ""
    if [[ "${_bad}" -eq 0 ]]; then
        echo "every release checked matches S3 in object count and byte count."
    else
        echo "${_bad} release(s) do not match S3."
        echo "  Re-run the sync; it skips what is already there at a matching"
        echo "  size, so this costs a listing for the parts already done:"
        echo "    sbatch --export=ALL,RELEASES=\"${rels}\" \\"
        echo "           scripts/slurm/cineca_hbn_download.sbatch"
    fi
    exit "$(( _bad > 0 ))"
    ;;

tdbrain)
    cat <<'MSG'
TDBRAIN is not downloaded by this script.

It requires an ORCID login and an accepted Data Use Agreement, and the DUA also
forbids redistribution -- so the transfer has to be yours, made under your own
acceptance of it.

  1. https://brainclinics.com/resources/tdbrain-dataset
  2. Register with an INSTITUTIONAL address. Their spam protection may drop
     mail to gmail and the like, and the download password comes by email.
  3. Take "TDBRAIN Dataset V3.1". ~15 GB zipped, ~20 GB unpacked.
  4. Put the zip under $EEG_ROOT/TDBRAIN/raw.

The zip is PASSWORD PROTECTED. The password is in the DUA email. Unpack with:

    TDBRAIN_PW='<from the email>' \
    python scripts/unpack_archive.py TDBRAIN_V3.1.zip \
        --extract-to . --password-env TDBRAIN_PW

--password-env rather than --password so it stays out of shell history and out
of `ps`. If it reports AES encryption, Python's zipfile cannot read that: use
`7z x` or `pip install pyzipper` and rerun.

Then, as with every other corpus:

  DATASET=tdbrain INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh

Read the coverage line before launching anything. TDBRAIN records 26 electrodes
against E32_512's 32 slots, so expect 26 of 32 (81%) and an empty-slot rate of
18.8% -- both inside the gates. The six unrecorded slots stay zero with
valid_channel_mask False. Nothing interpolates them: a spatially interpolated
channel is a smooth function of its neighbours, and the model would learn the
interpolation.

Two things to check in that report, both new in V3:
  * The format is BDF/BDF+. The BrainVision and .csv releases are withdrawn.
  * Every file carries a BioSemi "Status" channel as its LAST channel, all
    zeros in the resting recordings. It should appear under DROPPED and NOT in
    the filled slots. It is dropped because slots are matched by NAME; a
    positional rule would have put an all-zero channel in an electrode's slot.
MSG
    ;;

*) usage; exit 1 ;;
esac
