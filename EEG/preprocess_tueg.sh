#!/bin/bash
# ============================================================================
# TUEG -> the EEG C1 pretraining corpus (route E19_256).
#
#   bash EEG/preprocess_tueg.sh                    # everything, one process
#   SHARD=3/64 bash EEG/preprocess_tueg.sh         # one shard of an array
#
# TUEG is the largest corpus in the mixture and the only one that has to be
# processed in parallel: tens of thousands of EDF files, each read, filtered,
# resampled from whatever rate it was recorded at, and written as its own HDF5.
#
# LOOK BEFORE YOU RUN. The adapter's assumptions about layout and channel
# naming have been verified against a synthetic tree with TUH's conventions, not
# against this copy of the corpus. One command checks them, opens headers only
# and takes seconds:
#
#   INSPECT=200 bash EEG/preprocess_tueg.sh
#
# Then build the file listing ONCE, before the array. Walking this tree costs
# tens of minutes on Lustre and every task would otherwise pay it separately --
# which is exactly what made two of sixteen tasks spend forty-four minutes
# before processing a single file:
#
#   bash EEG/preprocess_tueg.sh --write-file-list
#
# Read the report before launching the array. What matters in it:
#   * identity rule should be 'filename' for every file. Any 'path' means the
#     filenames do not follow <subject>_s<NNN>_t<NNN>.edf here and the subject
#     ids -- and therefore the train/val split -- are coming from a fallback.
#   * slots filled should be 19 of 19. Anything less is an electrode this
#     adapter cannot name, and it would be silently masked out of every window.
#   * unmatched names should be only EKG / PHOTIC / IBI / BURSTS / SUPPR.
#
# SHARDING is by SUBJECT, not by file: a subject's recordings must all be
# handled by one task, because the train/val side is decided per subject and two
# tasks disagreeing about a subject is the leak the split exists to prevent.
# --split-mode hash follows automatically from --shard.
#
# RESUMABLE. A shard whose HDF5 already exists is reused, so a task that hits
# the walltime loses at most the recording it was in. Resubmit the same array.
#
# DISK. Each window is 19 x 1024 float32 = 78 KB before gzip, ~40 KB after.
# --inspect projects the total from the durations it reads, before anything is
# written. MAX_RECORDINGS caps the file count; STRIDE_SECONDS larger than the
# 4 s window subsamples instead, which keeps every recording represented rather
# than keeping some and dropping others entirely.
#
# HOW LONG IT TAKES is a property of this filesystem and these files, not
# something to guess. Measure it before sizing the array:
#
#   time JOBS=8 MAX_RECORDINGS=40 OUT_DIR=/tmp/tueg_probe \
#       bash EEG/preprocess_tueg.sh
#
# and divide: 70831 files / (40 files / elapsed) / N_TASKS.
#
# ENVIRONMENT VARIABLES:
#   TUEG_ROOT       raw corpus     ($SCRATCH/TUH_EEG/TUEG_v2.0.2)
#   DATA_ROOT       corpus root    ($PW_DATA_EEG/eeg_c1_corpus)
#   OUT_DIR         this dataset   ($DATA_ROOT/tueg)
#   SHARD           "I/N"          (unset = one process does everything)
#   INSPECT         N              (report and exit, process nothing)
#   MAINS_HZ                       (60 -- TUH is recorded in Philadelphia)
#   MAX_RECORDINGS  cap            (unset)
#   STRIDE_SECONDS  window stride  (unset = no overlap)
#   JOBS            worker processes in this task (1)
#   FILE_LIST       cached corpus listing ($OUT_DIR/tueg_files.txt)
#   MAX_MINUTES     cap on one recording, in memory terms (30)
#   REDO_TRUNCATED  1 to redo the recordings an earlier, lower cap
#                   cut short -- and only those. Raising
#                   MAX_MINUTES alone does nothing, because every
#                   shard exists and RESUME skips it.
#   RESUME          1 to reuse existing shards (1)
#   VAL_FRACTION / SPLIT_SEED      (0.10 / 42)
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Paths only: reading EDF needs mne, which lives in the preparation venv, and
# activating the training venv here would replace it.
PW_VARS_ONLY=1 source "$(pwd)/scripts/cineca_env.sh"

TUEG_ROOT="${TUEG_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/TUH_EEG/TUEG_v2.0.2}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/tueg}"
MAINS_HZ="${MAINS_HZ:-60}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
RESUME="${RESUME:-1}"
JOBS="${JOBS:-1}"
FILE_LIST="${FILE_LIST:-${OUT_DIR}/tueg_files.txt}"
MAX_MINUTES="${MAX_MINUTES:-30}"
REDO_TRUNCATED="${REDO_TRUNCATED:-0}"
SHARD="${SHARD:-}"
INSPECT="${INSPECT:-}"
MAX_RECORDINGS="${MAX_RECORDINGS:-}"
STRIDE_SECONDS="${STRIDE_SECONDS:-}"

if [[ ! -d "${TUEG_ROOT}" ]]; then
    echo "ERROR: no TUEG at ${TUEG_ROOT}" >&2
    echo "       Set TUEG_ROOT. This script never downloads the corpus -- TUEG" >&2
    echo "       is behind a data use agreement and fetching it on your behalf" >&2
    echo "       would be agreeing to that on your behalf." >&2
    exit 1
fi

if ! python -c "import mne" 2>/dev/null; then
    echo "ERROR: mne is not importable in $(python -c 'import sys;print(sys.prefix)')." >&2
    echo "       Reading EDF needs it, and it lives in the preparation venv:" >&2
    echo "         source \$HOME/pwprep/bin/activate" >&2
    echo "         PW_VARS_ONLY=1 source scripts/cineca_env.sh" >&2
    exit 1
fi

ARGS=(--dataset tueg --root "${TUEG_ROOT}" --out-dir "${OUT_DIR}"
      --mains-hz "${MAINS_HZ}" --val-fraction "${VAL_FRACTION}"
      --split-seed "${SPLIT_SEED}")
[[ -n "${INSPECT}" ]]        && ARGS+=(--inspect "${INSPECT}")
[[ -n "${SHARD}" ]]          && ARGS+=(--shard "${SHARD}")
[[ -n "${MAX_RECORDINGS}" ]] && ARGS+=(--max-recordings "${MAX_RECORDINGS}")
[[ -n "${STRIDE_SECONDS}" ]] && ARGS+=(--stride-seconds "${STRIDE_SECONDS}")
[[ "${RESUME}" == "1" ]]     && ARGS+=(--resume)
[[ "${JOBS}" -gt 1 ]]        && ARGS+=(--jobs "${JOBS}")
[[ -n "${FILE_LIST}" ]]      && ARGS+=(--file-list "${FILE_LIST}")
ARGS+=(--max-recording-minutes "${MAX_MINUTES}")
[[ "${REDO_TRUNCATED}" == "1" ]] && ARGS+=(--redo-truncated)

echo "============================================================"
echo "  TUEG -> E19_256  (19 x 1024 @ 256 Hz, 152 tokens)"
echo "  raw   ${TUEG_ROOT}"
echo "  out   ${OUT_DIR}"
echo "  mains ${MAINS_HZ} Hz   val ${VAL_FRACTION} (seed ${SPLIT_SEED})"
[[ -n "${SHARD}" ]]   && echo "  shard ${SHARD} (by subject)"
[[ "${JOBS}" -gt 1 ]] && echo "  ${JOBS} worker process(es)"
[[ -n "${INSPECT}" ]] && echo "  INSPECT ${INSPECT} -- reporting only, nothing written"
echo "============================================================"

# One-shot listing mode, so the array does not walk the tree sixteen times.
if [[ "${1:-}" == "--write-file-list" ]]; then
    mkdir -p "${OUT_DIR}"
    python EEG/preprocess_pretrain_corpus.py --dataset tueg \
        --root "${TUEG_ROOT}" --out-dir "${OUT_DIR}" \
        --write-file-list "${FILE_LIST}"
    exit $?
fi

python EEG/preprocess_pretrain_corpus.py "${ARGS[@]}"
_rc=$?

if [[ ${_rc} -eq 0 && -z "${INSPECT}" ]]; then
    echo ""
    echo "When every shard has finished, merge the manifests:"
    echo "  python scripts/build_eeg_c1_manifest.py --corpus-root ${DATA_ROOT} \\"
    echo "      --datasets tueg --allow-missing --check-shards"
fi
exit "${_rc}"
