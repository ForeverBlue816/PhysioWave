#!/bin/bash
# ============================================================================
# FACED / TDBRAIN / PhysioNetMI / M3CV / HBN / HGD -> the C1 pretraining corpus.
#
#   DATASET=physionet_mi INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh
#   DATASET=physionet_mi bash EEG/preprocess_eeg_corpus.sh
#   DATASET=hbn SHARD=3/32 bash EEG/preprocess_eeg_corpus.sh
#
# TUEG has its own script because it needs a cached file listing and a
# per-recording memory cap that the others do not. Everything else about the
# pipeline is identical, so these six share one driver.
#
# INSPECT FIRST, ALWAYS. Every adapter here has been verified against a
# synthetic tree with the corpus's naming conventions -- not against your copy
# of it. --inspect opens headers only and answers the three questions that
# decide whether an array is worth launching:
#
#   * PER-FILE slot coverage. Not the union. A corpus whose files each carry
#     two thirds of the montage reports a perfect union and produces windows
#     that are more mask than measurement.
#   * Which names were dropped, and whether any of them is a real electrode.
#   * With VERIFY_POWERLINE=1, whether the line peak is where the registry
#     (configs/pretrain/eeg_c1_datasets.yaml) says it is. FACED and PhysioNetMI
#     publish no trustworthy PowerLineFrequency field, and a notch at the wrong
#     frequency leaves the interference in AND removes real signal.
#
# The mains frequency comes from that registry, not from this script. MAINS_HZ
# overrides it per run; the registry is where a corrected value belongs.
#
# ENVIRONMENT VARIABLES:
#   DATASET         faced|tdbrain|physionet_mi|m3cv|hbn|hgd     (required)
#   RAW_ROOT        raw corpus            ($EEG_ROOT/<Name>/raw)
#   DATA_ROOT       corpus root           ($PW_DATA_EEG/eeg_c1_corpus)
#   OUT_DIR         this dataset          ($DATA_ROOT/$DATASET)
#   SHARD           "I/N" for an array    (unset = one process does everything)
#   JOBS            worker processes      (1)
#   INSPECT         N: report and exit, process nothing
#   DUMP_CHANNELS   N: print N recordings' channel lists verbatim and exit.
#                   What to reach for when INSPECT reports poor coverage.
#   DERIVE_SLOTS    PATH: write a candidate slot list for this route, derived
#                   from the corpus, plus the vocabulary names it needs.
#   FILE_LIST       cached corpus listing ($OUT_DIR/<dataset>_files.txt).
#                   Delete it after adding or removing raw files.
#   VERIFY_POWERLINE  1: with INSPECT, measure the 50/60 Hz bands
#   PSD_VERIFIED    1: acknowledge you have done that; silences the reminder
#   MAINS_HZ        override the registry (unset)
#   MIN_COVERAGE    per-recording slot coverage floor (0.75)
#   MAX_EMPTY_RATE  corpus-wide empty-slot ceiling    (0.25)
#   MAX_MINUTES     cap on one recording, in memory terms (30; 0 disables)
#   MAX_RECORDINGS  cap the file count    (unset)
#   STRIDE_SECONDS  window stride         (unset = no overlap)
#   RESUME          1 to reuse existing shards (1)
#   VAL_FRACTION / SPLIT_SEED   (0.10 / 42)  -- MUST match every other corpus
#   ALLOW_UPSAMPLE_FACED  1: accept FACED at a rate the registry does not list.
#                   The official configuration never sets this. FACED's 31
#                   genuinely-250 Hz subjects are accepted WITHOUT it.
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PW_VARS_ONLY=1 source "$(pwd)/scripts/cineca_env.sh"

DATASET="${DATASET:-}"
case "${DATASET}" in
    faced|tdbrain|physionet_mi|m3cv|hbn|hgd|siena) ;;
    "")  echo "ERROR: set DATASET. One of: faced tdbrain physionet_mi m3cv hbn hgd siena" >&2
         echo "       (TUEG has its own script: EEG/preprocess_tueg.sh)" >&2
         exit 1 ;;
    tueg) echo "ERROR: TUEG uses EEG/preprocess_tueg.sh -- it needs the cached" >&2
          echo "       file listing and the memory cap this driver does not set." >&2
          exit 1 ;;
    *)   echo "ERROR: unknown DATASET '${DATASET}'." >&2; exit 1 ;;
esac

# Default directory names follow the layout in the download guide. A case
# rather than an associative array: bash 3.2 has no `declare -A`, and this is
# the difference between the script being testable on a laptop and only ever
# being exercised for the first time inside a SLURM task.
case "${DATASET}" in
    faced)        DIRNAME=FACED;       ROUTE="E32_512   32 x 2048 @ 512 Hz, 256 tokens" ;;
    tdbrain)      DIRNAME=TDBRAIN;     ROUTE="E32_512   32 x 2048 @ 512 Hz, 256 tokens (26 recorded, 6 padded)" ;;
    physionet_mi) DIRNAME=PhysioNetMI; ROUTE="E64_256   64 x 1024 @ 256 Hz, 512 tokens" ;;
    m3cv)         DIRNAME=M3CV;        ROUTE="E64_256   64 x 1024 @ 256 Hz, 512 tokens" ;;
    hbn)          DIRNAME=HBN;         ROUTE="E128_512  128 x 2048 @ 512 Hz, 1024 tokens" ;;
    hgd)          DIRNAME=HGD;         ROUTE="E128_512  128 x 2048 @ 512 Hz, 1024 tokens" ;;
    siena)        DIRNAME=Siena;       ROUTE="E19_256   19 x 1024 @ 256 Hz, 152 tokens (continuation corpus)" ;;
esac

EEG_ROOT="${EEG_ROOT:-${PW_DATA_EEG}}"
RAW_ROOT="${RAW_ROOT:-${EEG_ROOT}/${DIRNAME}/raw}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/${DATASET}}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
RESUME="${RESUME:-1}"
JOBS="${JOBS:-1}"
MIN_COVERAGE="${MIN_COVERAGE:-0.75}"
MAX_EMPTY_RATE="${MAX_EMPTY_RATE:-0.25}"
MAX_MINUTES="${MAX_MINUTES:-30}"
SHARD="${SHARD:-}"
INSPECT="${INSPECT:-}"
DUMP_CHANNELS="${DUMP_CHANNELS:-}"
DERIVE_SLOTS="${DERIVE_SLOTS:-}"
# Cached file listing. Walking a BIDS tree of tens of thousands of entries on a
# parallel filesystem takes minutes, and an --array=0-63 would otherwise pay it
# once per task.
FILE_LIST="${FILE_LIST:-${OUT_DIR}/${DATASET}_files.txt}"
MAINS_HZ="${MAINS_HZ:-}"
MAX_RECORDINGS="${MAX_RECORDINGS:-}"
STRIDE_SECONDS="${STRIDE_SECONDS:-}"
VERIFY_POWERLINE="${VERIFY_POWERLINE:-0}"
PSD_VERIFIED="${PSD_VERIFIED:-0}"
ALLOW_UPSAMPLE_FACED="${ALLOW_UPSAMPLE_FACED:-0}"

# An EMPTY directory is not an absent one, and `bash ... layout` creates all six
# before anything is downloaded -- so the -d test passes for every corpus that
# has not been fetched yet, and the failure surfaces pages later as "no readable
# files". Count first.
if [[ -d "${RAW_ROOT}" ]]; then
    _n_files=$(find -L "${RAW_ROOT}" -type f \
        \( -iname '*.edf' -o -iname '*.bdf' -o -iname '*.set' \
           -o -iname '*.fif' -o -iname '*.cnt' -o -iname '*.vhdr' \) \
        2>/dev/null | head -1 | wc -l)
    if [[ "${_n_files}" -eq 0 ]]; then
        echo "ERROR: ${RAW_ROOT} exists but holds no EDF/BDF/SET/FIF/CNT/VHDR." >&2
        if compgen -G "${RAW_ROOT}/*.zip" >/dev/null 2>&1; then
            echo "       There IS a .zip here that has not been unpacked:" >&2
            ls -1 "${RAW_ROOT}"/*.zip 2>/dev/null | sed 's/^/         /' >&2
            echo "       unzip refuses these archives; use:" >&2
            echo "         python scripts/unpack_archive.py <file>.zip --extract-to ." >&2
            exit 1
        fi
        echo "       The download has not run, or it landed somewhere else." >&2
        echo "         bash scripts/download_eeg_pretrain_corpora.sh ${DATASET}" >&2
        echo "       Or point RAW_ROOT at where the files actually are:" >&2
        echo "         DATASET=${DATASET} RAW_ROOT=/path/to/corpus \\" >&2
        echo "             bash EEG/preprocess_eeg_corpus.sh" >&2
        exit 1
    fi
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
    echo "ERROR: no ${DATASET} corpus at ${RAW_ROOT}" >&2
    echo "       Set RAW_ROOT. This script never downloads a corpus: TDBRAIN" >&2
    echo "       and HBN are behind data use agreements, and fetching them on" >&2
    echo "       your behalf would be accepting those on your behalf." >&2
    echo "       See scripts/download_eeg_pretrain_corpora.sh for the commands." >&2
    exit 1
fi

if ! python -c "import mne" 2>/dev/null; then
    echo "ERROR: mne is not importable in $(python -c 'import sys;print(sys.prefix)')." >&2
    echo "       Reading EDF/BDF/SET needs it, and it lives in the prep venv:" >&2
    echo "         source \$HOME/pwprep/bin/activate" >&2
    exit 1
fi

ARGS=(--dataset "${DATASET}" --root "${RAW_ROOT}" --out-dir "${OUT_DIR}"
      --val-fraction "${VAL_FRACTION}" --split-seed "${SPLIT_SEED}"
      --min-slot-coverage "${MIN_COVERAGE}"
      --max-empty-slot-rate "${MAX_EMPTY_RATE}"
      --max-recording-minutes "${MAX_MINUTES}")
[[ -n "${MAINS_HZ}" ]]              && ARGS+=(--mains-hz "${MAINS_HZ}")
[[ -n "${INSPECT}" ]]               && ARGS+=(--inspect "${INSPECT}")
[[ -n "${DUMP_CHANNELS}" ]]         && ARGS+=(--dump-channels "${DUMP_CHANNELS}")
[[ -n "${DERIVE_SLOTS}" ]]          && ARGS+=(--derive-slots "${DERIVE_SLOTS}")
[[ -n "${FILE_LIST}" ]]             && ARGS+=(--file-list "${FILE_LIST}")
[[ -n "${SHARD}" ]]                 && ARGS+=(--shard "${SHARD}")
[[ -n "${MAX_RECORDINGS}" ]]        && ARGS+=(--max-recordings "${MAX_RECORDINGS}")
[[ -n "${STRIDE_SECONDS}" ]]        && ARGS+=(--stride-seconds "${STRIDE_SECONDS}")
[[ "${RESUME}" == "1" ]]            && ARGS+=(--resume)
[[ "${JOBS}" -gt 1 ]]               && ARGS+=(--jobs "${JOBS}")
[[ "${VERIFY_POWERLINE}" == "1" ]]  && ARGS+=(--verify-powerline)
[[ "${PSD_VERIFIED}" == "1" ]]      && ARGS+=(--psd-verified)
if [[ "${ALLOW_UPSAMPLE_FACED}" == "1" ]]; then
    echo "WARNING: --allow-upsample-faced accepts FACED at ANY rate, including" >&2
    echo "         the 250 Hz preprocessed release. The 31 subjects genuinely" >&2
    echo "         recorded at 250 Hz do NOT need it. Do not set this for the" >&2
    echo "         official run." >&2
    ARGS+=(--allow-upsample-faced)
fi

echo "============================================================"
echo "  ${DATASET} -> ${ROUTE}"
echo "  raw   ${RAW_ROOT}"
echo "  out   ${OUT_DIR}"
echo "  val   ${VAL_FRACTION} (seed ${SPLIT_SEED})"
[[ -n "${MAINS_HZ}" ]] && echo "  mains ${MAINS_HZ} Hz (OVERRIDING the registry)"
[[ -n "${SHARD}" ]]    && echo "  shard ${SHARD} (by subject)"
[[ "${JOBS}" -gt 1 ]]  && echo "  ${JOBS} worker process(es)"
[[ -n "${INSPECT}" ]]  && echo "  INSPECT ${INSPECT} -- reporting only, nothing written"
echo "============================================================"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${OUT_DIR}"
exec python EEG/preprocess_pretrain_corpus.py "${ARGS[@]}"
