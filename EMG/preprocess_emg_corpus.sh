#!/bin/bash
# ============================================================================
# One sEMG corpus -> the sEMG C1 pretraining corpus.
#
#   DATASET=hyser INSPECT=20 bash EMG/preprocess_emg_corpus.sh       # look
#   DATASET=hyser JOBS=16 bash EMG/preprocess_emg_corpus.sh          # do
#   DATASET=emg2pose LIST_ONLY=1 bash EMG/preprocess_emg_corpus.sh   # list once
#   sbatch --export=ALL,DATASET=emg2pose --array=0-3 \
#          scripts/slurm/cineca_emg_corpus_preprocess.sbatch         # in parallel
#
# The raw location follows scripts/download_emg_pretrain_corpora.sh:
# $EMG_ROOT/<Name>/raw. emg2pose is read straight out of its tar and CEMHSEY's
# GRASP zips member by member, so neither is unpacked.
#
# CEMHSEY IS TWO CORPORA, cemhsey_8x8 (its three 8x8 grids) and cemhsey_5x13
# (its two 5x13 grids), read from the same files: run both.
#
# INSPECT FIRST. It reads N records spread over the corpus and prints their
# rate, channels, duration, peak-to-peak amplitude and what QC would keep --
# the amplitude column is the unit check (sEMG is ~0.1-5 mV peak-to-peak; 100
# or more means the file is in uV and every threshold would be wrong).
#
# ENVIRONMENT VARIABLES:
#   DATASET        emg2pose | emg2qwerty | hyser | cemhsey_8x8 |
#                  cemhsey_5x13 | putemg                           (required)
#   EMG_ROOT       download root  ($SCRATCH/bio/emg)
#   RAW_ROOT       this corpus's raw download       ($EMG_ROOT/<Name>/raw[/x.tar])
#   DATA_ROOT      corpus root                      ($EMG_ROOT/emg_c1_corpus)
#   OUT_DIR        this dataset                     ($DATA_ROOT/$DATASET)
#   TASK           "I/N"; under a SLURM array it is taken from the array
#   JOBS           worker processes                 (1)
#   INSPECT        N: report on N records and exit
#   LIST_ONLY      1: write the record listing and exit
#   MAX_RECORDS    use this many records, spread over the corpus
#   MAX_WINDOWS_PER_RECORD   (none: every 1 s window is kept; 0 = no cap)
#   RECORDS_PER_UNIT         (the reader's: 200 emg2pose files, 4 emg2qwerty
#                             sessions, 64 Hyser files, 16 CEMHSEY trials,
#                             8 putEMG records)
#   MAINS_HZ       override the registry's mains frequency
#   NORMALIZATION  window_shared | window_per_lead | none   (window_shared)
#   VAL_FRACTION / SPLIT_SEED   (0.05 / 42) -- the same for every corpus
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PW_VARS_ONLY=1 source "$(pwd)/scripts/cineca_env.sh"

DATASET="${DATASET:-}"
case "${DATASET}" in
    emg2pose)                   NAME=emg2pose ;;
    emg2qwerty)                 NAME=emg2qwerty ;;
    hyser)                      NAME=Hyser ;;
    cemhsey_8x8|cemhsey_5x13)   NAME=CEMHSEY ;;
    putemg)                     NAME=putEMG ;;
    "") echo "ERROR: set DATASET. One of: emg2pose emg2qwerty hyser" >&2
        echo "       cemhsey_8x8 cemhsey_5x13 putemg" >&2
        exit 1 ;;
    *)  echo "ERROR: unknown DATASET '${DATASET}'." >&2; exit 1 ;;
esac

EMG_ROOT="${EMG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/emg}"
DATA_ROOT="${DATA_ROOT:-${EMG_ROOT}/emg_c1_corpus}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/${DATASET}}"

if [[ -z "${RAW_ROOT:-}" ]]; then
    RAW_ROOT="${EMG_ROOT}/${NAME}/raw"
    # emg2pose's tar is read in place; an unpacked tree works too.
    if [[ "${DATASET}" == "emg2pose" && -f "${RAW_ROOT}/emg2pose_dataset.tar" ]]; then
        if [[ -e "${RAW_ROOT}/emg2pose_dataset.tar.chunks" ]]; then
            echo "ERROR: ${RAW_ROOT}/emg2pose_dataset.tar is still downloading" >&2
            echo "       (its .chunks sidecar is there). Finish it first:" >&2
            echo "         bash scripts/download_emg_pretrain_corpora.sh emg2pose" >&2
            exit 1
        fi
        RAW_ROOT="${RAW_ROOT}/emg2pose_dataset.tar"
    fi
fi
if [[ ! -e "${RAW_ROOT}" ]]; then
    echo "ERROR: ${RAW_ROOT} does not exist." >&2
    echo "       bash scripts/download_emg_pretrain_corpora.sh ${DATASET}" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
_missing="$("${PYTHON}" - <<'PYEOF'
import importlib.util as u
print(" ".join(m for m in ("numpy", "scipy", "h5py", "wfdb", "torch")
               if u.find_spec(m) is None))
PYEOF
)"
if [[ -n "${_missing}" ]]; then
    echo "ERROR: $("${PYTHON}" -c 'import sys;print(sys.prefix)') lacks: ${_missing}" >&2
    echo "       The training venv has everything but wfdb:" >&2
    echo "         source \$HOME/pw/bin/activate && pip install wfdb" >&2
    echo "       (wfdb reads Hyser; torch is imported, not used: the shared" >&2
    echo "        preprocessing module pulls in the channel vocabulary.)" >&2
    exit 1
fi

TASK="${TASK:-}"
if [[ -z "${TASK}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi

ARGS=(--dataset "${DATASET}" --root "${RAW_ROOT}" --out-dir "${OUT_DIR}"
      --jobs "${JOBS:-1}")
[[ -n "${TASK}" ]]                        && ARGS+=(--task "${TASK}")
[[ -n "${INSPECT:-}" ]]                   && ARGS+=(--inspect "${INSPECT}")
[[ "${LIST_ONLY:-0}" == "1" ]]            && ARGS+=(--list-only)
[[ -n "${MAX_RECORDS:-}" ]]               && ARGS+=(--max-records "${MAX_RECORDS}")
[[ -n "${MAX_WINDOWS_PER_RECORD:-}" ]]    && ARGS+=(--max-windows-per-record "${MAX_WINDOWS_PER_RECORD}")
[[ -n "${RECORDS_PER_UNIT:-}" ]]          && ARGS+=(--records-per-unit "${RECORDS_PER_UNIT}")
[[ -n "${MAINS_HZ:-}" ]]                  && ARGS+=(--mains-hz "${MAINS_HZ}")
[[ -n "${NORMALIZATION:-}" ]]             && ARGS+=(--normalization "${NORMALIZATION}")
[[ -n "${VAL_FRACTION:-}" ]]              && ARGS+=(--val-fraction "${VAL_FRACTION}")
[[ -n "${SPLIT_SEED:-}" ]]                && ARGS+=(--split-seed "${SPLIT_SEED}")

echo "============================================================"
echo "  sEMG corpus: ${DATASET}"
echo "  raw   ${RAW_ROOT}"
echo "  out   ${OUT_DIR}"
[[ -n "${TASK}" ]] && echo "  task  ${TASK}"
echo "============================================================"
mkdir -p "${OUT_DIR}" || exit 1
"${PYTHON}" EMG/preprocess_emg_corpus.py "${ARGS[@]}"
_rc=$?
if [[ ${_rc} -eq 0 && -z "${INSPECT:-}" && "${LIST_ONLY:-0}" != "1" ]]; then
    echo ""
    echo "When every corpus you want is done, merge them:"
    echo "  python scripts/build_eeg_c1_manifest.py --modality emg \\"
    echo "      --corpus-root ${DATA_ROOT} --allow-missing --check-shards --jobs 16"
fi
exit ${_rc}
