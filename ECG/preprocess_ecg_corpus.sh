#!/bin/bash
# ============================================================================
# One ECG corpus -> the ECG C1 pretraining corpus.
#
#   DATASET=mimic_iv_ecg INSPECT=40 bash ECG/preprocess_ecg_corpus.sh   # look
#   DATASET=mimic_iv_ecg JOBS=16 bash ECG/preprocess_ecg_corpus.sh      # do
#   DATASET=pulsedb LIST_ONLY=1 bash ECG/preprocess_ecg_corpus.sh       # list once
#   sbatch --export=ALL,DATASET=pulsedb --array=0-7 \
#          scripts/slurm/cineca_ecg_corpus_preprocess.sbatch            # in parallel
#
# The raw location follows scripts/download_ecg_pretrain_corpora.sh:
# $ECG_ROOT/<Name>/raw. MIMIC-IV-ECG and Icentia11k are read straight out of
# their PhysioNet zips when the zip is what is there, so neither is unpacked.
#
# INSPECT FIRST. It reads N records spread over the corpus and prints their
# rate, leads, duration, peak-to-peak amplitude and what QC would keep -- the
# amplitude column is the unit check (a QRS is ~1 mV; 1000 means the file is
# in uV and every threshold would be wrong).
#
# ENVIRONMENT VARIABLES:
#   DATASET        mimic_iv_ecg | code15 | sph | georgia | medalcare_xl |
#                  icentia11k | pulsedb                            (required)
#   ECG_ROOT       download root  ($SCRATCH/bio/ecg)
#   RAW_ROOT       this corpus's raw download       ($ECG_ROOT/<Name>/raw[/x.zip])
#   DATA_ROOT      corpus root                      ($ECG_ROOT/ecg_c1_corpus)
#   OUT_DIR        this dataset                     ($DATA_ROOT/$DATASET)
#   TASK           "I/N"; under a SLURM array it is taken from the array
#   JOBS           worker processes                 (1)
#   INSPECT        N: report on N records and exit
#   LIST_ONLY      1: write the record listing and exit
#   MAX_RECORDS    use this many records, spread over the corpus
#   MAX_WINDOWS_PER_RECORD   (the registry's: 16 for Icentia11k; 0 = no cap)
#   RECORDS_PER_UNIT         (1000; 64 for Icentia11k segments)
#   MAINS_HZ       override the registry's mains frequency
#   NORMALIZATION  window_shared | window_per_lead | none   (window_shared)
#   VAL_FRACTION / SPLIT_SEED   (0.05 / 42) -- the same for every corpus
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PW_VARS_ONLY=1 source "$(pwd)/scripts/cineca_env.sh"

DATASET="${DATASET:-}"
case "${DATASET}" in
    mimic_iv_ecg)       NAME=MIMIC-IV-ECG ;;
    code15)             NAME=CODE-15 ;;
    sph)                NAME=SPH ;;
    georgia)            NAME=Georgia ;;
    medalcare_xl)       NAME=MedalCare-XL ;;
    icentia11k)         NAME=Icentia11k ;;
    pulsedb)            NAME=PulseDB ;;
    "") echo "ERROR: set DATASET. One of: mimic_iv_ecg code15 sph georgia" >&2
        echo "       medalcare_xl icentia11k pulsedb" >&2
        exit 1 ;;
    *)  echo "ERROR: unknown DATASET '${DATASET}'." >&2; exit 1 ;;
esac

ECG_ROOT="${ECG_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/ecg}"
DATA_ROOT="${DATA_ROOT:-${ECG_ROOT}/ecg_c1_corpus}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/${DATASET}}"

if [[ -z "${RAW_ROOT:-}" ]]; then
    RAW_ROOT="${ECG_ROOT}/${NAME}/raw"
    # The two zips that are read in place. Anything else is unpacked by the
    # download script and read as a directory.
    case "${DATASET}" in
        mimic_iv_ecg|icentia11k)
            shopt -s nullglob
            _zips=("${RAW_ROOT}"/*.zip)
            shopt -u nullglob
            if [[ ${#_zips[@]} -eq 1 ]]; then
                RAW_ROOT="${_zips[0]}"
            elif [[ ${#_zips[@]} -gt 1 ]]; then
                echo "ERROR: ${#_zips[@]} zips in ${RAW_ROOT}; set RAW_ROOT to one." >&2
                exit 1
            fi ;;
    esac
fi
if [[ ! -e "${RAW_ROOT}" ]]; then
    echo "ERROR: ${RAW_ROOT} does not exist." >&2
    echo "       bash scripts/download_ecg_pretrain_corpora.sh ${DATASET}" >&2
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
    echo "       (torch is imported, not used: the shared EEG preprocessing" >&2
    echo "        module pulls in the channel vocabulary, which imports it.)" >&2
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
echo "  ECG corpus: ${DATASET}"
echo "  raw   ${RAW_ROOT}"
echo "  out   ${OUT_DIR}"
[[ -n "${TASK}" ]] && echo "  task  ${TASK}"
echo "============================================================"
mkdir -p "${OUT_DIR}" || exit 1
"${PYTHON}" ECG/preprocess_ecg_corpus.py "${ARGS[@]}"
_rc=$?
if [[ ${_rc} -eq 0 && -z "${INSPECT:-}" && "${LIST_ONLY:-0}" != "1" ]]; then
    echo ""
    echo "When every corpus you want is done, merge them:"
    echo "  python scripts/build_eeg_c1_manifest.py --modality ecg \\"
    echo "      --corpus-root ${DATA_ROOT} --allow-missing --check-shards --jobs 16"
fi
exit ${_rc}
