#!/bin/bash
# ============================================================================
# Siena Scalp EEG Database (PhysioNet siena-scalp-eeg 1.0.0, ODC-By 1.0), for
# continued pretraining of the EEG C1 encoder (configs/pretrain/eeg_c1_siena.yaml).
# Login node; from PhysioNet's anonymous open S3 bucket (its own endpoints are
# ~100x slower), size-checked and resumable: rerun to fetch what is missing.
#
#   bash scripts/download_siena.sh            # ~22 GB -> $PW_DATA_EEG/Siena/raw
#
# Layout written: <dest>/PNxx/PNxx-k.edf (+ Seizures-list-PNxx.txt, the
# release's RECORDS / LICENSE / SHA256SUMS). The EDFs are what the preprocessing
# reads (EEG/preprocess_pretrain_corpus.py --dataset siena).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PW_DATA_EEG:-}" ]]; then
    PW_DATA_EEG="$(set +u; source "${HERE}/cineca_env.sh" >/dev/null 2>&1; echo "${PW_DATA_EEG}")"
fi
[[ -n "${PW_DATA_EEG}" ]] || { echo "ERROR: PW_DATA_EEG is not set" >&2; exit 1; }
DEST="${DEST:-${PW_DATA_EEG}/Siena/raw}"
PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || PYTHON=python
mkdir -p "${DEST}"
echo "==> s3://physionet-open/siena-scalp-eeg/1.0.0/ -> ${DEST}"
"${PYTHON}" "${HERE}/fetch_s3_open.py" --bucket physionet-open \
    --prefix siena-scalp-eeg/1.0.0/ --dest "${DEST}" \
    --include '(\.edf|\.txt|RECORDS|LICENSE\.txt|SHA256SUMS\.txt)$' --jobs "${JOBS:-16}"
n=$(find "${DEST}" -name '*.edf' | wc -l | tr -d ' ')
echo "    ${n} EDF recordings over $(find "${DEST}" -maxdepth 1 -type d -name 'PN*' | wc -l | tr -d ' ') patients in ${DEST}"
