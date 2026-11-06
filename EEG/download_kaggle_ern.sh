#!/bin/bash
# ============================================================================
# Fetch KaggleERN (BCI Challenge @ NER 2015) and lay it out for the converter.
#
#   bash EEG/download_kaggle_ern.sh $PW_DATA_EEG/kaggle_ern
#
# TWO THINGS ONLY YOU CAN DO, once, before this works:
#
#   1. Accept the competition rules while logged in to Kaggle:
#        https://www.kaggle.com/c/inria-bci-challenge/rules
#      Without it the API answers 403, which reads like a bad token.
#   2. Give the API a token: https://www.kaggle.com/settings/api ->
#      "Generate New Token", then
#          echo '<token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
#      (or export KAGGLE_API_TOKEN=<token>). Current CLI versions do not accept
#      the old kaggle.json username/key file.
#
# The kaggle CLI goes into a venv of its own, run with PYTHONPATH unset, for
# the reason scripts/upload_hf_release.sh does: the cineca-ai module puts its
# site-packages on PYTHONPATH ahead of any venv, so a package installed into
# the training venv can import the module's copy of its dependencies instead.
#
# Result, the layout EEG/kaggle_ern_finetune.py and EEGPT both expect:
#
#   <dest>/TrainLabels.csv  true_labels.csv  SampleSubmission.csv
#   <dest>/train/Data_S02_Sess01.csv ...     16 subjects x 5 sessions
#   <dest>/test/Data_S01_Sess01.csv  ...     10 subjects x 5 sessions
# ============================================================================

set -uo pipefail

DEST="${1:-${PW_DATA_EEG:-.}/kaggle_ern}"
COMP="inria-bci-challenge"
VENV="${KAGGLE_VENV:-${HOME}/kaggleenv}"
KAGGLE_BIN="${KAGGLE_BIN:-${VENV}/bin/kaggle}"

mkdir -p "${DEST}" || exit 1
DEST="$(cd "${DEST}" && pwd)"
echo "KaggleERN -> ${DEST}"

# --- the CLI, in its own venv ------------------------------------------------ #
if [[ "${KAGGLE_BIN}" == "${VENV}/bin/kaggle" ]]; then
    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "  creating ${VENV} (no system site-packages)"
        env -u PYTHONPATH python3 -m venv "${VENV}" || exit 1
    fi
    env -u PYTHONPATH "${VENV}/bin/pip" install -q -U pip kaggle || {
        echo "ERROR: could not install the kaggle CLI (outbound HTTPS?)" >&2; exit 1; }
fi

# The CLI changed its credentials. Current versions authenticate with a single
# API token -- KAGGLE_API_TOKEN, or ~/.kaggle/access_token -- or an OAuth login
# (`kaggle auth login`). The legacy kaggle.json (username + key) is still read
# by some versions and not by others, so its presence proves nothing; it is
# accepted here only so the CLI gets to say for itself whether it works.
if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -s "${HOME}/.kaggle/access_token" \
      && -z "${KAGGLE_USERNAME:-}" && ! -f "${HOME}/.kaggle/kaggle.json" \
      && ! -f "${HOME}/.config/kaggle/kaggle.json" ]]; then
    cat >&2 <<EOF
ERROR: no Kaggle credentials.
  1. https://www.kaggle.com/settings/api -> "Generate New Token", copy it
  2. mkdir -p ~/.kaggle && echo '<token>' > ~/.kaggle/access_token \\
       && chmod 600 ~/.kaggle/access_token
  3. accept the rules at https://www.kaggle.com/c/${COMP}/rules
EOF
    exit 1
fi

# --- download ---------------------------------------------------------------- #
ARCHIVE="${DEST}/${COMP}.zip"
if [[ ! -s "${ARCHIVE}" ]]; then
    echo "  downloading ${COMP} (a few GB)"
    log="$(mktemp)"
    env -u PYTHONPATH "${KAGGLE_BIN}" competitions download -c "${COMP}" -p "${DEST}" \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    # Diagnose from what the CLI actually said. The first version of this
    # printed "403: accept the rules" for every failure, including the CLI
    # saying outright that it had found no credentials it could use.
    if [[ ${rc} -ne 0 ]] || grep -qiE "authentication required|401|403" "${log}"; then
        if grep -qiE "authentication required|401|unauthori" "${log}"; then
            cat >&2 <<EOF
ERROR: the CLI did not accept any credentials. Current versions want an API
       TOKEN, not the old kaggle.json:
         1. https://www.kaggle.com/settings/api -> "Generate New Token"
         2. echo '<token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
            (or: export KAGGLE_API_TOKEN=<token>)
EOF
        elif grep -qiE "403|forbidden" "${log}"; then
            cat >&2 <<EOF
ERROR: 403 -- the token works, but this account has not accepted the
       competition rules: https://www.kaggle.com/c/${COMP}/rules
EOF
        else
            echo "ERROR: the download failed; the CLI's output is above." >&2
        fi
        rm -f "${log}"
        exit 1
    fi
    rm -f "${log}"
fi

# --- unpack ------------------------------------------------------------------ #
# The archive holds train.zip and test.zip, and whether those put their CSVs in
# a train/ folder or at the top level is not something to rely on. So every
# Data_S*.csv is found wherever it lands and moved to the folder its archive
# names -- the files are named identically in both, and which one a file came
# from is the only thing that says whether its labels are in TrainLabels.csv.
python3 - "${ARCHIVE}" "${DEST}" <<'PY'
import glob, os, shutil, sys, tempfile, zipfile

archive, dest = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp(dir=dest)
with zipfile.ZipFile(archive) as z:
    z.extractall(work)
for inner in glob.glob(os.path.join(work, "**", "*.zip"), recursive=True):
    with zipfile.ZipFile(inner) as z:
        z.extractall(inner[:-4])
    os.remove(inner)
for name in ("TrainLabels.csv", "true_labels.csv", "SampleSubmission.csv",
             "ChannelsLocation.csv"):
    hits = glob.glob(os.path.join(work, "**", name), recursive=True)
    if hits:
        shutil.move(hits[0], os.path.join(dest, name))
for split in ("train", "test"):
    os.makedirs(os.path.join(dest, split), exist_ok=True)
    roots = [d for d in glob.glob(os.path.join(work, "**", split), recursive=True)
             if os.path.isdir(d)]
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "Data_S*.csv"), recursive=True):
            shutil.move(f, os.path.join(dest, split, os.path.basename(f)))
shutil.rmtree(work)
PY
[[ $? -eq 0 ]] || { echo "ERROR: unpacking ${ARCHIVE} failed" >&2; exit 1; }

# --- verify ------------------------------------------------------------------ #
n_train=$(ls "${DEST}"/train/Data_S*.csv 2>/dev/null | wc -l | tr -d ' ')
n_test=$(ls "${DEST}"/test/Data_S*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  train sessions ${n_train} (expect 80)   test sessions ${n_test} (expect 50)"
rc=0
if [[ -f "${DEST}/TrainLabels.csv" ]]; then
    echo "  TrainLabels.csv  $(($(wc -l < "${DEST}/TrainLabels.csv") - 1)) rows"
else
    echo "  MISSING TrainLabels.csv" >&2
    rc=1
fi
if [[ -f "${DEST}/true_labels.csv" ]]; then
    echo "  true_labels.csv  $(($(wc -l < "${DEST}/true_labels.csv") - 1)) rows" \
         "-- the 10 Kaggle test subjects can be scored, as EEGPT does"
else
    # Not a failure: the Kaggle download does not contain it. The converter
    # then builds train/val/test from the 16 labelled subjects.
    cat <<EOF
  NOTE: no true_labels.csv (the Kaggle download does not include it).
        EEG/kaggle_ern_finetune.py will build train/val/test from the 16
        labelled subjects instead -- see its docstring.
EOF
fi
[[ "${n_train}" -gt 0 && "${n_test}" -gt 0 ]] || rc=1
[[ ${rc} -eq 0 ]] && rm -f "${ARCHIVE}" && echo "  done; archive removed"
exit ${rc}
