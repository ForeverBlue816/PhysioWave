#!/bin/bash
# ============================================================================
# Upload a prepared release to Hugging Face, from a venv of its own.
#
#   bash scripts/upload_hf_release.sh ~/hf_release ForeverBlue/EEG
#   PRIVATE=1 bash scripts/upload_hf_release.sh ~/hf_release_ecg ForeverBlue/ECG
#
# PRIVATE=1 creates the repo private and refuses to upload into one that
# already exists and is public: `hf upload` alone creates a missing repo
# PUBLIC, so a private release has to exist, private, before the first file.
#
# It asks before publishing, and refuses outright when stdin is not a terminal
# unless CONFIRM=1 is set. Every check before that point is safe to run.
#
# `pip install -U huggingface_hub` into the TRAINING venv is what this exists to
# avoid, and it fails twice over.
#
#   * $HOME/pw is built --system-site-packages on top of the cineca-ai module,
#     which ships transformers and tokenizers pinned to huggingface_hub<1.0.
#     Upgrading it to 1.x satisfies nothing and breaks those two, and pip says
#     so in a warning that scrolls past under a "Successfully installed" line.
#   * the module also puts its own site-packages on PYTHONPATH, which sys.path
#     honours BEFORE the venv. So `hf` -- the 1.x console script -- starts, and
#     `import huggingface_hub` lands on the module's 0.x copy, which has no
#     .cli submodule. ModuleNotFoundError, from a package that installed fine.
#
# A venv with no system site-packages and an empty PYTHONPATH has neither
# problem, and uploading needs nothing from the training environment: no torch,
# no cineca-ai, just requests over HTTPS.
#
# The training venv is left alone. If it was already upgraded, this prints how
# to put it back.
# ============================================================================

set -uo pipefail

DIR="${1:-${HOME}/hf_release}"
REPO="${2:-ForeverBlue/EEG}"
UPVENV="${HF_UPLOAD_VENV:-${HOME}/hfup}"

DIR="$(cd "${DIR}" 2>/dev/null && pwd)" || {
    echo "ERROR: no such directory: ${1:-${HOME}/hf_release}" >&2
    echo "       Run scripts/prepare_hf_release.py first." >&2
    exit 1
}
if [[ ! -f "${DIR}/README.md" ]]; then
    echo "ERROR: ${DIR} has no README.md -- that is not a prepared release." >&2
    exit 1
fi

PRIVATE="${PRIVATE:-0}"
echo "  release  ${DIR}"
echo "  repo     ${REPO}$([[ "${PRIVATE}" == "1" ]] && echo "   (private)")"
du -sh "${DIR}" | awk '{print "  size     " $1}'
echo ""

# --- the training venv, if it was already upgraded ------------------------- #
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${UPVENV}" ]]; then
    hv="$(python -c 'import huggingface_hub as h; print(h.__version__)' \
          2>/dev/null)"
    if [[ "${hv}" == 1.* ]]; then
        echo "  NOTE: ${VIRTUAL_ENV} has huggingface_hub ${hv}, which the"
        echo "        cineca-ai transformers/tokenizers cannot use. Nothing"
        echo "        here needs it. To put that venv back:"
        echo "            pip uninstall -y huggingface_hub hf-xet"
        echo "        (the module's own copy resurfaces underneath)"
        echo ""
    fi
fi

# --- a venv of its own ------------------------------------------------------ #
# PYTHONPATH is unset for every command below, not just for the install: the
# module's site-packages on it would shadow this venv exactly as it shadows the
# training one, and the failure would look identical.
if [[ ! -x "${UPVENV}/bin/python" ]]; then
    echo "  creating ${UPVENV} (no system site-packages)"
    env -u PYTHONPATH python3 -m venv "${UPVENV}" || {
        echo "ERROR: could not create ${UPVENV}" >&2; exit 1; }
fi

echo "  installing huggingface_hub into ${UPVENV}"
env -u PYTHONPATH "${UPVENV}/bin/pip" install -q -U pip huggingface_hub || {
    echo "ERROR: install failed. Is there outbound HTTPS from this node?" >&2
    exit 1; }

ver="$(env -u PYTHONPATH "${UPVENV}/bin/python" -c \
       'import huggingface_hub as h; print(h.__version__)')"
echo "  huggingface_hub ${ver} at $(env -u PYTHONPATH "${UPVENV}/bin/python" \
      -c 'import huggingface_hub as h; print(h.__file__)')"

# The console script is what broke; check it resolves here before relying on it.
if ! env -u PYTHONPATH "${UPVENV}/bin/hf" --help >/dev/null 2>&1; then
    echo "ERROR: ${UPVENV}/bin/hf still does not run. Upload with the Python" >&2
    echo "       API instead:" >&2
    echo "         ${UPVENV}/bin/python -c \"from huggingface_hub import HfApi;" >&2
    echo "           HfApi().upload_folder(folder_path='${DIR}'," >&2
    echo "           repo_id='${REPO}', repo_type='model')\"" >&2
    exit 1
fi

# --- authenticate ----------------------------------------------------------- #
# A token in the environment skips the prompt, which is what a batch job needs.
# Never echoed, and never written into the release.
if ! env -u PYTHONPATH "${UPVENV}/bin/hf" auth whoami >/dev/null 2>&1; then
    echo ""
    echo "  not logged in. A WRITE token from"
    echo "  https://huggingface.co/settings/tokens, then:"
    echo "      env -u PYTHONPATH ${UPVENV}/bin/hf auth login"
    echo "  or, without a prompt:"
    echo "      HF_TOKEN=hf_xxx bash $0 ${DIR} ${REPO}"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        exit 1
    fi
fi
# The account name, from the API. `hf auth whoami` changed its first line to
# "✓ Logged in" in 1.x, and that line is not who you are.
who="$(env -u PYTHONPATH "${UPVENV}/bin/python" -c \
       'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])' 2>/dev/null)"
[[ -n "${who}" ]] || who="$(env -u PYTHONPATH "${UPVENV}/bin/hf" auth whoami 2>/dev/null | head -1)"
echo "  authenticated as ${who}"

# --- confirm, because the next line publishes ------------------------------- #
# Not a formality. Running this script to see whether the venv fix worked is
# exactly what someone does, and with a token already in the environment every
# check above passes and the upload happens as a side effect of the test. A
# publish should require saying so, not merely reaching this line.
if [[ "${CONFIRM:-}" != "1" ]]; then
    echo ""
    echo "  About to publish ${DIR}"
    echo "  to https://huggingface.co/${REPO} as ${who}."
    if [[ "${PRIVATE}" == "1" ]]; then
        echo "  The repo is created PRIVATE (only you and your orgs see it)."
        echo "  Files stay in its git history even if deleted afterwards."
    else
        echo "  This is public and the files stay in the repo's git history"
        echo "  even if deleted afterwards. PRIVATE=1 for a private repo."
    fi
    if [[ -t 0 ]]; then
        # The exact string, quoted. "Type the repo id" is ambiguous the first
        # time someone meets it -- the name alone, or owner/name? -- and a
        # confirmation you have to guess at is a confirmation people paste
        # past without reading.
        read -r -p "  Type '${REPO}' to publish (anything else cancels): " answer
        if [[ "${answer}" != "${REPO}" ]]; then
            echo "  not published (you typed '${answer}')."
            exit 1
        fi
    else
        echo "  Not a terminal, so nothing was published."
        echo "  Re-run with CONFIRM=1 to publish:"
        echo "      CONFIRM=1 bash $0 ${DIR} ${REPO}"
        exit 1
    fi
fi

# --- a private repo exists, private, before any file goes up ----------------- #
if [[ "${PRIVATE}" == "1" ]]; then
    env -u PYTHONPATH "${UPVENV}/bin/python" - "${REPO}" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
if not api.repo_exists(repo, repo_type="model"):
    api.create_repo(repo, repo_type="model", private=True)
    print(f"  created {repo} (private)")
info = api.repo_info(repo, repo_type="model")
if not info.private:
    print(f"ERROR: {repo} already exists and is PUBLIC. Nothing was uploaded.\n"
          f"       Make it private on its settings page, or choose another name.",
          file=sys.stderr)
    sys.exit(1)
print(f"  {repo} is private")
PY
fi

echo ""
echo "  uploading ${DIR} -> ${REPO}"
env -u PYTHONPATH "${UPVENV}/bin/hf" upload "${REPO}" "${DIR}" . \
    --repo-type=model
rc=$?
if [[ ${rc} -eq 0 && "${PRIVATE}" == "1" ]]; then
    # Checked again after the upload, so a repo flipped public while the
    # files went up is reported rather than assumed.
    env -u PYTHONPATH "${UPVENV}/bin/python" -c "
import sys
from huggingface_hub import HfApi
p = HfApi().repo_info('${REPO}', repo_type='model').private
print('  visibility after upload:', 'private' if p else 'PUBLIC')
sys.exit(0 if p else 1)" || rc=1
fi
if [[ ${rc} -eq 0 ]]; then
    echo ""
    echo "  https://huggingface.co/${REPO}"
fi
exit ${rc}
