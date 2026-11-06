# shellcheck shell=bash
# =============================================================================
# PhysioWave -- shared environment for CINECA Leonardo.
#
# This file is *sourced*, never executed:
#
#     PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
#     source "${PROJECT_DIR}/scripts/cineca_env.sh"
#
# Fixed layout on Leonardo
# ------------------------
#   code         $HOME/PhysioWave-T           backed up, 50 GB quota
#   venv         $HOME/pw                     built on top of the cineca-ai module
#   checkpoints  $PW_FAST_ROOT/runs          permanent, project-shared, low latency
#   caches       $PW_FAST_ROOT/cache         SSL operators + preprocessed windows
#   manifests    $PW_FAST_ROOT/manifests
#   data         $SCRATCH/bio/{eeg,ecg,emg}   HDF5 training corpora
#
# WARNING: $SCRATCH is *temporary* -- files are purged 40 days after creation.
# The corpora under $SCRATCH/bio are reproducible from the raw archives, but
# nothing else may live there.  Anything you cannot rebuild belongs in $FAST.
#
# Every path below is an environment variable with a default, so a single
# `sbatch --export=ALL,PW_CKPT_ROOT=...` overrides it without editing this file.
#
# Two jobs in one file, and you often want only the first
# ------------------------------------------------------
#   1. export the PW_* paths
#   2. load the cineca-ai module and activate ${PW_VENV}, for training
#
# The data preparation scripts need (1) and nothing from (2) -- no CUDA, no
# torch -- and (2) actively gets in their way, because activating ${PW_VENV}
# replaces whatever environment they were being run from. So:
#
#     PW_VARS_ONLY=1 source scripts/cineca_env.sh
#
# exports the paths and touches neither the modules nor the virtualenv, and can
# be sourced before or after `source $HOME/pwprep/bin/activate` without
# disturbing it. PW_SKIP_MODULES=1 is the narrower version: skip the module
# load, still activate ${PW_VENV}.
# =============================================================================

# --------------------------------------------------------------------------- #
# 0. Where are we?
#
# Off-cluster the module/venv steps are skipped and every path falls back to a
# repository-relative default, so the same launch scripts work on a laptop
# without a second code path.
#
# THE ANSWER MUST NOT DEPEND ON ONE VARIABLE. This used to read $FAST alone. A
# batch job submitted from a shell where $FAST was not set therefore decided it
# was a laptop, skipped the module load and the venv, and ran on whatever
# `python` the submitting shell had on PATH -- which under `--export=ALL` is
# inherited. Four earlier jobs "worked" only because the submitting shell
# happened to have $HOME/pw active; the first one submitted from a shell
# without it got cineca-ai's own interpreter, which has torch and no pywt, and
# died on all sixteen ranks. The modulefile tree is a fact about the machine
# rather than about the shell, so it is the one that decides.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Site-specific values live OUTSIDE the repository, in ~/.physiowave_env
# (or $PW_LOCAL_ENV): the account, the per-user directories, anything that
# names a person or a project allocation. For example:
#
#     export PW_FAST_ROOT="$FAST/<your directory>"     # checkpoints, caches
#     export SBATCH_ACCOUNT=<account> SALLOC_ACCOUNT=<account>
#
# sbatch reads the account from $SBATCH_ACCOUNT of the SUBMITTING shell, so
# that line also belongs in ~/.bashrc; see docs/cineca_setup.md.
# --------------------------------------------------------------------------- #
PW_LOCAL_ENV="${PW_LOCAL_ENV:-${HOME}/.physiowave_env}"
# shellcheck disable=SC1090
[[ -f "${PW_LOCAL_ENV}" ]] && source "${PW_LOCAL_ENV}"

PW_ON_CINECA=0
[[ -n "${FAST:-}${PW_FAST:-}" ]] && PW_ON_CINECA=1
[[ -d /leonardo/prod/opt/modulefiles ]] && PW_ON_CINECA=1
export PW_ON_CINECA

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
[[ -d "${PROJECT_DIR}" ]] || PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Recorded before anything below assigns to PW_VENV. An inherited PW_VENV wins
# over the default, and the only symptom is the wrong interpreter running --
# worth saying out loud when it happens.
_pw_venv_inherited="${PW_VENV:-}"
PW_VENV_DEFAULT="${VENV:-${HOME}/pw}"
# Bound HERE, before either branch, because the training branch reads it inside
# its own module-failure warning. Under `set -u` that made the error handler
# abort with "PW_VENV: unbound variable" and hide the failure it was written to
# explain.
PW_VENV="${PW_VENV:-${PW_VENV_DEFAULT}}"

# --------------------------------------------------------------------------- #
# 1. Modules
#
# profile/deeplrn + cineca-ai provide a PyTorch built against Leonardo's GPUs
# and its NCCL/InfiniBand stack.  Two things must NOT be done here:
#
#   * `export PYTHONPATH=""` -- cineca-ai injects its site-packages through
#     PYTHONPATH.  Clearing it makes `import torch` fail on the compute node.
#   * `module load cuda/... gcc/...` on top -- cineca-ai already pins a matching
#     toolchain; loading another CUDA silently mixes runtimes.
# --------------------------------------------------------------------------- #
PW_CINECA_AI="${PW_CINECA_AI:-cineca-ai/4.3.0}"

# Is a module with this name (any version) already in the environment?
# $LOADEDMODULES is the colon-separated list the module system maintains.
pw_module_is_loaded() {
    # Matches by FAMILY: `cineca-ai/4.3.0` is satisfied by any cineca-ai,
    # because re-loading a different version of the same thing is the version
    # conflict this check exists to avoid.
    local want="${1%%/*}" entry
    local IFS=':'
    for entry in ${LOADEDMODULES:-}; do
        [[ "${entry%%/*}" == "${want}" ]] && return 0
    done
    return 1
}

pw_module_is_loaded_exact() {
    # Matches the FULL name. `profile/deeplrn` and `profile/base` are the same
    # family and not interchangeable: CINECA loads profile/base by default, so
    # the family check above says "profile is loaded" and skips deeplrn -- and
    # then cineca-ai is not on the module path at all. That is what made
    # `cineca-ai/4.3.0 did not load` appear on a system where it exists.
    local entry
    local IFS=':'
    for entry in ${LOADEDMODULES:-}; do
        [[ "${entry}" == "$1" ]] && return 0
    done
    return 1
}

if [[ "${PW_ON_CINECA}" -eq 1 && "${PW_VARS_ONLY:-0}" == "1" ]]; then
    # Paths only. Nothing here loads a module or activates a virtualenv, so
    # whatever environment is active stays active -- which is the point: the
    # preparation scripts run under $HOME/pwprep and want PW_DATA_EEG, not
    # torch.
    #
    # PW_VENV names the TRAINING venv, always -- never "whichever one happens to
    # be active". This branch is sourced from a shell running $HOME/pwprep, and
    # it used to set PW_VENV=${VIRTUAL_ENV} and export it. That export outlives
    # the command: every later `source scripts/cineca_env.sh` in the shell
    # honours it, and so does any `srun` that inherits the environment. So
    # preparing the data under pwprep and then training in the same shell
    # activated pwprep FOR TRAINING, with $HOME/pw sitting in the prompt, and
    # eighteen runs died on "missing pywt" pointing at the wrong venv.
    PW_VENV="${PW_VENV:-${PW_VENV_DEFAULT}}"
elif [[ "${PW_ON_CINECA}" -eq 1 ]]; then
    if ! type -t module >/dev/null 2>&1; then
        # Non-login shells do not always have the module function defined.
        # shellcheck disable=SC1091
        [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    fi

    # Loading a module that is already loaded is not a no-op: it re-resolves the
    # whole dependency chain and aborts on the first version conflict (typically
    # bzip2, pulled in by python/3.11). So load only what is missing, and let
    # PW_SKIP_MODULES=1 bypass the step entirely for an already-set-up shell.
    if [[ "${PW_SKIP_MODULES:-0}" == "1" ]]; then
        :
    elif pw_module_is_loaded "${PW_CINECA_AI}"; then
        :
    else
        # EXACT: profile/base is loaded by default and is not deeplrn.
        pw_module_is_loaded_exact profile/deeplrn || module load profile/deeplrn
        if ! module load "${PW_CINECA_AI}" 2>&1; then
            # module prints its own error and returns non-zero, and the script
            # used to carry straight on. ${PW_VENV} is built with
            # --system-site-packages on top of this module, so without it the
            # venv activates fine and then `import torch` fails somewhere much
            # later -- typically inside torchrun, after the job is allocated.
            echo "" >&2
            echo "WARNING: ${PW_CINECA_AI} did not load." >&2
            echo "  ${PW_VENV} is a venv built --system-site-packages on top of it," >&2
            echo "  so torch comes from the module. Without it the venv still" >&2
            echo "  activates and the failure surfaces later, usually as an" >&2
            echo "  ImportError inside torchrun once the job is running." >&2
            echo "" >&2
            echo "  Find what is actually available and pin it:" >&2
            echo "      module avail cineca-ai" >&2
            echo "      export PW_CINECA_AI=cineca-ai/<version>" >&2
            echo "" >&2
            echo "  Preparation and download need no CUDA, so this is harmless" >&2
            echo "  for those steps. Fix it before training." >&2
            echo "" >&2
        fi
    fi

    # ----------------------------------------------------------------------- #
    # 2. Virtualenv (created with --system-site-packages on top of cineca-ai)
    # ----------------------------------------------------------------------- #
    PW_VENV="${PW_VENV:-${PW_VENV_DEFAULT}}"
    if [[ -n "${_pw_venv_inherited}" && "${_pw_venv_inherited}" != "${PW_VENV_DEFAULT}" ]]; then
        echo "NOTE: PW_VENV was already set in your environment to ${_pw_venv_inherited}," >&2
        echo "      so that is the venv being activated, not ${PW_VENV_DEFAULT}." >&2
        echo "      \`unset PW_VENV\` if you did not mean to pin it." >&2
    fi
    # "Already active" means the venv's python is the one on PATH, not merely
    # that VIRTUAL_ENV names it. `sbatch --export=ALL` from a shell with the
    # venv active carries VIRTUAL_ENV into the job, and the module load above
    # then puts cineca-ai's own python ahead of the venv's on PATH: the
    # variable says pw, the interpreter is cineca-ai's, and trusting the
    # variable skipped the activation that would have fixed it -- on all
    # sixteen ranks of an ECG run.
    if [[ "${VIRTUAL_ENV:-}" == "${PW_VENV}" \
          && "$(command -v python 2>/dev/null)" == "${PW_VENV}/bin/python" ]]; then
        :                                   # already active; re-sourcing stacks PATH
    elif [[ -f "${PW_VENV}/bin/activate" ]]; then
        # Replacing a *different* virtualenv is silent otherwise, and the only
        # visible trace is the prompt changing. That has cost real time twice
        # on the P300 preparation: `source $HOME/pwprep/bin/activate` followed
        # by this script leaves pwprep inactive, and the next command fails on
        # `import mne` with no hint about why. This script has to win -- the
        # training launchers source it precisely to get ${PW_VENV} -- so say
        # what happened rather than change who wins.
        if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${PW_VENV}" ]]; then
            echo "NOTE: replacing the active virtualenv ${VIRTUAL_ENV} with ${PW_VENV}." >&2
            echo "      If you wanted the other one, source it AFTER this script:" >&2
            echo "        source scripts/cineca_env.sh          # first, for PW_DATA_*" >&2
            echo "        source ${VIRTUAL_ENV}/bin/activate    # second" >&2
        fi
        # shellcheck disable=SC1091
        source "${PW_VENV}/bin/activate"
    else
        echo "ERROR: virtualenv not found at ${PW_VENV}." >&2
        echo "       Create it on the login node with the cineca-ai module loaded:" >&2
        echo "         module load profile/deeplrn" >&2
        echo "         module load ${PW_CINECA_AI}" >&2
        echo "         python -m venv ${PW_VENV} --system-site-packages" >&2
        return 1 2>/dev/null || exit 1
    fi

    # The module and the venv can both look fine and still not produce a torch:
    # the venv is --system-site-packages on top of cineca-ai, so a module that
    # failed to load leaves an ImportError waiting inside torchrun, after the
    # allocation, on every rank at once. Two seconds here against a job that
    # dies twenty minutes in. PW_SKIP_TORCH_CHECK=1 bypasses it.
    if [[ "${PW_SKIP_TORCH_CHECK:-0}" != "1" ]]; then
        if ! python -c "import torch" >/dev/null 2>&1; then
            echo "" >&2
            echo "ERROR: torch does not import from ${PW_VENV}." >&2
            echo "  python: $(command -v python)" >&2
            echo "  loaded: ${LOADEDMODULES:-<none>}" >&2
            echo "" >&2
            echo "  The venv is --system-site-packages on top of ${PW_CINECA_AI}," >&2
            echo "  so torch comes from the module. Load it in this order:" >&2
            echo "      module load profile/deeplrn" >&2
            echo "      module load ${PW_CINECA_AI}" >&2
            echo "      source ${PW_VENV}/bin/activate" >&2
            echo "" >&2
            echo "  profile/deeplrn is not optional and is not implied by the" >&2
            echo "  profile/base loaded by default: without it cineca-ai is not" >&2
            echo "  on the module path at all." >&2
            echo "" >&2
            return 1 2>/dev/null || exit 1
        fi
    fi
else
    PW_VENV="${PW_VENV:-${VIRTUAL_ENV:-<none>}}"
fi

# --------------------------------------------------------------------------- #
# 3. Storage roots
#
# $SCRATCH is the documented Leonardo variable in some profiles and
# $CINECA_SCRATCH in others; accept either rather than silently falling back to
# $HOME, whose 50 GB quota a single pretraining run would blow through.
# --------------------------------------------------------------------------- #
if [[ "${PW_ON_CINECA}" -eq 1 ]]; then
    PW_SCRATCH="${PW_SCRATCH:-${SCRATCH:-${CINECA_SCRATCH:-}}}"
    PW_FAST="${PW_FAST:-${FAST:-}}"
    if [[ -z "${PW_SCRATCH}" ]]; then
        echo "ERROR: neither \$SCRATCH nor \$CINECA_SCRATCH is set. Set PW_SCRATCH=/path." >&2
        return 1 2>/dev/null || exit 1
    fi
    PW_FAST_ROOT="${PW_FAST_ROOT:-${PW_FAST}/${PW_FAST_USER:-$USER}}"
    PW_DATA_ROOT="${PW_DATA_ROOT:-${PW_SCRATCH}/bio}"
else
    # Off-cluster: reuse the repository layout that run_tpami.sh and .gitignore
    # already assume, rather than inventing a parallel one.
    PW_SCRATCH="${PW_SCRATCH:-${PROJECT_DIR}}"
    PW_FAST="${PW_FAST:-${PROJECT_DIR}}"
    PW_FAST_ROOT="${PW_FAST_ROOT:-${PROJECT_DIR}}"
    PW_DATA_ROOT="${PW_DATA_ROOT:-${PROJECT_DIR}/data}"
    PW_CKPT_ROOT="${PW_CKPT_ROOT:-${PROJECT_DIR}/outputs}"
fi

# Checkpoints, caches and manifests: permanent, project-shared, fast I/O.
PW_CKPT_ROOT="${PW_CKPT_ROOT:-${PW_FAST_ROOT}/runs}"
PW_CACHE_ROOT="${PW_CACHE_ROOT:-${PW_FAST_ROOT}/cache}"
PW_SSL_CACHE="${PW_SSL_CACHE:-${PW_CACHE_ROOT}/ssl}"
PW_PREP_CACHE="${PW_PREP_CACHE:-${PW_CACHE_ROOT}/preprocessed}"
PW_MANIFEST_DIR="${PW_MANIFEST_DIR:-${PW_FAST_ROOT}/manifests}"

# Training corpora: one directory per modality, each holding per-dataset
# subdirectories of HDF5 windows (e.g. $PW_DATA_EEG/tueg/*.h5).
PW_DATA_EEG="${PW_DATA_EEG:-${PW_DATA_ROOT}/eeg}"
PW_DATA_ECG="${PW_DATA_ECG:-${PW_DATA_ROOT}/ecg}"
PW_DATA_EMG="${PW_DATA_EMG:-${PW_DATA_ROOT}/emg}"

mkdir -p "${PW_CKPT_ROOT}" "${PW_SSL_CACHE}" "${PW_PREP_CACHE}" "${PW_MANIFEST_DIR}" \
         "${PW_DATA_EEG}" "${PW_DATA_ECG}" "${PW_DATA_EMG}"

export PROJECT_DIR PW_VENV PW_SCRATCH PW_FAST PW_FAST_ROOT
export PW_CKPT_ROOT PW_CACHE_ROOT PW_SSL_CACHE PW_PREP_CACHE PW_MANIFEST_DIR
export PW_DATA_ROOT PW_DATA_EEG PW_DATA_ECG PW_DATA_EMG

# --------------------------------------------------------------------------- #
# 4. Launcher
#
# Always `python -m torch.distributed.run`, never the `torchrun` console script.
#
# torch.distributed.run spawns its workers with sys.executable. The `torchrun`
# on PATH belongs to the cineca-ai environment, so its sys.executable is the
# *module's* python -- and the workers then cannot see anything pip installed
# into the venv (pywt, for one), even though the parent shell has the venv
# active. Invoking the module form makes the venv's python the parent, and the
# workers inherit its site-packages.
#
# On the login node `torchrun` is not on PATH at all; on a compute node it is.
# Preferring it would therefore fail only on the machine that has GPUs.
# --------------------------------------------------------------------------- #
PW_TORCHRUN=("${PYTHON:-python}" -m torch.distributed.run)

# Fail before the allocation is spent, not inside a worker: every module the
# training entry points import at top level, checked in the venv's interpreter.
# A path built from a variable that was empty. `$PW_CKPT_ROOT/run` becomes
# `/run` when PW_CKPT_ROOT is unset, and `sbatch --export=ALL` carries the empty
# value from the submitting shell straight into the job -- where the only
# symptom is rank 0 dying on a permission error after the allocation is granted
# and the other fifteen ranks reporting that they lost the rendezvous store.
#
# A directory whose parent is `/` is never a run directory on this system, and
# neither is one under a parent that does not exist. Both are the same mistake
# and both are cheap to see here.
pw_check_run_path() {
    local name="$1" path="$2" parent
    if [[ -z "${path}" ]]; then
        echo "ERROR: ${name} is empty." >&2
        return 1
    fi
    if [[ "${path}" != /* ]]; then
        echo "ERROR: ${name}=${path} is not an absolute path." >&2
        return 1
    fi
    parent="$(dirname "${path}")"
    if [[ "${parent}" == "/" ]]; then
        echo "ERROR: ${name}=${path} sits directly under /." >&2
        echo "" >&2
        echo "  That is what an unset variable expands to: \$PW_CKPT_ROOT/run" >&2
        echo "  becomes /run when PW_CKPT_ROOT is empty, and --export=ALL" >&2
        echo "  carries the empty value from your shell into the job." >&2
        echo "" >&2
        echo "  PW_CKPT_ROOT and PW_DATA_EEG come from scripts/cineca_env.sh," >&2
        echo "  not from your login profile. Before submitting:" >&2
        echo "      source scripts/cineca_env.sh" >&2
        echo "      echo \"\$PW_CKPT_ROOT\"      # must not be empty" >&2
        return 1
    fi
    if [[ ! -d "${parent}" ]]; then
        echo "ERROR: ${name}=${path} -- its parent ${parent} does not exist." >&2
        echo "       Nothing will create it, and rank 0 would die on it after" >&2
        echo "       the allocation is granted." >&2
        return 1
    fi
    return 0
}

# A training launcher runs on the training venv or it does not run. Everything
# that decides which interpreter is active is inherited from the submitting
# shell, so this is checked rather than assumed.
pw_require_training_venv() {
    local want have
    # Off-cluster there is no ${HOME}/pw and none is wanted: the laptop path
    # runs on whatever interpreter is active, which is the whole reason the
    # launch scripts have no second code path.
    [[ "${PW_ON_CINECA:-0}" -eq 1 ]] || return 0
    want="${PW_VENV_DEFAULT:-${HOME}/pw}"
    have="$("${PYTHON:-python}" -c 'import sys;print(sys.prefix)' 2>/dev/null)"
    [[ "${have}" == "${want}" ]] && return 0

    echo "ERROR: training needs ${want} and this is ${have:-<no python>}." >&2
    echo "" >&2
    pw_print_env_state
    echo "" >&2
    echo "  Every one of these is inherited by \`sbatch --export=ALL\` from the" >&2
    echo "  shell you submitted from. Submit from a clean shell, or fix the" >&2
    echo "  variable the state above names:" >&2
    echo "      unset PW_VENV PW_VARS_ONLY VIRTUAL_ENV" >&2
    echo "      sbatch ..." >&2
    return 1
}

pw_print_env_state() {
    echo "  python           $(command -v python 2>/dev/null || echo '<none>')" >&2
    echo "  sys.prefix       $("${PYTHON:-python}" -c 'import sys;print(sys.prefix)' 2>/dev/null || echo '<no python>')" >&2
    echo "  PW_ON_CINECA     ${PW_ON_CINECA:-<unset>}   (0 skips the module and the venv)" >&2
    echo "  PW_VARS_ONLY     ${PW_VARS_ONLY:-<unset>}   (1 skips them too)" >&2
    echo "  FAST             ${FAST:-<unset>}" >&2
    echo "  PW_VENV          ${PW_VENV:-<unset>}" >&2
    echo "  VIRTUAL_ENV      ${VIRTUAL_ENV:-<unset>}" >&2
    echo "  SLURM_JOB_ID     ${SLURM_JOB_ID:-<unset>}" >&2
    echo "  cineca-ai loaded $(pw_module_is_loaded_exact "${PW_CINECA_AI:-cineca-ai}" && echo yes || echo NO)" >&2
}

pw_require_python_deps() {
    local missing dists prefix
    # Two lines: the import names that are absent, then the distributions that
    # provide them. Three of the eight differ, and the old message printed the
    # import names into a `pip install` line that could not succeed.
    missing="$("${PYTHON:-python}" - <<'PYEOF'
import importlib.util as u
need = ["torch", "numpy", "scipy", "h5py", "pywt", "sklearn", "yaml", "tqdm"]
print(" ".join(m for m in need if u.find_spec(m) is None))
PYEOF
)"
    [[ -n "${missing}" ]] || return 0
    dists="$(PW_MISSING="${missing}" "${PYTHON:-python}" - <<'PYEOF'
import os
dist = {"pywt": "PyWavelets", "sklearn": "scikit-learn", "yaml": "PyYAML"}
print(" ".join(dist.get(m, m) for m in os.environ["PW_MISSING"].split()))
PYEOF
)"
    prefix="$("${PYTHON:-python}" -c 'import sys;print(sys.prefix)')"
    echo "ERROR: missing Python packages in ${prefix}: ${missing}" >&2
    if [[ "${prefix}" != "${PW_VENV_DEFAULT:-}" ]]; then
        # Far more often the wrong interpreter than a genuinely incomplete one.
        # Print the state rather than guess at which of the four ways it
        # happened: the guess used to name PW_VENV, and the last time this
        # fired it was PW_ON_CINECA.
        echo "" >&2
        echo "  That is not ${PW_VENV_DEFAULT:-\$HOME/pw}, the venv training uses." >&2
        pw_print_env_state
        echo "" >&2
        echo "  All of these are inherited by \`sbatch --export=ALL\`." >&2
        echo "" >&2
        echo "  If the interpreter is right after all, install into it:" >&2
    fi
    echo "       pip install --no-cache-dir --no-deps ${dists}" >&2
    return 1
}

# Refuse to start training on a login node.  finetune.py and pretrain_main both
# initialise NCCL, which needs a GPU; without one the job dies deep inside
# init_process_group instead of here, and CINECA frowns on login-node compute.
pw_require_gpu() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        return 0                      # inside a Slurm allocation (sbatch or srun)
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        return 0                      # a GPU is actually visible
    fi
    cat >&2 <<'MSG'
ERROR: no GPU and no Slurm allocation -- this looks like a login node.

  Training initialises NCCL, which requires a GPU. Get an allocation first:

    # interactive, for a smoke test
    srun --nodes=1 --gpus=1 --ntasks-per-node=1 --cpus-per-task=8 \
         -A <account> -p boost_usr_prod --time=0:30:00 --pty /bin/bash

    # or submit a batch job
    sbatch scripts/slurm/cineca_pretrain.sbatch

  Set PW_ALLOW_NO_GPU=1 to override (CPU-only debugging).
MSG
    return 1
}

# --------------------------------------------------------------------------- #
# 5. Helpers
# --------------------------------------------------------------------------- #

# Catch an OUTPUT_DIR that came from a variable the caller's shell had not set.
#
# `OUTPUT_DIR=$PW_CKPT_ROOT/run bash EMG/finetune_emg.sh` expands PW_CKPT_ROOT in
# the *caller's* shell, which happens before this file is sourced by the launcher
# -- so a shell that has not sourced it produces `/run`, and the failure is a
# permission error from mkdir at the filesystem root, several lines away from the
# cause. Fail here instead, with the fix in the message.
pw_check_output_dir() {
    local dir="${1:-}"
    [[ -n "${dir}" ]] || return 0
    local parent
    parent="$(dirname "${dir}")"
    if [[ "${parent}" == "/" ]]; then
        cat >&2 <<MSG
ERROR: OUTPUT_DIR is '${dir}' -- directly under the filesystem root.

  This is almost always an unset variable: OUTPUT_DIR=\$PW_CKPT_ROOT/name is
  expanded by your shell before this script runs, and \$PW_CKPT_ROOT is empty
  unless you have sourced the environment yourself:

    source scripts/cineca_env.sh

  Then re-run. Or leave OUTPUT_DIR unset and let the launcher choose it.
MSG
        return 1
    fi
    if [[ ! -d "${parent}" ]]; then
        echo "ERROR: OUTPUT_DIR '${dir}' -- parent '${parent}' does not exist" >&2
        return 1
    fi
    return 0
}

# Directory holding the corpora for one modality.  `emg` and `semg` are the same
# data: the registry calls it semg, the on-disk layout the user asked for is emg.
pw_data_dir() {
    case "$1" in
        eeg)        echo "${PW_DATA_EEG}" ;;
        ecg)        echo "${PW_DATA_ECG}" ;;
        emg|semg)   echo "${PW_DATA_EMG}" ;;
        *)          echo "ERROR: unknown modality '$1'" >&2; return 1 ;;
    esac
}

# Print `data.roots.<id>=<dir>/<id>` for each dataset id given, so a caller can
# splice them straight into `--set`.
pw_roots_for() {
    local modality="$1"; shift
    local dir; dir="$(pw_data_dir "${modality}")" || return 1
    local id
    for id in "$@"; do
        echo "data.roots.${id}=${dir}/${id}"
    done
}

pw_print_layout() {
    cat <<LAYOUT
  code        ${PROJECT_DIR}
  venv        ${PW_VENV}
  checkpoints ${PW_CKPT_ROOT}
  caches      ${PW_SSL_CACHE} , ${PW_PREP_CACHE}
  manifests   ${PW_MANIFEST_DIR}
  data        ${PW_DATA_EEG} , ${PW_DATA_ECG} , ${PW_DATA_EMG}
LAYOUT
}

# Echo the erpbci decode cache that actually exists, or nothing.
#
# The cache directory is named for what was DECODED, not for what a run wants:
# the current preparation writes every EEG channel the EDF has into c64 and the
# split stage takes its subset by name, so one decode serves --channels 58, 62
# and 64 alike. c58 is the legacy layout, written before that, and it holds
# only EEGPT's set -- so a 62-channel split cannot be built from it.
#
# Callers used to spell this "cache/c${IN_CHANNELS}", which agreed with the
# only cache that existed at the time and stopped agreeing the moment the
# montage became a choice: IN_CHANNELS=62 asked for a c62 that is never
# written. The search order here is find_subject_cache's in
# EEG/physio_p300_finetune.py, and the two must stay the same.
pw_p300_cache_dir() {
    local edf_dir="${1:?usage: pw_p300_cache_dir <edf-dir>}"
    local d
    for d in c64 c58; do
        if [[ -d "${edf_dir}/cache/${d}" ]]; then
            echo "${edf_dir}/cache/${d}"
            return 0
        fi
    done
    return 1
}
