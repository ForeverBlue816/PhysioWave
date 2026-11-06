#!/bin/bash
# ============================================================================
# sEMG C1 multi-route pretraining.
#
#   bash EMG/pretrain_emg_c1_moe.sh
#
# The EEG C1 model and loop on sEMG: a wavelet frontend per route (a 16-channel
# wristband, a 24-channel forearm array, one 8x8 HD grid; all 2000 Hz), one
# shared RoPE Transformer (384x6x6, ~11 M), C1 electrode embedding at the token
# site. configs/pretrain/emg_c1_moe.yaml says what differs and why.
#
# BEFORE THE FIRST RUN, once per corpus:
#
#   bash scripts/download_emg_pretrain_corpora.sh all          # login node
#   DATASET=emg2pose INSPECT=40 bash EMG/preprocess_emg_corpus.sh
#   sbatch --export=ALL,DATASET=emg2pose --array=0-15 \
#          scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#   python scripts/build_eeg_c1_manifest.py --modality emg \
#          --corpus-root ${PW_DATA_EMG}/emg_c1_corpus --allow-missing --check-shards --jobs 16
#
# ENVIRONMENT VARIABLES, all optional -- the same set as the EEG launcher:
#
#   NUM_GPUS              GPUs per node                              (4)
#   NNODES                nodes                            ($SLURM_NNODES or 1)
#   CONFIG                                               (pretrain/emg_c1_moe)
#   EPOCHS / GRAD_ACCUMULATION / LR / WEIGHT_DECAY / MASK_RATIO / SEED
#                                                             (the config's)
#   BATCH_SIZE_BY_ROUTE   "W16_2000=192,A24_2000=128,G64_2000=48"
#   WEIGHTS               balanced | proportional | temperature:0.5
#   DATA_ROOT             holds merged/manifest_{train,val}.jsonl
#                                           (${PW_DATA_EMG}/emg_c1_corpus)
#   OUTPUT_DIR            checkpoints and figures
#                                     (${PW_CKPT_ROOT}/pretrain_emg_c1_moe)
#   RESUME                'auto' to continue OUTPUT_DIR/latest.pth
#   INIT_FROM             another checkpoint's WEIGHTS, fresh schedule
#   STEPS_PER_EPOCH       override the epoch length
#   MAX_STEPS             stop after this many optimizer steps
#   SET                   "model.depth=6 model.mask_ratio=0.7" -- anything
#
# Unset means the config decides; see EEG/pretrain_eeg_c1_moe.sh for why none
# of the hyperparameters has a default here.
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

# In this order: the right interpreter, then what is installed in it.
# Reversed, a missing package is reported against an interpreter that was
# never meant to have it, and the message sends you to pip.
pw_require_training_venv || exit 1
pw_require_python_deps || exit 1

NUM_GPUS="${NUM_GPUS:-4}"          # GPUs per node
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
CONFIG="${CONFIG:-pretrain/emg_c1_moe}"
# No defaults for the hyperparameters: unset means the config decides.
EPOCHS="${EPOCHS:-}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
MASK_RATIO="${MASK_RATIO:-}"
SEED="${SEED:-}"
# balanced | proportional | temperature:A -- WITHOUT a space after the
# colon, which YAML would read as a mapping rather than the string the
# policy branch matches on.
WEIGHTS="${WEIGHTS:-}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EMG}/emg_c1_corpus}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_emg_c1_moe}"
BATCH_SIZE_BY_ROUTE="${BATCH_SIZE_BY_ROUTE:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME="${RESUME:-}"
# Weights only, fresh optimizer and schedule -- not a resume.
INIT_FROM="${INIT_FROM:-}"
# Caps the epoch. The config sets it explicitly (latest.pth is written only
# at epoch boundaries, so an epoch must fit a walltime); null derives it from
# the mixture. The trainer's banner prints what it resolved.
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
# Anything else, space separated: SET="model.depth=6 model.mask_ratio=0.7".
SET="${SET:-}"

MANIFEST_TRAIN="${MANIFEST_TRAIN:-${DATA_ROOT}/merged/manifest_train.jsonl}"
MANIFEST_VAL="${MANIFEST_VAL:-${DATA_ROOT}/merged/manifest_val.jsonl}"

# Say what is missing once, here, rather than as a stack trace after the model
# has been built on every rank.
for f in "${MANIFEST_TRAIN}" "${MANIFEST_VAL}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: no manifest at ${f}" >&2
        echo "       Preprocess the corpora first, one run per dataset:" >&2
        echo "         DATASET=<id> bash EMG/preprocess_emg_corpus.sh" >&2
        echo "       then merge them:" >&2
        echo "         python scripts/build_eeg_c1_manifest.py --modality emg \\" >&2
        echo "             --corpus-root ${DATA_ROOT} --allow-missing" >&2
        echo "       Nothing here falls back to synthetic data; --smoke-test is" >&2
        echo "       the explicit way to run without a corpus." >&2
        exit 1
    fi
done

OVERRIDES=(
  "data.manifest_train=${MANIFEST_TRAIN}"
  "data.manifest_val=${MANIFEST_VAL}"
)
# -n, not :+ -- MASK_RATIO=0 is a legitimate value and a presence test that
# reads it as unset would silently drop it.
[[ -n "${EPOCHS}" ]]           && OVERRIDES+=("train.epochs=${EPOCHS}")
[[ -n "${GRAD_ACCUMULATION}" ]] && OVERRIDES+=("train.grad_accumulation_steps=${GRAD_ACCUMULATION}")
[[ -n "${LR}" ]]               && OVERRIDES+=("train.lr=${LR}")
[[ -n "${WEIGHT_DECAY}" ]]     && OVERRIDES+=("train.weight_decay=${WEIGHT_DECAY}")
[[ -n "${MASK_RATIO}" ]]       && OVERRIDES+=("model.mask_ratio=${MASK_RATIO}")
[[ -n "${SEED}" ]]             && OVERRIDES+=("seed=${SEED}")
[[ -n "${WEIGHTS}" ]]          && OVERRIDES+=("data.weights=${WEIGHTS}")
[[ -n "${STEPS_PER_EPOCH}" ]]  && OVERRIDES+=("train.steps_per_epoch=${STEPS_PER_EPOCH}")
if [[ -n "${SET}" ]]; then
    read -r -a _extra_set <<< "${SET}"
    for _kv in "${_extra_set[@]}"; do
        [[ "${_kv}" == *=* ]] || { echo "ERROR: SET entry '${_kv}' is not key=value" >&2; exit 1; }
        OVERRIDES+=("${_kv}")
    done
fi

# "W16_2000=192,A24_2000=128,G64_2000=48" -> one dotted override per route.
if [[ -n "${BATCH_SIZE_BY_ROUTE}" ]]; then
    IFS=',' read -r -a _pairs <<< "${BATCH_SIZE_BY_ROUTE}"
    for _p in "${_pairs[@]}"; do
        [[ "${_p}" == *=* ]] || { echo "ERROR: BATCH_SIZE_BY_ROUTE entry '${_p}' is not ROUTE=N" >&2; exit 1; }
        OVERRIDES+=("train.batch_size_by_route.${_p}")
    done
fi

EXTRA=()
[[ -n "${MAX_STEPS}" ]] && EXTRA+=(--max-steps "${MAX_STEPS}")
[[ -n "${RESUME}" ]]    && EXTRA+=(--resume "${RESUME}")
[[ -n "${INIT_FROM}" ]] && EXTRA+=(--init-from "${INIT_FROM}")

# EVERY SHARD STAYS OPEN. A step reads its windows scattered over a corpus of
# hundreds of shards per dataset, and with the loader's default cache of 512
# open files per dataset most reads reopened an HDF5 file on Lustre -- which
# made the first ECG run ~50 s a step. Keeping them all open needs the
# descriptors for it: the soft limit is raised to the hard one, and the
# per-dataset cap is what that limit allows across six datasets with room to
# spare.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
_fd_limit="$(ulimit -n)"
if [[ "${_fd_limit}" == "unlimited" ]]; then _fd_limit=1048576; fi
_cap=$(( (_fd_limit - 1024) / 6 ))
export PW_MAX_OPEN_SHARDS="${PW_MAX_OPEN_SHARDS:-$(( _cap < 4096 ? _cap : 4096 ))}"
# ~2 shards a unit: emg2pose ~260, CEMHSEY ~800 per corpus at the defaults.
if [[ "${PW_MAX_OPEN_SHARDS}" -lt 1000 ]]; then
    echo "WARNING: open-file limit ${_fd_limit} allows ${PW_MAX_OPEN_SHARDS} shards per" >&2
    echo "  dataset, fewer than the largest corpus's shard count: reads will reopen files." >&2
fi
export PW_PREFETCH="${PW_PREFETCH:-1}"
# A rank that waits on a slow read must not take the job down: the first run
# died at step 408 on the 10-minute default collective timeout.
export PW_DIST_TIMEOUT_MIN="${PW_DIST_TIMEOUT_MIN:-60}"

pw_check_run_path OUTPUT_DIR "${OUTPUT_DIR}" || exit 1
# `set -e` is deliberately not on here, so an unchecked mkdir failure
# carries straight on to srun and dies inside Python on every rank.
mkdir -p "${OUTPUT_DIR}" || {
    echo "ERROR: cannot create ${OUTPUT_DIR}" >&2
    exit 1
}

echo "============================================================"
echo "  sEMG C1 multi-route pretraining"
echo "  config=${CONFIG}"
echo "  nodes=${NNODES} x ${NUM_GPUS} gpu = $((NNODES * NUM_GPUS)) rank(s)"
echo "  epochs=${EPOCHS:-<config>}  grad_accum=${GRAD_ACCUMULATION:-<config>}"
echo "  lr=${LR:-<config>}  wd=${WEIGHT_DECAY:-<config>}  mask_ratio=${MASK_RATIO:-<config>}  seed=${SEED:-<config>}"
echo "  weights=${WEIGHTS:-<config>}  steps/epoch=${STEPS_PER_EPOCH:-<derived>}"
[[ -n "${SET}" ]] && echo "  set            ${SET}"
echo "  (<config> means ${CONFIG}.yaml decides; the trainer's own banner"
echo "   below prints the values it actually resolved)"
echo "  train manifest ${MANIFEST_TRAIN}"
echo "  val   manifest ${MANIFEST_VAL}"
echo "  out            ${OUTPUT_DIR}"
echo "  open files     limit ${_fd_limit}, shards kept open per dataset ${PW_MAX_OPEN_SHARDS}, prefetch ${PW_PREFETCH}, collective timeout ${PW_DIST_TIMEOUT_MIN} min"
[[ -n "${BATCH_SIZE_BY_ROUTE}" ]] && echo "  batch/route    ${BATCH_SIZE_BY_ROUTE}"
[[ -n "${RESUME}" ]] && echo "  resume         ${RESUME}"
[[ -n "${INIT_FROM}" ]] && echo "  init from      ${INIT_FROM}  (weights only)"
echo "============================================================"

# The full command, recorded next to the checkpoints, so a run can be reproduced
# from its own output directory rather than from shell history.
# --standalone brings up its own rendezvous on localhost, which is right for
# one node and wrong for four: every node would elect itself rank 0 and the
# job would run as N independent one-node trainings that never all-reduce.
# Above one node, c10d against a named endpoint is what joins them.
if [[ "${NNODES}" -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        echo "ERROR: NNODES=${NNODES} but MASTER_ADDR is unset. The sbatch" >&2
        echo "       sets it from the nodelist; exporting it is how the ranks" >&2
        echo "       find each other." >&2
        exit 1
    fi
    RDZV=(--nnodes="${NNODES}" --nproc_per_node="${NUM_GPUS}"
          --node_rank="${SLURM_NODEID:-0}"
          --rdzv_id="${RDZV_ID:-${SLURM_JOB_ID:-0}}"
          --rdzv_backend=c10d
          --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT:-29500}")
else
    RDZV=(--standalone --nproc_per_node="${NUM_GPUS}")
fi

CMD=("${PW_TORCHRUN[@]}" "${RDZV[@]}"
     -m physiowave.train.pretrain_main
     --config "${CONFIG}"
     --output-dir "${OUTPUT_DIR}"
     ${EXTRA[@]+"${EXTRA[@]}"}
     --set "${OVERRIDES[@]}")

# One node writes this. Four nodes racing on the same path leaves whichever
# finished last, and the file is meant to record the run rather than a node.
if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/train_command.txt"
    echo >> "${OUTPUT_DIR}/train_command.txt"
    printf '%s\n' "$(cat "${OUTPUT_DIR}/train_command.txt")"
fi

"${CMD[@]}"
_rc=$?

if [[ ${_rc} -eq 0 ]]; then
    echo ""
    echo "Checkpoints: best.pth (= best_total.pth, lowest val total loss),"
    echo "  best_spec.pth, best_raw.pth, best_macro_total.pth, latest.pth"
    echo "Progress: metrics_epoch.jsonl (per route: route/W16_2000/..., route/G64_2000/...)"
fi
exit "${_rc}"
