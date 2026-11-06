#!/bin/bash
# ============================================================================
# EEG C1 multi-route pretraining -- the recommended entry point.
#
#   bash EEG/pretrain_eeg_c1_moe.sh
#
# Four wavelet frontends over seven corpora, one shared RoPE Transformer, C1
# channel-name embedding injected at the token site. This is NOT the WAST/TARE
# path: it shares no module with it, and configs/pretrain/eeg.yaml is a
# different run.
#
# BEFORE THE FIRST RUN the corpora have to be preprocessed, once per dataset:
#
#   python EEG/preprocess_pretrain_corpus.py --dataset tueg \
#       --root ${PW_DATA_EEG}/tueg --out-dir ${DATA_ROOT}/tueg --mains-hz 60
#
# and their manifests merged into ${DATA_ROOT}/merged. That step needs mne and
# belongs under $HOME/pwprep, not in the training venv.
#
# ENVIRONMENT VARIABLES, all optional:
#
#   NUM_GPUS              GPUs per node                              (4)
#   NNODES                nodes                            ($SLURM_NNODES or 1)
#   CONFIG                which objective          (pretrain/eeg_c1_moe = full)
#   EPOCHS                                                    (the config's)
#   MAX_STEPS             stop after this many optimizer steps      (unset)
#   BATCH_SIZE_BY_ROUTE   "E19_256=128,E32_512=96,E64_256=48,E128_512=24"
#   GRAD_ACCUMULATION                                         (the config's)
#   LR / WEIGHT_DECAY / MASK_RATIO / SEED                     (the config's)
#   WEIGHTS               balanced | proportional | temperature:0.5
#   DATA_ROOT             holds merged/manifest_{train,val}.jsonl
#   OUTPUT_DIR            checkpoints and figures
#   RESUME                'auto' to continue OUTPUT_DIR/latest.pth
#   INIT_FROM             another checkpoint's WEIGHTS, fresh schedule
#   STEPS_PER_EPOCH       override the mixture-derived epoch length
#   SET                   "model.embed_dim=512 model.depth=8" -- anything
#
# UNSET MEANS THE CONFIG DECIDES for the hyperparameters. None of them has a
# default here; a default here would be a second copy of a number that already
# has a home, and the copy would win.
#
# DATA_ROOT AND OUTPUT_DIR ARE PATHS AND ARE CHECKED. Both are usually built
# from ${PW_DATA_EEG} and ${PW_CKPT_ROOT}, which scripts/cineca_env.sh sets and
# your login profile does not. Submitting from a shell that never sourced it
# turns $PW_CKPT_ROOT/run into /run, which --export=ALL carries into the job.
#
# These are VALUES passed straight through, not `${VAR:+--flag}` presence
# tests: MASK_RATIO=0 is a non-empty string and that idiom would read it as on.
# The presence test that decides whether to override at all is `-n`, for the
# same reason.
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
# Which objective. pretrain/eeg_c1_moe is the full one -- 0.5 x spec MSE
# + 0.5 x raw SmoothL1 + 1e-3 x ScaleFold KL. The two ablations are pinned to
# the earlier 1.0/0.25 weighting they were run at and differ from each other in
# the raw_weight and mask_before_frontend lines.
CONFIG="${CONFIG:-pretrain/eeg_c1_moe}"
# NO DEFAULTS FOR THE HYPERPARAMETERS. Each of these used to carry a copy of
# the config's value and pass it through --set unconditionally, so the config
# was overridden by an identical number and editing the config did nothing --
# the file said mask_ratio 0.75 and the run banner said 0.5. Unset means "the
# config decides"; set means override, which is what an override is for.
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
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe}"
BATCH_SIZE_BY_ROUTE="${BATCH_SIZE_BY_ROUTE:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME="${RESUME:-}"
# Weights only, fresh optimizer and schedule -- not a resume. Use it when
# WEIGHTS or the epoch budget changes, because steps_per_epoch changes
# with the mixture and a resumed scheduler counts in the old epoch's
# units: 384-step epochs restored into 954-step ones leave the cosine
# a fifth short of annealed.
INIT_FROM="${INIT_FROM:-}"
# steps_per_epoch is DERIVED when the config leaves it null. Under the config's
# `proportional` policy that derivation is the EXACT ONE-PASS LENGTH over the
# corpus -- every dataset seen once, nothing revisited -- which on ~15.4 M
# windows is a large number. The banner prints what it resolved; do not carry
# the old `balanced` figure of 384 in your head, it is from a different policy.
#
# SETTING IT CAPS THE EPOCH. That is the point of the override: it turns "one
# pass over everything" back into a fixed budget, which is what you want for a
# short annealing run or for fitting an epoch inside a walltime:
#
#   EPOCHS=15 STEPS_PER_EPOCH=768 GRAD_ACCUMULATION=1 LR=1e-4 \
#   SET="train.warmup_epochs=0" \
#   INIT_FROM=$PW_CKPT_ROOT/pretrain_eeg_c1_moe/latest.pth \
#   bash EEG/pretrain_eeg_c1_moe.sh
#
# 768 x 15 = 11,520 optimizer updates at grad_accum 1. Note this is FAR SHORT
# of one pass over the corpus -- it is a deliberate cap, and under proportional
# it means most of TUEG goes unseen in those fifteen epochs. Use it to anneal a
# model that has already had its full passes, not to train one from scratch.
#
# INIT_FROM and not RESUME: a change of epoch length is a change of the unit
# the cosine counts in, and a change of objective weights is a different loss;
# --resume refuses both by design and says so.
#
# NOTHING HERE IS HARD-CODED IN PYTHON. These are values for this cluster and
# this corpus; the trainer's banner prints the steps/epoch, the total optimizer
# updates, the world size, the per-route batch and the passes/epoch it actually
# resolved, and warns when an epoch reads one dataset more than
# train.max_passes_per_epoch_warn (5) times -- which under proportional it
# never should, since every row is 1.00x by construction.
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
# Anything else, space separated: SET="model.embed_dim=512 model.depth=8".
# For the architecture, which has no business having an environment
# variable each, and which changing means no checkpoint transfers.
SET="${SET:-}"

MANIFEST_TRAIN="${MANIFEST_TRAIN:-${DATA_ROOT}/merged/manifest_train.jsonl}"
MANIFEST_VAL="${MANIFEST_VAL:-${DATA_ROOT}/merged/manifest_val.jsonl}"

# Say what is missing once, here, rather than as a stack trace after the model
# has been built on every rank.
for f in "${MANIFEST_TRAIN}" "${MANIFEST_VAL}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: no manifest at ${f}" >&2
        echo "       Preprocess the corpora first, one run per dataset:" >&2
        echo "         python EEG/preprocess_pretrain_corpus.py --dataset <id> \\" >&2
        echo "             --root <raw> --out-dir ${DATA_ROOT}/<id> --mains-hz <50|60>" >&2
        echo "       then merge the per-dataset manifests into ${DATA_ROOT}/merged." >&2
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

# "E19_256=64,E32_512=48" -> one dotted override per route.
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

pw_check_run_path OUTPUT_DIR "${OUTPUT_DIR}" || exit 1
# `set -e` is deliberately not on here, so an unchecked mkdir failure
# carries straight on to srun and dies inside Python on every rank.
mkdir -p "${OUTPUT_DIR}" || {
    echo "ERROR: cannot create ${OUTPUT_DIR}" >&2
    exit 1
}

echo "============================================================"
echo "  EEG C1 multi-route pretraining"
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
     "${EXTRA[@]}"
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
    echo "Figures:"
    echo "  python scripts/visualize_eeg_pretraining.py \\"
    echo "      --run-dir ${OUTPUT_DIR} --checkpoint best.pth --split val --format svg"
    echo ""
    echo "Progress:"
    echo "  python scripts/eeg_c1_progress.py ${OUTPUT_DIR}"
    echo ""
    echo "Checkpoints: best.pth (= best_total.pth, lowest val total loss),"
    echo "  best_spec.pth, best_raw.pth, best_macro_total.pth, latest.pth"
fi
exit "${_rc}"
