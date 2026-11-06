#!/bin/bash
# =============================================================================
# PhysioWave RALF fusion training on CINECA Leonardo.
#
#   # pick up the newest pretrained encoder of each modality automatically
#   sbatch scripts/slurm/cineca_fusion.sbatch
#
#   # or pin them explicitly
#   sbatch --export=ALL,EEG_CKPT=...,ECG_CKPT=...,SEMG_CKPT=... \
#          scripts/slurm/cineca_fusion.sbatch
#
# Paths, modules and the venv all come from scripts/cineca_env.sh:
#   code $HOME/PhysioWave-T | ckpt $PW_FAST_ROOT/runs | data $SCRATCH/bio/<mod>
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=physiowave_ralf
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# -------- Environment (modules + venv + storage layout) --------
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/fusion_ralf_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"

# -------- Pretrained encoders --------
# Default to the most recent pretrain_<modality>_<jobid>/best.pth under
# $PW_FAST_ROOT/runs.  An encoder with no checkpoint is trained from scratch,
# which is a legitimate ablation -- so this warns rather than fails.
# Always succeeds: a missing encoder prints nothing and returns 0, so `set -e`
# does not kill the job over an ablation that deliberately has no checkpoint.
latest_ckpt() {
    local m="$1" newest=""
    newest="$(ls -1dt "${PW_CKPT_ROOT}"/pretrain_"${m}"_*/best.pth 2>/dev/null | head -n 1 || true)"
    printf '%s' "${newest}"
    return 0
}
EEG_CKPT="${EEG_CKPT:-$(latest_ckpt eeg)}"
ECG_CKPT="${ECG_CKPT:-$(latest_ckpt ecg)}"
SEMG_CKPT="${SEMG_CKPT:-$(latest_ckpt semg)}"

for m in EEG ECG SEMG; do
    v="${m}_CKPT"
    if [[ -z "${!v}" ]]; then
        echo "WARNING: no pretrained ${m} encoder found under ${PW_CKPT_ROOT}; " \
             "that branch starts from random init." >&2
    fi
done

SET_ARGS=(
    "train.epochs=${EPOCHS:-50}"
    "train.batch_size=${BATCH_SIZE:-8}"
    "train.precision=${PRECISION:-bf16}"
    "data.manifest_dir=${PW_MANIFEST_DIR}"
    "data.preprocess.cache_dir=${PW_PREP_CACHE}"
    "model.encoders.eeg.spatial.ssl.cache_dir=${PW_SSL_CACHE}"
)
[[ -n "${DATASETS:-}" ]] && SET_ARGS+=("data.datasets=[${DATASETS}]")
[[ -n "${EEG_CKPT}"  ]] && SET_ARGS+=("fusion.pretrained.eeg=${EEG_CKPT}")
[[ -n "${ECG_CKPT}"  ]] && SET_ARGS+=("fusion.pretrained.ecg=${ECG_CKPT}")
[[ -n "${SEMG_CKPT}" ]] && SET_ARGS+=("fusion.pretrained.semg=${SEMG_CKPT}")
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && SET_ARGS+=(${EXTRA})

if [[ -z "${DATASETS:-}" ]]; then
    echo "WARNING: DATASETS is unset -- RALF will train on the SYNTHETIC corpus. " \
         "Pass --export=ALL,DATASETS=<registry ids> for a real run." >&2
fi

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29501}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE))}"

echo "============================================================"
echo "  RALF fusion: nodes=${SLURM_NNODES} gpus/node=${GPUS_PER_NODE}"
echo "  eeg=${EEG_CKPT:-<scratch>}"
echo "  ecg=${ECG_CKPT:-<scratch>}"
echo "  semg=${SEMG_CKPT:-<scratch>}"
pw_print_layout
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================================"

srun --kill-on-bad-exit=1 --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    "${PW_TORCHRUN[@]}" --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${SLURM_NNODES}" \
    --rdzv_id="${SLURM_JOB_ID}" --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m physiowave.train.fusion_main \
    --config "${CONFIG:-fusion/ralf}" --output-dir "${OUTPUT_DIR}" \
    --resume auto --set "${SET_ARGS[@]}"

echo "Done. Outputs in ${OUTPUT_DIR}"
