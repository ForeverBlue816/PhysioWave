#!/bin/bash
# =============================================================================
# PhysioWave pretraining on CINECA Leonardo (Booster partition).
#
#   sbatch --export=ALL,MODALITY=eeg  scripts/slurm/cineca_pretrain.sbatch
#   sbatch --export=ALL,MODALITY=ecg  scripts/slurm/cineca_pretrain.sbatch
#   sbatch --export=ALL,MODALITY=semg scripts/slurm/cineca_pretrain.sbatch
#
# Paths, modules and the venv all come from scripts/cineca_env.sh:
#   code $HOME/PhysioWave-T | ckpt $PW_FAST_ROOT/runs | data $SCRATCH/bio/<mod>
# Override any of them through --export (PW_CKPT_ROOT, PW_DATA_EEG, ...).
# Local runs do not depend on Slurm at all -- use scripts/run_tpami.sh.
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --job-name=physiowave_pretrain
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# -------- Environment (modules + venv + storage layout) --------
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

MODALITY="${MODALITY:-eeg}"
CONFIG="${CONFIG:-pretrain/${MODALITY}}"

# -------- Corpora --------
# Registry ids per modality; override with DATASETS="tueg,siena".  Each id is
# expected at $SCRATCH/bio/<modality>/<id>/*.h5 -- see scripts/cineca_env.sh.
if [[ -z "${DATASETS:-}" ]]; then
    case "${MODALITY}" in
        eeg)   DATASETS="tueg,siena" ;;
        ecg)   DATASETS="mimic_iv_ecg" ;;
        semg)  DATASETS="ninapro_db6" ;;
        *)     echo "ERROR: unknown MODALITY '${MODALITY}'" >&2; exit 1 ;;
    esac
fi
IFS=',' read -r -a DATASET_IDS <<< "${DATASETS}"

# An empty corpus makes physiowave fall back to the synthetic smoke dataset,
# which would burn a 24 h four-node reservation on noise.  Fail instead.
DATA_DIR="$(pw_data_dir "${MODALITY}")"
MODALITY_UC="$(printf '%s' "${MODALITY}" | tr '[:lower:]' '[:upper:]')"
for id in "${DATASET_IDS[@]}"; do
    found=0
    for f in "${DATA_DIR}/${id}"/*.h5; do
        [[ -e "${f}" ]] && { found=1; break; }
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "ERROR: no *.h5 under ${DATA_DIR}/${id}." >&2
        echo "       Build the corpus first (EEG/tueg_pretrain.py, ECG/mimic_pretrain.py," >&2
        echo "       EMG/db6_pretrain.py) or set DATASETS / PW_DATA_${MODALITY_UC}." >&2
        exit 1
    fi
done

ROOT_ARGS=()
while IFS= read -r line; do
    ROOT_ARGS+=("${line}")
done < <(pw_roots_for "${MODALITY}" "${DATASET_IDS[@]}")

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_${MODALITY}_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"

# -------- Hyperparameters (all overridable via --export) --------
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA="${EXTRA:-}"

# -------- Multi-node rendezvous --------
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE))}"

echo "============================================================"
echo "  PhysioWave pretraining -- modality=${MODALITY}"
echo "  config=${CONFIG}  datasets=${DATASETS}"
echo "  nodes=${SLURM_NNODES} gpus/node=${GPUS_PER_NODE}"
echo "  MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "  epochs=${EPOCHS} bs=${BATCH_SIZE} accum=${GRAD_ACCUM} lr=${LR} precision=${PRECISION}"
pw_print_layout
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================================"

TRAIN_ARGS=(
    --config "${CONFIG}"
    --output-dir "${OUTPUT_DIR}"
    --resume auto
    --set
    "data.datasets=[${DATASETS}]"
    "${ROOT_ARGS[@]}"
    "data.manifest_dir=${PW_MANIFEST_DIR}"
    "data.preprocess.cache_dir=${PW_PREP_CACHE}"
    "model.spatial.ssl.cache_dir=${PW_SSL_CACHE}"
    "train.epochs=${EPOCHS}"
    "train.batch_size=${BATCH_SIZE}"
    "train.grad_accumulation_steps=${GRAD_ACCUM}"
    "train.lr=${LR}"
    "train.weight_decay=${WEIGHT_DECAY}"
    "train.warmup_epochs=${WARMUP_EPOCHS}"
    "train.precision=${PRECISION}"
    "train.num_workers=${NUM_WORKERS}"
)
# shellcheck disable=SC2206
[[ -n "${EXTRA}" ]] && TRAIN_ARGS+=(${EXTRA})

srun --kill-on-bad-exit=1 \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    "${PW_TORCHRUN[@]}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${SLURM_NNODES}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m physiowave.train.pretrain_main "${TRAIN_ARGS[@]}"

echo "Done. Outputs in ${OUTPUT_DIR}"
