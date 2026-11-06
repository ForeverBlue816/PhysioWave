#!/bin/bash
# ============================================================================
# EEG C1 multi-route pretraining, as a batch job.
#
#   sbatch scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# Overrides go through --export, same as the other sbatch scripts here. Note the
# commas: --export takes a comma-separated list, so quote any value containing
# one:
#
#   sbatch --export=ALL,EPOCHS=80,"BATCH_SIZE_BY_ROUTE=E19_256=32,E128_512=8" \
#          scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# RESUMABLE. OUTPUT_DIR carries no job id and the trainer writes latest.pth
# every epoch with the optimizer, scheduler, AMP scaler, sampler position, RNG
# state and history in it, so RESUME=auto continues the sequence the interrupted
# run would have drawn next rather than restarting the epoch.
#
#   sbatch --export=ALL,RESUME=auto scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# The corpora are NOT preprocessed here. That step needs mne and belongs under
# $HOME/pwprep; this job reads the manifests it produced.
# ============================================================================
# SCALE. Four nodes of four GPUs, sixteen ranks. batch_size_by_route is PER
# RANK, so the effective batch is sixteen times it and the optimizer sees
# sixteen times as many windows per step as a single-GPU run at the same
# setting. Two consequences worth deciding on rather than discovering:
#
#   * steps_per_epoch falls by the same factor -- the sampler divides the
#     corpus by batch x world_size -- so an epoch is a quarter as many steps as
#     it was on four GPUs, and a schedule expressed in epochs is unchanged
#     while one expressed in steps is not.
#   * the learning rate was chosen at a smaller effective batch. LR is NOT
#     scaled here, because doing it silently is worse than not doing it: decide
#     it and pass LR= explicitly.
#
# Falling back to one node is NUM_GPUS=4 with --nodes=1; the launcher switches
# rendezvous accordingly.
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=physiowave_eeg_c1_moe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"

# sbatch defaults to --export=ALL, so this job inherits the submitting shell's
# environment -- including a PW_VENV left pinned to $HOME/pwprep by the
# preprocessing step, which is exactly the shell someone submits this from.
if [[ -n "${PW_VENV:-}" && "${PW_VENV}" != "${VENV:-${HOME}/pw}" \
      && "${PW_ALLOW_INHERITED_VENV:-0}" != "1" ]]; then
    echo "NOTE: ignoring inherited PW_VENV=${PW_VENV}; using ${VENV:-${HOME}/pw}." >&2
    unset PW_VENV
fi

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

export NUM_GPUS="${NUM_GPUS:-4}"                  # per node
export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / NUM_GPUS))}"

# Which objective to train. The default is the full one; the ablations exist to
# separate its three contributions and are selected here, not by editing a file.
export CONFIG="${CONFIG:-pretrain/eeg_c1_moe}"

# WHERE THE RANKS MEET. One task per node, and torchrun starts NUM_GPUS local
# ranks inside each. The endpoint is the first node in the allocation, and the
# port is derived from the job id so two concurrent jobs on overlapping nodes
# do not try to rendezvous with each other.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}"
export RDZV_ID="${SLURM_JOB_ID}"

# NCCL over the fabric rather than over whatever it picks first. If a run hangs
# at startup with all ranks joined, this is the first thing to check.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Large caches and checkpoints belong on $FAST, never in $HOME (50 GB quota).
export DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe}"

# Both arrive from the submitting shell through --export=ALL, and both are
# built from variables that scripts/cineca_env.sh sets rather than the login
# profile. Checked here, before srun, because the alternative is finding out
# from rank 0's permission error and fifteen rendezvous timeouts.
pw_check_run_path OUTPUT_DIR "${OUTPUT_DIR}" || exit 1
pw_check_run_path DATA_ROOT "${DATA_ROOT}" || exit 1

echo "============================================================"
echo "  EEG C1 multi-route pretraining  (job ${SLURM_JOB_ID})"
echo "  config     ${CONFIG}"
# Print the torch the job will actually import. The development machine
# runs a much newer torch than cineca-ai ships, and an API that exists
# in one and not the other only shows up as a traceback on rank 0.
echo "  torch      $(python -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"
echo "  ${NNODES} node(s) x ${NUM_GPUS} gpu = $((NNODES * NUM_GPUS)) rank(s)"
echo "  rendezvous ${MASTER_ADDR}:${MASTER_PORT} (id ${RDZV_ID})"
echo "  nodes      $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  epochs=${EPOCHS:-<script default>}  resume=${RESUME:-<none>}"
echo "  progress=${PW_PROGRESS:-auto (log, under sbatch)}""  heartbeat=${PW_LOG_SECONDS:-120}s"
pw_print_layout
echo "  DATA_ROOT  =${DATA_ROOT}"
echo "  OUTPUT_DIR =${OUTPUT_DIR}   (resubmit with RESUME=auto to continue)"
echo "============================================================"

# One task per node; torchrun forks NUM_GPUS ranks inside each. --export=ALL so
# MASTER_ADDR, MASTER_PORT and CONFIG reach every node.
srun --ntasks-per-node=1 --export=ALL bash EEG/pretrain_eeg_c1_moe.sh
_rc=$?

echo ""
echo "============================================================"
echo "  Figures:"
echo "    python scripts/visualize_eeg_pretraining.py \\"
echo "        --run-dir ${OUTPUT_DIR} --checkpoint best.pth --split val --format svg"
echo "    Several candidate windows per route, ranked, to choose one from:"
echo "    python scripts/visualize_eeg_pretraining.py \\"
echo "        --run-dir ${OUTPUT_DIR} --checkpoint best.pth --format png --recon-windows 8"
echo "  Resume:"
echo "    sbatch --export=ALL,RESUME=auto \\"
echo "        scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch"
echo "============================================================"
exit "${_rc}"
