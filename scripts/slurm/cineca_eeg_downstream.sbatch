#!/bin/bash
# ============================================================================
# EEG C1 downstream on Sleep-EDF and PhysioNet P300: every task x mode in ONE
# job, one node, four GPUs (EEG/finetune_eeg_c1.sh). GPU i takes runs i, i+4,
# ... of the TASKS x MODES list, then the table of every finished run
# (scripts/collect_eeg_downstream.py: per mode, the run with the best
# VALIDATION score is the one compared).
#
#   sbatch scripts/slurm/cineca_eeg_downstream.sbatch
#
#   TASKS       "sleep_edf physio_p300"
#   MODES       "ft@1e-4 ft@3e-5 probe@1e-3 scratch"   MODE or MODE@LR:
#                 ft       pretrained encoder fine-tuned, layer decay 0.75
#                 probe    pretrained encoder frozen; frontend, pool, head train
#                 scratch  same model, random initialisation (config lr)
#               a MODE token can also carry variants, each a "+" suffix:
#                 +route   on the pretraining route ROUTE_ID (E64_256): the
#                          pretrained frontend loads too (EEG/finetune_eeg_c1.sh)
#                 +cw      class-balanced cross-entropy
#                 +els<x>  ft: pretrained parameters at x times the lr
#                 +tp      EEGPT's PhysioP300 head (physiowave/eeg_c1/heads.py):
#                          a gain per electrode, per-position projection kept
#                          in time order and flattened, dropout 0.5, max-norm
#                          output. A frozen encoder's tokens averaged over
#                          time cannot say "at 300 ms"; this head can.
#               e.g. "ft@1e-3+route+cw+els0.1 probe@1e-3+route+cw scratch+route+cw"
#   SEEDS       "42"; several, e.g. "42 43 44", run every token once per seed
#               (directories other than seed 42 end in _s<seed>; the table
#               reports mean and std over the seeds of each configuration)
#   EPOCHS      20 (the configs carry EEGPT's 40 / 100)
#   PRETRAINED  $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth
#   TAG, SET, EXTRA, LAYER_DECAY   passed through
#
# The splits are built beforehand (they need mne, the P300 download a login
# node); a missing split stops that run with the command that builds it. A run
# whose results.json has a test block is skipped: resubmit to finish what the
# walltime cut off. FORCE=1 reruns everything.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_eeg_downstream
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"
TASKS="${TASKS:-sleep_edf physio_p300}"
MODES="${MODES:-ft@1e-4 ft@3e-5 probe@1e-3 scratch}"
N_GPUS="${N_GPUS:-4}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/eeg_downstream}"
export EPOCHS="${EPOCHS:-20}" NUM_GPUS=1 PROGRESS="${PROGRESS:-log}" OUT_ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS="${SEEDS:-42}"
ROUTE_ID="${ROUTE_ID:-E64_256}"
TP_HEAD="model.eeg_c1.pool=time model.eeg_c1.channel_pool=attn model.eeg_c1.spatial_filter=scale model.eeg_c1.head_dropout=0.5 model.eeg_c1.head_max_norm=1.0"
RUNS=()
for t in ${TASKS}; do for m in ${MODES}; do for s in ${SEEDS}; do
    RUNS+=("${t}:${m}:${s}"); done; done; done
mkdir -p "${OUT_ROOT}"
echo "============================================================"
echo "  EEG C1 downstream  (job ${SLURM_JOB_ID:-?}, 1 node, ${N_GPUS} GPUs)"
echo "  tasks   ${TASKS}"
echo "  modes   ${MODES}   epochs ${EPOCHS}   seeds ${SEEDS}"
echo "  encoder ${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
echo "  output  ${OUT_ROOT}"
echo "============================================================"

done_already() {   # results.json with a test block
    python - "$1/results.json" <<'PY' 2>/dev/null
import json, sys
sys.exit(0 if "test" in json.load(open(sys.argv[1])) else 1)
PY
}
worker() {
    local gpu="$1" i spec task token seed base mode lr v route cw els runset out rc=0
    local -a parts flags
    for ((i = gpu; i < ${#RUNS[@]}; i += N_GPUS)); do
        IFS=: read -r task token seed <<< "${RUNS[$i]}"
        IFS=+ read -r -a flags <<< "${token}"
        base="${flags[0]}"; flags=("${flags[@]:1}")
        mode="${base%%@*}"; lr=""; [[ "${base}" == *@* ]] && lr="${base#*@}"
        route=""; cw=""; els=""; runset=""
        parts=(); [[ -n "${lr}" ]] && parts+=("lr${lr}")
        for v in ${flags[@]+"${flags[@]}"}; do
            case "${v}" in
                route) route="${ROUTE_ID}"; parts+=(route) ;;
                cw)    cw=balanced; parts+=(cw) ;;
                els*)  els="${v#els}"; parts+=("${v}") ;;
                tp)    runset="${TP_HEAD}"; parts+=(tp) ;;
                *) echo "[gpu ${gpu}] ${token}: unknown variant +${v}"; rc=1; continue 2 ;;
            esac
        done
        [[ -n "${TAG:-}" ]] && parts+=("${TAG}")
        [[ "${seed}" != "42" ]] && parts+=("s${seed}")
        out="${OUT_ROOT}/${task}_${mode}"; for v in ${parts[@]+"${parts[@]}"}; do out+="_${v}"; done
        if [[ "${FORCE:-0}" != "1" ]] && done_already "${out}"; then
            echo "[gpu ${gpu}] ${task} ${token} seed ${seed}: done, skipped"; continue
        fi
        echo "[gpu ${gpu}] ${task} ${token} seed ${seed}: start $(date +%H:%M)"
        if CUDA_VISIBLE_DEVICES="${gpu}" TASK="${task}" MODE="${mode}" LR="${lr}" \
                SEED="${seed}" ROUTE="${route}" CLASS_WEIGHT="${cw}" ENC_LR_SCALE="${els}" \
                SET="${SET:-}${runset:+ ${runset}}" \
                OUTPUT_DIR="${out}" bash EEG/finetune_eeg_c1.sh \
                > "${out}.log" 2>&1; then
            echo "[gpu ${gpu}] ${task} ${token} seed ${seed}: done $(date +%H:%M)"
        else
            echo "[gpu ${gpu}] ${task} ${token} seed ${seed}: FAILED -- see ${out}.log"; rc=1
        fi
    done
    return ${rc}
}
pids=()
for ((g = 0; g < N_GPUS && g < ${#RUNS[@]}; g++)); do worker "${g}" & pids+=($!); done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
python scripts/collect_eeg_downstream.py --root "${OUT_ROOT}" \
    --markdown "${OUT_ROOT}/summary.md" | tee "${OUT_ROOT}/summary.txt"
exit ${rc}
