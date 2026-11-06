#!/bin/bash
# ============================================================================
# Multimodal fusion on the three C1 encoders: every run of the comparison in
# ONE job, one node, four GPUs (physiowave/fusion/train.py).
#
#   sbatch scripts/slurm/cineca_fusion_c1.sbatch
#   sbatch --export=ALL,TASKS=mpdb scripts/slurm/cineca_fusion_c1.sbatch     # MPDB alone
#
# GPU i takes runs i, i+4, ... of RUNS. A run whose results.json exists is
# skipped and one with a latest.pth resumes after its last finished epoch, so
# resubmitting the same command finishes what the walltime cut off. Ends with the table
# (scripts/collect_fusion.py).
#
#   RUNS      "ralf:frozen concat:frozen weighted_logit:frozen ralf:finetune ralf:scratch"
#             fusion:encoder_mode pairs. frozen = the pretrained core fixed,
#             finetune = layer-wise decay, scratch = no encoder loaded.
#             A VARIANT is fusion:mode:label:key=value,key=value -- the label
#             names the directory (<fusion>_<mode>_<label>[_<TAG>]) and the
#             overrides go to --set, e.g.
#               "ralf:frozen:lr1e-4:train.lr=1e-4,train.epochs=15,train.patience=6"
#             Pass such RUNS in the environment (RUNS="..." sbatch <this>), not
#             through --export, which splits its argument at every comma.
#   TASKS     "mpdb seizeit2"   (data from scripts/slurm/cineca_fusion_prep.sbatch)
#   EEG_CKPT  $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth
#   ECG_CKPT  $PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth
#   EMG_CKPT  $PW_CKPT_ROOT/pretrain_emg_c1_moe/epoch38.pth
#   SEEDS     "42"; several seeds run every RUNS entry, directories suffixed _s<seed> except 42
#   TAG       suffix on every run directory
#   SET       extra --set overrides, space separated
#   EXTRA     raw physiowave.fusion.train flags, e.g. "--device cpu --max-steps 2"
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_fusion_c1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

FUSION_ROOT="${FUSION_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/fusion}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/fusion}"
RUNS="${RUNS:-ralf:frozen concat:frozen weighted_logit:frozen ralf:finetune ralf:scratch}"
SEEDS="${SEEDS:-42}"
TASKS="${TASKS:-mpdb seizeit2}"
EEG_CKPT="${EEG_CKPT:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
ECG_CKPT="${ECG_CKPT:-${PW_CKPT_ROOT}/pretrain_ecg_c1_moe/best.pth}"
EMG_CKPT="${EMG_CKPT:-${PW_CKPT_ROOT}/pretrain_emg_c1_moe/epoch38.pth}"
N_GPUS=4
RUN_LIST=()
for seed in ${SEEDS}; do for spec in ${RUNS}; do RUN_LIST+=("${spec}:${seed}"); done; done
mkdir -p "${OUT_ROOT}"

TASK_SETS=()
for t in ${TASKS}; do
    d="${FUSION_ROOT}/c1/${t}"
    [[ -f "${d}/train.h5" ]] || { echo "ERROR: ${d}/train.h5 missing -- run cineca_fusion_prep.sbatch"; exit 1; }
    TASK_SETS+=("data.tasks.${t}.dir=${d}")
done
for m in EEG ECG EMG; do
    v="${m}_CKPT"; [[ -f "${!v}" ]] || { echo "ERROR: no ${m} checkpoint at ${!v}"; exit 1; }
done
echo "runs    ${RUNS}"
echo "tasks   ${TASKS}"
echo "eeg     ${EEG_CKPT}"; echo "ecg     ${ECG_CKPT}"; echo "emg     ${EMG_CKPT}"
echo "output  ${OUT_ROOT}"

worker() {
    local gpu="$1" i spec fusion mode label overrides seed out log rc=0
    for ((i = gpu; i < ${#RUN_LIST[@]}; i += N_GPUS)); do
        spec="${RUN_LIST[$i]}"; seed="${spec##*:}"; spec="${spec%:*}"
        IFS=: read -r fusion mode label overrides <<< "${spec}"
        local -a extra=()
        [[ -n "${overrides:-}" ]] && IFS=, read -r -a extra <<< "${overrides}"
        out="${OUT_ROOT}/${fusion}_${mode}${label:+_${label}}${TAG:+_${TAG}}$([[ "${seed}" != "42" ]] && echo "_s${seed}")"
        if [[ -f "${out}/results.json" && "${FORCE:-0}" != "1" ]]; then
            echo "[gpu ${gpu}] ${spec}: already done, skipped"; continue
        fi
        log="${out}.log"; mkdir -p "${out}"
        echo "[gpu ${gpu}] ${spec}: start $(date +%H:%M) -> ${log}"
        # shellcheck disable=SC2086
        if CUDA_VISIBLE_DEVICES="${gpu}" python -m physiowave.fusion.train \
                --config fusion/c1_ralf --output-dir "${out}" ${EXTRA:-} \
                --set "${TASK_SETS[@]}" "encoders.eeg=${EEG_CKPT}" "encoders.ecg=${ECG_CKPT}" \
                      "encoders.emg=${EMG_CKPT}" "model.fusion=${fusion}" \
                      "model.encoder_mode=${mode}" "seed=${seed}" ${extra[@]+"${extra[@]}"} ${SET:-} \
                      > "${log}" 2>&1; then
            echo "[gpu ${gpu}] ${spec}: done $(date +%H:%M)"
        else
            echo "[gpu ${gpu}] ${spec}: FAILED -- see ${log}"; rc=1
        fi
    done
    return ${rc}
}

pids=()
for ((g = 0; g < N_GPUS && g < ${#RUN_LIST[@]}; g++)); do
    worker "${g}" &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done

echo ""
python scripts/collect_fusion.py --root "${OUT_ROOT}" \
    --markdown "${OUT_ROOT}/summary.md" --json "${OUT_ROOT}/summary.json" | tee "${OUT_ROOT}/summary.txt"
exit ${rc}
