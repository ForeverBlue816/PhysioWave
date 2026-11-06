#!/bin/bash
# =============================================================================
# PhysioP300, all nine LOSO folds, in one job.
#
#   sbatch scripts/slurm/cineca_p300_folds.sbatch
#
# Overrides go through --export, same as the other sbatch scripts here:
#
#   sbatch --export=ALL,EPOCHS=15,DROPOUT=0.2,HEAD_DROPOUT=0.3,WEIGHT_DECAY=0.05 \
#          scripts/slurm/cineca_p300_folds.sbatch
#
# Nine folds because one is a pilot. A single held-out subject is the whole
# test set, and between-subject variance is the dominant term on an ERP task --
# EEGPT's own +-0.0139 on kappa is across folds, not across seeds. The number
# to report is the mean over the nine, with the spread.
#
# Four hours at 15 epochs a fold. Raise --time before raising EPOCHS.
#
# The decode is NOT repeated: the .npz cache under $PW_DATA_EEG/erpbci is built
# once by EEG/physio_p300_finetune.py --stage cache and shared by every fold.
# This job only builds each fold's HDF5 split, which is file assembly.
#
# Disk: each fold's train/val/test is ~2.5 GB. KEEP_SPLITS=1 leaves all nine in
# place (22.5 GB); the default removes a fold's HDF5 once that fold has been
# scored, since rebuilding one from the cache takes a couple of minutes.
# =============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=physiowave_p300_folds
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

export NUM_GPUS="${NUM_GPUS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / NUM_GPUS))}"

EDF_DIR="${EDF_DIR:-${PW_DATA_EEG}/erpbci}"
SWEEP_ROOT="${SWEEP_ROOT:-${PW_CKPT_ROOT}/p300_folds_${SLURM_JOB_ID}}"
KEEP_SPLITS="${KEEP_SPLITS:-0}"
mkdir -p "${SWEEP_ROOT}"

# The cache is the expensive part and this job does not build it. Say so here
# rather than nine times over, each as a split-stage failure.
# The montage. One variable, so the split that is written and the model that
# reads it cannot disagree -- IN_CHANNELS reaches the split as --channels and
# EEG/finetune_p300.sh as itself.
IN_CHANNELS="${IN_CHANNELS:-64}"
export IN_CHANNELS

if ! CACHE_DIR="$(pw_p300_cache_dir "${EDF_DIR}")"; then
    echo "ERROR: no decode cache under ${EDF_DIR}/cache (looked for c64, c58)" >&2
    echo "       Build it first, on a node with mne available:" >&2
    echo "         source \$HOME/pwprep/bin/activate" >&2
    echo "         PW_VARS_ONLY=1 source scripts/cineca_env.sh" >&2
    echo "         python EEG/physio_p300_finetune.py --edf-dir ${EDF_DIR} \\" >&2
    echo "             --out-dir ${PW_DATA_EEG}/p300_f0 --stage cache --jobs 1" >&2
    exit 1
fi
if [[ "$(basename "${CACHE_DIR}")" == "c58" && "${IN_CHANNELS}" != "58" ]]; then
    # Not a warning: the split stage would fail nine times over, once per fold,
    # and the reason is the same each time.
    echo "ERROR: ${CACHE_DIR} is the legacy cache and holds only EEGPT's 58" >&2
    echo "       electrodes, so no ${IN_CHANNELS}-channel split can come out" >&2
    echo "       of it -- AF7, AF8, AFz and Iz were never decoded." >&2
    echo "       Re-decode once and every channel set becomes a split:" >&2
    echo "         python EEG/physio_p300_finetune.py --edf-dir ${EDF_DIR} \\" >&2
    echo "             --out-dir ${PW_DATA_EEG}/p300_f0 --stage cache --jobs 2" >&2
    echo "       Or run this job at the montage that is on disk: IN_CHANNELS=58" >&2
    exit 1
fi
_n_cached="$(find "${CACHE_DIR}" -maxdepth 1 -name 'sub*.npz' | wc -l)"
echo "Decode cache: ${CACHE_DIR} (${_n_cached} subjects)"

echo "============================================================"
echo "  PhysioP300 -- 9 LOSO folds"
echo "  montage=${IN_CHANNELS} channels"
echo "  epochs=${EPOCHS:-<script default>}  select_by=${SELECT_BY:-<default>}"
echo "  dropout=${DROPOUT:-<default>} head_dropout=${HEAD_DROPOUT:-<default>}"
echo "  weight_decay=${WEIGHT_DECAY:-<default>}"
echo "  gpus=${NUM_GPUS}  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
pw_print_layout
echo "  SWEEP_ROOT=${SWEEP_ROOT}"
echo "============================================================"

failed=()
for k in $(seq 0 8); do
    DATA_DIR="${PW_DATA_EEG}/p300_f${k}"
    OUT_DIR="${SWEEP_ROOT}/fold${k}"

    echo ""
    echo "----- fold ${k} -----------------------------------------"

    # Rebuild the split every time rather than trusting what is on disk: a
    # directory left by an interrupted run can hold a train.h5 from a different
    # fold, and nothing downstream would notice.
    if ! python EEG/physio_p300_finetune.py --edf-dir "${EDF_DIR}" \
             --out-dir "${DATA_DIR}" --stage split --fold "${k}" \
             --channels "${IN_CHANNELS}"; then
        echo "fold ${k}: split failed, skipped" >&2
        failed+=("${k}")
        continue
    fi

    if ! FOLD="${k}" DATA_DIR="${DATA_DIR}" OUTPUT_DIR="${OUT_DIR}" \
         bash EEG/finetune_p300.sh; then
        echo "fold ${k}: training failed" >&2
        failed+=("${k}")
    fi

    # One fold's HDF5 is 2.5 GB and nine of them do not need to coexist. The
    # cache stays, so rebuilding any of them is a couple of minutes.
    [[ "${KEEP_SPLITS}" == "1" ]] || rm -f "${DATA_DIR}"/{train,val,test}.h5
done

echo ""
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "  ${#failed[@]} fold(s) failed: ${failed[*]}"
fi
echo "  Results under ${SWEEP_ROOT}/fold*/test_results.json"
echo "  Summarise with: python scripts/collect_p300_folds.py ${SWEEP_ROOT}"
echo "============================================================"

python scripts/collect_p300_folds.py "${SWEEP_ROOT}" || true
