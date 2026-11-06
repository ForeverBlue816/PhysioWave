#!/bin/bash
# ============================================================================
# Draw the SeizeIT2 multimodal figures (scripts/visualize_seizeit2.py) from the
# finished runs and put them under docs/runs/<NAME> for the repository.
#
#   sbatch scripts/slurm/cineca_visualize_seizeit2.sbatch
#   sbatch --export=ALL,EXAMPLE=sub-101_run-03 scripts/slurm/cineca_visualize_seizeit2.sbatch
#
# then, on a login node:
#   git add docs/runs/<NAME> && git commit -m "<NAME>: figures" && git push
#
# A job rather than a login-node command: it reads one EDF recording and the
# 778 cached prediction files. CPU only.
#
#   RUN       the main run directory under $PW_CKPT_ROOT/fusion   (ralf_frozen_szo)
#   COMPARE   other trimodal runs   (concat_frozen_szo weighted_logit_frozen_szo
#                                    ralf_finetune_szo ralf_scratch_szo)
#   EEG_RUNS  EEG-only runs: frozen and fine-tuned for PhysioWave-T (c1),
#             LUNA-L, LaBraM and EEGPT (fig_main's panel b)
#   ONLY      figure names to draw, e.g. ONLY=fig_main (default: all)
#   EXAMPLE   recording for the example figure (default: the ranked choice)
#   WINDOW_START  start of the 4 s per-modality input window, seconds from the
#             seizure onset (default: centred on the seizure)
#   NAME      docs/runs/<NAME>      (seizeit2_fusion)
#   FORMAT    svg,pdf (vector, text editable); png also accepted
#   PAPER     1: no titles/notes/stamp (kept in figure_metadata); 0: working figures
# A run named in COMPARE / EEG_RUNS that has no events_cross.json yet is
# skipped from the curves and said so in the log.
# ============================================================================
#SBATCH -p boost_usr_prod
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000
#SBATCH --job-name=physiowave_viz_sz2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
FUSION_ROOT="${FUSION_ROOT:-${PW_DATA_ROOT:-${SCRATCH:-${CINECA_SCRATCH}}/bio}/fusion}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/fusion}"
RUN="${RUN:-ralf_frozen_szo}"
COMPARE="${COMPARE:-concat_frozen_szo weighted_logit_frozen_szo ralf_finetune_szo ralf_scratch_szo}"
EEG_RUNS="${EEG_RUNS:-eeg_c1_frozen_sze eeg_c1_finetune_sze eeg_luna_large_frozen_sze eeg_luna_large_finetune_sze eeg_labram_frozen_sze eeg_labram_finetune_sze eeg_eegpt_frozen_sze eeg_eegpt_finetune_sze}"
NAME="${NAME:-seizeit2_fusion}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

present() {   # keep the named runs that have an events file
    local out=() d
    for d in $1; do
        if [[ -f "${OUT_ROOT}/${d}/events_cross.json" ]]; then out+=("${OUT_ROOT}/${d}")
        else echo "  ${d}: no events_cross.json yet, left out"; fi
    done
    echo "${out[@]:-}"
}
[[ -f "${OUT_ROOT}/${RUN}/events_cross.json" ]] || { echo "ERROR: ${RUN} has no events_cross.json"; exit 1; }
COMPARE_DIRS="$(present "${COMPARE}")"
EEG_DIRS="$(present "${EEG_RUNS}")"
# shellcheck disable=SC2086
python scripts/visualize_seizeit2.py --raw-dir "${FUSION_ROOT}/raw/seizeit2" \
    --run-dir "${OUT_ROOT}/${RUN}" --compare ${COMPARE_DIRS} --eeg-runs ${EEG_DIRS} \
    --out-dir "docs/runs/${NAME}" --format "${FORMAT:-svg,pdf}" \
    ${EXAMPLE:+--example "${EXAMPLE}"} ${WINDOW_START:+--window-start "${WINDOW_START}"} \
    ${ONLY:+--only ${ONLY}} \
    $([[ "${PAPER:-1}" == "1" ]] && echo --paper) || exit 1
rm -rf "docs/runs/${NAME}/figure_data"          # the arrays: large, redrawable
echo "figures in docs/runs/${NAME}; on a login node: git add docs/runs/${NAME} && git commit && git push"
