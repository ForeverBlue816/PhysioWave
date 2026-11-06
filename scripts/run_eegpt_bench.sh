#!/bin/bash
# ============================================================================
# One command: BCIC-IV-2a, BCIC-IV-2b and KaggleERN on the C1 encoder.
#
#   bash scripts/run_eegpt_bench.sh
#
# Run it on the LOGIN node. It
#   1. makes sure each dataset is downloaded -- fetching what is missing,
#      skipping what is complete (compute nodes have no internet);
#   2. drops any task whose data cannot be had, rather than submitting a job
#      that would fail at its first step;
#   3. submits one job per task. Each builds its split (the pretraining
#      preprocessing), then fine-tunes the pretrained encoder (ft) and trains
#      the same model from scratch (scratch) side by side, and scores both on
#      held-out subjects;
#   4. tells you where the results will be.
#
# Knobs, all optional:
#   PRETRAINED   encoder or pretraining checkpoint
#                (default $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth)
#   TASKS        default "bcic2a bcic2b kaggleern"
#   FOLD         default 0
#   TAG          suffix on the result directories, to keep encoders apart:
#                  PRETRAINED=.../latest.pth TAG=final MODES=ft bash scripts/run_eegpt_bench.sh
#   MODES        default "ft scratch"
#   PREP         pretrain (default) or eegpt -- EEGPT's preprocessing, BCIC
#                only. Results get an "eegptprep" tag, beside the default run:
#                  PREP=eegpt TASKS="bcic2a bcic2b" bash scripts/run_eegpt_bench.sh
#   SWEEP=1      the hyper-parameter sweep instead of one run per mode: every
#                configuration in EEG/eegpt_sweep_grid.sh, for ft and scratch,
#                each selected on its own VALIDATION score. CONFIGS restricts it.
#   SKIP_DOWNLOAD=1  only check what is on disk
#   DRY_RUN=1        print the sbatch commands instead of submitting
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/cineca_env.sh
# shellcheck disable=SC1091
source EEG/eegpt_bench_lib.sh

PRETRAINED="${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
TASKS="${TASKS:-${EEGPT_TASKS[*]}}"
FOLD="${FOLD:-0}"
MODES="${MODES:-ft scratch}"

echo "============================================================"
echo "  EEGPT benchmarks   tasks: ${TASKS}   fold ${FOLD}   modes: ${MODES}   prep: ${PREP:-pretrain}"
[[ "${SWEEP:-0}" == 1 ]] && echo "  SWEEP: ${CONFIGS:-every configuration in EEG/eegpt_sweep_grid.sh}"
echo "  encoder  ${PRETRAINED}"
echo "  data     ${PW_DATA_EEG}"
echo "  results  ${PW_CKPT_ROOT}/eegpt_bench"
echo "============================================================"

if [[ "${MODES}" != "scratch" && ! -f "${PRETRAINED}" ]]; then
    echo "ERROR: no encoder at ${PRETRAINED}. Set PRETRAINED=<path>." >&2
    exit 1
fi

# --- 1. data ----------------------------------------------------------------- #
count() { find -L "$1" -maxdepth "${3:-1}" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

ready=()
for t in ${TASKS}; do
    raw="$(eegpt_raw_dir "${t}")" || exit 1
    case "${t}" in
        bcic2a|bcic2b)
            ds="${t#bcic}"
            if [[ "$(count "${raw}" '*.mat')" -lt 18 && "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
                echo "[${t}] fetching BCIC-IV-${ds} into ${raw}"
                python EEG/download_eegpt_benchmarks.py --dataset "${ds}" --dest "${raw}"
            fi
            n="$(count "${raw}" '*.mat')"
            if [[ "${n}" -ge 18 ]]; then
                echo "[${t}] data ready: ${n} files"; ready+=("${t}")
            else
                echo "[${t}] SKIPPED: ${n}/18 files in ${raw}" >&2
            fi ;;
        kaggleern)
            if [[ "$(count "${raw}/train" 'Data_S*.csv')" -lt 1 && "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
                echo "[${t}] fetching KaggleERN into ${raw}"
                bash EEG/download_kaggle_ern.sh "${raw}"
            fi
            n="$(count "${raw}/train" 'Data_S*.csv')"
            if [[ "${n}" -ge 1 && -f "${raw}/TrainLabels.csv" ]]; then
                echo "[${t}] data ready: ${n} labelled sessions"; ready+=("${t}")
            else
                echo "[${t}] SKIPPED: no labelled sessions in ${raw} --" \
                     "see EEG/download_kaggle_ern.sh (needs a Kaggle token)" >&2
            fi ;;
    esac
done
if [[ ${#ready[@]} -eq 0 ]]; then
    echo "ERROR: no task has its data; nothing submitted." >&2
    exit 1
fi

# --- 2. submit ----------------------------------------------------------------- #
ids=()
for t in "${ready[@]}"; do
    if [[ "${SWEEP:-0}" == 1 ]]; then
        # TAG is the sweep's own (sw_<config>), so it is not passed on; CONFIGS
        # is, when set, and must not contain the comma --export splits on.
        cmd=(sbatch --parsable
             "--export=ALL,TASK=${t},FOLD=${FOLD},MODES=${MODES},PREP=${PREP:-pretrain},PRETRAINED=${PRETRAINED}${CONFIGS:+,CONFIGS=${CONFIGS}}"
             scripts/slurm/cineca_eegpt_sweep.sbatch)
    else
        cmd=(sbatch --parsable
             "--export=ALL,TASK=${t},FOLD=${FOLD},MODES=${MODES},PREP=${PREP:-pretrain},PRETRAINED=${PRETRAINED}${TAG:+,TAG=${TAG}}"
             scripts/slurm/cineca_eegpt_bench.sbatch)
    fi
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "[${t}] would run: ${cmd[*]}"
        continue
    fi
    if id="$("${cmd[@]}")"; then
        echo "[${t}] submitted job ${id}"
        ids+=("${id%%;*}")
    else
        echo "[${t}] sbatch FAILED" >&2
    fi
done

echo ""
[[ "${DRY_RUN:-0}" == 1 ]] && { echo "dry run: nothing submitted"; exit 0; }
[[ ${#ids[@]} -gt 0 ]] || { echo "ERROR: nothing was submitted" >&2; exit 1; }
cat <<EOF
Submitted ${#ids[@]} job(s): ${ids[*]}

  watch        squeue --me
  per run      ${PW_CKPT_ROOT}/eegpt_bench/<task>_f${FOLD}_<mode>${TAG:+_${TAG}}.log
  summary      ${PW_CKPT_ROOT}/eegpt_bench/summary.txt
               (rewritten by each job as it finishes; complete once all are done)
  any time     python scripts/collect_eegpt_bench.py --root ${PW_CKPT_ROOT}/eegpt_bench
EOF
