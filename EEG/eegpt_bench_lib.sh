#!/bin/bash
# ============================================================================
# Shared by EEG/finetune_eegpt_bench.sh, scripts/slurm/cineca_eegpt_bench.sbatch
# and scripts/run_eegpt_bench.sh: where each task's raw data and split live,
# and how a split is built. One copy, so the three cannot disagree about which
# directory a split is in.
#
# Sourced, not run. Needs PW_DATA_EEG (scripts/cineca_env.sh) and the repo root
# as the working directory.
# ============================================================================

EEGPT_TASKS=(bcic2a bcic2b kaggleern)

# The preprocessing version, read from the converters' own constant. It names
# the split directory, so a split built by an earlier pipeline -- the EEGPT-style
# one, say -- is never picked up by a later one just because its test.h5 exists.
eegpt_prep_version() {
    python -c "import sys; sys.path.insert(0, 'EEG'); \
import eegpt_bench_common as b; print(b.PREPS['${1:-pretrain}'])"
}

# PREP=pretrain (default) or eegpt. eegpt is BCIC-only; KaggleERN always uses
# the pretraining pipeline, so a PREP=eegpt run leaves its ERN split and results
# exactly where the default run put them.
eegpt_prep_for() {
    case "$1" in kaggleern) echo pretrain ;; *) echo "${PREP:-pretrain}" ;; esac
}

# The result-directory tag: TAG, plus the preprocessing when it is not the
# default, so a PREP=eegpt run sits next to the default one instead of
# overwriting it. Word characters only -- the collector parses it.
eegpt_tag() {
    local t="${TAG:-}"
    [[ "$(eegpt_prep_for "$1")" != pretrain ]] && t="${t:+${t}_}$(eegpt_prep_for "$1")prep"
    echo "${t}"
}

eegpt_num_classes() {
    case "$1" in bcic2a) echo 4 ;; bcic2b|kaggleern) echo 2 ;;
        *) echo "ERROR: unknown task '$1'" >&2; return 1 ;; esac
}

eegpt_raw_dir() {
    case "$1" in
        bcic2a)    echo "${PW_DATA_EEG}/bcic_iv2a" ;;
        bcic2b)    echo "${PW_DATA_EEG}/bcic_iv2b" ;;
        kaggleern) echo "${PW_DATA_EEG}/kaggle_ern" ;;
        *) echo "ERROR: unknown task '$1'" >&2; return 1 ;;
    esac
}

# eegpt_split_dir TASK FOLD [VERSION]
eegpt_split_dir() {
    local v="${3:-$(eegpt_prep_version "$(eegpt_prep_for "$1")")}"
    echo "${PW_DATA_EEG}/eegpt_bench/${1}_f${2}_${v}"
}

# eegpt_build_split TASK FOLD OUT_DIR RAW_DIR -- a no-op if it already exists.
eegpt_build_split() {
    local task="$1" fold="$2" out="$3" raw="$4"
    [[ -f "${out}/test.h5" ]] && { echo "split ${out} exists"; return 0; }
    if [[ ! -d "${raw}" ]]; then
        echo "ERROR: no raw data for ${task} at ${raw}; download it on the" \
             "login node (scripts/run_eegpt_bench.sh does)" >&2
        return 1
    fi
    echo "building the ${task} fold-${fold} split -> ${out}"
    case "${task}" in
        bcic2a)    python EEG/bcic_iv2_finetune.py --dataset 2a \
                       --prep "$(eegpt_prep_for "${task}")" \
                       --raw-dir "${raw}" --out-dir "${out}" --fold "${fold}" ;;
        bcic2b)    python EEG/bcic_iv2_finetune.py --dataset 2b \
                       --prep "$(eegpt_prep_for "${task}")" \
                       --raw-dir "${raw}" --out-dir "${out}" --fold "${fold}" ;;
        kaggleern) python EEG/kaggle_ern_finetune.py \
                       --raw-dir "${raw}" --out-dir "${out}" --fold "${fold}" ;;
    esac && return 0
    # A half-written split would pass the test.h5 check next time if the
    # converter died after writing it; remove it -- but only something that is
    # visibly one of these split directories, never a path an unset variable
    # could have shortened.
    [[ "${out}" == */eegpt_bench/*_f*_* && -d "${out}" ]] && rm -rf -- "${out}"
    return 1
}
