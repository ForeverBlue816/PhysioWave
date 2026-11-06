#!/bin/bash
# ============================================================================
# The hyper-parameter grid for the EEGPT-benchmark sweep. Sourced by
# scripts/slurm/cineca_eegpt_sweep.sbatch; the collector reads the results it
# produces from their "sw_<name>" directory tags.
#
# Aimed at what the first full run showed: every model peaked within its first
# 4-14 epochs of 30 and then overfitted, and the pretrained-minus-scratch gap
# on BCIC was small -- a 25M-parameter encoder fine-tuned at a single rate on
# ~5k trials overwrites what pretraining gave it. So the grid moves the overall
# rate, the rate of the PRETRAINED parameters relative to the fresh ones
# (encoder_lr_scale), and regularisation. Everything else stays at the config.
#
# Each entry: NAME|LR|extra finetune_main flags|config overrides (SET)
# The same grid runs for ft AND scratch, and each is selected on its own
# validation subjects, so pretraining gets no tuning advantage the control did
# not also get.
# ============================================================================

EEGPT_SWEEP_GRID=(
    "base|2.5e-4||"
    "lr1e4|1e-4||"
    "lr5e5|5e-5||"
    "enc01|2.5e-4|--encoder-lr-scale 0.1|"
    "enc005|1e-3|--encoder-lr-scale 0.05|"
    "reg|1e-4|--weight-decay 0.05|model.dropout=0.3 model.eeg_c1.head_dropout=0.3"
)

# eegpt_sweep_names -> the configuration names, space-separated
eegpt_sweep_names() {
    local e; for e in "${EEGPT_SWEEP_GRID[@]}"; do printf '%s ' "${e%%|*}"; done
}

# eegpt_sweep_env NAME -> exports LR, EXTRA, SET and TAG for that configuration
eegpt_sweep_env() {
    local e name lr extra set
    for e in "${EEGPT_SWEEP_GRID[@]}"; do
        IFS='|' read -r name lr extra set <<< "${e}"
        if [[ "${name}" == "$1" ]]; then
            # BASE_SET / BASE_EXTRA apply to every configuration -- an override
            # meant for the whole sweep, e.g. BASE_SET="model.eeg_c1.patch_samples=128".
            # The configuration's own come after, so they win on a clash.
            export LR="${lr}" TAG="sw_${name}" \
                   EXTRA="${BASE_EXTRA:-}${BASE_EXTRA:+ }${extra}" \
                   SET="${BASE_SET:-}${BASE_SET:+ }${set}"
            return 0
        fi
    done
    echo "ERROR: no sweep configuration '$1' (have: $(eegpt_sweep_names))" >&2
    return 1
}
