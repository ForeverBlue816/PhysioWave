# Collected run logs

Logs from runs on the cluster, stripped of tqdm's carriage-return rewrites and
committed so a result can be read in a diff instead of a terminal paste.

Collected with, from the repository root:

    bash scripts/collect_runs.sh docs/runs/<experiment> ~/<pattern>*.log

Each run contributes two files: `<name>.log` (the hyper-parameter header, one
line per epoch, the test block) and `<name>.json` (`test_results.json` copied
from the run directory, which is the authoritative result -- the log is a
transcript and can be truncated by a job hitting its wall clock).

Checkpoints are not collected. They are large, `*.pth` is in `.gitignore`, and
nothing about a result needs them.

## Pretraining runs

C1 pretraining runs (EEG, ECG, sEMG) are collected with their figures:

    sbatch --export=ALL,MODALITY=ecg scripts/slurm/cineca_visualize_run.sbatch
    git add docs/runs/ecg_c1_moe && git commit -m "ecg_c1_moe: figures" && git push

The job draws the figures from `best.pth` into the run directory, then
`scripts/collect_pretrain_run.sh` copies the figures (SVG and PDF, text left editable), each figure's metadata, the
per-epoch metrics, the resolved config and the progress tables into
`docs/runs/<name>/`, with a README that shows the best epochs and every figure
inline. Checkpoints, the per-step metrics and the arrays behind each figure
stay in the run directory.

## Journal figures (TPAMI extension)

Four composites drawn at their printed width (7.16 in, text 8 pt and up, no
figure-level title, no corner stamp -- the caption material is in each
`figure_metadata/<name>.json`):

| figure | what | drawn by |
|---|---|---|
| `fig_paper_mask_reconstruction` | both heads, four routes, eight panels each; masked correlation in the error panels | `cineca_visualize_run.sbatch` |
| `fig_paper_raw_waveform` | the raw head on one channel per route, prediction on masked patches only | `cineca_visualize_run.sbatch` |
| `fig_paper_channel_topology` | name-embedding cosine similarity against scalp distance, seed topomaps, MDS vs. true layout (needs `mne`) | `cineca_visualize_run.sbatch` |
| `fig_paper_fold_by_stage` | dynamic ScaleFold alpha per scale and sleep stage, one night, alpha vs. delta share | `cineca_visualize_sleep_fold.sbatch` |

    ONLY="fig_paper_mask_reconstruction fig_paper_raw_waveform fig_paper_channel_topology" \
        RECON_WINDOWS=8 sbatch --export=ALL,MODALITY=eeg scripts/slurm/cineca_visualize_run.sbatch
    sbatch scripts/slurm/cineca_visualize_sleep_fold.sbatch

The first lands in `docs/runs/eeg_c1_moe/`, the second in `docs/runs/sleep_edf_fold/`.
