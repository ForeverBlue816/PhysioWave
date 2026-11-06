# EEG C1 downstream: Sleep-EDF and PhysioNet P300

The pretrained EEG C1 encoder in three modes on two tasks, one job:

| mode | what trains |
|---|---|
| `ft` | everything, layer-wise lr decay 0.75 over the Transformer |
| `probe` | what pretraining did not supply: the montage's fresh frontend, the pool, the head (`freeze_scope: pretrained`) |
| `scratch` | everything, from random initialisation (the control) |

Neither montage sits on a pretraining route (Sleep-EDF: Fpz-Cz and Pz-Oz at
100 Hz; P300: 64 electrodes at 256 Hz with 512-sample windows), so in both the
frontend is built fresh and what transfers is the channel embedding, its gate
and the Transformer. A probe that also froze the fresh frontend would freeze
random filters.

```bash
sbatch scripts/slurm/cineca_eeg_downstream.sbatch     # -> $PW_CKPT_ROOT/eeg_downstream
```

Defaults: `TASKS="sleep_edf physio_p300"`, `MODES="ft@1e-4 ft@3e-5 probe@1e-3
scratch"` (scratch at the config's lr: 4e-4 / 5e-4), `EPOCHS=20` (the configs
carry EEGPT's 40 / 100). Per mode, the table (`summary.txt`,
`scripts/collect_eeg_downstream.py`) marks the run with the best validation
score; only that one is compared, beside EEGPT's and LaBraM's published rows,
which come from fold-averaged protocols scored on the folds they select on.
One subject-disjoint split here, with a test set nothing selects on.

The splits are built beforehand (`EEG/finetune_eeg_c1.sh` prints the command
when one is missing):

```bash
source $HOME/pwprep/bin/activate          # mne
python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf --split holdout
python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci       # login node
python EEG/physio_p300_finetune.py --edf-dir $PW_DATA_EEG/erpbci \
    --out-dir $PW_DATA_EEG/p300_f0 --fold 0
```

A Siena-continued encoder: `PRETRAINED=$PW_CKPT_ROOT/pretrain_eeg_c1_siena/latest.pth
TAG=siena` (the runs get `_siena` in their directory and their own rows).

## Variants: route, class weights, encoder lr, seeds

A `MODES` token takes `+` suffixes, and `SEEDS` repeats each token:

```bash
sbatch --export=ALL,TASKS=physio_p300,SEEDS="42 43 44",MODES="ft@1e-3+cw+els0.1 \
ft@1e-3+route+cw+els0.1 probe@1e-3+cw probe@1e-3+route+cw scratch+cw scratch+route+cw" \
    scripts/slurm/cineca_eeg_downstream.sbatch
```

- `+route` places the montage on the pretraining route `ROUTE_ID` (E64_256; P300's
  58 electrodes are 58 of its 64 slots, the other six stay zero, as in
  pretraining). The pretrained wavelet frontend and patcher load with the
  Transformer, so a probe freezes the encoder end to end and trains the head
  alone, which is EEGPT's linear probe. Without it the montage gets a fresh
  frontend and only the Transformer transfers (the configs' default, the
  narrower question).
- `+cw`: cross-entropy weighted N / (K n_c) over the training labels
  (`--class-weight balanced`). P300 is about one target in six, and the argmax
  of an unweighted loss leans to the majority class; balanced accuracy and
  kappa see that, AUROC does not.
- `+els<x>` (ft): the pretrained parameters, the route frontend among them
  when it was loaded, learn at x times the lr under layer decay 0.75; the fresh
  frontend and the head learn at the lr. `ft@1e-3+els0.1` puts the fresh parts
  at the probe's 1e-3 and the Transformer at 1e-4 and below. A single lr of
  1e-4 trains the fresh frontend and head ten times slower than the probe does.
- `+tp`: EEGPT's PhysioP300 head (`physiowave/eeg_c1/heads.py`): a learnable
  gain per electrode, the tokens pooled over electrodes by one learned query,
  projected per time position to 16 and flattened in time order, dropout 0.5,
  a max-norm output (about 10k trainable on the full model). With the default
  head (all tokens averaged, then a linear layer: 2k parameters) the
  route-placed probe stays at chance, train loss included, because a mean over
  the time positions cannot say where in the window the P300 is; the
  fresh-frontend probe hides this because its frontend trains. Give the
  scratch arm the same head.
- Seeds other than 42 add `_s<seed>` to the directory. The table averages the
  seeds of one configuration (std underneath) and selects on the mean
  validation score.
