# Multimodal fusion on the three C1 encoders

Three independently pretrained encoders (EEG `pretrain_eeg_c1_moe`, ECG
`pretrain_ecg_c1_moe`, sEMG `pretrain_emg_c1_moe`, epoch 38) each hand the
fusion a few **summary tokens**; **RALF** (Reliability-Aware Latent Fusion)
fuses them with per-sample reliability, hard masks for missing modalities and
a residual of reliability-weighted unimodal logits. Two tasks share the
encoders and the fusion and keep their own heads:

| task | data | classes | split |
|---|---|---|---|
| `mpdb` | MPDB: 35 drivers, 59-ch EEG + chest ECG + 4 limb sEMG, 4 s windows around the simulator events | 5 behaviours (smooth, acceleration, deceleration, lane change, turning) | by subject, 25 / 5 / 5 |
| `seizeit2` | SeizeIT2: 125 patients, 2 behind-the-ear EEG + ECG + deltoid EMG, 4 s windows | seizure vs background | by patient, 85 / 15 / 25 |

The comparison is three fusion cores under one training loop and one test
matrix:

| `model.fusion` | what it is |
|---|---|
| `ralf` | 8 fusion tokens, two cross-attention layers over the summary tokens; a reliability per modality and sample gates attention and weights the unimodal residual; absent modalities are masked out of the attention |
| `concat` | mean-pool each modality, concatenate, classify; absence is a zero vector |
| `weighted_logit` | the earlier scheme: a logit per modality and one global learned weight per modality, renormalised over what is present |

and three encoder settings, `model.encoder_mode`: `frozen` (what pretraining
supplied stays fixed; the montage adapters, pools, fusion and heads train),
`finetune` (layer-wise decay 0.75 over the pretrained core), `scratch` (no
encoder loaded, the control).

## Commands (Leonardo)

```bash
cd ~/PhysioWave-T && git pull && source $HOME/pw/bin/activate
pip install mne                      # once; the MPDB release is EEGLAB .set files

# 1. data (login node)
bash scripts/download_fusion_data.sh mpdb_raw            # 23 GB, two .rar; the authors' epochs: mpdb
SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' bash scripts/download_fusion_data.sh seizeit2   # 39 patients first
bash scripts/download_fusion_data.sh seizeit2            # all 125, 97.5 GB

# 2. windows (compute node, CPU)
sbatch scripts/slurm/cineca_fusion_prep.sbatch
#    a subset: sbatch --export=ALL,TASKS=seizeit2,SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' scripts/slurm/cineca_fusion_prep.sbatch

# 3. the comparison: five runs on four GPUs, one job
sbatch scripts/slurm/cineca_fusion_c1.sbatch
#    MPDB alone first: sbatch --export=ALL,TASKS=mpdb scripts/slurm/cineca_fusion_c1.sbatch

# 4. the table: $PW_CKPT_ROOT/fusion/summary.{txt,md,json}; to share it
bash scripts/sync_results.sh
```

If the login node has no `unrar`, `7z` or `bsdtar`, the MPDB step prints the
three lines that put rarlab's `unrar` in `$HOME/bin`.

## Data

### MPDB (`Fusion/mpdb_raw_prep.py`; the authors' epochs via `Fusion/mpdb_prep.py`)

The source is the RAW release (figshare 22193119, 23 GB): per subject four
experiments (brake, throttle, turn, change), each a continuous EEGLAB file
per device -- EEG&ECG (59 electrodes and the chest ECG on one Neuracle
amplifier) and EMG (four limb channels on a NeuSen WM), 1 kHz, with the
simulator's markers and their times. The marker codes of the paper's
Table 2 give the five classes: smooth 133; acceleration 135, 137 (the
release's code for overtaking), 143; deceleration 139, 141, 145; lane change
129, 131; turning 125, 127.

Windows are 4 s, from 0.5 s before the marker, so the onset sits where the
authors' 2 s epochs put it and the window is the EEG encoder's pretraining
length. An EMG window is paired with the EEG/ECG window of the same event:
within an experiment the two devices' marker sequences are matched in order
(their LCS when one device missed a trigger) and their inter-event intervals
compared; a pair whose intervals disagree by more than 50 ms is refused and
the window keeps EEG/ECG without EMG (`present_emg` False). `split.json`
records per experiment the pairing counts and the devices' clock offset.
Each record is filtered whole with its modality's pretraining pipeline, then
cut, resampled (EEG 256 Hz, ECG 250 Hz, EMG 2000 Hz) and normalised as that
corpus was. No ICA and no epoch rejection: quality is the reliability
estimator's problem, which is the point.

The authors' preprocessed release (figshare 22192831: `EEG_n / ECG_n / EMG_n.set`,
2 s epochs, ICA-pruned) carries no timestamps and its three files differ in
length, so it can only be aligned by marker order with an ambiguity the
fusion would inherit. `Fusion/mpdb_prep.py` still reads it (`TASKS=mpdb_release`)
for comparison; it is not the paper's data.

The sEMG of the pedal leg can carry the label directly. That is why every
modality subset is reported, not only the complete one.

### SeizeIT2 (`Fusion/seizeit2_prep.py`)

OpenNeuro ds005873 (CC0), BIDS: per run EEG, ECG and EMG EDFs at 256 Hz and
an `events.tsv` (bckg / impd / sz_*). 4 s windows. Seizure windows every
second across each seizure (overlap at least 2 s); background windows on a
4 s grid at least 30 s from any seizure and clear of impedance checks,
subsampled to 30 per recorded hour (seed per run). Each run is filtered
whole (EEG: detrend, 50 Hz notch, 0.5 Hz; ECG: notch, 0.5 Hz; EMG: notch
with harmonics, 20 Hz), then windows are cut, resampled per modality and
normalised. ECG/EMG missing from a run (3% of the data) is kept as real
absence. `--bg-per-hour`, `--guard`, `--sz-stride` change the sampling;
`--subjects` restricts to a regex for a first pass.

**Split.** `--split official` (the default) is the dataset paper's fixed
split (`net/datasets/SZ2_*.tsv` in github.com/biomedepi/seizeit2):
sub-001..sub-096 train and sub-097..sub-125 the set the paper reports on,
which is our test set. 12% of the training patients (seed 42) are held out
for our own checkpoint selection, so the test patients are never seen before
the final evaluation.

### SeizeIT2 event-level evaluation (`Fusion/seizeit2_events.py`)

The window-level numbers above sub-sample the background (30 windows per
hour, seizures ~8% of the windows) and are not comparable with the
literature. The event-level evaluation runs the selected checkpoint over the
test patients' complete continuous recordings: 4 s windows every second,
centred on the second they score, the encoders run once and the fusion core
under every modality subset. Each second gets a label (its centre inside an
annotated `sz*` event) and the reference RMS gate (13 < RMS < 150 µV on the
2 s cross-head EEG window after a 0.5-60 Hz band and a 50 Hz stop; the
reference code gates on the cross-head channel alone, and so does
`--gate cross`, the default).

Scoring (`physiowave/fusion/seizure_events.py`) is a port of the reference
code: threshold, merge predicted events closer than 2 s, drop events shorter
than 8 s, any-overlap. It reports, at thresholds 0, 0.02, ..., 1:

- the dataset paper's figures, means over recordings: any-overlap
  sensitivity (whose TP counts overlapping prediction-seizure pairs, as the
  reference does), any-overlap and epoch FA/h, and the challenge score
  `100·sens − 0.4·FA_epoch/h`;
- pooled event figures over all recordings: seizures hit by at least one
  predicted event / seizures, and FA/24h;
- the sensitivity reachable at 0.5, 1, 2, 6.22 and 11 FA/h;
- the paper's AUROC and AUPR (`auroc_paper`, `aupr_paper`). The paper
  derives them "from metrics calculated using the traditional epoch-based
  method for computing sensitivity/recall, specificity and precision". So
  they are taken after the gate and post-processing, per recording,
  averaged over recordings, as areas over the 51 thresholds. The released
  code does not compute them, so this is a reading of that sentence;
- the pooled per-second AUROC and AUPRC, before post-processing. These are
  ours. The AUPRC sits near the per-second seizure prevalence (~0.1%), an
  order of magnitude under the paper's AUPR, and is **not** comparable with
  it.

A change to the scoring needs no new inference. The per-second
probabilities are cached in `<run>/predictions/*.npz`, and
`SCORE_ONLY=1 sbatch scripts/slurm/cineca_seizeit2_events.sbatch` re-scores
every run from them in parallel.

`scripts/collect_seizeit2_events.py` puts the runs beside the published
rows: the dataset paper's SVM (71.1% / 11.0 FA/h / AUROC 0.70 / AUPR 0.081)
and ChronoNet (84.2% / 100.5 FA/h / 0.79 / 0.104) on bte-EEG with this split
and scoring, and the Epilepsia Open 2026 bte-EEG + ECG algorithm (78%
per-patient sensitivity at 6.22 FA/h, its own cohort and protocol; indicative).

### What transfers

| modality | montage | route | what loads |
|---|---|---|---|
| EEG | MPDB's 59 electrodes | E64_256 if every name is one of its slots, else routeless | core (+ frontend on a route) |
| EEG | SeizeIT2's two bte channels | routeless (names not in the vocabulary: a new embedding row each, seeded from `<unk>`, trained as an adapter) | core |
| ECG | one chest derivation, named `patch1` | L1_250 | core + frontend |
| sEMG | 4 limb channels / 1 deltoid | routeless | core |

`encoders.json` in every run directory records, per task and modality, the
route and every key taken or skipped. The second task's encoder of a modality
shares the first task's pretrained core (`shares_core_with`), so one
Transformer per modality trains on both tasks and only the montage adapters
differ.

## Training (`physiowave/fusion/train.py`, `configs/fusion/c1_ralf.yaml`)

Per batch of one task (the two tasks' batches are interleaved in a shuffled
order, every batch of each once per epoch):

```
L = CE(fused)                                    task
  + 0.2 * sum_m CE(unimodal_m)                   on items that have m
  + 1.0 * MSE(reliability_m, 1 - corruption_m)   RALF; corruption applied with a known severity
  + 1.0 * symKL(fused | random subset, fused | all present)
```

The corruption (`physiowave.pretrain.corruption`: noise, drift, saturation,
bad channel, temporal dropout) hits each modality's raw window with
probability 0.5 before the encoders. The subset view drops present
modalities with probability 0.3 each, keeping at least one. SeizeIT2's class
imbalance is met with square-root inverse-frequency class weights.

| setting | value |
|---|---|
| fusion width | 256; 4 summary tokens per modality; 8 fusion tokens, 2 layers, 8 heads |
| epochs / batch | 30 / 32, patience 8 |
| lr | 1e-4 (fusion, pools, adapters, heads); `finetune`: the core at lr with layer decay 0.75 |
| selection | validation, complete modalities, no corruption: mean of MPDB balanced accuracy and SeizeIT2 AUPRC |

## The test matrix

The test set is scored once, at the selected checkpoint, under: complete;
each modality alone; each pair; and each of eeg~noise, eeg~bad channel,
eeg~drift, ecg~saturation, ecg~noise, emg~dropout, emg~noise at severity
0.7 on top of the complete set. A condition an item cannot meet (its only
present modality excluded) drops it from that condition; the count is in
`results.json`. `scripts/collect_fusion.py` prints one block per task with
the headline metric (MPDB balanced accuracy, SeizeIT2 AUPRC) per run and
condition.

Every finished epoch is checkpointed to `latest.pth` (weights, optimizer,
schedule, history, early-stopping state, RNG). Starting a run in a directory
that holds one continues after that epoch, so a job the walltime cut off is
finished by resubmitting the same command; `--fresh` starts over. The file
is removed once `results.json` is written.

## SeizeIT2, EEG alone: C1 against EEGPT, LaBraM and LUNA

Published EEG foundation models report on their own preprocessing, splits
and folds, so their numbers do not sit beside ours. Here they are run in
our setting instead: the official-split windows, one head, one schedule,
selection on validation AUPRC, the window-level test and the continuous
event-level scoring above -- the encoder is the only thing that differs.

| encoder | weights | input |
|---|---|---|
| C1 | `pretrain_eeg_c1_moe/best.pth` | per-channel z-score, 256 Hz |
| EEGPT large (NeurIPS 2024) | `eegpt_mcae_58chs_4s_large4E`, target encoder | mV, 256 Hz, 64-sample patches |
| LaBraM-Base (ICLR 2024) | `labram-base.pth`, student | 0.1 mV, 200 Hz, 200-sample patches |
| LUNA large / huge (NeurIPS 2025) | `LUNA_large/huge.safetensors`, encoder | per-channel z-score, 256 Hz, 40-sample patches, the centred 1000 samples |

`physiowave/fusion/baselines.py` re-implements the published encoders
with the released parameter names. The released weights load strictly,
and on random input they reproduce the authors' code: EEGPT exactly,
LaBraM to 7e-5 on outputs of size ~3, LUNA base and large to 4e-6 and
7e-6 on outputs of size ~0.7 (attention kernel rounding).

LUNA is topology-agnostic: each channel is placed by its standard_1005
coordinates (MNE head frame; a bipolar name at its electrodes' midpoint),
encoded by a NeRF-style sine bank and summarised by learned queries. Its
highest bands multiply the normalised coordinates by 2^15, so the
coordinates must be LUNA's exactly; `electrode_positions` returns them at
full precision. Its weights are CC BY-ND 4.0: fine-tuned copies stay on the
cluster and are never shared (the results sync takes no checkpoints).

**Montage.** SeizeIT2's two behind-the-ear channels are electrodes of no
model's vocabulary. Every encoder, C1 included, gets the same learned 1x1
channel adapter, EEGPT's own recipe for two-channel data (its Sleep-EDF
script): a max-norm-1 convolution from the two channels onto F7 T7 P7 /
F8 T8 P8. Every model knows those six electrodes, and they put C1 on its
E19_256 route, so its pretrained frontend transfers. The adapter trains
even when the encoder is frozen. `c1native` runs C1 the way the fusion
runs do: the raw channel names, new embedding rows, no adapter.

**Amplitude.** The stored windows are z-scored. After the adapter each
channel is z-scored again, then scaled to its model's pretraining units
for a 30 uV RMS channel (`encoders.eeg_input_uv`). EEGPT's common-average
re-reference is not applied, because it is meaningless for two bipolar
channels.

**Recipe** (`configs/fusion/seizeit2_eeg.yaml`): concat head on 4 summary
tokens, lr 1e-4 for 15 epochs (patience 5), layer decay 0.75 when
fine-tuning, no corruption augmentation, no multimodal losses. Both
`frozen` (the encoder fixed, the adapter and head train) and `finetune`
are run for each encoder.

```bash
bash scripts/download_eeg_baselines.sh          # login node, once; SHA-256 checked
sbatch scripts/slurm/cineca_seizeit2_eeg.sbatch # 8 runs, then their event-level scoring
sbatch --export=ALL,RUNS="luna_large:frozen luna_large:finetune luna_huge:frozen luna_huge:finetune" \
    scripts/slurm/cineca_seizeit2_eeg.sbatch     # LUNA, into the same tables
```

LaBraM's file comes from the authors' repository at a fixed commit. EEGPT's
authors release theirs on figshare, which does not serve scripted
downloads, so the script takes a safetensors conversion of the same file
from huggingface.co/eeg-telecom-paris/eegpt-large-official at a fixed
revision. That is a third-party mirror; its target encoder is the one
checked above. To use the authors' `.ckpt`, pass `EEGPT_CKPT=<path>` to the
job; both formats load. The tables land in
`$PW_CKPT_ROOT/fusion/seizeit2_eeg_sze_{windows,events}.txt`.

**Results** (official split, 29 test patients, 181 seizures; the event
columns are the cross-channel-gated scoring above, s@11 the sensitivity at
the SVM's 11 FA/h, AUROC and AUPR the paper's epoch-based areas, AUPRC the
window-level test set):

| encoder | mode | s@11 % | AUROC | AUPR | window AUPRC |
|---|---|---|---|---|---|
| LUNA huge | finetune | **63.4** | 0.723 | **0.0305** | 0.358 |
| C1 | finetune | 61.4 | 0.723 | 0.0107 | 0.321 |
| LUNA large | finetune | 60.9 | **0.732** | 0.0206 | **0.365** |
| LaBraM | finetune | 58.0 | 0.725 | 0.0161 | 0.345 |
| LUNA huge | frozen | 58.2 | 0.725 | 0.0240 | 0.347 |
| C1 | frozen | 54.5 | 0.715 | 0.0131 | 0.322 |
| EEGPT | frozen | 53.3 | 0.710 | 0.0146 | 0.302 |
| SVM (Bhagubai et al.) | | 71.1 | 0.700 | 0.0810 | |

Fine-tuned C1 is level with LUNA on sensitivity and AUROC (two points of
s@11 are 3-4 of 181 seizures) and ahead of LaBraM and EEGPT. It trails on
AUPR: its most confident non-seizure scores are higher, so precision at the
top of the ranking is lower. It also has the best VALIDATION AUPRC of every
encoder (0.385, LaBraM 0.370, LUNA huge 0.327) and the worst test AUPRC of
the fine-tuned four, and every C1 fine-tuning picks epoch 0: the few
validation patients flatter it. No EEG-alone encoder reaches the SVM; the
multimodal runs do (weighted_logit 78.9 %, RALF 74.4 % at <= 11 FA/h).

## Figures (`scripts/visualize_seizeit2.py`)

`fig_main` is the four-panel summary for the paper, SeizeIT2 only, in a
Nature-style layout of its own (the house Times New Roman bold, bold panel
letters, short left-aligned titles, thin spines; one hue per encoder: PhysioWave-T navy, LUNA-L teal, LaBraM violet, EEGPT red; the
encoders are named PhysioWave-T in every label, the run directories keep
`c1`). (a) sensitivity against FA/h: the main fusion run and every EEG-alone
encoder at its better of frozen / fine-tuned; (b) those encoders frozen (the
hue lightened) and fine-tuned at 11 FA/h; (c) the RALF run's reliability head
on the clean set and under each corruption, three dots per condition, an arrow
from the clean level to the corrupted modality's value; (d) the window-level AUPRC, the metric the
runs are selected on, of every modality subset of the fusion runs, the rules
labelled at their ends. The published SVM and ChronoNet are in the tables,
not on the figure. `ONLY=fig_main` draws it alone; (c) always shows the RALF
run among those given. With the nature-figure skill installed
(`~/ai-skills/nature-skills`), `PW_FIGURE_AUDIT=<its scripts dir>` runs its
panel-alignment gate at render time; its `audit_pdf_text.py` (5 pt floor) and
`audit_figure_collisions.py` pass on the exported PDF, the remaining warnings
being labels that sit outside a panel's frame on purpose. The figure needs no
cluster artefact, so it is drawn locally from the synced results into
`docs/runs/seizeit2_fusion_main/` (svg, pdf, png, metadata). Panel (b)'s
parameter counts are `ENC_PARAMS_M` in the script, the encoder alone, counted
from the architectures:

```python
from physiowave.fusion.baselines import EEGPTBackbone, LaBraMBackbone, LUNABackbone, LUNA_SIZES
from physiowave.eeg_c1.downstream import EEGC1Downstream
from physiowave.eeg_c1.routes import SLOTS_19
n = lambda m, excl=(): sum(p.numel() for k, p in m.named_parameters() if not k.startswith(excl))
n(EEGPTBackbone()), n(LaBraMBackbone()), n(LUNABackbone(**LUNA_SIZES["luna_large"]))
n(EEGC1Downstream(in_channels=19, window_samples=1024, sampling_rate=256, patch_samples=128,
                  num_classes=2, channel_names=list(SLOTS_19), route_id="E19_256", embed_dim=512,
                  depth=8, num_heads=8, mlp_ratio=4.0, norm="rmsnorm", ffn="swiglu", qk_norm=True,
                  channel_embed_dim=64), ("head",))
```
The other figures,
from the same artefacts: the cached per-second
probabilities, `events_cross.json`, `results.json`, `history.json` and one
test recording. `fig_seizure_example` (a seizure with the recorded EEG, ECG
and EMG, the probability under every modality subset, the gate and the
detection; the recording is chosen by a ranking that is kept in the
metadata), `fig_sensitivity_fa` (sensitivity against FA/h for every run and
subset, the published SVM and ChronoNet as points), `fig_modality_subsets`,
`fig_reliability` (RALF's reliability under each corruption; the AUPRC every
run loses under the same), `fig_per_patient`, `fig_learning_curves`, and per
modality `fig_signal_<m>` / `fig_window_<m>`: the example's recorded trace
with scale bars, and one 4 s encoder input window, bare signal in the
modality's colour for the framework figure and single-modality panels. Same style
and plumbing as the pretraining figures: Times New Roman bold, Okabe-Ito,
svg + pdf with editable text, `figure_metadata/` beside each.

```bash
sbatch scripts/slurm/cineca_visualize_seizeit2.sbatch       # -> docs/runs/seizeit2_fusion
git add docs/runs/seizeit2_fusion && git commit -m "seizeit2_fusion: figures" && git push
```

`EXAMPLE=sub-101_run-03` picks the recording for the example figure;
`RUN`, `COMPARE`, `EEG_RUNS` choose the runs. The framework (architecture)
figure is drawn by hand from `docs/fusion_framework_figure_brief.md`.

### C1 continued on Siena

Continued pretraining from the original run's `latest.pth`
(`configs/pretrain/eeg_c1_siena.yaml`). The Siena Scalp EEG Database
(PhysioNet, 14 adults with focal epilepsy, 512 Hz, about 22 GB) is mixed with
the seven original corpora. Siena takes a fixed 0.3 of the steps, and the
rest go to the original corpora under the original `temperature:0.5`
policy, so the encoder keeps rehearsing what it learned. An epoch is one
read of Siena at that share, and two epochs read it about twice. The
learning rate restarts at 1e-4, warms up over a tenth of an epoch from a
fresh optimizer, and decays to 1e-5. Siena is registered as a continuation
corpus (`routes.CONTINUATION_DATASETS`), so the original seven-corpus
definition is unchanged, and a run uses Siena only when its config names it.

The merge starts from the verified original manifest (`--base-dir merged`),
so the seven corpora keep exactly the shards the original run read; only
Siena's are checked. Re-merging the seven from their per-dataset manifests
brings back the shards the original `--check-shards --drop-unreadable`
left out. The first continuation attempt did that and died on a TUEG
shard whose data does not inflate.

```bash
bash scripts/download_siena.sh                                    # login node
sbatch --export=ALL,DATASET=siena --array=0-3 scripts/slurm/cineca_eeg_corpus_preprocess.sbatch
source scripts/cineca_env.sh && python scripts/build_eeg_c1_manifest.py \
    --corpus-root $PW_DATA_EEG/eeg_c1_corpus --base-dir $PW_DATA_EEG/eeg_c1_corpus/merged \
    --datasets siena --check-shards --drop-unreadable --jobs 4 \
    --out-dir $PW_DATA_EEG/eeg_c1_corpus/merged_siena
sbatch --nodes=1 --export=ALL,CONFIG=pretrain/eeg_c1_siena,INIT_FROM=$PW_CKPT_ROOT/pretrain_eeg_c1_moe/latest.pth,MANIFEST_TRAIN=$PW_DATA_EEG/eeg_c1_corpus/merged_siena/manifest_train.jsonl,MANIFEST_VAL=$PW_DATA_EEG/eeg_c1_corpus/merged_siena/manifest_val.jsonl,OUTPUT_DIR=$PW_CKPT_ROOT/pretrain_eeg_c1_siena \
    scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
sbatch --export=ALL,RUNS="c1siena:finetune:1e-5 c1siena:finetune:3e-5 c1siena:finetune c1siena:frozen" \
    scripts/slurm/cineca_seizeit2_eeg.sbatch
```

An `encoder:mode:lr` entry in `RUNS` sets `train.lr` for that run and adds
`_lr<lr>` to its directory. `c1siena:frozen` is compared with `c1:frozen`.

**Result: no gain.** Two epochs move the original corpora's validation
loss little (macro 0.2403 -> 0.2441, TUEG 0.3034 -> 0.3041) and bring
Siena's from 0.4245 to 0.4225. On SeizeIT2:

| encoder | lr | val AUPRC | s@11 % | AUROC | AUPR |
|---|---|---|---|---|---|
| C1 | 1e-4 | 0.385 | **61.4** | 0.723 | 0.0107 |
| C1 + Siena | 1e-4 (chosen on validation) | **0.397** | 59.8 | 0.722 | 0.0106 |
| C1 + Siena | 3e-5 | 0.352 | 59.2 | 0.722 | 0.0168 |
| C1 + Siena | 1e-5 | 0.356 | 54.9 | 0.719 | **0.0220** |
| C1, frozen | | 0.348 | 54.5 | 0.715 | 0.0131 |
| C1 + Siena, frozen | | 0.347 | 53.4 | 0.714 | 0.0111 |

Validation picks the continued encoder at 1e-4, and on test it is the
original within noise (1.6 points of s@11 are 3 seizures). Lower learning
rates do not help sensitivity; 1e-5 doubles AUPR but loses 6.5 points of
s@11, and validation does not choose it. The original `pretrain_eeg_c1_moe`
encoder stays the one reported.

## RALF: why it peaks early and collapses, and the sweep that follows

The frozen RALF run reaches its best validation AUPRC at epoch 5 (0.380)
and epoch 7 (0.378), then falls to 0.14 by epoch 12 with AUROC under 0.5;
the fine-tuned and from-scratch RALF runs do the same, while concat and
weighted logit at the same settings stay flat around 0.34. The training
losses say what breaks: the reliability loss, 0.04 through epoch 7, is 0.13
at epoch 8 and 0.47 by epoch 15, and the task loss follows it (0.31 to
0.58). Epoch 7 is where the schedule reaches its peak: `warmup_ratio` 0.1
of 60 epochs is six epochs of warm-up, and with `patience` 10 the cosine
never comes down. The gate is the exposure: a modality's tokens are scaled
by its reliability before the cross-attention, so a reliability head driven
off course by the full learning rate takes the fusion's input with it.
RALF's best epochs are the ones trained at a fraction of 3e-4.

Two switches on `RALFCore` address the mechanism (`configs/fusion/c1_ralf.yaml`,
both default to the original behaviour): `gate_floor` keeps a modality's
tokens at least that fraction of their size at reliability 0, and
`gate_detach` stops the task loss from reaching the reliability head through
the gate, so the head learns from its corruption target alone. The fusion
job takes a per-run variant, `fusion:mode:label:key=value,...`, so one job
runs the sweep (frozen encoders, SeizeIT2, one GPU each, about an hour an
epoch):

```bash
# RUNS goes in the environment, not in --export: sbatch splits --export at every
# comma, which cuts the key=value,key=value overrides apart (sbatch exports the
# whole environment by default, so the prefix form carries RUNS intact).
J=$(TASKS=seizeit2 TAG=szo RUNS="\
ralf:frozen:e15:train.lr=1e-4,train.epochs=15,train.patience=6 \
ralf:frozen:e15rw:train.lr=1e-4,train.epochs=15,train.patience=6,train.reliability_weight=0.3 \
ralf:frozen:e15gd:train.lr=1e-4,train.epochs=15,train.patience=6,model.gate_detach=true,model.gate_floor=0.3 \
ralf:frozen:ep12:train.epochs=12,train.patience=6" sbatch --parsable scripts/slurm/cineca_fusion_c1.sbatch)
RUN_DIRS="ralf_frozen_e15_szo ralf_frozen_e15rw_szo ralf_frozen_e15gd_szo ralf_frozen_ep12_szo" \
    sbatch --dependency=afterok:$J scripts/slurm/cineca_seizeit2_events.sbatch
```

`e15` is the schedule fix alone (peak at epoch 1.5, decayed by 15);
`e15rw` adds a lighter reliability loss; `e15gd` the detached, floored
gate; `ep12` keeps 3e-4 but lets the cosine finish, to separate the rate
from the schedule. The directories sit beside `ralf_frozen_szo`, and
`collect_seizeit2_events.py` tabulates them with it. Choose on validation
AUPRC as before; the event-level numbers follow.

**Result** (official split, same scoring; `e15siena` is `e15` with the
Siena-continued EEG encoder; `lr1e-4` did not finish). Validation AUPRC is
the selection metric, at the epoch each run selected:

| run | best epoch | val AUPRC | test AUPRC | s@11 % | s@2 % | AUROC | AUPR |
|---|---|---|---|---|---|---|---|
| `ralf_frozen_szo` (original, 3e-4, 60-epoch cosine) | 5 | 0.380 | 0.399 | **74.4** | 32.0 | **0.741** | 0.0121 |
| `e15` | 5 | **0.401** | 0.394 | 69.3 | 31.8 | 0.737 | 0.0147 |
| `e15rw` | 2 | 0.389 | **0.414** | 66.6 | 37.1 | 0.738 | **0.0232** |
| `e15gd` | 4 | 0.376 | 0.402 | 70.8 | 34.0 | 0.740 | 0.0159 |
| `ep12` | 4 | 0.359 | 0.391 | 65.8 | 33.5 | 0.738 | 0.0133 |
| `e15siena` | 5 | 0.360 | 0.384 | 70.8 | 31.4 | 0.739 | 0.0154 |
| `weighted_logit_frozen_szo` | 2 | 0.372 | 0.372 | 78.9 | 26.7 | 0.750 | 0.0132 |
| `concat_frozen_szo` | 2 | 0.370 | 0.347 | 68.5 | 24.2 | 0.735 | 0.0177 |

The schedule fix does what it was meant to: the reliability loss stays at
0.03-0.04 to the end and no variant collapses. It does not move the test
numbers beyond the spread of one run: validation prefers `e15` by 0.02 over
the original, and on test the two are level (0.394 against 0.399 AUPRC,
69.3 against 74.4 % at 11 FA/h, nine seizures of 181). `e15rw` has the
best test AUPRC and twice the original's AUPR, and the lowest s@11 of the
RALF runs. Every RALF variant (0.384-0.414) is above weighted logit (0.372)
and concat (0.347) on the complete window-level AUPRC; none reaches
weighted logit's 78.9 % at 11 FA/h. The Siena encoder changes nothing
here either.

The paper keeps `ralf_frozen_szo` as the RALF run: it is the configuration
the comparison was designed with, its selection is by validation within the
run, and the sweep is reported as its spread. `fig_main` draws the six
frozen RALF runs as a min-max band behind the original's line in (a) and
(d), and adds the weighted-logit and concat curves to (a):

```bash
R=<synced runs/fusion>   # branch cineca-results
python scripts/visualize_seizeit2.py --raw-dir /nonexistent --run-dir $R/ralf_frozen_szo \
    --compare $R/weighted_logit_frozen_szo $R/concat_frozen_szo \
              $R/ralf_frozen_{e15,e15rw,e15gd,ep12,e15siena}_szo \
    --eeg-runs $R/eeg_{c1,luna_large,labram,eegpt}_{frozen,finetune}_sze \
    --only fig_main --paper --out-dir docs/runs/seizeit2_fusion_main --format svg,pdf,png
```

## Files

| file | what |
|---|---|
| `scripts/download_fusion_data.sh` | MPDB (figshare, .rar) and SeizeIT2 (OpenNeuro S3) |
| `Fusion/mpdb_raw_prep.py`, `Fusion/mpdb_prep.py`, `Fusion/seizeit2_prep.py` | raw -> `{train,val,test}.h5` + `split.json` |
| `physiowave/fusion/prep_common.py` | rates, windows, per-modality filtering and normalisation, the HDF5 layout, an EDF reader |
| `physiowave/fusion/{data,encoders,model,train}.py` | dataset, per-task C1 encoders, the three cores, the entry point |
| `configs/fusion/c1_ralf.yaml` | the run's settings |
| `scripts/slurm/cineca_fusion_prep.sbatch`, `scripts/slurm/cineca_fusion_c1.sbatch` | preprocessing; the five-run comparison |
| `scripts/collect_fusion.py` | the window-level table |
| `Fusion/seizeit2_events.py`, `physiowave/fusion/seizure_events.py` | SeizeIT2 continuous inference and event-level scoring |
| `scripts/slurm/cineca_seizeit2_events.sbatch`, `scripts/collect_seizeit2_events.py` | the evaluation job; the event-level table with the published rows |
| `physiowave/fusion/baselines.py`, `configs/fusion/seizeit2_eeg.yaml` | EEGPT and LaBraM encoders, the channel adapter; the EEG-only recipe |
| `scripts/download_eeg_baselines.sh`, `scripts/slurm/cineca_seizeit2_eeg.sbatch` | the released weights; the eight EEG-only runs and their event-level scoring |
| `scripts/visualize_seizeit2.py`, `scripts/slurm/cineca_visualize_seizeit2.sbatch` | the SeizeIT2 figures, from the runs' artefacts |
| `docs/fusion_framework_figure_brief.md` | the specification of the RALF framework figure |
| `configs/pretrain/eeg_c1_siena.yaml`, `scripts/download_siena.sh` | C1 continued on Siena mixed with the original corpora |
| `tests/test_fusion_c1.py` | both preparations on synthetic files, the three cores, an end-to-end run on smoke encoders |

The earlier legacy path (`physiowave/models/fusion.py`, `configs/fusion/ralf.yaml`,
`physiowave/train/fusion_main.py`) is kept for its tests; it ran on synthetic
data with the pre-C1 encoders and is not what the paper reports.
