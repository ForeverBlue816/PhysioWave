# sEMG C1 multi-route pretraining

The EEG C1 model, trained on surface EMG. Each route has a wavelet frontend
with a dynamic ScaleFold. One RoPE Transformer is shared across routes, and
the C1 electrode-name embedding is added at the token site.

The shared Transformer is 384 wide, 6 deep, with 6 heads: 10.63 M parameters.
The model is 12.05 M in total, 11.56 M of it in the encoder that fine-tuning
inherits.

## From download to a running job

```bash
# 1. download: bytes on a login node, then verify + unpack on a compute node
source $HOME/pw/bin/activate
bash scripts/download_emg_pretrain_corpora.sh status
STAGE=fetch nohup bash scripts/download_emg_pretrain_corpora.sh all > ~/emg_dl.log 2>&1 &   # ~1.25 TB
sbatch scripts/slurm/cineca_emg_download_finish.sbatch     # once every corpus says "downloaded"

# 2. look at each corpus before processing it -- the amplitude column is the unit check
DATASET=hyser INSPECT=20 bash EMG/preprocess_emg_corpus.sh

# 3. preprocess, one array per corpus (see the sbatch header for sizes)
sbatch --export=ALL,DATASET=emg2pose --array=0-3 scripts/slurm/cineca_emg_corpus_preprocess.sbatch
#    ... emg2qwerty 0-3, hyser 0-1, cemhsey_8x8 0-3, cemhsey_5x13 0-3, putemg 0-0

# 4. check, then merge whatever has finished
python scripts/ecg_corpus_status.py --modality emg
python scripts/build_eeg_c1_manifest.py --modality emg \
    --corpus-root $EMG_ROOT/emg_c1_corpus --allow-missing --check-shards --jobs 16

# 5. train (RESUME=auto to continue)
sbatch scripts/slurm/cineca_emg_c1_moe_pretrain.sbatch
```

`EMG_ROOT` defaults to `$SCRATCH/bio/emg`,
next to the EEG and ECG corpora. Scratch is purged after 40 days. The corpus
can be rebuilt from the raw downloads; checkpoints cannot, so they go to `$FAST`.

emg2pose, emg2qwerty and CEMHSEY's zips are each single files of 15–463 GB.
`scripts/fetch_ranged.py` fetches them in parallel 256 MB ranged pieces,
written straight into place. A login node that kills the transfer costs only
the unfinished pieces, and a rerun fetches the rest. Zenodo serves CEMHSEY at
about 0.6 MB/s per connection, so its 19 zips share 24 connections by default
(`CEMHSEY_JOBS`).

The download runs in two stages. Login nodes kill processes that use a lot
of CPU; the first full run had its downloader SIGKILLed three times. Compute
nodes have no internet.

- `STAGE=fetch`, on a login node, only moves bytes.
- `STAGE=finish`, a compute-node job with no network, does the rest: it checks
  each file is whole, verifies CEMHSEY's md5s, unpacks GESTURE and emg2qwerty,
  reads emg2pose's tar headers, and marks each corpus complete.

Nothing loops to restart after a kill. Rerunning is a person's decision, and
if kills keep coming, CINECA's data-transfer service is the route meant for
this volume.

Progress and figures use the EEG scripts, which recognise an sEMG run by its
checkpoint:

```bash
python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_emg_c1_moe --by dataset
python scripts/visualize_eeg_pretraining.py --run-dir $PW_CKPT_ROOT/pretrain_emg_c1_moe \
    --checkpoint best.pth --format svg,pdf
```

A smoke run needs no data at all:

```bash
python -m physiowave.train.pretrain_main --config pretrain/emg_c1_moe --smoke-test --max-steps 24
```

## Routes

All routes use a 1 s window and 0.125 s patches at 2000 Hz.

| route      | channels | tokens | corpora |
|------------|----------|--------|---------|
| `W16_2000` | 16       | 128    | emg2pose, emg2qwerty (each wrist a sample) |
| `A24_2000` | 24       | 192    | putEMG (three forearm rings of 8) |
| `G64_2000` | 64       | 512    | one high-density grid: Hyser's 8x8; CEMHSEY's 8x8 and 5x13 |

**One grid per sample.** Hyser records four 8x8 grids (256 channels) and
CEMHSEY five grids (320 channels). Each grid is its own sample, the way
emg2qwerty's two wrists are two samples.

Taking a whole recording as one route would cost far more. At 256 and 320
channels, the frontend's cross-channel FFN alone would be 9.8 M and 15.3 M
parameters, and every window would be 2,048 or 2,560 tokens. One grid costs
0.64 M parameters and 512 tokens. It keeps what HD-sEMG is for: 64
neighbouring electrodes over one muscle group.

**2000 Hz** is the native rate of emg2pose and emg2qwerty, which are most of
the corpus, so they are not resampled. Hyser and CEMHSEY (2048 Hz) and putEMG
(5120 Hz) are resampled down. sEMG power lies below about 500 Hz, well inside
the 1000 Hz Nyquist limit.

## What differs from EEG and ECG, and why

**Wavelets.** The frontend starts from db2, db4, db7, sym4 and sym5. The
Daubechies filters suit the sharp motor-unit action potentials, and the
symlets suit near-symmetric ones; sEMG denoising and feature-extraction work
most often uses these families. All are 4–14 taps, so the `pad`
initialisation places each one exactly.

**Signal settings** (`EMG_SETTINGS` in `EMG/preprocess_emg_corpus.py`):

- a 20 Hz high-pass, the usual sEMG cut for movement artefact and electrode
  drift;
- a mains notch with three harmonics;
- a window is dropped for a channel flatter than 0.5 µV, for any sample above
  25 mV, or for a non-finite sample.

The rest of the pipeline is ECG's, unchanged: one scale per window across all
channels, so a window keeps which electrodes were active relative to which.

**Its own electrode vocabulary** (`physiowave/emg_c1/electrodes.py`), with
its own hash:

| names | meaning |
|-------|---------|
| `band01..16` | the wristband's electrodes; the same in emg2pose and emg2qwerty |
| `putemg_r{1-3}e{1-8}` | putEMG's rings |
| `hyser_r{r}c{c}` | a position in a Hyser 8x8 grid |
| `cemhsey_r{r}c{c}` | a position in a CEMHSEY 8x8 grid |
| `cemhsey13_r{r}c{c}` | a position in a CEMHSEY 5x13 grid (no electrode at r5c1) |

CEMHSEY's rows are mapped to positions with the channel maps in the dataset's
own `PreProcess.m`.

## The numerical guard, and resuming after a divergence

The first 60k-step run diverged at epoch 39: one hyser batch produced a
gradient norm of 4e4, a later one inf, and `clip_grad_norm_` with an inf
norm scales every gradient by NaN. Under bf16 there is no GradScaler to
refuse such a step, so the weights became NaN and the job trained on them
for five more epochs.

The trainer (`physiowave/eeg_c1/train.py`) now

- skips a step whose pre-clip gradient norm is not finite or is above
  `train.skip_step_grad_norm_above` (100 here; `null` keeps only the
  non-finite check). The weights are untouched and the schedule advances. DDP
  has already averaged the gradients, so every rank skips the same steps.
  `train/skipped_steps` and `train/max_grad_norm` are in
  `metrics_epoch.jsonl`, and each skip is a `SKIPPED step` line in the log;
- stops the run after `train.max_skipped_steps_per_epoch` (50) skips in one
  epoch, which is a model gone bad rather than a bad batch;
- refuses to resume from a checkpoint whose weights are not finite, and never
  overwrites `latest.pth` with such weights.

To continue a run that diverged before this guard existed, resume from the
last checkpoint written before the divergence. `best_raw.pth`,
`best_total.pth` and `best_macro_total.pth` are full trainer states, and each
one's `epoch` field says where it stands:

```bash
sbatch --export=ALL,RESUME=$PW_CKPT_ROOT/pretrain_emg_c1_moe/best_raw.pth \
    scripts/slurm/cineca_emg_c1_moe_pretrain.sbatch
```

The resume starts at the epoch after that checkpoint's, replays its sampler
state, and moves the metric rows the diverged continuation wrote for those
epochs to `superseded/resumed/`, so the curve keeps one row per epoch.

## Corpora

| id | route | source | notes |
|----|-------|--------|-------|
| `emg2pose` | W16 | Meta, 463 GB tar | 193 users, 25,253 files (one per wrist per stage). HDF5 `emg2pose/timeseries["emg"]`, µV. **Read in place from the tar** (uncompressed, so each file is a contiguous byte range the listing records) |
| `emg2qwerty` | W16 | Meta, 308 GB tar.gz | 108 users, ~1,135 sessions; `emg_left`/`emg_right`, µV; each wrist a record. Unpacked (gzip cannot be read in place); the tar.gz is then removed |
| `hyser` | G64 | PhysioNet hd-semg 2.0.0, open S3 | 20 subjects × 2 sessions. Only the `*_raw_*` records (76 GB); the `preprocess` copies are the authors' band-passed versions. Channels `XX-i-j` → grid XX, `hyser_r{i}c{j}`. **The headers say V; the values are mV** (0.4 V peak-to-peak is not a surface signal), so the unit is set, not read |
| `cemhsey_8x8` | G64 | Zenodo 15077957 + 15070187 | channels 1–192: three 8x8 grids (10 mm), proximal forearm. `data_sEMG`, 320 × N, mV, 2048 Hz. GRASP (13 subjects) and GESTURE (6 subjects) over 11 days |
| `cemhsey_5x13` | G64 | the same files | channels 193–320: two 5x13 grids (8 mm), distal forearm |
| `putemg` | A24 | putEMG WebDAV share | 44 subjects, 712 records; pandas `frame_table`, columns found by name. Raw 12-bit counts × 5/4096 × 1000/200 mV; 5120 Hz |

**CEMHSEY is two corpora because a corpus's shards carry one electrode map,
and its two grid types place their 64 electrodes differently.** Both read the
same files. GRASP's zips use deflate and are read member by member, without
unpacking. GESTURE's zips use deflate64, which Python's zipfile cannot
inflate, so the download script unpacks them next to themselves. GRASP and
GESTURE recruited separately, so their subject numbers name different people;
the subjects are `grasp_S4` and `gesture_S4`. The four trials the authors list
as failed recordings are left out.

**Never pretrained on:** NinaPro DB5 and EPN-612, the downstream benchmarks.
Preprocessing and the merge refuse them by name. NinaPro DB6/7/8 are not
used.

The split is by subject hash (5% validation), the same as on EEG and ECG.

The readers were checked against real files on 2026-10-06:

- two emg2pose files, read from a tar;
- an emg2qwerty session;
- three Hyser raw records (1dof, dynamic, maintenance);
- a CEMHSEY GRASP trial, both unpacked and read from a deflate zip;
- two putEMG records (gesture and force).

Every one maps fully onto its slots, and QC keeps every window. The
median-channel peak-to-peak is 0.35–1.0 mV, about 3–5 mV for putEMG with its
baseline. A 2-rank training run on the shards they produce covers all three
routes.

## Licences

As each source states them:

| corpus | licence |
|--------|---------|
| emg2pose | CC BY-NC-SA 4.0 |
| emg2qwerty | CC BY-NC-SA 4.0 in its LICENSE file; its README says CC BY-NC 4.0 |
| CEMHSEY | CC BY 4.0 on Zenodo |
| Hyser | ODC-By 1.0 |
| putEMG | CC BY-NC 4.0 |

The non-commercial terms carry over to checkpoints trained on these corpora.
