# ECG C1 multi-route pretraining

The EEG C1 model — per-route wavelet frontend with a dynamic ScaleFold, one
shared RoPE Transformer, C1 channel-name embedding at the token site — trained
on ECG. About 30 M parameters (512 wide, 9 deep, 8 heads: 30.08 M in total,
28.64 M in the encoder that fine-tuning inherits).

## From download to a running job

```bash
# 1. download (login node; resumable, rerun to continue)
bash scripts/download_ecg_pretrain_corpora.sh status
nohup bash scripts/download_ecg_pretrain_corpora.sh all > ~/ecg_download.log 2>&1 &   # ~730 GB

# 2. look at each corpus before processing it -- the amplitude column is the unit check
DATASET=mimic_iv_ecg INSPECT=40 bash ECG/preprocess_ecg_corpus.sh

# 3. preprocess, one array per corpus (see the sbatch header for sizes)
sbatch --export=ALL,DATASET=mimic_iv_ecg --array=0-15 scripts/slurm/cineca_ecg_corpus_preprocess.sbatch

# 4. merge whatever has finished
python scripts/build_eeg_c1_manifest.py --modality ecg \
    --corpus-root $ECG_ROOT/ecg_c1_corpus --allow-missing --check-shards --jobs 16

# 5. train (RESUME=auto to continue)
sbatch scripts/slurm/cineca_ecg_c1_moe_pretrain.sbatch
```

PhysioNet's own download endpoints were measured at ~0.04 MB/s per
connection on 2026-09-29, so Icentia11k is fetched file by file from
PhysioNet's open S3 bucket (`scripts/fetch_s3_open.py`), every fifth
70-minute segment per patient (~230 GB). MIMIC-IV-ECG has no open S3 copy
and still comes from the slow zip endpoint; it is last in `all` and can run
for days, resumably.

`ECG_ROOT` defaults to `$SCRATCH/bio/ecg`,
next to the EEG corpora. Scratch is purged after 40 days; the corpus is
rebuildable from the raw downloads, checkpoints are not and go to `$FAST`.

Progress and figures use the EEG scripts, which recognise an ECG run by its
checkpoint:

```bash
python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_ecg_c1_moe            # per route
python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_ecg_c1_moe --by dataset
python scripts/visualize_eeg_pretraining.py --run-dir $PW_CKPT_ROOT/pretrain_ecg_c1_moe \
    --checkpoint best.pth --format svg,pdf
```

A smoke run needs no data at all:

```bash
python -m physiowave.train.pretrain_main --config pretrain/ecg_c1_moe --smoke-test --max-steps 24
```

## Routes

| route     | leads | rate   | window | patch | tokens | corpora |
|-----------|-------|--------|--------|-------|--------|---------|
| `L12_500` | 12    | 500 Hz | 10 s   | 0.5 s | 240    | MIMIC-IV-ECG, CODE-15%, SPH, Georgia, MedalCare-XL |
| `L1_250`  | 1     | 250 Hz | 10 s   | 0.5 s | 20     | Icentia11k (chest patch), PulseDB (lead II) |

As on the EEG side, the route is a property of the recording, carried in the
data, never learned. Each route has its own wavelet frontend (the filters are
per lead), each sampling rate its own patch embedding and decoders, and
everything else is shared. CODE is recorded at 400 Hz and resampled to 500;
PulseDB is 125 Hz and resampled to 250 (a 0.5 s patch at 125 Hz would be 62.5
samples, so it cannot have a route of its own). Each shard records the
resampling. The two single-lead corpora share one frontend and are told apart
by their lead names: `patch1` for Icentia's chest patch, `II` for PulseDB.

**What 10 s costs.** Most corpora store 10 s records, so a record is one
window. CODE-15% does not: about 58% of its exams are 7.3 s and give no 10 s
window. Preprocessing reports them as `records_shorter_than_window`, and
Georgia's 52 five-second records are dropped the same way.
`WINDOW_SECONDS = 5.0` in `physiowave/ecg_c1/routes.py` keeps all of them,
and a 10 s record then gives two windows. The price is half the rhythm
context per window, and every corpus must be preprocessed again.

## What differs from EEG, and why

**Wavelets.** The frontend starts from db4, db6, sym4, sym8 and coif2, the
families used in the ECG literature for QRS and P/T-wave decomposition. All
are 8–16 taps, so the `pad` initialisation places each one in the 16-tap
kernel exactly. coif3, the original default, has 18 taps and would not fit.
The filter bank is shared across levels and runs at full rate, so the family
sets each filter's starting shape, not an octave per level.

**Limb leads are masked together.** I, II, III, aVR, aVL and aVF are
projections of one frontal-plane vector, and any two of them determine the
other four exactly. With per-token masking at 0.70, a masked limb-lead patch
has two or more of its siblings visible about 47% of the time, so
reconstructing it is arithmetic, not inference. A time patch of the limb group
is therefore masked as one unit, and V1–V6 are masked per token. Each part is
masked at the configured ratio, so the overall ratio does not change.
`model.lead_group_masking: false` is the EEG behaviour exactly.

**Normalisation keeps inter-lead amplitude.** Each window has each lead's mean
removed and is then divided by one standard deviation taken over all its
leads. Per-lead z-scoring, which EEG uses, would erase the ratios between
leads. In ECG those ratios carry diagnostic information: axis, voltage
criteria and inversions.

**Its own lead vocabulary.** The ECG leads have their own vocabulary
(`physiowave/ecg_c1/leads.py`) with its own hash. Adding them to
`CHANNEL_VOCAB` would have changed the EEG vocabulary's hash and made every
existing EEG checkpoint unloadable.

**Preprocessing** (`physiowave/ecg_c1/preprocess.py`):

1. Convert to mV and remove DC.
2. Notch at the corpus's mains frequency (none for simulated data).
3. 0.5 Hz high-pass, zero-phase, in second-order sections, mirror-padded.
4. Polyphase resample, then place leads on slots by name. III, aVR, aVL and
   aVF are derived from I and II when a file stores only the eight independent
   leads.

No band-pass is applied. A window is dropped if it contains a non-finite
sample, has a flat lead, or exceeds 25 mV, and drops are counted per reason.

## Corpora

| id | leads @ rate | access | notes |
|----|--------------|--------|-------|
| `mimic_iv_ecg` | 12 @ 500 | open (PhysioNet) | read from the 36 GB zip, not unpacked. Files store aVF before aVL; leads are placed by name |
| `code15` | 12 @ 400 | open (Zenodo) | zero padding stripped; the extra `exam_id 0` row per file skipped; values are mV despite the README's "1e-4 V"; patient from exams.csv |
| `sph` | 12 @ 500 | open, CC0 (figshare) | Shandong Provincial Hospital, 25,770 ECGs of 10–56 s, so a record gives 1–5 windows. Float16 mV, already filtered by the machine. Patient from `metadata.csv`; 50 Hz mains |
| `medalcare_xl` | 12 @ 500 | open (Zenodo) | simulated. One rendering of three (`noise`, via `--medalcare-variant`); the subject is the torso model `run_SXX` |
| `georgia` | 12 @ 500 | open (PhysioNet Challenge 2021) | recursive HTTP, since there is no zip or open S3 copy. No patient id |
| `icentia11k` | 1 @ 250 | open (PhysioNet) | read from the 202 GB zip; 16 random windows per 70-minute segment |
| `pulsedb` | 1 @ 125 | open (Box); MIMIC half ODbL, VitalDB half CC BY-NC-SA | ~5.2 M 10 s segments of lead II from MIMIC-III and VitalDB, resampled to 250 Hz. Read from the 26 downloaded pieces (388 GB) without joining or unpacking them: one subject's `.mat` is inflated at a time. Takes `ECG_Record` (mV), not `ECG_Raw`/`ECG_F`, which are min-max scaled. The subject key includes the half, since 28 ids occur in both |

Never pretrained on: `ptbxl`, `cpsc2018` and `chapman_shaoxing`, the
repository's ECG fine-tuning benchmarks. Preprocessing and the merge refuse
them by name.

The split is by subject hash (5% validation), so every array task puts a given
patient on the same side. Georgia carries no patient id, so each of its
records is its own subject.

The adapters were checked against real files on 2026-09-29:

- a MIMIC-IV-ECG record, read from a zip;
- Georgia `.mat` records, including a 5 s one;
- an Icentia11k segment, read from a zip;
- CODE-15%'s `exams_part17`;
- 7 SPH records;
- a PulseDB MIMIC subject and a VitalDB subject, read both unpacked and from
  deflated, split pieces.

MedalCare-XL was checked against its documented layout only. Run `INSPECT`
before an array on it.

The Norwegian athlete set, HEEDB and CODE-II were dropped on 2026-09-29. The
athletes' leads are each rescaled to full int16 range, so their amplitudes are
not physical. HEEDB and CODE-II need credentials or a request.
