# EEG

EEG-specific data preparation and launch scripts, alongside `ECG/` and `EMG/`.

| File | Purpose |
|------|---------|
| `tueg_pretrain.py` | TUH EEG / Siena EDF → HDF5 windows for pretraining |
| `tuab_finetune.py` | TUAB EDF → labelled HDF5 for normal-vs-abnormal fine-tuning |
| `download_sleep_edf.py` | Fetch Sleep-EDFx SC from PhysioNet's S3 mirror in parallel, and verify it |
| `sleep_edf_finetune.py` | Sleep-EDFx SC → labelled HDF5 for 5-class sleep staging, on EEGPT's preprocessing |
| `download_p300.py` | Fetch PhysioNet ERP-BCI (erpbci) from the S3 mirror in parallel, and verify it |
| `physio_p300_finetune.py` | erpbci → labelled HDF5 for binary P300 detection, on EEGPT's preprocessing |
| `pretrain_eeg.sh` | Launch EEG pretraining (extension pipeline) |
| `finetune_eeg.sh` | Launch downstream fine-tuning (shared `finetune.py`) |
| `finetune_sleep.sh` | Launch Sleep-EDF sleep staging — see [docs/sleep_edf.md](../docs/sleep_edf.md) |
| `finetune_p300.sh` | Launch PhysioP300 detection — see [docs/physio_p300.md](../docs/physio_p300.md) |
| `download_eegpt_benchmarks.py` | Fetch BCIC-IV-2a / 2b from BNCI Horizon 2020, resumable and size-verified |
| `download_kaggle_ern.sh` | Fetch KaggleERN via the Kaggle API, in an isolated venv |
| `bcic_iv2_finetune.py` | BCIC-IV-2a / 2b → labelled HDF5, on the pretraining preprocessing |
| `kaggle_ern_finetune.py` | KaggleERN → labelled HDF5, on the pretraining preprocessing |
| `eegpt_bench_common.py`, `eegpt_bench_lib.sh` | What the two converters share: the pretraining preprocessing, and where splits live |
| `finetune_eegpt_bench.sh` | One EEGPT-benchmark run (ft / scratch). All three at once: `bash scripts/run_eegpt_bench.sh` — see [docs/eegpt_benchmarks.md](../docs/eegpt_benchmarks.md) |

## Why EEG is not just "ECG with more channels"

The ECG and EMG folders drive the legacy top-level `pretrain.py`. EEG pretraining
instead goes through `physiowave.train.pretrain_main`, because it needs three
things the legacy path has no representation for:

- **A montage, not a channel count.** Channels are electrodes at known scalp
  positions. `physiowave/data/montages.py` supplies template 10-20/10-10
  coordinates; a recording without positions automatically disables the
  spherical-spline Laplacian branch instead of silently running on zeros.
- **Reference invariance.** An EEG montage is a choice of reference, not a
  property of the brain. `configs/pretrain/eeg.yaml` turns on the
  reference-consistency objective so representations do not encode the montage.
- **A variable, unordered channel set.** TARE channel compression handles
  19-channel and 64-channel recordings with the same encoder.

Model and objective settings therefore live in `configs/pretrain/eeg.yaml`, not
in the shell script; the script only supplies paths and cluster settings.

## Pipeline

```bash
# 1. Build the pretraining corpus (needs a local copy; nothing is downloaded)
python EEG/tueg_pretrain.py --dataset tueg  --root /data/tuh_eeg --out-dir ./data/eeg_pretrain
python EEG/tueg_pretrain.py --dataset siena --root /data/siena  --out-dir ./data/eeg_pretrain

# 2. Pretrain
bash EEG/pretrain_eeg.sh

# 3. Robustness evaluation of the pretrained encoder
bash scripts/run_tpami.sh eval --suite eeg

# 4. Downstream task
python EEG/tuab_finetune.py --root /data/tuh_abnormal/v3.0.0 --out-dir ./data/tuab
bash EEG/finetune_eeg.sh
```

## Output format

Both preparation scripts write HDF5 that `physiowave.data.manifest.build_manifest`
can scan directly:

- `data` — `(N, C, T)` float32 windows
- `label` — `(N,)` int64 (fine-tuning files only)
- `channel_names` — `(C,)` fixed-length bytes; **without this the SSL spatial
  branch has no electrode positions and is disabled for the whole corpus**

Filtering and normalisation are *not* baked into the files. Only resampling is,
because windows are cut in samples and a common rate is what makes a window a
fixed duration. Notch, bandpass and z-scoring are applied at load time from
`data.preprocess` in the config, so one cached corpus serves several filter
settings — see `configs/data/eeg_real.yaml`.

## Extra dependency

EDF reading needs `mne`, which is not in `requirements.txt` (the ECG scripts
similarly need `wfdb`):

```bash
pip install mne
```

## Data access

`tueg`, `tuab`, `tuar`, `tusl` and `seizeit2` require a signed data use
agreement and are never downloaded automatically — point `--root` at a copy you
already obtained. `siena` and `bci_iv_2a` are open. The full list, with
sampling rates, montages and which corpora carry coordinates, is in
`physiowave/data/registry.py`.
