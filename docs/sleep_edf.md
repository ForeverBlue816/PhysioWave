# Sleep-EDF: 5-class sleep staging, from scratch

Everything needed to go from an empty directory to a number that sits next to
EEGPT's published row.

## The comparison, stated first

| | protocol | BalAcc | Kappa |
|---|---|---|---|
| EEGPT (NeurIPS'24) | pretrained encoder, **frozen**, 4-layer decoder head trained on top; 10 folds; scored on the validation fold it also selects on | 0.6917 ± 0.0069 | 0.6857 ± 0.0019 |
| this repo, first pass | **from scratch**, every parameter trained; held-out test set nothing selects on | — | — |

Three differences that all favour EEGPT, listed so no one has to find them
later: they start from a pretrained encoder and we do not; they train a head on
frozen features, which is a much smaller optimisation than ours; and their
reported fold is the one they early-stop on. Matching 0.69 from scratch would
be a strong result. Not matching it is the expected starting point, and it is
the argument *for* pretraining rather than against the architecture.

What is identical is the data: same source, same channels, same cropping, same
filtering, same windows, same labels, same normalisation. That part is not
approximate — `EEG/sleep_edf_finetune.py` reproduces EEGPT's
`datasets/downstream/prepare_sleep.py` step by step.

## 1. Install the reader

```bash
pip install braindecode          # pulls in mne
python -c "import pydantic; print(pydantic.VERSION, pydantic.__file__)"
```

**braindecode's dependency chain needs pydantic 2.** With pydantic 1 every
subject fails identically with `module 'pydantic' has no attribute
'TypeAdapter'`. A shared cluster environment may well pin pydantic 1; install
pydantic 2 into your own environment so it shadows the shared one, and use the
line above to confirm which one actually imports. braindecode 1.2.0 with
pydantic 2.13.4 is a combination that works.

The preparation script checks this before it starts, so the failure arrives
once with a fix in it rather than 64 times with a bare exception repr.

braindecode is used rather than mne directly because these windows have to be
EEGPT's windows. An mne-only reimplementation was tried and abandoned: whether
the raw is preloaded changes where `raw.crop` lands — 2508001 samples against
2523000 for the same recording — and `create_windows_from_events` with
`drop_last_window=False` adds one overlapping window per trial. A hand-rolled
version came out 15 windows different on subject 0 alone. Small, and fatal to
the only reason this dataset is here.

Sleep-EDF is open, so nothing needs a data use agreement. braindecode's
`SleepPhysionet` fetches Sleep-EDFx Sleep Cassette from PhysioNet through MNE
the first time it is asked. MNE caches it under `~/mne_data`; on a cluster
point that somewhere with room:

```bash
export MNE_DATA=$SCRATCH/mne_data      # ~8 GB for the full SC set
```

## 2. Where each step has to run

On Leonardo the download and the training do not belong on the same node.
Compute nodes generally have no route to the internet; login nodes do. Check
before committing to anything:

```bash
curl -sI --max-time 10 https://physionet.org | head -1
```

If that prints `HTTP/2 200` you have a route; if it hangs or fails, you are on
a node without one, and `--stage cache` has to run somewhere else.

| step | where | needs |
|---|---|---|
| `EEG/download_sleep_edf.py` | **login node** | internet, 5.8 GiB of `$MNE_DATA` |
| `--stage cache` | login node (or anywhere, once the EDFs are local) | the EDFs, no GPU |
| `--stage split` | either | the cache, no internet, no GPU |
| `finetune_sleep.sh` | **compute node** | GPUs, no internet |

### Fetching the EDFs yourself

MNE downloads one recording at a time, serially, with no resume. One command
does it concurrently from PhysioNet's S3 mirror instead:

```bash
export MNE_DATA=$SCRATCH/mne_data
python EEG/download_sleep_edf.py
```

No credentials and no extra tools -- the mirror is public and the script uses
only the standard library. It defaults to the 64 subjects EEGPT runs its folds
over, which is **5.8 GiB** rather than the 7.1 GiB of the whole cassette set;
`--subjects all` fetches everything. `--jobs` sets the number of concurrent
transfers (16 by default) and `--mirror` falls back to physionet.org if S3 is
blocked.

Files land in `$MNE_DATA/physionet-sleep-data/`, which is where MNE looks, so
the preparation step below downloads nothing.

Rerunning is safe and is how you recover from an interruption: a file whose
size already matches the mirror is skipped, a partial one is refetched, and
downloads are written to `.part` and renamed only when complete, so an
interrupted run never leaves something that looks finished.

**The verification is the part that matters.** MNE's `_fetch_one` returns a
pre-existing file without looking at it:

```python
destination = op.join(path, fname)
if op.isfile(destination) and not force_update:
    return destination, False
```

The SHA1 in its records is consulted only when pooch performs the download. A
file truncated by a dropped connection is therefore read as a short recording,
and surfaces as a subject with fewer windows -- which looks like data, not like
an error. The script hashes every file against MNE's own `SHA1SUMS` and names
what fails. `--verify-only` re-checks without refetching.

### Then preprocess

The cache stage is single-threaded per subject and 64 subjects take a while, so
run it detached:

```bash
nohup python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf \
      --stage cache > ~/sleep_cache.log 2>&1 &
tail -f ~/sleep_cache.log
```

It writes one `.npz` per subject and skips what is already there, so an
interrupted run resumes. Disjoint subject ranges are safe to run in parallel
because each writes its own file:

```bash
for r in "0,2,4,5,6,7,8,9,11,12,13,14,15,16" "17,18,19,21,22,23,24,25,26,29,30,31,32,33" \
         "34,35,37,38,40,42,44,45,46,47,48,49,51,52" "53,54,55,56,57,58,59,61,62,63,64,65,66,71" \
         "72,73,74,75,76,77,81,82"; do
    nohup python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf \
          --stage cache --subjects "$r" >> ~/sleep_cache.log 2>&1 &
done
```

Cache one subject serially first, so MNE's own setup happens once rather than
five times at once.

## 2b. Smoke test the whole pipeline first

Three subjects is enough to prove every stage works, and it takes minutes
rather than hours:

```bash
python EEG/sleep_edf_finetune.py --out-dir /tmp/sleep_smoke --subjects 0,2,4

DATA_DIR=/tmp/sleep_smoke OUTPUT_DIR=$SCRATCH/runs/sleep_smoke \
  EPOCHS=2 WARMUP_EPOCHS=0 BATCH_SIZE=16 NUM_GPUS=1 \
  bash EEG/finetune_sleep.sh
```

If that reaches a `[Test]` line, the remaining risk is only the size of the
download.

## 3. Build the HDF5 files

```bash
python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf
```

This runs two stages. The first downloads and preprocesses each subject into
`cache/subNN.npz`; it is the slow one and it is resumable — rerun the command
after an interruption and cached subjects are skipped. The second turns the
cache into `train.h5` / `val.h5` / `test.h5`.

The preprocessing, matching EEGPT exactly:

| step | value |
|---|---|
| channels | `EEG Fpz-Cz`, `EEG Pz-Oz` at 100 Hz |
| cropping | `crop_wake_mins=30` — only 30 min of wake either side of sleep |
| scaling | volts → microvolts, then a 30 Hz lowpass |
| windows | 30 s, non-overlapping, hypnogram-aligned → 3000 samples |
| labels | W / N1 / N2 / N3 / REM, AASM merge of stages 3 and 4 |
| normalising | per-channel z-score over the whole recording — see below |

`crop_wake_mins` is load bearing. Without it the recordings carry hours of
pre-bed wake and the W class swamps everything; the class balance the script
prints should come out near W 34% / N1 12% / N2 32% / N3 7% / REM 14%.

### What the z-score normalises

Not each window — the whole recording. With no `reject`, `picks` or `flat`
argument, `create_windows_from_events` returns an `EEGWindowsDataset`, whose
windows are lazy views into the Raw, so the preprocessor lands on the
recording. braindecode says so during the run:

```
UserWarning: Applying preprocessors [...] to the mne.io.Raw of an EEGWindowsDataset.
```

Measured on subject 0: per-window std ranges over **[0.306, 3.836]**, where
per-window scaling would pin it at 1.000, while all windows concatenated have
mean 0.000 and std 0.999 per channel.

This matches EEGPT rather than diverging from it. Their pinned braindecode
0.8.1 computes

```python
use_mne_epochs = (reject is not None) or (picks is not None) \
                 or (flat is not None) or (drop_bad_windows is True)
```

and they pass none of them, so their windows are lazy too.

It is also the right choice here. N3 is *defined* by high-amplitude slow waves;
z-scoring inside each window would divide exactly that amplitude away and take
the strongest cue for the rarest class with it.

**N3 is 7% of the data.** That is why balanced accuracy and Cohen's kappa are
the headline metrics and plain accuracy is not — a model that never predicts
N3 loses 7 points of accuracy and much more of either of the others.

### Splits

`--split holdout` (default) partitions the 64 subjects 60/20/20. The split is
always **by subject**: consecutive 30 s epochs of one night are near-duplicates,
so a window-level split puts a sleeper's own neighbouring epochs on both sides
and reports a number that says nothing about a new sleeper.

`--split eegpt-fold --fold K` reproduces EEGPT's 10-fold partition of the
subject list their finetune script hard-codes. Use this for the row that goes
next to their number. It differs from theirs in one place on purpose: the
validation set is carved from the *training* subjects, so the held-out fold is
only ever scored. That makes our number pessimistic relative to theirs.

```bash
# re-splitting is cheap; the cache is already built
python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf_f0 \
    --cache-dir $PW_DATA_EEG/sleep_edf/cache \
    --stage split --split eegpt-fold --fold 0
```

## 4. Train

```bash
bash EEG/finetune_sleep.sh 2>&1 | tee ~/sleep_edf.log
```

Defaults, all overridable by environment variable:

| | value | why |
|---|---|---|
| `IN_CHANNELS` | 2 | Fpz-Cz and Pz-Oz |
| `PATCH_SIZE` | 50 | 0.5 s at 100 Hz — the timescale of a sleep spindle (0.5–2 s) and a K-complex (0.5–1.5 s). 3000 samples give 60 time patches, so the folded model sees 2 × 60 = **120 tokens** where the unfolded one would see 480 |
| `EMBED_DIM` / `DEPTH` / `NUM_HEADS` | 384 / 6 / 6 | **11.02 M parameters** |
| `BATCH_SIZE` | 32 | 1.27 GiB of fp32 activations on a 64 GiB card; 64 is 2.51 GiB and equally safe |
| `EPOCHS` | 20 | |
| `SCALE_FOLD` | `dynamic` | plus `FOLD_SYNTHESIS=3`, the configuration that scored best on DB5 |
| `WAVE_INIT_MODE` | `pad` | see below |
| `WAVELET_NAMES` | `sym4 sym5 db6 sym8 db8` | every one is ≤ 16 taps and orthogonal |
| `SELECT_BY` | `balanced_acc` | the metric EEGPT reports |

## 5. Why the wavelet set changed

`WAVE_INIT_MODE=pad` centres a wavelet's native taps in the kernel and pads
with zeros. The original code stretched them by linear interpolation, which
does not produce a wavelet filter bank:

| | native | ‖h_lo‖² (want 1) | half-band cutoff (want 0.5π) | PR error (want 0, max 2) |
|---|---|---|---|---|
| sym4, interpolated to 16 | 8 | 0.396 | 0.203π | **1.999** |
| db6, interpolated to 16 | 12 | 0.646 | 0.336π | 1.979 |
| coif3, interpolated to 16 | 18 | 1.003 | 0.520π | 0.538 |
| **sym4, zero-padded to 16** | 8 | **1.000** | **0.500π** | **0.000** |
| **db8, native** | 16 | 1.000 | 0.500π | 0.000 |

PR error is `max |H_lo(ω)|² + |H_hi(ω)|² − 2|`, the power-complementary
condition. At 1.999 out of a maximum of 2 there is a band of the spectrum that
*neither* the lowpass nor the highpass passes. Stretching a filter in time
compresses it in frequency, so the cutoff moves off the half band and the pair
stops partitioning the spectrum.

Only `coif3` survived interpolation, because 18 → 16 is a mild compression
rather than a 2× stretch. `db8` and `sym8` are 16 taps natively and were never
touched by it.

This probably also explains the energy imbalance the DB5 model shows — the
approximation band carries ~84% and each detail band ~5% — since a lowpass
cutting at 0.2π still captures most of the energy of a 1/f spectrum while the
middle of the band goes to neither filter. The soft gate's
`g·approx + (1−g)·up_a` is what puts that content back, i.e. the gate has been
compensating for a broken filter bank.

`coif3` and `bior4.4` are gone from the EEG set: coif3 is 18 taps and does not
fit in a 16-tap kernel, and bior4.4 is biorthogonal, so it is not
power-complementary by construction (measured PR error 0.21 even when padded).
Neither is wrong to use — they just cannot be defended as "initialised to a
wavelet filter bank".

The EMG launcher keeps `interp` as its default so the DB5 numbers reproduce.
Whether `pad` improves them is an open question and a cheap run:

```bash
env $BASE WAVE_INIT_MODE=pad WAVELET_NAMES="sym4 sym5 db6 sym8 db8" \
  OUTPUT_DIR=$R/db5_padinit bash EMG/finetune_emg.sh
```

## 6. If it runs out of memory

It did once, and the cause was not the model. `CrossScaleCAFFN`'s cross-scale
attention runs at full temporal resolution, and `nn.MultiheadAttention`
defaults to `need_weights=True`, which materialises the whole
`[B, heads, len(q), len(kv)]` score matrix and takes the unfused path. The
weights are then thrown away on the next line.

That matrix is quadratic in the window length. At an sEMG window of 256 samples
it is 0.13 GiB and nobody notices. At a 30 s EEG epoch, the third decomposition
level attends 3000 queries against 6000 keys, so batch 64 with 4 heads needs
**17.2 GiB for that one level** — which is exactly the allocation the first
Sleep-EDF run died on.

Measured at B=8, T=3000, activations saved for backward:

| | saved | outputs agree to |
|---|---|---|
| `need_weights=True` | 2200.6 MiB | — |
| `need_weights=False` | 3.7 MiB | 2.8e-07 |

A 595x reduction with no change to the mathematics. Fixed; the whole model at
T=3000 now saves 0.66 GiB at batch 16, 1.27 at 32, 2.51 at 64.

If you hit OOM anyway, lower `BATCH_SIZE` — but check first that you have the
fix, because before it no batch size small enough to fit would have been large
enough to be worth running.

## 7. Reading the result

```bash
python scripts/inspect_checkpoint.py $OUTPUT_DIR/best_model.pth \
    --data $PW_DATA_EEG/sleep_edf/test.h5
```

Report `test_results.json`, not the log. Report **balanced accuracy and kappa**,
say which split was used, and say the model was trained from scratch.
