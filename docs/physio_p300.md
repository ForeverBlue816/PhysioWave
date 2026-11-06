# PhysioP300: binary P300 detection, from scratch

Everything needed to go from an empty directory to a number that sits next to
EEGPT's published row.

## The comparison, stated first

| | protocol | BalAcc | Kappa | AUROC |
|---|---|---|---|---|
| EEGPT (NeurIPS'24) | pretrained encoder, **frozen**, linear probe on top; LOSO; scored on the held-out subject it also selects on | 0.6502 ± 0.0063 | 0.2999 ± 0.0139 | 0.7168 ± 0.0051 |
| LaBraM (ICLR'24) | linear probe | 0.6477 ± 0.0110 | 0.2935 ± 0.0227 | 0.7068 ± 0.0134 |
| BENDR | full fine-tune | 0.6114 ± 0.0118 | 0.2227 ± 0.0237 | 0.6588 ± 0.0163 |
| BIOT (NeurIPS'23) | linear probe | 0.5485 ± 0.0325 | 0.0968 ± 0.0647 | 0.5308 ± 0.0333 |
| this repo, first pass | **from scratch**, every parameter trained; held-out subject nothing selects on | — | — | — |

Read the absolute numbers before reading the gaps: a kappa of 0.30 is a *hard*
task. Single-trial P300 detection at 1:5 imbalance is close to the ceiling of
what one flash can support, which is why speller systems average over many
repetitions. Do not expect Sleep-EDF-shaped numbers here.

Three differences that all favour EEGPT, listed so no one has to find them
later: they start from a pretrained encoder and we do not; they train a linear
probe on frozen features, which is a far smaller optimisation than ours; and
the fold they report is the one they train against. Matching 0.65 / 0.30 / 0.72
from scratch would be a strong result. Not matching it is the expected starting
point, and it is the argument *for* pretraining rather than against the
architecture.

What is identical is the data: same source, same channels, same epoching, same
filtering, same resampling, same labels. `EEG/physio_p300_finetune.py`
reproduces EEGPT's `datasets/downstream/prepare_PhysioNetP300.py` step by step,
with three departures documented in that file's header and repeated below.

## What their code actually does

Worth knowing before comparing, because two details differ from their paper.

**There is no test set.** `downstream/linear_probe_EEGPT_PhysioP300.py` builds
only `train_dataset` and `valid_dataset`, and passes `callbacks = [lr_monitor]`
to the Lightning trainer — no `ModelCheckpoint`, no `EarlyStopping`. It trains
100 epochs and the reported number comes off the validation subject. Same
pattern as their Sleep-EDF script.

**Subject 1 is never preprocessed.** The paper says subjects 8, 10 and 12 are
dropped and the remaining nine kept. Their LOSO loop agrees:

```python
all_subjects = [1,2,3,4,5,6,7,9,11]
```

But `prepare_PhysioNetP300.py` writes only

```python
for sub in [2,3,4,5,6,7,9,11]:
```

— eight subjects. The fold that holds out subject 1 has nothing to hold out.
This repo follows the paper and includes subject 1; `--eegpt-subjects`
reproduces their eight if you want to see the difference.

## 1. Install the reader

The preparation needs MNE and nothing else — no braindecode this time, because
the epoching is driven by the EDF's own annotations rather than by a dataset
class.

```bash
source $HOME/pwprep/bin/activate
PW_VARS_ONLY=1 source scripts/cineca_env.sh
python -c "import sys, mne; print(sys.prefix, mne.__version__)"
```

**`PW_VARS_ONLY=1` matters.** `scripts/cineca_env.sh` does two jobs: export the
`PW_*` paths, and set up the *training* environment by loading the cineca-ai
module and activating `$HOME/pw`. Steps 2 and 3 need the first and nothing from
the second — no CUDA, no torch — and the second gets in the way, because
activating `$HOME/pw` replaces `pwprep`. Plain `source scripts/cineca_env.sh`
flips the prompt from `(pwprep)` to `(pw)` and `import mne` then fails.

`PW_VARS_ONLY=1` touches neither the modules nor the virtualenv, so the order
above is not load-bearing and the `cineca-ai/4.3.0` module error cannot appear.
Step 4 sources the script normally, because training is what the module and
`$HOME/pw` are for.

The `sys.prefix` in the check must end in `pwprep`. Verified against mne 1.12.1.

## 2. Download

2.19 GiB, 245 EDF recordings, twelve subjects at roughly twenty runs each.
Small enough that there is no reason to fetch a subset.

```bash
source $HOME/pwprep/bin/activate
PW_VARS_ONLY=1 source scripts/cineca_env.sh

python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci
```

If `--dest` comes out as `/erpbci`, `$PW_DATA_EEG` was empty: your shell expands
it before the script starts, so sourcing the environment afterwards is too late.
The script says so rather than failing on a permission error at the filesystem
root.

Run it where there is internet — on Leonardo that is a login node; compute
nodes have no route out and the script says so rather than hanging.

Files are pulled concurrently from PhysioNet's S3 mirror and then checked
against `SHA256SUMS.txt`, which the archive ships itself. That check matters:
a recording truncated by a dropped connection is not an error downstream. MNE
reads it as a short run, the preparation step writes fewer epochs, and the
result looks like data. Rerunning skips whatever is already complete, so an
interrupted download is resumed rather than restarted.

`--mirror` falls back to physionet.org if S3 is unreachable.

## 3. Preprocess

```bash
python EEG/physio_p300_finetune.py \
    --edf-dir $PW_DATA_EEG/erpbci \
    --out-dir $PW_DATA_EEG/p300_f0 \
    --fold 0
```

Two stages. `--stage cache` decodes each subject into one `.npz`; `--stage
split` turns those into HDF5 for one LOSO fold. The default runs both, and the
cache lives under `--edf-dir` so **the nine folds share one decode** — only the
split re-runs per fold. Budget roughly 3 s per run, ~20 runs per subject, two
subjects in parallel by default (`--jobs`).

### If the decode gets killed on the login node

Observed: `BrokenProcessPool` on two of nine subjects, which means a worker was
**killed** rather than that it raised — there is no traceback because no Python
exception happened. On a login node that is the per-user cgroup. The peak is
`mne.Epochs(preload=True)` holding a whole run at 2048 Hz in float64, about
0.5 GiB per worker before the filter and resample make their copies.

Three ways out, in order of preference:

```bash
# 1. Decode on a compute node. It needs no internet, only the EDFs.
srun -N1 -n1 -c8 -t 0:30:00 -A <account> -p <partition>      $HOME/pwprep/bin/python EEG/physio_p300_finetune.py      --edf-dir $PW_DATA_EEG/erpbci --out-dir $PW_DATA_EEG/p300_f0 --stage cache

# 2. Stay on the login node, one subject at a time
python EEG/physio_p300_finetune.py --edf-dir ... --out-dir ... --stage cache --jobs 1
```

Either way **nothing is redone**: subjects already cached are skipped, so a
partial run is resumed rather than restarted. The script also retries a killed
subject serially on its own before giving up, which recovers most of these
without intervention.

**The split stage refuses to run on an incomplete cache.** If any subject has
no `.npz`, nothing is written and the missing ids are named. A split assembled
from whatever happened to be cached is the failure this file exists to prevent:
it loads, it trains, and it reports a number for a corpus that is quietly
missing subjects — and with LOSO, every fold would then be scored on a
different corpus. `--allow-missing` overrides it, which is only ever right for
a pipeline smoke test.

### The preprocessing, matching EEGPT

| step | value |
|---|---|
| source | PhysioNet erpbci 1.0.0, EDF, 2048 Hz |
| channels | the 58 EEGPT's encoder consumes, in their order |
| epochs | `tmin=-0.1 s`, `tmax=2.0 s`, from the EDF's annotations |
| filtering | IIR 0–120 Hz, applied **after** epoching |
| resampling | 256 Hz, after filtering |
| labels | 1 when the flashed row/column contains the run's target character |

The label rate is exactly 1/6. A Donchin speller flashes 6 rows and 6 columns
per repetition and 2 of the 12 contain the target. Measured on `s01/rc01`: 240
epochs, 40 positive, 0.167.

The target character comes from the run's own annotation — `#TgtM_RC01_SOA63`
means the target is `M` — and the flash codes name their six characters
directly (`ABCDEF`, `AGMSY5`, …), so the label is `target_char in code`.

### Three departures, and why

**1. The window is 512 samples, not 538.** Their epoch is 2.1 s at 256 Hz =
538 samples. 538 = 2 × 269 with 269 prime, so no sensible patch size divides
it, and `patchify` in [model.py:284](../model.py#L284) asserts
`T % patch_size == 0`. The array is cropped to the first 512 samples, i.e.
−0.1 s to 1.9 s. The P300 is a 250–500 ms deflection; the tail is context only.
512 / 64 = 8 time patches, and 64 samples is 250 ms — the same patch duration
EEGPT uses (their `d=64` at 256 Hz).

**2. A per-channel z-score is applied.** They scale volts by `1e3` and stop,
because their classifier begins with a learnable per-channel scaling factor
(their "adaptive spatial filter", Appendix C.2.5) that absorbs the units. Ours
has no such layer, and `1e3` leaves the signal at std **0.0265** — measured on
`s01/rc01` — which is not a scale to hand a wavelet filter bank.

The z-score is over the **run**, not over each epoch. The P300 is defined by
its amplitude relative to the non-target response, and per-epoch scaling would
divide exactly that away. Same choice, same reason, as the recording-level
z-score in `sleep_edf_finetune.py`.

**3. Subject 1 is included.** See above.

### The split

`--fold K` makes `subjects[K]` the **test** set and carves a validation subject
out of the remaining eight, so train/val/test is 7/1/1 of the nine. EEGPT holds
out one subject, calls it validation, and reports it. Here the held-out subject
selects nothing.

```
LOSO fold 0 (test subject 1), by subject:
  train  7 subjects   ...  nontarget=...(83.3%)  target=...(16.7%)
  val    1 subjects   ...
  test   1 subjects   ...
```

## 4. Train

Back in the **training** environment, not `pwprep`:

```bash
deactivate
source scripts/cineca_env.sh

FOLD=0 DATA_DIR=$PW_DATA_EEG/p300_f0 bash EEG/finetune_p300.sh
```

All nine folds — this is the number to report, not fold 0:

```bash
for k in $(seq 0 8); do
  # cache is shared; only the split re-runs
  python EEG/physio_p300_finetune.py --edf-dir $PW_DATA_EEG/erpbci \
      --out-dir $PW_DATA_EEG/p300_f$k --stage split --fold $k
  FOLD=$k DATA_DIR=$PW_DATA_EEG/p300_f$k bash EEG/finetune_p300.sh
done
```

Between-subject variance is the dominant term on ERP tasks — EEGPT's own
±0.0139 on kappa is across folds. One fold is a pilot, not a number.

### Knobs that matter here

| variable | default | why |
|---|---|---|
| `IN_CHANNELS` | `62` | must match what the preprocessing wrote — `--channels 62` is the converter's default; a split made with `--channels 58` needs `IN_CHANNELS=58` |
| `PATCH_SIZE` | `64` | 250 ms at 256 Hz, EEGPT's `d`; 512/64 = 8 time patches |
| `NUM_CLASSES` | `2` | binary |
| `CLASS_WEIGHT` | `balanced` | see below — **do not turn this off** |
| `SELECT_BY` | `auroc` | what EEGPT monitors for binary tasks (their Appendix D) |
| `SCALE_FOLD` | `dynamic` | see below |
| `EPOCHS` | `30` | |

**`CLASS_WEIGHT=balanced` is not optional at 1:6.** Unweighted, argmax at 0.5
collapses onto the majority: plain accuracy reports 0.833 and balanced accuracy
sits at 0.5. The weights come out `[0.333, 1.667]` — inverse frequency,
normalised to mean 1 so the loss stays on the same scale as an unweighted run
and a tuned learning rate does not have to move.

**The scale fold earns its keep here more than on Sleep-EDF.** With 58 channels
and `max_level=3`, the unfolded spectrogram is (3+1) × 58 = 232 rows, which at
8 time patches is **1856 tokens**. The fold collapses the four scales back onto
58 rows: **464 tokens**. Sleep-EDF's 2-channel montage never made that
difference visible.

## 5. Read the result

```
$OUTPUT_DIR/test_results.json   the number to report
$OUTPUT_DIR/history.json        per-epoch validation curve
$OUTPUT_DIR/best_model.pth      inspect with scripts/inspect_checkpoint.py
```

Report `test_*`, not the validation line. The validation subject is what
`SELECT_BY` optimises against; the test subject is the one nothing touched.

## Two fixes this task forced

Both are in `finetune.py` and both affect any binary task, not just this one.

**AUROC was silently NaN on every binary task.**
`roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')` routes on
the shape of `y_true`: with two classes present sklearn calls the *binary*
path, which rejects a 2-column `y_score` with `y should be a 1d array`. The
surrounding `except Exception` turned that into `float('nan')`. With
`--select_by auroc`, NaN compares False against everything, so **no checkpoint
would ever have been saved after initialisation**. Fixed to take `y_prob[:, 1]`
when there are two columns; the multi-class `ovo` branch is unchanged, so
Sleep-EDF numbers are unaffected.

**Selection is now NaN-safe.** AUROC is genuinely undefined when a split holds
one class, which is possible on a per-subject fold. That epoch is now skipped
with a printed reason rather than poisoning selection for the rest of the run.

`--class_weight` is new and defaults to `none`, so existing runs are unchanged;
verified that `weight=None` reproduces the previous loss bit-for-bit, and that
with `--label_smoothing 0` the weighted form matches
`nn.CrossEntropyLoss(weight=...)` to 1e-6.
