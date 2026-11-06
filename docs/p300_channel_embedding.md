# PhysioP300 channel embedding

The same six variants as the Sleep-EDF ablation, on the montage the question
was actually about.

## Why this dataset and not the other one

Sleep-EDF has two bipolar derivations and never varies them, and the model's
existing 2-D position embedding already distinguishes two rows. It is the least
favourable place a channel code could be tested — which was the point of
running it there first, but it cannot answer whether channel identity matters.

erpbci has **58 electrodes** in the montage this ablation ran on. That is
enough for an identity to be worth learning and enough for a geometry to be
worth having.

> **The default montage is now 62**, not 58 — every electrode erpbci records
> that `E64_256` has a slot for, so a C1 pretrained frontend transfers into it
> whole. Every number on this page was measured at 58 (`--channels 58`,
> `IN_CHANNELS=58`) and is left at 58 rather than restated, because the rows
> are a record of runs that happened. A re-run at 62 is a different input.

## Monopolar, and what that changes

This montage is monopolar: each channel is one electrode against **one common
reference**, not a difference of an electrode pair. That is the opposite of Sleep-EDF, and the
encoder has to describe both without pretending either is the other.

A monopolar channel has a position and **no direction**. Both endpoint indices
point at the same electrode, so in the encoder

```
mid  = 0.5 * (phi(A) + phi(A)) = phi(A)      the electrode's position
dirn =        phi(A) - phi(A)  = 0           exactly zero, not small
```

`dir_proj` therefore cannot influence the code however it is weighted — a test
multiplies its weights by 1000 and asserts the output is bit-identical. What is
left is a position encoding, and the code carries a `monopolar_token` rather
than the `bipolar_token`, so a model that has seen both kinds can tell which it
is looking at.

**No reference position is invented.** erpbci's ear electrodes are dropped
before preprocessing and `standard_1020` has no scalp coordinate for them, so
subtracting a made-up reference would be fabricating geometry to make a formula
look symmetric.

So `C2` means *signed derivation* on Sleep-EDF and *position* on P300. Same
encoder, degenerating honestly. Say which one when quoting a result.

## The confound to state before looking at the numbers

`CHANNELS_58` is in EEGPT's **topographic** order, and the 2-D position
embedding indexes rows by that order. The model therefore already has a
one-dimensional walk over the scalp, for free, in every variant including C0.
What C2 adds is genuine 3-D position.

A null result here means *"the topographic row order was already enough"*, not
*"position does not matter"*. Shuffling the row order in a further arm would
separate those, and is not part of this ablation.

## Vocabulary

`CHANNEL_VOCAB` in `channel_embedding.py` is **append-only**: the ids are stored
in every HDF5 and indexed by every checkpoint's embedding table, so inserting a
name anywhere but the end silently relabels every channel of every file written
before. The 58 monopolar names were appended after the bipolar ones; `Fpz-Cz`
and `Pz-Oz` keep ids 2 and 3.

`"Cz"` and `"Fz-Cz"` are different entries with different ids, on purpose. One
is a potential at a site, the other a difference of two, and a shared id would
tell the model they are the same measurement.

A channel that is not in the vocabulary is **refused** at preparation time, not
mapped to `UNK`: 58 channels all landing on one `UNK` row would make `id` encode
"some channel" 58 times, and that null would look like a measurement.

## Running it

### 1. The decode cache, once

Needs `mne`, so it runs under `$HOME/pwprep`:

```bash
source $HOME/pwprep/bin/activate
PW_VARS_ONLY=1 source scripts/cineca_env.sh
python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci
python EEG/physio_p300_finetune.py --edf-dir $PW_DATA_EEG/erpbci \
    --out-dir $PW_DATA_EEG/p300_channel/fold0 --stage cache --jobs 2
```

Do not train from the shell that prepared the data — `PW_VENV` is exported and
`srun` inherits it. Start a fresh shell, or `unset PW_VENV`.

### 2. The ablation

```bash
# plumbing check: two variants, one fold
VARIANTS=C0,C4 FOLDS=0 SEEDS=42 NUM_GPUS=4 EPOCHS=2 \
    bash EEG/run_p300_channel_ablation.sh

# the real thing: nine LOSO folds
VARIANTS=C0,C1,C2,C3,C4,C5 FOLDS=0,1,2,3,4,5,6,7,8 SEEDS=42 \
    bash EEG/run_p300_channel_ablation.sh
```

Nine folds rather than one, because between-subject variance is the dominant
term on an ERP task — EEGPT's own ±0.0139 on kappa is across folds, not seeds.
A single fold on this task says almost nothing.

The runner builds each fold's HDF5 once, shares it across every variant, and
removes it afterwards (`KEEP_SPLITS=1` to keep). The decode cache stays.

### 3. Collect

```bash
python scripts/collect_channel_ablation.py \
    $PW_CKPT_ROOT/p300_channel_ablation --classes nontarget,target
```

The number that matters is the **paired** delta against C0 within each
(fold, seed): the fold *is* a subject, and subject-to-subject variation is
larger than the effect being looked for, so an unpaired difference of means is
mostly a difference of subjects.

## Hyper-parameters

In `EEG/run_p300_channel_ablation.sh`, not in the caller's shell:

```
EPOCHS 15   WARMUP_EPOCHS 2   BATCH_SIZE 64   LR 3e-4   MIN_LR 1e-6
WEIGHT_DECAY 1e-2   DROPOUT 0.1   HEAD_DROPOUT 0.1   LABEL_SMOOTHING 0.1
FOLD_KL 1e-3   SELECT_BY auroc   IN_CHANNELS 58
```

`SELECT_BY auroc` because the task is 1/6 positive by construction and accuracy
on it is nearly uninformative; it is also what EEGPT monitors for binary tasks.

## What this can and cannot conclude

A stable C2/C4 gain would license:

> Electrode-position-aware channel metadata improves a 58-channel ERP model,
> beyond what a topographically ordered position embedding already provides.

It would **not** license a claim about cross-montage generalisation — every run
sees the same 58 electrodes — nor about reference invariance, which a learned
map of a common-referenced signal does not have.

## Compatibility

* **Old P300 HDF5, `--channel_encoding none`**: runs unchanged.
* **Old P300 HDF5, any other mode**: refused, with the rebuild command. Reading
  a missing key as zero would train on a montage of origin points.
* **`bipolar_endpoints`** is optional in the schema and absent here. The trainer
  compares its presence across train/val/test, so a bipolar and a monopolar file
  cannot be mixed into one run.
* `qk_norm` is **on** here as it is for Sleep-EDF. Runs recorded before that
  change had it off and are not directly comparable.

### When is a run "already finished"?

Not "a `test_results.json` exists". The runner asks `scripts/check_run_current.py`
what produced that result and re-runs it unless it matches:

* every hyper-parameter above, plus the variant's encoding/injection and seed;
* `result_schema_version`, which is bumped when a result stops being comparable
  for a reason `provenance` cannot show. Version 2 means val and test were
  scored on the **whole** set; version 1 results came from one rank's
  `1/world_size` shard and are refused.

This is not hypothetical. A result written before the evaluation fix had
identical hyper-parameters and sat in the table looking current. `FORCE=1`
re-runs everything regardless.

Deleting a stale result by hand is the other way, and has its own trap:

```bash
source scripts/cineca_env.sh      # FIRST -- the runner sources it in a subshell,
                                  # so $PW_CKPT_ROOT is unset in your own
echo "[$PW_CKPT_ROOT]"            # empty here means the rm below deletes nothing
rm -rf "$PW_CKPT_ROOT/<sweep>/fold0/C1"
```

`rm -rf` on a path that does not exist succeeds silently, so an unset variable
turns the command into a no-op that reports success.
