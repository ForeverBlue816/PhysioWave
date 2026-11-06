#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KaggleERN (BCI Challenge @ NER 2015) -> labelled HDF5, on EEGPT's setup.
-------------------------------------------------------------------------
    python EEG/kaggle_ern_finetune.py \\
        --raw-dir $PW_DATA_EEG/kaggle_ern --out-dir $PW_DATA_EEG/ern_f0 --fold 0

SOURCE. Kaggle competition `inria-bci-challenge`, fetched by
EEG/download_kaggle_ern.sh. Expected layout -- the one EEGPT's readme gives:

    <raw-dir>/TrainLabels.csv        IdFeedBack,Prediction   (16 subjects)
    <raw-dir>/true_labels.csv        the 10 test subjects' labels
    <raw-dir>/SampleSubmission.csv   the test IdFeedBack order (optional)
    <raw-dir>/train/Data_S02_Sess01.csv ...
    <raw-dir>/test/Data_S01_Sess01.csv  ...

Each CSV is one session at 200 Hz: a Time column, the 56 EEG channels, EOG,
and FeedBackEvent (1 at the sample the feedback appeared). The channel names
are read from the header rather than assumed.

The task: after each spelled letter the subject sees feedback, and the label
says whether the speller got it right (1) or wrong (0). An error elicits an
error-related potential. The classes are unbalanced -- roughly seven correct to
three wrong -- which is why EEGPT reports balanced accuracy and AUROC.

PREPROCESSING is the pretraining pipeline (EEG/eegpt_bench_common.py), run on
each continuous session before it is epoched: detrend, 50 Hz notch, 0.5 Hz
high-pass, 256 Hz, then a per-window z-score. Not EEGPT's per-trial min-max and
common average -- the encoder saw neither in pretraining.

THE TASK, as EEGPT defines it:

    epoch       -0.7 s to +1.3 s around feedback onset (their tmin=-0.7, tlen=2)
    channels    all 56 by default; `--channels eegpt19` gives their 19
    folds       their four, 0-based here (their Folds[fold + 1])
    S22 Sess05  EEGPT skips it as an "error file". Here any session whose
                feedback count disagrees with its label count is skipped and
                named -- S22 Sess05 included, if that is what is wrong with it

TEST LABELS. The Kaggle download does not include true_labels.csv, and it is
not published anywhere this could find. Without it (the default --test-split
auto picks this) train, val and test all come from the 16 labelled subjects:
EEGPT's fold leaves 4 of them out, those are the test set, and 2 of its 12
training subjects validate. That is a valid subject-disjoint benchmark, but its
test subjects are not EEGPT's, so their Table 4 row is a reference, not a
comparison on the same data.

Output: see EEG/eegpt_bench_common.py. 512 samples at 256 Hz per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eegpt_bench_common as bc                                  # noqa: E402

FS_IN = 200.0
TMIN_S, TLEN_S = -0.7, 2.0
CLASSES = ["error", "correct"]
SESSIONS = [1, 2, 3, 4, 5]
TRAIN_SUBJECTS = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
TEST_SUBJECTS = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
# EEGPT's Folds dict, 1-based there, 0-based here. Each is the training set;
# the four training subjects it omits are this pipeline's validation set.
EEGPT_FOLDS = [
    [12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
    [2, 6, 7, 11, 17, 18, 20, 21, 22, 23, 24, 26],
    [2, 6, 7, 11, 12, 13, 14, 16, 22, 23, 24, 26],
    [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21],
]
EEGPT19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
           "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def session_path(raw_dir: str, split: str, s: int, k: int) -> str:
    return os.path.join(raw_dir, split, f"Data_S{s:02d}_Sess{k:02d}.csv")


def fb_id(s: int, k: int, fb: int) -> str:
    return f"S{s:02d}_Sess{k:02d}_FB{fb:03d}"


#: Header names in the Kaggle CSVs that are not the electrode's name. The files
#: write PO8 as "P08" -- digit zero for the letter O -- between PO7/POz and
#: O1/O2, where only PO8 fits. EEGPT hard-codes the names instead of reading
#: them and uses 19 channels without PO8, so it never met this. Mapped here and
#: announced; any OTHER unknown name still stops the converter.
HEADER_ALIASES = {"P08": "PO8"}


def read_session(path: str):
    """``(eeg [C, T] float64, channel names, feedback sample indices)``."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = list(df.columns)
    if cols[0].strip().lower() != "time" or "FeedBackEvent" not in cols:
        raise SystemExit(f"{path}: unexpected header {cols[:3]} ... {cols[-3:]}")
    eeg_cols = [c for c in cols[1:] if c not in ("EOG", "FeedBackEvent")]
    x = df[eeg_cols].to_numpy(dtype=np.float64).T
    onsets = np.flatnonzero(df["FeedBackEvent"].to_numpy() > 0)
    names = [HEADER_ALIASES.get(c.strip(), c.strip()) for c in eeg_cols]
    return x, names, onsets


def train_labels(raw_dir: str) -> dict:
    import pandas as pd
    path = os.path.join(raw_dir, "TrainLabels.csv")
    df = pd.read_csv(path)
    return dict(zip(df["IdFeedBack"].astype(str), df["Prediction"].astype(int)))


def test_labels(raw_dir: str, order: list) -> dict:
    """``{IdFeedBack: label}`` for the test subjects.

    By ID when either file carries one. Only if neither does is the label
    column read by POSITION against ``order`` -- the (subject, session,
    feedback) sequence EEGPT consumes it in -- and the counts must then agree
    exactly, because a positional read that is off by one is a label file
    shifted by one, silently.
    """
    import pandas as pd
    path = os.path.join(raw_dir, "true_labels.csv")
    if not os.path.isfile(path):
        raise SystemExit(
            f"no {path}.\n  The test subjects' labels are not in TrainLabels.csv"
            f" -- EEGPT scores against true_labels.csv. Check what the Kaggle "
            f"download contained (EEG/download_kaggle_ern.sh lists it).")
    df = pd.read_csv(path)
    lab_col = next((c for c in df.columns
                    if c.lower() in ("label", "prediction", "true_label")), None)
    if lab_col is None:
        raise SystemExit(f"{path}: no label column in {list(df.columns)}")
    labels = df[lab_col].astype(int).to_numpy()
    if "IdFeedBack" in df.columns:
        return dict(zip(df["IdFeedBack"].astype(str), labels))
    sub = os.path.join(raw_dir, "SampleSubmission.csv")
    if os.path.isfile(sub):
        ids = pd.read_csv(sub)["IdFeedBack"].astype(str).tolist()
        if len(ids) != len(labels):
            raise SystemExit(f"{path} has {len(labels)} labels and "
                             f"SampleSubmission.csv {len(ids)} ids")
        return dict(zip(ids, labels))
    if len(order) != len(labels):
        raise SystemExit(
            f"{path} has {len(labels)} labels and the test sessions have "
            f"{len(order)} feedback events; a positional match would be wrong")
    print("  NOTE: true_labels.csv has no ids and SampleSubmission.csv is "
          "absent; matching by position, as EEGPT does", file=sys.stderr)
    return dict(zip(order, labels))


def prepare_session(x: np.ndarray, names: list, onsets: np.ndarray,
                    keep: list):
    """One session -> ``(epochs [N, C, 512], kept feedback indices)``."""
    lower = {n.lower(): i for i, n in enumerate(names)}
    missing = [c for c in keep if c.lower() not in lower]
    if missing:
        raise SystemExit(f"channels {missing} not in the recording {names}")
    x = x[[lower[c.lower()] for c in keep]]
    # The CSVs carry no unit; the per-window z-score at the end makes the
    # absolute scale irrelevant, so "uV" is a label, not a claim.
    x = bc.pretrain_pipeline(x, FS_IN, unit="uV")
    length = int(round(TLEN_S * bc.FS_OUT))
    starts = np.round(onsets * bc.FS_OUT / FS_IN + TMIN_S * bc.FS_OUT).astype(np.int64)
    ok = (starts >= 0) & (starts + length <= x.shape[1])
    X = bc.normalise_windows(bc.epoch_at(x, starts[ok], length))
    return X.astype(np.float32), np.flatnonzero(ok)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0, choices=range(len(EEGPT_FOLDS)),
                   help="EEGPT's fold, 0-based (their Folds[fold + 1])")
    p.add_argument("--channels", default="all", choices=["all", "eegpt19"],
                   help="all 56 (default) or EEGPT's 19")
    p.add_argument("--test-split", default="auto",
                   choices=["auto", "kaggle", "labelled"],
                   help="kaggle: the 10 Kaggle test subjects, scored against "
                        "true_labels.csv (EEGPT's). labelled: train/val/test all "
                        "from the 16 labelled subjects. auto (default): kaggle if "
                        "true_labels.csv exists, else labelled")
    p.add_argument("--train-subjects", default=None,
                   help="override, comma-separated -- for smoke tests")
    p.add_argument("--test-subjects", default=None,
                   help="override, comma-separated -- for smoke tests")
    args = p.parse_args(argv)

    have_true = os.path.isfile(os.path.join(args.raw_dir, "true_labels.csv"))
    mode = args.test_split
    if mode == "auto":
        mode = "kaggle" if have_true else "labelled"
        if not have_true:
            print("  NOTE: no true_labels.csv, so the 10 Kaggle test subjects "
                  "cannot be scored.\n        Building train/val/test from the "
                  "16 LABELLED subjects instead (--test-split labelled).\n"
                  "        The test set is therefore not EEGPT's: their Table 4 "
                  "row is a different set of subjects.", file=sys.stderr)
    if mode == "kaggle":
        train = EEGPT_FOLDS[args.fold]
        val = [s for s in TRAIN_SUBJECTS if s not in train]
        test = TEST_SUBJECTS
    else:
        train, val, test = labelled_split(args.fold)
    if args.train_subjects:
        pool = [int(v) for v in args.train_subjects.split(",")]
        train, val = pool[:-1], pool[-1:]
    if args.test_subjects:
        test = [int(v) for v in args.test_subjects.split(",")]
    if not args.test_subjects and not args.train_subjects:
        assert not set(train) & set(val) and not set(train) & set(test) \
            and not set(val) & set(test), "split is not subject-disjoint"
    split = {"train": train, "val": val, "test": test}

    # The montage: from the first file's header, so a column order different
    # from the one EEGPT hard-codes cannot silently relabel electrodes.
    first = session_path(args.raw_dir, "train", train[0], 1)
    if not os.path.isfile(first):
        raise SystemExit(f"missing {first}\n  fetch the data with: "
                         f"bash EEG/download_kaggle_ern.sh {args.raw_dir}")
    _, names, _ = read_session(first)
    import pandas as pd
    raw_header = [c.strip() for c in pd.read_csv(first, nrows=0).columns]
    for bad, good in HEADER_ALIASES.items():
        if bad in raw_header:
            print(f"  header {bad!r} read as {good!r} (a typo in the Kaggle "
                  f"files: it sits between PO7/POz and O1/O2)", file=sys.stderr)
    keep = EEGPT19 if args.channels == "eegpt19" else names
    bc.check_vocabulary(keep)

    print(f"KaggleERN: fold {args.fold}  train {train}  val {val}  test {test}")
    print(f"  {len(keep)} channels, feedback {TMIN_S:+g}..{TMIN_S + TLEN_S:+g} s, "
          f"pretraining pipeline ({bc.PREP_VERSION}): detrend, {bc.MAINS_HZ:g} Hz "
          f"notch, {bc.CFG.highpass_hz:g} Hz high-pass, {bc.FS_OUT} Hz, "
          f"window z-score")

    tr_lab = train_labels(args.raw_dir)
    te_lab = None
    skipped, dropped = [], 0
    provenance = bc.provenance({
        "dataset": "KaggleERN (inria-bci-challenge)",
        "source_sampling_rate": FS_IN,
        "epoch": f"feedback {TMIN_S:+g} s, {TLEN_S:g} s long",
        "fold": args.fold, "eegpt_fold": args.fold + 1, "split": split,
        "channels": keep, "test_split": mode,
        "protocol": ("EEGPT's fold; left-out training subjects validate, "
                     "the 10 Kaggle test subjects are the test set"
                     if mode == "kaggle" else
                     "labelled subjects only (no true_labels.csv): EEGPT fold's "
                     "4 left-out subjects are the test set, 2 of its 12 "
                     "training subjects validate -- NOT EEGPT's test set"),
    })

    rows = []
    for name, subs in split.items():
        folder = "test" if (name == "test" and mode == "kaggle") else "train"
        X, Y, S, K = [], [], [], []
        for s in subs:
            for k in SESSIONS:
                path = session_path(args.raw_dir, folder, s, k)
                if not os.path.isfile(path):
                    skipped.append((os.path.basename(path), "missing"))
                    continue
                x, names_k, onsets = read_session(path)
                if names_k != names:
                    raise SystemExit(f"{path}: channel order differs from {first}")
                ids = [fb_id(s, k, i) for i in range(1, len(onsets) + 1)]
                if folder == "test":
                    if te_lab is None:
                        te_lab = test_labels(args.raw_dir, test_order(args.raw_dir, test))
                    lab = te_lab
                else:
                    lab = tr_lab
                have = [i for i in ids if i in lab]
                if len(have) != len(ids) or not ids:
                    skipped.append((os.path.basename(path),
                                    f"{len(onsets)} feedback events, "
                                    f"{len(have)} labels"))
                    continue
                Xs, kept = prepare_session(x, names, onsets, keep)
                dropped += len(ids) - len(kept)
                X.append(Xs)
                Y.append(np.array([lab[ids[i]] for i in kept], dtype=np.int64))
                S.append(np.full(len(kept), s)); K.append(np.full(len(kept), k))
        rows.append(bc.write_split(os.path.join(args.out_dir, f"{name}.h5"),
                                   X, Y, S, K, keep, CLASSES, provenance))

    for f, why in skipped:
        print(f"  SKIPPED {f}: {why}", file=sys.stderr)
    if dropped:
        print(f"  dropped {dropped} epoch(s) that ran off the end of a session",
              file=sys.stderr)
    bc.print_summary(rows)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"provenance": provenance, "files": rows,
                   "skipped_sessions": skipped, "dropped_epochs": dropped}, f, indent=2)
    print(f"  wrote {args.out_dir}/{{train,val,test}}.h5  (--num-classes 2)")
    return 0


def labelled_split(fold: int, seed: int = 7):
    """Train/val/test from the 16 labelled subjects, in EEGPT's fold shape.

    For when true_labels.csv is not available and the Kaggle test subjects
    cannot be scored -- the usual situation, since the Kaggle download does not
    contain it. EEGPT's fold `fold` trains on 12 of the 16 and leaves 4 out;
    those 4 become the TEST set here, and 2 of the 12, drawn deterministically,
    the validation set. 10/2/4, subject-disjoint, every label from
    TrainLabels.csv.
    """
    twelve = EEGPT_FOLDS[fold]
    test = [s for s in TRAIN_SUBJECTS if s not in twelve]
    rng = np.random.default_rng(seed + fold)
    val = sorted(int(v) for v in rng.choice(twelve, size=2, replace=False))
    train = [s for s in twelve if s not in val]
    return train, val, test


def test_order(raw_dir: str, subjects: list) -> list:
    """Every test feedback id, in the order EEGPT consumes true_labels.csv."""
    order = []
    for s in subjects:
        for k in SESSIONS:
            path = session_path(raw_dir, "test", s, k)
            if os.path.isfile(path):
                n = len(read_session(path)[2])
                order += [fb_id(s, k, i) for i in range(1, n + 1)]
    return order


if __name__ == "__main__":
    sys.exit(main())
