"""NinaPro DB5 -> labelled HDF5 windows for PhysioWave fine-tuning.

DB5 is 10 subjects wearing two Myo armbands (16 channels at 200 Hz), performing
52 movements plus rest, 6 repetitions each, split across three exercise files
per subject.

Two things about the raw files are easy to get wrong:

* **The ``exercise`` field inside the .mat disagrees with the file name.**  In
  ``S1_E1_A1.mat`` the field reads 3, in ``S1_E2_A1.mat`` it reads 1.  The file
  name is what matches the movement count (E1->12, E2->17, E3->23), so the
  exercise is taken from the name and the movement count is verified against
  the data.
* **``restimulus`` restarts at 1 in every exercise file.**  Concatenating the
  three files without an offset silently merges 52 distinct movements into 23.
  The offsets below make the labels globally unique.

Output, matching what ``finetune.py`` expects:

    data   (N, C, T) float32
    label  (N,)      int64, contiguous from 0

Usage
-----
    python EMG/db5_finetune.py --root $SCRATCH/bio/emg/db5/raw \\
                               --out-dir $SCRATCH/bio/emg/db5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
from scipy.io import loadmat

# Movements per exercise, indexed by the number in the FILE NAME (not the
# unreliable `exercise` field). Offsets make restimulus globally unique.
EXERCISE_MOVEMENTS = {1: 12, 2: 17, 3: 23}
EXERCISE_OFFSET = {1: 0, 2: 12, 3: 29}          # 1..12, 13..29, 30..52
NUM_CHANNELS = 16
SAMPLING_RATE = 200.0

FNAME_RE = re.compile(r"S(\d+)_E(\d+)_A\d+\.mat$", re.IGNORECASE)


def contiguous_segments(*arrays: np.ndarray):
    """Yield (start, stop) of runs where every input array is constant."""
    if not arrays:
        return
    n = len(arrays[0])
    change = np.zeros(n - 1, dtype=bool)
    for a in arrays:
        change |= np.diff(a) != 0
    bounds = [0, *(np.flatnonzero(change) + 1).tolist(), n]
    for i in range(len(bounds) - 1):
        yield bounds[i], bounds[i + 1]


def normalise(win: np.ndarray, mode: str) -> np.ndarray:
    """Normalise one (C, T) window."""
    if mode == "none":
        return win
    if mode == "scale":
        # Myo output is int8-valued; /128 puts it in [-1, 1] while preserving
        # amplitude differences between windows.
        return win / 128.0
    if mode == "maxabs":
        # Per-channel max-abs, the convention EMG/epn_finetune.py uses, so a
        # DB5 window and an EPN612 window reach the encoder on the same scale.
        peak = np.max(np.abs(win), axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        return win / peak
    if mode == "zscore":
        mu = win.mean(axis=1, keepdims=True)
        sd = win.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (win - mu) / sd
    raise ValueError(f"unknown normalisation {mode!r}")


class SplitWriter:
    """Incrementally append windows to one HDF5 file.

    Written incrementally rather than accumulated in RAM: the full corpus is a
    few hundred MB and the login node is shared.
    """

    def __init__(self, path: str, channels: int, window: int):
        self.path = path
        self.f = h5py.File(path, "w")
        self.data = self.f.create_dataset(
            "data", shape=(0, channels, window), maxshape=(None, channels, window),
            dtype="float32", chunks=(32, channels, window), compression="gzip",
            compression_opts=1,
        )
        self.label = self.f.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype="int64", chunks=(1024,),
        )
        self.n = 0

    def append(self, windows: np.ndarray, labels: np.ndarray) -> None:
        k = len(windows)
        if k == 0:
            return
        self.data.resize(self.n + k, axis=0)
        self.label.resize(self.n + k, axis=0)
        self.data[self.n:self.n + k] = windows
        self.label[self.n:self.n + k] = labels
        self.n += k

    def close(self, attrs: dict) -> None:
        for k, v in attrs.items():
            self.f.attrs[k] = v
        self.f.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="directory holding the unzipped s1/ ... s10/ folders")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=int, default=512,
                   help="window length in samples; MUST be a multiple of the "
                        "model's patch_size (default 512 = 2.56 s at 200 Hz)")
    p.add_argument("--stride", type=int, default=128, help="window hop in samples")
    p.add_argument("--patch-size", type=int, default=64,
                   help="the patch_size fine-tuning will use; --window is checked "
                        "against it because model.py asserts T %% patch_size == 0")
    p.add_argument("--exercises", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="subject numbers to include (default: every one found)")
    p.add_argument("--split-by", choices=["repetition", "subject"], default="repetition",
                   help="'repetition' is the subject-dependent protocol most DB5 papers "
                        "report (87-93%%); 'subject' is leave-one-subject-out, which is "
                        "far harder (published LOSO baselines sit near 40%%) and is the "
                        "setting that actually says something about generalisation")
    p.add_argument("--train-reps", type=int, nargs="+", default=[1, 3, 4, 6])
    p.add_argument("--val-reps", type=int, nargs="+", default=[2])
    p.add_argument("--test-reps", type=int, nargs="+", default=[5])
    p.add_argument("--val-subjects", type=int, nargs="+", default=[9],
                   help="--split-by subject only")
    p.add_argument("--test-subjects", type=int, nargs="+", default=[10],
                   help="--split-by subject only")
    p.add_argument("--normalize", choices=["maxabs", "zscore", "scale", "none"],
                   default="maxabs")
    p.add_argument("--no-rest", action="store_true",
                   help="drop the rest class; labels then cover the 52 movements only")
    p.add_argument("--rest-ratio", type=float, default=1.0,
                   help="cap rest windows at this multiple of the median per-movement "
                        "class count (rest is ~45%% of the recording otherwise)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.window % args.patch_size:
        print(f"ERROR: --window {args.window} is not a multiple of --patch-size "
              f"{args.patch_size}. model.py asserts T % patch_size == 0, so "
              f"fine-tuning would die in patchify().", file=sys.stderr)
        return 1

    if args.split_by == "repetition":
        overlap = set(args.train_reps) & set(args.val_reps) | \
                  set(args.train_reps) & set(args.test_reps) | \
                  set(args.val_reps) & set(args.test_reps)
        unit = "repetitions"
    else:
        overlap = set(args.val_subjects) & set(args.test_subjects)
        unit = "subjects"
    if overlap:
        print(f"ERROR: {unit} {sorted(overlap)} appear in more than one split; "
              f"that leaks test data into training.", file=sys.stderr)
        return 1

    subj_split = {}
    if args.split_by == "subject":
        for sj in args.val_subjects:
            subj_split[sj] = "val"
        for sj in args.test_subjects:
            subj_split[sj] = "test"

    rep_split = {}
    for name, reps in (("train", args.train_reps), ("val", args.val_reps),
                       ("test", args.test_reps)):
        for r in reps:
            rep_split[r] = name

    files = sorted(glob.glob(os.path.join(args.root, "**", "*.mat"), recursive=True))
    if not files:
        print(f"ERROR: no .mat files under {args.root}", file=sys.stderr)
        return 1

    # ---- pass 1: cut windows, keyed by split ------------------------------ #
    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}
    used_files = 0

    for path in files:
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f"  skip (unrecognised name): {os.path.basename(path)}")
            continue
        subject, exercise = int(m.group(1)), int(m.group(2))
        if exercise not in args.exercises:
            continue
        if args.subjects and subject not in args.subjects:
            continue

        mat = loadmat(path)
        emg = np.asarray(mat["emg"], dtype=np.float32)
        stim = np.asarray(mat["restimulus"], dtype=np.int64).ravel()
        rep = np.asarray(mat["rerepetition"], dtype=np.int64).ravel()

        if emg.shape[1] != NUM_CHANNELS:
            print(f"  skip {os.path.basename(path)}: {emg.shape[1]} channels, "
                  f"expected {NUM_CHANNELS}", file=sys.stderr)
            continue

        # The file name decides the exercise; verify the data agrees before
        # trusting the offset, because a wrong offset silently merges classes.
        expected = EXERCISE_MOVEMENTS[exercise]
        seen = int(stim.max())
        if seen != expected:
            print(f"  WARNING {os.path.basename(path)}: E{exercise} should hold "
                  f"{expected} movements, restimulus tops out at {seen}. "
                  f"Labels may be wrong.", file=sys.stderr)
        offset = EXERCISE_OFFSET[exercise]

        emg = emg.T                                   # (C, N)
        n_win = 0
        for start, stop in contiguous_segments(stim, rep):
            label_raw, r = int(stim[start]), int(rep[start])
            if r == 0:                                 # unassigned filler samples
                continue
            if label_raw == 0 and args.no_rest:
                continue
            if args.split_by == "subject":
                # Every repetition of a held-out subject goes to that split, and
                # a training subject contributes all of its repetitions.
                split = subj_split.get(subject, "train")
            else:
                split = rep_split.get(r)
            if split is None:
                continue
            label = 0 if label_raw == 0 else label_raw + offset

            seg = emg[:, start:stop]
            for s in range(0, seg.shape[1] - args.window + 1, args.stride):
                win = normalise(seg[:, s:s + args.window].astype(np.float32),
                                args.normalize)
                buckets[split][0].append(win)
                buckets[split][1].append(label)
                n_win += 1

        used_files += 1
        print(f"  {os.path.basename(path):20s} subject={subject:2d} E{exercise} "
              f"-> {n_win:6d} windows")

    if not used_files:
        print("ERROR: no usable files matched the filters", file=sys.stderr)
        return 1

    # ---- rest capping ----------------------------------------------------- #
    # Rest is roughly 45% of the recording. Left alone it would outweigh every
    # movement class put together, so it is subsampled to the same order of
    # magnitude as an average movement class.
    if not args.no_rest and args.rest_ratio > 0:
        for split, (wins, labs) in buckets.items():
            labs_arr = np.asarray(labs)
            rest_idx = np.flatnonzero(labs_arr == 0)
            move_labels, move_counts = np.unique(labs_arr[labs_arr != 0], return_counts=True)
            if len(rest_idx) == 0 or len(move_counts) == 0:
                continue
            cap = int(args.rest_ratio * np.median(move_counts))
            if len(rest_idx) <= cap:
                continue
            keep_rest = set(rng.choice(rest_idx, size=cap, replace=False).tolist())
            keep = [i for i in range(len(labs)) if labs[i] != 0 or i in keep_rest]
            buckets[split] = ([wins[i] for i in keep], [labs[i] for i in keep])
            print(f"  {split}: rest windows {len(rest_idx)} -> {cap}")

    # ---- contiguous label remap ------------------------------------------- #
    # CrossEntropy needs labels in [0, num_classes). Dropping rest, or a subset
    # of exercises, leaves gaps that would otherwise inflate num_classes.
    present = sorted({l for _, labs in buckets.values() for l in labs})
    remap = {old: new for new, old in enumerate(present)}
    num_classes = len(present)

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {}
    for split, (wins, labs) in buckets.items():
        path = os.path.join(args.out_dir, f"{split}.h5")
        w = SplitWriter(path, NUM_CHANNELS, args.window)
        if wins:
            order = rng.permutation(len(wins))
            block = 2048
            for i in range(0, len(order), block):
                idx = order[i:i + block]
                w.append(np.stack([wins[j] for j in idx]),
                         np.asarray([remap[labs[j]] for j in idx], dtype=np.int64))
        w.close({"sampling_rate": SAMPLING_RATE, "window": args.window,
                 "num_classes": num_classes, "normalize": args.normalize})
        counts[split] = w.n
        print(f"wrote {path}: {w.n} windows")

    meta = {
        "dataset": "ninapro_db5",
        "sampling_rate": SAMPLING_RATE,
        "num_channels": NUM_CHANNELS,
        "window": args.window,
        "stride": args.stride,
        "normalize": args.normalize,
        "num_classes": num_classes,
        "counts": counts,
        "split_by": args.split_by,
        "splits": ({"train": args.train_reps, "val": args.val_reps, "test": args.test_reps}
                   if args.split_by == "repetition"
                   else {"val_subjects": args.val_subjects,
                         "test_subjects": args.test_subjects}),
        # original DB5 id (0 = rest, 1..52 = movements) -> contiguous training label
        "label_map": {str(k): v for k, v in remap.items()},
    }
    meta_path = os.path.join(args.out_dir, "db5_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nnum_classes = {num_classes}")
    print(f"metadata    = {meta_path}")
    print(f"\nFine-tune with:\n"
          f"  TASK=db5 IN_CHANNELS={NUM_CHANNELS} NUM_CLASSES={num_classes} "
          f"PATCH_SIZE={args.patch_size} bash EMG/finetune_emg.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
