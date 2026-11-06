"""NinaPro DB6 -> labelled HDF5 windows for PhysioWave fine-tuning.

DB6 is 10 subjects performing 7 grasps, 12 repetitions per session, recorded
twice a day over 5 days -- 10 sessions each. The point of the dataset is
repeatability *across sessions*, so `--split-by session` is the protocol it was
built for: train on the early days, test on the later ones, and the number you
get reflects electrode-shift and day-to-day variation rather than how well the
model memorised one afternoon.

Compared with DB5 this is a much easier label problem (8 classes against 53) on
far more data, which is what makes it the right dataset for telling apart "the
model is too big" from "the corpus is too small".

Differences from DB5 the conversion has to handle:

* 2 kHz, not 200 Hz. A 512-sample window is 256 ms here and 2.56 s there.
* The EMG matrix has 16 columns for 14 electrodes; the two unused ones are
  detected rather than hardcoded, because which columns they are is a property
  of the recording rig, not something to assume.
* Sessions are separate files, `S<subject>_D<day>_T<time>.mat`.

Output, matching what ``finetune.py`` expects:

    data   (N, C, T) float32
    label  (N,)      int64, contiguous from 0

Usage
-----
    python EMG/db6_finetune.py --root $SCRATCH/bio/emg/db6/raw \\
                               --out-dir $SCRATCH/bio/emg/db6
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

NUM_GRASPS = 7                  # plus rest -> 8 classes
SAMPLING_RATE = 2000.0
EXPECTED_ELECTRODES = 14

FNAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)\.mat$", re.IGNORECASE)


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


def live_channels(emg: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Indices of columns that carry signal.

    DB6 stores 14 electrodes in a 16-column matrix. Which two columns are the
    padding is a property of the rig, so it is measured per file instead of
    assumed -- a hardcoded pair silently drops two real electrodes if the
    layout differs.
    """
    return np.flatnonzero(emg.std(axis=0) > tol)


def normalise(win: np.ndarray, mode: str) -> np.ndarray:
    """Normalise one (C, T) window. Statistics never cross a window boundary,
    so no information leaks from the test split into the training split."""
    if mode == "none":
        return win
    if mode == "maxabs":
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
    def __init__(self, path: str, channels: int, window: int):
        self.path = path
        self.f = h5py.File(path, "w")
        self.data = self.f.create_dataset(
            "data", shape=(0, channels, window), maxshape=(None, channels, window),
            dtype="float32", chunks=(32, channels, window), compression="gzip",
            compression_opts=1)
        self.label = self.f.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype="int64", chunks=(1024,))
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
    p.add_argument("--root", required=True, help="directory holding the unzipped .mat files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=int, default=512,
                   help="window length in samples; 512 at 2 kHz is 256 ms, the usual "
                        "sEMG analysis window. Must be a multiple of --patch-size.")
    p.add_argument("--stride", type=int, default=512,
                   help="window hop. The default does not overlap: DB6 is large enough "
                        "that overlapping windows mostly buy disk. Halve it for more data.")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--channel-probe", type=int, default=5,
                   help="how many files to scan when deciding which columns are "
                        "electrodes; the union is taken, so one session with a dead "
                        "electrode cannot shrink the set")
    p.add_argument("--split-by", choices=["session", "repetition", "subject"],
                   default="session",
                   help="'session' is what DB6 was designed to measure: train on early "
                        "days, test on later ones, so the score reflects electrode shift "
                        "over days. 'repetition' is the easier within-session protocol. "
                        "'subject' is leave-one-subject-out.")
    p.add_argument("--train-days", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--val-days", type=int, nargs="+", default=[4])
    p.add_argument("--test-days", type=int, nargs="+", default=[5])
    p.add_argument("--train-reps", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--val-reps", type=int, nargs="+", default=[9, 10])
    p.add_argument("--test-reps", type=int, nargs="+", default=[11, 12])
    p.add_argument("--val-subjects", type=int, nargs="+", default=[9])
    p.add_argument("--test-subjects", type=int, nargs="+", default=[10])
    p.add_argument("--normalize", choices=["maxabs", "zscore", "none"], default="zscore",
                   help="zscore by default: Delsys Trigno output has no fixed full-scale "
                        "the way the Myo's int8 range does")
    p.add_argument("--no-rest", action="store_true")
    p.add_argument("--rest-ratio", type=float, default=1.0,
                   help="cap rest windows at this multiple of the median per-grasp count")
    p.add_argument("--max-windows-per-class", type=int, default=0,
                   help="cap each class per split (0 = no cap). DB6 at --stride 256 "
                        "yields hundreds of thousands of windows; a cap keeps the HDF5 "
                        "to a size that still fits in page cache during training.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.window % args.patch_size:
        print(f"ERROR: --window {args.window} is not a multiple of --patch-size "
              f"{args.patch_size}; model.py asserts T % patch_size == 0.", file=sys.stderr)
        return 1

    if args.split_by == "session":
        groups = [("train", args.train_days), ("val", args.val_days), ("test", args.test_days)]
        unit = "days"
    elif args.split_by == "repetition":
        groups = [("train", args.train_reps), ("val", args.val_reps), ("test", args.test_reps)]
        unit = "repetitions"
    else:
        groups = [("val", args.val_subjects), ("test", args.test_subjects)]
        unit = "subjects"
    seen: dict[int, str] = {}
    for name, vals in groups:
        for v in vals:
            if v in seen:
                print(f"ERROR: {unit[:-1]} {v} is in both {seen[v]} and {name}; that "
                      f"leaks test data into training.", file=sys.stderr)
                return 1
            seen[v] = name

    files = sorted(glob.glob(os.path.join(args.root, "**", "*.mat"), recursive=True))
    if not files:
        print(f"ERROR: no .mat files under {args.root}", file=sys.stderr)
        return 1

    # Fix the channel set once, from the union of live columns over the first
    # few files, then take those same columns everywhere. Deciding per file
    # makes the array shape depend on which electrodes happened to work that
    # session; deciding from one file makes the whole run hostage to whichever
    # file is read first.
    probe = files[:args.channel_probe]
    union: set = set()
    n_cols = 0
    for probe_path in probe:
        probe_emg = np.asarray(loadmat(probe_path, variable_names=["emg"])["emg"],
                               dtype=np.float32)
        n_cols = probe_emg.shape[1]
        union |= set(live_channels(probe_emg).tolist())
    keep = np.array(sorted(union), dtype=int)
    channels = len(keep)
    print(f"channel set from {len(probe)} probed files: {channels} of {n_cols} columns "
          f"(dropped {sorted(set(range(n_cols)) - union)})")
    if channels != EXPECTED_ELECTRODES:
        print(f"WARNING: {channels} channels, DB6 documents {EXPECTED_ELECTRODES}",
              file=sys.stderr)

    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}
    used = 0

    for path in files:
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f"  skip (unrecognised name): {os.path.basename(path)}")
            continue
        subject, day, _time = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if args.subjects and subject not in args.subjects:
            continue

        mat = loadmat(path)
        emg = np.asarray(mat["emg"], dtype=np.float32)
        stim = np.asarray(mat["restimulus"], dtype=np.int64).ravel()
        rep = np.asarray(mat["rerepetition"], dtype=np.int64).ravel()

        if emg.shape[1] <= keep.max():
            print(f"  skip {os.path.basename(path)}: {emg.shape[1]} columns, the channel "
                  f"set needs at least {keep.max() + 1}", file=sys.stderr)
            continue
        live_here = set(live_channels(emg).tolist())
        flat = [int(c) for c in keep if c not in live_here]
        if flat:
            # An electrode that came loose for one session is a fact about that
            # recording, not a reason to drop it: the column is kept, flat, so
            # the array shape stays fixed and the session still contributes.
            print(f"  note {os.path.basename(path)}: channels {flat} flat here; keeping "
                  f"the session with those channels zeroed", file=sys.stderr)
        extra = sorted(live_here - set(keep.tolist()))
        if extra:
            print(f"  WARNING {os.path.basename(path)}: columns {extra} carry signal but "
                  f"sit outside the channel set; the electrode layout differs from the "
                  f"probed files", file=sys.stderr)

        # DB6's 7 grasps keep their indices from the full NinaPro movement set,
        # so restimulus runs to 11 with gaps. The count of distinct non-zero
        # labels is what is worth checking; the maximum says nothing. The gaps
        # are closed by the contiguous remap before anything is written.
        distinct = int((np.unique(stim) != 0).sum())
        if distinct != NUM_GRASPS:
            print(f"  WARNING {os.path.basename(path)}: {distinct} distinct grasps, "
                  f"expected {NUM_GRASPS}", file=sys.stderr)

        emg = emg[:, keep].T                       # (C, N)
        n_win = 0
        for start, stop in contiguous_segments(stim, rep):
            label, r = int(stim[start]), int(rep[start])
            if r == 0 or (label == 0 and args.no_rest):
                continue
            if args.split_by == "session":
                split = seen.get(day)
            elif args.split_by == "repetition":
                split = seen.get(r)
            else:
                split = seen.get(subject, "train")
            if split is None:
                continue

            seg = emg[:, start:stop]
            for s in range(0, seg.shape[1] - args.window + 1, args.stride):
                buckets[split][0].append(
                    normalise(seg[:, s:s + args.window].astype(np.float32), args.normalize))
                buckets[split][1].append(label)
                n_win += 1

        used += 1
        print(f"  {os.path.basename(path):18s} subj={subject:2d} day={day} "
              f"-> {n_win:6d} windows")

    if not used:
        print("ERROR: no usable files matched the filters", file=sys.stderr)
        return 1

    # Rest capping, then an optional per-class cap. Both subsample rather than
    # truncate, so a cap does not silently prefer the earliest sessions.
    for split, (wins, labs) in buckets.items():
        labs_arr = np.asarray(labs)
        if len(labs_arr) == 0:
            continue
        keep_idx = np.arange(len(labs))

        if not args.no_rest and args.rest_ratio > 0:
            rest = np.flatnonzero(labs_arr == 0)
            _, counts = np.unique(labs_arr[labs_arr != 0], return_counts=True)
            if len(rest) and len(counts):
                cap = int(args.rest_ratio * np.median(counts))
                if len(rest) > cap:
                    drop = set(rng.choice(rest, size=len(rest) - cap, replace=False).tolist())
                    keep_idx = np.array([i for i in keep_idx if i not in drop])
                    print(f"  {split}: rest windows {len(rest)} -> {cap}")

        if args.max_windows_per_class > 0:
            sel = []
            for lab in np.unique(labs_arr[keep_idx]):
                idx = keep_idx[labs_arr[keep_idx] == lab]
                if len(idx) > args.max_windows_per_class:
                    idx = rng.choice(idx, size=args.max_windows_per_class, replace=False)
                sel.append(idx)
            keep_idx = np.concatenate(sel)

        if len(keep_idx) != len(labs):
            buckets[split] = ([wins[i] for i in keep_idx], [labs[i] for i in keep_idx])

    present = sorted({l for _, labs in buckets.values() for l in labs})
    remap = {old: new for new, old in enumerate(present)}
    num_classes = len(present)

    total = sum(len(w) for w, _ in buckets.values())
    gb = total * channels * args.window * 4 / 1e9
    print(f"\n{total} windows total, about {gb:.1f} GB before compression")

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {}
    for split, (wins, labs) in buckets.items():
        path = os.path.join(args.out_dir, f"{split}.h5")
        w = SplitWriter(path, channels, args.window)
        if wins:
            order = rng.permutation(len(wins))
            for i in range(0, len(order), 2048):
                idx = order[i:i + 2048]
                w.append(np.stack([wins[j] for j in idx]),
                         np.asarray([remap[labs[j]] for j in idx], dtype=np.int64))
        w.close({"sampling_rate": SAMPLING_RATE, "window": args.window,
                 "num_classes": num_classes, "normalize": args.normalize})
        counts[split] = w.n
        print(f"wrote {path}: {w.n} windows")

    meta = {
        "dataset": "ninapro_db6",
        "sampling_rate": SAMPLING_RATE,
        "num_channels": channels,
        "window": args.window,
        "stride": args.stride,
        "normalize": args.normalize,
        "num_classes": num_classes,
        "counts": counts,
        "split_by": args.split_by,
        "splits": {name: vals for name, vals in groups},
        "label_map": {str(k): v for k, v in remap.items()},
    }
    with open(os.path.join(args.out_dir, "db6_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nnum_classes = {num_classes}   channels = {channels}")
    print(f"\nFine-tune with:\n"
          f"  TASK=$(basename {args.out_dir}) IN_CHANNELS={channels} "
          f"NUM_CLASSES={num_classes} PATCH_SIZE={args.patch_size} "
          f"bash EMG/finetune_emg.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
