#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TUAB (TUH Abnormal EEG) Fine-tuning Data Preprocessing
---------------------------------------------------------
• Binary downstream task: normal (0) vs abnormal (1)
• Labels come from the corpus directory layout: .../{normal,abnormal}/...
• Uses the official edf/train and edf/eval split; val is carved out of train
  by subject, so no subject appears in two splits
• Window length: 2048 samples @ 256 Hz (8 s), non-overlapping by default
• Output HDF5 format: data (N, 19, 2048) + label (N,) + channel_names

Every window inherits its recording's label.  That is the standard TUAB
protocol, but it means window-level accuracy is optimistic; report the
recording-level vote as well when the finetune script exposes it.

Usage:
    python EEG/tuab_finetune.py --root /data/tuh_abnormal/v3.0.0 \
        --out-dir ./data/tuab

Requires `mne` for EDF reading:  pip install mne
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiowave.data.montages import MONTAGES  # noqa: E402
from physiowave.data.preprocess import PreprocessConfig, preprocess  # noqa: E402

from tueg_pretrain import (  # noqa: E402
    append_h5,
    read_edf,
    select_montage_channels,
    sliding_window,
    subject_of,
)

# ──────────────────── Global Parameters ────────────────────
FS = 256.0
WINDOW_SIZE = 2048     # 8 seconds @ 256 Hz
STEP_SIZE = 2048       # Non-overlapping: overlap between train windows of the
                       # same recording is near-duplication, not augmentation
CHUNK_SIZE = 200
MONTAGE = "standard_1020_19"

LABEL_MAP = {"normal": 0, "abnormal": 1}
NUM_CLASSES = len(LABEL_MAP)

VAL_RATIO = 0.1        # Carved out of the official train split, by subject


# ──────────────────── Label Parsing ────────────────────
def label_of(path):
    """
    Read the class from the corpus directory layout.

    TUAB stores the label as a path component:
        .../edf/train/abnormal/01_tcp_ar/aaaaaaaa_s004_t000.edf

    Returns:
        0 for normal, 1 for abnormal, or -1 when the path carries no label
    """
    parts = {p.lower() for p in path.split(os.sep)}
    hits = [name for name in LABEL_MAP if name in parts]
    if len(hits) != 1:
        return -1
    return LABEL_MAP[hits[0]]


def official_split(path):
    """'train', 'eval' or None, from the corpus layout"""
    parts = {p.lower() for p in path.split(os.sep)}
    if "eval" in parts or "test" in parts:
        return "eval"
    if "train" in parts:
        return "train"
    return None


# ──────────────────── HDF5 Operations ────────────────────
def create_finetune_h5(path, data_shape, channel_names, chunk_size=CHUNK_SIZE):
    """Create fine-tuning HDF5 file (data + label)"""
    f = h5py.File(path, "w")
    f.create_dataset(
        "data",
        shape=(0,) + data_shape,
        maxshape=(None,) + data_shape,
        chunks=(chunk_size,) + data_shape,
        dtype="float32",
        compression="gzip",
        compression_opts=4,
    )
    f.create_dataset(
        "label", shape=(0,), maxshape=(None,), chunks=(chunk_size,), dtype="int64"
    )
    f.create_dataset("channel_names", data=np.array(channel_names, dtype="S16"))
    return f


def append_labels(f, labels):
    """Append a batch of labels to an open HDF5 file"""
    if not len(labels):
        return
    arr = np.asarray(labels, dtype=np.int64)
    d = f["label"]
    n = d.shape[0]
    d.resize(n + arr.shape[0], axis=0)
    d[n:] = arr


# ──────────────────── Main ────────────────────
def process_split(records, out_path, args, montage_names):
    """Preprocess one split ([(path, label), ...]) into a single HDF5 file"""
    pp = PreprocessConfig(target_sampling_rate=args.target_fs, normalize="none")
    f = create_finetune_h5(out_path, (len(montage_names), args.window), montage_names)

    counts = {name: 0 for name in LABEL_MAP}
    skipped = 0
    win_buf, lab_buf = [], []
    try:
        for path, label in tqdm(records, desc=os.path.basename(out_path)):
            try:
                sig, ch_names, fs = read_edf(path)
            except Exception as exc:
                print(f"  skip {os.path.basename(path)}: unreadable ({exc})")
                skipped += 1
                continue

            aligned = select_montage_channels(sig, ch_names, montage_names)
            if aligned is None:
                skipped += 1
                continue

            resampled, _ = preprocess(aligned, fs, pp)
            windows = sliding_window(resampled, args.window, args.step)
            if args.max_windows_per_record:
                windows = windows[:args.max_windows_per_record]

            win_buf.extend(windows)
            lab_buf.extend([label] * len(windows))
            for name, value in LABEL_MAP.items():
                if value == label:
                    counts[name] += len(windows)

            if len(win_buf) >= CHUNK_SIZE:
                append_h5(f, win_buf)
                append_labels(f, lab_buf)
                win_buf, lab_buf = [], []
        append_h5(f, win_buf)
        append_labels(f, lab_buf)
    finally:
        f.close()

    total = sum(counts.values())
    dist = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  -> {out_path}: {total} windows ({dist}), {skipped} recordings skipped")
    return total


def parse_args():
    p = argparse.ArgumentParser(description="TUAB fine-tuning data preprocessing")
    p.add_argument("--root", required=True,
                   help="local path to TUAB; requires a signed data use agreement")
    p.add_argument("--out-dir", default="./data/tuab")
    p.add_argument("--montage", default=MONTAGE, choices=sorted(MONTAGES))
    p.add_argument("--target-fs", type=float, default=FS)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--step", type=int, default=STEP_SIZE)
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    p.add_argument("--max-windows-per-record", type=int, default=None,
                   help="cap windows per recording so long recordings do not dominate")
    p.add_argument("--max-files", type=int, default=None, help="cap for a dry run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    montage_names = MONTAGES[args.montage]

    if not os.path.isdir(args.root):
        raise SystemExit(f"Corpus root not found: {args.root}")

    train_records, eval_records, unlabelled = [], [], 0
    for dirpath, _, filenames in os.walk(args.root):
        for name in sorted(filenames):
            if not name.lower().endswith(".edf"):
                continue
            path = os.path.join(dirpath, name)
            label = label_of(path)
            if label < 0:
                unlabelled += 1
                continue
            (eval_records if official_split(path) == "eval" else
             train_records).append((path, label))

    if not train_records:
        raise SystemExit(
            f"No labelled .edf files under {args.root}; TUAB is expected to store "
            "the class as a 'normal'/'abnormal' path component."
        )
    if unlabelled:
        print(f"warning: {unlabelled} recordings carry no normal/abnormal label; ignored")
    if args.max_files:
        train_records = train_records[:args.max_files]
        eval_records = eval_records[:args.max_files]

    # Val split by subject, out of the official train split only: the official
    # eval split stays untouched so numbers remain comparable to the literature.
    subjects = sorted({subject_of(p, "tueg") for p, _ in train_records})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * args.val_ratio)))
    val_subjects = set(subjects[:n_val])

    val_records = [(p, y) for p, y in train_records
                   if subject_of(p, "tueg") in val_subjects]
    fit_records = [(p, y) for p, y in train_records
                   if subject_of(p, "tueg") not in val_subjects]

    print(f"TUAB: {len(fit_records)} train / {len(val_records)} val / "
          f"{len(eval_records)} eval recordings "
          f"({len(subjects)} train subjects, {len(val_subjects)} held out)")
    print(f"montage {args.montage} ({len(montage_names)} channels), "
          f"{args.window} samples @ {args.target_fs:g} Hz")

    os.makedirs(args.out_dir, exist_ok=True)
    process_split(fit_records, os.path.join(args.out_dir, "tuab_train.h5"),
                  args, montage_names)
    process_split(val_records, os.path.join(args.out_dir, "tuab_val.h5"),
                  args, montage_names)
    if eval_records:
        process_split(eval_records, os.path.join(args.out_dir, "tuab_test.h5"),
                      args, montage_names)

    print(f"\nDone. {NUM_CLASSES} classes: " +
          ", ".join(f"{v}={k}" for k, v in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])))
    print("Run EEG/finetune_eeg.sh against this directory.")


if __name__ == "__main__":
    main()
