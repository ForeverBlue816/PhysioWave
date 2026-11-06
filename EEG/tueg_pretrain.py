#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TUH EEG / Siena Scalp EEG Pretraining Data Preprocessing
---------------------------------------------------------
• Reads EDF recordings and emits fixed-size analysis windows as HDF5
• Channel selection + reordering onto a template montage (default: 10-20, 19ch)
• Window length: 2048 samples @ 256 Hz (8 s), sliding step: 1024 (50% overlap)
• Generates only train and val files for pretraining (data key only)
• Output HDF5 format: (N, 19, 2048) plus a `channel_names` dataset
• Subject-level split: no recording of the same subject spans train and val

Filtering and normalisation are deliberately NOT baked in here.  The pipeline
applies them at load time from `data.preprocess` in the config (see
configs/data/eeg_real.yaml), so one cached corpus serves several filter
settings.  Only resampling is done here, because windows are cut in samples and
a common rate is what makes a window a fixed duration.

Datasets behind a data use agreement (TUEG, TUAB, ...) are never downloaded;
point --root at a local copy you already obtained.

Usage:
    python EEG/tueg_pretrain.py --dataset tueg --root /data/tuh_eeg \
        --out-dir ./data/eeg_pretrain
    python EEG/tueg_pretrain.py --dataset siena --root /data/siena \
        --out-dir ./data/eeg_pretrain

Requires `mne` for EDF reading:  pip install mne
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiowave.data.montages import MONTAGES, canonical_name  # noqa: E402
from physiowave.data.preprocess import PreprocessConfig, preprocess  # noqa: E402

# ──────────────────── Global Parameters ────────────────────
FS = 256.0             # Target sampling rate (matches configs/pretrain/eeg.yaml)
WINDOW_SIZE = 2048     # Window size (8 seconds @ 256 Hz)
STEP_SIZE = 1024       # Sliding step (50% overlap)
CHUNK_SIZE = 200       # HDF5 write batch size
MONTAGE = "standard_1020_19"

TRAIN_RATIO = 0.9      # 90% train
VAL_RATIO = 0.1        # 10% val

# Minimum fraction of the montage a recording must carry to be usable.  A
# recording missing half its electrodes is not a partial montage, it is a
# different montage, and averaging the two corrupts the spatial branch.
MIN_CHANNEL_FRACTION = 1.0

#: Subject id per corpus.  Getting this wrong silently leaks subjects across
#: splits, which is the one preprocessing bug that inflates every number.
SUBJECT_PATTERNS = {
    "tueg": r"([a-z]{8})_s\d+_t\d+",     # aaaaaaaa_s001_t000.edf
    "siena": r"(PN\d+)",                 # PN00-1.edf
}


# ──────────────────── EDF Reading ────────────────────
def read_edf(path):
    """
    Read one EDF recording.

    Args:
        path: Path to the .edf file

    Returns:
        (signal (C, T) float64 in microvolts, channel names list, sampling rate)
    """
    try:
        import mne
    except ImportError as exc:                                    # pragma: no cover
        raise SystemExit(
            "EDF reading needs mne: pip install mne"
        ) from exc

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    # mne returns volts; EEG is conventionally handled in microvolts
    return raw.get_data() * 1e6, list(raw.ch_names), float(raw.info["sfreq"])


# ──────────────────── Montage Alignment ────────────────────
def select_montage_channels(sig, ch_names, montage_names):
    """
    Reorder recording channels onto the template montage.

    TUH labels look like 'EEG FP1-REF'; `canonical_name` strips the prefix,
    the reference suffix and maps the old T3/T4/T5/T6 names onto T7/T8/P7/P8.

    Args:
        sig: (C_raw, T) - Raw recording
        ch_names: Channel labels as stored in the EDF
        montage_names: Template channel order to produce

    Returns:
        (C_montage, T) array, or None if the recording is missing electrodes
    """
    index = {}
    for i, name in enumerate(ch_names):
        canon = canonical_name(name)
        index.setdefault(canon, i)          # first occurrence wins

    rows, missing = [], []
    for name in montage_names:
        if name in index:
            rows.append(sig[index[name]])
        else:
            missing.append(name)

    if len(missing) > len(montage_names) * (1.0 - MIN_CHANNEL_FRACTION):
        return None
    return np.stack(rows, axis=0)


# ──────────────────── Sliding Window ────────────────────
def sliding_window(eeg_2d, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Sliding window segmentation for EEG signals

    Args:
        eeg_2d: (C, L) - EEG signal
        window_size: Size of each window
        step_size: Step between windows

    Returns:
        list of (C, window_size) arrays
        Incomplete trailing windows are dropped: a zero-padded EEG window is an
        artefact the frequency-guided masking would happily learn.
    """
    segs = []
    L = eeg_2d.shape[1]

    for start in range(0, L - window_size + 1, step_size):
        segs.append(eeg_2d[:, start:start + window_size])

    return segs


# ──────────────────── HDF5 Operations ────────────────────
def create_pretrain_h5(path, data_shape, channel_names, chunk_size=CHUNK_SIZE):
    """Create pretraining HDF5 file (data key only, plus channel names)"""
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
    # build_manifest reads this back; without it the SSL branch has no
    # electrode positions and is disabled for the whole corpus.
    f.create_dataset("channel_names", data=np.array(channel_names, dtype="S16"))
    return f


def append_h5(f, windows):
    """Append a batch of windows to an open HDF5 file"""
    if not len(windows):
        return
    arr = np.asarray(windows, dtype=np.float32)
    d = f["data"]
    n = d.shape[0]
    d.resize(n + arr.shape[0], axis=0)
    d[n:] = arr


# ──────────────────── Subject-Level Split ────────────────────
def subject_of(path, dataset):
    """Extract the subject id from a recording path"""
    pattern = SUBJECT_PATTERNS.get(dataset)
    if pattern:
        m = re.search(pattern, path)
        if m:
            return m.group(1)
    # Fallback: the containing directory, which is one subject in most corpora
    return os.path.basename(os.path.dirname(path))


def split_subjects(files, dataset, val_ratio=VAL_RATIO, seed=42):
    """
    Split recordings by subject, never by recording.

    Returns:
        (train_files, val_files)
    """
    by_subject = {}
    for path in files:
        by_subject.setdefault(subject_of(path, dataset), []).append(path)

    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = set(subjects[:n_val])

    train, val = [], []
    for subject, paths in by_subject.items():
        (val if subject in val_subjects else train).extend(paths)
    print(f"  subjects: {len(subjects)} total, {len(val_subjects)} held out for val")
    return sorted(train), sorted(val)


# ──────────────────── Main ────────────────────
def process_split(files, out_path, args, montage_names):
    """Preprocess one split into a single HDF5 file"""
    pp = PreprocessConfig(target_sampling_rate=args.target_fs, normalize="none")
    f = create_pretrain_h5(out_path, (len(montage_names), args.window), montage_names)

    total, skipped, buffer = 0, 0, []
    try:
        for path in tqdm(files, desc=os.path.basename(out_path)):
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

            buffer.extend(windows)
            total += len(windows)
            if len(buffer) >= CHUNK_SIZE:
                append_h5(f, buffer)
                buffer = []
        append_h5(f, buffer)
    finally:
        f.close()

    print(f"  -> {out_path}: {total} windows, {skipped} recordings skipped")
    return total


def parse_args():
    p = argparse.ArgumentParser(description="EEG pretraining data preprocessing")
    p.add_argument("--dataset", default="tueg", choices=sorted(SUBJECT_PATTERNS),
                   help="corpus layout (decides how the subject id is parsed)")
    p.add_argument("--root", required=True,
                   help="local path to the corpus; nothing is ever downloaded")
    p.add_argument("--out-dir", default="./data/eeg_pretrain")
    p.add_argument("--montage", default=MONTAGE, choices=sorted(MONTAGES),
                   help="template montage every recording is projected onto")
    p.add_argument("--target-fs", type=float, default=FS)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--step", type=int, default=STEP_SIZE)
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    p.add_argument("--max-files", type=int, default=None, help="cap for a dry run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    montage_names = MONTAGES[args.montage]

    if not os.path.isdir(args.root):
        raise SystemExit(f"Corpus root not found: {args.root}")

    files = []
    for dirpath, _, filenames in os.walk(args.root):
        for name in sorted(filenames):
            if name.lower().endswith(".edf"):
                files.append(os.path.join(dirpath, name))
    files.sort()
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No .edf files under {args.root}")

    print(f"{args.dataset}: {len(files)} recordings under {args.root}")
    print(f"montage {args.montage} ({len(montage_names)} channels), "
          f"{args.window} samples @ {args.target_fs:g} Hz "
          f"({args.window / args.target_fs:.1f} s per window)")

    train_files, val_files = split_subjects(files, args.dataset, args.val_ratio, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    process_split(train_files, os.path.join(args.out_dir, f"{args.dataset}_train.h5"),
                  args, montage_names)
    process_split(val_files, os.path.join(args.out_dir, f"{args.dataset}_val.h5"),
                  args, montage_names)

    print("\nDone. Point the config at this directory, e.g.:")
    print(f"  --set data.datasets=[{args.dataset}] data.roots.{args.dataset}={args.out_dir}")


if __name__ == "__main__":
    main()
