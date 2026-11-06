#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep-EDF Expanded (Sleep Cassette) -> labelled HDF5 for 5-class sleep staging.
------------------------------------------------------------------------------
The preprocessing deliberately reproduces EEGPT's `datasets/downstream/
prepare_sleep.py` step for step, because the point of running this dataset is
to sit next to their published number rather than beside it:

    * source        braindecode SleepPhysionet, which fetches Sleep-EDFx SC
                    from PhysioNet through MNE the first time it is asked
    * channels      the two EEG derivations, Fpz-Cz and Pz-Oz, at 100 Hz
    * cropping      crop_wake_mins=30, so only 30 minutes of wake either side
                    of the sleep period is kept -- without this the class
                    balance is dominated by hours of pre-bed wake
    * scaling       volts to microvolts, then a 30 Hz lowpass
    * windows       30 s, non-overlapping, aligned to the hypnogram events,
                    i.e. 3000 samples
    * labels        W / N1 / N2 / N3 / REM, with the AASM merge of stages 3
                    and 4 into N3
    * normalising   per-channel z-score over the whole recording (see below)

What is NOT reproduced is their evaluation protocol. EEGPT trains 40 epochs
with no checkpoint callback and reports the validation fold, so the model is
selected on the set it is scored on. `--split holdout` keeps a test set that
nothing selects on; `--split eegpt-fold` reproduces their partition for a
like-for-like row. Report both and say which is which.

Two stages, because the download and the decoding are slow and the split is
not. `--stage cache` writes one .npz per subject; `--stage split` turns those
into HDF5. The default runs both.

Usage:
    pip install braindecode          # pulls in mne
    python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf

Output, matching what finetune.py reads:
    data          (N, 2, 3000) float32
    label         (N,)         int64      0=W 1=N1 2=N2 3=N3 4=REM
    channel_names (2,)         bytes
    subject       (N,)         int64      provenance; unused by the trainer
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

FS = 100.0
WINDOW_SAMPLES = 30 * int(FS)      # 3000
LOWPASS_HZ = 30.0
CROP_WAKE_MINS = 30
# braindecode strips the "EEG " prefix in SleepPhysionet._load_raw.
CHANNELS = ["Fpz-Cz", "Pz-Oz"]

# --------------------------------------------------------------------------- #
# Channel metadata
#
# Sleep-EDF's two signals are BIPOLAR DERIVATIONS, not four independent
# electrodes and not two electrodes against a shared reference. Fpz-Cz and
# Pz-Oz share nothing: Cz is the negative end of the first and Oz of the second.
# Three things follow, and all three have been got wrong before:
#
#   * The recordings arrive already re-referenced. Calling set_eeg_reference()
#     on them would re-reference a derivation and produce a signal that is not
#     any montage.
#   * DERIVATION_MATRIX states what the channels *mean*. It is never multiplied
#     into the data -- the subtraction it describes already happened in the
#     recording hardware.
#   * The endpoints are declared here rather than parsed out of the name.
#     `split("-")` happens to work for "Fpz-Cz" and does not for "T3", "A1-A2"
#     or "EEG Fpz-Cz-REF"; a dataset adapter is the right place to know.
# --------------------------------------------------------------------------- #
ELECTRODES = ["Fpz", "Cz", "Pz", "Oz"]

BIPOLAR_ENDPOINTS = [
    ("Fpz", "Cz"),
    ("Pz", "Oz"),
]

DERIVATION_MATRIX = np.array([
    [+1.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, +1.0, -1.0],
], dtype=np.float32)

#: Bumped whenever the set of metadata datasets changes. The trainer refuses a
#: file whose schema it does not know rather than reading a missing key as zero.
METADATA_SCHEMA_VERSION = 1
COORDINATE_SOURCE = "mne_standard_1020"
COORDINATE_TYPE = "template_not_subject_digitized"
COORDINATE_SYSTEM = "RAS"
COORDINATE_UNIT = "m"
DERIVATION_TYPE = "bipolar"
MONTAGE_TYPE = "clinical_bipolar"

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
MAPPING = {                        # AASM: stages 3 and 4 are one stage
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# braindecode's SleepPhysionet covers subjects 0..82 with these absent.
MISSING = {39, 68, 69, 78, 79}

VERBOSE = False    # set by --verbose

# The subject list EEGPT's finetune_EEGPT_SleepEDF.py runs its 10 folds over.
# Copied verbatim so the partition is theirs, not a reconstruction of theirs.
EEGPT_SUBJECTS = [
    0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24,
    25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 42, 44, 45, 46, 47, 48, 49,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74,
    75, 76, 77, 81, 82,
]


# --------------------------------------------------------------------------- #
# Stage 1: fetch and preprocess one subject
# --------------------------------------------------------------------------- #
def cache_subject(subject: int, cache_dir: str, overwrite: bool = False) -> str | None:
    """Download, preprocess and window one subject into a .npz. Returns its path."""
    out = os.path.join(cache_dir, f"sub{subject:02d}.npz")
    if os.path.exists(out) and not overwrite:
        return out

    from braindecode.datasets import SleepPhysionet
    from braindecode.preprocessing import Preprocessor, preprocess
    from braindecode.preprocessing import create_windows_from_events
    from sklearn.preprocessing import scale as standard_scale

    if not VERBOSE:
        # 64 subjects of MNE's filter-design banner scrolls the progress bar
        # off the screen and buries anything that actually went wrong.
        import warnings

        import mne
        mne.set_log_level("ERROR")
        warnings.filterwarnings("ignore", category=UserWarning, module="braindecode")

    ds = SleepPhysionet(subject_ids=[subject], crop_wake_mins=CROP_WAKE_MINS)
    # Put the channels in CHANNELS order explicitly and fail if they are not all
    # there. The default order happens to be [Fpz-Cz, Pz-Oz] for this corpus, and
    # relying on that would make the channel metadata -- which says row 0 is
    # Fpz+ Cz- -- describe whichever row the EDF happened to put first. A
    # silently transposed pair is not an error anywhere downstream; it is a
    # model trained with the electrodes swapped.
    for rec in ds.datasets:
        have = list(rec.raw.ch_names)
        missing = [c for c in CHANNELS if c not in have]
        if missing:
            raise RuntimeError(
                f"subject {subject}: channels {missing} absent; the recording "
                f"has {have}")
        dupes = [c for c in CHANNELS if have.count(c) > 1]
        if dupes:
            raise RuntimeError(f"subject {subject}: channels {dupes} appear twice")
        sfreq = float(rec.raw.info["sfreq"])
        if abs(sfreq - FS) > 1e-6:
            raise RuntimeError(
                f"subject {subject}: sampling rate {sfreq} Hz, expected {FS}. "
                f"Every window length and the metadata's sampling_rate assume "
                f"{FS} Hz.")
        rec.raw.pick(CHANNELS)          # picks AND reorders, in one call
        assert list(rec.raw.ch_names) == CHANNELS, rec.raw.ch_names

    preprocess(ds, [
        Preprocessor(lambda d: d * 1e6),                      # volts -> microvolts
        Preprocessor("filter", l_freq=None, h_freq=LOWPASS_HZ, n_jobs=1),
    ])
    windows = create_windows_from_events(
        ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=WINDOW_SAMPLES, window_stride_samples=WINDOW_SAMPLES,
        preload=True, mapping=MAPPING,
    )
    # Per-channel z-score. Note what this actually normalises: with no reject,
    # picks or flat argument, create_windows_from_events returns an
    # EEGWindowsDataset -- lazy views into the Raw -- so the preprocessor lands
    # on the *recording*, not on each window. braindecode says so out loud
    # ("Applying preprocessors ... to the mne.io.Raw of an EEGWindowsDataset").
    #
    # Measured on subject 0: per-window std ranges over [0.306, 3.836] where
    # per-window scaling would pin it at 1.000, while the concatenation of all
    # windows has mean 0.000 and std 0.999 per channel.
    #
    # This matches EEGPT rather than diverging from it. Their pinned
    # braindecode 0.8.1 computes
    #     use_mne_epochs = (reject is not None) or (picks is not None)
    #                      or (flat is not None) or (drop_bad_windows is True)
    # and they pass none of them, so their windows are lazy too.
    #
    # It is also the right choice for sleep staging: N3 is defined by
    # high-amplitude slow waves, and per-window scaling would divide exactly
    # that amplitude away.
    preprocess(windows, [Preprocessor(standard_scale, channel_wise=True)])

    X = np.stack([w[0] for w in windows]).astype(np.float32)   # (N, C, 3000)
    y = np.array([w[1] for w in windows], dtype=np.int64)
    if X.shape[1] != len(CHANNELS) or X.shape[2] != WINDOW_SAMPLES:
        raise RuntimeError(f"subject {subject}: unexpected shape {X.shape}")

    np.savez_compressed(out, data=X, label=y)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: subject-level splits
# --------------------------------------------------------------------------- #
def holdout_split(subjects: list[int], seed: int = 7) -> dict[str, list[int]]:
    """60/20/20 by subject.

    The split is by subject and not by window: consecutive 30 s epochs of one
    night are near-duplicates, so a window-level split would put a subject's
    own neighbouring epochs on both sides and report a number that says
    nothing about a new sleeper.
    """
    if len(subjects) < 3:
        raise SystemExit(
            f"a subject-level train/val/test split needs at least 3 subjects, got "
            f"{len(subjects)}. For a pipeline smoke test use 3 or more, e.g. "
            f"--subjects 0,2,4."
        )
    rng = np.random.default_rng(seed)
    order = np.array(subjects)[rng.permutation(len(subjects))]
    # Rounding down twice empties the validation split for small subject counts
    # -- 4 subjects would give 2/0/2 -- and an empty val set does not fail, it
    # silently disables model selection. One subject each is the floor.
    n_tr = max(1, int(0.6 * len(order)))
    n_va = max(1, int(0.2 * len(order)))
    n_tr = min(n_tr, len(order) - 2)
    n_va = min(n_va, len(order) - n_tr - 1)
    return {
        "train": sorted(order[:n_tr].tolist()),
        "val": sorted(order[n_tr:n_tr + n_va].tolist()),
        "test": sorted(order[n_tr + n_va:].tolist()),
    }


def eegpt_fold_split(fold: int, val_ratio: float = 0.15,
                     seed: int = 7) -> dict[str, list[int]]:
    """EEGPT's 10-fold partition of its 64-subject list, fold ``fold`` held out.

    EEGPT scores the held-out fold and selects on it. To keep the comparison
    honest a validation set is carved out of the *training* subjects here, so
    the held-out fold is only ever scored. That makes our number pessimistic
    relative to theirs rather than the other way round.
    """
    n = len(EEGPT_SUBJECTS) // 10
    held = EEGPT_SUBJECTS[fold * n:(fold + 1) * n]
    rest = [s for s in EEGPT_SUBJECTS if s not in set(held)]
    rng = np.random.default_rng(seed + fold)
    rest = np.array(rest)[rng.permutation(len(rest))]
    n_va = max(1, int(val_ratio * len(rest)))
    return {"train": sorted(rest[n_va:].tolist()),
            "val": sorted(rest[:n_va].tolist()),
            "test": sorted(held)}


def electrode_coordinates() -> np.ndarray:
    """``[4, 3]`` template positions for ELECTRODES, from MNE's standard_1020.

    Read here, in preprocessing, and written into the HDF5. The training loop
    must never import MNE: it runs on compute nodes whose environment is built
    for torch, and a montage lookup at step time would make the model's
    behaviour depend on which MNE happened to be installed.

    These are template coordinates, not this subject's digitised positions.
    Sleep-EDF ships no digitisation, so nothing better exists for it, and the
    HDF5 records the distinction rather than letting a reader assume.
    """
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    missing = [e for e in ELECTRODES if e not in pos]
    if missing:
        raise SystemExit(
            f"standard_1020 has no position for {missing}.\n"
            f"  MNE {mne.__version__} names them differently, or the montage "
            f"changed. Fix ELECTRODES rather than guessing a coordinate."
        )
    return np.stack([np.asarray(pos[e], dtype=np.float32) for e in ELECTRODES])


def build_channel_metadata() -> dict:
    """Every per-dataset field the trainer needs, as plain numpy.

    One copy per file, not one per 30 s window: the montage is a property of the
    recording set-up and is identical for all 100k of them. Storing it per
    window would multiply a 200-byte fact by the corpus.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from channel_embedding import CHANNEL_VOCAB, channel_id

    xyz = electrode_coordinates()
    index = {e: i for i, e in enumerate(ELECTRODES)}
    pos_idx = np.array([index[a] for a, _ in BIPOLAR_ENDPOINTS], dtype=np.int64)
    neg_idx = np.array([index[b] for _, b in BIPOLAR_ENDPOINTS], dtype=np.int64)

    # Sanity against the matrix, so the two descriptions cannot drift apart.
    for c, (p, n) in enumerate(zip(pos_idx, neg_idx)):
        row = DERIVATION_MATRIX[c]
        assert row[p] == +1.0 and row[n] == -1.0 and np.abs(row).sum() == 2.0, (
            f"DERIVATION_MATRIX row {c} disagrees with BIPOLAR_ENDPOINTS")

    # Midpoint of the two endpoints on the unit sphere, renormalised. This is a
    # convenience for anything that wants one point per channel; it is NOT what
    # the signed encoder consumes, because a midpoint is identical for A-B and
    # B-A and the ordering is the whole point.
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True).clip(1e-8)
    centre = 0.5 * (unit[pos_idx] + unit[neg_idx])
    centre = centre / np.linalg.norm(centre, axis=1, keepdims=True).clip(1e-8)

    meta = {
        "channel_names": np.array([c.encode() for c in CHANNELS], dtype="S32"),
        "channel_ids": np.array([channel_id(c) for c in CHANNELS], dtype=np.int64),
        "electrode_names": np.array([e.encode() for e in ELECTRODES], dtype="S32"),
        "electrode_xyz": xyz.astype(np.float32),
        "positive_electrode_index": pos_idx,
        "negative_electrode_index": neg_idx,
        "bipolar_endpoints": np.array(
            [[a.encode(), b.encode()] for a, b in BIPOLAR_ENDPOINTS], dtype="S32"),
        "derivation_matrix": DERIVATION_MATRIX.copy(),
        "channel_center_xyz": centre.astype(np.float32),
        "valid_channel_mask": np.ones(len(CHANNELS), dtype=bool),
    }
    attrs = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "coordinate_source": COORDINATE_SOURCE,
        "coordinate_type": COORDINATE_TYPE,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_unit": COORDINATE_UNIT,
        "sampling_rate": float(FS),
        "derivation_type": DERIVATION_TYPE,
        "montage_type": MONTAGE_TYPE,
        "channel_vocab_size": len(CHANNEL_VOCAB),
    }
    attrs["metadata_hash"] = metadata_hash(meta, attrs)
    return {"datasets": meta, "attrs": attrs}


def metadata_hash(datasets: dict, attrs: dict) -> str:
    """A digest over everything a variant must agree on.

    The trainer compares this across train/val/test and across multi-file
    inputs. Two files that differ in a coordinate or an electrode order would
    otherwise concatenate silently into a corpus with no single montage.
    """
    import hashlib
    h = hashlib.sha256()
    for k in sorted(datasets):
        h.update(k.encode())
        h.update(np.ascontiguousarray(datasets[k]).tobytes())
    for k in sorted(attrs):
        if k == "metadata_hash":
            continue
        h.update(f"{k}={attrs[k]}".encode())
    return h.hexdigest()[:32]


def readable_metadata(bundle: dict) -> dict:
    """The same facts as JSON, for split.json and for a human."""
    d, a = bundle["datasets"], bundle["attrs"]
    return {
        **{k: v for k, v in a.items()},
        "channels": [c.decode() for c in d["channel_names"]],
        "channel_ids": d["channel_ids"].tolist(),
        "electrodes": [e.decode() for e in d["electrode_names"]],
        "electrode_xyz": d["electrode_xyz"].tolist(),
        "bipolar_endpoints": [[a_.decode(), b_.decode()]
                              for a_, b_ in d["bipolar_endpoints"]],
        "derivation_matrix": d["derivation_matrix"].tolist(),
        "channel_center_xyz": d["channel_center_xyz"].tolist(),
        "positive_electrode_index": d["positive_electrode_index"].tolist(),
        "negative_electrode_index": d["negative_electrode_index"].tolist(),
        "valid_channel_mask": d["valid_channel_mask"].tolist(),
    }


def write_split(name: str, subjects: list[int], cache_dir: str, out_dir: str,
                meta_bundle: dict | None = None) -> dict:
    """Concatenate cached subjects into one HDF5, growing it incrementally."""
    path = os.path.join(out_dir, f"{name}.h5")
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total = 0
    with h5py.File(path, "w") as f:
        data = f.create_dataset("data", shape=(0, len(CHANNELS), WINDOW_SAMPLES),
                                maxshape=(None, len(CHANNELS), WINDOW_SAMPLES),
                                dtype="float32", chunks=(64, len(CHANNELS), WINDOW_SAMPLES),
                                compression="lzf")
        label = f.create_dataset("label", shape=(0,), maxshape=(None,), dtype="int64")
        subj = f.create_dataset("subject", shape=(0,), maxshape=(None,), dtype="int64")
        for s in subjects:
            npz = os.path.join(cache_dir, f"sub{s:02d}.npz")
            if not os.path.exists(npz):
                print(f"  [{name}] subject {s}: no cache, skipped", file=sys.stderr)
                continue
            with np.load(npz) as z:
                X, y = z["data"], z["label"]
            n = len(y)
            data.resize(total + n, axis=0); data[total:total + n] = X
            label.resize(total + n, axis=0); label[total:total + n] = y
            subj.resize(total + n, axis=0); subj[total:total + n] = s
            counts += np.bincount(y, minlength=len(CLASS_NAMES))
            total += n
        # Unconditional, and before the bundle. These describe the array that
        # was just written, so they are not the channel metadata's to carry --
        # a file built without --channel-metadata still has a sampling rate,
        # and finetune_main refusing to run without one is the failure this
        # placement caused.
        f.attrs["sampling_rate"] = float(FS)
        f.attrs["window_samples"] = int(WINDOW_SAMPLES)
        if meta_bundle is None:
            f.create_dataset("channel_names",
                             data=np.array([c.encode() for c in CHANNELS], dtype="S32"))
        else:
            # One copy per file. channel_names is part of the bundle, so the
            # legacy line above is the no-metadata fallback and not a duplicate.
            for key, arr in meta_bundle["datasets"].items():
                f.create_dataset(key, data=arr)
            for key, val in meta_bundle["attrs"].items():
                f.attrs[key] = val
    share = (counts / max(counts.sum(), 1) * 100)
    print(f"  {name:5s} {len(subjects):2d} subjects  {total:6d} windows  "
          + "  ".join(f"{c}={v}({p:.0f}%)" for c, v, p in zip(CLASS_NAMES, counts, share)))
    return {"subjects": subjects, "windows": total, "class_counts": counts.tolist()}


def preflight() -> None:
    """Import braindecode once, before 64 subjects fail the same way.

    braindecode is used rather than mne directly because the point of this
    dataset is to sit beside EEGPT's number, and their windows come from
    braindecode. An mne-only reimplementation was tried and abandoned: whether
    the raw is preloaded changes where `raw.crop` lands (2508001 samples
    against 2523000 for the same recording), and `create_windows_from_events`
    with drop_last_window=False adds one overlapping window per trial, so a
    hand-rolled version came out 15 windows different on subject 0 alone.
    Fifteen windows is small and comparability is the whole reason this file
    exists, so the dependency stays.
    """
    try:
        import braindecode  # noqa: F401
        from braindecode.datasets import SleepPhysionet  # noqa: F401
        from braindecode.preprocessing import (  # noqa: F401
            Preprocessor, create_windows_from_events, preprocess,
        )
        from sklearn.preprocessing import scale  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n\nThis script needs braindecode (which pulls in mne):\n"
            f"    pip install braindecode"
        ) from exc
    except AttributeError as exc:
        if "pydantic" in str(exc):
            import pydantic
            raise SystemExit(
                f"{exc}\n\n"
                f"braindecode's dependencies need pydantic 2, and pydantic "
                f"{pydantic.VERSION} is what imports here\n"
                f"({os.path.dirname(pydantic.__file__)}).\n\n"
                f"  pip install -U 'pydantic>=2'\n\n"
                f"On a cluster whose shared environment pins pydantic 1, install into\n"
                f"your own environment so it shadows the shared one, and check with:\n\n"
                f"  python -c \"import pydantic; print(pydantic.VERSION, pydantic.__file__)\"\n\n"
                f"braindecode 1.2.0 with pydantic 2.13.4 is a combination known to work."
            ) from exc
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default=None,
                   help="per-subject .npz cache (default: <out-dir>/cache)")
    p.add_argument("--stage", choices=["cache", "split", "all"], default="all")
    p.add_argument("--split", choices=["holdout", "eegpt-fold"], default="holdout")
    p.add_argument("--fold", type=int, default=0, help="0-9, for --split eegpt-fold")
    p.add_argument("--subjects", default=None,
                   help="comma-separated subject ids to cache; default is the "
                        "64 EEGPT runs its folds over")
    p.add_argument("--no-channel-metadata", action="store_true",
                   help="write the pre-channel-embedding HDF5 layout. Only "
                        "--channel_encoding none can train on the result; this "
                        "exists to reproduce a file from before the schema.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="let MNE and braindecode log; off by default because "
                        "64 subjects of filter-design banners bury the failures")
    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    # `--out-dir $PW_DATA_EEG/sleep_edf` with PW_DATA_EEG unset expands to
    # `/sleep_edf` before this process ever starts, and the failure that
    # follows is a permission error on the filesystem root rather than
    # anything about the variable. Same guard as the shell launchers'
    # pw_check_output_dir.
    for label, path in (("--out-dir", args.out_dir),
                        ("--cache-dir", args.cache_dir)):
        if path and os.path.dirname(os.path.abspath(path)) == os.sep:
            raise SystemExit(
                f"{label} is '{path}' -- directly under the filesystem root.\n\n"
                f"  That is almost always an unset variable: your shell expands\n"
                f"  $PW_DATA_EEG before this script runs, and it is empty unless\n"
                f"  you have sourced the environment yourself:\n\n"
                f"      source scripts/cineca_env.sh\n\n"
                f"  Then re-run, or pass an absolute path."
            )

    cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        subjects = list(EEGPT_SUBJECTS)
    bad = sorted(set(subjects) & MISSING)
    if bad:
        raise SystemExit(f"subjects {bad} are not in Sleep-EDFx SC")

    if args.stage in ("cache", "all"):
        preflight()
        print(f"Caching {len(subjects)} subjects into {cache_dir}")
        print("The first subject downloads Sleep-EDFx SC through MNE; expect a wait.")
        import traceback

        from tqdm import tqdm
        failed = []
        for s in tqdm(subjects, ncols=100):
            try:
                cache_subject(s, cache_dir, args.overwrite)
            except SystemExit:
                raise
            except Exception as exc:                       # noqa: BLE001
                # One unreadable recording should not cost the whole corpus,
                # but it must be named -- a silently short dataset looks like
                # a modelling result. The first failure prints in full: a bare
                # repr of an exception from somewhere inside mne/braindecode
                # says nothing about which package raised it, and when every
                # subject fails the same way it is an environment problem that
                # only the traceback identifies.
                if not failed:
                    print("\n\nFirst failure, in full:\n", file=sys.stderr)
                    traceback.print_exc()
                failed.append((s, repr(exc)))
        if failed:
            print(f"\n{len(failed)} of {len(subjects)} subject(s) failed.",
                  file=sys.stderr)
            for s, e in failed[:10]:
                print(f"  subject {s}: {e[:120]}", file=sys.stderr)
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more", file=sys.stderr)
            if len(failed) == len(subjects):
                raise SystemExit(
                    "\nEvery subject failed, so this is the environment rather "
                    "than the data.\nThe traceback above names the package. "
                    "Nothing was written."
                )

    if args.stage in ("split", "all"):
        if args.split == "holdout":
            split = holdout_split(subjects)
            tag = "holdout"
        else:
            split = eegpt_fold_split(args.fold)
            tag = f"eegpt-fold{args.fold}"
        print(f"\nSplit ({tag}), by subject:")
        bundle = None
        if not args.no_channel_metadata:
            try:
                bundle = build_channel_metadata()
            except ImportError as exc:
                raise SystemExit(
                    f"{exc}\n\nThe channel metadata needs mne for the "
                    f"standard_1020 template.\n"
                    f"  Install it, or pass --no-channel-metadata to write the "
                    f"old format\n  (which only --channel_encoding none can train on)."
                ) from exc
            print(f"  channel metadata: schema v{bundle['attrs']['metadata_schema_version']} "
                  f"hash {bundle['attrs']['metadata_hash']} "
                  f"({bundle['attrs']['coordinate_source']})")
        meta = {"split": tag, "fs": FS, "window_samples": WINDOW_SAMPLES,
                "channels": CHANNELS, "classes": CLASS_NAMES, "splits": {}}
        if bundle is not None:
            meta["channel_metadata"] = readable_metadata(bundle)
            meta["metadata_hash"] = bundle["attrs"]["metadata_hash"]
        for name in ("train", "val", "test"):
            meta["splits"][name] = write_split(name, split[name], cache_dir,
                                               args.out_dir, bundle)
        empty = [n for n, m in meta["splits"].items() if m["windows"] == 0]
        if empty:
            # Writing a 0-window HDF5 and printing "Wrote ..." is the failure
            # mode this whole script is meant to avoid: the trainer would load
            # it, report a loss over nothing, and look like it ran.
            for n in empty:
                os.remove(os.path.join(args.out_dir, f"{n}.h5"))
            raise SystemExit(
                f"\n{', '.join(empty)} came out empty -- no cached subject had "
                f"any windows.\nThe cache stage has not run, or it failed. "
                f"Removed the empty file(s); nothing usable was written."
            )
        overlap = (set(split["train"]) & set(split["val"])) | \
                  (set(split["train"]) & set(split["test"])) | \
                  (set(split["val"]) & set(split["test"]))
        assert not overlap, f"subject appears in two splits: {sorted(overlap)}"
        with open(os.path.join(args.out_dir, "split.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nWrote {args.out_dir}/{{train,val,test}}.h5 and split.json")


if __name__ == "__main__":
    main()
