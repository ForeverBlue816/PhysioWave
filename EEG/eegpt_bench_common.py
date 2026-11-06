#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the three benchmark converters share: BCIC-IV-2a, BCIC-IV-2b, KaggleERN.

    EEG/bcic_iv2_finetune.py    --dataset 2a | 2b
    EEG/kaggle_ern_finetune.py

THE PREPROCESSING IS THE PRETRAINING PIPELINE, not EEGPT's. These are the
tasks EEGPT reports, run on the tasks' own recordings, but the signal reaching
the encoder is prepared the way the signal it was pretrained on was -- by the
same functions, from physiowave.eeg_c1.preprocess, with the same defaults:

    microvolts -> linear detrend -> mains notch (50 Hz + harmonic)
    -> 0.5 Hz high-pass -> polyphase resample to 256 Hz
    -> epoch -> per-window, per-channel z-score, clipped at +-20

applied to each continuous run or session before it is epoched, exactly as
EEG/preprocess_pretrain_corpus.py applies it to each recording before it is
windowed. A fine-tuned encoder is being asked to reuse what it learned; handing
it input prepared differently -- EEGPT's 0-38 Hz band-pass, Euclidean
alignment, common average, per-trial min-max -- changes the question to
whether it can also adapt to a new input distribution.

What that means per task, stated because each differs from EEGPT's:

  * no band-pass. BCIC keeps everything above 0.5 Hz up to Nyquist (EEGPT cuts
    at 38 Hz), because the pretraining corpus did.
  * no re-referencing and no Euclidean alignment. Neither was applied in
    pretraining.
  * per-WINDOW z-score. The encoder never saw an un-normalised window. It
    removes each channel's absolute power in the window -- and with it part of
    the C3/C4 power asymmetry motor imagery produces -- but the spectral and
    temporal structure within the window, which is what an ERD and an ERN are
    made of, survive it. It is also close to what EEGPT does on 2a
    (braindecode's exponential moving standardisation, a running per-channel
    z-score).
  * mains 50 Hz: both datasets were recorded in Europe (Graz, Toulouse).

What IS kept from EEGPT is what makes the tasks the same tasks: the epochs
(cue + 0..4 s on BCIC, feedback -0.7..+1.3 s on KaggleERN), the classes, the
channel sets, and the cross-subject structure of their folds. The evaluation
differs too: held-out subjects are a TEST set that selects nothing, validation
is a separate set of subjects, and one fold is run rather than a mean over
folds. EEGPT's rows are printed beside ours as a reference, not as a
like-for-like comparison.

Output, what physiowave.train.finetune_main reads:

    data           (N, C, T) float32
    label          (N,)      int64
    subject        (N,)      int64     provenance; the trainer ignores it
    session        (N,)      int64     provenance; the trainer ignores it
    channel_names  (C,)      bytes     the montage, by name, in order
    attrs          sampling_rate, window_samples, prep_version, provenance
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.eeg_c1 import preprocess as pp                  # noqa: E402

FS_OUT = 256
MAINS_HZ = 50.0
#: Two preprocessings, each with a version stamped into every file and into the
#: split directory's name, so a split built by one is never picked up by a run
#: that asked for the other.
#:
#:   pretrain  the pretraining pipeline (the default; see the top of this file)
#:   eegpt     EEGPT's: 0-38 Hz low-pass, then per session Euclidean alignment
#:             and a common average. BCIC only. Kept because on BCIC the
#:             pretraining pipeline trailed EEGPT by a wide margin, and EA --
#:             which aligns each subject's spatial covariance to the identity --
#:             is the one step with no counterpart in pretraining that is known
#:             to matter for cross-subject motor imagery.
PREPS = {"pretrain": "pretrain-v1", "eegpt": "eegpt-v1"}
PREP_VERSION = PREPS["pretrain"]
EEGPT_LOWPASS_HZ = 38.0
#: The pretraining defaults, one object, so nothing here restates a number.
CFG = pp.PreprocessConfig(notch_hz=MAINS_HZ)


def pretrain_pipeline(x: np.ndarray, fs: float, unit: str = "uV") -> np.ndarray:
    """A continuous ``[C, T]`` recording, prepared as pretraining prepared one.

    The same calls, in the same order, as `process_recording` in
    EEG/preprocess_pretrain_corpus.py, up to (not including) windowing.
    """
    x = pp.to_microvolts(np.asarray(x, dtype=np.float64), unit)
    x = pp.detrend(x, CFG.detrend)
    x = pp.notch(x, fs, CFG.notch_hz, CFG.notch_harmonics, CFG.notch_quality)
    x = pp.highpass(x, fs, CFG.highpass_hz)
    return pp.resample_to(x, fs, FS_OUT)


def normalise_windows(epochs: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score, clipped -- pretraining's zscore_windows."""
    valid = np.ones(epochs.shape[1], dtype=bool)
    return pp.zscore_windows(epochs, valid, CFG.zscore_eps, CFG.clip_sigma)


def provenance(extra: Dict, prep: str = "pretrain") -> Dict:
    if prep == "pretrain":
        return CFG.provenance({"prep": prep, "prep_version": PREPS[prep],
                               "target_sampling_rate": FS_OUT, **extra})
    return {"prep": prep, "prep_version": PREPS[prep],
            "pipeline": (f"butterworth low-pass {EEGPT_LOWPASS_HZ:g} Hz (order 4, "
                         f"zero-phase) -> resample_poly {FS_OUT} Hz -> epoch -> "
                         f"Euclidean alignment per session file -> common "
                         f"average -> clip +-{CFG.clip_sigma:g}"),
            "target_sampling_rate": FS_OUT, **extra}


# --------------------------------------------------------------------------- #
# The EEGPT preprocessing (--prep eegpt), BCIC only. Restored as it was before
# the pretraining pipeline replaced it: same filter, same order, same checks.
# --------------------------------------------------------------------------- #

def eegpt_continuous(x: np.ndarray, fs: float) -> np.ndarray:
    """0-38 Hz and 256 Hz, on a CONTINUOUS run.

    MNE's `filter(0, 38, method='iir')` is a 4th-order Butterworth run forward
    and backward, which is what this is. EEGPT applies it to 4 s epochs; here
    it runs on the whole run before epoching, so the edge transient falls at
    the ends of the recording rather than at the ends of every trial.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, EEGPT_LOWPASS_HZ / (fs / 2.0), btype="lowpass", output="sos")
    return pp.resample_to(sosfiltfilt(sos, np.asarray(x, np.float64), axis=-1),
                          fs, FS_OUT)


def euclidean_alignment(epochs: np.ndarray) -> np.ndarray:
    """``R^{-1/2} x`` with ``R`` the mean trial covariance of this set.

    EEGPT's `Data_process.utils.EA`, applied the way they apply it: once per
    session file, over every trial in it. Label-free, so it is applied to the
    test subject's sessions too -- as theirs is -- using that subject's own
    unlabelled trials.

    MUST RUN BEFORE the common-average reference, which is EEGPT's order too.
    A common average makes the channels sum to zero, so R loses a rank:
    singular outright on 2b's three channels, singular to within rounding on
    2a's twenty-two -- where R^{-1/2} still comes out finite. Hence eigh and an
    explicit conditioning check rather than a finiteness test.
    """
    x = epochs.astype(np.float64)
    R = np.einsum("nct,ndt->cd", x, x) / len(x)
    w, V = np.linalg.eigh((R + R.T) / 2.0)
    if w.min() <= w.max() * 1e-8:
        raise ValueError(
            f"Euclidean alignment: the session covariance is rank-deficient "
            f"(eigenvalues {w.min():.3g} .. {w.max():.3g}). A common-average "
            f"reference applied before EA does this; so does a flat or "
            f"duplicated channel.")
    # errstate: numpy's matmul on some BLAS builds reports divide-by-zero on
    # well-conditioned input. Conditioning is checked above, the outcome below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Rm = (V / np.sqrt(w)) @ V.T
        out = np.einsum("dc,nct->ndt", Rm, x)
    if not (np.all(np.isfinite(Rm)) and np.all(np.isfinite(out))):
        raise ValueError("Euclidean alignment produced non-finite values")
    return out


def common_average(x: np.ndarray) -> np.ndarray:
    """Subtract the channel mean at every sample -- AFTER alignment."""
    return x - x.mean(axis=-2, keepdims=True)


def eegpt_session(epochs: np.ndarray) -> np.ndarray:
    """One session file's epochs: EA, then common average, then the ±20 clip."""
    x = common_average(euclidean_alignment(epochs))
    return np.clip(x, -CFG.clip_sigma, CFG.clip_sigma).astype(np.float32)


def epoch_at(x: np.ndarray, onsets: Sequence[int], length: int) -> np.ndarray:
    """``[C, T] -> [N, C, length]``. An epoch running off either end is an error.

    Not clipped and not padded: a short epoch stretched or zero-filled to length
    is a different duration passed off as the same one.
    """
    C, T = x.shape
    out = np.empty((len(onsets), C, length), dtype=np.float32)
    for i, s in enumerate(onsets):
        s = int(s)
        if s < 0 or s + length > T:
            raise ValueError(f"epoch {i} at sample {s} (+{length}) runs off a "
                             f"{T}-sample recording")
        out[i] = x[:, s:s + length]
    return out


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def loso_split(subjects: List[int], fold: int, seed: int = 7) -> Dict[str, List[int]]:
    """Subject ``subjects[fold]`` is the test set; one more becomes validation.

    The same rule as physio_p300_finetune.py. EEGPT's BCIC fold trains on the
    other eight subjects and scores the ninth, validating on it as well; here
    the ninth is the TEST set, and a validation subject is drawn from the eight
    deterministically, so the split is 7/1/1.
    """
    if not 0 <= fold < len(subjects):
        raise SystemExit(f"--fold must be in [0, {len(subjects)}), got {fold}")
    if len(subjects) < 3:
        raise SystemExit(f"a subject-level train/val/test split needs at least "
                         f"3 subjects, got {len(subjects)}")
    test = [subjects[fold]]
    rest = [s for s in subjects if s != subjects[fold]]
    rng = np.random.default_rng(seed + fold)
    val = [rest[int(rng.integers(len(rest)))]]
    train = [s for s in rest if s not in val]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_split(path: str, epochs: List[np.ndarray], labels: List[np.ndarray],
                subjects: List[np.ndarray], sessions: List[np.ndarray],
                channels: Sequence[str], class_names: Sequence[str],
                prov: Dict) -> Dict:
    """One HDF5 in the layout finetune_main reads. Returns a summary row."""
    import h5py

    if not epochs:
        raise SystemExit(f"{path}: no epochs -- refusing to write an empty split")
    X = np.concatenate(epochs).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    s = np.concatenate(subjects).astype(np.int64)
    k = np.concatenate(sessions).astype(np.int64)
    if X.shape[1] != len(channels):
        raise SystemExit(f"{path}: {X.shape[1]} channels in the data, "
                         f"{len(channels)} names")
    if not np.all(np.isfinite(X)):
        raise SystemExit(f"{path}: non-finite values in the epochs")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X, chunks=(min(64, len(X)),) + X.shape[1:],
                         compression="lzf")
        f.create_dataset("label", data=y)
        f.create_dataset("subject", data=s)
        f.create_dataset("session", data=k)
        f.create_dataset("channel_names",
                         data=np.array([c.encode() for c in channels], dtype="S32"))
        f.attrs["sampling_rate"] = float(FS_OUT)
        f.attrs["window_samples"] = int(X.shape[-1])
        f.attrs["prep_version"] = prov.get("prep_version", PREP_VERSION)
        f.attrs["class_names"] = json.dumps(list(class_names))
        f.attrs["provenance"] = json.dumps(prov, default=str)

    counts = np.bincount(y, minlength=len(class_names))
    return {"file": os.path.basename(path), "epochs": int(len(y)),
            "subjects": sorted(set(s.tolist())),
            "class_counts": {c: int(n) for c, n in zip(class_names, counts)},
            "shape": list(X.shape)}


def check_vocabulary(channels: Sequence[str]) -> None:
    """Refuse a montage the channel-name embedding has no rows for.

    An unknown name would resolve to UNK and share one embedding row with every
    other unknown name -- a montage quietly losing its identity.
    """
    from channel_embedding import UNK_ID, channel_id
    unknown = [c for c in channels if channel_id(c) == UNK_ID]
    if unknown:
        raise SystemExit(f"channel(s) not in the vocabulary: {unknown}")


def print_summary(rows: List[Dict]) -> None:
    for r in rows:
        cc = "  ".join(f"{k}={v}" for k, v in r["class_counts"].items())
        print(f"  {r['file']:9s} {r['epochs']:6d} epochs  subjects "
              f"{r['subjects']}  {cc}  shape {tuple(r['shape'])}")
