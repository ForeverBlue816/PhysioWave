"""The DB5 converter must not leak raw samples between splits.

Overlapping windows are only safe because the split is decided per
(movement, repetition) segment *before* the windows are cut. Cut the windows
first and split them afterwards -- the protocol most DB5 papers with 90%+
numbers use -- and adjacent windows share half their samples across the
train/test boundary, which is worth tens of points of apparent accuracy.

These tests encode each sample's index as its own value, so a shared sample
between two splits is directly observable rather than inferred.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
sio = pytest.importorskip("scipy.io")

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "EMG" / "db5_finetune.py"


def _write_subject(root: Path, subject: int, first_sample: int,
                   movements: int = 3, reps: int = 6,
                   rest: int = 200, hold: int = 900) -> int:
    """One fake DB5 exercise file whose EMG values are global sample indices."""
    stim: list[int] = []
    rep: list[int] = []
    for m in range(1, movements + 1):
        for r in range(1, reps + 1):
            stim += [0] * rest + [m] * hold
            rep += [r] * (rest + hold)
    n = len(stim)
    emg = np.tile(np.arange(first_sample, first_sample + n, dtype=np.float32)[:, None],
                  (1, 16))
    d = root / f"s{subject}"
    d.mkdir(parents=True, exist_ok=True)
    sio.savemat(d / f"S{subject}_E1_A1.mat", {
        "emg": emg,
        "restimulus": np.array(stim, dtype=np.int8)[:, None],
        "rerepetition": np.array(rep, dtype=np.int8)[:, None],
        "stimulus": np.array(stim, dtype=np.int8)[:, None],
        "repetition": np.array(rep, dtype=np.int8)[:, None],
        "exercise": np.array([[1]], dtype=np.uint8),
    })
    return n


def _convert(root: Path, out: Path, *extra: str) -> None:
    cmd = [sys.executable, str(CONVERTER), "--root", str(root), "--out-dir", str(out),
           "--window", "512", "--stride", "32", "--normalize", "none",
           "--rest-ratio", "0", "--exercises", "1", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _samples(path: Path) -> set[int]:
    with h5py.File(path) as f:
        data = f["data"][:]
    seen: set[int] = set()
    for window in data:
        seen.update(window[0].astype(np.int64).tolist())
    return seen


def test_repetition_split_shares_no_samples(tmp_path):
    _write_subject(tmp_path / "raw", 1, 0)
    out = tmp_path / "out"
    _convert(tmp_path / "raw", out)

    splits = {name: _samples(out / f"{name}.h5") for name in ("train", "val", "test")}
    assert all(splits.values()), "a split came out empty"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = splits[a] & splits[b]
        assert not shared, f"{len(shared)} raw samples shared between {a} and {b}"


def test_subject_split_holds_out_whole_subjects(tmp_path):
    raw = tmp_path / "raw"
    bounds = {}
    start = 0
    for subject in (1, 2, 3):
        n = _write_subject(raw, subject, start)
        bounds[subject] = (start, start + n)
        start += n

    out = tmp_path / "out"
    _convert(raw, out, "--split-by", "subject",
             "--val-subjects", "2", "--test-subjects", "3")

    for name, subject in (("train", 1), ("val", 2), ("test", 3)):
        lo, hi = bounds[subject]
        got = _samples(out / f"{name}.h5")
        assert got, f"{name} came out empty"
        assert min(got) >= lo and max(got) < hi, \
            f"{name} contains samples from outside subject {subject}"


def test_overlapping_split_units_are_rejected(tmp_path):
    _write_subject(tmp_path / "raw", 1, 0)
    proc = subprocess.run(
        [sys.executable, str(CONVERTER), "--root", str(tmp_path / "raw"),
         "--out-dir", str(tmp_path / "out"), "--train-reps", "1", "2",
         "--val-reps", "2"],
        capture_output=True, text=True)
    assert proc.returncode == 1
    assert "more than one split" in proc.stderr


def test_window_must_divide_patch_size(tmp_path):
    _write_subject(tmp_path / "raw", 1, 0)
    proc = subprocess.run(
        [sys.executable, str(CONVERTER), "--root", str(tmp_path / "raw"),
         "--out-dir", str(tmp_path / "out"), "--window", "500", "--patch-size", "64"],
        capture_output=True, text=True)
    assert proc.returncode == 1
    assert "not a multiple of" in proc.stderr
