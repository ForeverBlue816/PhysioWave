"""
The preprocessing every corpus goes through, and the HDF5 it writes.

The adapters differ only in how a recording is *read*. Everything after that --
units, detrending, notch, resampling, slot mapping, windowing, normalisation --
is this file, applied identically, so that a difference between two datasets in
training is a difference between the datasets and not between two people's idea
of a filter.

Deliberate choices worth stating, because the usual alternatives are worse here:

* **No 0.5-45 Hz band-pass.** That band is a sleep-staging and ERP convention.
  It would delete the gamma range this corpus contains on purpose -- HGD is a
  high-gamma dataset -- and a pretrained encoder that has never seen above 45 Hz
  cannot be fine-tuned onto anything that needs it. The default is a 0.5 Hz
  high-pass and whatever anti-aliasing the resampler needs, and nothing else.

* **Polyphase resampling.** ``scipy.signal.resample_poly`` filters before it
  decimates. Plain slicing or ``scipy.signal.resample`` (which is FFT-based and
  assumes periodicity) both alias mains harmonics and muscle activity down into
  the EEG band, and the result looks like clean data.

* **Per-window z-score, not per-recording.** Amplitude drifts over a session
  with impedance and with the subject. Normalising per window makes the model's
  input scale-free at the scale the model actually sees.

* **Incomplete tail windows are dropped, never zero-padded.** A padded tail is a
  training example that is part signal and part fiction, and the model cannot
  tell which half to trust.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from channel_embedding import channel_ids_for, normalize_channel_name
from .routes import Route


@dataclass
class PreprocessConfig:
    """Everything that changes the numbers, recorded into every shard."""

    highpass_hz: float = 0.5
    notch_hz: Optional[float] = None        # 50, 60, or None; never guessed
    notch_harmonics: int = 2
    notch_quality: float = 30.0
    clip_sigma: float = 20.0
    zscore_eps: float = 1e-6
    window_seconds: float = 4.0
    stride_seconds: Optional[float] = None  # None -> no overlap
    val_fraction: float = 0.10
    split_seed: int = 42
    detrend: str = "linear"                 # 'linear' | 'constant' | 'none'

    def stride(self) -> float:
        return self.window_seconds if self.stride_seconds is None else self.stride_seconds

    def provenance(self, extra: Optional[Dict] = None) -> Dict:
        d = asdict(self)
        d["pipeline"] = ("units_uV -> detrend -> notch -> highpass -> "
                         "resample_poly -> slot_map -> window -> zscore_clip")
        d["resampler"] = "scipy.signal.resample_poly (polyphase, anti-aliased)"
        d["band_pass"] = "none (0.5 Hz high-pass + resampling anti-alias only)"
        if extra:
            d.update(extra)
        d["config_sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in sorted(d.items())},
                       default=str).encode()).hexdigest()
        return d


class PreprocessError(RuntimeError):
    """A recording that cannot be processed. Recorded, never silently skipped."""


# --------------------------------------------------------------------------- #
# Signal steps
# --------------------------------------------------------------------------- #

def to_microvolts(x: np.ndarray, unit: str) -> np.ndarray:
    """Bring a recording to µV. ``unit`` is what the file says it is in."""
    scale = {"uv": 1.0, "µv": 1.0, "microvolt": 1.0,
             "mv": 1e3, "v": 1e6}.get(unit.strip().lower())
    if scale is None:
        raise PreprocessError(
            f"unknown unit {unit!r}; say uV, mV or V rather than let a "
            f"factor of a million pass as a scaling difference")
    return x.astype(np.float64) * scale


def detrend(x: np.ndarray, mode: str) -> np.ndarray:
    """Remove DC or a linear trend, per channel.

    scipy 1.15's detrend emits a divide-by-zero RuntimeWarning from an internal
    matmul path on ordinary well-conditioned input, and on a corpus the size of
    TUEG that warning is printed once per recording for no reason. It is
    silenced here and replaced with something that actually checks the outcome:
    a finiteness assertion on the result. A flat channel really can produce a
    non-finite trend fit, and that is a recording to fail rather than to pass on
    as NaN that only surfaces as a NaN loss thousands of steps later.
    """
    if mode == "none":
        return x
    from scipy.signal import detrend as _d
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = _d(x, axis=-1, type="linear" if mode == "linear" else "constant")
    if not np.all(np.isfinite(y)):
        bad = np.where(~np.isfinite(y).all(axis=-1))[0]
        raise PreprocessError(
            f"detrending produced non-finite values on channel row(s) "
            f"{bad[:8].tolist()} -- typically a flat or all-NaN channel in the "
            f"source recording.")
    return y


def notch(x: np.ndarray, fs: float, freq: Optional[float], harmonics: int,
          q: float) -> np.ndarray:
    """IIR notch at the mains frequency and its harmonics below Nyquist."""
    if not freq:
        return x
    from scipy.signal import filtfilt, iirnotch
    y = x
    for k in range(1, max(1, harmonics) + 1):
        f0 = freq * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(f0, q, fs)
        y = filtfilt(b, a, y, axis=-1)
    return y


def highpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Zero-phase Butterworth high-pass. Removes drift, keeps everything above."""
    if not cutoff:
        return x
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, x, axis=-1)


def resample_to(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    """Polyphase resample with the anti-alias filter the ratio requires."""
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from fractions import Fraction
    from scipy.signal import resample_poly
    ratio = Fraction(int(round(fs_out)), int(round(fs_in))).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


# --------------------------------------------------------------------------- #
# Montage -> canonical slots
# --------------------------------------------------------------------------- #

@dataclass
class SlotMapping:
    """How a recording's channels landed on a route's canonical slots."""

    matrix_rows: List[int]              # source row for each filled slot
    slot_of_row: List[int]              # which slot each filled row went to
    valid: np.ndarray                   # [n_slots] bool
    unmatched_sources: List[str]        # recorded channels with no slot
    empty_slots: List[str]              # slots no channel filled
    unknown_names: List[str]            # names outside the vocabulary


def map_to_slots(channel_names: Sequence[str], slots: Sequence[str]) -> SlotMapping:
    """Place channels into the route's slots **by name**.

    Never by position. Two datasets that both report "26 channels" do not have
    the same 26, and lining them up by index would train one electrode's
    embedding on another electrode's signal. A slot nothing fills stays zero and
    is marked invalid; a recorded channel that fits no slot is dropped and
    named in the return, because dropping data quietly is how a montage silently
    becomes a different montage.
    """
    canonical = [normalize_channel_name(n) for n in channel_names]
    slot_index = {s: i for i, s in enumerate(slots)}
    matrix_rows: List[int] = []
    slot_of_row: List[int] = []
    valid = np.zeros(len(slots), dtype=bool)
    unmatched: List[str] = []
    taken: Dict[int, int] = {}

    for row, name in enumerate(canonical):
        j = slot_index.get(name)
        if j is None:
            unmatched.append(str(channel_names[row]))
            continue
        if j in taken:
            # The same electrode twice: keep the first and say so, rather than
            # let the second overwrite it depending on file order.
            unmatched.append(f"{channel_names[row]} (duplicate of slot {slots[j]})")
            continue
        taken[j] = row
        matrix_rows.append(row)
        slot_of_row.append(j)
        valid[j] = True

    _, unknown = channel_ids_for(channel_names)
    empty = [s for i, s in enumerate(slots) if not valid[i]]
    return SlotMapping(matrix_rows, slot_of_row, valid, unmatched, empty, unknown)


def place_on_slots(x: np.ndarray, mapping: SlotMapping, n_slots: int) -> np.ndarray:
    """``[C_src, T] -> [n_slots, T]``, zeros where no channel was recorded."""
    out = np.zeros((n_slots, x.shape[-1]), dtype=x.dtype)
    for src_row, slot in zip(mapping.matrix_rows, mapping.slot_of_row):
        out[slot] = x[src_row]
    return out


# --------------------------------------------------------------------------- #
# Windowing and normalisation
# --------------------------------------------------------------------------- #

def window_signal(x: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """``[C, T] -> ([N, C, window], [N] start sample)``. The tail is dropped.

    A recording shorter than one window yields nothing, which is correct: there
    is no 4-second example in it, and zero-padding one into existence would be
    training on a signal that is partly a constant.
    """
    C, T = x.shape
    if T < window:
        return (np.zeros((0, C, window), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    starts = np.arange(0, T - window + 1, stride, dtype=np.int64)
    out = np.stack([x[:, s:s + window] for s in starts]).astype(np.float32)
    return out, starts


def zscore_windows(w: np.ndarray, valid: np.ndarray, eps: float,
                   clip_sigma: float) -> np.ndarray:
    """Per window, per valid channel. Invalid slots are left exactly zero.

    Normalising a padded slot would divide zeros by ~eps and turn a slot that
    holds no measurement into whatever numerical noise survives, which then
    reaches the model looking like signal.
    """
    out = w.astype(np.float32, copy=True)
    v = np.asarray(valid, dtype=bool)
    mu = out[:, v, :].mean(axis=-1, keepdims=True)
    sd = out[:, v, :].std(axis=-1, keepdims=True)
    z = (out[:, v, :] - mu) / np.maximum(sd, eps)
    if clip_sigma and clip_sigma > 0:
        z = np.clip(z, -clip_sigma, clip_sigma)
    out[:, v, :] = z
    out[:, ~v, :] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #

def window_qc(w: np.ndarray, valid: np.ndarray, fs: int) -> Dict[str, float]:
    """Amplitude and spectral summary of one recording's windows."""
    if w.size == 0:
        return {}
    v = np.asarray(valid, dtype=bool)
    sig = w[:, v, :]
    qc = {
        "n_windows": int(w.shape[0]),
        "amplitude_mean": float(sig.mean()),
        "amplitude_std": float(sig.std()),
        "amplitude_min": float(sig.min()),
        "amplitude_max": float(sig.max()),
        "channel_missing_rate": float(1.0 - v.mean()),
    }
    # PSD of a bounded sample: enough to see a dead channel or a mains spike,
    # cheap enough to run on every recording.
    take = sig[: min(64, sig.shape[0])]
    freqs = np.fft.rfftfreq(take.shape[-1], d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(take, axis=-1)) ** 2).mean(axis=(0, 1))
    for lo, hi, name in ((0.5, 4, "delta"), (4, 8, "theta"), (8, 13, "alpha"),
                         (13, 30, "beta"), (30, 45, "gamma"),
                         (45, min(100, fs / 2 - 1), "high_gamma")):
        band = (freqs >= lo) & (freqs < hi)
        qc[f"psd_{name}"] = float(psd[band].mean()) if band.any() else 0.0
    return qc


# --------------------------------------------------------------------------- #
# Subject-level split
# --------------------------------------------------------------------------- #

def split_subjects(subjects: Sequence[str], val_fraction: float,
                   seed: int) -> Tuple[List[str], List[str]]:
    """Partition SUBJECTS, never windows.

    A window-level split puts two windows of the same recording on both sides,
    and the val loss then measures memorisation of a subject rather than
    generalisation to one. Sorted before shuffling so the partition depends on
    the seed and not on filesystem order.
    """
    uniq = sorted({str(s) for s in subjects})
    if len(uniq) < 2:
        return uniq, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val = sorted(uniq[i] for i in order[:n_val])
    train = sorted(uniq[i] for i in order[n_val:])
    return train, val


def subject_split_side(subject_id: str, val_fraction: float, seed: int) -> str:
    """``"train"`` or ``"val"`` for one subject, decided without seeing the rest.

    TUEG is preprocessed as a SLURM array: no single process sees the whole
    subject list, so split_subjects' shuffle-and-cut cannot be used -- each task
    would shuffle its own subjects and the same subject could land in train in
    one shard and val in another, which is the leak the subject-level split
    exists to prevent.

    Hashing the subject id instead makes the side a property of the subject.
    Any task, in any order, on any number of shards, and on a re-run months
    later, puts a given subject on the same side. SHA-256 rather than hash():
    Python's string hash is salted per process and would give a different
    partition on every task.
    """
    h = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    # First 8 bytes as a fraction of the range, compared against the target.
    frac = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "val" if frac < val_fraction else "train"


# --------------------------------------------------------------------------- #
# HDF5
# --------------------------------------------------------------------------- #

def write_shard(path: str, windows: np.ndarray, route: Route,
                dataset_id: str, channel_names: Sequence[str],
                channel_ids: Sequence[int], valid: np.ndarray,
                subject_ids: Sequence[str], recording_ids: Sequence[str],
                window_starts: Sequence[float], source_rate: float,
                provenance: Dict) -> Dict:
    """One HDF5 per recording (or per shard). Never one file for a corpus.

    Recording-level files are what make TUEG loadable at all: the index holds
    counts, the trainer opens the handful of files a step touches, and nothing
    ever has to hold the corpus in memory.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = int(windows.shape[0])
    # Write to a private name and rename on success. A shard is gzip-chunked,
    # and HDF5 lays the header down before the chunks, so a task killed
    # part-way through (a walltime, an OOM, a node failure) used to leave a
    # file at the FINAL path whose shape and attributes read back perfectly and
    # whose data raised "inflate() failed" on the first training epoch that
    # touched it -- months later, on sixteen ranks at once. os.replace is
    # atomic within a filesystem, so the final path now only ever names a file
    # that was written all the way through.
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        _write_shard_file(tmp, windows, route, dataset_id, channel_names,
                          channel_ids, valid, subject_ids, recording_ids,
                          window_starts, source_rate, provenance, n)
    except BaseException:
        # A signal cannot be caught here, which is the case this design is
        # actually for; this only keeps a failed run from littering scratch.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    return {"path": path, "dataset_id": dataset_id, "route_id": route.route_id,
            "n_windows": n, "subjects": sorted({str(s) for s in subject_ids})}


def _write_shard_file(path: str, windows: np.ndarray, route: Route,
                      dataset_id: str, channel_names: Sequence[str],
                      channel_ids: Sequence[int], valid: np.ndarray,
                      subject_ids: Sequence[str], recording_ids: Sequence[str],
                      window_starts: Sequence[float], source_rate: float,
                      provenance: Dict, n: int) -> None:
    """The bytes themselves. Separate so write_shard owns only the publish."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=windows.astype(np.float32),
                         compression="gzip", compression_opts=4,
                         chunks=(1,) + windows.shape[1:] if n else None)
        f.create_dataset("channel_names",
                         data=np.array([str(c) for c in channel_names],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("channel_ids",
                         data=np.asarray(channel_ids, dtype=np.int64))
        f.create_dataset("valid_channel_mask",
                         data=np.asarray(valid, dtype=bool))
        f.create_dataset("subject_id",
                         data=np.array([str(s) for s in subject_ids],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("recording_id",
                         data=np.array([str(s) for s in recording_ids],
                                       dtype=h5py.string_dtype()))
        f.create_dataset("window_start_seconds",
                         data=np.asarray(window_starts, dtype=np.float64))
        f.attrs["dataset_id"] = dataset_id
        f.attrs["route_id"] = route.route_id
        f.attrs["source_sampling_rate"] = float(source_rate)
        f.attrs["target_sampling_rate"] = float(route.sampling_rate)
        f.attrs["preprocessing_provenance"] = json.dumps(provenance)
