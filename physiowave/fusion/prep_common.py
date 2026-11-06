"""What the multimodal benchmark preprocessing scripts share.

THE WINDOW LAYOUT. One benchmark window is the same stretch of time seen by
up to three modalities, each stored at ITS ENCODER'S pretraining rate and
normalised the way THAT corpus was, so the pretrained frontends meet the
input distribution they were trained on:

    modality   rate     patch        normalisation (physiowave.<m>_c1.preprocess)
    eeg        256 Hz   0.5 s        per channel: z-score, clip 20 sigma
    ecg        250 Hz   0.5 s        window-shared: demean, one RMS, clip
    emg        2000 Hz  0.125 s      window-shared, as ECG's

A 4 s window is therefore eeg [C, 1024], ecg [1, 1000], emg [C, 8000] -- a
whole number of patches on every route, which is all the encoders require.

THE HDF5 LAYOUT (one file per split):

    <m>              (N, C_m, T_m) float32   zeros when the modality is absent
    present_<m>      (N,) bool
    label            (N,) int64
    subject, segment (N,) bytes
    attrs            task, class_names (json), window_seconds, prep_version,
                     <m>_sampling_rate, <m>_channel_names (json)

Absence is a FLAG, never a zero vector the model has to recognise: the
dataset hands the flag to the fusion as a mask (physiowave.fusion.data).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.eeg_c1.preprocess import (detrend as _eeg_detrend,  # noqa: E402
                                          highpass as _eeg_highpass,
                                          notch as _eeg_notch, resample_to)
from physiowave.ecg_c1.preprocess import (highpass as _ecg_highpass,  # noqa: E402
                                          notch as _ecg_notch)

MODALITIES: Tuple[str, ...] = ("eeg", "ecg", "emg")
#: The encoders' rates and patch lengths (samples), from their route tables.
RATES: Dict[str, int] = {"eeg": 256, "ecg": 250, "emg": 2000}
PATCH: Dict[str, int] = {"eeg": 128, "ecg": 125, "emg": 250}
PREP_VERSION = "fusion-1"
CLIP_SIGMA = 20.0
EPS = 1e-6


def window_samples(modality: str, seconds: float) -> int:
    n = int(round(seconds * RATES[modality]))
    if n % PATCH[modality]:
        raise ValueError(f"a {seconds} s window is {n} {modality} samples, not a "
                         f"whole number of {PATCH[modality]}-sample patches")
    return n


def filter_continuous(x: np.ndarray, fs: float, modality: str,
                      mains_hz: Optional[float]) -> np.ndarray:
    """The modality's pretraining filters on a CONTINUOUS record, [C, T].

    Done on the record, not the window, so the filters' transients fall
    outside every window. EEG: linear detrend, notch, 0.5 Hz high-pass. ECG:
    notch, 0.5 Hz. sEMG: notch with three harmonics, 20 Hz high-pass.
    """
    x = np.asarray(x, dtype=np.float64)
    if modality == "eeg":
        x = _eeg_detrend(x, "linear")
        x = _eeg_notch(x, fs, mains_hz, 2, 30.0)
        return _eeg_highpass(x, fs, 0.5)
    if modality == "ecg":
        x = _ecg_notch(x, fs, mains_hz, 2, 30.0)
        return _ecg_highpass(x, fs, 0.5)
    if modality == "emg":
        x = _ecg_notch(x, fs, mains_hz, 3, 30.0)
        return _ecg_highpass(x, fs, 20.0)
    raise ValueError(modality)


def to_encoder_rate(x: np.ndarray, fs: float, modality: str) -> np.ndarray:
    """Polyphase resample [C, T] to the modality's pretraining rate."""
    return resample_to(np.asarray(x, dtype=np.float64), float(fs), RATES[modality])


def normalise_window(w: np.ndarray, modality: str) -> np.ndarray:
    """One window [C, T] -> the stored representation, as its corpus did it.

    A channel with no variance (a flat or absent electrode) stays exactly
    zero rather than becoming eps-scaled noise.
    """
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    if modality == "eeg":
        sd = x.std(axis=-1, keepdims=True)
        live = sd > EPS
        x = np.where(live, x / np.maximum(sd, EPS), 0.0)
    else:
        sd = np.sqrt((x ** 2).mean())
        x = x / sd if sd > EPS else np.zeros_like(x)
    return np.clip(x, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)


def normalise_batch(w: np.ndarray, modality: str) -> np.ndarray:
    """``normalise_window`` over ``[N, C, T]`` at once (continuous inference)."""
    x = np.asarray(w, dtype=np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    if modality == "eeg":
        sd = x.std(axis=-1, keepdims=True)
        x = np.where(sd > EPS, x / np.maximum(sd, EPS), 0.0)
    else:
        sd = np.sqrt((x ** 2).mean(axis=(-1, -2), keepdims=True))
        x = np.where(sd > EPS, x / np.maximum(sd, EPS), 0.0)
    return np.clip(x, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)


def cut(x: np.ndarray, fs: float, start_s: float, seconds: float) -> Optional[np.ndarray]:
    """``[C, T]`` from ``start_s`` for ``seconds``, or None if it runs off the end."""
    a = int(round(start_s * fs))
    b = a + int(round(seconds * fs))
    if a < 0 or b > x.shape[-1]:
        return None
    return x[:, a:b]


def prepare_window(seg: np.ndarray, fs: float, modality: str, seconds: float) -> np.ndarray:
    """A cut segment at its native rate -> ``[C, T_m]`` normalised at the encoder rate.

    Resampling rounds; the window is trimmed or zero-padded to the exact
    sample count the patch grid needs (at most a sample or two).
    """
    y = to_encoder_rate(seg, fs, modality)
    n = window_samples(modality, seconds)
    if y.shape[-1] > n:
        y = y[:, :n]
    elif y.shape[-1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[-1])))
    return normalise_window(y, modality)


# --------------------------------------------------------------------------- #
# HDF5 writer
# --------------------------------------------------------------------------- #

class MultimodalWriter:
    """Append windows to one split's HDF5 file."""

    def __init__(self, path: str, task: str, seconds: float, class_names: Sequence[str],
                 channels: Dict[str, Sequence[str]], extra_attrs: Optional[Dict] = None):
        import h5py
        self.path = path
        self.channels = {m: list(n) for m, n in channels.items()}
        self.seconds = float(seconds)
        self.n = 0
        self.f = h5py.File(path, "w")
        for m, names in self.channels.items():
            T = window_samples(m, seconds)
            self.f.create_dataset(m, shape=(0, len(names), T), maxshape=(None, len(names), T),
                                  dtype="float32", chunks=(16, len(names), T),
                                  compression="gzip", compression_opts=1)
            self.f.create_dataset(f"present_{m}", shape=(0,), maxshape=(None,), dtype="bool")
            self.f.attrs[f"{m}_sampling_rate"] = float(RATES[m])
            self.f.attrs[f"{m}_channel_names"] = json.dumps(list(names))
        self.f.create_dataset("label", shape=(0,), maxshape=(None,), dtype="int64")
        str_t = h5py.special_dtype(vlen=bytes)
        self.f.create_dataset("subject", shape=(0,), maxshape=(None,), dtype=str_t)
        self.f.create_dataset("segment", shape=(0,), maxshape=(None,), dtype=str_t)
        self.f.attrs["task"] = task
        self.f.attrs["class_names"] = json.dumps(list(class_names))
        self.f.attrs["window_seconds"] = self.seconds
        self.f.attrs["prep_version"] = PREP_VERSION
        self.f.attrs["modalities"] = json.dumps(list(self.channels))
        for k, v in (extra_attrs or {}).items():
            self.f.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
        self._buf: List[dict] = []

    def add(self, item: dict) -> None:
        """``item``: label, subject, segment, and per modality an array or None."""
        self._buf.append(item)
        if len(self._buf) >= 256:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        k = len(self._buf)
        a, b = self.n, self.n + k
        for m, names in self.channels.items():
            d = self.f[m]
            d.resize(b, axis=0)
            block = np.zeros((k, len(names), d.shape[-1]), np.float32)
            pres = np.zeros(k, bool)
            for i, it in enumerate(self._buf):
                x = it.get(m)
                if x is not None:
                    if x.shape != block.shape[1:]:
                        raise ValueError(f"{m} window {x.shape} != {block.shape[1:]}")
                    block[i] = x
                    pres[i] = True
            d[a:b] = block
            p = self.f[f"present_{m}"]
            p.resize(b, axis=0)
            p[a:b] = pres
        for key, dtype in (("label", np.int64),):
            d = self.f[key]
            d.resize(b, axis=0)
            d[a:b] = np.asarray([it["label"] for it in self._buf], dtype)
        for key in ("subject", "segment"):
            d = self.f[key]
            d.resize(b, axis=0)
            d[a:b] = [str(it[key]).encode() for it in self._buf]
        self.n = b
        self._buf = []

    def close(self) -> None:
        self.flush()
        self.f.close()


def subject_split(subjects: Sequence[str], n_val: int, n_test: int, seed: int
                  ) -> Dict[str, str]:
    """Deterministic subject -> split. Counts are clipped to what there is."""
    subs = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    order = [subs[i] for i in rng.permutation(len(subs))]
    n_test = min(n_test, max(0, len(order) - 2))
    n_val = min(n_val, max(0, len(order) - n_test - 1))
    side = {}
    for i, s in enumerate(order):
        side[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return side


def split_summary(stats: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for s, v in stats.items():
        out[s] = {"windows": v["windows"], "subjects": len(v["subjects"]),
                  "per_class": v["per_class"].tolist(),
                  "present": {m: int(n) for m, n in v["present"].items()}}
    return out


# --------------------------------------------------------------------------- #
# EDF
# --------------------------------------------------------------------------- #

def read_edf(path: str, want: Optional[Sequence[str]] = None
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], float]:
    """A plain EDF/EDF+ reader: ``({label: [T] float32}, {label: Hz}, seconds)``.

    Standard library and numpy only, so the 11,000-file SeizeIT2 mirror needs
    no extra package on the cluster. Physical values; the EDF Annotations
    channel is skipped. ``want`` keeps only those labels (after stripping).
    """
    with open(path, "rb") as f:
        hdr = f.read(256)
        if len(hdr) < 256:
            raise ValueError(f"{path}: not an EDF file")
        n_rec = int(hdr[236:244].decode().strip() or -1)
        rec_s = float(hdr[244:252].decode().strip() or 0)
        ns = int(hdr[252:256].decode().strip())
        ch = f.read(ns * 256)
        labels = [ch[16 * i:16 * (i + 1)].decode("latin-1").strip() for i in range(ns)]
        o = 16 * ns + 80 * ns + 8 * ns                      # after transducer, unit
        pmin = [float(ch[o + 8 * i:o + 8 * (i + 1)]) for i in range(ns)]; o += 8 * ns
        pmax = [float(ch[o + 8 * i:o + 8 * (i + 1)]) for i in range(ns)]; o += 8 * ns
        dmin = [int(ch[o + 8 * i:o + 8 * (i + 1)]) for i in range(ns)]; o += 8 * ns
        dmax = [int(ch[o + 8 * i:o + 8 * (i + 1)]) for i in range(ns)]; o += 8 * ns
        o += 80 * ns                                         # prefiltering
        nsamp = [int(ch[o + 8 * i:o + 8 * (i + 1)]) for i in range(ns)]
        raw = np.fromfile(f, dtype="<i2")
    per_rec = sum(nsamp)
    if n_rec < 0:
        n_rec = raw.size // per_rec
    raw = raw[:n_rec * per_rec].reshape(n_rec, per_rec)
    signals, rates = {}, {}
    col = 0
    for i, lab in enumerate(labels):
        a, b = col, col + nsamp[i]
        col = b
        if lab.upper().startswith("EDF ANNOTATIONS"):
            continue
        if want is not None and lab not in want:
            continue
        gain = (pmax[i] - pmin[i]) / float(max(dmax[i] - dmin[i], 1))
        x = raw[:, a:b].astype(np.float32).reshape(-1)
        signals[lab] = (x - dmin[i]) * gain + pmin[i]
        rates[lab] = nsamp[i] / rec_s if rec_s else float("nan")
    return signals, rates, n_rec * rec_s


def edf_units(path: str) -> Dict[str, str]:
    """``{label: physical dimension}`` from an EDF header ('uV', 'mV', 'V', ...)."""
    with open(path, "rb") as f:
        hdr = f.read(256)
        ns = int(hdr[252:256].decode().strip())
        ch = f.read(ns * 256)
    labels = [ch[16 * i:16 * (i + 1)].decode("latin-1").strip() for i in range(ns)]
    o = 16 * ns + 80 * ns
    dims = [ch[o + 8 * i:o + 8 * (i + 1)].decode("latin-1").strip() for i in range(ns)]
    return dict(zip(labels, dims))


def to_microvolts(x: np.ndarray, dim: str) -> np.ndarray:
    d = (dim or "").strip().lower().replace("µ", "u")
    scale = {"uv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}.get(d)
    if scale is None:
        raise ValueError(f"unknown EEG unit {dim!r}")
    return x * scale


def write_edf(path: str, signals: Dict[str, np.ndarray], fs: float,
              rec_s: float = 1.0, start: str = "01.01.00") -> None:
    """A minimal EDF writer, for tests and for round-trip checks of read_edf."""
    labels = list(signals)
    ns = len(labels)
    nsamp = int(round(fs * rec_s))
    T = min(len(v) for v in signals.values())
    n_rec = T // nsamp
    pmin, pmax = -5000.0, 5000.0
    dmin, dmax = -32768, 32767
    hdr = ("0".ljust(8) + "X".ljust(80) + "X".ljust(80) + start.ljust(8)
           + "00.00.00" + str(256 * (ns + 1)).ljust(8) + "".ljust(44)
           + str(n_rec).ljust(8) + f"{rec_s:g}".ljust(8) + str(ns).ljust(4))
    f16 = lambda s: str(s)[:16].ljust(16)  # noqa: E731
    f8 = lambda s: str(s)[:8].ljust(8)     # noqa: E731
    f80 = lambda s: str(s)[:80].ljust(80)  # noqa: E731
    chdr = ("".join(f16(l) for l in labels) + "".join(f80("") for _ in labels)
            + "".join(f8("uV") for _ in labels)
            + "".join(f8(f"{pmin:g}") for _ in labels) + "".join(f8(f"{pmax:g}") for _ in labels)
            + "".join(f8(dmin) for _ in labels) + "".join(f8(dmax) for _ in labels)
            + "".join(f80("") for _ in labels) + "".join(f8(nsamp) for _ in labels)
            + "".join(f80("")[:32] for _ in labels))
    gain = (pmax - pmin) / (dmax - dmin)
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(chdr.encode("ascii"))
        for r in range(n_rec):
            for l in labels:
                seg = np.asarray(signals[l][r * nsamp:(r + 1) * nsamp], np.float64)
                d = np.clip(np.round((seg - pmin) / gain + dmin), dmin, dmax).astype("<i2")
                f.write(d.tobytes())


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
