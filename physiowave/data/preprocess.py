"""Configurable signal preprocessing with an on-disk cache.

Order matters and is fixed: notch -> band-pass -> resample -> normalise.  Notching
before resampling removes line noise while its harmonics are still resolvable;
normalising last means the statistics describe the signal the model actually sees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Preprocessing switches; all optional, all recorded in the cache key."""

    target_sampling_rate: Optional[float] = None
    notch_freq: Optional[float] = None          # 50.0 in Europe, 60.0 in the US
    notch_q: float = 30.0
    bandpass: Optional[Tuple[float, float]] = None
    normalize: str = "zscore"                   # 'zscore' | 'minmax' | 'maxabs' | 'none'
    clip_sigma: Optional[float] = None
    cache_dir: Optional[str] = None

    def key(self) -> str:
        payload = json.dumps(
            {k: v for k, v in self.__dict__.items() if k != "cache_dir"}, sort_keys=True
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _butter_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        logger.warning("bandpass (%.2f, %.2f) invalid at fs=%.1f; skipping", lo, hi, fs)
        return x
    sos = butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1).copy()


def _notch(x: np.ndarray, fs: float, freq: float, q: float) -> np.ndarray:
    from scipy.signal import filtfilt, iirnotch

    if freq >= fs / 2.0:
        logger.warning("notch %.1f Hz above Nyquist at fs=%.1f; skipping", freq, fs)
        return x
    b, a = iirnotch(freq / (fs / 2.0), q)
    return filtfilt(b, a, x, axis=-1).copy()


def _resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    if abs(fs - target) < 1e-6:
        return x
    up, down = int(round(target)), int(round(fs))
    g = gcd(up, down)
    return resample_poly(x, up // g, down // g, axis=-1)


def normalize_signal(x: np.ndarray, mode: str, clip_sigma: Optional[float] = None) -> np.ndarray:
    """Per-channel normalisation."""
    if mode == "none":
        return x
    if mode == "zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        out = (x - mu) / np.maximum(sd, 1e-8)
    elif mode == "minmax":
        lo = x.min(axis=-1, keepdims=True)
        hi = x.max(axis=-1, keepdims=True)
        out = 2.0 * (x - lo) / np.maximum(hi - lo, 1e-8) - 1.0
    elif mode == "maxabs":
        out = x / np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-8)
    else:
        raise ValueError(f"Unknown normalize mode {mode!r}")
    if clip_sigma:
        out = np.clip(out, -clip_sigma, clip_sigma)
    return out


def preprocess(x: np.ndarray, fs: float, cfg: PreprocessConfig) -> Tuple[np.ndarray, float]:
    """Apply the configured chain to ``[C, T]``; returns ``(x, new_fs)``."""
    assert x.ndim == 2, f"expected [C, T], got {x.shape}"
    x = np.asarray(x, dtype=np.float64)
    if cfg.notch_freq:
        x = _notch(x, fs, cfg.notch_freq, cfg.notch_q)
    if cfg.bandpass:
        x = _butter_bandpass(x, fs, cfg.bandpass[0], cfg.bandpass[1])
    if cfg.target_sampling_rate:
        x = _resample(x, fs, cfg.target_sampling_rate)
        fs = cfg.target_sampling_rate
    x = normalize_signal(x, cfg.normalize, cfg.clip_sigma)
    return x.astype(np.float32), fs


class PreprocessCache:
    """Content-addressed cache of preprocessed arrays."""

    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.dir = cfg.cache_dir
        self.hits = 0
        self.misses = 0
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def path(self, item_key: str) -> Optional[str]:
        if not self.dir:
            return None
        h = hashlib.sha1(f"{item_key}::{self.cfg.key()}".encode()).hexdigest()[:20]
        return os.path.join(self.dir, f"{h}.npy")

    def get(self, item_key: str) -> Optional[np.ndarray]:
        p = self.path(item_key)
        if p and os.path.exists(p):
            self.hits += 1
            return np.load(p)
        self.misses += 1
        return None

    def put(self, item_key: str, arr: np.ndarray) -> None:
        p = self.path(item_key)
        if not p:
            return
        # Write through a handle: np.save would append a second ".npy" to a
        # temporary name that does not already end in it.
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, p)
