"""Synthetic corpora for smoke tests, unit tests and dry runs.

No real recording is ever fabricated as a *result*: these generators exist so the
whole pipeline (forward, backward, checkpoint, resume, evaluation, benchmarks)
can be exercised on a laptop without any protected dataset.  Anything computed on
synthetic data is labelled as such in every report.

The EEG generator is deliberately more than white noise: it mixes a small number
of dipole-like sources through a distance-based lead field so that the resulting
channel statistics show the volume-conduction structure the spatial modules are
designed around (smooth spatial fall-off, strong zero-lag correlation between
neighbours).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..spatial.spatial_stats import DEFAULT_BANDS
from .montages import montage
from .schema import Sample


def _band_noise(rng: np.random.Generator, T: int, fs: float, lo: float, hi: float) -> np.ndarray:
    """Band-limited noise of unit variance."""
    X = rng.normal(size=T)
    F = np.fft.rfft(X)
    f = np.fft.rfftfreq(T, 1.0 / fs)
    F[(f < lo) | (f >= hi)] = 0.0
    y = np.fft.irfft(F, n=T)
    return y / max(y.std(), 1e-8)


def synth_eeg(
    n_channels: int, T: int, fs: float, xyz: torch.Tensor, seed: int = 0, n_sources: int = 6
) -> np.ndarray:
    """Volume-conduction-like EEG: sources mixed through a distance lead field."""
    rng = np.random.default_rng(seed)
    # float64 throughout: some BLAS builds raise spurious FP flags on float32 matmul
    pos = xyz.double().numpy()
    src = rng.normal(size=(n_sources, 3))
    src /= np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1e-8)
    src *= 0.7                                          # sources sit below the surface
    d = np.linalg.norm(pos[:, None, :] - src[None, :, :], axis=-1)
    lead = np.exp(-(d ** 2) / 0.35)                     # smooth spatial spread
    bands = list(DEFAULT_BANDS.values())
    S = np.stack([
        _band_noise(rng, T, fs, *bands[i % len(bands)]) * (1.0 / (1 + i))
        for i in range(n_sources)
    ])
    # Some BLAS builds (notably Apple Accelerate) raise spurious FP status flags
    # on this matmul even though every operand and the result are finite, so the
    # flags are suppressed and the result is asserted finite instead.
    with np.errstate(all="ignore"):
        x = np.asarray(lead, dtype=np.float64) @ np.asarray(S, dtype=np.float64)
    assert np.isfinite(x).all(), "synthetic EEG generator produced non-finite values"
    x += 0.1 * rng.normal(size=(n_channels, T))         # sensor noise
    x += 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * np.arange(T) / fs)[None, :]  # drift
    return x.astype(np.float32)


def synth_ecg(n_channels: int, T: int, fs: float, seed: int = 0, hr: float = 70.0) -> np.ndarray:
    """Simple 12-lead-like ECG: a QRS-shaped template repeated at a heart rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fs
    period = 60.0 / hr
    phase = (t % period) / period
    qrs = np.exp(-((phase - 0.2) ** 2) / 2e-4) - 0.25 * np.exp(-((phase - 0.16) ** 2) / 1e-4)
    tw = 0.3 * np.exp(-((phase - 0.45) ** 2) / 3e-3)
    pw = 0.15 * np.exp(-((phase - 0.08) ** 2) / 1e-3)
    beat = qrs + tw + pw
    gains = rng.uniform(-1.0, 1.5, size=(n_channels, 1))
    x = gains * beat[None, :] + 0.05 * rng.normal(size=(n_channels, T))
    return x.astype(np.float32)


def synth_semg(n_channels: int, T: int, fs: float, seed: int = 0, n_bursts: int = 4) -> np.ndarray:
    """Limb sEMG: band-limited noise gated by smooth activation bursts."""
    rng = np.random.default_rng(seed)
    base = np.stack([_band_noise(rng, T, fs, 20.0, min(450.0, fs / 2 - 1)) for _ in range(n_channels)])
    env = np.zeros(T)
    for _ in range(n_bursts):
        c = rng.integers(0, T)
        w = rng.integers(T // 20, T // 6)
        env += np.exp(-((np.arange(T) - c) ** 2) / (2 * w ** 2))
    env = env / max(env.max(), 1e-8)
    gains = rng.uniform(0.4, 1.0, size=(n_channels, 1))
    return (base * (0.1 + gains * env[None, :])).astype(np.float32)


@dataclass
class SyntheticConfig:
    """Shape and size of a synthetic corpus."""

    modality: str = "eeg"
    num_samples: int = 64
    num_subjects: int = 8
    window_samples: int = 512
    sampling_rate: float = 256.0
    num_channels: Optional[int] = None
    montage_name: str = "standard_1020_19"
    num_classes: Optional[int] = 4
    seed: int = 0
    missing_channel_prob: float = 0.0


class SyntheticDataset(Dataset):
    """Deterministic synthetic dataset yielding :class:`Sample` objects."""

    def __init__(self, cfg: SyntheticConfig) -> None:
        self.cfg = cfg
        if cfg.modality == "eeg":
            self.channel_names, self.xyz = montage(cfg.montage_name)
            if cfg.num_channels:
                self.channel_names = self.channel_names[: cfg.num_channels]
                self.xyz = self.xyz[: cfg.num_channels]
            self.reference_type, self.derivation_type = "original", "monopolar"
            self.montage_type = "standard_1020" if "1020" in cfg.montage_name else "standard_1010"
            self.emg_region = "unknown"
        else:
            C = cfg.num_channels or (12 if cfg.modality == "ecg" else 8)
            prefix = "LEAD" if cfg.modality == "ecg" else "EMG"
            self.channel_names = [f"{prefix}{i + 1}" for i in range(C)]
            self.xyz = torch.zeros(C, 3)
            self.reference_type = "unknown"
            self.derivation_type = "ecg_12lead" if cfg.modality == "ecg" else "monopolar"
            self.montage_type = "custom"
            self.emg_region = "limb" if cfg.modality == "semg" else "unknown"

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> Sample:
        cfg = self.cfg
        C = len(self.channel_names)
        seed = cfg.seed * 100003 + idx
        if cfg.modality == "eeg":
            x = synth_eeg(C, cfg.window_samples, cfg.sampling_rate, self.xyz, seed)
        elif cfg.modality == "ecg":
            x = synth_ecg(C, cfg.window_samples, cfg.sampling_rate, seed)
        else:
            x = synth_semg(C, cfg.window_samples, cfg.sampling_rate, seed)

        mask = torch.ones(C, dtype=torch.bool)
        if cfg.missing_channel_prob > 0:
            g = torch.Generator().manual_seed(seed)
            drop = torch.rand(C, generator=g) < cfg.missing_channel_prob
            drop[0] = False                              # never drop everything
            mask = ~drop
            x = x * mask.numpy()[:, None]

        subject = f"S{idx % cfg.num_subjects:03d}"
        label = None
        if cfg.num_classes:
            label = torch.tensor((idx // max(cfg.num_subjects, 1)) % cfg.num_classes)
        dur = cfg.window_samples / cfg.sampling_rate
        return Sample(
            signal=torch.from_numpy(np.ascontiguousarray(x)),
            modality=cfg.modality,
            sampling_rate=cfg.sampling_rate,
            subject_id=subject,
            recording_id=f"{subject}_R{idx % 3}",
            dataset_id=f"synthetic_{cfg.modality}",
            channel_names=list(self.channel_names),
            channel_xyz=self.xyz.clone(),
            channel_mask=mask,
            montage_type=self.montage_type,
            reference_type=self.reference_type,
            derivation_type=self.derivation_type,
            label=label,
            window_start=idx * dur,
            window_end=(idx + 1) * dur,
            emg_region=self.emg_region,
        ).validate()


def synthetic_multimodal(
    num_samples: int = 32, window_samples: int = 512, seed: int = 0
) -> List[dict]:
    """Aligned EEG/ECG/sEMG windows for RALF smoke tests."""
    eeg = SyntheticDataset(SyntheticConfig("eeg", num_samples, window_samples=window_samples,
                                           sampling_rate=256.0, seed=seed))
    ecg = SyntheticDataset(SyntheticConfig("ecg", num_samples, window_samples=window_samples,
                                           sampling_rate=256.0, seed=seed + 1))
    emg = SyntheticDataset(SyntheticConfig("semg", num_samples, window_samples=window_samples,
                                           sampling_rate=256.0, seed=seed + 2))
    return [{"eeg": eeg[i], "ecg": ecg[i], "semg": emg[i]} for i in range(num_samples)]
