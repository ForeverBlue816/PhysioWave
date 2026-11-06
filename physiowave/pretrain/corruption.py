"""Controlled signal corruption with a known severity, used to supervise reliability.

RALF predicts a scalar reliability per modality.  The only place a *ground truth*
for that scalar exists is synthetic corruption applied by this module: the caller
knows exactly how badly it degraded each stream, so the target is
``1 - corruption_level`` rather than a heuristic.

Corruptions cover the failure modes that actually occur on wearable and clinical
recordings: additive noise, low-frequency drift, amplifier saturation, dead or
railing electrodes, and packet loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

CORRUPTIONS = ("noise", "drift", "saturation", "bad_channel", "dropout")


@dataclass
class CorruptionConfig:
    """Configuration for :class:`SignalCorruptor`."""

    enabled: bool = True
    prob: float = 0.5                    # probability a given stream is corrupted
    kinds: Sequence[str] = field(default_factory=lambda: list(CORRUPTIONS))
    max_level: float = 1.0
    noise_snr_db_range: Tuple[float, float] = (20.0, -5.0)
    drift_amp_range: Tuple[float, float] = (0.1, 3.0)
    saturation_quantile_range: Tuple[float, float] = (0.999, 0.5)
    bad_channel_frac_range: Tuple[float, float] = (0.0, 0.5)
    dropout_frac_range: Tuple[float, float] = (0.0, 0.5)


class SignalCorruptor:
    """Applies one corruption per item and reports the severity that was used.

    The generator lives on the CPU whatever device the signal is on: every
    random draw is made there and moved, so one seeded generator serves a
    CUDA batch (a CUDA tensor cannot be drawn from a CPU generator, and the
    first cluster run of the fusion died on exactly that).
    """

    def __init__(self, cfg: CorruptionConfig, generator: Optional[torch.Generator] = None) -> None:
        self.cfg = cfg
        self.g = generator

    def _u(self, n: int = 1, device=None) -> torch.Tensor:
        return torch.rand(n, generator=self.g, device=device)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Corrupt ``[B, C, T]``.

        Returns:
            ``(x_corrupt, level [B] in [0, 1], kinds [B])``.  ``level`` is the
            reliability-supervision target complement: 0 = untouched.
        """
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        out = x.clone()
        level = torch.zeros(B, device=x.device)
        kinds: List[str] = ["none"] * B
        if not self.cfg.enabled:
            return out, level, kinds

        for b in range(B):
            if float(self._u(1).item()) > self.cfg.prob:
                continue
            kind = self.cfg.kinds[int(torch.randint(len(self.cfg.kinds), (1,),
                                                    generator=self.g).item())]
            sev = float(self._u(1).item()) * self.cfg.max_level
            out[b] = self._apply(out[b], kind, sev)
            level[b] = sev
            kinds[b] = kind
        return out, level, kinds

    def _apply(self, x: torch.Tensor, kind: str, sev: float) -> torch.Tensor:
        """Apply one corruption of severity ``sev in [0, 1]`` to ``[C, T]``."""
        C, T = x.shape
        if kind == "noise":
            lo, hi = self.cfg.noise_snr_db_range
            snr_db = lo + (hi - lo) * sev
            power = x.pow(2).mean().clamp_min(1e-12)
            noise_power = power / (10 ** (snr_db / 10))
            return x + torch.randn(x.shape, generator=self.g).to(x.device) * noise_power.sqrt()
        if kind == "drift":
            lo, hi = self.cfg.drift_amp_range
            amp = (lo + (hi - lo) * sev) * x.std().clamp_min(1e-6)
            t = torch.linspace(0, 1, T, device=x.device)
            phase = (torch.rand(C, 1, generator=self.g) * 6.283).to(x.device)
            freq = (0.5 + 2.0 * torch.rand(C, 1, generator=self.g)).to(x.device)
            return x + amp * torch.sin(6.283 * freq * t.view(1, T) + phase)
        if kind == "saturation":
            lo, hi = self.cfg.saturation_quantile_range
            q = lo + (hi - lo) * sev
            thr = torch.quantile(x.abs().flatten(), max(min(q, 0.9999), 0.01))
            return x.clamp(-thr, thr)
        if kind == "bad_channel":
            lo, hi = self.cfg.bad_channel_frac_range
            n_bad = int(round(C * (lo + (hi - lo) * sev)))
            if n_bad <= 0:
                return x
            idx = torch.randperm(C, generator=self.g)[:n_bad].to(x.device)
            y = x.clone()
            # A dead electrode rails or goes flat with high-frequency noise.
            y[idx] = torch.randn(len(idx), T, generator=self.g).to(x.device) * 1e-3
            return y
        if kind == "dropout":
            lo, hi = self.cfg.dropout_frac_range
            frac = lo + (hi - lo) * sev
            n = max(1, int(round(T * frac)))
            start = int(torch.randint(max(T - n, 1), (1,), generator=self.g).item())
            y = x.clone()
            y[:, start:start + n] = 0.0
            return y
        raise ValueError(f"Unknown corruption kind {kind!r}")
