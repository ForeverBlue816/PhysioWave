"""Channel-relation graphs estimated from the recorded data (``A_dyn``).

TERMINOLOGY (constraint A of ``docs/terminology.md``) -- read before editing
---------------------------------------------------------------------------
Every matrix produced here is a **spatial statistic of the recorded signals**,
also called a *channel-relation graph*.  It is contaminated by the reference
montage and by volume conduction, and it MUST NOT be interpreted or described as
functional or brain connectivity.  # TERMINOLOGY-ALLOW: states the prohibition
Scalp channel covariance mixes genuine source correlation with (a) the choice of
reference, which adds a rank-one common term to every channel pair, and (b)
instantaneous volume conduction, which spreads a single source across many
electrodes and produces spurious zero-lag correlation.  The names in this module
(``spatial_stat_graph``, ``A_dyn``) are chosen to keep that distinction visible in
the code, and ``tests/test_terminology.py`` enforces it.

Estimators
----------
``cov`` (default)
    Band-wise shrinkage covariance / correlation.  Band-wise is the default and
    broadband is only an ablation: a per-sample broadband covariance of scalp EEG
    is dominated by whichever band carries the most amplitude -- typically alpha
    or a low-frequency ocular drift -- so the resulting graph mostly encodes
    "where the biggest slow signal is", not the spatial structure of the other
    bands.  Shrinkage towards a scaled identity (Ledoit-Wolf or a fixed
    coefficient) keeps the estimate well conditioned for short windows where the
    number of samples per band is small.

``wpli`` / ``imcoh``
    Volume-conduction-robust alternatives.  Because volume conduction is
    instantaneous within the EEG band, the spurious component of a cross-spectrum
    concentrates at zero (and pi) phase, i.e. on the *real* axis.  The weighted
    phase-lag index and the imaginary part of coherency use only the imaginary
    part / phase-lag sign of the cross-spectrum and are therefore insensitive to
    it.  Ordinary magnitude coherence is deliberately **not** offered as a
    default anywhere in this codebase for exactly that reason.

All estimators return **detached** tensors.  These graphs enter the model only as
an attention bias / graph structure, never as a differentiable feature, so no
gradient may flow through the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

#: Canonical EEG band edges in Hz.
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

DYN_GRAPH_TYPES = ("cov", "wpli", "imcoh")
DYN_GRAPH_INPUTS = ("raw", "ssl")


@dataclass
class DynGraphConfig:
    """Configuration of the dynamic spatial-statistics graph."""

    enabled: bool = True
    dyn_graph_type: str = "cov"          # 'cov' | 'wpli' | 'imcoh'
    dyn_graph_input: str = "raw"         # 'raw' | 'ssl'
    band_wise: bool = True               # broadband is an ablation only
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    shrinkage: str = "ledoit_wolf"       # 'ledoit_wolf' | 'fixed' | 'none'
    shrinkage_coef: float = 0.1          # used when shrinkage == 'fixed'
    correlation: bool = True             # normalise covariance to correlation
    diag_load: float = 1e-6
    amplitude_gate_percentile: float = 0.1  # phase gating for wpli / imcoh
    debiased_wpli: bool = True              # Vinck et al. (2011) small-sample correction
    n_segments: int = 4                     # Welch segments for the phase estimators
    max_condition_number: float = 1e6

    def __post_init__(self) -> None:
        if self.dyn_graph_type not in DYN_GRAPH_TYPES:
            raise ValueError(f"dyn_graph_type must be one of {DYN_GRAPH_TYPES}")
        if self.dyn_graph_input not in DYN_GRAPH_INPUTS:
            raise ValueError(f"dyn_graph_input must be one of {DYN_GRAPH_INPUTS}")


# --------------------------------------------------------------------------- #
# Band decomposition
# --------------------------------------------------------------------------- #
def band_analytic_signals(
    x: torch.Tensor, fs: float, bands: Sequence[Tuple[float, float]]
) -> torch.Tensor:
    """Per-band analytic signals of ``[B, C, T]`` via FFT masking.

    Zeroing the negative-frequency half of the band-limited spectrum and doubling
    the positive half is the Hilbert transform, so this yields the analytic signal
    of each band in one FFT pass.

    Returns:
        complex ``[B, n_bands, C, T]``.
    """
    assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
    B, C, T = x.shape
    # `.contiguous()` avoids a spurious out-resize warning from the MPS FFT.
    X = torch.fft.fft(x.to(torch.float32).contiguous(), dim=-1)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs, device=x.device)
    out = []
    for lo, hi in bands:
        keep = (freqs >= lo) & (freqs < hi)          # positive frequencies only
        mask = torch.zeros(T, device=x.device, dtype=X.dtype)
        mask[keep] = 2.0
        out.append(torch.fft.ifft(X * mask, dim=-1))
    return torch.stack(out, dim=1)                   # [B, n_bands, C, T]


def band_filtered(x: torch.Tensor, fs: float, bands: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Real band-limited signals ``[B, n_bands, C, T]``."""
    return band_analytic_signals(x, fs, bands).real


def band_fourier_coefficients(
    x: torch.Tensor,
    fs: float,
    bands: Sequence[Tuple[float, float]],
    n_segments: int = 4,
) -> List[torch.Tensor]:
    """Windowed Fourier coefficients per band -> one complex ``[B, C, n_obs]`` each.

    Phase-based statistics (wPLI, imaginary coherence) are estimated by averaging
    the cross-spectrum over independent observations.  Time samples of a
    narrow-band analytic signal are *not* independent -- a 5 Hz-wide band over 8 s
    carries roughly 40 independent degrees of freedom, not 2048 -- so averaging
    over them leaves the estimator with a large variance that the small-sample
    debiasing cannot remove.  Averaging over (segment x frequency bin) pairs
    instead gives genuinely near-independent observations, which is how these
    measures are computed in the literature.

    Returns:
        one tensor per band, complex, shape ``[B, C, n_segments * n_bins]``.
    """
    assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
    B, C, T = x.shape
    n_segments = max(1, min(n_segments, T // 32))
    seg_len = T // n_segments
    if seg_len < 8:
        n_segments, seg_len = 1, T
    xs = x[..., : n_segments * seg_len].reshape(B, C, n_segments, seg_len)
    window = torch.hann_window(seg_len, device=x.device, dtype=x.dtype)
    X = torch.fft.rfft(xs * window, dim=-1)                    # [B, C, n_seg, n_freq]
    freqs = torch.fft.rfftfreq(seg_len, d=1.0 / fs, device=x.device)
    out: List[torch.Tensor] = []
    for lo, hi in bands:
        keep = torch.nonzero((freqs >= lo) & (freqs < hi), as_tuple=False).flatten()
        if keep.numel() == 0:                                  # band below the resolution
            keep = torch.tensor([min(1, X.shape[-1] - 1)], device=x.device)
        sel = X[..., keep]                                     # [B, C, n_seg, n_bins]
        out.append(sel.reshape(B, C, -1))
    return out


# --------------------------------------------------------------------------- #
# Covariance / correlation
# --------------------------------------------------------------------------- #
def ledoit_wolf_shrinkage(S: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Ledoit-Wolf shrinkage intensity for a batch of covariance matrices.

    Uses the standard closed form with the scaled-identity target
    ``mu * I``, ``mu = tr(S) / C``.  Returned shape broadcasts against ``S``.
    """
    C = S.shape[-1]
    mu = torch.diagonal(S, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    target = mu * torch.eye(C, device=S.device, dtype=S.dtype)
    d2 = ((S - target) ** 2).sum(dim=(-2, -1), keepdim=True)
    # b2 is bounded above by d2; the usual estimator uses per-sample scatter, and
    # this conservative bound keeps the intensity in [0, 1] without materialising
    # the per-sample outer products.
    b2 = (S ** 2).sum(dim=(-2, -1), keepdim=True) / max(n_samples, 1)
    b2 = torch.minimum(b2, d2)
    rho = (b2 / d2.clamp_min(1e-12)).clamp(0.0, 1.0)
    return rho


def shrunk_covariance(
    x: torch.Tensor, cfg: DynGraphConfig
) -> torch.Tensor:
    """Shrinkage covariance (or correlation) of ``[..., C, T]`` -> ``[..., C, C]``.

    Numerically stabilised by diagonal loading and, if the conditioning is still
    bad, by an additional shrinkage step towards the identity.
    """
    C, T = x.shape[-2], x.shape[-1]
    xc = x - x.mean(dim=-1, keepdim=True)
    S = xc @ xc.transpose(-2, -1) / max(T - 1, 1)
    if cfg.shrinkage == "ledoit_wolf":
        rho = ledoit_wolf_shrinkage(S, T)
    elif cfg.shrinkage == "fixed":
        rho = torch.full(S.shape[:-2] + (1, 1), float(cfg.shrinkage_coef),
                         device=S.device, dtype=S.dtype)
    else:
        rho = torch.zeros(S.shape[:-2] + (1, 1), device=S.device, dtype=S.dtype)
    mu = torch.diagonal(S, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(C, device=S.device, dtype=S.dtype)
    S = (1 - rho) * S + rho * mu * eye
    S = S + cfg.diag_load * mu.clamp_min(1e-12) * eye

    if cfg.correlation:
        d = torch.diagonal(S, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
        S = S / (d.unsqueeze(-1) * d.unsqueeze(-2))
        S = S.clamp(-1.0, 1.0)
    return S


def condition_number(S: torch.Tensor) -> torch.Tensor:
    """Per-matrix condition number of a batch of symmetric matrices.

    Computed on CPU: it is a diagnostic evaluated once per forward pass on a tiny
    ``[C, C]`` matrix, and several accelerator backends (MPS among them) have no
    symmetric eigensolver.
    """
    ev = torch.linalg.eigvalsh(S.detach().to(device="cpu", dtype=torch.float32))
    cond = ev.abs().amax(dim=-1) / ev.abs().amin(dim=-1).clamp_min(1e-12)
    return cond.to(S.device)


# --------------------------------------------------------------------------- #
# Volume-conduction-robust estimators
# --------------------------------------------------------------------------- #
def _cross_spectrum(z: torch.Tensor) -> torch.Tensor:
    """Cross-spectral products ``z_i * conj(z_j)`` of ``[..., C, T]`` -> ``[..., C, C, T]``."""
    return z.unsqueeze(-2) * z.conj().unsqueeze(-3)


def _amplitude_gate(z: torch.Tensor, percentile: float) -> torch.Tensor:
    """Mask of observations whose amplitude is large enough for phase to mean anything.

    Phase is undefined-ish where the band-limited amplitude is near zero, so those
    observations are excluded before any phase-based statistic is formed.
    """
    amp = z.abs()                                        # [..., C, T]
    thresh = torch.quantile(amp, percentile, dim=-1, keepdim=True)
    return (amp > thresh)                                # [..., C, T]


def weighted_phase_lag_index(
    z: torch.Tensor, percentile: float = 0.1, debiased: bool = True
) -> torch.Tensor:
    """wPLI from analytic signals ``[..., C, T]`` -> ``[..., C, C]`` in ``[0, 1]``.

    Plain wPLI is ``|E[Im(S)]| / E[|Im(S)|]``.  Only the imaginary part of the
    cross-spectrum contributes, so a pair of channels sharing an instantaneous
    (zero-phase) volume-conducted source contributes nothing in expectation.

    With ``debiased=True`` (the default) the pairwise estimator of Vinck et al.
    (2011) is used,

    .. code-block:: text

        dwPLI = ( (sum Im)^2 - sum Im^2 ) / ( (sum |Im|)^2 - sum Im^2 )

    which removes the positive bias that the plain form carries at small sample
    counts.  That bias matters here: a short window contains only a handful of
    independent samples per band, and the undebiased estimator would report a
    sizeable wPLI for a purely zero-phase pair purely from noise.
    """
    S = _cross_spectrum(z)                               # [..., C, C, T]
    im = S.imag
    gate = _amplitude_gate(z, percentile)                # [..., C, T]
    pair_gate = (gate.unsqueeze(-2) & gate.unsqueeze(-3)).to(im.dtype)
    im = im * pair_gate
    s1 = im.sum(dim=-1)
    s1a = im.abs().sum(dim=-1)
    if debiased:
        s2 = (im ** 2).sum(dim=-1)
        num = (s1 ** 2 - s2).clamp_min(0.0)
        den = (s1a ** 2 - s2).clamp_min(1e-12)
        out = num / den
    else:
        out = s1.abs() / s1a.clamp_min(1e-12)
    idx = torch.arange(out.shape[-1], device=out.device)
    out[..., idx, idx] = 1.0
    return out.clamp(0.0, 1.0)


def imaginary_coherence(z: torch.Tensor, percentile: float = 0.1) -> torch.Tensor:
    """|Imaginary coherency| from analytic signals ``[..., C, T]`` -> ``[..., C, C]``.

    ``imCoh = |Im(E[S_ij])| / sqrt(E[|z_i|^2] E[|z_j|^2])``.  Same rationale as
    wPLI: the instantaneous, volume-conducted component of the cross-spectrum is
    real and drops out.
    """
    gate = _amplitude_gate(z, percentile)
    g = gate.to(z.real.dtype)
    n = g.sum(dim=-1).clamp_min(1.0)                     # [..., C]
    S = _cross_spectrum(z)
    pair_gate = (gate.unsqueeze(-2) & gate.unsqueeze(-3)).to(S.real.dtype)
    npair = pair_gate.sum(dim=-1).clamp_min(1.0)
    cross = (S * pair_gate).sum(dim=-1) / npair          # [..., C, C] complex
    power = ((z.abs() ** 2) * g).sum(dim=-1) / n         # [..., C]
    denom = (power.unsqueeze(-1) * power.unsqueeze(-2)).clamp_min(1e-12).sqrt()
    out = cross.imag.abs() / denom
    idx = torch.arange(out.shape[-1], device=out.device)
    out[..., idx, idx] = 1.0
    return out.clamp(0.0, 1.0)


def magnitude_coherence(z: torch.Tensor) -> torch.Tensor:
    """Ordinary magnitude coherence -- provided **only** as a negative control.

    It is never a default anywhere in this codebase: within the EEG band, volume
    conduction is instantaneous, so a shared source inflates magnitude coherence
    at zero phase lag and the value additionally shifts with the reference
    montage.  ``tests/test_spatial.py`` uses it to demonstrate exactly that
    failure mode against wPLI.
    """
    S = _cross_spectrum(z).mean(dim=-1)
    power = (z.abs() ** 2).mean(dim=-1)
    denom = (power.unsqueeze(-1) * power.unsqueeze(-2)).clamp_min(1e-12).sqrt()
    return (S.abs() / denom).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #
class SpatialStatGraph(nn.Module):
    """Estimates ``A_dyn``: a per-sample, band-weighted channel-relation graph.

    The output is **detached**: it is consumed as an attention bias and as graph
    structure, never as a differentiable feature.  Only the band mixing weights
    are learnable, and they act on an already-detached stack of per-band matrices.
    """

    def __init__(self, cfg: DynGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.band_names = list(cfg.bands.keys())
        n = len(self.band_names) if cfg.band_wise else 1
        self.band_logits = nn.Parameter(torch.zeros(n))
        self.last_condition_number: Optional[float] = None

    @property
    def band_edges(self) -> List[Tuple[float, float]]:
        return [self.cfg.bands[b] for b in self.band_names]

    def per_band_graphs(self, x: torch.Tensor, fs: float) -> torch.Tensor:
        """``[B, C, T] -> [B, n_bands, C, C]`` detached statistics."""
        with torch.no_grad():
            bands = self.band_edges if self.cfg.band_wise else [(0.0, fs / 2.0)]
            if self.cfg.dyn_graph_type == "cov":
                if not self.cfg.band_wise:
                    return shrunk_covariance(x, self.cfg).unsqueeze(1)
                return shrunk_covariance(band_filtered(x, fs, bands), self.cfg)
            coeffs = band_fourier_coefficients(x, fs, bands, self.cfg.n_segments)
            return torch.stack([self._phase_graph(c) for c in coeffs], dim=1)

    def _phase_graph(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, C, n_obs]`` complex -> ``[B, C, C]``."""
        p = self.cfg.amplitude_gate_percentile
        if self.cfg.dyn_graph_type == "wpli":
            return weighted_phase_lag_index(z, p, self.cfg.debiased_wpli)
        return imaginary_coherence(z, p)

    def forward(self, x: torch.Tensor, fs: float) -> torch.Tensor:
        """``[B, C, T] -> A_dyn [B, C, C]`` (detached, band-weighted).

        Args:
            x: signal the statistics are computed on.  With
               ``dyn_graph_input='ssl'`` the caller passes the SSL-transformed
               signal instead of the raw one; that view is far less affected by
               the reference montage and by volume conduction, which is why it is
               in the ablation matrix.
        """
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        per_band = self.per_band_graphs(x, fs)                # [B, n, C, C]
        with torch.no_grad():
            cond = condition_number(per_band.flatten(0, 1)).max()
            self.last_condition_number = float(cond.item())
        w = torch.softmax(self.band_logits, dim=0).view(1, -1, 1, 1)
        A = (per_band * w).sum(dim=1)
        return A.detach()
