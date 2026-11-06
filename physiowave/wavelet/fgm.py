"""Frequency-guided masking (FgM) driven by critically-sampled subband energies.

The original PhysioWave FgM scored patches with an FFT of the *token* tensor.
Here the score comes from the wavelet subband energies that WAST already
computes, which is both cheaper and better matched to the decomposition.

Two properties are load-bearing and are enforced by the tests:

* the scores are computed from **detached** energies, so the masking decision
  never back-propagates into the wavelet filters (a masking policy that can be
  optimised to make reconstruction easy is a degenerate objective);
* the score is a per-patch scalar over the ``S`` temporal patches, so adding
  decomposition levels changes the *quality* of the score but never the number
  of maskable units -- multi-scale analysis must not inflate the token count.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def band_energies(coeffs: Sequence[torch.Tensor], B: int, C: int, S: int) -> torch.Tensor:
    """Per-patch, per-band mean energy.

    Args:
        coeffs: subbands ``[cA_J, cD_J, ..., cD_1]``, each ``[B*C*S, P_k]``.
        B, C, S: batch, channel and patch counts used to unflatten.

    Returns:
        ``[B, C, S, J+1]`` detached energies.
    """
    out: List[torch.Tensor] = []
    for c in coeffs:
        e = c.detach().pow(2).mean(dim=-1)  # [B*C*S]
        out.append(e.view(B, C, S))
    return torch.stack(out, dim=-1)


def frequency_guided_patch_scores(
    energies: torch.Tensor,
    band_weights: torch.Tensor | None = None,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine ``[B, C, S, J+1]`` band energies into ``[B, S]`` patch scores.

    Energies are log-compressed (physiological band power spans several decades)
    and standardised per band before being pooled over channels, so a single
    high-amplitude band cannot dominate the ranking.
    """
    e = torch.log1p(energies.clamp_min(0.0)).detach()
    mean = e.mean(dim=(1, 2), keepdim=True)
    std = e.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    e = (e - mean) / std
    if band_weights is not None:
        e = e * band_weights.detach().view(1, 1, 1, -1)
    per_channel = e.mean(dim=-1)  # [B, C, S]
    if channel_mask is not None:
        m = channel_mask.to(per_channel.dtype)
        if m.dim() == 1:                      # [C] -> [B, C]
            m = m.unsqueeze(0).expand(per_channel.shape[0], -1)
        assert m.shape == per_channel.shape[:2], (
            f"channel_mask must be [C] or [B, C]; got {tuple(channel_mask.shape)} "
            f"for tokens with shape {tuple(per_channel.shape)}"
        )
        m = m.unsqueeze(-1)                   # [B, C, 1]
        per_channel = per_channel * m
        denom = m.sum(dim=1).clamp_min(1.0)   # [B, 1]
        return per_channel.sum(dim=1) / denom
    return per_channel.mean(dim=1)


def frequency_guided_mask(
    scores: torch.Tensor,
    mask_ratio: float,
    importance_ratio: float = 0.6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask ``[B, S]`` selecting the highest-scoring patches.

    ``importance_ratio`` blends the (detached) importance score with uniform
    noise, exactly as in the original PhysioWave FgM: 1.0 is purely
    energy-driven, 0.0 is uniform random masking.
    """
    B, S = scores.shape
    num_mask = max(1, int(round(S * mask_ratio)))
    s = scores - scores.mean(dim=1, keepdim=True)
    s = s / s.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = torch.rand(B, S, device=scores.device, dtype=scores.dtype, generator=generator)
    combined = importance_ratio * s + (1.0 - importance_ratio) * noise
    idx = torch.topk(combined, num_mask, dim=1).indices
    mask = torch.zeros(B, S, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, idx, True)
    return mask


def random_patch_mask(
    B: int, S: int, mask_ratio: float, device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Uniform random boolean mask ``[B, S]``, sampled independently per item."""
    num_mask = max(1, int(round(S * mask_ratio)))
    noise = torch.rand(B, S, device=device, generator=generator)
    idx = torch.topk(noise, num_mask, dim=1).indices
    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask
