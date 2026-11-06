"""Pretraining loss terms.

The full objective is

.. code-block:: text

    L = L_masked_raw
      + lambda_wave * L_wavelet
      + lambda_ref  * L_reference_consistency
      + lambda_spec * L_query_specialization
      + lambda_cov  * L_covariance

Every term can be switched off or reweighted from YAML, and each is logged
separately -- a single scalar tells you nothing about which term stalled.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..spatial.spatial_stats import DEFAULT_BANDS, band_filtered


def masked_patch_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over masked patches only.

    Args:
        pred/target: ``[B, S, P]``
        mask: ``[B, S]`` bool, True = masked (i.e. predicted).
    """
    assert pred.shape == target.shape, f"{tuple(pred.shape)} vs {tuple(target.shape)}"
    if mask.sum() == 0:
        return pred.new_zeros(())
    m = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target) ** 2 * m).sum() / (m.sum() * pred.shape[-1]).clamp_min(1.0)


def wavelet_coefficient_loss(
    pred_patches: torch.Tensor,
    target_coeffs: Sequence[torch.Tensor],
    transform,
    mask: torch.Tensor,
    B: int,
    C: int,
    S: int,
) -> torch.Tensor:
    """Reconstruction loss in the **critically sampled coefficient domain**.

    The predicted patches are re-analysed with the same transform and compared
    band by band against the target coefficients.  Because the transform is
    critically sampled, this adds no extra targets -- it reweights the same
    information towards a scale-aware error, which a pure time-domain MSE does not
    do (a time-domain MSE is dominated by whichever band carries most amplitude).
    """
    if mask.sum() == 0:
        return pred_patches.new_zeros(())
    P = pred_patches.shape[-1]
    pred_coeffs = transform.analysis(pred_patches.reshape(-1, P))
    losses = []
    m = mask.view(B, 1, S, 1).to(pred_patches.dtype)
    for pc, tc in zip(pred_coeffs, target_coeffs, strict=True):
        L = pc.shape[-1]
        p = pc.view(B, -1, S, L)
        t = tc.view(B, -1, S, L).detach()
        # Normalise per band so a low-amplitude band still contributes, using a
        # *symmetric* scale (target plus detached prediction).  Dividing by the
        # target RMS alone makes the term explode at initialisation, when the
        # prediction has unit scale and a fine detail band has RMS ~1e-2.
        scale = (t.pow(2).mean() + p.detach().pow(2).mean()).clamp_min(1e-8)
        losses.append((((p - t) ** 2 / scale) * m).sum() / (m.sum() * L * p.shape[1]).clamp_min(1.0))
    return torch.stack(losses).mean()


def reference_consistency_loss(
    view_embeddings: Sequence[torch.Tensor],
    anchor: Optional[torch.Tensor] = None,
    metric: str = "cosine",
) -> torch.Tensor:
    """Pull every reference view's global representation to a common point.

    ``anchor`` given (the default ``anchor='ssl'`` path)
        Each view is aligned to ``stopgrad(anchor)``, where the anchor is the
        representation of the surface-Laplacian view.  ``L_ssl`` annihilates the
        all-ones channel direction, so the SSL view is *the* reference-invariant
        view of the same data -- the physical statement being optimised is "every
        reference convention should collapse onto the reference-free one".

    ``anchor`` omitted (the ``pairwise`` fallback)
        Symmetric pairwise alignment between views.  Used when the montage is too
        sparse for a trustworthy spline Laplacian.
    """
    if len(view_embeddings) == 0:
        raise ValueError("reference_consistency_loss needs at least one view")

    def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if metric == "cosine":
            return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()
        return F.mse_loss(a, b)

    if anchor is not None:
        tgt = anchor.detach()
        return torch.stack([dist(z, tgt) for z in view_embeddings]).mean()
    if len(view_embeddings) < 2:
        return view_embeddings[0].new_zeros(())
    terms = [dist(view_embeddings[i], view_embeddings[j].detach())
             + dist(view_embeddings[j], view_embeddings[i].detach())
             for i in range(len(view_embeddings))
             for j in range(i + 1, len(view_embeddings))]
    return 0.5 * torch.stack(terms).mean()


def band_covariance_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> torch.Tensor:
    """Keep the reconstruction's **per-band** spatial covariance close to the target's.

    Band-wise, not broadband, for the same reason ``A_dyn`` is band-wise: a
    broadband covariance of scalp EEG is dominated by the highest-amplitude band,
    so matching it says almost nothing about the others.

    This is a reconstruction-fidelity term on a spatial statistic.  It is **not**
    a channel-relation graph and carries no interaction interpretation; see
    ``docs/terminology.md``.
    """
    assert recon.shape == target.shape, f"{tuple(recon.shape)} vs {tuple(target.shape)}"
    edges = list((bands or DEFAULT_BANDS).values())
    rb = band_filtered(recon, fs, edges)                        # [B, n, C, T]
    tb = band_filtered(target, fs, edges).detach()
    rb = rb - rb.mean(-1, keepdim=True)
    tb = tb - tb.mean(-1, keepdim=True)
    T = rb.shape[-1]
    cr = rb @ rb.transpose(-2, -1) / max(T - 1, 1)
    ct = tb @ tb.transpose(-2, -1) / max(T - 1, 1)
    scale = ct.pow(2).mean().clamp_min(1e-10).sqrt()
    return ((cr - ct) ** 2).mean() / scale


def corruption_targets(levels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Turn known corruption levels into reliability targets ``1 - level``."""
    return {m: (1.0 - lvl).clamp(0.0, 1.0) for m, lvl in levels.items()}
