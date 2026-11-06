"""Electrode geometry helpers: sphere projection, distances and the static graph.

Terminology (see ``docs/terminology.md``)
-----------------------------------------
Everything in this file is derived from *electrode positions only*.  It is
geometry, never a statement about neural interaction.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np
import torch


def normalize_to_sphere(xyz: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project ``[C, 3]`` electrode positions onto the unit sphere.

    The spherical spline Laplacian is defined on a sphere, so positions coming
    from a digitiser or a template montage must be radially projected first.
    Rows that are all-zero (used as the "unknown coordinate" sentinel) are left
    at the origin and must be excluded by the caller via ``channel_mask``.
    """
    assert xyz.dim() == 2 and xyz.shape[-1] == 3, f"expected [C, 3], got {tuple(xyz.shape)}"
    centred = xyz - xyz.mean(dim=0, keepdim=True)
    r = centred.norm(dim=-1, keepdim=True)
    known = (r > eps).squeeze(-1)
    out = torch.zeros_like(centred)
    out[known] = centred[known] / r[known]
    return out


def cos_angles(sphere_xyz: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine of the angle between unit-sphere positions -> ``[C, C]``."""
    cos = sphere_xyz @ sphere_xyz.transpose(0, 1)
    return cos.clamp(-1.0, 1.0)


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances ``[M, N]`` between ``[M, D]`` and ``[N, D]``.

    Written out rather than using ``torch.cdist`` because several accelerator
    backends (MPS at the time of writing) have no ``cdist`` backward kernel, and
    these distances sit on the gradient path through the learnable query anchors.
    """
    a2 = (a * a).sum(-1, keepdim=True)              # [M, 1]
    b2 = (b * b).sum(-1).unsqueeze(0)               # [1, N]
    return (a2 + b2 - 2.0 * (a @ b.transpose(-2, -1))).clamp_min(0.0)


def pairwise_distance(xyz: torch.Tensor) -> torch.Tensor:
    """Euclidean distance matrix ``[C, C]`` of ``[C, 3]`` positions."""
    return pairwise_sq_dist(xyz, xyz).clamp_min(0.0).sqrt()


def geometric_graph(
    xyz: torch.Tensor,
    sigma: float = 0.3,
    channel_mask: Optional[torch.Tensor] = None,
    self_loops: bool = True,
) -> torch.Tensor:
    """Static geometric affinity ``A_geo[i, j] = exp(-d(i, j)^2 / sigma^2)``.

    Purely a function of electrode positions; independent of the data.  Missing
    channels (``channel_mask == 0``) get zero affinity in both directions so they
    cannot leak into a neighbour's neighbourhood.
    """
    d2 = pairwise_sq_dist(xyz, xyz)
    A = torch.exp(-d2 / (sigma ** 2))
    if not self_loops:
        A = A - torch.diag(torch.diagonal(A))
    if channel_mask is not None:
        m = channel_mask.to(A.dtype)
        A = A * m.unsqueeze(-1) * m.unsqueeze(-2)
    return A


def normalized_graph_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric normalised Laplacian ``L = I - D^-1/2 A D^-1/2`` of ``[C, C]``."""
    deg = A.sum(dim=-1)
    dinv = torch.where(deg > eps, deg.pow(-0.5), torch.zeros_like(deg))
    L = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype) - (
        dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2)
    )
    return L


def montage_hash(
    channel_names: Sequence[str],
    xyz: torch.Tensor,
    channel_mask: Optional[torch.Tensor] = None,
    decimals: int = 4,
) -> str:
    """Stable hash of a montage, used as the SSL operator cache key.

    Includes names, rounded coordinates and the good-channel pattern, because the
    SSL operator changes when any of those change (bad channels are interpolated
    *before* the Laplacian is formed, so the operator is mask-specific).
    """
    arr = np.round(xyz.detach().cpu().double().numpy(), decimals)
    payload = ["|".join(channel_names), arr.tobytes().hex()]
    if channel_mask is not None:
        payload.append("".join("1" if bool(v) else "0" for v in channel_mask.tolist()))
    return hashlib.sha1("::".join(payload).encode()).hexdigest()[:16]
