"""GL branch -- learnable Graph-Laplacian spatial filter (**CSD-inspired**).

Naming (constraint B of ``docs/terminology.md``)
------------------------------------------------
This branch applies a normalised graph Laplacian built from electrode geometry,
with learnable edge weights and a learnable output gate.  It is a coarse,
trainable relative of a surface Laplacian -- essentially a learnable Hjorth-style
local spatial filter -- and is therefore called **CSD-inspired**.  It is *not*
CSD.  The strict spherical-spline CSD lives in
:mod:`physiowave.spatial.spline_laplacian` and is called the **SSL branch**.
The two are parallel branches; the names are never used interchangeably.

Like SSL, this branch is *fused* with the raw branch through a gate rather than
replacing it: a Laplacian is a spatial band-pass that attenuates deep and widely
distributed generators, so using it alone throws away real signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .geometry import geometric_graph, normalized_graph_laplacian


@dataclass
class GLConfig:
    """Configuration for the GL branch."""

    enabled: bool = True
    sigma: float = 0.5
    gate_init: float = 0.1
    learnable_edges: bool = True


class GraphLaplacianBranch(nn.Module):
    """``X_lap = L_geo @ X`` with a learnable edge modulation and output gate.

    Args:
        max_channels: upper bound on ``C`` used to size the learnable edge
            modulation table; montages with fewer channels use the top-left block.
    """

    def __init__(self, cfg: GLConfig, max_channels: int = 128) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_channels = max_channels
        self.gate = nn.Parameter(torch.full((1,), float(cfg.gate_init)))
        if cfg.learnable_edges:
            # Modulates the geometric affinity multiplicatively via exp(delta),
            # initialised at 0 so the branch starts as pure geometry.
            self.edge_delta = nn.Parameter(torch.zeros(max_channels, max_channels))
        else:
            self.register_parameter("edge_delta", None)

    def laplacian(
        self,
        xyz: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalised graph Laplacian ``[C, C]`` for one montage."""
        C = xyz.shape[0]
        A = geometric_graph(xyz, self.cfg.sigma, channel_mask)
        if self.edge_delta is not None:
            assert C <= self.max_channels, (
                f"C={C} exceeds max_channels={self.max_channels}; raise gl.max_channels"
            )
            d = self.edge_delta[:C, :C]
            d = 0.5 * (d + d.transpose(0, 1))          # keep the graph undirected
            A = A * torch.exp(d.clamp(-3.0, 3.0))
            if channel_mask is not None:
                m = channel_mask.to(A.dtype)
                A = A * m.unsqueeze(-1) * m.unsqueeze(-2)
        return normalized_graph_laplacian(A)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[B, C, T] -> [B, C, T]``: the gated Laplacian-filtered signal.

        Returns the *contribution* ``g * (L @ X)``; the caller adds it to the raw
        branch.  The gate starts small so training begins from the raw path.
        """
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        L = self.laplacian(xyz, channel_mask).to(x.dtype)
        return self.gate * torch.einsum("ij,bjt->bit", L, x)
