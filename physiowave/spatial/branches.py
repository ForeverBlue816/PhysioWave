"""Spatial front-end: raw + SSL + GL branches and the channel-relation graph ``A``.

The four parts, with their names fixed by ``docs/terminology.md``:

``A_geo``   static geometric affinity from electrode coordinates (data independent)
``A_dyn``   *spatial statistics* / channel-relation graph estimated from the data.
            It is contaminated by the reference montage and by volume conduction
            and is never described as connectivity.  # TERMINOLOGY-ALLOW
``SSL``     strict spherical-spline surface Laplacian (Perrin et al. 1989)
``GL``      learnable graph Laplacian, *CSD-inspired* only

Both Laplacian branches are **added to** the raw branch through learnable gates,
never substituted for it: a surface Laplacian is a spatial band-pass that
suppresses deep and broadly distributed generators, so a Laplacian-only model
throws away real signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..channels.tare import ChannelMeta
from .geometry import geometric_graph
from .graph_laplacian import GLConfig, GraphLaplacianBranch
from .spatial_stats import DynGraphConfig, SpatialStatGraph
from .spline_laplacian import SSLConfig, SSLOperatorCache

logger = logging.getLogger(__name__)


@dataclass
class SpatialConfig:
    """Configuration of the whole spatial front-end."""

    use_raw: bool = True
    ssl: SSLConfig = field(default_factory=SSLConfig)
    gl: GLConfig = field(default_factory=GLConfig)
    dyn: DynGraphConfig = field(default_factory=DynGraphConfig)
    geo_sigma: float = 0.5
    lambda_geo_init: float = 1.0
    lambda_dyn_init: float = 0.5
    learnable_lambdas: bool = True
    max_channels: int = 128


class SpatialFrontend(nn.Module):
    """Produces the branch-fused signal and the relation graph ``A``.

    Forward returns:
        ``signal``   ``[B, C, T]`` -- ``X_raw + g_gl * (L_geo X) + g_ssl * (L_ssl X)``
        ``A``        ``[B, C, C]`` -- ``lambda_g A_geo + lambda_d A_dyn`` (detached)
        ``ssl_signal`` ``[B, C, T]`` or ``None`` -- the pure SSL view, used as the
            reference-invariant anchor by the pretraining objective
        ``info``     dict of which branches actually ran and why any were skipped
    """

    def __init__(self, cfg: SpatialConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ssl_cache = SSLOperatorCache(cfg.ssl.cache_dir)
        self.ssl_gate = nn.Parameter(torch.full((1,), float(cfg.ssl.gate_init)))
        self.gl = GraphLaplacianBranch(cfg.gl, cfg.max_channels) if cfg.gl.enabled else None
        self.dyn = SpatialStatGraph(cfg.dyn) if cfg.dyn.enabled else None
        if cfg.learnable_lambdas:
            self.lambda_geo = nn.Parameter(torch.tensor(float(cfg.lambda_geo_init)))
            self.lambda_dyn = nn.Parameter(torch.tensor(float(cfg.lambda_dyn_init)))
        else:
            self.register_buffer("lambda_geo", torch.tensor(float(cfg.lambda_geo_init)))
            self.register_buffer("lambda_dyn", torch.tensor(float(cfg.lambda_dyn_init)))

    # -- SSL -------------------------------------------------------------------
    def ssl_operator(self, meta: ChannelMeta, device) -> Optional[torch.Tensor]:
        """Cached ``[C, C]`` surface-Laplacian operator, or ``None`` if skipped."""
        if not self.cfg.ssl.enabled:
            return None
        L = self.ssl_cache.get(
            meta.channel_names, meta.channel_xyz, meta.channel_mask,
            self.cfg.ssl, meta.derivation_type,
        )
        return None if L is None else L.to(device)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        meta: ChannelMeta,
        fs: float,
    ) -> Dict[str, object]:
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        device = x.device
        xyz = meta.channel_xyz.to(device=device, dtype=x.dtype)
        mask = None if meta.channel_mask is None else meta.channel_mask.to(device)

        info: Dict[str, object] = {"raw": self.cfg.use_raw, "ssl": False, "gl": False}
        out = x if self.cfg.use_raw else torch.zeros_like(x)

        L_ssl = self.ssl_operator(meta, device)
        ssl_signal: Optional[torch.Tensor] = None
        if L_ssl is not None:
            ssl_signal = torch.einsum("ij,bjt->bit", L_ssl.to(x.dtype), x)
            out = out + self.ssl_gate * ssl_signal
            info["ssl"] = True
        else:
            info["ssl_skip_reason"] = dict(self.ssl_cache.skips) or "disabled"

        if self.gl is not None:
            out = out + self.gl(x, xyz, mask)
            info["gl"] = True

        A_geo = geometric_graph(xyz, self.cfg.geo_sigma, mask)          # [C, C]
        A = self.lambda_geo * A_geo.unsqueeze(0).expand(B, -1, -1)
        if self.dyn is not None:
            stat_input = x
            if self.cfg.dyn.dyn_graph_input == "ssl":
                if ssl_signal is None:
                    logger.warning(
                        "dyn_graph_input='ssl' requested but the SSL branch is "
                        "unavailable for this montage; falling back to raw."
                    )
                    info["dyn_input_fallback"] = "raw"
                else:
                    stat_input = ssl_signal
            # `spatial_stat_graph` == A_dyn: a spatial statistic of the recorded
            # signals, contaminated by reference and volume conduction.  It is a
            # channel-relation graph, not a statement about neural interaction.
            spatial_stat_graph = self.dyn(stat_input, fs)
            A = A + self.lambda_dyn * spatial_stat_graph
            info["dyn_graph_type"] = self.cfg.dyn.dyn_graph_type
            info["dyn_condition_number"] = self.dyn.last_condition_number

        return {"signal": out, "A": A.detach(), "ssl_signal": ssl_signal,
                "A_geo": A_geo.detach(), "info": info}

    def ssl_cache_stats(self) -> Dict[str, object]:
        return self.ssl_cache.stats()
