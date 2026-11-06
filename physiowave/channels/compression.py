"""Topology-aware channel compression: ``[B, C, S, D] -> [B, K, S, D]`` with ``K << C``.

Why ``K`` can be far smaller than ``C``
---------------------------------------
Volume conduction low-pass filters the potential field on its way from cortex to
scalp, so neighbouring electrodes see strongly overlapping mixtures.  The
*effective* spatial degrees of freedom of a scalp EEG recording are therefore far
fewer than the number of electrodes -- which is why classical pipelines get away
with a handful of CSP or ICA components.  A small set of topology-aware queries,
each anchored at a learnable scalp location, is enough to retain that spatial
information, and it makes the token count independent of the montage: a 19-channel
clinical recording and a 64-channel HD recording both produce ``K * S`` tokens.

Cost
----
One query-channel attention is ``O(B * S * K * C * D)``; the backbone that follows
never re-expands to the ``(J + 1) * C * S`` sequence the legacy model used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spatial.geometry import pairwise_sq_dist
from .tare import FourierPositionEncoding


@dataclass
class CompressionConfig:
    """Configuration for :class:`ChannelCompressor`."""

    num_queries: int = 16              # K; 4 / 8 / 16 / 32 are the ablation points
    embed_dim: int = 256
    num_heads: int = 4
    dist_bias_scale: float = 1.0
    graph_bias_scale: float = 1.0
    use_graph_bias: bool = True
    dropout: float = 0.0
    anchor_init_spread: float = 0.8


class ChannelCompressor(nn.Module):
    """Learned-query cross-attention over the channel axis.

    Each query owns a learnable 3-D scalp anchor.  Attention logits combine

    * content similarity between the query and the channel's token features,
    * the channel's TARE embedding (so metadata steers the pooling),
    * a geometric bias from the anchor-to-electrode distance,
    * a bias derived from the channel-relation graph ``A`` (detached).

    ``channel_mask`` hard-masks missing or bad channels: their attention weight is
    exactly zero, which ``tests/test_channels.py`` asserts.
    """

    def __init__(self, cfg: CompressionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, K, H = cfg.embed_dim, cfg.num_queries, cfg.num_heads
        assert D % H == 0, f"embed_dim={D} must be divisible by num_heads={H}"
        self.head_dim = D // H

        self.query = nn.Parameter(torch.randn(K, D) * 0.02)
        # Anchors start spread over the upper hemisphere so queries begin
        # specialised rather than collapsed on one spot.
        anchors = torch.randn(K, 3)
        anchors = anchors / anchors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        anchors[:, 2] = anchors[:, 2].abs()
        self.anchors = nn.Parameter(anchors * cfg.anchor_init_spread)

        self.anchor_pe = FourierPositionEncoding(n_bands=6)
        self.anchor_proj = nn.Linear(self.anchor_pe.out_dim, D)

        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.q_proj = nn.Linear(D, D)
        self.chan_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(D)

        self.log_tau = nn.Parameter(torch.zeros(K))          # anchor kernel width
        self.graph_gain = nn.Parameter(torch.zeros(1))
        self._last_attn: Optional[torch.Tensor] = None

    # -- biases ----------------------------------------------------------------
    def _distance_bias(self, xyz: torch.Tensor) -> torch.Tensor:
        """``[K, C]`` bias from anchor-to-electrode distance."""
        d2 = pairwise_sq_dist(self.anchors, xyz)
        tau = F.softplus(self.log_tau).clamp_min(1e-3).unsqueeze(-1)
        return -self.cfg.dist_bias_scale * d2 / tau

    def _graph_bias(self, A: torch.Tensor, dist_bias: torch.Tensor) -> torch.Tensor:
        """``[B, K, C]`` bias injecting the channel-relation graph.

        The anchor kernel gives each query a soft neighbourhood over channels;
        propagating that neighbourhood through ``A`` scores a channel by how
        strongly it relates to the query's region.  ``A`` is detached -- these
        statistics shape attention but never receive gradient.
        """
        w = torch.softmax(dist_bias, dim=-1)                 # [K, C]
        if A.dim() == 2:
            A = A.unsqueeze(0)
        prop = torch.einsum("kc,bcd->bkd", w, A.detach().to(w.dtype))
        prop = prop / prop.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.cfg.graph_bias_scale * self.graph_gain * prop

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        channel_embedding: torch.Tensor,
        xyz: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        relation_graph: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compress the channel axis.

        Args:
            tokens: ``[B, C, S, D]`` per-channel patch tokens from WAST.
            channel_embedding: ``[C, D]`` TARE embeddings.
            xyz: ``[C, 3]`` electrode positions.
            channel_mask: ``[C]`` or ``[B, C]`` bool; False = missing/bad.
            relation_graph: ``[C, C]`` or ``[B, C, C]`` detached ``A``.

        Returns:
            ``{'tokens': [B, K, S, D], 'attn': [B, S, K, C]}``
        """
        assert tokens.dim() == 4, f"expected [B, C, S, D], got {tuple(tokens.shape)}"
        B, C, S, D = tokens.shape
        K, H, hd = self.cfg.num_queries, self.cfg.num_heads, self.head_dim
        assert channel_embedding.shape == (C, D), (
            f"channel_embedding must be [C, D] = {(C, D)}, got {tuple(channel_embedding.shape)}"
        )

        chan = self.chan_proj(channel_embedding).view(1, C, 1, D)
        feat = tokens + chan                                  # metadata-conditioned features
        k = self.k_proj(feat).view(B, C, S, H, hd)
        v = self.v_proj(feat).view(B, C, S, H, hd)

        q = self.query + self.anchor_proj(self.anchor_pe(self.anchors))
        q = self.q_proj(q).view(K, H, hd)

        logits = torch.einsum("khd,bcshd->bskc", q, k) / (hd ** 0.5)   # [B, S, K, C]

        dist_bias = self._distance_bias(xyz)                  # [K, C]
        logits = logits + dist_bias.view(1, 1, K, C)
        if relation_graph is not None and self.cfg.use_graph_bias:
            gb = self._graph_bias(relation_graph, dist_bias)  # [B, K, C]
            logits = logits + gb.unsqueeze(1)

        if channel_mask is not None:
            m = channel_mask.to(torch.bool)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(B, -1)
            logits = logits.masked_fill(~m.view(B, 1, 1, C), float("-inf"))
            # A query whose entire neighbourhood is masked would produce NaN;
            # fall back to a uniform distribution over the surviving channels.
            all_masked = (~m).all(dim=-1)
            if bool(all_masked.any()):
                logits = torch.where(all_masked.view(B, 1, 1, 1),
                                     torch.zeros_like(logits), logits)

        attn = torch.softmax(logits, dim=-1)
        if channel_mask is not None:
            attn = attn.masked_fill(~m.view(B, 1, 1, C), 0.0)
        attn = self.drop(attn)
        self._last_attn = attn

        out = torch.einsum("bskc,bcshd->bskhd", attn, v).reshape(B, S, K, D)
        out = self.out_proj(out).permute(0, 2, 1, 3).contiguous()      # [B, K, S, D]
        out = self.norm(out)
        assert out.shape == (B, K, S, D), f"expected [B, K, S, D], got {tuple(out.shape)}"
        return {"tokens": out, "attn": attn}

    # -- regularisation --------------------------------------------------------
    def query_specialization_loss(self, attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Penalise queries that attend to the same channels (query collapse).

        Two terms: the mean pairwise cosine similarity between query attention
        distributions, and a repulsion term between the 3-D anchors.  Without this
        the queries reliably collapse onto the highest-variance electrodes.
        """
        a = attn if attn is not None else self._last_attn
        K = self.cfg.num_queries
        eye = torch.eye(K, device=self.anchors.device)
        if a is None:
            sim_loss = self.anchors.new_zeros(())
        else:
            p = a.mean(dim=(0, 1))                            # [K, C]
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sim = p @ p.transpose(0, 1)
            sim_loss = (sim - eye).abs().sum() / max(K * (K - 1), 1)
        d2 = pairwise_sq_dist(self.anchors, self.anchors)
        rep = torch.exp(-d2 / 0.1) * (1 - eye)
        anchor_loss = rep.sum() / max(K * (K - 1), 1)
        return sim_loss + anchor_loss
