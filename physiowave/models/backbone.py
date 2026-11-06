"""Factorized backbone over the compressed latent grid ``[B, K, S, D]``.

The sequence is never flattened back to ``(J + 1) * C * S``.  Each block

1. attends along the **time** axis ``S`` independently per latent slot
   (cost ``O(B K S^2 D)``), then
2. mixes the ``K`` latent slots with a light attention over the slot axis
   (cost ``O(B S K^2 D)``, and ``K`` is 4-32), then
3. applies a position-wise FFN.

Compared with full attention over ``K * S`` tokens this replaces an
``O((KS)^2)`` term with ``O(K S^2 + S K^2)``, which is what keeps long windows
affordable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    slot_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_rope: bool = True
    # Post-BERT defaults. Each is switchable so the paper can ablate it rather
    # than assert it: 'layernorm'/'mlp'/qk_norm=False reproduce the original
    # block exactly.
    norm: str = "rmsnorm"          # 'layernorm' | 'rmsnorm'
    ffn: str = "swiglu"            # 'mlp' | 'swiglu'
    qk_norm: bool = True           # normalise q and k per head before attention
    #: Levels of Haar DWT applied to the *keys and values* of temporal attention.
    #: 0 disables it. Each level halves the K/V sequence -- and only the K/V:
    #: the query and the output keep every one of the S positions, so the token
    #: lattice the encoder hands on is unchanged. This is the one compression
    #: that is safe here, because attention already summarises its keys; what is
    #: not safe is compressing the tokens themselves, which is what the channel
    #: compressor did.
    #:
    #: Haar rather than the tokenizer's wavelet: the K/V sequence is short (S is
    #: a handful of patches) and a longer filter's support would run past the
    #: whole axis. Both subbands are kept and packed into the feature dimension,
    #: so the halving discards nothing -- unlike a stride-2 pool, which drops the
    #: high-frequency half outright.
    #: How tokens mix.
    #:
    #: ``factorized``  temporal attention within each channel, then slot
    #:                 attention within each time step. O(K*S^2 + S*K^2), which
    #:                 is what makes a long lattice affordable.
    #: ``full``        one attention over all K*S tokens, the way the legacy
    #:                 model does it. O((K*S)^2) and no cheaper, but a single
    #:                 layer can relate channel i at time t to channel j at time
    #:                 t' -- a pair the factorized form can only reach across
    #:                 two layers, and only through an intermediate that has
    #:                 already been averaged over one axis. On sEMG the class is
    #:                 exactly such a cross-channel, cross-time pattern.
    mixing: str = "factorized"      # 'factorized' | 'full'
    kv_wavelet_level: int = 0
    #: How the two subbands are recombined after each halving.
    #:
    #: ``linear``  a full ``Linear(2D, D)`` per level. Expressive, but it costs
    #:             ``2 * K * S/2 * 2D * D`` per block against an attention saving
    #:             of ``2 * K * S^2 * D / 2`` -- so it only pays for itself once
    #:             ``S`` is past a few hundred. At DB5's S=16 it is a ~30x net
    #:             loss; the arithmetic is in the class docstring below.
    #: ``gated``   ``lo + g * hi`` with a learned per-feature ``g``. Effectively
    #:             free, keeps both subbands, and is the one to use when ``S`` is
    #:             short -- which is every sEMG window in this project.
    kv_mix: str = "linear"          # 'linear' | 'gated'
    #: Init scale of the slot (channel) and time position embeddings. 0.02 is the
    #: usual transformer default, but it is worth naming here: when no channel
    #: metadata encoder is present, the slot embedding is the *only* thing that
    #: tells one channel from another, and at 0.02 it starts around 3% of the
    #: token magnitude. That is learnable but slow, so a run that has to discover
    #: channel identity from scratch may want it larger.
    pos_embed_init: float = 0.02


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction or the bias.

    Two fewer ops per token than LayerNorm and one fewer parameter tensor, at
    no measured cost in quality. The reduction runs in fp32 because under bf16
    autocast the mean of squares over a few hundred channels loses enough
    mantissa to move the normaliser visibly.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm {kind!r}; expected 'rmsnorm' or 'layernorm'")


class SwiGLU(nn.Module):
    """Gated FFN: ``down(silu(gate(x)) * up(x))``.

    Three projections instead of two, so the hidden width is scaled by 2/3 to
    keep the parameter count level with the GELU MLP it replaces -- otherwise a
    'better FFN' comparison is really just a bigger one.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def make_ffn(cfg: "BackboneConfig") -> nn.Module:
    D = cfg.embed_dim
    if cfg.ffn == "swiglu":
        hidden = int(D * cfg.mlp_ratio * 2 / 3)
        hidden += (-hidden) % 8                       # keep the GEMM aligned
        return SwiGLU(D, hidden, cfg.dropout)
    if cfg.ffn == "mlp":
        hidden = int(D * cfg.mlp_ratio)
        return nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                             nn.Linear(hidden, D), nn.Dropout(cfg.dropout))
    raise ValueError(f"unknown ffn {cfg.ffn!r}; expected 'swiglu' or 'mlp'")


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to the temporal axis."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim, self.base = dim, base

    def forward(self, x: torch.Tensor, stride: float = 1.0) -> torch.Tensor:
        """``[B, H, N, D] -> [B, H, N, D]``.

        ``stride`` scales the position index. A key that summarises ``stride``
        original positions sits at ``stride * i`` on the original axis, so
        passing it keeps queries and compressed keys on one coordinate system --
        without it the relative offsets the rotation encodes would be wrong by
        exactly that factor.
        """
        B, H, N, D = x.shape
        pos = torch.arange(N, device=x.device, dtype=x.dtype) * stride
        inv = 1.0 / (self.base ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D))
        freqs = pos[:, None] * inv[None, :]                # [N, D/2]
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        xe, xo = x[..., 0::2], x[..., 1::2]
        out = torch.stack([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1)
        return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    """Standard MHA with optional RoPE, used for both factorized axes."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True,
                 qk_norm: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.h, self.hd = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.hd) if use_rope else None
        # Bounding the norm of q and k keeps the logits from drifting into the
        # range where softmax saturates and gradients stop flowing -- the usual
        # cause of a loss that plateaus early at high learning rates.
        self.q_norm = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                kv: Optional[torch.Tensor] = None,
                kv_stride: float = 1.0) -> torch.Tensor:
        """``[N, L, D] -> [N, L, D]``; ``key_padding_mask`` is ``[N, L]``, True = keep.

        ``kv`` supplies a *separate, shorter* source for the keys and values --
        the wavelet-compressed sequence when that is enabled. The query and the
        output still have ``L`` positions, so the caller's token count does not
        change; only the width of the score matrix does. ``kv_stride`` is how
        many original positions each compressed one covers, and goes to RoPE so
        both sides share a coordinate system.
        """
        N, L, D = x.shape
        # Slice the fused qkv weight rather than projecting both tensors three
        # times: the parameter stays one [3D, D] block, so checkpoints keep
        # loading, but a compressed-K/V block does not pay for keys it discards.
        w, b = self.qkv.weight, self.qkv.bias
        q = F.linear(x, w[:D], b[:D]).reshape(N, L, self.h, self.hd).transpose(1, 2)
        src = x if kv is None else kv
        Lk = src.shape[1]
        kv_pack = F.linear(src, w[D:], b[D:]).reshape(
            N, Lk, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        k, v = kv_pack[0], kv_pack[1]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k, stride=kv_stride)
        attn_mask = None
        if key_padding_mask is not None:
            if kv is not None:
                # A padding mask indexes the original axis; there is no
                # meaningful way to carry it onto a wavelet-mixed key, whose
                # every position blends kept and padded ones.
                raise ValueError("key_padding_mask is not supported with compressed K/V")
            attn_mask = key_padding_mask.view(N, 1, 1, Lk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(N, L, D)
        return self.drop(self.proj(out))


class WaveletKV(nn.Module):
    """Halve a sequence ``J`` times with a Haar DWT, packing subbands into features.

    ``[N, L, D] -> [N, L / 2**J, D]``. At each level the sequence is split into
    its low- and high-pass halves and both are concatenated on the feature axis
    before a linear map takes the width back to ``D``; nothing is thrown away,
    which is what separates this from a stride-2 pool. An odd length is left
    alone -- there is no correct way to halve it, and silently dropping the last
    position would misalign the key positions against the query's.
    """

    def __init__(self, dim: int, levels: int, mix: str = "linear") -> None:
        super().__init__()
        assert levels >= 1
        if mix not in ("linear", "gated"):
            raise ValueError(f"kv_mix must be 'linear' or 'gated', got {mix!r}")
        self.levels, self.mix_kind = levels, mix
        if mix == "linear":
            self.mix = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(levels))
        else:
            # One scalar per feature on the high-pass half: the low-pass half
            # passes through at unit gain, so the layer starts as a plain Haar
            # average and learns how much detail to fold back in.
            self.gate = nn.ParameterList(
                nn.Parameter(torch.zeros(dim)) for _ in range(levels))
        self.norm = nn.ModuleList(make_norm("rmsnorm", dim) for _ in range(levels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Returns the compressed sequence and the stride it actually achieved.

        The stride is measured, not assumed: the loop stops early on an odd
        length, and a caller that trusted ``2 ** levels`` would then hand RoPE a
        key-position scale that does not match the tensor it just received.
        """
        L0 = x.shape[1]
        for i, norm in enumerate(self.norm):
            L = x.shape[1]
            if L < 2 or L % 2 != 0:
                break
            even, odd = x[:, 0::2], x[:, 1::2]
            lo = (even + odd) / math.sqrt(2.0)          # Haar low-pass
            hi = (even - odd) / math.sqrt(2.0)          # Haar high-pass
            if self.mix_kind == "linear":
                x = norm(self.mix[i](torch.cat([lo, hi], dim=-1)))
            else:
                x = norm(lo + self.gate[i] * hi)
        return x, L0 / x.shape[1]


class FactorizedBlock(nn.Module):
    """Temporal attention -> slot mixing -> FFN, each pre-normed and residual."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        D = cfg.embed_dim
        self.n1 = make_norm(cfg.norm, D)
        self.time_attn = MultiHeadAttention(D, cfg.num_heads, cfg.dropout, cfg.use_rope,
                                            cfg.qk_norm)
        # Temporal K/V only. The slot axis is the channel axis and is already
        # short; compressing it would be compressing channel evidence, which is
        # the thing this architecture exists to stop doing.
        self.kv_wave = WaveletKV(D, cfg.kv_wavelet_level, cfg.kv_mix) \
            if cfg.kv_wavelet_level else None
        self.full = cfg.mixing == "full"
        # Not allocated in 'full': one attention already spans both axes, and an
        # unused module would still be counted in the parameter budget the
        # comparison is supposed to hold fixed.
        self.n2 = None if self.full else make_norm(cfg.norm, D)
        self.slot_attn = None if self.full else MultiHeadAttention(
            D, cfg.slot_heads, cfg.dropout, use_rope=False, qk_norm=cfg.qk_norm)
        self.n3 = make_norm(cfg.norm, D)
        self.mlp = make_ffn(cfg)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, K, S, D] -> [B, K, S, D]``."""
        B, K, S, D = x.shape
        if self.full:
            if time_mask is not None:
                raise ValueError("mixing='full' does not take a time mask")
            h = self.n1(x).reshape(B, K * S, D)
            x = x + self.time_attn(h).view(B, K, S, D)
            return x + self.mlp(self.n3(x))
        h = self.n1(x).reshape(B * K, S, D)
        tm = None
        if time_mask is not None:
            tm = time_mask.unsqueeze(1).expand(B, K, S).reshape(B * K, S)
        if self.kv_wave is not None:
            if time_mask is not None:
                raise ValueError("kv_wavelet_level cannot be combined with a time mask")
            kv, stride = self.kv_wave(h)
            x = x + self.time_attn(h, None, kv=kv, kv_stride=stride).view(B, K, S, D)
        else:
            x = x + self.time_attn(h, tm).view(B, K, S, D)

        h = self.n2(x).permute(0, 2, 1, 3).reshape(B * S, K, D)
        x = x + self.slot_attn(h).view(B, S, K, D).permute(0, 2, 1, 3)

        return x + self.mlp(self.n3(x))


class FactorizedBackbone(nn.Module):
    """Stack of :class:`FactorizedBlock` with a learned temporal position embedding."""

    def __init__(self, cfg: BackboneConfig, max_patches: int = 512) -> None:
        super().__init__()
        self.cfg = cfg
        std = cfg.pos_embed_init
        self.slot_embed = nn.Parameter(torch.randn(1, 64, 1, cfg.embed_dim) * std)
        self.time_embed = nn.Parameter(torch.randn(1, 1, max_patches, cfg.embed_dim) * std)
        self.blocks = nn.ModuleList(FactorizedBlock(cfg) for _ in range(cfg.depth))
        self.norm = make_norm(cfg.norm, cfg.embed_dim)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 4, f"expected [B, K, S, D], got {tuple(x.shape)}"
        B, K, S, D = x.shape
        assert K <= self.slot_embed.shape[1], f"K={K} exceeds slot embedding capacity"
        assert S <= self.time_embed.shape[2], f"S={S} exceeds max_patches"
        x = x + self.slot_embed[:, :K] + self.time_embed[:, :, :S]
        for blk in self.blocks:
            x = blk(x, time_mask)
        return self.norm(x)
