"""WAST -- Wavelet Analysis-Synthesis Tokenizer.

WAST turns a raw multi-channel recording ``[B, C, T]`` into per-channel patch
tokens ``[B, C, S, D]`` while performing a genuine multi-level wavelet analysis
on the way.  The whole point is that the analysis is *token neutral*:

.. code-block:: text

    legacy PhysioWave : bands are upsampled back to T and concatenated on the
                        channel axis  ->  N_old = (J + 1) * C * S tokens
    WAST              : bands are processed in the coefficient domain and
                        synthesised back to one patch  ->  C * S tokens,
                        which channel compression then reduces to K * S

Adding decomposition levels therefore changes what a token *contains*, never how
many tokens there are.  ``token_report`` returns both counts so the compression
ratio in the paper tables is computed, not asserted.

Pipeline (``placement='post_patch'``, the default)
--------------------------------------------------
1. ``[B, C, T]``            -- input, asserted 3-D
2. ``[B, C, S, P]``         -- split the time axis into ``S = T // P`` patches
3. ``[B*C*S, P]``           -- flatten; run the critically-sampled DWT along time
   only.  EEG channels are *never* treated as a transform axis: the channel
   ordering of a montage carries no metric structure, so a 2-D DWT over
   (channel, time) would filter across an arbitrary permutation.  Spatial
   structure is handled by TARE instead.
4. per-subband depthwise conv + norm + learnable gate (:class:`SubbandProcessor`)
5. inverse DWT back to ``[B*C*S, P]`` -- fixed patch length, no upsampled copies
6. ``[B, C, S, D]``         -- linear projection of each processed patch

``placement='pre_patch'`` performs the DWT on the full ``T`` axis before
patching.  It sees longer support at coarse scales but couples patches, so
``post_patch`` is the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dwt import (
    DEFAULT_BOUNDARY_MODE,
    DEFAULT_WAVELET,
    WaveletTransform1D,
    coeff_lengths,
)
from .fgm import band_energies, frequency_guided_patch_scores


@dataclass
class WASTConfig:
    """Configuration for :class:`WAST`."""

    patch_size: int = 64
    embed_dim: int = 256
    wavelet: str = DEFAULT_WAVELET
    level: int = 3
    boundary_mode: str = DEFAULT_BOUNDARY_MODE
    learnable_filters: bool = True
    placement: str = "post_patch"       # 'post_patch' | 'pre_patch'
    subband_width: int = 8              # hidden width of the per-subband processor
    subband_kernel: int = 5
    subband_dropout: float = 0.0
    residual_scale_init: float = 0.1
    norm_patches: bool = True
    use_band_features: bool = True      # inject per-band log-energy into the token
    #: How a patch becomes a token.  All three produce ``C * S`` tokens; they
    #: differ in what a token is built from, so they are the A/B/C rungs of the
    #: tokenizer ablation.
    #:
    #: ``synthesis``  DWT -> per-subband processing -> IDWT -> project the
    #:                reconstructed patch.  The original WAST.
    #: ``fold``       DWT -> per-subband processing -> project each subband into
    #:                its own slice of the embedding -> mix.  The scales never
    #:                re-mix in the time domain, so each keeps a private
    #:                subspace; nothing is upsampled and no token is added.
    #: ``raw``        no wavelet at all: project the patch.  The floor the other
    #:                two have to beat.
    tokenizer: str = "synthesis"        # 'synthesis' | 'fold' | 'raw'
    #: One learnable filter bank per channel instead of one shared by all.
    #:
    #: The legacy tokenizer has always done this -- its wavelet convolutions use
    #: ``groups=in_channels``, so every channel gets its own filters. WAST shares
    #: a single bank across the whole montage, which is the right default for a
    #: scalp recording where the electrodes see one field. A forearm ring is the
    #: opposite case: each electrode sits over a different muscle with different
    #: spectral content, and one bank has to compromise across all of them.
    #:
    #: Banks are initialised from the same wavelet and diverge during training,
    #: which is also what the legacy filters do. Indexed by channel position, so
    #: this is montage-specific by construction.
    channel_filters: bool = False
    max_channels: int = 64              # 'channel_filters' bank count
    #: 'fold' only: the raw patch is projected too and the folded wavelet token
    #: is added through a gate initialised here, so training starts at a plain
    #: patch embedding and earns its way into the multi-scale path.
    fold_raw_residual: bool = True
    fold_gate_init: float = 0.1

    def __post_init__(self) -> None:
        if self.placement not in ("post_patch", "pre_patch"):
            raise ValueError("placement must be 'post_patch' or 'pre_patch'")
        if self.tokenizer not in ("synthesis", "fold", "raw"):
            raise ValueError(
                f"tokenizer must be 'synthesis', 'fold' or 'raw', got {self.tokenizer!r}")
        if self.tokenizer == "fold" and self.placement != "post_patch":
            raise ValueError("tokenizer='fold' packs a patch's own subbands and "
                             "requires placement='post_patch'")
        if self.patch_size % (2 ** self.level) != 0:
            raise ValueError(
                f"patch_size={self.patch_size} must be divisible by "
                f"2**level={2 ** self.level} for a critically sampled transform"
            )


def split_embedding(embed_dim: int, coeff_lens: Sequence[int]) -> List[int]:
    """Divide ``embed_dim`` among the subbands in proportion to their length.

    A finer scale carries more coefficients and therefore more of the patch's
    variance, so it gets more of the embedding; every band is guaranteed at
    least one dimension, and the remainder lands on the coarsest band. The parts
    sum to ``embed_dim`` exactly -- the concatenation below has no room to be
    off by one.
    """
    total = sum(coeff_lens)
    if embed_dim < len(coeff_lens):
        raise ValueError(f"embed_dim={embed_dim} cannot be split across "
                         f"{len(coeff_lens)} subbands")
    dims = [max(1, (embed_dim * L) // total) for L in coeff_lens]
    slack = embed_dim - sum(dims)
    dims[0] += slack                                  # cA_J absorbs the rounding
    if dims[0] < 1:                                   # only if rounding overshot
        raise ValueError(f"cannot split embed_dim={embed_dim} across {coeff_lens}")
    assert sum(dims) == embed_dim
    return dims


class SubbandProcessor(nn.Module):
    """Lightweight per-subband operator: expand -> depthwise conv -> norm -> gate.

    Operates on ``[N, 1, P_k]`` coefficient sequences.  The gate is the product of
    a learnable per-subband scalar and a content-dependent scalar derived from the
    subband's own energy, so the model can attenuate or emphasise a scale without
    ever changing its length.
    """

    def __init__(self, length: int, width: int = 8, kernel: int = 5, dropout: float = 0.0,
                 residual_scale_init: float = 0.1) -> None:
        super().__init__()
        self.length = length
        k = min(kernel, length if length % 2 == 1 else length - 1)
        k = max(k, 1)
        self.expand = nn.Conv1d(1, width, kernel_size=1)
        self.depthwise = nn.Conv1d(width, width, kernel_size=k, padding=k // 2, groups=width)
        self.norm = nn.GroupNorm(1, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.project = nn.Conv1d(width, 1, kernel_size=1)
        self.content_gate = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, 1))
        self.band_gate = nn.Parameter(torch.zeros(1))          # sigmoid(0) = 0.5
        self.res_scale = nn.Parameter(torch.full((1,), residual_scale_init))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """``[N, P_k] -> [N, P_k]`` (length preserved -- critical sampling is invariant)."""
        N, P_k = c.shape
        h = c.unsqueeze(1)                       # [N, 1, P_k]
        h = self.expand(h)
        h = self.depthwise(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.project(h).squeeze(1)           # [N, P_k]
        energy = c.detach().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).log()
        gate = torch.sigmoid(self.band_gate + self.content_gate(energy))  # [N, 1]
        out = c + self.res_scale * gate * h
        assert out.shape == c.shape, "subband processing must preserve the coefficient count"
        return out


class WAST(nn.Module):
    """Wavelet Analysis-Synthesis Tokenizer.

    Args:
        cfg: :class:`WASTConfig`.

    Forward returns a dict with:
        ``tokens``    ``[B, C, S, D]``
        ``patches``   ``[B, C, S, P]`` reconstructed (post-processing) patches
        ``raw_patches`` ``[B, C, S, P]`` the untouched input patches
        ``coeffs``    list of ``[B*C*S, P_k]`` processed subbands
        ``raw_coeffs`` list of ``[B*C*S, P_k]`` analysis subbands (pre-processing)
        ``patch_scores`` ``[B, S]`` detached FgM importance scores
    """

    def __init__(self, cfg: WASTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        def _bank() -> WaveletTransform1D:
            return WaveletTransform1D(
                wavelet=cfg.wavelet,
                level=cfg.level,
                boundary_mode=cfg.boundary_mode,
                learnable=cfg.learnable_filters,
            )

        self.wt = _bank()               # shared bank; also the source of lengths
        self.channel_wt = nn.ModuleList(
            _bank() for _ in range(cfg.max_channels)) if cfg.channel_filters else None
        lens = coeff_lengths(cfg.patch_size, cfg.level)
        self.coeff_lens: List[int] = lens
        self.subbands = nn.ModuleList(
            SubbandProcessor(L, cfg.subband_width, cfg.subband_kernel,
                             cfg.subband_dropout, cfg.residual_scale_init)
            for L in lens
        )
        self.patch_norm = nn.LayerNorm(cfg.patch_size) if cfg.norm_patches else nn.Identity()
        self.proj = nn.Linear(cfg.patch_size, cfg.embed_dim)
        self.band_weights = nn.Parameter(torch.zeros(len(lens)))

        # -- 'fold': one private slice of the embedding per subband ------------
        self.band_dims: List[int] = []
        if cfg.tokenizer == "fold":
            self.band_dims = split_embedding(cfg.embed_dim, lens)
            self.band_norms = nn.ModuleList(nn.LayerNorm(L) for L in lens)
            self.band_projs = nn.ModuleList(
                nn.Linear(L, d) for L, d in zip(lens, self.band_dims, strict=True))
            self.fold_mix = nn.Linear(cfg.embed_dim, cfg.embed_dim)
            # atanh(gate_init), so tanh() of the parameter starts at gate_init.
            g = float(min(max(cfg.fold_gate_init, -0.999), 0.999))
            self.fold_gate = nn.Parameter(torch.full((1,), math.atanh(g)))
        # Per-band log-energy features.  Analysis followed by synthesis is an exact
        # inverse, so without this path the choice of wavelet basis would only
        # reach the token through the subband gates.  Feeding the (differentiable,
        # *not* detached) band energies into the projection makes the basis a
        # first-class part of the representation -- this is the path that
        # `tests/test_wavelet.py::test_wavelet_parameters_change_output` exercises.
        self.band_feat = nn.Sequential(
            nn.Linear(len(lens), cfg.embed_dim // 2), nn.GELU(),
            nn.Linear(cfg.embed_dim // 2, cfg.embed_dim),
        ) if cfg.use_band_features else None

    # -- helpers ---------------------------------------------------------------
    @property
    def num_bands(self) -> int:
        return self.cfg.level + 1

    def num_patches(self, T: int) -> int:
        return T // self.cfg.patch_size

    def token_report(self, C: int, T: int, K: Optional[int] = None,
                     kv_stride: int = 1) -> Dict[str, float]:
        """Token accounting for the legacy vs WAST vs compressed paths.

        The number worth reporting is ``N_out``: what the backbone carries and
        what the head eventually pools. ``N_kv`` is separate on purpose --
        compressing the keys of an attention is not the same act as deleting
        tokens from the representation, and collapsing the two into one
        "compression ratio" is how a table ends up claiming information was kept
        when it was not.
        """
        S = self.num_patches(T)
        J = self.cfg.level
        n_old = (J + 1) * C * S            # legacy: one token per (scale, channel, patch)
        n_lattice = C * S                  # the channel-time lattice the input arrives as
        n_out = n_lattice if K is None else K * S
        report = {
            "S": S, "C": C, "J": J,
            "N_patch_in": n_lattice,
            "N_old_legacy": n_old,
            "N_wast": n_lattice,
            "N_out": n_out,
            # What removing the per-scale token expansion bought, on its own.
            "scale_token_reduction": n_old / max(n_lattice, 1),
            "compression_vs_legacy": n_old / max(n_lattice, 1),
            "N_kv": max(n_out // max(kv_stride, 1), 1) if kv_stride > 1 else n_out,
            "kv_stride": kv_stride,
        }
        if K is not None:
            report["K"] = K
            report["N_new"] = n_out
            # Channel compression on top -- reported separately because it is a
            # different claim: the lattice itself got smaller.
            report["channel_compression"] = n_lattice / max(n_out, 1)
            report["compression_vs_legacy_with_compression"] = n_old / max(n_out, 1)
        return report

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, T] -> [B, C, S, P]``."""
        assert x.dim() == 3, f"WAST expects [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        P = self.cfg.patch_size
        assert T % P == 0, (
            f"signal length T={T} must be a multiple of patch_size={P}; "
            "crop or pad the window in the data layer"
        )
        return x.view(B, C, T // P, P)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert x.dim() == 3, f"WAST expects [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        P = self.cfg.patch_size

        if self.cfg.placement == "pre_patch":
            # Analyse the whole time axis, then split the synthesised signal.
            flat = x.reshape(B * C, T)                      # [B*C, T]
            coeffs = self.wt.analysis(flat)
            proc = [sb(c) for sb, c in zip(self.subbands_for(T), coeffs, strict=True)]
            rec = self.wt.synthesis(proc, T)                # [B*C, T]
            raw_patches = x.view(B, C, T // P, P)
            patches = rec.view(B, C, T // P, P)
            raw_coeffs, coeffs_out = coeffs, proc
            S = T // P
        else:
            raw_patches = self.patchify(x)                  # [B, C, S, P]
            S = raw_patches.shape[2]
            flat = raw_patches.reshape(B * C * S, P)        # [N, P]
            raw_coeffs = self.analyse(flat, B, C, S, P)
            assert sum(c.shape[-1] for c in raw_coeffs) == P, (
                "WAST analysis broke critical sampling"
            )
            if self.cfg.tokenizer == "raw":
                coeffs_out = raw_coeffs
                patches = raw_patches
            else:
                coeffs_out = [sb(c) for sb, c in zip(self.subbands, raw_coeffs, strict=True)]
                if self.cfg.tokenizer == "fold":
                    # No synthesis: the subbands stay separate all the way into
                    # the embedding. Reconstructing first would re-mix the scales
                    # in the time domain, and a linear projection of the mixture
                    # cannot undo that -- which is the whole reason this rung
                    # exists.
                    patches = raw_patches
                else:
                    rec = self.synthesise(coeffs_out, B, C, S, P)
                    patches = rec.view(B, C, S, P)

        energies = band_energies(raw_coeffs, B, C, S) if self.cfg.placement == "post_patch" \
            else self._pre_patch_energies(raw_coeffs, B, C, S, P)
        scores = frequency_guided_patch_scores(
            energies, torch.softmax(self.band_weights, dim=0), channel_mask
        )

        h = self.patch_norm(patches)
        tokens = self.proj(h)                               # [B, C, S, D]
        if self.cfg.tokenizer == "fold":
            parts = [proj(norm(c)) for c, norm, proj
                     in zip(coeffs_out, self.band_norms, self.band_projs, strict=True)]
            wave = self.fold_mix(torch.cat(parts, dim=-1))  # [N, D]
            wave = wave.view(B, C, S, self.cfg.embed_dim)
            if self.cfg.fold_raw_residual:
                tokens = tokens + torch.tanh(self.fold_gate) * wave
            else:
                tokens = wave
        if self.band_feat is not None:
            # Differentiable band energies (the detached copy above is only for FgM).
            be = torch.stack(
                [c.pow(2).mean(dim=-1).view(B, C, S) for c in coeffs_out], dim=-1
            )
            tokens = tokens + self.band_feat(torch.log1p(be.clamp_min(0.0)))
        assert tokens.shape == (B, C, S, self.cfg.embed_dim), (
            f"expected [B, C, S, D] = {(B, C, S, self.cfg.embed_dim)}, got {tuple(tokens.shape)}"
        )
        return {
            "tokens": tokens,
            "patches": patches,
            "raw_patches": raw_patches,
            "coeffs": coeffs_out,
            "raw_coeffs": raw_coeffs,
            "patch_scores": scores,
            "num_patches": S,
        }

    def analyse(self, flat: torch.Tensor, B: int, C: int, S: int,
                P: int) -> List[torch.Tensor]:
        """``[B*C*S, P] -> per-band ``[B*C*S, P_k]``, honouring per-channel banks.

        ``flat`` is indexed ``((b * C + c) * S + s)``, so a per-channel result is
        rebuilt by stacking on axis 1 and flattening again -- getting that order
        wrong would silently pair each channel's signal with another's filters.
        """
        if self.channel_wt is None:
            return self.wt.analysis(flat)
        if C > len(self.channel_wt):
            raise ValueError(
                f"channel_filters holds {len(self.channel_wt)} banks but the signal "
                f"has {C} channels; raise wast.max_channels")
        per_channel = [self.channel_wt[c].analysis(
            flat.view(B, C, S, P)[:, c].reshape(B * S, P)) for c in range(C)]
        return [torch.stack([per_channel[c][k].view(B, S, -1) for c in range(C)],
                            dim=1).reshape(B * C * S, -1)
                for k in range(len(per_channel[0]))]

    def synthesise(self, coeffs: List[torch.Tensor], B: int, C: int, S: int,
                   P: int) -> torch.Tensor:
        """Inverse of :meth:`analyse`, with the same channel bookkeeping."""
        if self.channel_wt is None:
            return self.wt.synthesis(coeffs, P)
        out = []
        for c in range(C):
            per_band = [ck.view(B, C, S, -1)[:, c].reshape(B * S, -1) for ck in coeffs]
            out.append(self.channel_wt[c].synthesis(per_band, P).view(B, S, P))
        return torch.stack(out, dim=1).reshape(B * C * S, P)

    def subbands_for(self, T: int) -> nn.ModuleList:
        """Subband processors for ``pre_patch`` placement (lengths depend on ``T``)."""
        if not hasattr(self, "_pre_patch_subbands"):
            lens = coeff_lengths(T, self.cfg.level)
            self._pre_patch_subbands = nn.ModuleList(
                SubbandProcessor(L, self.cfg.subband_width, self.cfg.subband_kernel,
                                 self.cfg.subband_dropout, self.cfg.residual_scale_init)
                for L in lens
            ).to(self.proj.weight.device)
        return self._pre_patch_subbands

    @staticmethod
    def _pre_patch_energies(coeffs, B, C, S, P) -> torch.Tensor:
        """Pool whole-signal subband energies onto the ``S`` patch slots."""
        out = []
        for c in coeffs:
            e = c.detach().pow(2).view(B, C, -1)
            e = F.adaptive_avg_pool1d(e, S)                # [B, C, S]
            out.append(e)
        return torch.stack(out, dim=-1)                    # [B, C, S, J+1]
