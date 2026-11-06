"""Unified per-modality encoder: EEG, ECG and limb sEMG.

The three modalities share the *macro interface* (WAST tokenizer -> channel
metadata encoder -> topology-aware compression -> factorized backbone -> summary
tokens) but **do not share backbone parameters**: their spectral content, channel
counts and spatial semantics are too different for a shared trunk to be honest.
Each is pretrained separately and later combined by RALF.

Per-modality specialisation
---------------------------
EEG
    Full TARE with both spatial branches (SSL + GL) and the ``A_geo``/``A_dyn``
    relation graph.
ECG
    Lead identity, lead/derivation metadata and a missing-lead mask.  The SSL
    branch is **explicitly disabled**: a spherical-spline surface Laplacian is
    defined on potentials sampled over a sphere approximating the scalp, and the
    12-lead system has no such geometry (its leads are already a fixed linear
    derivation of a small number of body-surface electrodes).  Forcing a spline
    fit on chest-lead positions would produce a meaningless operator.
limb sEMG
    Wavelet tokenisation and token reduction only: ``channel_embedding:
    channel_id`` and no spatial frontend (see ``configs/pretrain/semg.yaml``).
    Both spatial branches need electrode coordinates and a forearm ring has
    none -- SSL because an array on a limb is not a sphere, GL because with no
    coordinates its geometric graph is uniform and the branch degenerates into a
    common-average reference.  TARE is off for the same reason: with neither a
    coordinate nor a label it recognises it returns one vector for every
    channel, and the encoder loses the ability to tell electrodes apart at all.
    A free vector per channel index claims no geometry and keeps the identity.
    **Region check:** this encoder is for *limb / skeletal* sEMG.  Facial EMG has
    different generators, different bandwidth and different artefact structure,
    and must not be mixed into the limb sEMG pretraining corpus; the dataset
    registry carries an ``emg_region`` field and
    :func:`physiowave.data.registry.assert_limb_semg` enforces it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..channels.compression import ChannelCompressor, CompressionConfig
from ..channels.tare import ChannelEncoder, ChannelMeta, TAREConfig
from ..spatial.branches import SpatialConfig, SpatialFrontend
from ..wavelet.fgm import frequency_guided_mask, random_patch_mask
from ..wavelet.wast import WAST, WASTConfig
from .backbone import BackboneConfig, FactorizedBackbone

logger = logging.getLogger(__name__)

MODALITIES = ("eeg", "ecg", "semg")

#: How the per-channel embedding the compressor is conditioned on is produced.
#:
#: ``tare``        the metadata encoder: coordinates, labels, reference scheme.
#: ``channel_id``  a free vector per channel *index*, no metadata at all.  The
#:                 weakest thing that still tells the model which electrode is
#:                 which, and the right choice for an array whose geometry is not
#:                 in the montage tables -- a forearm sEMG ring, say, where TARE
#:                 has neither a coordinate nor a label it recognises and returns
#:                 the same vector for every channel.  Indexed by position, so it
#:                 is montage-specific by construction: it cannot transfer to a
#:                 permuted or differently sized array.
#: ``none``        no channel embedding.  Note that this leaves the encoder
#:                 *invariant* to permuting the channel axis, because WAST shares
#:                 its parameters across channels and the compressor then has
#:                 nothing left that distinguishes one channel from another.
CHANNEL_EMBEDDINGS = ("tare", "channel_id", "none")


@dataclass
class EncoderConfig:
    """Configuration of a single-modality :class:`PhysioWaveEncoder`."""

    modality: str = "eeg"
    sampling_rate: float = 256.0
    embed_dim: int = 256
    num_summary_tokens: int = 4
    wast: WASTConfig = field(default_factory=WASTConfig)
    tare: TAREConfig = field(default_factory=TAREConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    use_spatial_frontend: bool = True
    use_tare: bool = True
    use_compression: bool = True
    channel_embedding: str = "tare"     # 'tare' | 'channel_id' | 'none'
    max_channels: int = 256             # table size for 'channel_id'
    #: What happens to the channel axis when ``use_compression`` is False.
    #:
    #: ``none``  keep it: the backbone's slot axis becomes the channel axis and
    #:           the lattice stays ``C * S``. The backbone already attends along
    #:           that axis, so nothing needs compressing for it to work.
    #: ``mean``  average the channels into a single slot. This is the older
    #:           fallback and it throws the entire channel–time lattice away for
    #:           ``1 * S`` tokens; it also makes the encoder permutation
    #:           invariant. Kept only so the earlier ablation stays runnable.
    channel_reduction: str = "none"     # 'none' | 'mean'
    mask_ratio: float = 0.5
    masking_strategy: str = "frequency_guided"      # 'frequency_guided' | 'random'
    importance_ratio: float = 0.6
    num_classes: Optional[int] = None
    pooling: str = "mean"
    #: How ``[B, K, S, D]`` becomes ``[B, D]`` for the head.
    #:
    #: ``mean``          mean of the summary tokens, which ``num_summary_tokens``
    #:                   learned queries produce by cross-attending over the whole
    #:                   lattice. Expressive, and montage-independent, which is
    #:                   what the multimodal path needs.
    #: ``first``         the first summary token.
    #: ``lattice_mean``  plain mean over every channel and patch, bypassing the
    #:                   summary attention entirely. This is what the legacy model
    #:                   does, and on a task whose class is a roughly stationary
    #:                   amplitude pattern across channels it is a much stronger
    #:                   prior: averaging over time cannot latch onto the
    #:                   idiosyncrasies of the particular repetitions in the
    #:                   training split, and a learned query can. Summary tokens
    #:                   are still produced for downstream consumers.
    #: Dropout before the classification layer. The legacy fine-tuning path has
    #: had this since the beginning (``--head_dropout``, 0.1 by default) and this
    #: encoder had no equivalent, so a downstream run here was training a bare
    #: LayerNorm+Linear on top of pooled features with no regularisation at all.
    head_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        # Keep the sub-configs dimensionally consistent with the encoder.
        self.wast.embed_dim = self.embed_dim
        self.tare.embed_dim = self.embed_dim
        self.compression.embed_dim = self.embed_dim
        self.backbone.embed_dim = self.embed_dim
        if self.pooling not in ("mean", "first", "lattice_mean"):
            raise ValueError(
                f"pooling must be 'mean', 'first' or 'lattice_mean', got {self.pooling!r}")
        if self.channel_reduction not in ("none", "mean"):
            raise ValueError(
                f"channel_reduction must be 'none' or 'mean', got {self.channel_reduction!r}")
        if self.channel_embedding not in CHANNEL_EMBEDDINGS:
            raise ValueError(
                f"channel_embedding must be one of {CHANNEL_EMBEDDINGS}, "
                f"got {self.channel_embedding!r}")
        # ``use_tare: false`` is the older spelling of the tokenizer-only ablation
        # and keeps meaning "no channel embedding at all".
        if not self.use_tare and self.channel_embedding == "tare":
            self.channel_embedding = "none"
        if self.channel_embedding != "tare":
            self.use_tare = False
        if self.modality in ("ecg", "semg"):
            # See the module docstring: no spherical scalp geometry exists here.
            self.spatial.ssl.enabled = False


class PhysioWaveEncoder(nn.Module):
    """Single-modality encoder.

    Forward returns a dict with
        ``summary_tokens``   ``[B, n_summary, D]`` -- fixed count, montage independent
        ``pooled``           ``[B, D]``
        ``logits``           ``[B, num_classes]`` when a head is configured
        ``quality``          ``[B]`` in ``[0, 1]`` -- self-estimated signal quality
        ``tokens``           ``[B, K, S, D]``
        ``token_stats``      dict with the legacy / WAST / compressed token counts
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wast = WAST(cfg.wast)
        self.tare = ChannelEncoder(cfg.tare) if cfg.channel_embedding == "tare" else None
        if cfg.channel_embedding == "channel_id":
            self.channel_id = nn.Embedding(cfg.max_channels, cfg.embed_dim)
            nn.init.normal_(self.channel_id.weight, std=0.02)
            # TARE ends in a LayerNorm, and the compressor's chan_proj is scaled
            # for what that delivers. Without the same normalisation here a
            # std=0.02 table reaches the attention logits ~50x weaker than the
            # branch it replaces, and starts far enough below the token
            # magnitudes that channel identity is nearly invisible at init.
            self.channel_id_norm = nn.LayerNorm(cfg.embed_dim)
        else:
            self.channel_id = None
        self.spatial = SpatialFrontend(cfg.spatial) if cfg.use_spatial_frontend else None
        self.compressor = ChannelCompressor(cfg.compression) if cfg.use_compression else None
        self.backbone = FactorizedBackbone(cfg.backbone)

        D = cfg.embed_dim
        self.summary_query = nn.Parameter(torch.randn(cfg.num_summary_tokens, D) * 0.02)
        self.summary_attn = nn.MultiheadAttention(D, cfg.backbone.slot_heads, batch_first=True)
        self.summary_norm = nn.LayerNorm(D)
        self.quality_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(),
                                          nn.Linear(D // 2, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Dropout(cfg.head_dropout) if cfg.head_dropout > 0 else nn.Identity(),
            nn.Linear(D, cfg.num_classes),
        ) if cfg.num_classes else None
        # Reconstruction heads for pretraining.
        P = cfg.wast.patch_size
        self.recon_raw = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, P))
        self.recon_wave = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, P))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)
        # Uncompressed fallback used when compression is disabled (legacy-like path).
        self.channel_pool = nn.Linear(D, D)

    # -- helpers ---------------------------------------------------------------
    def default_meta(self, C: int, device) -> ChannelMeta:
        """Metadata for callers that pass a bare tensor (ECG/sEMG convenience)."""
        return ChannelMeta(
            channel_names=[f"{self.cfg.modality.upper()}{i}" for i in range(C)],
            channel_xyz=torch.zeros(C, 3, device=device),
            montage_type="unknown",
            reference_type="unknown",
            derivation_type="monopolar",
        )

    def token_stats(self, C: int, T: int) -> Dict[str, float]:
        kv_stride = 2 ** self.cfg.backbone.kv_wavelet_level
        if self.compressor is not None:
            return self.wast.token_report(C, T, self.cfg.compression.num_queries, kv_stride)
        if self.cfg.channel_reduction == "mean":
            return self.wast.token_report(C, T, 1, kv_stride)
        # No channel compression: the lattice survives, so there is no K to report.
        return self.wast.token_report(C, T, None, kv_stride)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        meta: Optional[ChannelMeta] = None,
        mask_patches: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, object]:
        assert x.dim() == 3, f"encoder expects [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        meta = meta if meta is not None else self.default_meta(C, x.device)
        assert meta.num_channels() == C, (
            f"metadata describes {meta.num_channels()} channels but the signal has {C}"
        )

        info: Dict[str, object] = {}
        signal, A, ssl_signal = x, None, None
        if self.spatial is not None:
            sp = self.spatial(x, meta, self.cfg.sampling_rate)
            signal, A, ssl_signal = sp["signal"], sp["A"], sp["ssl_signal"]
            info.update(sp["info"])

        chan_mask = None if meta.channel_mask is None else meta.channel_mask.to(x.device)
        wast_out = self.wast(signal, chan_mask)
        tokens = wast_out["tokens"]                       # [B, C, S, D]
        S = tokens.shape[2]

        if self.tare is not None:
            chan_emb = self.tare(meta, device=x.device)   # [C, D]
        elif self.channel_id is not None:
            if C > self.channel_id.num_embeddings:
                raise ValueError(
                    f"channel_id embedding holds {self.channel_id.num_embeddings} rows "
                    f"but the signal has {C} channels; raise model.max_channels")
            chan_emb = self.channel_id_norm(self.channel_id(torch.arange(C, device=x.device)))
        else:
            chan_emb = torch.zeros(C, self.cfg.embed_dim, device=x.device, dtype=tokens.dtype)

        xyz = meta.channel_xyz.to(device=x.device, dtype=tokens.dtype)
        if self.compressor is not None:
            comp = self.compressor(tokens, chan_emb, xyz, chan_mask, A)
            latent, attn = comp["tokens"], comp["attn"]   # [B, K, S, D]
        elif self.cfg.channel_reduction == "none":
            # Keep the full channel-time lattice. The backbone's slot axis is the
            # channel axis here, and its slot embedding is what tells one channel
            # from another -- so this path needs no channel embedding of its own.
            if chan_mask is not None:
                tokens = tokens * chan_mask.to(tokens.dtype).view(1, C, 1, 1)
            latent, attn = tokens, None
        else:
            # Average over channels so the interface still yields [B, K, S, D]
            # with K = 1. This discards the channel axis entirely; see the note
            # on EncoderConfig.channel_reduction.
            m = torch.ones(C, device=x.device, dtype=tokens.dtype) if chan_mask is None \
                else chan_mask.to(tokens.dtype)
            latent = (tokens * m.view(1, C, 1, 1)).sum(1, keepdim=True) / m.sum().clamp_min(1.0)
            latent = self.channel_pool(latent)
            attn = None

        if mask_patches is not None:
            assert mask_patches.shape == (B, S), (
                f"mask must be [B, S] = {(B, S)}, got {tuple(mask_patches.shape)}"
            )
            latent = torch.where(mask_patches.view(B, 1, S, 1),
                                 self.mask_token.to(latent.dtype).expand_as(latent), latent)

        feats = self.backbone(latent)                     # [B, K, S, D]
        Bk, K, Sk, D = feats.shape
        flat = feats.reshape(B, K * S, D)
        q = self.summary_query.unsqueeze(0).expand(B, -1, -1)
        summary, _ = self.summary_attn(q, flat, flat)
        summary = self.summary_norm(summary)              # [B, n_summary, D]
        if self.cfg.pooling == "lattice_mean":
            pooled = feats.mean(dim=(1, 2))               # [B, D]
        elif self.cfg.pooling == "mean":
            pooled = summary.mean(dim=1)
        else:
            pooled = summary[:, 0]

        out: Dict[str, object] = {
            "summary_tokens": summary,
            "pooled": pooled,
            "tokens": feats,
            "quality": torch.sigmoid(self.quality_head(pooled)).squeeze(-1),
            "patch_scores": wast_out["patch_scores"],
            "raw_patches": wast_out["raw_patches"],
            "raw_coeffs": wast_out["raw_coeffs"],
            "num_patches": S,
            "token_stats": self.token_stats(C, T),
            "spatial_info": info,
            "attn": attn,
            "ssl_signal": ssl_signal,
            "A": A,
        }
        if self.head is not None:
            out["logits"] = self.head(pooled)
        if return_all:
            out["channel_embedding"] = chan_emb
            out["channel_tokens"] = tokens
        return out

    # -- pretraining helpers ---------------------------------------------------
    def sample_mask(self, scores: torch.Tensor, generator=None) -> torch.Tensor:
        """Patch mask ``[B, S]`` according to the configured masking strategy."""
        if self.cfg.masking_strategy == "frequency_guided":
            return frequency_guided_mask(scores, self.cfg.mask_ratio,
                                         self.cfg.importance_ratio, generator)
        return random_patch_mask(scores.shape[0], scores.shape[1], self.cfg.mask_ratio,
                                 scores.device, generator)

    def reconstruct(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-patch reconstructions from ``[B, K, S, D]`` latent features."""
        pooled = feats.mean(dim=1)                        # [B, S, D]
        return {"raw": self.recon_raw(pooled), "wave": self.recon_wave(pooled)}
