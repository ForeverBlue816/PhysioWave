"""RALF -- Reliability-Aware Latent Fusion of EEG, ECG and limb sEMG.

Design
------
Each modality encoder is pretrained independently and contributes only its
**summary tokens** (4-8 per modality), not its full token grid.  Fusion is a
bottleneck cross-attention: a small set of learnable fusion tokens attends over
the concatenated summary tokens.  Concatenating full token sequences would make
the fusion cost scale with every modality's window length and would let the
modality with the most tokens dominate by sheer count.

Reliability
-----------
Every modality gets a scalar reliability score in ``[0, 1]``, predicted from its
own summary tokens.  It is used twice: as a multiplicative gate on that
modality's contribution to the fusion attention, and as the weight of its
single-modality residual logit.  During training the score is supervised against
the *known* corruption level applied by :class:`SignalCorruptor`, which is the
only place a ground-truth reliability exists.

Missing modalities
------------------
Any subset of modalities may be present, and the subset may differ per batch
item.  Absent modalities are masked out of the attention entirely (not zero-filled,
which would let a zero vector act as evidence).  An all-absent item is a hard
error rather than a silent zero prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_ORDER = ("eeg", "ecg", "semg")


@dataclass
class RALFConfig:
    """Configuration for :class:`ReliabilityAwareLatentFusion`."""

    modalities: Sequence[str] = field(default_factory=lambda: list(MODALITY_ORDER))
    embed_dim: int = 256
    num_fusion_tokens: int = 8
    num_heads: int = 8
    depth: int = 2
    dropout: float = 0.1
    num_classes: int = 5
    modality_dropout: float = 0.2
    residual_weight: float = 0.5
    consistency_weight: float = 1.0
    reliability_weight: float = 1.0


class ReliabilityAwareLatentFusion(nn.Module):
    """Bottleneck cross-attention fusion with per-modality reliability weighting."""

    def __init__(self, cfg: RALFConfig, encoder_dims: Optional[Dict[str, int]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.modalities = list(cfg.modalities)
        D = cfg.embed_dim
        dims = encoder_dims or {m: D for m in self.modalities}

        # Modality-specific projection aligns heterogeneous encoder widths, and the
        # modality token tells the fusion which stream a summary token came from.
        self.proj = nn.ModuleDict({m: nn.Linear(dims.get(m, D), D) for m in self.modalities})
        self.modality_token = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(1, 1, D) * 0.02) for m in self.modalities}
        )
        self.reliability_head = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(),
                             nn.Linear(D // 2, 1))
            for m in self.modalities
        })
        self.unimodal_head = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, cfg.num_classes))
            for m in self.modalities
        })

        self.fusion_tokens = nn.Parameter(torch.randn(cfg.num_fusion_tokens, D) * 0.02)
        self.cross = nn.ModuleList(
            nn.MultiheadAttention(D, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(cfg.depth)
        )
        self.cross_norm = nn.ModuleList(nn.LayerNorm(D) for _ in range(cfg.depth))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(),
                          nn.Dropout(cfg.dropout), nn.Linear(4 * D, D))
            for _ in range(cfg.depth)
        )
        self.out_norm = nn.LayerNorm(D)
        self.fusion_head = nn.Linear(D, cfg.num_classes)

    # -- helpers ---------------------------------------------------------------
    def _present(self, features: Dict[str, torch.Tensor],
                 modality_mask: Optional[Dict[str, torch.Tensor]],
                 B: int, device) -> Dict[str, torch.Tensor]:
        """Per-modality ``[B]`` float presence, combining absence and batch masks."""
        out = {}
        for m in self.modalities:
            if m not in features or features[m] is None:
                out[m] = torch.zeros(B, device=device)
                continue
            if modality_mask is not None and m in modality_mask and modality_mask[m] is not None:
                out[m] = modality_mask[m].to(device=device, dtype=torch.float32).view(B)
            else:
                out[m] = torch.ones(B, device=device)
        return out

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        """Fuse per-modality summary tokens.

        Args:
            features: ``{modality: [B, n_tokens_m, D_m]}``.  ``n_tokens_m`` may
                differ across modalities; the interface does not care.
            modality_mask: ``{modality: [B] bool}``, False = that item lacks that
                modality.

        Returns:
            ``logits``, ``fusion_logits``, ``unimodal_logits``, ``reliability``,
            ``present``, ``fused`` (the pooled fusion embedding).
        """
        assert features, "RALF received no modalities at all"
        any_feat = next(v for v in features.values() if v is not None)
        B, device = any_feat.shape[0], any_feat.device
        present = self._present(features, modality_mask, B, device)

        total = torch.stack([present[m] for m in self.modalities], dim=0).sum(0)
        if bool((total <= 0).any()):
            idx = torch.nonzero(total <= 0).flatten().tolist()
            raise ValueError(
                f"RALF received no available modality for batch items {idx}. "
                "At least one modality must be present per sample."
            )

        seq, key_mask, reliability, uni_logits = [], [], {}, {}
        for m in self.modalities:
            f = features.get(m)
            if f is None:
                continue
            h = self.proj[m](f) + self.modality_token[m]          # [B, n_m, D]
            pooled = h.mean(dim=1)
            r = torch.sigmoid(self.reliability_head[m](pooled)).squeeze(-1)  # [B]
            r = r * present[m]
            reliability[m] = r
            uni_logits[m] = self.unimodal_head[m](pooled)
            # Reliability gates the contribution; presence hard-masks it.
            seq.append(h * r.view(B, 1, 1))
            key_mask.append(present[m].view(B, 1).expand(B, h.shape[1]) > 0.5)

        mem = torch.cat(seq, dim=1)                               # [B, sum_n, D]
        mem_mask = torch.cat(key_mask, dim=1)                     # [B, sum_n] True = keep

        x = self.fusion_tokens.unsqueeze(0).expand(B, -1, -1)
        for attn, norm, ffn in zip(self.cross, self.cross_norm, self.ffn, strict=True):
            a, _ = attn(norm(x), mem, mem, key_padding_mask=~mem_mask)
            x = x + a
            x = x + ffn(x)
        fused = self.out_norm(x).mean(dim=1)                      # [B, D]

        fusion_logits = self.fusion_head(fused)
        logits = fusion_logits
        denom = torch.zeros(B, device=device)
        residual = torch.zeros_like(fusion_logits)
        for m, lg in uni_logits.items():
            w = reliability[m] * present[m]
            residual = residual + w.view(B, 1) * lg
            denom = denom + w
        logits = logits + self.cfg.residual_weight * residual / denom.clamp_min(1e-6).view(B, 1)

        return {
            "logits": logits,
            "fusion_logits": fusion_logits,
            "unimodal_logits": uni_logits,
            "reliability": reliability,
            "present": present,
            "fused": fused,
        }

    # -- losses ----------------------------------------------------------------
    def reliability_loss(
        self, reliability: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        present: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """MSE between predicted reliability and ``1 - corruption_level``."""
        losses = []
        for m, r in reliability.items():
            if m not in target or target[m] is None:
                continue
            w = present[m]
            if w.sum() <= 0:
                continue
            losses.append((((r - target[m].to(r.dtype)) ** 2) * w).sum() / w.sum())
        if not losses:
            return next(iter(reliability.values())).new_zeros(())
        return torch.stack(losses).mean()

    @staticmethod
    def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        """Symmetric KL between two views' predictive distributions.

        Used to tie the full-modality prediction to the degraded/subset ones, so
        dropping or corrupting a modality moves the prediction as little as the
        remaining evidence allows.
        """
        pa, pb = F.log_softmax(logits_a, -1), F.log_softmax(logits_b, -1)
        return 0.5 * (F.kl_div(pb, pa, log_target=True, reduction="batchmean")
                      + F.kl_div(pa, pb, log_target=True, reduction="batchmean"))


class ConcatFusionBaseline(nn.Module):
    """Baseline fusion: mean-pool each modality, concatenate, classify.

    This is the "original multimodal fusion" arm of the tier-3 comparison against
    RALF.  It deliberately has neither reliability weighting nor bottleneck
    cross-attention: a missing modality is zero-filled, which is exactly the
    failure mode RALF exists to avoid -- a zero vector is indistinguishable from a
    genuine all-quiet measurement, so the classifier learns to treat absence as
    evidence.  It shares RALF's interface so the two are drop-in comparable.
    """

    def __init__(self, cfg: RALFConfig, encoder_dims: Optional[Dict[str, int]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.modalities = list(cfg.modalities)
        D = cfg.embed_dim
        dims = encoder_dims or {m: D for m in self.modalities}
        self.proj = nn.ModuleDict({m: nn.Linear(dims.get(m, D), D) for m in self.modalities})
        self.head = nn.Sequential(
            nn.LayerNorm(D * len(self.modalities)),
            nn.Linear(D * len(self.modalities), D), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(D, cfg.num_classes),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        any_feat = next(v for v in features.values() if v is not None)
        B, device = any_feat.shape[0], any_feat.device
        parts, present = [], {}
        for m in self.modalities:
            f = features.get(m)
            if f is None:
                parts.append(torch.zeros(B, self.cfg.embed_dim, device=device))
                present[m] = torch.zeros(B, device=device)
                continue
            p = torch.ones(B, device=device) if modality_mask is None or m not in modality_mask \
                else modality_mask[m].to(device=device, dtype=torch.float32).view(B)
            present[m] = p
            parts.append(self.proj[m](f).mean(dim=1) * p.view(B, 1))
        logits = self.head(torch.cat(parts, dim=-1))
        return {"logits": logits, "fusion_logits": logits, "unimodal_logits": {},
                "reliability": {m: present[m] for m in present}, "present": present,
                "fused": torch.cat(parts, dim=-1)}

    def reliability_loss(self, *args, **kwargs) -> torch.Tensor:
        """No reliability model to supervise; returns zero so training code is shared."""
        return next(self.parameters()).new_zeros(())

    @staticmethod
    def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        return ReliabilityAwareLatentFusion.consistency_loss(logits_a, logits_b)


class MultimodalPhysioWave(nn.Module):
    """Container tying per-modality encoders to :class:`ReliabilityAwareLatentFusion`.

    Encoders can be loaded from independent pretrained checkpoints; see
    :func:`physiowave.models.build.load_pretrained_encoders`.
    """

    def __init__(self, encoders: Dict[str, nn.Module], cfg: RALFConfig,
                 fusion_cls: type = ReliabilityAwareLatentFusion) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        dims = {m: e.cfg.embed_dim for m, e in encoders.items()}
        cfg.modalities = list(encoders.keys())
        self.fusion = fusion_cls(cfg, dims)
        self.cfg = cfg

    def encode(
        self,
        inputs: Dict[str, torch.Tensor],
        metas: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        for m, enc in self.encoders.items():
            x = inputs.get(m)
            if x is None:
                continue
            meta = (metas or {}).get(m)
            feats[m] = enc(x, meta)["summary_tokens"]
        return feats

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        metas: Optional[Dict[str, object]] = None,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        return self.fusion(self.encode(inputs, metas), modality_mask)
