"""The combined self-supervised objective for a single-modality encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..channels.tare import ChannelMeta
from .losses import (
    band_covariance_loss,
    masked_patch_loss,
    reference_consistency_loss,
    wavelet_coefficient_loss,
)
from .reference import ReferenceConfig, build_views

logger = logging.getLogger(__name__)

ANCHOR_MODES = ("ssl", "pairwise")


@dataclass
class RefConsistencyConfig:
    """Configuration of the reference-consistency term."""

    enabled: bool = True
    anchor: str = "ssl"              # 'ssl' | 'pairwise'
    metric: str = "cosine"
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)

    def __post_init__(self) -> None:
        if self.anchor not in ANCHOR_MODES:
            raise ValueError(f"anchor must be one of {ANCHOR_MODES}")


@dataclass
class PretrainObjectiveConfig:
    """Weights and switches for every loss term."""

    use_masked_raw: bool = True
    use_wavelet: bool = True
    use_reference_consistency: bool = True
    use_query_specialization: bool = True
    use_covariance: bool = True

    lambda_wave: float = 0.5
    lambda_ref: float = 0.5
    lambda_spec: float = 0.05
    lambda_cov: float = 0.1

    ref_consistency: RefConsistencyConfig = field(default_factory=RefConsistencyConfig)


class PretrainObjective(nn.Module):
    """Computes the combined loss and a per-term log dict.

    The reference-consistency anchor follows the physics: the SSL view is
    reference invariant (``L_ssl 1 == 0``), so every re-referenced view is pulled
    towards it.  When the montage is too sparse for a trustworthy spline Laplacian
    the SSL branch reports itself unavailable and this module **automatically
    falls back** to pairwise alignment, recording ``ref_anchor_fallback`` in the
    log so the change is visible in the run history rather than hidden.
    """

    def __init__(self, cfg: PretrainObjectiveConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fallback_count = 0

    def forward(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        meta: ChannelMeta,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, object]:
        """Run the encoder on all views and return ``{'loss', 'logs', ...}``."""
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        B, C, T = x.shape
        logs: Dict[str, float] = {}
        device = x.device

        # --- primary view: masked reconstruction ------------------------------
        base = encoder(x, meta)
        S = base["num_patches"]
        mask = encoder.sample_mask(base["patch_scores"], generator)
        masked_out = encoder(x, meta, mask_patches=mask)
        recon = encoder.reconstruct(masked_out["tokens"])

        # Target patches are the channel-mean patch (the encoder is compressed over
        # channels, so a per-channel target would be unidentifiable).
        raw_patches = base["raw_patches"]                     # [B, C, S, P]
        cm = meta.channel_mask
        if cm is not None:
            w = cm.to(device=device, dtype=raw_patches.dtype).view(1, C, 1, 1)
            target_raw = (raw_patches * w).sum(1) / w.sum().clamp_min(1.0)
        else:
            target_raw = raw_patches.mean(dim=1)              # [B, S, P]

        total = x.new_zeros(())
        if self.cfg.use_masked_raw:
            l_raw = masked_patch_loss(recon["raw"], target_raw.detach(), mask)
            total = total + l_raw
            logs["loss_masked_raw"] = float(l_raw.item())

        if self.cfg.use_wavelet:
            l_wave = wavelet_coefficient_loss(
                recon["wave"].unsqueeze(1).expand(-1, 1, -1, -1).reshape(B, 1, S, -1),
                [c.view(B, C, S, -1).mean(1).reshape(B * S, -1) for c in base["raw_coeffs"]],
                encoder.wast.wt, mask, B, 1, S,
            )
            total = total + self.cfg.lambda_wave * l_wave
            logs["loss_wavelet"] = float(l_wave.item())

        # --- reference consistency --------------------------------------------
        used_anchor = "none"
        if self.cfg.use_reference_consistency and self.cfg.ref_consistency.enabled:
            views = build_views(x, meta, self.cfg.ref_consistency.reference, generator=generator)
            embs, tiers = [], []
            for v in views:
                out_v = encoder(v["signal"], v["meta"])
                embs.append(out_v["pooled"])
                tiers.append(v["tier"])
            anchor = None
            if self.cfg.ref_consistency.anchor == "ssl":
                ssl_sig = base.get("ssl_signal")
                if ssl_sig is not None:
                    # The SSL view is reference free; encode it as its own view.
                    ssl_meta = ChannelMeta(
                        channel_names=meta.channel_names, channel_xyz=meta.channel_xyz,
                        channel_mask=meta.channel_mask, channel_quality=meta.channel_quality,
                        montage_type=meta.montage_type, reference_type="unknown",
                        derivation_type="csd",
                    )
                    anchor = encoder(ssl_sig, ssl_meta)["pooled"]
                    used_anchor = "ssl"
                else:
                    self.fallback_count += 1
                    used_anchor = "pairwise_fallback"
                    logs["ref_anchor_fallback"] = 1.0
                    if self.fallback_count == 1:
                        logger.warning(
                            "reference-consistency anchor 'ssl' unavailable for this "
                            "montage (SSL branch skipped); falling back to pairwise "
                            "alignment. Reason: %s", base["spatial_info"].get("ssl_skip_reason"),
                        )
            else:
                used_anchor = "pairwise"
            l_ref = reference_consistency_loss(embs, anchor, self.cfg.ref_consistency.metric)
            total = total + self.cfg.lambda_ref * l_ref
            logs["loss_reference_consistency"] = float(l_ref.item())
            # Log standard and hard tiers separately: a lateralised reference is a
            # much harder target and averaging the two hides that.
            for tier in ("standard", "hard"):
                sel = [e for e, t in zip(embs, tiers, strict=True) if t == tier]
                if sel:
                    lt = reference_consistency_loss(sel, anchor, self.cfg.ref_consistency.metric)
                    logs[f"loss_ref_{tier}"] = float(lt.item())

        # --- query specialisation ---------------------------------------------
        if self.cfg.use_query_specialization and encoder.compressor is not None:
            l_spec = encoder.compressor.query_specialization_loss(base["attn"])
            total = total + self.cfg.lambda_spec * l_spec
            logs["loss_query_specialization"] = float(l_spec.item())

        # --- band-wise spatial covariance fidelity ----------------------------
        if self.cfg.use_covariance:
            P = raw_patches.shape[-1]
            rec_sig = recon["raw"].reshape(B, 1, S * P).expand(B, C, S * P)
            l_cov = band_covariance_loss(rec_sig, x[..., : S * P], encoder.cfg.sampling_rate)
            total = total + self.cfg.lambda_cov * l_cov
            logs["loss_covariance"] = float(l_cov.item())

        logs["loss_total"] = float(total.item())
        logs["mask_ratio"] = float(mask.float().mean().item())
        return {"loss": total, "logs": logs, "mask": mask,
                "ref_anchor": used_anchor, "token_stats": base["token_stats"],
                "spatial_info": base["spatial_info"]}
