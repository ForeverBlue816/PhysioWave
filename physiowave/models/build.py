"""Config -> model construction, including the legacy path.

``model.name`` selects the architecture:

``legacy``      the original :class:`BERTWaveletTransformer`, untouched
``wast``        WAST tokenizer only (no TARE, no spatial branches, no compression)
``wast_tare``   the full model
``ralf``        multimodal fusion over per-modality encoders
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch.nn as nn

from ..channels.compression import CompressionConfig
from ..channels.tare import TAREConfig
from ..config import instantiate
from ..spatial.branches import SpatialConfig
from ..spatial.graph_laplacian import GLConfig
from ..spatial.spatial_stats import DynGraphConfig
from ..spatial.spline_laplacian import SSLConfig
from ..wavelet.wast import WASTConfig
from .backbone import BackboneConfig
from .encoder import EncoderConfig, PhysioWaveEncoder
from .fusion import MultimodalPhysioWave, RALFConfig
from .legacy import (LegacyForFinetuneMain, LegacyWithChannelID,
                     build_legacy_model)

logger = logging.getLogger(__name__)


def build_encoder_config(cfg: Dict[str, Any]) -> EncoderConfig:
    """Turn a ``model:`` config block into an :class:`EncoderConfig`."""
    cfg = dict(cfg or {})
    cfg.pop("name", None)
    spatial = dict(cfg.pop("spatial", {}) or {})
    ec = EncoderConfig(
        modality=cfg.pop("modality", "eeg"),
        sampling_rate=float(cfg.pop("sampling_rate", 256.0)),
        embed_dim=int(cfg.pop("embed_dim", 256)),
        num_summary_tokens=int(cfg.pop("num_summary_tokens", 4)),
        wast=instantiate(WASTConfig, cfg.pop("wast", {})),
        tare=instantiate(TAREConfig, cfg.pop("tare", {})),
        compression=instantiate(CompressionConfig, cfg.pop("compression", {})),
        spatial=SpatialConfig(
            use_raw=spatial.pop("use_raw", True),
            ssl=instantiate(SSLConfig, spatial.pop("ssl", {})),
            gl=instantiate(GLConfig, spatial.pop("gl", {})),
            dyn=instantiate(DynGraphConfig, spatial.pop("dyn", {})),
            **{k: v for k, v in spatial.items()},
        ),
        backbone=instantiate(BackboneConfig, cfg.pop("backbone", {})),
        use_spatial_frontend=cfg.pop("use_spatial_frontend", True),
        use_tare=cfg.pop("use_tare", True),
        use_compression=cfg.pop("use_compression", True),
        channel_embedding=cfg.pop("channel_embedding", "tare"),
        channel_reduction=cfg.pop("channel_reduction", "none"),
        max_channels=int(cfg.pop("max_channels", 256)),
        mask_ratio=float(cfg.pop("mask_ratio", 0.5)),
        masking_strategy=cfg.pop("masking_strategy", "frequency_guided"),
        importance_ratio=float(cfg.pop("importance_ratio", 0.6)),
        num_classes=cfg.pop("num_classes", None),
        pooling=cfg.pop("pooling", "mean"),
        head_dropout=float(cfg.pop("head_dropout", 0.0)),
    )
    if cfg:
        raise ValueError(f"Unknown model config keys: {sorted(cfg)}")
    return ec


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate whatever ``cfg['model']['name']`` asks for."""
    model_cfg = dict(cfg.get("model", {}) or {})
    name = model_cfg.get("name", "wast_tare")

    if name in ("eeg_c1", "ecg_c1", "emg_c1"):
        # The C1 pretrained encoder with a classification head. Its parameters
        # live under model.eeg_c1 rather than at the top of model, because the
        # keys a downstream montage needs -- window_samples, sampling_rate,
        # patch_samples -- are facts about the DATA and share no names with the
        # other backbones' architecture blocks.
        if name == "ecg_c1":
            # The same model with the ECG routes, lead vocabulary and
            # wavelets (physiowave/ecg_c1/downstream.py).
            from physiowave.ecg_c1.downstream import ECGC1Downstream as EEGC1Downstream
        elif name == "emg_c1":
            from physiowave.emg_c1.downstream import EMGC1Downstream as EEGC1Downstream
        else:
            from physiowave.eeg_c1.downstream import EEGC1Downstream

        params = dict(model_cfg.get(name, {}) or {})
        if model_cfg.get("num_classes") is not None:
            params["num_classes"] = model_cfg["num_classes"]
        for k in ("embed_dim", "depth", "num_heads", "mlp_ratio", "dropout",
                  "norm", "ffn", "qk_norm", "channel_encoding",
                  "channel_embed_dim", "max_level", "wave_kernel_size",
                  "wavelet_names", "wave_init_mode", "use_separate_channel",
                  "fold_synthesis", "fold_gamma"):
            if k in model_cfg and k not in params:
                params[k] = model_cfg[k]
        pretrained = params.pop("pretrained", None)
        allow_missing_gate = bool(params.pop("allow_missing_gate", False))
        report_to = params.pop("_report", None)
        logger.info("Building the %s C1 downstream model with %s",
                    name[:3].upper(), params)
        model = EEGC1Downstream(**params)
        sf = getattr(model, "spatial_filter", None)
        if sf is not None:
            # Worth its own line: with `mix` the montage the model sees is not
            # the montage in the file, and the log above prints the file's.
            logger.info("adaptive spatial filter: %s, %d -> %d channel(s)%s",
                        sf.kind, sf.in_channels, sf.out_channels,
                        f" {sf.out_names}" if sf.out_names else "")
        if pretrained:
            report = model.load_pretrained(
                pretrained, allow_missing_gate=allow_missing_gate)
            logger.info("%s <- %s", model.describe_transfer(report), pretrained)
            if report_to is not None:
                report_to.update(report)
        else:
            # The control, and it has to be sayable out loud: a downstream
            # number with nothing to compare it against says only that the
            # architecture works.
            logger.info("%s C1 downstream: NO pretrained weights "
                        "(the from-scratch control)", name[:3].upper())
        return model

    if name in ("legacy", "legacy_channel_id"):
        params = dict(model_cfg.get("legacy", {}) or {})
        patch = params.get("patch_size", (1, 20))
        if isinstance(patch, (int, float)):
            params["patch_size"] = (1, int(patch))
        elif isinstance(patch, list):
            params["patch_size"] = tuple(patch)
        if name == "legacy_channel_id":
            logger.info("Building legacy + channel-ID with %s", params)
            return LegacyWithChannelID(**params)
        if model_cfg.get("num_classes") is not None:
            # Driven by finetune_main, which passes ChannelMeta and reads a dict.
            params.setdefault("task_type", "classification")
            params["num_classes"] = model_cfg["num_classes"]
            logger.info("Building the legacy PhysioWave model for finetune_main with %s",
                        params)
            return LegacyForFinetuneMain(**params)
        logger.info("Building the legacy PhysioWave model with %s", params)
        return build_legacy_model(**params)

    if name == "wast":
        # Tokenizer-only ablation: no channel metadata, no spatial branches, no
        # compression -- isolates the effect of the wavelet tokenizer alone.
        model_cfg = deep_set(model_cfg, use_tare=False, use_spatial_frontend=False,
                             use_compression=False)
    elif name not in ("wast_tare", "ralf", "concat_fusion"):
        raise ValueError(f"Unknown model name {name!r}")

    if name in ("ralf", "concat_fusion"):
        encoders = {}
        for modality, sub in (model_cfg.get("encoders", {}) or {}).items():
            sub = dict(sub or {})
            sub["modality"] = modality
            encoders[modality] = PhysioWaveEncoder(build_encoder_config(sub))
        ralf = instantiate(RALFConfig, model_cfg.get("ralf", {}))
        if name == "concat_fusion":
            from .fusion import ConcatFusionBaseline

            return MultimodalPhysioWave(encoders, ralf, fusion_cls=ConcatFusionBaseline)
        return MultimodalPhysioWave(encoders, ralf)

    return PhysioWaveEncoder(build_encoder_config(model_cfg))


def deep_set(cfg: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    out = dict(cfg)
    out.update(kwargs)
    return out


def load_pretrained_encoders(
    model: MultimodalPhysioWave,
    paths: Dict[str, str],
    strict: bool = False,
) -> Dict[str, Any]:
    """Load an independent pretrained checkpoint into each modality encoder."""
    from .checkpoint import load_checkpoint

    reports = {}
    for modality, path in paths.items():
        if not path:
            continue
        if modality not in model.encoders:
            raise KeyError(f"No encoder for modality {modality!r}")
        logger.info("Loading pretrained %s encoder from %s", modality, path)
        payload = load_checkpoint(path, model.encoders[modality], strict=strict)
        reports[modality] = payload.get("metrics", {})
    return reports


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
