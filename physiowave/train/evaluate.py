"""Evaluation suites: channel robustness, reference robustness and multimodal RALF.

Every suite writes raw JSON so the aggregation in
:mod:`physiowave.experiments.report` can build CSV / Markdown / LaTeX tables and
the performance-efficiency Pareto data without re-running anything.

Results computed on the synthetic corpus are tagged ``"synthetic": true`` in the
output so they can never be mistaken for benchmark numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..channels.tare import ChannelMeta
from ..config import load_config
from ..data.schema import batch_to_meta, collate_samples
from ..data.synthetic import SyntheticConfig, SyntheticDataset, synthetic_multimodal
from ..models.build import build_model
from ..models.checkpoint import load_checkpoint
from ..pretrain.corruption import CorruptionConfig, SignalCorruptor
from ..pretrain.reference import ReferenceConfig, build_views
from ..spatial.spline_laplacian import SSLConfig, SSLOperatorCache
from .utils import pick_device, set_seed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Representation-stability metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embedding_shift(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """How far two pooled representations of the same data drifted apart."""
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    rel = ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)).mean().item()
    return {"cosine_similarity": cos, "relative_l2": rel}


@torch.no_grad()
def run_channel_robustness(
    model, meta: ChannelMeta, x: torch.Tensor, seed: int = 0
) -> Dict[str, Dict[str, float]]:
    """Permutation, missing-channel, bad-channel and unknown-montage stability.

    Index tensors are kept on CPU for the (CPU-resident) metadata and moved to the
    signal's device for the signal itself; several accelerator backends refuse a
    cross-device index.
    """
    g = torch.Generator().manual_seed(seed)
    dev = x.device
    base = model(x, meta)["pooled"]
    C = x.shape[1]
    out: Dict[str, Dict[str, float]] = {}

    perm = torch.randperm(C, generator=g)                       # CPU
    pmeta = ChannelMeta(
        channel_names=[meta.channel_names[i] for i in perm.tolist()],
        channel_xyz=meta.channel_xyz.cpu()[perm].to(dev),
        channel_mask=None if meta.channel_mask is None else meta.channel_mask.cpu()[perm],
        montage_type=meta.montage_type, reference_type=meta.reference_type,
        derivation_type=meta.derivation_type,
    )
    out["permutation"] = embedding_shift(base, model(x[:, perm.to(dev)], pmeta)["pooled"])

    for frac in (0.1, 0.25):
        n_drop = max(1, int(round(C * frac)))
        idx = torch.randperm(C, generator=g)[:n_drop]           # CPU
        mask = torch.ones(C, dtype=torch.bool)
        mask[idx] = False
        mmeta = ChannelMeta(meta.channel_names, meta.channel_xyz, channel_mask=mask,
                            montage_type=meta.montage_type, reference_type=meta.reference_type,
                            derivation_type=meta.derivation_type)
        xm = x.clone()
        xm[:, idx.to(dev)] = 0.0
        out[f"missing_{int(frac * 100)}pct"] = embedding_shift(base, model(xm, mmeta)["pooled"])

    idx = torch.randperm(C, generator=g)[:max(1, C // 10)]
    xb = x.clone()
    noise = (torch.randn(len(idx), x.shape[-1], generator=g) * 10.0).to(dev)
    xb[:, idx.to(dev)] = noise.unsqueeze(0).expand(x.shape[0], -1, -1)
    out["bad_channel"] = embedding_shift(base, model(xb, meta)["pooled"])

    umeta = ChannelMeta(meta.channel_names, torch.zeros_like(meta.channel_xyz),
                        montage_type="unknown", reference_type=meta.reference_type,
                        derivation_type=meta.derivation_type)
    out["unknown_montage"] = embedding_shift(base, model(x, umeta)["pooled"])
    return out


@torch.no_grad()
def run_reference_robustness(model, meta: ChannelMeta, x: torch.Tensor) -> Dict[str, Any]:
    """Representation stability under every physically legal reference view.

    Hard (lateralised) views appear *here only* -- never as a training anchor and
    never as the default downstream evaluation view.
    """
    base = model(x, meta)["pooled"]
    cfg = ReferenceConfig(num_views=1)
    out: Dict[str, Any] = {"standard": {}, "hard": {}}
    for tier, names in (("standard", cfg.standard_views), ("hard", cfg.hard_views)):
        for name in names:
            views = build_views(x, meta, cfg, view_names=[name])
            if not views or views[0]["name"] != name:
                out[tier][name] = {"skipped": True}
                continue
            v = views[0]
            out[tier][name] = embedding_shift(base, model(v["signal"], v["meta"])["pooled"])

    # Offline spline-CSD preprocessed input as a further view.
    cache = SSLOperatorCache()
    L = cache.get(meta.channel_names, meta.channel_xyz.cpu(), meta.channel_mask,
                  SSLConfig(), meta.derivation_type)
    if L is not None:
        csd = torch.einsum("ij,bjt->bit", L.to(device=x.device, dtype=x.dtype), x)
        cmeta = ChannelMeta(meta.channel_names, meta.channel_xyz, meta.channel_mask,
                            montage_type=meta.montage_type, reference_type="unknown",
                            derivation_type="csd")
        out["offline_spline_csd"] = embedding_shift(base, model(csd, cmeta)["pooled"])
    else:
        out["offline_spline_csd"] = {"skipped": True, "reason": cache.stats()["skips"]}
    return out


@torch.no_grad()
def run_montage_transfer(model, from_montage: str, to_montage: str, T: int = 1024,
                         batch: int = 4, seed: int = 0) -> Dict[str, float]:
    """Encode the same synthetic sources under two montages and compare embeddings.

    A model whose spatial reasoning is genuinely geometric should map the same
    underlying activity to similar representations whether it was sampled by 19 or
    64 electrodes.
    """
    ds_a = SyntheticDataset(SyntheticConfig("eeg", batch, window_samples=T,
                                            montage_name=from_montage, seed=seed))
    ds_b = SyntheticDataset(SyntheticConfig("eeg", batch, window_samples=T,
                                            montage_name=to_montage, seed=seed))
    ba = collate_samples([ds_a[i] for i in range(batch)])
    bb = collate_samples([ds_b[i] for i in range(batch)])
    device = next(model.parameters()).device
    ma, mb = batch_to_meta(ba), batch_to_meta(bb)
    ma.channel_xyz = ma.channel_xyz.to(device)
    mb.channel_xyz = mb.channel_xyz.to(device)
    ea = model(ba["signal"].to(device), ma)["pooled"]
    eb = model(bb["signal"].to(device), mb)["pooled"]
    return embedding_shift(ea, eb)


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #
def run_eeg_suite(config: str = "pretrain/eeg", overrides: Sequence[str] = (),
                  checkpoint: Optional[str] = None, output: Optional[str] = None,
                  quick: bool = False, seed: int = 0) -> Dict[str, Any]:
    """Channel + reference robustness for one EEG model."""
    set_seed(seed)
    cfg = load_config(config, list(overrides))
    device = pick_device()
    model = build_model(cfg).to(device).eval()
    synthetic = True
    if checkpoint and os.path.exists(checkpoint):
        load_checkpoint(checkpoint, model, strict=False)
    T = 512 if quick else 1024
    ds = SyntheticDataset(SyntheticConfig("eeg", 4, window_samples=T,
                                          montage_name="standard_1010_64", seed=seed))
    batch = collate_samples([ds[i] for i in range(4)])
    x = batch["signal"].to(device)
    meta = batch_to_meta(batch)
    meta.channel_xyz = meta.channel_xyz.to(device)

    result = {
        "suite": "eeg", "config": config, "seed": seed, "synthetic": synthetic,
        "channel_robustness": run_channel_robustness(model, meta, x, seed),
        "reference_robustness": run_reference_robustness(model, meta, x),
        "montage_transfer": {
            "19_to_64": run_montage_transfer(model, "standard_1020_19", "standard_1010_64", T),
            "64_to_19": run_montage_transfer(model, "standard_1010_64", "standard_1020_19", T),
        },
        "ssl_cache": model.spatial.ssl_cache_stats() if getattr(model, "spatial", None) else {},
    }
    if output:
        _write(result, output)
    return result


def run_multimodal_robustness(config: str = "experiments/multimodal_robustness",
                              overrides: Sequence[str] = (), output: Optional[str] = None,
                              quick: bool = False, checkpoint: Optional[str] = None,
                              seed: int = 0) -> Dict[str, Any]:
    """RALF under every modality subset and every corruption condition."""
    set_seed(seed)
    cfg = load_config(config, list(overrides))
    device = pick_device()
    model = build_model(cfg).to(device).eval()
    if checkpoint and os.path.exists(checkpoint):
        load_checkpoint(checkpoint, model, strict=False)

    n = 4 if quick else 8
    T = 512 if quick else 1024
    items = synthetic_multimodal(n, T, seed)
    modalities = list(model.encoders.keys())
    batch = {m: collate_samples([it[m] for it in items]) for m in modalities}
    labels = batch["eeg"].get("label")

    conditions = cfg.get("experiment", {}).get("conditions", [])
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for cond in conditions:
            present = cond.get("modalities", modalities)
            corrupt = cond.get("corrupt", {}) or {}
            inputs, metas = {}, {}
            levels = {}
            for m in modalities:
                if m not in present:
                    continue
                x = batch[m]["signal"].to(device)
                if m in corrupt:
                    c = SignalCorruptor(CorruptionConfig(prob=1.0, kinds=[corrupt[m]],
                                                         max_level=1.0))
                    x, lvl, _ = c(x)
                    levels[m] = float(lvl.mean().item())
                inputs[m] = x
                metas[m] = batch_to_meta(batch[m])
            mask = {m: torch.ones(n, dtype=torch.bool, device=device) for m in inputs}
            out = model.fusion(model.encode(inputs, metas), mask)
            row: Dict[str, Any] = {
                "condition": cond["id"],
                "modalities": present,
                "corrupt": corrupt,
                "corruption_level": levels,
                "reliability": {m: float(r.mean().item()) for m, r in out["reliability"].items()},
            }
            if labels is not None:
                pred = out["logits"].argmax(-1).cpu()
                row["accuracy"] = float((pred == labels).float().mean().item())
            rows.append(row)
            logger.info("%-16s modalities=%s reliability=%s", cond["id"], present,
                        {k: round(v, 3) for k, v in row["reliability"].items()})

    # Explicit check: an all-absent input must fail loudly rather than predict.
    all_missing_ok = False
    try:
        model.fusion(model.encode({}, {}), {})
    except (ValueError, AssertionError):
        all_missing_ok = True
    result = {"suite": "multimodal", "synthetic": True, "seed": seed, "rows": rows,
              "all_modalities_missing_raises": all_missing_ok}
    if output:
        _write(result, output)
    return result


def _write(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote %s", path)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="PhysioWave evaluation suites")
    p.add_argument("--suite", default="eeg", choices=["eeg", "multimodal"])
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.suite == "eeg":
        run_eeg_suite(args.config or "pretrain/eeg", args.overrides, args.checkpoint,
                      args.output or "./results/eval_eeg.json", args.quick, args.seed)
    else:
        run_multimodal_robustness(args.config or "experiments/multimodal_robustness",
                                  args.overrides, args.output or "./results/eval_multimodal.json",
                                  args.quick, args.checkpoint, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
