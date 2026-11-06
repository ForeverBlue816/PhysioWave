"""RALF multimodal fusion training.

Trains the fusion module (optionally with frozen, independently pretrained
encoders) with four objectives:

* task cross-entropy on the fused logits;
* reliability regression against the *known* synthetic corruption level;
* prediction consistency between the complete-modality view and the
  dropped/corrupted views;
* per-modality auxiliary cross-entropy on the unimodal residual heads.

Usage::

    python -m physiowave.train.fusion_main --config fusion/ralf
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import instantiate, load_config, save_resolved
from ..data.schema import batch_to_meta, collate_samples
from ..data.synthetic import synthetic_multimodal
from ..models.build import build_model, count_parameters, load_pretrained_encoders
from ..models.checkpoint import load_checkpoint, save_checkpoint
from ..pretrain.corruption import CorruptionConfig, SignalCorruptor
from .utils import (
    MetricLogger,
    TensorBoardWriter,
    Throughput,
    autocast_ctx,
    build_optimizer,
    build_scheduler,
    check_finite,
    cleanup_distributed,
    resolve_precision,
    set_seed,
    setup_distributed,
)

logger = logging.getLogger(__name__)
MODALITIES = ("eeg", "ecg", "semg")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhysioWave RALF fusion training")
    p.add_argument("--config", default="fusion/ralf")
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


class MultimodalSyntheticLoader:
    """Minimal batching over aligned synthetic EEG/ECG/sEMG windows."""

    def __init__(self, num_samples: int, batch_size: int, window_samples: int, seed: int = 0):
        self.items = synthetic_multimodal(num_samples, window_samples, seed)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(len(self.items) // self.batch_size, 1)

    def __iter__(self):
        for i in range(0, len(self.items) - self.batch_size + 1, self.batch_size):
            chunk = self.items[i:i + self.batch_size]
            yield {m: collate_samples([c[m] for c in chunk]) for m in MODALITIES}


def sample_modality_mask(
    B: int, modalities: List[str], dropout: float, device
) -> Dict[str, torch.Tensor]:
    """Random per-item modality availability, guaranteeing at least one present."""
    mask = {m: torch.rand(B, device=device) > dropout for m in modalities}
    stacked = torch.stack([mask[m] for m in modalities], dim=0)
    empty = ~stacked.any(dim=0)
    if bool(empty.any()):
        keep = modalities[0]
        mask[keep] = mask[keep] | empty
    return mask


def train_step(model, batch, cfg, corruptor, device, amp_dtype) -> Dict[str, Any]:
    core = model.module if isinstance(model, DDP) else model
    inputs, metas, levels = {}, {}, {}
    for m in core.encoders.keys():
        x = batch[m]["signal"].to(device)
        x, lvl, _ = corruptor(x)
        inputs[m] = x
        metas[m] = batch_to_meta(batch[m])
        levels[m] = lvl
    labels = batch["eeg"].get("label")
    labels = labels.to(device) if labels is not None else None

    modalities = list(core.encoders.keys())
    B = inputs[modalities[0]].shape[0]
    with autocast_ctx(device, amp_dtype):
        feats = core.encode(inputs, metas)
        full = core.fusion(feats, {m: torch.ones(B, dtype=torch.bool, device=device)
                                   for m in modalities})
        drop_mask = sample_modality_mask(B, modalities, cfg["model"]["ralf"].get(
            "modality_dropout", 0.2), device)
        partial = core.fusion(feats, drop_mask)

        loss = torch.zeros((), device=device)
        logs: Dict[str, float] = {}
        if labels is not None:
            l_task = F.cross_entropy(full["logits"], labels)
            loss = loss + l_task
            logs["loss_task"] = float(l_task.item())
            for m, lg in full["unimodal_logits"].items():
                l_uni = F.cross_entropy(lg, labels)
                loss = loss + 0.2 * l_uni
                logs[f"loss_uni_{m}"] = float(l_uni.item())

        targets = {m: (1.0 - levels[m]).clamp(0, 1) for m in modalities}
        l_rel = core.fusion.reliability_loss(full["reliability"], targets, full["present"])
        loss = loss + float(cfg["model"]["ralf"].get("reliability_weight", 1.0)) * l_rel
        logs["loss_reliability"] = float(l_rel.item())

        l_cons = core.fusion.consistency_loss(partial["logits"], full["logits"].detach())
        loss = loss + float(cfg["model"]["ralf"].get("consistency_weight", 1.0)) * l_cons
        logs["loss_consistency"] = float(l_cons.item())

        for m in modalities:
            logs[f"reliability_{m}"] = float(full["reliability"][m].mean().item())
            logs[f"corruption_{m}"] = float(levels[m].mean().item())
    logs["loss_total"] = float(loss.item())
    return {"loss": loss, "logs": logs}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    out_dir = args.output_dir or cfg.get("output_dir", "./outputs/fusion").replace(
        "${run_name}", str(cfg.get("run_name", "fusion")))

    if args.dry_run:
        model = build_model(cfg)
        print(f"[dry-run] RALF model params={count_parameters(model)['total']:,} "
              f"modalities={list(model.encoders.keys())} output_dir={out_dir}")
        return 0

    info = setup_distributed()
    set_seed(int(cfg.get("seed", 42)) + info.rank, cfg["train"].get("deterministic", True))
    if info.is_main:
        os.makedirs(out_dir, exist_ok=True)
        save_resolved(cfg, os.path.join(out_dir, "config_resolved.yaml"))

    model = build_model(cfg).to(info.device)
    pre = (cfg.get("fusion", {}) or {}).get("pretrained", {}) or {}
    if any(pre.values()):
        load_pretrained_encoders(model, pre, strict=False)
    if info.is_main:
        logger.info("RALF params=%s", count_parameters(model))
    if info.distributed:
        model = DDP(model, device_ids=[info.local_rank] if info.device.type == "cuda" else None,
                    find_unused_parameters=True)

    corruptor = SignalCorruptor(instantiate(
        CorruptionConfig, (cfg.get("fusion", {}) or {}).get("corruption", {})))
    precision, amp_dtype = resolve_precision(cfg["train"].get("precision", "bf16"), info.device)
    optimizer = build_optimizer(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    syn = cfg.get("data", {}).get("synthetic", {}) or {}
    loader = MultimodalSyntheticLoader(int(syn.get("num_samples", 64)),
                                       int(cfg["train"]["batch_size"]),
                                       int(syn.get("window_samples", 1024)),
                                       int(cfg.get("seed", 42)))
    scheduler = build_scheduler(optimizer, max(len(loader), 1), max(epochs * len(loader), 1))

    start_epoch, step = 0, 0
    if args.resume:
        path = args.resume if args.resume != "auto" else os.path.join(out_dir, "latest.pth")
        if os.path.exists(path):
            payload = load_checkpoint(path, model, optimizer, scheduler, strict=True)
            start_epoch = int(payload.get("epoch", 0)) + 1
            step = int(payload.get("step", 0))

    writer = TensorBoardWriter(os.path.join(out_dir, "tb"), info.is_main)
    for epoch in range(start_epoch, epochs):
        model.train()
        meters, tp = MetricLogger(), Throughput()
        for batch in loader:
            out = train_step(model, batch, cfg, corruptor, info.device, amp_dtype)
            if not check_finite(out["loss"], step, out["logs"]):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            step += 1
            meters.update(out["logs"], n=batch["eeg"]["signal"].shape[0])
            tp.update(batch["eeg"]["signal"].shape[0])
            if args.max_steps and step >= args.max_steps:
                break
        stats = {**meters.averages(), **tp.rates()}
        if info.is_main:
            writer.add_scalars(stats, step, prefix="fusion/")
            logger.info("epoch %d | %s", epoch,
                        " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items())))
            save_checkpoint(os.path.join(out_dir, "latest.pth"), model, optimizer, scheduler,
                            None, epoch, step, stats, cfg)
        if args.max_steps and step >= args.max_steps:
            break

    if info.is_main:
        writer.close()
        logger.info("Fusion training finished; outputs in %s", out_dir)
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
