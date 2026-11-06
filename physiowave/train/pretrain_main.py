"""Single-modality self-supervised pretraining entry point.

Supports torchrun/DDP, single-GPU, MPS and CPU; bf16 with automatic fallback;
gradient accumulation and clipping; deterministic seeding; distributed sampling;
automatic resume from a full training checkpoint; and per-term loss logging with
throughput, peak memory, FLOPs and the token compression ratio.

Usage::

    python -m physiowave.train.pretrain_main --config pretrain/eeg
    torchrun --nproc_per_node=4 -m physiowave.train.pretrain_main --config pretrain/eeg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import instantiate, load_config, save_resolved
from ..data.datasets import LoaderConfig, build_dataloader
from ..data.schema import batch_to_meta
from ..models.build import build_model, count_parameters
from ..models.checkpoint import environment_info, load_checkpoint, save_checkpoint
from ..pretrain.objectives import PretrainObjective, PretrainObjectiveConfig
from .data_builder import build_datasets, maybe_weighted_sampler
from .utils import (
    DistInfo,
    MetricLogger,
    TensorBoardWriter,
    Throughput,
    all_reduce_mean,
    autocast_ctx,
    build_optimizer,
    build_scheduler,
    check_finite,
    cleanup_distributed,
    make_grad_scaler,
    peak_memory_mb,
    reset_peak_memory,
    resolve_precision,
    set_seed,
    setup_distributed,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhysioWave single-modality pretraining")
    p.add_argument("--config", required=True, help="config path under configs/ (no .yaml needed)")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="dotted overrides, e.g. --set model.wast.level=4 train.epochs=2")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from ('auto' = latest)")
    p.add_argument("--init-from", default=None,
                   help="start a NEW run from another checkpoint's weights: no "
                        "optimizer, no scheduler position, no step count. What "
                        "you want when the mixture or the epoch length changes, "
                        "because a resumed scheduler counts in the old epoch's "
                        "units. eeg_c1 and ecg_c1 only.")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="validate config/data and exit")
    p.add_argument("--smoke-test", action="store_true",
                   help="run on synthetic data. Only the eeg_c1_moe and "
                        "ecg_c1_moe trainers implement it, and it is the ONLY way that path "
                        "fabricates signal -- nothing falls back to it.")
    return p.parse_args(argv)


def resolve_output_dir(cfg: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    out = cfg.get("output_dir", "./outputs/run")
    return out.replace("${run_name}", str(cfg.get("run_name", "run")))


def run_dry_run(cfg: Dict[str, Any], out_dir: str) -> int:
    """Validate the config, the data paths and the model build, then stop."""
    modality = cfg.get("data", {}).get("modality", "eeg")
    print(f"[dry-run] config resolved; run_name={cfg.get('run_name')} modality={modality}")
    model = build_model(cfg)
    print(f"[dry-run] model={type(model).__name__} params={count_parameters(model)['total']:,}")
    train, val, stats = build_datasets(cfg.get("data", {}), modality)
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    print(f"[dry-run] train={len(train)} val={len(val) if val else 0} "
          f"stats={json.dumps(public)[:400]}")
    print(f"[dry-run] output_dir={out_dir}")
    print(f"[dry-run] env={json.dumps(environment_info())}")
    return 0


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    objective: PretrainObjective,
    loader,
    optimizer,
    scheduler,
    scaler,
    info: DistInfo,
    cfg: Dict[str, Any],
    writer: TensorBoardWriter,
    global_step: int,
    amp_dtype,
    max_steps: Optional[int],
) -> Dict[str, float]:
    model.train()
    meters = MetricLogger()
    tp = Throughput()
    tcfg = cfg["train"]
    accum = int(tcfg.get("grad_accumulation_steps", 1))
    clip = float(tcfg.get("grad_clip", 1.0))
    reset_peak_memory(info.device)
    core = model.module if isinstance(model, DDP) else model

    optimizer.zero_grad(set_to_none=True)
    for it, batch in enumerate(loader):
        x = batch["signal"].to(info.device, non_blocking=True)
        meta = batch_to_meta(batch)
        with autocast_ctx(info.device, amp_dtype):
            out = objective(core, x, meta)
            loss = out["loss"] / accum

        if tcfg.get("check_nan", True) and not check_finite(loss, global_step, out["logs"]):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum == 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        ts = out["token_stats"]
        tp.update(x.shape[0], int(ts.get("N_new", ts.get("N_wast", 0))) * x.shape[0])
        meters.update(out["logs"], n=x.shape[0])

        if info.is_main and global_step % int(tcfg.get("log_every", 10)) == 0:
            rates = tp.rates()
            payload = {**out["logs"], **rates,
                       "lr": optimizer.param_groups[0]["lr"],
                       "peak_mem_mb": peak_memory_mb(info.device),
                       "token_compression": ts.get("compression_vs_legacy_with_compression",
                                                   ts.get("compression_vs_legacy", 1.0))}
            writer.add_scalars(payload, global_step, prefix="train/")
            logger.info("epoch %d step %d | %s", epoch, global_step,
                        " ".join(f"{k}={v:.4f}" for k, v in sorted(payload.items())))
        if max_steps and global_step >= max_steps:
            break

    stats = meters.averages()
    stats["global_step"] = global_step
    stats.update(tp.rates())
    stats["peak_mem_mb"] = peak_memory_mb(info.device)
    return stats


@torch.no_grad()
def validate(model, objective, loader, info: DistInfo, amp_dtype, max_steps=None) -> Dict[str, float]:
    model.eval()
    meters = MetricLogger()
    core = model.module if isinstance(model, DDP) else model
    for it, batch in enumerate(loader):
        x = batch["signal"].to(info.device, non_blocking=True)
        with autocast_ctx(info.device, amp_dtype):
            out = objective(core, x, batch_to_meta(batch))
        meters.update(out["logs"], n=x.shape[0])
        if max_steps and it + 1 >= max_steps:
            break
    return {k: all_reduce_mean(v, info) for k, v in meters.averages().items()}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    out_dir = resolve_output_dir(cfg, args.output_dir)

    # The EEG C1 multi-route path shares no model, objective or data builder
    # with what follows, so it is dispatched whole rather than threaded
    # through a loop whose every stage would have to become conditional.
    if cfg.get("trainer") == "eeg_c1_moe":
        from ..eeg_c1.entry import run as run_eeg_c1
        return run_eeg_c1(cfg, out_dir, args)
    if cfg.get("trainer") == "ecg_c1_moe":
        from ..ecg_c1.entry import run as run_ecg_c1
        return run_ecg_c1(cfg, out_dir, args)
    if cfg.get("trainer") == "emg_c1_moe":
        from ..emg_c1.entry import run as run_emg_c1
        return run_emg_c1(cfg, out_dir, args)

    if args.dry_run:
        return run_dry_run(cfg, out_dir)

    info = setup_distributed()
    set_seed(int(cfg.get("seed", 42)) + info.rank, cfg["train"].get("deterministic", True))
    if info.is_main:
        os.makedirs(out_dir, exist_ok=True)
        save_resolved(cfg, os.path.join(out_dir, "config_resolved.yaml"))
        with open(os.path.join(out_dir, "environment.json"), "w") as f:
            json.dump(environment_info(), f, indent=2)

    modality = cfg.get("data", {}).get("modality", "eeg")
    train_ds, val_ds, data_stats = build_datasets(cfg.get("data", {}), modality)
    lcfg = LoaderConfig(batch_size=int(cfg["train"]["batch_size"]),
                        num_workers=int(cfg["train"].get("num_workers", 0)))
    sampler = maybe_weighted_sampler(cfg.get("data", {}), data_stats, info.distributed)
    train_loader = build_dataloader(train_ds, lcfg, info.distributed, sampler=sampler,
                                    seed=int(cfg.get("seed", 42)))
    val_loader = build_dataloader(
        val_ds, LoaderConfig(batch_size=lcfg.batch_size, num_workers=lcfg.num_workers,
                             shuffle=False, drop_last=False),
        info.distributed, seed=int(cfg.get("seed", 42))) if val_ds is not None else None

    model = build_model(cfg).to(info.device)
    objective = PretrainObjective(instantiate(PretrainObjectiveConfig, cfg.get("pretrain", {})))
    if info.is_main:
        public_stats = {k: v for k, v in data_stats.items() if not k.startswith("_")}
        logger.info("model=%s params=%s data=%s", type(model).__name__,
                    count_parameters(model), json.dumps(public_stats)[:300])

    if info.distributed:
        model = DDP(model, device_ids=[info.local_rank] if info.device.type == "cuda" else None,
                    find_unused_parameters=True)

    precision, amp_dtype = resolve_precision(cfg["train"].get("precision", "bf16"), info.device)
    scaler = make_grad_scaler(info.device.type, precision == "fp16")
    optimizer = build_optimizer(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(len(train_loader) // int(cfg["train"].get("grad_accumulation_steps", 1)), 1)
    scheduler = build_scheduler(optimizer, int(cfg["train"].get("warmup_epochs", 1)) * steps_per_epoch,
                                epochs * steps_per_epoch,
                                float(cfg["train"].get("min_lr_ratio", 0.01)))

    start_epoch, global_step, best = 0, 0, float("inf")
    resume = args.resume
    if resume == "auto":
        cand = os.path.join(out_dir, "latest.pth")
        resume = cand if os.path.exists(cand) else None
    if resume:
        payload = load_checkpoint(resume, model, optimizer, scheduler, scaler, strict=True)
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("step", 0))
        best = float(payload.get("metrics", {}).get("val_loss_total", float("inf")))
        if info.is_main:
            logger.info("Resumed from %s at epoch %d step %d", resume, start_epoch, global_step)

    writer = TensorBoardWriter(os.path.join(out_dir, "tb"),
                               cfg["train"].get("tensorboard", True) and info.is_main)
    history = []
    for epoch in range(start_epoch, epochs):
        if info.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        stats = train_one_epoch(epoch, model, objective, train_loader, optimizer, scheduler,
                                scaler, info, cfg, writer, global_step, amp_dtype, args.max_steps)
        global_step = int(stats.pop("global_step"))
        stats["epoch_seconds"] = time.time() - t0

        val_stats = {}
        if val_loader is not None:
            val_stats = {f"val_{k}": v for k, v in
                         validate(model, objective, val_loader, info, amp_dtype).items()}
        merged = {**stats, **val_stats}
        if info.is_main:
            writer.add_scalars(merged, global_step, prefix="epoch/")
            logger.info("epoch %d done | %s", epoch,
                        " ".join(f"{k}={v:.4f}" for k, v in sorted(merged.items())))
            history.append({"epoch": epoch, **merged})
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            save_checkpoint(os.path.join(out_dir, "latest.pth"), model, optimizer, scheduler,
                            scaler, epoch, global_step, merged, cfg)
            score = merged.get("val_loss_total", merged.get("loss_total", float("inf")))
            if score < best:
                best = score
                save_checkpoint(os.path.join(out_dir, "best.pth"), model, optimizer, scheduler,
                                scaler, epoch, global_step, merged, cfg)
            if (epoch + 1) % int(cfg["train"].get("save_every_epochs", 1)) == 0:
                save_checkpoint(os.path.join(out_dir, f"epoch_{epoch:04d}.pth"), model, optimizer,
                                scheduler, scaler, epoch, global_step, merged, cfg)
        if args.max_steps and global_step >= args.max_steps:
            break

    if info.is_main:
        writer.close()
        logger.info("Pretraining finished; best score %.5f; outputs in %s", best, out_dir)
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
