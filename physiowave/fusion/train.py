"""Multimodal fusion on the three C1 encoders: training, selection, the test matrix.

    python -m physiowave.fusion.train --config fusion/c1_ralf \\
        --set data.tasks.mpdb.dir=$D/mpdb data.tasks.seizeit2.dir=$D/seizeit2 \\
              encoders.eeg=... encoders.ecg=... encoders.emg=... \\
              model.fusion=ralf model.encoder_mode=frozen \\
        --output-dir $PW_CKPT_ROOT/fusion/ralf_frozen

WHAT TRAINS. One C1 encoder per modality (shared across the tasks that
have it), a summary pool per modality, and one fusion core with a head per
task. ``model.encoder_mode``: ``frozen`` keeps what pretraining supplied
fixed (the montage adapters and everything downstream train), ``finetune``
trains it under layer-wise decay, ``scratch`` loads nothing (the control).

THE LOSS, per batch of one task:

    L = CE(fused) + uni_w * sum_m CE(unimodal_m)       [on items that have m]
      + rel_w * MSE(reliability_m, 1 - corruption_m)   [RALF only]
      + cons_w * symKL(fused | random modality subset, fused | all present)

The corruption (physiowave.pretrain.corruption) is applied to the raw
windows before the encoders with a known severity, which is the only ground
truth a reliability score has. The subset view drops present modalities at
random (always leaving one), so the prediction learns not to lurch when a
stream goes.

SELECTION on validation, complete modalities, no corruption: the mean over
tasks of each task's ``select_by`` metric. THE TEST SET IS SCORED ONCE, at
that checkpoint, under every condition of the matrix -- complete, each
modality alone, each pair, and each modality corrupted at a fixed severity --
and written to results.json. A condition an item cannot meet (its only
present modality excluded) drops that item from that condition, and the
count is recorded.

Every finished epoch goes to latest.pth; a run started again in the same
directory resumes after it (--fresh starts over), and the file is removed
once results.json is written (--keep-latest keeps it).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config, save_resolved
from ..pretrain.corruption import CORRUPTIONS, CorruptionConfig, SignalCorruptor
from ..train.finetune_main import multiclass_metrics
from ..train.utils import build_optimizer, build_scheduler, set_seed
from .data import MultimodalWindows, batch_to, collate_multimodal
from .encoders import build_encoder, channel_meta, encoder_lr_scale, share_core
from .model import C1Multimodal
from .prep_common import MODALITIES

logger = logging.getLogger("physiowave.fusion")

#: (condition id, modalities kept, {modality: corruption kind}). Pairs and
#: singles are subsets of what the item has; corruption applies on top of
#: the complete set.
DEFAULT_CONDITIONS: List[Tuple[str, Tuple[str, ...], Dict[str, str]]] = [
    ("complete", MODALITIES, {}),
    ("eeg_only", ("eeg",), {}), ("ecg_only", ("ecg",), {}), ("emg_only", ("emg",), {}),
    ("eeg_ecg", ("eeg", "ecg"), {}), ("eeg_emg", ("eeg", "emg"), {}),
    ("ecg_emg", ("ecg", "emg"), {}),
    ("eeg_noise", MODALITIES, {"eeg": "noise"}),
    ("eeg_bad_channel", MODALITIES, {"eeg": "bad_channel"}),
    ("eeg_drift", MODALITIES, {"eeg": "drift"}),
    ("ecg_saturation", MODALITIES, {"ecg": "saturation"}),
    ("ecg_noise", MODALITIES, {"ecg": "noise"}),
    ("emg_dropout", MODALITIES, {"emg": "dropout"}),
    ("emg_noise", MODALITIES, {"emg": "noise"}),
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="fusion/c1_ralf")
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="per epoch; smoke tests")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--progress", default="log", choices=["log", "none"])
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing latest.pth and start the run over")
    p.add_argument("--keep-latest", action="store_true",
                   help="keep latest.pth after results.json is written")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def load_tasks(cfg: Dict, modalities: List[str]) -> Dict[str, Dict[str, MultimodalWindows]]:
    tasks = {}
    for name, spec in (cfg["data"].get("tasks") or {}).items():
        d = (spec or {}).get("dir")
        if not d:
            continue
        splits = {}
        for s in ("train", "val", "test"):
            path = os.path.join(d, f"{s}.h5")
            if os.path.isfile(path):
                splits[s] = MultimodalWindows(path, modalities)
        if "train" not in splits:
            raise SystemExit(f"{name}: no train.h5 under {d}")
        tasks[name] = splits
    if not tasks:
        raise SystemExit("no task has a data dir: set data.tasks.<name>.dir")
    return tasks


def build_model(cfg: Dict, tasks: Dict[str, Dict[str, MultimodalWindows]], out_dir: str
                ) -> Tuple[C1Multimodal, Dict]:
    mcfg = cfg["model"]
    mode = mcfg.get("encoder_mode", "frozen")
    if mode not in ("frozen", "finetune", "scratch"):
        raise SystemExit(f"model.encoder_mode must be frozen, finetune or scratch, not {mode!r}")
    ecfg = cfg.get("encoders") or {}
    ck = {m: ecfg.get(m) for m in MODALITIES}
    arch = {m: str(ecfg.get(f"{m}_arch") or "c1") for m in MODALITIES}
    eeg_adapter = mcfg.get("eeg_adapter") or None
    if mode != "scratch" and not any(ck.values()):
        raise SystemExit("encoder_mode is not scratch but no encoders.<m> path is set")
    encoders, metas, reports, keep = {}, {}, {}, {}
    primary: Dict[str, torch.nn.Module] = {}
    for t, splits in tasks.items():
        ds = splits["train"]
        encoders[t], metas[t], reports[t], keep[t] = {}, {}, {}, {}
        for m in ds.modalities:
            names = ds.channel_names[m]
            enc, rep = build_encoder(
                m, names, ds.rates[m], ds.shapes[m][-1],
                None if mode == "scratch" else ck[m],
                os.path.join(out_dir, "encoders"), freeze=(mode == "frozen"),
                dropout=float(mcfg.get("encoder_dropout", 0.0)),
                max_drop=int(mcfg.get("route_drop_max", 0)), arch=arch[m],
                adapter=eeg_adapter if m == "eeg" else None,
                input_uv=float(ecfg.get("eeg_input_uv", 30.0)))
            keep[t][m] = rep.get("keep_index")
            if rep.get("dropped_channels"):
                names = [n for n in names if n not in rep["dropped_channels"]]
            if m in primary and arch[m] != "c1":
                raise SystemExit(f"{m}: a {arch[m]} encoder is not shared across tasks; "
                                 "train one task")
            if m in primary:
                share_core(primary[m], enc)
                rep["shares_core_with"] = next(tt for tt in encoders if m in encoders[tt])
            else:
                primary[m] = enc
            encoders[t][m] = enc
            metas[t][m] = channel_meta(names)
            reports[t][m] = rep
    n_classes = {t: len(s["train"].class_names) for t, s in tasks.items()}
    core_kw = {k: mcfg[k] for k in ("embed_dim", "num_fusion_tokens", "num_heads", "depth",
                                    "dropout", "residual_weight", "gate_floor", "gate_detach")
               if k in mcfg}
    model = C1Multimodal(encoders, metas, n_classes, fusion=mcfg.get("fusion", "ralf"),
                         summary_tokens=int(mcfg.get("summary_tokens", 4)),
                         summary_heads=int(mcfg.get("summary_heads", 4)),
                         keep_index=keep, **core_kw)
    return model, reports


def class_weights(ds: MultimodalWindows, mode: str, device) -> Optional[torch.Tensor]:
    if mode in (None, "none"):
        return None
    counts = ds.class_counts().astype(np.float64)
    inv = counts.sum() / (len(counts) * np.maximum(counts, 1))
    w = np.sqrt(inv) if mode == "sqrt" else inv
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# Corruption and masks
# --------------------------------------------------------------------------- #

def corrupt_fixed(corruptor: SignalCorruptor, x: torch.Tensor, kind: str, sev: float
                  ) -> torch.Tensor:
    """Every item corrupted by ``kind`` at severity ``sev`` (the test matrix)."""
    out = x.clone()
    for b in range(x.shape[0]):
        out[b] = corruptor._apply(x[b].float(), kind, sev).to(x.dtype)
    return out


def subset_mask(present: Dict[str, torch.Tensor], keep: Tuple[str, ...]
                ) -> Dict[str, torch.Tensor]:
    return {m: (p if m in keep else torch.zeros_like(p)) for m, p in present.items()}


def random_subset(present: Dict[str, torch.Tensor], p_drop: float, gen: torch.Generator
                  ) -> Dict[str, torch.Tensor]:
    """Drop present modalities at random; an item always keeps at least one."""
    mods = list(present)
    B = next(iter(present.values())).shape[0]
    dev = next(iter(present.values())).device
    drop = {m: (torch.rand(B, generator=gen, device="cpu").to(dev) < p_drop) for m in mods}
    mask = {m: present[m] & ~drop[m] for m in mods}
    none_left = ~torch.stack([mask[m] for m in mods]).any(0)
    if bool(none_left.any()):
        for m in mods:                       # restore the item's own modalities
            mask[m] = torch.where(none_left, present[m], mask[m])
    return mask


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def train_step(model, batch, task, cfg, corruptor, cw, amp, gen) -> Tuple[torch.Tensor, Dict]:
    tcfg = cfg["train"]
    x, present, y = batch["x"], batch["present"], batch["label"]
    levels = {}
    corrupted = {}
    for m in model.modalities:
        if corruptor.cfg.enabled:
            xm, lvl, _ = corruptor(x[m])
        else:
            xm, lvl = x[m], torch.zeros(x[m].shape[0], device=x[m].device)
        corrupted[m], levels[m] = xm, lvl
    with amp:
        feats = model.encode(corrupted, task)
        full = model.core(feats, task, present)
        loss_task = F.cross_entropy(full["logits"].float(), y, weight=cw)
        loss = loss_task
        logs = {"loss_task": float(loss_task)}
        uni_w = float(tcfg.get("uni_weight", 0.2))
        for m, lg in full["unimodal_logits"].items():
            w = present[m].float()
            if w.sum() <= 0:
                continue
            l_m = (F.cross_entropy(lg.float(), y, weight=cw, reduction="none") * w).sum() / w.sum()
            loss = loss + uni_w * l_m
            logs[f"loss_uni_{m}"] = float(l_m)
        rel_w = float(tcfg.get("reliability_weight", 1.0))
        if rel_w and full["reliability"]:
            targets = {m: (1.0 - levels[m]).clamp(0, 1) for m in levels}
            l_rel = model.core.reliability_loss(full["reliability"], targets, full["present"])
            loss = loss + rel_w * l_rel
            logs["loss_reliability"] = float(l_rel)
        cons_w = float(tcfg.get("consistency_weight", 1.0))
        if cons_w:
            sub = random_subset(present, float(tcfg.get("modality_dropout", 0.3)), gen)
            partial = model.core(feats, task, sub)
            l_cons = model.core.consistency_loss(partial["logits"], full["logits"].detach())
            loss = loss + cons_w * l_cons
            logs["loss_consistency"] = float(l_cons)
    logs["loss"] = float(loss)
    return loss, logs


@torch.no_grad()
def evaluate(model, loader, task, device, amp, keep: Tuple[str, ...] = MODALITIES,
             corrupt: Optional[Dict[str, str]] = None, severity: float = 0.7,
             corruptor: Optional[SignalCorruptor] = None, num_classes: int = 2) -> Dict:
    model.eval()
    probs, labels, rel = [], [], {m: [] for m in model.modalities}
    dropped = 0
    for batch in loader:
        batch = batch_to(batch, device)
        x, present = batch["x"], batch["present"]
        mask = subset_mask(present, keep)
        ok = torch.stack([mask[m] for m in mask]).any(0)
        dropped += int((~ok).sum())
        if not bool(ok.any()):
            continue
        if not bool(ok.all()):
            x = {m: v[ok] for m, v in x.items()}
            mask = {m: v[ok] for m, v in mask.items()}
            y = batch["label"][ok]
        else:
            y = batch["label"]
        if corrupt:
            x = dict(x)
            for m, kind in corrupt.items():
                if m in x:
                    x[m] = corrupt_fixed(corruptor, x[m], kind, severity)
        with amp:
            out = model(x, task, mask)
        probs.append(torch.softmax(out["logits"].float(), -1).cpu().numpy())
        labels.append(y.cpu().numpy())
        for m, r in out["reliability"].items():
            rel[m].append(r.float().cpu().numpy()[mask[m].cpu().numpy()])
    model.train()
    if not probs:
        return {"n": 0, "dropped": dropped}
    y_prob, y_true = np.concatenate(probs), np.concatenate(labels)
    metrics = multiclass_metrics(y_prob, y_true, num_classes)
    if num_classes == 2 and len(np.unique(y_true)) == 2:
        from sklearn.metrics import average_precision_score
        metrics["auprc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        pred = y_prob[:, 1] >= 0.5
        metrics["sensitivity"] = float(pred[y_true == 1].mean())
        metrics["specificity"] = float((~pred[y_true == 0]).mean())
    metrics["n"] = int(len(y_true))
    metrics["dropped"] = dropped
    for m, parts in rel.items():
        if parts and sum(len(p) for p in parts):
            metrics[f"reliability_{m}"] = float(np.concatenate(parts).mean())
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    save_resolved(cfg, os.path.join(out_dir, "config_resolved.yaml"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.get("seed", 42)), True)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))

    modalities = list(cfg["data"].get("modalities") or MODALITIES)
    tasks = load_tasks(cfg, modalities)
    model, reports = build_model(cfg, tasks, out_dir)
    model.to(device)
    with open(os.path.join(out_dir, "encoders.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)
    for t, per in reports.items():
        for m, rep in per.items():
            logger.info("%s/%s: route %s, %s%s%s", t, m, rep.get("route"), rep.get("summary"),
                        f" (shares core with {rep['shares_core_with']})"
                        if rep.get("shares_core_with") else "",
                        f" [electrodes left out for the route: {rep['dropped_channels']}]"
                        if rep.get("dropped_channels") else "")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model %.2f M parameters, %.2f M trainable; fusion=%s encoder_mode=%s",
                n_total / 1e6, n_train / 1e6, model.fusion_name, cfg["model"]["encoder_mode"])

    tcfg = cfg["train"]
    nw = int(cfg["data"].get("num_workers", 4))
    bs = int(tcfg["batch_size"])

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                          collate_fn=collate_multimodal, pin_memory=device.type == "cuda",
                          drop_last=False, persistent_workers=nw > 0)
    loaders = {t: {s: loader(ds, s == "train") for s, ds in splits.items()}
               for t, splits in tasks.items()}
    cw = {t: class_weights(s["train"], tcfg.get("class_weight", "sqrt"), device)
          for t, s in tasks.items()}
    n_classes = {t: len(s["train"].class_names) for t, s in tasks.items()}
    select_by = {t: (cfg["data"]["tasks"][t] or {}).get("select_by", "balanced_acc")
                 for t in tasks}

    # One schedule over both tasks: every training batch of every task once per
    # epoch, in a shuffled order, so neither task waits for the other.
    steps_per_task = {t: len(loaders[t]["train"]) for t in tasks}
    if args.max_steps:
        steps_per_task = {t: min(n, args.max_steps) for t, n in steps_per_task.items()}
    epochs = int(tcfg["epochs"])
    total_steps = epochs * sum(steps_per_task.values())

    lr_scale = None
    if cfg["model"]["encoder_mode"] == "finetune":
        lr_scale = encoder_lr_scale("encoders", model.encoder_depth(),
                                    float(tcfg.get("encoder_lr_scale", 1.0)),
                                    float(tcfg.get("layer_decay", 1.0)), model.encoders)
    optimizer = build_optimizer(model, float(tcfg["lr"]), float(tcfg["weight_decay"]),
                                lr_scale=lr_scale)
    scheduler = build_scheduler(optimizer, int(float(tcfg.get("warmup_ratio", 0.1)) * total_steps),
                                total_steps, float(tcfg.get("min_lr_ratio", 0.01)))
    corruptor = SignalCorruptor(CorruptionConfig(**(cfg.get("corruption") or {})),
                                torch.Generator().manual_seed(int(cfg.get("seed", 42)) + 1))
    use_amp = device.type == "cuda" and tcfg.get("precision", "bf16") == "bf16"
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if use_amp else _Null()
    logger.info("tasks %s; steps/epoch %s; %d epochs", {t: n_classes[t] for t in tasks},
                steps_per_task, epochs)

    history, best, best_epoch, bad = [], -math.inf, -1, 0
    patience = int(tcfg.get("patience", 0))
    t_run = time.time()

    # -- resume: every finished epoch is in latest.pth ------------------------- #
    latest = os.path.join(out_dir, "latest.pth")
    start_epoch, stopped_early = 0, False
    if os.path.isfile(latest) and not args.fresh:
        st = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        history, best, best_epoch, bad = st["history"], st["best"], st["best_epoch"], st["bad"]
        start_epoch, stopped_early = int(st["epoch"]) + 1, bool(st.get("stopped_early"))
        torch.set_rng_state(st["rng"]["torch"])
        gen.set_state(st["rng"]["gen"])
        if corruptor.g is not None and st["rng"].get("corruptor") is not None:
            corruptor.g.set_state(st["rng"]["corruptor"])
        logger.info("resuming after epoch %d (best %.4f at epoch %d%s)", st["epoch"], best,
                    best_epoch, "; training had stopped early" if stopped_early else "")
        t_run -= float(st.get("seconds", 0.0))

    def save_latest(epoch: int, stopped: bool) -> None:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history, "best": best,
                 "best_epoch": best_epoch, "bad": bad, "epoch": epoch, "stopped_early": stopped,
                 "seconds": time.time() - t_run,
                 "rng": {"torch": torch.get_rng_state(), "gen": gen.get_state(),
                         "corruptor": corruptor.g.get_state() if corruptor.g is not None else None}}
        torch.save(state, latest + ".tmp")
        os.replace(latest + ".tmp", latest)

    for epoch in range(epochs if stopped_early else start_epoch, epochs):
        model.train()
        order = [t for t, n in steps_per_task.items() for _ in range(n)]
        rng = np.random.default_rng(int(cfg.get("seed", 42)) + epoch)
        order = [order[i] for i in rng.permutation(len(order))]
        its = {t: iter(loaders[t]["train"]) for t in tasks}
        sums: Dict[str, Dict[str, float]] = {t: {} for t in tasks}
        counts = {t: 0 for t in tasks}
        t0 = time.time()
        for k, t in enumerate(order):
            try:
                batch = next(its[t])
            except StopIteration:
                its[t] = iter(loaders[t]["train"])
                batch = next(its[t])
            batch = batch_to(batch, device)
            loss, logs = train_step(model, batch, t, cfg, corruptor, cw[t], amp, gen)
            if not torch.isfinite(loss):
                logger.warning("non-finite loss on %s at step %d; batch skipped", t, k)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            for key, v in logs.items():
                sums[t][key] = sums[t].get(key, 0.0) + v
            counts[t] += 1
            if args.progress == "log" and (k % 50 == 0 or k == len(order) - 1):
                logger.info("epoch %d step %d/%d [%s] loss %.4f lr %.2e", epoch, k + 1,
                            len(order), t, logs["loss"], optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "seconds": time.time() - t0,
               "train": {t: {key: v / max(counts[t], 1) for key, v in sums[t].items()}
                         for t in tasks}, "val": {}}
        scores = []
        for t in tasks:
            if "val" in loaders[t]:
                mv = evaluate(model, loaders[t]["val"], t, device, amp, num_classes=n_classes[t])
                row["val"][t] = mv
                scores.append(mv.get(select_by[t], float("nan")))
        finite = [v for v in scores if v == v]
        if scores and not finite:
            logger.warning("no task has a validation score this epoch; selecting on the "
                           "training loss instead")
        score = (float(np.mean(finite)) if finite
                 else -float(np.mean([row["train"][t].get("loss", float("nan")) for t in tasks])))
        row["score"] = score
        history.append(row)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        logger.info("epoch %d done: %s | score %.4f%s", epoch,
                    " ".join(f"{t}: " + ", ".join(f"{k}={v:.4f}" for k, v in mv.items()
                                                 if k in (select_by[t], "acc", "auroc"))
                             for t, mv in row["val"].items()),
                    score, "  (best)" if score > best else "")
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score,
                        "config": cfg}, os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
        stop = bool(patience and bad >= patience)
        save_latest(epoch, stop)
        if stop:
            logger.info("no improvement for %d epochs; stopping", patience)
            break

    # -- the test matrix, once, at the selected checkpoint -------------------- #
    if best_epoch < 0:
        logger.warning("no epoch produced a finite selection score; the final weights are "
                       "scored and marked as such")
        torch.save({"model": model.state_dict(), "epoch": len(history) - 1, "score": None,
                    "config": cfg}, os.path.join(out_dir, "best.pth"))
    ck = torch.load(os.path.join(out_dir, "best.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    sev = float((cfg.get("eval") or {}).get("corruption_severity", 0.7))
    conditions, seen = [], set()
    for cid, keep, corr in DEFAULT_CONDITIONS:
        kept = tuple(m for m in keep if m in modalities)
        if not kept or not all(m in modalities for m in corr):
            continue
        key = (kept, tuple(sorted(corr.items())))
        if key in seen:              # e.g. eeg+ecg on an EEG-only model is eeg alone
            continue
        seen.add(key)
        conditions.append((cid, kept, corr))
    test_corruptor = SignalCorruptor(CorruptionConfig(prob=1.0, kinds=list(CORRUPTIONS)),
                                     torch.Generator().manual_seed(1234))
    results = {"fusion": model.fusion_name, "encoder_mode": cfg["model"]["encoder_mode"],
               "seed": int(cfg.get("seed", 42)),
               "best_epoch": best_epoch, "best_score": best, "select_by": select_by,
               "epochs_run": len(history),
               "hparams": {k: tcfg.get(k) for k in ("epochs", "batch_size", "lr", "layer_decay",
                                                    "weight_decay", "modality_dropout",
                                                    "uni_weight", "reliability_weight",
                                                    "consistency_weight", "class_weight")},
               "corruption_severity": sev, "encoders": reports,
               "seconds": time.time() - t_run, "tasks": {}}
    for t in tasks:
        entry = {"classes": tasks[t]["train"].class_names,
                 "val": history[best_epoch]["val"].get(t) if best_epoch >= 0 else None,
                 "test": {}}
        if "test" in loaders[t]:
            for cid, keep, corr in conditions:
                entry["test"][cid] = evaluate(model, loaders[t]["test"], t, device, amp, keep,
                                              corr or None, sev, test_corruptor, n_classes[t])
                logger.info("test %s/%s: %s", t, cid,
                            ", ".join(f"{k}={v:.4f}" for k, v in entry["test"][cid].items()
                                      if isinstance(v, float) and k in
                                      ("acc", "balanced_acc", "macro_f1", "auroc", "auprc")))
        results["tasks"][t] = entry
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("wrote %s", os.path.join(out_dir, "results.json"))
    if os.path.isfile(latest) and not args.keep_latest:
        os.remove(latest)
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
