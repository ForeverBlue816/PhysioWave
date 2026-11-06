"""Supervised fine-tuning for the extension encoder (WAST + TARE + backbone).

The legacy ``finetune.py`` at the repository root builds
``model.BERTWaveletTransformer`` and cannot construct this model at all, so
until now the only supervised numbers in the project were the legacy
architecture's. This is the entry point that puts the token-efficient path on
the same downstream tasks, which is what makes the two comparable.

Data comes straight from the HDF5 that the dataset converters write
(``EMG/db5_finetune.py``, ``EMG/db6_finetune.py``, ``EEG/tuab_finetune.py``):

    data   (N, C, T) float32
    label  (N,)      int64, contiguous from 0

That is deliberately not the manifest path ``pretrain_main`` uses. Pretraining
mixes corpora and needs the registry to know what each one is; fine-tuning is
handed three files and a label column, and routing that through a manifest adds
a registry entry per downstream task without answering a question.

Channel metadata
----------------
TARE wants 3-D electrode coordinates. A forearm sEMG ring has no standard ones,
so the coordinates are left as the all-zero "unknown" row and TARE falls back to
its name branch, which for ``ch00..ch13`` amounts to a learned per-channel
embedding. That is weaker than TARE on a scalp montage and it is the honest
default: inventing cylinder coordinates for the ring would be a prior of ours,
not a measurement, and belongs in an ablation rather than in the main result.
``--channel-names`` and ``--channel-xyz`` supply real metadata when there is any.

Usage
-----
    torchrun --standalone --nproc_per_node=4 -m physiowave.train.finetune_main \\
        --config pretrain/semg \\
        --data-dir $SCRATCH/bio/emg/db6 --num-classes 8 \\
        --output-dir $PW_FAST_ROOT/runs/db6_wast
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..channels.tare import ChannelMeta
from ..config import load_config
from ..models.build import build_model
from ..models.checkpoint import load_checkpoint, save_checkpoint
from .published import CHANCE, EEGPT, PUBLISHED, TASK_METRICS, task_of
from .utils import (
    AverageMeter,
    autocast_ctx,
    bar_write,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    close_bar,
    epoch_bar,
    fmt_eta,
    make_grad_scaler,
    PROGRESS_MODES,
    progress,
    resolve_progress,
    set_postfix,
    set_postfix_str,
    sparkbar,
    pick_device,
    resolve_precision,
    set_seed,
    setup_distributed,
)

logger = logging.getLogger(__name__)

SELECTION_METRICS = ("loss", "acc", "balanced_acc", "kappa", "weighted_f1", "auroc",
                     # multi-label (CPSC 2018, Chapman-Shaoxing): `auroc` is then
                     # the macro AUROC over labels, and these are defined too
                     "macro_f1", "micro_f1", "micro_auroc", "macro_auprc")

#: Command line, then the config's ``train:`` block, then these.
#:
#: The middle step is new and it was a real hole: nothing in this file ever read
#: ``cfg["train"]``, so every value came from an argparse default and the
#: ``lr: 1.0e-4`` in ``finetune/eeg_c1_p300.yaml`` and the ``batch_size: 32`` in
#: ``finetune/eeg_c1_sleep.yaml`` described runs that never happened -- the
#: sleep runs were at 64. A config block that is read by nobody is worse than
#: no config block, because it is quoted in a paper as if it ran.
HPARAM_FALLBACKS = {
    "epochs": 60, "batch_size": 64, "lr": 3e-4, "weight_decay": 0.05,
    "warmup_epochs": 5, "warmup_ratio": None, "grad_clip": 1.0,
    "label_smoothing": 0.1, "min_lr_ratio": 0.01, "precision": "bf16",
    "select_by": "balanced_acc",
    # Early stopping is a training hyper-parameter like the rest, and it is in
    # this table for the reason the others are: the table is LOGGED. Left out,
    # `--patience` was a flag whose absence looked exactly like its presence
    # until the run failed to stop, twenty epochs later.
    "patience": 0, "min_delta": 0.0,
    # The learning rate of the PRETRAINED parameters as a multiple of lr (EEG
    # C1: channel embedding, its gate, shared transformer). 1.0 is plain
    # fine-tuning; below it, the fresh frontend, patcher and head learn at lr
    # while what pretraining learned moves more slowly.
    "encoder_lr_scale": 1.0,
    # Layer-wise learning-rate decay (BEiT, MAE) over the pretrained Transformer:
    # the top block at lr x decay, each block below at decay more, the channel
    # encoder under block 0 at decay^(depth+1). 1.0 is off. Multiplies
    # encoder_lr_scale. On sEMG, plain fine-tuning at one rate reproduced the
    # from-scratch curve -- the pretrained features were overwritten -- while
    # a frozen probe beat it on cross-subject GRABMyo; this is the remedy.
    "layer_decay": 1.0,
    # Cross-entropy class weights: "none", or "balanced" -- class c weighted
    # N / (K * n_c) over the TRAINING labels (sklearn's "balanced"). For a
    # skewed binary task (P300, about 1 target in 6) the argmax of a plain CE
    # leans to the majority class, which balanced accuracy and kappa punish
    # while AUROC does not see it.
    "class_weight": "none",
}
HPARAM_TYPES = {
    "epochs": int, "batch_size": int, "warmup_epochs": int, "lr": float,
    "weight_decay": float, "warmup_ratio": float, "grad_clip": float,
    "label_smoothing": float, "min_lr_ratio": float, "precision": str,
    "select_by": str, "patience": int, "min_delta": float,
    "encoder_lr_scale": float, "layer_decay": float, "class_weight": str,
}


CLASS_WEIGHTS = ("none", "balanced")


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """N / (K * n_c) per class over ``labels``; a class absent from them is refused."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if (counts == 0).any():
        raise SystemExit(f"class_weight=balanced: class(es) {np.flatnonzero(counts == 0).tolist()} "
                         f"have no training window")
    return torch.as_tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def c1_lr_scale(depth: int, encoder_lr_scale: float = 1.0, layer_decay: float = 1.0,
                pretrained_prefixes: Optional[Tuple[str, ...]] = None):
    """``name -> multiple of lr`` for a C1 downstream model's parameters.

    What pretraining supplied (TRANSFERABLE) learns at ``encoder_lr_scale``
    times lr; the fresh frontend, patcher and head at lr. ``layer_decay``
    below 1 is layer-wise decay: the top Transformer block at ``decay``, each
    block below at ``decay`` more, the final norm above the top block at 1,
    and the channel encoder and gate, which feed block 0, at
    ``decay ** (depth + 1)``. The two factors multiply.

    ``pretrained_prefixes`` is what the checkpoint supplied (the model's
    ``pretrained_prefixes``); on a route it adds the frontend and patcher,
    which then learn like the channel encoder under block 0.
    """
    from physiowave.eeg_c1.downstream import TRANSFERABLE
    s, d = float(encoder_lr_scale), float(layer_decay)
    prefixes = tuple(pretrained_prefixes or TRANSFERABLE)

    def scale(name: str) -> float:
        if not name.startswith(prefixes):
            return 1.0
        f = s
        if d != 1.0:
            m = re.match(r"shared_transformer\.blocks\.(\d+)\.", name)
            if m:
                f *= d ** (depth - int(m.group(1)))
            elif not name.startswith("shared_transformer."):
                f *= d ** (depth + 1)
        return f
    return scale


def resolve_hparams(args, cfg: Dict[str, Any]) -> List[Tuple[str, Any, str]]:
    """Fill the unset training hyper-parameters and say where each came from."""
    tcfg = dict(cfg.get("train", {}) or {})
    rows: List[Tuple[str, Any, str]] = []
    for key, fallback in HPARAM_FALLBACKS.items():
        value = getattr(args, key, None)
        source = "cli"
        if value is None:
            if tcfg.get(key) is not None:
                value, source = tcfg[key], f"config:{args.config}"
            else:
                value, source = fallback, "builtin"
        if value is not None:
            value = HPARAM_TYPES[key](value)
        setattr(args, key, value)
        rows.append((key, value, source))
    if args.select_by not in SELECTION_METRICS:
        raise SystemExit(f"--select-by must be one of {SELECTION_METRICS}, "
                         f"got {args.select_by!r}")
    if args.class_weight not in CLASS_WEIGHTS:
        raise SystemExit(f"--class-weight must be one of {CLASS_WEIGHTS}, "
                         f"got {args.class_weight!r}")
    return rows


def say(message: str, progress_mode: str, is_main: bool = True) -> None:
    """A line for a human, printed without tearing a redrawing bar in half."""
    if not is_main:
        return
    if progress_mode == "bar":
        bar_write(None, message)
    else:
        logger.info("%s", message)


class LabelledWindows(Dataset):
    """The (N, C, T) + (N,) HDF5 the converters write, held in memory.

    Read once into RAM rather than kept open per worker: the downstream files
    are a few GB, the windows are touched in random order every epoch, and an
    open h5py handle is not safe to share across forked workers.
    """

    def __init__(self, path: str) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            self.data = np.asarray(f["data"][:], dtype=np.float32)
            raw_labels = np.asarray(f["label"][:])
            # (N,) int is one class per window; (N, K) is MULTI-LABEL, a 0/1
            # column per class (CPSC 2018, Chapman-Shaoxing), trained with a
            # sigmoid per class rather than one softmax.
            self.multilabel = raw_labels.ndim == 2
            self.labels = (raw_labels.astype(np.float32) if self.multilabel
                           else raw_labels.astype(np.int64))
            # Which recording each window came from, when a recording is cut
            # into several windows (CPSC's 6-60 s records). Evaluation then
            # averages a recording's window probabilities and scores records,
            # which is how the challenge scored them; without it every window
            # is its own sample.
            self.records = ([r.decode() if isinstance(r, bytes) else str(r)
                             for r in f["record"][:]] if "record" in f else None)
            self.class_names = (json.loads(f.attrs["class_names"])
                                if "class_names" in f.attrs else None)
            # The montage is a property of the FILE. Every converter writes it,
            # and a copy typed into a config is a copy that can be wrong -- the
            # first hand-transcribed one had two electrodes the montage does
            # not contain.
            self.channel_names = ([c.decode() if isinstance(c, bytes) else str(c)
                                   for c in f["channel_names"][:]]
                                  if "channel_names" in f else None)
            self.sampling_rate = float(f.attrs.get("sampling_rate", 0.0)) or None
        if len(self.data) != len(self.labels):
            raise ValueError(f"{path}: {len(self.data)} windows but {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        y = (torch.from_numpy(self.labels[i]) if self.multilabel
             else int(self.labels[i]))
        # The index rides along so evaluation can undo DistributedSampler's
        # interleaving and padding, and find each window's recording.
        return torch.from_numpy(self.data[i]), y, i

    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def window(self) -> int:
        return self.data.shape[2]

    def class_counts(self, num_classes: int) -> np.ndarray:
        if self.multilabel:
            return self.labels.sum(axis=0).astype(np.int64)
        return np.bincount(self.labels, minlength=num_classes)


def build_channel_meta(num_channels: int, names: Optional[List[str]],
                       xyz_path: Optional[str], device: torch.device) -> ChannelMeta:
    """Channel metadata for TARE; unknown coordinates are the all-zero row."""
    if names is None:
        names = [f"ch{i:02d}" for i in range(num_channels)]
    if len(names) != num_channels:
        raise ValueError(f"{len(names)} channel names for {num_channels} channels")
    if xyz_path:
        xyz = torch.as_tensor(np.load(xyz_path), dtype=torch.float32)
        if xyz.shape != (num_channels, 3):
            raise ValueError(f"channel_xyz must be ({num_channels}, 3), got {tuple(xyz.shape)}")
    else:
        xyz = torch.zeros(num_channels, 3, dtype=torch.float32)
    return ChannelMeta(channel_names=names, channel_xyz=xyz.to(device))


def forward_logits(model: nn.Module, x: torch.Tensor, meta: ChannelMeta) -> torch.Tensor:
    out = model(x, meta)
    if "logits" not in out:
        raise RuntimeError(
            "the encoder produced no logits; model.num_classes must be set "
            "(pass --num-classes, which this entry point does)")
    return out["logits"]


def _gather(t: torch.Tensor, device) -> torch.Tensor:
    """Concatenate a per-rank tensor over ranks (sizes may differ)."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    sizes = [torch.zeros(1, dtype=torch.long, device=device)
             for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, torch.tensor([len(t)], dtype=torch.long, device=device))
    sizes = [int(x.item()) for x in sizes]
    pad = torch.zeros((max(sizes),) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
    pad[:len(t)] = t.to(device)
    out = [torch.zeros_like(pad) for _ in sizes]
    dist.all_gather(out, pad)
    return torch.cat([o[:n] for o, n in zip(out, sizes)]).cpu()


def by_record(y_prob: np.ndarray, y_true: np.ndarray, records) -> Tuple[np.ndarray, np.ndarray]:
    """One row per recording: the mean of its windows' probabilities."""
    keys = {}
    order = []
    for i, r in enumerate(records):
        if r not in keys:
            keys[r] = []
            order.append(r)
        keys[r].append(i)
    P = np.stack([y_prob[keys[r]].mean(0) for r in order])
    Y = np.stack([y_true[keys[r][0]] for r in order])
    return P, Y


def multiclass_metrics(y_prob: np.ndarray, y_true: np.ndarray,
                       num_classes: int) -> Dict[str, float]:
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                                 roc_auc_score)
    y_pred = y_prob.argmax(1)
    present = np.unique(y_true)
    metrics = {
        "acc": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                       if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                    if len(y_true) else 0.0,
    }
    # AUROC is one-vs-rest over the classes that actually occur. Dropping the
    # absent columns leaves rows that no longer sum to one, which sklearn
    # rejects, so they are renormalised -- without this a single unrepresented
    # class turns the whole metric into nan, which on a 53-class split is the
    # common case rather than the corner one.
    try:
        if len(present) >= 2:
            sub = y_prob[:, present]
            sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
            if len(present) == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, sub[:, 1],
                                                       labels=present))
            else:
                metrics["auroc"] = float(roc_auc_score(
                    y_true, sub, multi_class="ovr", average="macro", labels=present))
        else:
            metrics["auroc"] = float("nan")
    except ValueError as exc:                      # degenerate split
        logger.warning("AUROC unavailable: %s", exc)
        metrics["auroc"] = float("nan")
    if len(present) < num_classes:
        logger.warning("%d of %d classes absent from this split; AUROC is over the "
                       "%d present", num_classes - len(present), num_classes, len(present))
    return metrics


def multilabel_metrics(y_prob: np.ndarray, y_true: np.ndarray,
                       thresholds: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Per-class sigmoid scores against 0/1 columns.

    ``auroc`` is the MACRO AUROC over the classes with both outcomes present,
    so the selection metric means the same thing on every task. F1 needs a
    threshold: 0.5 always, and ``thresholds`` (one per class, chosen on the
    validation split) when given, reported as ``*_tuned``.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    y_true = (y_true > 0.5).astype(np.int64)
    ok = [k for k in range(y_true.shape[1]) if 0 < y_true[:, k].sum() < len(y_true)]
    m: Dict[str, float] = {}
    if ok:
        m["auroc"] = float(np.mean([roc_auc_score(y_true[:, k], y_prob[:, k]) for k in ok]))
        m["micro_auroc"] = float(roc_auc_score(y_true[:, ok].ravel(), y_prob[:, ok].ravel()))
        m["macro_auprc"] = float(np.mean([average_precision_score(y_true[:, k], y_prob[:, k])
                                          for k in ok]))
    else:
        m["auroc"] = m["micro_auroc"] = m["macro_auprc"] = float("nan")
    if len(ok) < y_true.shape[1]:
        logger.warning("%d of %d labels have only one outcome in this split; macro "
                       "AUROC is over the %d that have both",
                       y_true.shape[1] - len(ok), y_true.shape[1], len(ok))

    def thresholded(th, suffix):
        pred = (y_prob >= th).astype(np.int64)
        m[f"macro_f1{suffix}"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        m[f"micro_f1{suffix}"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
        m[f"weighted_f1{suffix}"] = float(f1_score(y_true, pred, average="weighted",
                                                   zero_division=0))
        m[f"acc{suffix}"] = float((pred == y_true).all(axis=1).mean())    # exact match
        m[f"hamming{suffix}"] = float((pred != y_true).mean())

    thresholded(0.5, "")
    if thresholds is not None:
        thresholded(np.asarray(thresholds)[None, :], "_tuned")
    # Not defined for multi-label; present so a line that prints them does not fail.
    m["balanced_acc"] = m["kappa"] = float("nan")
    return m


def tune_thresholds(y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per class, the threshold maximising F1 on THIS split (use: validation)."""
    from sklearn.metrics import f1_score
    y_true = (y_true > 0.5).astype(np.int64)
    grid = np.linspace(0.05, 0.95, 19)
    out = np.full(y_true.shape[1], 0.5)
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        f1s = [f1_score(y_true[:, k], (y_prob[:, k] >= t).astype(int), zero_division=0)
               for t in grid]
        out[k] = float(grid[int(np.argmax(f1s))])
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, meta: ChannelMeta, device: torch.device,
             amp_dtype: torch.dtype, criterion: nn.Module, num_classes: int,
             desc: str = "eval", progress_mode: str = "none",
             is_main: bool = False, thresholds: Optional[np.ndarray] = None,
             return_outputs: bool = False):
    """Metrics over the whole split, gathered across ranks.

    When the split records which recording each window came from, the
    headline metrics are per RECORDING (window probabilities averaged) and the
    per-window ones are kept under ``window_*``.
    """
    model.eval()
    ds = loader.dataset
    multilabel = bool(getattr(ds, "multilabel", False))
    loss_meter = AverageMeter()
    probs_all, target_all, idx_all = [], [], []
    for x, y, idx in progress(loader, desc, progress_mode, is_main):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp_dtype):
            logits = forward_logits(model, x, meta)
            loss = criterion(logits.float(), y)
        loss_meter.update(loss.item(), len(y))
        probs_all.append((torch.sigmoid(logits.float()) if multilabel
                          else F.softmax(logits.float(), dim=-1)).cpu())
        target_all.append(y.cpu())
        idx_all.append(torch.as_tensor(idx).cpu())

    probs = torch.cat(probs_all) if probs_all else torch.zeros(0, num_classes)
    target = torch.cat(target_all) if target_all else torch.zeros(0, dtype=torch.long)
    index = torch.cat(idx_all) if idx_all else torch.zeros(0, dtype=torch.long)
    # Every rank sees a disjoint shard; the metrics below are not averages of
    # per-shard metrics, so the predictions have to be gathered before
    # anything is computed. DistributedSampler pads the last shard with
    # repeats; the window index removes them.
    probs, target, index = (_gather(probs, device), _gather(target, device),
                            _gather(index.long(), device))
    _, first = np.unique(index.numpy(), return_index=True)
    y_prob, y_true, index = probs.numpy()[first], target.numpy()[first], index.numpy()[first]

    score = (lambda P, Y: multilabel_metrics(P, Y, thresholds)) if multilabel else \
            (lambda P, Y: multiclass_metrics(P, Y, num_classes))
    metrics: Dict[str, float] = {"loss": loss_meter.avg}
    records = getattr(ds, "records", None)
    if records is not None:
        win = score(y_prob, y_true)
        metrics.update({f"window_{k}": v for k, v in win.items()})
        y_prob, y_true = by_record(y_prob, y_true, [records[i] for i in index])
        metrics["n_records"] = int(len(y_true))
    metrics.update(score(y_prob, y_true))
    if return_outputs:
        return metrics, y_prob, y_true
    return metrics


def train_one_epoch(model, loader, meta, device, amp_dtype, criterion, optimizer, scheduler,
                    scaler, grad_clip: float, epoch: int, log_every: int,
                    is_main: bool, progress_mode: str = "auto") -> Dict[str, float]:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    bar = progress(loader, f"train {epoch}", progress_mode, is_main)
    total = len(loader)
    started = time.monotonic()
    for step, (x, y, _idx) in enumerate(bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp_dtype):
            logits = forward_logits(model, x, meta)
            loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), len(y))
        if y.dim() == 2:      # multi-label: exact match at 0.5
            hit = ((logits.detach() > 0).float() == y).all(dim=1).float().mean()
        else:
            hit = (logits.argmax(-1) == y).float().mean()
        acc_meter.update(float(hit.item()), len(y))
        set_postfix(bar, loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if is_main and progress_mode == "log" and log_every and step % log_every == 0:
            # The percentage and ETA are the part a bar would have given; they
            # are what makes a periodic line answer "how much longer" in a log
            # file, where a redrawing bar cannot go.
            done = step + 1
            eta = (time.monotonic() - started) / done * (total - done)
            logger.info("epoch %d step %d/%d (%3.0f%%) loss %.4f acc %.4f lr %.2e eta %s",
                        epoch, step, total, 100.0 * done / max(total, 1),
                        loss_meter.avg, acc_meter.avg,
                        optimizer.param_groups[0]["lr"], fmt_eta(eta))
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


RULE = "\u2500" * 78


def render_test_block(metrics: Dict[str, float], args, run_name: str) -> str:
    """The test row as something readable, with the published row beside it.

    A bar per metric, drawn from the metric's own floor rather than from zero:
    a kappa of 0.30 and an AUROC of 0.72 are similar results and look nothing
    alike when both are drawn from 0. And the published row is printed with the
    reason it is not directly comparable, every time, because a table that
    shows the two side by side without that line is a table someone will quote.
    """
    task = task_of(args.config) or task_of(run_name)
    shown = TASK_METRICS.get(task, (("balanced_acc", "BalAcc"), ("kappa", "Kappa"),
                                    ("auroc", "AUROC"), ("weighted_f1", "W-F1")))
    if getattr(args, "_multilabel", False):
        shown = (("micro_f1_tuned", "F1 mic"), ("macro_f1_tuned", "F1 mac"),
                 ("auroc", "AUROC"), ("micro_auroc", "AUROCu"),
                 ("micro_f1", "F1u@.5"), ("macro_f1", "F1m@.5"))
    ref = PUBLISHED.get(task, {}).get(EEGPT, {})

    mode = "linear probe" if args.freeze_encoder else "full fine-tune"
    init = "pretrained" if getattr(args, "_pretrained", False) else "from scratch"
    trainable = getattr(args, "_trainable_params", None)
    header = f"  {run_name}  \u00b7  TEST  \u00b7  {args.config}  \u00b7  {mode}  \u00b7  {init}"
    if trainable is not None:
        header += f"  \u00b7  {trainable:,} trainable"
    out = [RULE, header, RULE]
    for key, label in shown:
        value = metrics.get(key, float("nan"))
        if value != value:                       # nan: not defined here
            continue
        floor = CHANCE.get(key)
        lo = floor if floor is not None else 0.0
        line = f"  {label:<7} {value:>7.4f}  {sparkbar(value, 22, lo=lo)}"
        line += f"  chance {floor:.2f}" if floor is not None else " " * 13
        if key in ref:
            line += f"   EEGPT {ref[key]:.4f}  {value - ref[key]:+.4f}"
        out.append(line)
    rest = [f"{k} {metrics[k]:.4f}" for k in ("acc", "weighted_f1", "loss")
            if k in metrics and k not in dict(shown)]
    if rest:
        out.append("  " + "   ".join(rest))
    if ref:
        out += [RULE,
                "  EEGPT's row is a fold-averaged score on the split it also monitors,",
                "  with no held-out test set. This row is one subject-disjoint split,",
                "  scored on a test set nothing selected on -- the harder of the two",
                "  numbers. Say which is which wherever they appear together."]
    out.append(RULE)
    return "\n".join(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pretrain/semg",
                   help="encoder config; the model block is reused, the pretraining "
                        "objectives are not")
    p.add_argument("--data-dir", help="directory holding train.h5 / val.h5 / test.h5")
    p.add_argument("--train-file")
    p.add_argument("--val-file")
    p.add_argument("--test-file")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pretrained", help="extension-format checkpoint to start from")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train the classification head only (linear probe)")
    p.add_argument("--channel-names", nargs="+", default=None)
    p.add_argument("--channel-xyz", default=None,
                   help=".npy of shape (C, 3); without it TARE sees unknown coordinates")
    # default=None on every one of these, so "not given" is distinguishable
    # from "given the same value the default happened to have" and the
    # config's train: block gets its turn. resolve_hparams applies the order.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="warmup as a fraction of the whole run, which is how "
                        "OneCycle's pct_start is written; overrides "
                        "--warmup-epochs. EEGPT uses 0.2 downstream.")
    p.add_argument("--min-lr-ratio", type=float, default=None,
                   help="floor of the cosine decay, as a fraction of --lr")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--layer-decay", type=float, default=None,
                   help="layer-wise lr decay over the pretrained Transformer "
                        "(train.layer_decay; 1.0 is off)")
    p.add_argument("--encoder-lr-scale", type=float, default=None,
                   help="learning rate of the pretrained encoder parameters as a "
                        "fraction of --lr (EEG C1 only); default 1.0")
    p.add_argument("--label-smoothing", type=float, default=None,
                   help="0.0 to match EEGPT, which trains a plain "
                        "CrossEntropyLoss. Smoothing puts a floor under the "
                        "loss (0.20 for 2 classes, 0.39 for 5) that reads as "
                        "a run that stopped improving.")
    p.add_argument("--class-weight", default=None, choices=CLASS_WEIGHTS,
                   help="'balanced' weights the cross-entropy by N / (K * n_c) "
                        "over the training labels (train.class_weight); default none")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"],
                   help="override the automatic pick; useful when the default "
                        "accelerator is the thing under suspicion")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50,
                   help="periodic in-epoch log lines; 0 disables. Ignored while a "
                        "progress bar is showing the same numbers.")
    p.add_argument("--progress", default="auto", choices=list(PROGRESS_MODES),
                   help="'auto' shows a tqdm bar on a terminal and periodic log lines "
                        "with a percentage and an ETA when the stream is redirected, "
                        "where a redrawing bar would become one enormous line")
    p.add_argument("--select-by", default=None, choices=SELECTION_METRICS,
                   help="validation metric that decides best.pth. EEGPT monitors "
                        "AUROC for binary tasks and Cohen's kappa for multi-class; "
                        "the finetune configs set the matching one.")
    p.add_argument("--patience", type=int, default=None,
                   help="stop when --select-by has not improved for this many "
                        "epochs. 0, the default, never stops early -- the run "
                        "goes to --epochs. The resolved value is logged.")
    p.add_argument("--min-delta", type=float, default=None,
                   help="how much better counts as an improvement, for --patience")
    # 'extend' rather than the default 'store': the launch scripts pass a --set of
    # their own and EXTRA can carry another, and with 'store' the second silently
    # replaces the first -- an override that looks applied and is not.
    # default=None because argparse mutates a shared list default in place.
    p.add_argument("--set", nargs="*", action="extend", default=None,
                   help="config overrides, key=value; repeatable")
    args = p.parse_args(argv)
    args.set = args.set or []
    return args


def resolve_files(args) -> Tuple[str, Optional[str], Optional[str]]:
    if args.data_dir:
        j = lambda n: os.path.join(args.data_dir, f"{n}.h5")  # noqa: E731
        train = args.train_file or j("train")
        val = args.val_file or (j("val") if os.path.exists(j("val")) else None)
        test = args.test_file or (j("test") if os.path.exists(j("test")) else None)
    else:
        train, val, test = args.train_file, args.val_file, args.test_file
    if not train or not os.path.exists(train):
        raise SystemExit(f"ERROR: training file not found: {train}")
    return train, val, test


def main(argv=None) -> int:
    args = parse_args(argv)
    info = setup_distributed()
    device = torch.device(args.device) if args.device else pick_device()
    set_seed(args.seed + info.rank)
    logging.basicConfig(level=logging.INFO if info.is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.progress = resolve_progress(args.progress)

    train_path, val_path, test_path = resolve_files(args)
    cfg = load_config(args.config, args.set)
    hparams = resolve_hparams(args, cfg)
    if info.is_main:
        logger.info("hyper-parameters (cli > config train: > builtin):")
        for key, value, source in hparams:
            logger.info("    %-16s %-10s  [%s]", key, value, source)
    model_cfg = dict(cfg.get("model", {}) or {})
    model_cfg["num_classes"] = args.num_classes
    cfg["model"] = model_cfg

    if info.is_main:
        # Say this before the read, not after. LabelledWindows pulls the whole
        # file into RAM and every rank holds its own copy, so a few GB of
        # densely-strided windows is minutes of complete silence otherwise --
        # indistinguishable from a hung job, which is how an allocation gets
        # killed and rerun for no reason.
        sizes = " ".join(
            f"{os.path.basename(p)} {os.path.getsize(p) / 2**30:.2f} GiB"
            for p in (train_path, val_path, test_path)
            if p and os.path.exists(p))
        logger.info("loading %s into memory on each of %d rank(s): %s",
                    os.path.dirname(train_path) or ".", info.world_size, sizes)

    train_set = LabelledWindows(train_path)
    C, T = train_set.num_channels, train_set.window
    multilabel = train_set.multilabel
    args._multilabel = multilabel
    if multilabel and train_set.labels.shape[1] != args.num_classes:
        raise SystemExit(f"{train_path} has {train_set.labels.shape[1]} label columns "
                         f"and --num-classes is {args.num_classes}")
    if info.is_main:
        counts = train_set.class_counts(args.num_classes)
        logger.info("train %d windows, %d channels, %d samples/window", len(train_set), C, T)
        logger.info("class counts: %s", counts.tolist())
        if (counts == 0).any():
            logger.warning("classes %s have no training windows",
                           np.flatnonzero(counts == 0).tolist())

    c1_name = model_cfg.get("name") if model_cfg.get("name") in ("eeg_c1", "ecg_c1", "emg_c1") else None
    if c1_name:
        # The montage, the window and the rate come from the FILE unless the
        # config names them. They are facts about the data, and a second copy
        # in a config is one that can disagree with it silently.
        c1 = dict(model_cfg.get(c1_name, {}) or {})
        # Through the model, not only through requires_grad. EEGC1Downstream
        # overrides train() on this flag and holds the encoder in eval mode, so
        # the probe reads the representation the encoder actually produces
        # rather than a fresh dropout sample of it every step.
        if args.freeze_encoder:
            c1["freeze_encoder"] = True
        c1.setdefault("in_channels", C)
        c1.setdefault("window_samples", T)
        if train_set.channel_names and "channel_names" not in c1:
            c1["channel_names"] = train_set.channel_names
        if train_set.sampling_rate and "sampling_rate" not in c1:
            c1["sampling_rate"] = train_set.sampling_rate
        elif train_set.sampling_rate and "sampling_rate" in c1:
            # Both have one. The config wins by construction above, so a
            # disagreement would be silent -- and a rate that is wrong by a
            # factor of 2.56 makes every patch a different duration than the
            # config claims while nothing anywhere fails.
            if abs(float(c1["sampling_rate"]) - train_set.sampling_rate) > 1e-6:
                raise SystemExit(
                    f"the config says model.{c1_name}.sampling_rate="
                    f"{c1['sampling_rate']} and {train_path} says "
                    f"{train_set.sampling_rate}. One of them is wrong and "
                    f"guessing which would change the duration of every patch.")
        missing = [k for k in ("sampling_rate", "patch_samples") if k not in c1]
        if missing:
            raise SystemExit(
                f"model.{c1_name} needs {missing} and neither the config nor "
                f"{train_path} supplies them.\n"
                f"  A patch length is a modelling choice and belongs in the "
                f"config. A sampling rate should be an attribute of the file, "
                f"and is written by the converters -- but only since the fix "
                f"that made it unconditional, so a file built before that has "
                f"none.\n"
                f"  Either re-run the converter's --stage split, or state it: "
                f"--set model.{c1_name}.sampling_rate=<Hz>")
        model_cfg[c1_name] = c1
        cfg["model"] = model_cfg
        if info.is_main:
            logger.info("%s C1 downstream: %d channels at %s Hz, %d-sample "
                        "windows, %d-sample patches, route %s", c1_name[:3].upper(),
                        c1["in_channels"], c1["sampling_rate"],
                        c1["window_samples"], c1["patch_samples"],
                        c1.get("route_id") or "<its own frontend>")

    model = build_model(cfg).to(device)
    if args.pretrained:
        # strict=False on purpose: the classification head does not exist in a
        # pretraining checkpoint, and an encoder trained on a different channel
        # count keeps everything except the per-channel pieces.
        payload = load_checkpoint(args.pretrained, model=model, strict=False,
                                  restore_rng=False)
        if info.is_main:
            logger.info("loaded encoder from %s (epoch %s)", args.pretrained,
                        payload.get("epoch", "?"))
    if args.freeze_encoder and not getattr(model, "_frozen", False):
        # The fallback, for architectures that do not take freeze_encoder
        # themselves. EEGC1Downstream does, and it knows that its adaptive
        # spatial filter is an adapter between montages rather than part of the
        # representation -- freezing that too would measure the encoder's
        # response to the wrong electrode gains. Re-running this loop over it
        # would undo exactly that.
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
    if info.distributed:
        model = DDP(model, device_ids=[info.local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=True)
    core = model.module if hasattr(model, "module") else model
    # Hoisted out of the logging block: the trainable count belongs in
    # results.json as well as in the log. A probe's score is only evidence
    # about the encoder to the extent that its head is small, and "small" is a
    # number a reader has to be able to see next to the score.
    n_total = sum(p.numel() for p in core.parameters())
    n_trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if info.is_main:
        # State the block that is actually running. The components are dataclass
        # defaults rather than YAML entries, so reading the configs does not tell
        # you which ones are on, and an ablation that silently kept the defaults
        # would be indistinguishable from one that took effect.
        bb = getattr(getattr(core, "cfg", None), "backbone", None)
        if bb is not None:
            logger.info("backbone: depth=%d dim=%d | rope=%s norm=%s ffn=%s qk_norm=%s",
                        bb.depth, bb.embed_dim, bb.use_rope, bb.norm, bb.ffn, bb.qk_norm)
        # Raw counts as well as millions: a linear probe leaves a few thousand
        # parameters trainable, which "0.00 M" reads as "none at all".
        logger.info("model %.2f M parameters (%s trainable, %.1f%%)",
                    n_total / 1e6, f"{n_trainable:,}",
                    100.0 * n_trainable / max(n_total, 1))

    meta = build_channel_meta(C, args.channel_names, args.channel_xyz, device)

    def loader_for(dataset, shuffle):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if info.distributed else None
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                          shuffle=(sampler is None and shuffle),
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          drop_last=False), sampler

    train_loader, train_sampler = loader_for(train_set, True)
    val_loader = test_loader = None
    if val_path and os.path.exists(val_path):
        val_loader, _ = loader_for(LabelledWindows(val_path), False)
    if test_path and os.path.exists(test_path):
        test_loader, _ = loader_for(LabelledWindows(test_path), False)

    precision, amp_dtype = resolve_precision(args.precision, device)
    scaler = make_grad_scaler(device.type, precision == "fp16")
    if multilabel:
        if args.class_weight != "none":
            raise SystemExit("class_weight applies to single-label cross-entropy; "
                             "this split is multi-label")
        criterion = nn.BCEWithLogitsLoss()
        if info.is_main:
            logger.info("multi-label: %d sigmoid outputs, BCE; label smoothing "
                        "does not apply", args.num_classes)
    else:
        weight = None
        if args.class_weight == "balanced":
            weight = balanced_class_weights(train_set.labels, args.num_classes).to(device)
            if info.is_main:
                logger.info("class weights (balanced, from the training labels): %s",
                            [round(float(w), 4) for w in weight])
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    lr_scale = None
    scaled = args.encoder_lr_scale != 1.0 or args.layer_decay != 1.0
    has_pretrained = bool(args.pretrained
                          or (model_cfg.get(c1_name or "eeg_c1") or {}).get("pretrained"))
    if scaled and not has_pretrained:
        # Both scales protect what pretraining supplied. From scratch there is
        # nothing to protect, and the control should learn at one rate.
        if info.is_main:
            logger.info("encoder_lr_scale/layer_decay ignored: no pretrained "
                        "encoder loaded, every parameter learns at lr")
        scaled = False
    if scaled:
        if not c1_name:
            raise SystemExit("encoder_lr_scale and layer_decay are defined for "
                             "model.name=eeg_c1/ecg_c1/emg_c1 only: they scale "
                             "what that model loads from pretraining")
        blocks = getattr(getattr(core, "shared_transformer", None), "blocks", None)
        if blocks is None:
            raise SystemExit("layer_decay needs the model's shared_transformer.blocks")
        lr_scale = c1_lr_scale(len(blocks), args.encoder_lr_scale, args.layer_decay,
                               getattr(core, "pretrained_prefixes", None))
    optimizer = build_optimizer(core, args.lr, args.weight_decay, lr_scale=lr_scale)
    if info.is_main and lr_scale is not None:
        # Every distinct rate with how many parameters learn at it, highest
        # first: the fresh parts at lr, then the Transformer top to bottom.
        at = {}
        for n, p in core.named_parameters():
            if p.requires_grad:
                k = round(lr_scale(n), 6)
                at[k] = at.get(k, 0) + p.numel()
        logger.info("learning rates (encoder_lr_scale %g, layer_decay %g): %s",
                    args.encoder_lr_scale, args.layer_decay,
                    ", ".join(f"{args.lr * k:.2e} on {v:,}"
                              for k, v in sorted(at.items(), reverse=True)))
    steps = max(len(train_loader), 1)
    total_steps = args.epochs * steps
    if args.warmup_ratio is not None:
        # OneCycle's pct_start, which is how EEGPT writes it (0.2 on every
        # downstream task). Expressed in steps it survives a change of epoch
        # count, where "2 warmup epochs" silently becomes 20% of a 10-epoch run
        # and 5% of a 40-epoch one.
        warmup_steps = int(round(args.warmup_ratio * total_steps))
    else:
        warmup_steps = args.warmup_epochs * steps
    if warmup_steps >= total_steps:
        warmup_steps = max(total_steps // 10, 0)
        if info.is_main:
            logger.warning("warmup covers the whole run; using %d steps (10%%) "
                           "so the decay happens", warmup_steps)
    if info.is_main:
        logger.info("schedule: %d steps/epoch, %d total, %d warmup (%.0f%%), "
                    "lr %.2e -> %.2e", steps, total_steps, warmup_steps,
                    100.0 * warmup_steps / max(total_steps, 1), args.lr,
                    args.lr * args.min_lr_ratio)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps,
                                args.min_lr_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_score, best_epoch, stale = float("-inf"), -1, 0

    if info.is_main and val_loader is not None:
        logger.info("selection: best.pth follows val %s%s", args.select_by,
                    f"; early stop after {args.patience} epochs without an "
                    f"improvement of >{args.min_delta:g}" if args.patience > 0
                    else "; no early stop, so this runs all "
                         f"{args.epochs} epochs (--patience N to stop sooner)")
    run_started = time.monotonic()
    outer = epoch_bar(args.epochs, args.progress, info.is_main)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, meta, device, amp_dtype, criterion,
                             optimizer, scheduler, scaler, args.grad_clip, epoch,
                             args.log_every, info.is_main, args.progress)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}}

        if val_loader is not None:
            va = evaluate(model, val_loader, meta, device, amp_dtype, criterion,
                          args.num_classes, f"val {epoch}", args.progress, info.is_main)
            row.update({f"val_{k}": v for k, v in va.items()})
            score = -va["loss"] if args.select_by == "loss" else va[args.select_by]
            improved = score > best_score + args.min_delta
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
                if info.is_main:
                    save_checkpoint(os.path.join(args.output_dir, "best.pth"), core,
                                    optimizer, epoch=epoch, metrics=va,
                                    config={"args": vars(args), "config": args.config})
            else:
                stale += 1
            if info.is_main:
                done = epoch + 1
                eta = (time.monotonic() - run_started) / done * (args.epochs - done)
                head = (f"epoch {epoch:>3}/{args.epochs - 1}  "
                        f"train_loss {tr['loss']:.4f}  val_loss {va['loss']:.4f}  ")
                body = (f"auroc {va['auroc']:.4f}  microF1 {va['micro_f1']:.4f}  "
                        f"macroF1 {va['macro_f1']:.4f}  auprc {va['macro_auprc']:.4f}  "
                        if multilabel else
                        f"acc {va['acc']:.4f}  bal {va['balanced_acc']:.4f}  "
                        f"kappa {va['kappa']:.4f}  auroc {va['auroc']:.4f}  ")
                say(head + body + f"eta {fmt_eta(eta)}" + ("  *best" if improved else ""),
                    args.progress)
                set_postfix_str(outer, f"best {args.select_by} {best_score:.4f} "
                                       f"@ep{best_epoch}")
        if outer is not None:
            outer.update(1)
        history.append(row)
        if info.is_main:
            with open(os.path.join(args.output_dir, "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

        # Early stop: rank 0 decides, every rank obeys. A rank that leaves the
        # loop alone strands the others in the next all-reduce.
        if args.patience > 0 and val_loader is not None:
            flag = torch.zeros(1, device=device)
            if info.is_main and stale >= args.patience:
                logger.info("early stop at epoch %d: val %s flat for %d epochs (best %d)",
                            epoch, args.select_by, args.patience, best_epoch)
                flag[0] = 1.0
            if info.distributed:
                dist.broadcast(flag, src=0)
            if flag.item() > 0:
                break

    close_bar(outer)
    results: Dict[str, Any] = {"best_epoch": best_epoch, "select_by": args.select_by,
                               "best_val": best_score if best_epoch >= 0 else None,
                               "hparams": {k: v for k, v, _ in hparams},
                               "frozen_encoder": bool(args.freeze_encoder),
                               "total_params": n_total,
                               "trainable_params": n_trainable,
                               "multilabel": multilabel,
                               "class_names": train_set.class_names,
                               "pretrained": bool(
                                   args.pretrained
                                   or (model_cfg.get(c1_name or "eeg_c1") or {}).get("pretrained"))}
    if test_loader is not None:
        best = os.path.join(args.output_dir, "best.pth")
        if os.path.exists(best):
            load_checkpoint(best, model=core, strict=False, restore_rng=False)
            if info.distributed:
                for p in core.parameters():
                    dist.broadcast(p.data, src=0)
        thresholds = None
        if multilabel and val_loader is not None:
            # F1 needs a threshold per class. Chosen on VALIDATION with the
            # selected checkpoint and applied unchanged to test; the 0.5 row is
            # reported beside it.
            _, vp, vy = evaluate(model, val_loader, meta, device, amp_dtype, criterion,
                                 args.num_classes, "val (thresholds)", args.progress,
                                 info.is_main, return_outputs=True)
            thresholds = tune_thresholds(vp, vy)
            results["thresholds"] = thresholds.tolist()
        te = evaluate(model, test_loader, meta, device, amp_dtype, criterion,
                      args.num_classes, "test", args.progress, info.is_main,
                      thresholds=thresholds)
        results["test"] = te
        if info.is_main:
            # The grep target stays exactly as it was -- scripts and habits
            # depend on a line starting with TEST -- and the block under it is
            # for the human reading the tail of a log.
            if multilabel:
                logger.info("TEST macroAUROC %.4f microAUROC %.4f microF1 %.4f "
                            "macroF1 %.4f (tuned thresholds: micro %.4f macro %.4f)",
                            te["auroc"], te["micro_auroc"], te["micro_f1"],
                            te["macro_f1"], te.get("micro_f1_tuned", float("nan")),
                            te.get("macro_f1_tuned", float("nan")))
            else:
                logger.info("TEST acc %.4f bal %.4f kappa %.4f wf1 %.4f auroc %.4f",
                            te["acc"], te["balanced_acc"], te["kappa"], te["weighted_f1"],
                            te["auroc"])
            args._pretrained = results["pretrained"]
            args._trainable_params = n_trainable
            print(render_test_block(te, args, os.path.basename(
                os.path.normpath(args.output_dir))), flush=True)
    if info.is_main:
        with open(os.path.join(args.output_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("outputs in %s", args.output_dir)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
