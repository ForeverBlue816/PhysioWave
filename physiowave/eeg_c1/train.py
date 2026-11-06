"""
Training loop for the EEG C1 multi-route pretrainer.

The objective:

    loss = spec_weight * masked MSE on the DETACHED clean folded-wavelet
           patches, normalised per patch
         + raw_weight  * masked SmoothL1 on the DETACHED preprocessed EEG
           patches
         + fold_kl     * ScaleFold KL

    at spec_weight = raw_weight = 0.5, resolved from the config by
    ``physiowave.eeg_c1.objective.resolve_eeg_c1_objective`` -- the one place
    those numbers are read, so the banner, the figures and the progress report
    cannot each quote a different default.

    Under mask_before_frontend the masked patches are zeroed in the SIGNAL
    before the wavelet frontend runs, so nothing inside a masked patch can
    reach a visible token through the frontend.

and nothing else. No reference consistency, no contrastive term, no query
specialisation -- those belong to the WAST/TARE path, which this one does not
touch.

Four things here are less obvious than the loop:

**The two losses are not comparable in magnitude.** An MSE on normalised
wavelet coefficients and a SmoothL1 on z-scored volts are different units with
a different penalty shape; the raw term is the smaller number for reasons that
have nothing to do with which head is doing better. The comparison is
``masked_{spec,raw}_corr`` and ``masked_{spec,raw}_nmse``, both dimensionless,
both reported globally and per route and per dataset.

**Validation metrics are aggregated across ranks, breakdowns included.** The
sweep is partitioned, so a per-route mean computed on one rank is a mean over
that rank's slice. Sums and counts are all-reduced and the mean taken from the
totals.

**Validation masks are fixed.** A validation loss computed under a fresh random
mask each epoch moves because the mask moved, and the curve then measures the
sampler. Each validation batch draws its mask from a generator seeded by
(mask seed, dataset, first window index), so epoch 12 masks exactly what epoch 1
masked and a change in the curve is a change in the model.

**Resume restores the sampling sequence, not just the weights.** The schedule is
a pure function of (seed, epoch), and the step within the epoch is checkpointed,
so a resumed run draws the batches the interrupted one would have drawn next.
Restoring the model and letting the sampler start over would quietly re-train on
the first half of the epoch and skip the second.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from channel_embedding import save_vocab, vocab_payload
from .data import (DEFAULT_BATCH_BY_ROUTE, CorpusIndex, EEGWindowDataset,
                   RouteBatchLoader, RouteSchedule, collate_windows)
from .model import MultiRouteEEGPretrainer, masked_reconstruction_loss
from .objective import (objective_banner, objective_equation,
                        resolve_eeg_c1_objective, resume_incompatibilities,
                        resume_refusal_message)
from .routes import PRETRAIN_DATASETS, ROUTES, datasets_with
from ..train.utils import (fmt_eta, make_grad_scaler, progress,
                           resolve_progress, set_postfix)


# --------------------------------------------------------------------------- #
# Deterministic validation
# --------------------------------------------------------------------------- #

def _mask_generator(seed: int, dataset_id: str, first_index: int) -> torch.Generator:
    """A CPU generator fixed by the sample identity, not by the epoch."""
    g = torch.Generator()
    h = (seed * 1_000_003
         + (abs(hash(dataset_id)) % 1_000_003) * 1009
         + int(first_index))
    g.manual_seed(h % (2 ** 63 - 1))
    return g


class ValIterator:
    """Every validation window once, in file order, in route-pure batches.

    Not the training schedule: validation is not a mixture to be sampled, it is
    a fixed set to be swept, and sweeping it in a fixed order is what lets the
    mask seed depend on the window's identity.
    """

    def __init__(self, index: CorpusIndex, batch_by_route: Dict[str, int],
                 num_replicas: int = 1, rank: int = 0,
                 max_batches_per_dataset: Optional[int] = None,
                 routes: Optional[Dict] = None):
        self.datasets = {d: EEGWindowDataset(index, d, routes=routes)
                         for d in sorted(index.by_dataset())}
        self.batch_by_route = dict(batch_by_route)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.max_batches = max_batches_per_dataset

    def __len__(self) -> int:
        """Exact, so a progress bar over validation can show a total.

        Mirrors __iter__'s bounds rather than estimating them: the stride keeps
        every rank's slice full, so the inner short-batch break never fires and
        the count is the number of starts.
        """
        total = 0
        for ds in self.datasets.values():
            stride = self.batch_by_route[ds.route_id] * self.num_replicas
            emitted = len(range(0, max(0, len(ds) - stride + 1), stride))
            if self.max_batches:
                emitted = min(emitted, self.max_batches)
            total += emitted
        return total

    def __iter__(self):
        for dataset_id, ds in self.datasets.items():
            bs = self.batch_by_route[ds.route_id]
            stride = bs * self.num_replicas
            n = len(ds)
            emitted = 0
            for start in range(0, n - stride + 1, stride):
                lo = start + self.rank * bs
                idx = list(range(lo, min(lo + bs, n)))
                if len(idx) < bs:
                    break
                batch = collate_windows([ds[i] for i in idx])
                batch["channel_meta"] = ds.montage()
                batch["mask_seed_index"] = start
                yield batch
                emitted += 1
                if self.max_batches and emitted >= self.max_batches:
                    break

    def close(self):
        for ds in self.datasets.values():
            ds.close()


# --------------------------------------------------------------------------- #
# Metric accumulation
# --------------------------------------------------------------------------- #

#: Bin edges for the reconstruction-error histograms, log-spaced over |error|.
#: A histogram rather than per-batch quantiles because counts add exactly and
#: averaged quantiles do not: the mean of per-batch medians is not the median.
ERROR_BIN_EDGES = [0.0] + [10.0 ** (-3 + 0.125 * i) for i in range(41)]


class ErrorHistogram:
    """Counts of |prediction - target| per bin, split masked vs visible.

    The visible half is the control. A masked-reconstruction objective can be
    satisfied by a model that has learned to copy its input, and the way that
    shows is a visible error near zero while the masked error stays high. There
    is no way to see it from the loss, which only ever looks at masked tokens.
    """

    KEYS = ("spec_masked", "spec_visible", "raw_masked", "raw_visible")

    def __init__(self):
        n = len(ERROR_BIN_EDGES) - 1
        self.counts = {k: [0] * n for k in self.KEYS}
        self.sums = {k: 0.0 for k in self.KEYS}
        self.n = {k: 0 for k in self.KEYS}

    @torch.no_grad()
    def add(self, out: Dict):
        edges = torch.tensor(ERROR_BIN_EDGES, device=out["mask"].device)
        mask = out["mask"]
        for tag, pk, tk in (("spec", "pred_spec", "target_spec"),
                            ("raw", "pred_raw", "target_raw")):
            pred, target = out.get(pk), out.get(tk)
            if pred is None or target is None:
                continue
            err = (pred.float() - target.float()).abs()
            sel = mask.unsqueeze(-1).expand_as(err)
            valid = out.get("valid_tokens")
            if valid is not None:
                ok = valid.unsqueeze(-1).expand_as(err)
            else:
                ok = torch.ones_like(sel)
            for name, take in ((f"{tag}_masked", sel & ok),
                               (f"{tag}_visible", (~sel) & ok)):
                v = err[take]
                if v.numel() == 0:
                    continue
                idx = torch.bucketize(v, edges, right=False).clamp_(
                    1, len(ERROR_BIN_EDGES) - 1) - 1
                hist = torch.bincount(idx,
                                      minlength=len(ERROR_BIN_EDGES) - 1)
                counts = self.counts[name]
                for i, c in enumerate(hist.tolist()):
                    counts[i] += int(c)
                self.sums[name] += float(v.sum())
                self.n[name] += int(v.numel())

    def payload(self) -> Dict:
        return {
            "edges": ERROR_BIN_EDGES,
            "counts": self.counts,
            "mean_abs_error": {k: (self.sums[k] / self.n[k] if self.n[k] else 0.0)
                               for k in self.KEYS},
            "n": dict(self.n),
        }


@torch.no_grad()
def module_grad_norms(model) -> Dict[str, float]:
    """Gradient norm per branch, so a dead one is visible rather than inferred.

    A global norm hides which part produced it. With two decoders, a frontend
    run twice and a gated channel path that starts at zero, "the gradient is
    fine" is not a statement anyone should have to take on the aggregate.
    """
    groups: Dict[str, List[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        head = name.split(".")[0]
        if head in ("wavelet_frontends", "patch_embed_by_rate",
                    "reconstruction_heads", "raw_reconstruction_heads"):
            # Keep the route or rate: E128_512's frontend and E19_256's are
            # trained on different steps and averaging them says nothing.
            head = ".".join(name.split(".")[:2])
        groups.setdefault(head, []).append(p.grad.detach().float().reshape(-1))
    out = {}
    for k, v in groups.items():
        out[f"gradnorm/{k}"] = float(torch.cat(v).norm())
    return out


#: The route-macro summaries, as ``macro name -> the metric it averages``.
#: Unweighted across the routes present in validation, deliberately: the
#: validation sweep is proportional to corpus size, so a sample-weighted
#: summary is 97% TUEG and HBN and says almost nothing about E32_512.
MACRO_ROUTE_METRICS = {
    "macro_route_loss_total": "loss_total",
    "macro_route_loss_spec": "loss_masked_spec_mse",
    "macro_route_loss_raw": "loss_masked_raw_smoothl1",
    "macro_route_spec_corr": "masked_spec_corr",
    "macro_route_raw_corr": "masked_raw_corr",
    "macro_route_spec_nmse": "masked_spec_nmse",
    "macro_route_raw_nmse": "masked_raw_nmse",
}


class Accumulator:
    """Running sums and counts, globally and per route and per dataset.

    The validation sweep is PARTITIONED: each rank walks its own windows of
    every dataset. Four global scalars used to be all-reduced and NOTHING else
    was, so every ``route/<id>/...`` and ``dataset/<id>/...`` number written to
    metrics_epoch.jsonl was rank 0's quarter of the sweep -- and those are
    precisely the columns the per-route report exists to show.

    Sums and counts rather than means because that is what combines under any
    partition. ValIterator happens to emit the same number of batches on every
    rank -- its stride keeps each rank's slice full -- so a mean of means would
    give the same answer today. It stops doing so the moment that changes:
    ``max_batches_per_dataset``, an uneven route, a dataset shorter than one
    rank's batch. Reducing the totals is right in all of those cases and costs
    one collective an epoch.
    """

    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, metrics: Dict[str, float], route_id: str, dataset_id: str):
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            for key in (k, f"route/{route_id}/{k}", f"dataset/{dataset_id}/{k}"):
                self.sums[key] = self.sums.get(key, 0.0) + float(v)
                self.counts[key] = self.counts.get(key, 0) + 1

    def merge(self, other_sums: Dict[str, float], other_counts: Dict[str, int]):
        for k, v in other_sums.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        for k, c in other_counts.items():
            self.counts[k] = self.counts.get(k, 0) + int(c)

    def all_reduce(self, distributed: bool, device=None):
        """Sum every rank's (sum, count) for EVERY key, including the breakdowns.

        ``all_gather_object`` rather than a tensor all-reduce, because the key
        SETS differ between ranks: a rank whose slice of a small dataset was
        shorter than one batch contributes no ``dataset/faced/...`` keys at
        all, and a fixed-layout reduction would need the union of the keys
        before it could start. This exchanges the dictionaries and adds them,
        so every rank ends with the same one.
        """
        if not distributed or not dist.is_available() or not dist.is_initialized():
            return self
        payload = [None] * dist.get_world_size()
        dist.all_gather_object(payload, (self.sums, self.counts))
        merged = Accumulator()
        for sums, counts in payload:
            merged.merge(sums or {}, counts or {})
        self.sums, self.counts = merged.sums, merged.counts
        return self

    def mean(self) -> Dict[str, float]:
        return {k: self.sums[k] / self.counts[k] for k in self.sums
                if self.counts[k]}


def macro_route_metrics(means: Dict[str, float],
                        macro: Optional[Dict[str, str]] = None
                        ) -> Dict[str, float]:
    """Unweighted mean across the routes present, one entry per macro name.

    Reads the already-globally-aggregated ``route/<id>/<metric>`` values, so it
    must run AFTER ``Accumulator.all_reduce`` -- a macro built from rank 0's
    slice is a macro over a quarter of the sweep.
    """
    macro = macro or MACRO_ROUTE_METRICS
    by_metric: Dict[str, List[float]] = {}
    for key, value in means.items():
        if not key.startswith("route/"):
            continue
        parts = key.split("/", 2)
        if len(parts) != 3:
            continue
        _, _route_id, metric = parts
        if isinstance(value, (int, float)) and math.isfinite(value):
            by_metric.setdefault(metric, []).append(float(value))
    out: Dict[str, float] = {}
    for name, metric in macro.items():
        vals = by_metric.get(metric)
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


# --------------------------------------------------------------------------- #
# Checkpoint selection
# --------------------------------------------------------------------------- #

#: ``criterion -> (validation metric, files written when it improves)``.
#:
#: best.pth is a copy of best_total.pth. It used to be the best SPEC loss under
#: the name ``loss_masked_mse``, which is the compatibility alias for the spec
#: term -- so once the objective became half raw, the file every downstream
#: script loads by default was being selected by half of it. It now means what
#: its name has always implied: the best model under the loss that was trained.
BEST_SELECTION = {
    "total": ("loss_total", ("best_total.pth", "best.pth")),
    "spec": ("loss_masked_spec_mse", ("best_spec.pth",)),
    "raw": ("loss_masked_raw_smoothl1", ("best_raw.pth",)),
    "macro_total": ("macro_route_loss_total", ("best_macro_total.pth",)),
}
BEST_KEYS = tuple(BEST_SELECTION)

#: What each file on disk was chosen by, recorded IN the checkpoint so a file
#: found a year later does not have to be matched to the code that wrote it.
CHECKPOINT_SELECTION = {
    "best.pth": "val/loss_total",
    "best_total.pth": "val/loss_total",
    "best_spec.pth": "val/loss_masked_spec_mse",
    "best_raw.pth": "val/loss_masked_raw_smoothl1",
    "best_macro_total.pth": "val/macro_route_loss_total",
}

#: Human-readable names for the "new best ..." lines.
BEST_LABEL = {"total": "total", "spec": "spec", "raw": "raw",
              "macro_total": "macro-route total"}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class EEGC1Trainer:
    #: The registry this trainer draws from, and the label its banner prints.
    #: Class attributes rather than module globals read in place, so that
    #: physiowave.ecg_c1 can reuse this loop -- schedule, validation, metrics,
    #: checkpoint selection, resume -- by naming its own tables and model,
    #: rather than by keeping a second copy of all of it.
    routes_table = ROUTES
    datasets_table = PRETRAIN_DATASETS
    default_batch_by_route = DEFAULT_BATCH_BY_ROUTE
    banner_title = "EEG C1 multi-route pretraining"

    def __init__(self, cfg: Dict, out_dir: str, info, max_steps: Optional[int] = None):
        self.cfg = cfg
        self.out_dir = out_dir
        self.info = info
        self.max_steps = max_steps
        self.is_main = getattr(info, "is_main", True)
        self.device = getattr(info, "device", torch.device("cpu"))
        self.distributed = getattr(info, "distributed", False)

        tcfg = cfg.get("train", {})
        mcfg = cfg.get("model", {})
        dcfg = cfg.get("data", {})
        # Continued pretraining may add corpora outside the original seven
        # (routes.CONTINUATION_DATASETS), only when the config names them.
        if dcfg.get("continuation_datasets"):
            if self.datasets_table is not PRETRAIN_DATASETS:
                raise SystemExit("data.continuation_datasets is for the EEG corpus only")
            self.datasets_table = datasets_with(dcfg["continuation_datasets"])

        self.epochs = int(tcfg.get("epochs", 10))
        self.grad_accum = int(tcfg.get("grad_accumulation_steps", 1))
        # The objective's weights, resolved ONCE, here, by the same helper
        # the figures and the progress report call. They used to have a default
        # each; that agreed only for as long as every default was 1.0/0.25, and
        # moving the config to 0.5 is precisely the edit those defaults would
        # not have followed.
        self.objective = resolve_eeg_c1_objective(cfg)
        self.spec_weight = float(self.objective["spec_weight"])
        self.raw_weight = float(self.objective["raw_weight"])
        self.raw_beta = float(self.objective["raw_beta"])
        self.fold_kl = float(self.objective["fold_kl"])
        self.mask_before_frontend = bool(self.objective["mask_before_frontend"])
        self.normalize_spec_target = bool(self.objective["normalize_spec_target"])
        self.mask_ratio = float(mcfg.get("mask_ratio", 0.5))
        self.val_mask_seed = int(tcfg.get("val_mask_seed", 1234))
        # How many validation batches per dataset each epoch sweeps. None is
        # every one of them, which is what a final number needs and what ten
        # epochs of it cannot afford: the sweep is proportional to corpus size,
        # so it is mostly TUEG. Capping is safe for a CURVE because the mask
        # seed is fixed by window identity -- a cap takes the same windows every
        # epoch, so epoch 9 is comparable to epoch 0 rather than being a
        # different sample of the validation set.
        _vcap = tcfg.get("val_max_batches_per_dataset")
        self.val_max_batches = int(_vcap) if _vcap else None
        self.clip_grad = float(tcfg.get("clip_grad_norm", 1.0))
        # NUMERICAL GUARD. A step whose gradient norm is not finite, or is
        # absurdly large, is skipped: the weights are left alone and the
        # schedule advances. Without it one bad batch ends the run. The sEMG
        # 60k-step run went 38 epochs at a norm of 0.1-0.5; at epoch 39 one
        # hyser batch came in at 4e4, a later one at inf, and clip_grad_norm_
        # with an inf norm scales every gradient by NaN. Under bf16 there is
        # no GradScaler to refuse the step, so the weights became NaN and five
        # more epochs trained on them. The norm compared here is the pre-clip
        # one, which DDP has already averaged across ranks, so every rank
        # reaches the same decision without another collective. null turns
        # the threshold off; the non-finite check stays.
        _skip = tcfg.get("skip_step_grad_norm_above", 100.0)
        self.skip_grad_norm_above = float(_skip) if _skip else None
        # An epoch with this many skipped steps is not training; stop it.
        self.max_skipped_steps_per_epoch = int(
            tcfg.get("max_skipped_steps_per_epoch", 50))
        # Test hook: at this global step rank 0's loss is made infinite, so a
        # test can see the guard skip one step on EVERY rank and carry on.
        _dbg = os.environ.get("PW_DEBUG_NONFINITE_STEP")
        self._debug_nonfinite_step = int(_dbg) if _dbg else None
        # How many passes over one dataset per epoch is worth a line in the
        # banner. Configurable rather than hard-coded: `balanced` on a corpus
        # with a 30x size spread reaches 20x on the small ones by design.
        self.passes_warn_threshold = float(
            tcfg.get("max_passes_per_epoch_warn", 5.0))
        # 'auto' is a bar on a terminal and periodic lines under sbatch, where a
        # bar that redraws with carriage returns becomes one enormous line in
        # the .out. PW_PROGRESS overrides it without editing a config, which is
        # what you want when the question is "is this thing alive" and the
        # answer has to come from a job that is already queued.
        self.progress_mode = resolve_progress(
            os.environ.get("PW_PROGRESS") or str(tcfg.get("progress", "auto")))
        # Every fiftieth step is a cadence in STEPS, and how long fifty steps
        # take is the thing nobody knows before the run. A floor in seconds
        # makes "is it alive" answerable at a rate a person can wait through,
        # whatever a step turns out to cost.
        self.log_seconds = float(os.environ.get("PW_LOG_SECONDS")
                                 or tcfg.get("log_seconds", 120))
        self.batch_by_route = {**self.default_batch_by_route,
                               **(tcfg.get("batch_size_by_route") or {})}

        # -- data ---------------------------------------------------------- #
        self.train_index = CorpusIndex.from_manifest(dcfg["manifest_train"])
        self.val_index = (CorpusIndex.from_manifest(dcfg["manifest_val"])
                          if dcfg.get("manifest_val") else None)

        self.schedule = RouteSchedule(
            self.train_index, weights=dcfg.get("weights"),
            steps_per_epoch=tcfg.get("steps_per_epoch"),
            seed=int(cfg.get("seed", 42)),
            batch_by_route=self.batch_by_route,
            num_replicas=getattr(info, "world_size", 1),
            rank=getattr(info, "rank", 0),
            routes=self.routes_table, datasets=self.datasets_table,
            fixed_weights=dcfg.get("weights_fixed"),
            epoch_pass_of=tcfg.get("epoch_pass_of"))
        # train.corpus_passes_per_run: size the epoch so the whole run reads
        # the corpus that many times over, instead of fixing steps by hand.
        # Only when steps_per_epoch is not given; an explicit one wins.
        self.corpus_passes_per_run = tcfg.get("corpus_passes_per_run")
        if (self.corpus_passes_per_run is not None
                and tcfg.get("steps_per_epoch") is None):
            self.schedule.size_for_run(float(self.corpus_passes_per_run),
                                       self.epochs)
        self.loader = RouteBatchLoader(self.train_index, self.schedule)

        # -- model --------------------------------------------------------- #
        self.model = self.build_model(mcfg).to(self.device)

        self.raw_model = self.model
        if self.distributed:
            # find_unused_parameters is required and not merely defensive: a
            # step runs ONE route's frontend, so the other three frontends and
            # the other rate's patcher and decoder produce no gradient. Every
            # rank runs the same route, so the unused set is identical across
            # ranks and the reducer's bookkeeping stays consistent.
            self.model = DDP(
                self.model,
                device_ids=[info.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True)

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(), lr=float(tcfg.get("lr", 3e-4)),
            weight_decay=float(tcfg.get("weight_decay", 0.05)),
            betas=(0.9, 0.95))
        total_steps = max(1, self.epochs * len(self.schedule) // self.grad_accum)
        # Fractions allowed: a two-epoch continuation warms up over a tenth of one.
        warmup = int(float(tcfg.get("warmup_epochs", 1)) * max(
            1, len(self.schedule) // self.grad_accum))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: self._lr_lambda(s, warmup, total_steps,
                                                      float(tcfg.get("min_lr_ratio", 0.01))))
        self.use_amp = (self.device.type == "cuda"
                        and tcfg.get("precision", "bf16") != "fp32")
        self.amp_dtype = (torch.bfloat16 if tcfg.get("precision", "bf16") == "bf16"
                          else torch.float16)
        self.scaler = make_grad_scaler(
            self.device.type, self.use_amp and self.amp_dtype is torch.float16)

        self.epoch = 0
        self.global_step = 0
        # One bar per selection criterion. best.pth used to mean "best
        # loss_masked_mse", which is the compatibility alias for the SPEC term
        # alone -- so with a balanced dual objective the exported checkpoint was
        # chosen by half the loss it was trained on.
        self.best_scores: Dict[str, float] = {k: float("inf")
                                              for k in BEST_KEYS}
        self.best_epochs: Dict[str, Optional[int]] = {k: None for k in BEST_KEYS}
        self.history: Dict[str, List] = {"train": [], "val": []}
        self.tb = None
        if self.is_main:
            try:
                from ..train.utils import TensorBoardWriter
                self.tb = TensorBoardWriter(os.path.join(out_dir, "tensorboard"))
            except Exception:                                 # noqa: BLE001
                self.tb = None

    # -- what a subclass for another modality replaces ------------------- #
    def build_model(self, mcfg: Dict):
        """The pretrainer, from the config's ``model:`` block."""
        return MultiRouteEEGPretrainer(
            embed_dim=int(mcfg.get("embed_dim", 384)),
            depth=int(mcfg.get("depth", 6)),
            num_heads=int(mcfg.get("num_heads", 6)),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            dropout=float(mcfg.get("dropout", 0.1)),
            norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
            qk_norm=bool(mcfg.get("qk_norm", True)),
            max_level=int(mcfg.get("max_level", 3)),
            wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
            wavelet_names=mcfg.get("wavelet_names"),
            wave_init_mode=mcfg.get("wave_init_mode", "pad"),
            use_separate_channel=bool(mcfg.get("use_separate_channel", True)),
            mask_before_frontend=self.mask_before_frontend,
            normalize_spec_target=self.normalize_spec_target,
            fold_synthesis=int(mcfg.get("fold_synthesis", 3)),
            fold_gamma=float(mcfg.get("fold_gamma", 0.1)),
            masking_strategy=mcfg.get("masking_strategy", "frequency_guided"),
            importance_ratio=float(mcfg.get("importance_ratio", 0.6)),
            mask_ratio=self.mask_ratio,
            channel_encoding=mcfg.get("channel_encoding", "id"),
            channel_injection=mcfg.get("channel_injection", "token"),
            channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
            channel_token_gate_init=float(mcfg.get("channel_token_gate_init", 0.0)),
        )

    def vocab_payload(self) -> Dict:
        """The channel vocabulary's size and hash, as recorded in checkpoints."""
        return vocab_payload()

    def save_vocab(self, path: str):
        save_vocab(path)

    @staticmethod
    def _lr_lambda(step, warmup, total, min_ratio):
        if warmup and step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # -- files ------------------------------------------------------------- #
    def _write_startup(self):
        if not self.is_main:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "figures"), exist_ok=True)
        self.save_vocab(os.path.join(self.out_dir, "channel_vocab.json"))
        # Written here rather than in pretrain_main: this trainer is dispatched
        # before that function reaches its own copy of these, so leaving them to
        # it produced a run directory with no record of what produced it.
        try:
            from ..config import save_resolved
            save_resolved(self.cfg, os.path.join(self.out_dir,
                                                 "config_resolved.yaml"))
        except Exception as exc:                              # noqa: BLE001
            print(f"warning: could not write config_resolved.yaml ({exc})")
        try:
            from ..models.checkpoint import environment_info
            with open(os.path.join(self.out_dir, "environment.json"), "w") as f:
                json.dump(environment_info(), f, indent=2)
        except Exception as exc:                              # noqa: BLE001
            print(f"warning: could not write environment.json ({exc})")
        report = self.raw_model.parameter_report()
        mixture = self.schedule.realised_mixture()
        manifest = {
            "routes": {rid: {"n_channels": r.n_channels,
                             "window_samples": r.window_samples,
                             "sampling_rate": r.sampling_rate,
                             "patch_size": list(r.patch_size),
                             "n_tokens": r.n_tokens}
                       for rid, r in self.routes_table.items()},
            "datasets": {d: {"route_id": self.datasets_table[d].route_id,
                             "n_windows": self.train_index.window_counts().get(d, 0)}
                         for d in self.schedule.dataset_ids},
            "target_weights": self.schedule.weights,
            "realised_mixture": mixture,
            "batch_size_by_route": self.batch_by_route,
            "steps_per_epoch": self.schedule.steps_per_epoch,
            "parameters": report,
            **self.raw_model.vocab_fingerprint(),
        }
        with open(os.path.join(self.out_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print("=" * 66)
        print(f"  {self.banner_title}")
        print(f"  total parameters        {report['total']:,}")
        print(f"  shared transformer      {report['shared_transformer']:,}")
        for rid in self.routes_table:
            print(f"  frontend {rid:<10s}     {report[f'wavelet_frontend.{rid}']:,}")
        for rate in sorted(self.raw_model.patch_embed_by_rate):
            print(f"  patch_embed {rate:<7s}    {report[f'patch_embed.{rate}']:,}"
                  f"   spec decoder {report[f'reconstruction_head.{rate}']:,}"
                  f"   raw decoder {report[f'raw_reconstruction_head.{rate}']:,}")
        print(f"  downstream encoder      {report['downstream_encoder']:,}"
              f"   (+{report['pretraining_only']:,} pretraining-only:"
              f" both decoders and the mask token)")
        for line in objective_banner(self.objective):
            print(f"  {line}")
        print(f"  L_total = {objective_equation(self.objective)}")
        print(f"  spec target: {'normalised per patch' if self.normalize_spec_target else 'RAW -- the loss will track the frontend scale'}",
              flush=True)
        print(f"  masking: {'signal before the frontend' if self.mask_before_frontend else 'tokens only, after the frontend'}"
              f"   targets: detached")
        if "channel_encoder" in report:
            print(f"  channel encoder (C1)    {report['channel_encoder']:,}"
                  f"  + proj {report['channel_to_token']:,}")
        print("  routes:")
        for rid, r in self.routes_table.items():
            print(f"    {r.describe()}")
        counts = self.train_index.window_counts()
        total_w = max(1, sum(counts.values()))
        policy = getattr(self.schedule, "weight_policy", "explicit")
        explicit_steps = self.cfg.get("train", {}).get("steps_per_epoch") is not None
        oversampled: Dict[str, float] = {}
        print(f"  mixture policy: {policy}")
        print("    dataset          corpus%   step%  window%   passes/epoch")
        for d in self.schedule.dataset_ids:
            b = self.schedule.batch_by_route[self.datasets_table[d].route_id]
            seen = (mixture["by_step"][d] * self.schedule.steps_per_epoch
                    * b * self.schedule.num_replicas)
            # passes/epoch is the number that says whether a small corpus is
            # being read a hundred times to fill a quota. Under `proportional`
            # every row here is 1.0.
            passes = seen / max(1, counts.get(d, 0))
            oversampled[d] = passes
            print(f"    {d:<14s} {counts.get(d,0)/total_w*100:7.2f}% "
                  f"{mixture['by_step'][d]*100:6.2f}% "
                  f"{mixture['by_window'][d]*100:7.2f}%   {passes:6.2f}x")
        print(f"  channel vocab sha256    {self.vocab_payload()['channel_vocab_sha256'][:16]}")

        # -- the training budget, spelled out ------------------------------- #
        # These four numbers are what an epoch length actually buys, and every
        # one of them is a derived quantity that nobody computes by hand
        # correctly under sbatch. steps_per_epoch is SCHEDULED steps; the
        # optimizer sees steps_per_epoch / grad_accum of them.
        steps_per_epoch = int(self.schedule.steps_per_epoch)
        updates_per_epoch = max(1, steps_per_epoch // self.grad_accum)
        world = int(getattr(self.info, "world_size", 1) or 1)
        print("  budget:")
        if explicit_steps:
            how = "  (explicit override)"
        elif self.corpus_passes_per_run is not None:
            how = (f"  (sized so {self.epochs} epochs read "
                   f"{float(self.corpus_passes_per_run):g}x the corpus)")
        else:
            how = "  (derived from the mixture)"
        print(f"    resolved steps per epoch   {steps_per_epoch}{how}")
        read = steps_per_epoch * self.epochs * self.schedule.windows_per_step()
        print(f"    windows read over the run  {read:,.0f} of a {total_w:,}-window "
              f"corpus ({read / total_w:.2f}x)")
        print(f"    total optimizer updates    {updates_per_epoch * self.epochs}"
              f"   ({updates_per_epoch}/epoch x {self.epochs} epochs,"
              f" grad_accum {self.grad_accum})")
        print(f"    effective world size       {world}")
        print(f"    validation per epoch       "
              + ("every window" if self.val_max_batches is None
                 else f"first {self.val_max_batches} batch(es) per dataset "
                      f"-- a CURVE, not a final number"))
        print(f"    per-route batch size       " + "  ".join(
            f"{rid}={self.batch_by_route[rid]}" for rid in self.routes_table
            if rid in self.batch_by_route))
        print(f"    passes per dataset/epoch   " + "  ".join(
            f"{d}={oversampled[d]:.2f}x" for d in self.schedule.dataset_ids))
        # A warning, never a refusal: repeated sampling is a legitimate choice
        # under `balanced`, and an explicit STEPS_PER_EPOCH is how the final
        # annealing runs buy their updates. What is NOT acceptable is it
        # happening without anybody seeing it.
        loud = {d: n for d, n in oversampled.items()
                if n > self.passes_warn_threshold}
        if loud:
            print(f"  WARNING: at {steps_per_epoch} steps/epoch these datasets "
                  f"are read more than {self.passes_warn_threshold:g}x per "
                  f"epoch:")
            for d, n in sorted(loud.items(), key=lambda kv: -kv[1]):
                print(f"    {d:<14s} {n:6.2f} passes/epoch "
                      f"({counts.get(d, 0):,} windows)")
            print("    An epoch is then a repeat of the same windows rather "
                  "than new exposure. Training continues; "
                  "train.max_passes_per_epoch_warn sets the threshold.")
        print(f"  steps/epoch {steps_per_epoch}  "
              f"epochs {self.epochs}  grad_accum {self.grad_accum}")
        print("=" * 66, flush=True)

    #: Written by appending, so a run that starts fresh in a directory a
    #: previous attempt used would interleave its curves with that attempt's.
    METRICS_FILES = ("metrics_step.jsonl", "metrics_epoch.jsonl", "history.json")

    def retire_previous_metrics(self) -> Optional[str]:
        """Move an earlier attempt's metrics aside. Only on a fresh start.

        Four jobs died in this directory before one got past the first epoch,
        each leaving its rows behind, and the file that was supposed to answer
        "how far along is it" answered with a total across all five. A resume
        must NOT do this -- there the earlier rows are this run's own history.

        Moved rather than deleted: the crashed attempts are how you find out
        why they crashed.
        """
        if not self.is_main:
            return None
        present = [n for n in self.METRICS_FILES
                   if os.path.isfile(os.path.join(self.out_dir, n))]
        if not present:
            return None
        n = 0
        while os.path.exists(os.path.join(self.out_dir, "superseded", str(n))):
            n += 1
        dest = os.path.join(self.out_dir, "superseded", str(n))
        os.makedirs(dest, exist_ok=True)
        for name in present:
            os.replace(os.path.join(self.out_dir, name),
                       os.path.join(dest, name))
        print(f"  moved {len(present)} file(s) from an earlier attempt in this "
              f"directory to {dest}", flush=True)
        return dest

    def _append_jsonl(self, name: str, row: Dict):
        if not self.is_main:
            return
        with open(os.path.join(self.out_dir, name), "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- checkpointing ----------------------------------------------------- #
    def state_dict(self) -> Dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_scores": dict(self.best_scores),
            "best_epochs": dict(self.best_epochs),
            "checkpoint_selection": dict(CHECKPOINT_SELECTION),
            # Backward-compatible ALIAS for the spec bar, and nothing selects
            # on it any more. A reader that only knows this name gets the
            # number it always meant.
            "best_val_loss_masked_mse": self.best_scores["spec"],
            "objective": dict(self.objective),
            "sampler": self.schedule.state_dict(),
            "history": self.history,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
            "config": self.cfg,
            **self.vocab_payload(),
        }

    def save(self, name: str):
        if not self.is_main:
            return
        path = os.path.join(self.out_dir, name)
        torch.save(self.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)

    def _restore_best_records(self, ck: Dict):
        """The four bars, migrating a checkpoint that only had one.

        Before the dual objective there was a single ``best_val_loss_masked_mse``
        and it meant the spec loss. Reading it into the TOTAL bar would start
        the new run with a total-loss threshold borrowed from a different
        quantity, and a genuinely better model could then fail to clear it for
        the whole run.
        """
        scores = ck.get("best_scores")
        epochs = ck.get("best_epochs") or {}
        if isinstance(scores, dict) and scores:
            self.best_scores = {k: float(scores.get(k, float("inf")))
                                for k in BEST_KEYS}
            self.best_epochs = {k: epochs.get(k) for k in BEST_KEYS}
            return
        legacy = ck.get("best_val_loss_masked_mse")
        self.best_scores = {k: float("inf") for k in BEST_KEYS}
        self.best_epochs = {k: None for k in BEST_KEYS}
        if legacy is not None and math.isfinite(float(legacy)):
            self.best_scores["spec"] = float(legacy)
            self.best_epochs["spec"] = ck.get("epoch")
        print(f"  migrated legacy checkpoint-selection state: "
              f"best spec = {self.best_scores['spec']:.5f} from "
              f"best_val_loss_masked_mse; total/raw/macro_total start at "
              f"infinity (they were never recorded)", flush=True)

    def _check_vocab(self, ck, path):
        """An embedding row means whichever electrode held that id when it was
        learned. A checkpoint from a different vocabulary is relabelled."""
        recorded = ck.get("channel_vocab_sha256")
        current = self.vocab_payload()["channel_vocab_sha256"]
        if recorded and recorded != current:
            raise SystemExit(
                f"{path} was trained under channel vocabulary "
                f"{recorded[:16]} and this one is {current[:16]}. Every "
                f"embedding row would mean a different electrode. Check out the "
                f"commit that produced the checkpoint, or retrain.")

    def init_from(self, path: str):
        """The WEIGHTS, and nothing else. Not a resume.

        Resuming continues one run: it restores the optimizer, the scheduler's
        position, the step count and the sampler, because those are the run.
        Initialising STARTS a run from another's representation -- and the
        difference matters most exactly when you want it, because a new mixture
        changes steps_per_epoch, and a scheduler resumed across that change is
        counting in the old epoch's units. Restoring one from a 384-step epoch
        into a 954-step one leaves the cosine 20% short of annealed at the end
        of the run, at a learning rate an order of magnitude above where it
        should finish.

        So this takes the model and leaves the optimizer, the schedule, the
        epoch counter and the best-so-far bar fresh.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self._check_vocab(ck, path)
        self.raw_model.load_state_dict(ck["model"])
        # Deliberately NOT checked against resume_incompatibilities: a change
        # of objective is the ordinary REASON to use --init-from. The weights
        # transfer; nothing that was accumulated under the old weighting --
        # optimizer moments, cosine position, best bars -- comes with them.
        old_obj = resolve_eeg_c1_objective(ck.get("config") or {})
        print(f"initialised from {path} "
              f"(weights only; it was at epoch {ck.get('epoch', '?')}, "
              f"step {ck.get('global_step', '?')}, best spec "
              f"{float(ck.get('best_val_loss_masked_mse', float('nan'))):.5f})",
              flush=True)
        if objective_equation(old_obj) != objective_equation(self.objective):
            print(f"  its objective was {objective_equation(old_obj)}; "
                  f"this run trains {objective_equation(self.objective)}. "
                  f"Weights only -- optimizer, schedule, epoch counter and "
                  f"best-so-far records all start fresh.", flush=True)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # BEFORE the weights: an exact resume claims this process continues the
        # one that wrote the file. If the loss it is continuing is not the loss
        # that produced the optimizer moments, the cosine's position and the
        # best-so-far bars, the continuation is a fiction and the run's own
        # metrics file is the only place the change would ever have shown --
        # unlabelled, as one curve.
        diffs = resume_incompatibilities(ck.get("config"), self.cfg,
                                         ck.get("sampler"))
        if diffs:
            raise SystemExit(resume_refusal_message(path, diffs))
        # train.epochs is NOT in that list, because extending a run is the
        # ordinary reason to resume one. It is not free, though: the cosine's
        # total is epochs x steps_per_epoch, so a longer run re-shapes the
        # anneal from here on rather than continuing the old curve. Said out
        # loud, because the learning rate then does something the first
        # submission's plot does not predict.
        old_epochs = ((ck.get("config") or {}).get("train") or {}).get("epochs")
        if old_epochs is not None and int(old_epochs) != self.epochs:
            print(f"  note: this checkpoint was scheduled for {old_epochs} "
                  f"epochs and this run is scheduled for {self.epochs}. The "
                  f"cosine's horizon moves with it, so the learning rate from "
                  f"here follows the new schedule, not the old one.",
                  flush=True)
        # STRICT, deliberately. A checkpoint from the single-decoder objective
        # has no raw_reconstruction_heads, and loading it loosely would resume
        # a run whose optimizer state, scheduler position and step count all
        # belong to a different objective -- reported as a continuation of it.
        # Resuming and initialising from are not the same operation.
        try:
            self.raw_model.load_state_dict(ck["model"])
        except RuntimeError as exc:
            missing = [k for k in self.raw_model.state_dict()
                       if k not in ck["model"]]
            raw_missing = [k for k in missing
                           if k.startswith("raw_reconstruction_heads.")]
            if raw_missing:
                raise SystemExit(
                    f"{path} has no raw reconstruction head, so it was written "
                    f"by the single-decoder objective and is not a resume of "
                    f"this one.\n\n"
                    f"  Its optimizer state, scheduler position and step count "
                    f"belong to a different loss, and continuing from them "
                    f"would be reported as one run when it is two.\n\n"
                    f"  To carry the representation over instead, export it and "
                    f"start a new run:\n"
                    f"    python scripts/export_eeg_pretrained_encoder.py "
                    f"--checkpoint {path} ...\n\n"
                    f"  ({len(raw_missing)} missing key(s), e.g. "
                    f"{raw_missing[0]})") from exc
            raise
        if not self._weights_finite():
            others = sorted(n for n in os.listdir(os.path.dirname(path) or ".")
                            if n.startswith("best") and n.endswith(".pth"))
            raise SystemExit(
                f"{path} holds non-finite weights: the run that wrote it had "
                f"already diverged, and resuming it would train NaN on NaN. "
                f"Resume from the last checkpoint written before the "
                f"divergence instead, e.g. RESUME=<out_dir>/best_raw.pth "
                f"(the best_*.pth files here: {', '.join(others) or 'none'}; "
                f"each one's epoch is in its 'epoch' field).")
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.epoch = int(ck.get("epoch", 0))
        self.global_step = int(ck.get("global_step", 0))
        self._restore_best_records(ck)
        self.history = ck.get("history", {"train": [], "val": []})
        if ck.get("sampler"):
            self.schedule.load_state_dict(ck["sampler"])
        rng = ck.get("rng") or {}
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"])
                                else rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        self._check_vocab(ck, path)
        bars = "  ".join(
            f"{BEST_LABEL[k]} {self.best_scores[k]:.5f}"
            f"@{self.best_epochs[k] if self.best_epochs[k] is not None else '-'}"
            for k in BEST_KEYS if math.isfinite(self.best_scores[k]))
        print(f"resumed from {path}: epoch {self.epoch}, step "
              f"{self.global_step}" + (f"  best: {bars}" if bars else ""))
        self._retire_metrics_from(self.epoch)

    def _weights_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all())
                   for p in self.raw_model.parameters())

    def _step_is_bad(self, grad_norm: float) -> bool:
        if not math.isfinite(grad_norm):
            return True
        return (self.skip_grad_norm_above is not None
                and grad_norm > self.skip_grad_norm_above)

    def _retire_metrics_from(self, epoch: int) -> None:
        """Drop the metric rows of epochs this resume is about to redo.

        Resuming from best_raw.pth after latest.pth went NaN restarts at the
        epoch after best_raw's, and the rows the diverged continuation wrote
        for those epochs are still in the files. Left there, the curve has
        two rows per epoch and the figure draws both. Only rank 0 writes.
        """
        if not self.is_main:
            return
        moved = 0
        for name in ("metrics_epoch.jsonl", "metrics_step.jsonl"):
            path = os.path.join(self.out_dir, name)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                rows = f.readlines()
            keep, drop = [], []
            for line in rows:
                try:
                    e = json.loads(line).get("epoch")
                except ValueError:
                    keep.append(line)
                    continue
                (drop if e is not None and int(e) >= epoch else keep).append(line)
            if not drop:
                continue
            dest = os.path.join(self.out_dir, "superseded", "resumed")
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, name), "a") as f:
                f.writelines(drop)
            with open(path + ".tmp", "w") as f:
                f.writelines(keep)
            os.replace(path + ".tmp", path)
            moved += len(drop)
        if moved:
            print(f"  moved {moved} metric row(s) of epoch >= {epoch} to "
                  f"superseded/resumed/: this resume redoes those epochs",
                  flush=True)

    # -- one epoch --------------------------------------------------------- #
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.schedule.set_epoch(self.epoch)
        acc = Accumulator()
        t0 = time.time()
        windows = 0
        skipped = 0                    # steps the numerical guard withheld
        max_norm = 0.0                 # largest finite pre-clip gradient norm
        # Captured before the loop. len(self.loader) is the schedule's REMAINING
        # steps -- it has to be, so a resumed epoch iterates only what is left --
        # and self.schedule.start_step advances every iteration, so reading
        # either one inside the loop counts down instead of up.
        first_step = int(self.schedule.start_step)
        epoch_steps = int(self.schedule.steps_per_epoch)
        last_log_t = t0
        route_seconds: Dict[str, float] = {}
        route_windows: Dict[str, int] = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        bar = progress(self.loader, f"train {self.epoch}", self.progress_mode,
                       self.is_main)
        for i, batch in enumerate(bar):
            x = batch["x"].to(self.device, non_blocking=True)
            meta = {k: v.to(self.device) for k, v in batch["channel_meta"].items()}
            is_accum = (i + 1) % self.grad_accum != 0
            step_t0 = time.time()

            ctx = (torch.autocast(self.device.type, dtype=self.amp_dtype)
                   if self.use_amp else _nullcontext())
            with ctx:
                out = self.model(x, batch["route_id"], channel_meta=meta,
                                 mask_ratio=self.mask_ratio)
                loss, metrics = masked_reconstruction_loss(
                    out, spec_weight=self.spec_weight,
                    raw_weight=self.raw_weight, fold_kl=self.fold_kl,
                    raw_beta=self.raw_beta)
            if (self._debug_nonfinite_step is not None and self.is_main
                    and self.global_step == self._debug_nonfinite_step):
                loss = loss * float("inf")

            self.scaler.scale(loss / self.grad_accum).backward()
            grad_norm = float("nan")
            branch_norms: Dict[str, float] = {}
            # first_step + i + 1, not len(self.loader): the schedule's length is
            # what REMAINS, so "i + 1 == len(self.loader)" is not the last step
            # of the epoch -- it is the step where the count passed the
            # countdown, exactly halfway, and the last step was never logged.
            want_log = self.is_main and (
                i % 50 == 0
                or first_step + i + 1 == epoch_steps
                or time.time() - last_log_t >= self.log_seconds)
            skipped_this = 0.0
            if not is_accum:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), self.clip_grad))
                # HERE, not after the step: zero_grad(set_to_none=True) is
                # below and every gradient is None by then. Only on the
                # logged steps, because walking every parameter is nothing at
                # every fiftieth step and not nothing at every one.
                if want_log:
                    branch_norms = module_grad_norms(self.raw_model)
                if self._step_is_bad(grad_norm):
                    # The update is what is withheld; the schedule still
                    # advances, because it is a function of time and a skipped
                    # step took time. clip_grad_norm_ has already scaled the
                    # gradients by NaN when the norm was inf, which is why
                    # they must not reach the optimizer.
                    skipped_this = 1.0
                    skipped += 1
                    if self.is_main and skipped <= 20:
                        # Which branch blew up is the whole diagnosis, and a
                        # skipped step is rare enough to afford the walk.
                        worst = sorted(module_grad_norms(self.raw_model).items(),
                                       key=lambda kv: -kv[1])[:3]
                        print(f"  SKIPPED step {self.global_step} "
                              f"[{batch['route_id']} {batch['dataset_id']}]: "
                              f"pre-clip grad norm {grad_norm:.3g}, loss "
                              f"{float(loss.detach()):.4g}; weights unchanged. "
                              f"Largest: " + ", ".join(
                                  f"{k.split('/', 1)[-1]} {v:.3g}" for k, v in worst),
                              flush=True)
                    if skipped > self.max_skipped_steps_per_epoch:
                        raise SystemExit(
                            f"{skipped} steps skipped in epoch {self.epoch}: "
                            f"the gradient is not finite (or is above "
                            f"train.skip_step_grad_norm_above="
                            f"{self.skip_grad_norm_above}) on most batches, "
                            f"so this is not one bad batch but a model that "
                            f"has gone bad. Resume from the last best_*.pth "
                            f"with a lower learning rate: "
                            f"RESUME=<out_dir>/best_raw.pth LR=...")
                else:
                    self.scaler.step(self.optimizer)
                    if math.isfinite(grad_norm):
                        max_norm = max(max_norm, grad_norm)
                # Also on a skipped step: for an fp16 scaler this closes the
                # unscale_ it opened; disabled (bf16, fp32) it is a no-op.
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            windows += x.shape[0]
            # Per route, so the mixture figure can show what each one COST as
            # well as what share it got. E128_512 draws 12 windows to
            # E19_256's 64 and each is 6.7x the tokens; share of steps says
            # nothing about where the hours went.
            route_seconds[batch["route_id"]] = route_seconds.get(
                batch["route_id"], 0.0) + (time.time() - step_t0)
            route_windows[batch["route_id"]] = route_windows.get(
                batch["route_id"], 0) + x.shape[0]

            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            # Of the steps taken: a skipped step's norm is the reason it was
            # skipped, and in the epoch mean it would bury the other 999.
            if math.isfinite(grad_norm) and not skipped_this:
                metrics["grad_norm"] = grad_norm
            if not is_accum:
                metrics["skipped_step"] = skipped_this
            gate = self.raw_model.channel_gate_value()
            if gate is not None:
                metrics["channel_token_gate_tanh"] = gate
            alpha = out.get("fold_alpha")
            if alpha is not None:
                for s, a in enumerate(alpha.detach().float().cpu().tolist()):
                    metrics[f"fold_alpha_{s}"] = a
            acc.add(metrics, batch["route_id"], batch["dataset_id"])

            if want_log:
                last_log_t = time.time()
                self._append_jsonl("metrics_step.jsonl", {
                    "epoch": self.epoch, "step": self.global_step,
                    # Absolute, not elapsed: it survives a resume, and it is
                    # what tells you whether the last row is a minute old or an
                    # hour old without watching the file grow.
                    "unix_time": last_log_t,
                    "route_id": batch["route_id"],
                    "dataset_id": batch["dataset_id"],
                    **metrics, **branch_norms})
                # Sixteen GPUs printing nothing between the banner and the end
                # of the first epoch is indistinguishable from sixteen GPUs
                # deadlocked, and the difference matters at this price. Same
                # cadence as the jsonl row, so the cost is one line per fifty
                # steps rather than a second stream to throttle.
                done = first_step + i + 1
                took = max(1e-9, time.time() - t0)
                rate = (i + 1) / took
                eta = (epoch_steps - done) / rate
                if self.progress_mode == "log":
                    print(f"  [{time.strftime('%H:%M:%S')}] "
                          f"epoch {self.epoch} step {done}/{epoch_steps} "
                          f"[{batch['route_id']} {batch['dataset_id']}] "
                          f"loss {metrics.get('loss_total', float('nan')):.5f} "
                          f"lr {metrics['lr']:.2e} "
                          f"{windows / took:.0f} win/s "
                          f"eta {fmt_eta(eta)}", flush=True)
            # Outside want_log: a bar that only moved every fiftieth step would
            # be a worse version of the log line it replaces.
            set_postfix(bar,
                        loss=f"{metrics.get('loss_total', float('nan')):.4f}",
                        route=batch["route_id"],
                        win_s=f"{windows / max(1e-9, time.time() - t0):.0f}")
            self.schedule.start_step = i + 1
            if self.max_steps and self.global_step >= self.max_steps:
                break

        out = acc.mean()
        elapsed = max(1e-9, time.time() - t0)
        out["throughput_windows_per_s"] = windows / elapsed
        for rid, secs in route_seconds.items():
            out[f"route_seconds/{rid}"] = secs
            out[f"route_share_of_time/{rid}"] = secs / elapsed
        for rid, n in route_windows.items():
            out[f"route_windows/{rid}"] = float(n)
        out["epoch_seconds"] = elapsed
        out["skipped_steps"] = float(skipped)
        out["max_grad_norm"] = max_norm
        if self.device.type == "cuda":
            out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(
                self.device) / 1e6
        mixture = self.schedule.realised_mixture()
        for d, v in mixture["by_step"].items():
            out[f"mixture_step/{d}"] = v
        for d, v in mixture["by_window"].items():
            out[f"mixture_window/{d}"] = v
        return out

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.val_index is None:
            return {}
        self.model.eval()
        acc = Accumulator()
        hist = ErrorHistogram()
        it = ValIterator(self.val_index, self.batch_by_route,
                         getattr(self.info, "world_size", 1),
                         getattr(self.info, "rank", 0),
                         max_batches_per_dataset=max_batches,
                         routes=self.routes_table)
        val_t0 = last_val_log = time.time()
        n_val_batches = len(it)
        for vb, batch in enumerate(progress(it, f"val {self.epoch}",
                                            self.progress_mode, self.is_main)):
            if (self.is_main and self.progress_mode == "log"
                    and time.time() - last_val_log >= self.log_seconds):
                # The sweep is the other half of an epoch and used to print
                # nothing at all, so the gap between the last training line and
                # "epoch N:" looked exactly like a hang.
                last_val_log = time.time()
                rate = (vb + 1) / max(1e-9, last_val_log - val_t0)
                print(f"  [{time.strftime('%H:%M:%S')}] "
                      f"epoch {self.epoch} val {vb + 1}/{n_val_batches} "
                      f"eta {fmt_eta((n_val_batches - vb - 1) / max(1e-9, rate))}",
                      flush=True)
            x = batch["x"].to(self.device, non_blocking=True)
            meta = {k: v.to(self.device) for k, v in batch["channel_meta"].items()}
            gen = _mask_generator(self.val_mask_seed, batch["dataset_id"],
                                  batch["mask_seed_index"])
            ctx = (torch.autocast(self.device.type, dtype=self.amp_dtype)
                   if self.use_amp else _nullcontext())
            with ctx:
                out = self.raw_model(x, batch["route_id"], channel_meta=meta,
                                     mask_ratio=self.mask_ratio,
                                     mask_generator=gen)
                _, metrics = masked_reconstruction_loss(
                    out, spec_weight=self.spec_weight,
                    raw_weight=self.raw_weight, fold_kl=self.fold_kl,
                    raw_beta=self.raw_beta)
            hist.add(out)
            # The control the loss cannot provide: error on the tokens the
            # model could SEE. A masked-reconstruction objective can be
            # satisfied by a model that has learned to copy its input, and the
            # way that shows is a near-zero visible error beside an unimproved
            # masked one. Nothing in the loss looks at visible tokens.
            with torch.no_grad():
                sel = out["mask"].unsqueeze(-1).expand_as(out["pred_spec"])
                vis = ~sel
                if out.get("valid_tokens") is not None:
                    vis = vis & out["valid_tokens"].unsqueeze(-1).expand_as(sel)
                if bool(vis.any()):
                    for tag, pk, tk in (("spec", "pred_spec", "target_spec"),
                                        ("raw", "pred_raw", "target_raw")):
                        e = (out[pk].float() - out[tk].float()).abs()[vis]
                        metrics[f"visible_{tag}_mae"] = float(e.mean())
            acc.add(metrics, batch["route_id"], batch["dataset_id"])
        it.close()
        # EVERY key, not just the four global scalars that used to be reduced.
        # ValIterator partitions each dataset across ranks, so
        # `route/E32_512/masked_spec_corr` on rank 0 was a mean over rank 0's
        # windows: four ranks held four different per-route tables and only
        # rank 0's reached metrics_epoch.jsonl. Sums and counts go over the
        # wire and the mean is taken from the totals, which is the form that
        # stays correct if the slices ever stop being equal.
        acc.all_reduce(self.distributed, self.device)
        means = acc.mean()
        # After the reduction, so the macro is over every rank's routes.
        means.update(macro_route_metrics(means))
        means["val_seconds"] = time.time() - val_t0
        means["val_batches"] = float(n_val_batches)
        if self.is_main:
            payload = hist.payload()
            payload["epoch"] = self.epoch
            payload["global_step"] = self.global_step
            with open(os.path.join(self.out_dir, "error_histogram.json"),
                      "w") as f:
                json.dump(payload, f)
            self._append_jsonl("error_histogram_by_epoch.jsonl", payload)
        return means

    # -- driver ------------------------------------------------------------ #
    def fit(self) -> int:
        self._write_startup()
        start = self.epoch
        run_t0 = time.time()
        epochs_here = 0
        for epoch in range(start, self.epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate(max_batches=self.val_max_batches)
            epochs_here += 1
            row = {"epoch": epoch, "global_step": self.global_step,
                   **{f"train/{k}": v for k, v in train_metrics.items()},
                   **{f"val/{k}": v for k, v in val_metrics.items()}}
            self._append_jsonl("metrics_epoch.jsonl", row)
            if self.tb is not None:
                self.tb.add_scalars(
                    {k: float(v) for k, v in row.items()
                     if isinstance(v, (int, float)) and math.isfinite(v)},
                    self.global_step)
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)
            if self.is_main:
                with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                    json.dump(self.history, f, indent=2)
                # The question a 50-epoch job is actually asked -- will it
                # fit in the walltime -- is answerable from here and nowhere
                # else, so it is answered here rather than left to arithmetic
                # over a jsonl file.
                spent = time.time() - run_t0
                left = (self.epochs - epoch - 1) * spent / max(1, epochs_here)
                # "epoch 0" and not "epoch 1 of 50": the identifier has to
                # match metrics_epoch.jsonl, which is 0-based. The fraction is
                # a count of epochs FINISHED, which is a different number and
                # is spelled as one.
                nan = float("nan")
                print(f"epoch {epoch} done ({epoch + 1}/{self.epochs}): "
                      f"train total "
                      f"{train_metrics.get('loss_total', nan):.5f}  "
                      f"val total {val_metrics.get('loss_total', nan):.5f} "
                      f"(spec {val_metrics.get('loss_masked_spec_mse', nan):.5f} "
                      f"r {val_metrics.get('masked_spec_corr', nan):.3f} "
                      f"nmse {val_metrics.get('masked_spec_nmse', nan):.3f} | "
                      f"raw {val_metrics.get('loss_masked_raw_smoothl1', nan):.5f} "
                      f"r {val_metrics.get('masked_raw_corr', nan):.3f} "
                      f"nmse {val_metrics.get('masked_raw_nmse', nan):.3f} | "
                      f"macro {val_metrics.get('macro_route_loss_total', nan):.5f})  "
                      f"gate {train_metrics.get('channel_token_gate_tanh', 0):.4f}  "
                      f"[{fmt_eta(train_metrics.get('epoch_seconds', nan))} "
                      f"train + {fmt_eta(val_metrics.get('val_seconds', nan))} "
                      f"val, elapsed {fmt_eta(spent)}, run eta {fmt_eta(left)}]",
                      flush=True)

            self.schedule.start_step = 0
            self.epoch = epoch + 1
            # The bars are updated BEFORE latest.pth is written. Written the
            # other way round, latest.pth records the previous epoch's bests,
            # and a resume from it then believes the best is worse than it is --
            # letting the next epoch overwrite a best checkpoint with a worse
            # one.
            if not self._weights_finite():
                # Belt to the step guard's braces: never write NaN weights
                # over the last good latest.pth.
                raise SystemExit(
                    f"the weights are not finite after epoch {epoch}; "
                    f"latest.pth is left as it was. Resume from it or from "
                    f"the last best_*.pth with a lower learning rate.")
            improved = self._update_best_records(epoch, val_metrics)
            self.save("latest.pth")
            self._save_best(improved)
            if self.max_steps and self.global_step >= self.max_steps:
                break
        return 0

    def _save_best(self, improved: List[str]):
        """Write the file(s) each improved criterion owns. best.pth is in
        `total`'s list, which is what makes it a copy of best_total.pth."""
        for key in improved:
            for name in BEST_SELECTION[key][1]:
                self.save(name)

    def _update_best_records(self, epoch: int,
                             val_metrics: Dict[str, float]) -> List[str]:
        """Which criteria improved this epoch. Every rank agrees on the answer.

        val_metrics is identical on every rank after Accumulator.all_reduce, so
        the four bars move in step even though only rank 0 writes files.
        """
        improved: List[str] = []
        for key, (metric, _files) in BEST_SELECTION.items():
            v = val_metrics.get(metric)
            if v is None or not isinstance(v, (int, float)) \
                    or not math.isfinite(float(v)):
                continue
            if float(v) < self.best_scores[key]:
                self.best_scores[key] = float(v)
                self.best_epochs[key] = epoch
                improved.append(key)
                if self.is_main:
                    files = ", ".join(BEST_SELECTION[key][1])
                    print(f"  new best {BEST_LABEL[key]}: {float(v):.5f} "
                          f"at epoch {epoch}  ({metric} -> {files})",
                          flush=True)
        return improved


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
