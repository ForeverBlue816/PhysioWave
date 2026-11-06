"""
Corpus index, lazy HDF5 windows, and a route-aware sampler that works under DDP.

Three problems this file exists to solve.

**Nothing may be loaded eagerly.** TUEG does not fit in memory and the top-level
pretrain.py loads every HDF5 it is given. Here a shard is opened per worker on
first touch and read window by window; the index holds counts and offsets, not
signal.

**A batch may not mix routes.** 19x1024 and 128x2048 are different shapes, and
padding one to the other would mean explaining the padding to the attention. So
a step picks one route and every sample in it comes from that route.

**Every rank must agree on the route, without talking.** If rank 0 runs E19 while
rank 1 runs E128, the all-reduce at the end of the step waits for gradients on
parameters the other rank never touched, and the job deadlocks. The route
sequence is therefore drawn from a generator seeded by (seed, epoch) alone --
identical on every rank by construction, no broadcast needed -- and only the
sample indices within the chosen route differ between ranks.

The mixture is enforced by the schedule, not by a sampler weight.
``WeightedRandomSampler`` is skipped outright under DistributedSampler (see
physiowave/train/data_builder.py), which is exactly how TUEG would have come to
be 90% of a run that was configured for 25%.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .routes import (PRETRAIN_DATASETS, ROUTES, Route,
                     balanced_sampling_weights,
                     proportional_sampling_weights,
                     temperature_sampling_weights)


# --------------------------------------------------------------------------- #
# HDF5 schema
# --------------------------------------------------------------------------- #

#: How many shards one dataset may hold open. Bounded because the alternative
#: is unbounded: nothing in the access pattern closes a file, and the corpus
#: has 94,938 of them. PW_MAX_OPEN_SHARDS overrides it.
MAX_OPEN_SHARDS = 512

#: Datasets every shard carries. `data` is [N, C, T] float32 in z-scored units;
#: everything else is per-window provenance of the same length N, except the
#: montage arrays which are per-channel and identical for the whole shard.
WINDOW_FIELDS = ("subject_id", "recording_id", "window_start_seconds")
MONTAGE_FIELDS = ("channel_names", "channel_ids", "valid_channel_mask")
SHARD_ATTRS = ("dataset_id", "route_id", "source_sampling_rate",
               "target_sampling_rate", "preprocessing_provenance")


@dataclass
class ShardInfo:
    """One HDF5 file's place in the corpus. No signal is held here."""

    path: str
    dataset_id: str
    route_id: str
    n_windows: int
    subjects: Tuple[str, ...] = ()

    @property
    def route(self) -> Route:
        return ROUTES[self.route_id]


@dataclass
class CorpusIndex:
    """Which shards exist, per (route, dataset), and how many windows each holds."""

    shards: List[ShardInfo] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "CorpusIndex":
        """Read a ``manifest_{train,val}.jsonl`` written by preprocessing."""
        shards: List[ShardInfo] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                shards.append(ShardInfo(
                    path=rec["path"],
                    dataset_id=rec["dataset_id"],
                    route_id=rec["route_id"],
                    n_windows=int(rec["n_windows"]),
                    subjects=tuple(rec.get("subjects", ())),
                ))
        if not shards:
            raise SystemExit(
                f"{manifest_path} lists no shards. Preprocessing wrote nothing, "
                f"or wrote somewhere else -- check preprocessing_failures.jsonl "
                f"beside it.")
        return cls(shards)

    def by_dataset(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for i, s in enumerate(self.shards):
            out.setdefault(s.dataset_id, []).append(i)
        return out

    def by_route(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for s in self.shards:
            members = out.setdefault(s.route_id, [])
            if s.dataset_id not in members:
                members.append(s.dataset_id)
        return out

    def window_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.shards:
            counts[s.dataset_id] = counts.get(s.dataset_id, 0) + s.n_windows
        return counts

    def total_windows(self) -> int:
        return sum(s.n_windows for s in self.shards)


class EEGWindowDataset(Dataset):
    """Windows of one dataset_id, addressed globally, read lazily.

    Files are opened on first access **in the worker that touches them** and
    cached there. An h5py handle cannot cross a fork, so opening in __init__ and
    inheriting it is the classic way to get silent corruption with
    num_workers > 0.
    """

    def __init__(self, index: CorpusIndex, dataset_id: str,
                 routes: Optional[Dict[str, Route]] = None):
        self.dataset_id = dataset_id
        self.shards = [s for s in index.shards if s.dataset_id == dataset_id]
        if not self.shards:
            raise KeyError(f"no shards for dataset {dataset_id!r}")
        self.route_id = self.shards[0].route_id
        # ``routes`` is the registry the corpus belongs to -- EEG's four by
        # default, and physiowave.ecg_c1 passes its own. The window reader is
        # identical for both; only the table of shapes differs.
        self.route = (ROUTES if routes is None else routes)[self.route_id]
        # offsets[i] is the global index of shard i's first window
        self.offsets = np.cumsum([0] + [s.n_windows for s in self.shards])
        self._handles: "OrderedDict[int, object]" = OrderedDict()
        self.max_open = max(1, int(os.environ.get("PW_MAX_OPEN_SHARDS")
                                   or MAX_OPEN_SHARDS))
        self._montage: Optional[Dict[str, torch.Tensor]] = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _handle(self, shard_i: int):
        """An open shard, least-recently-used ones closed to a bound.

        The cache used to be unbounded. A rank draws ~29,000 windows an epoch
        from a corpus of 94,938 shards, so it opened a file it would mostly
        never touch again and held it for the rest of the run: the process's
        file table grew without limit, every later open got slower, and 50
        epochs would have walked into the descriptor limit.

        A bound loses almost nothing. Sampling ~29,000 windows out of 15.4M
        makes a second touch on the same shard rare, so the hit rate is low at
        any cache size and the reads it saves are the few that were going to
        repeat anyway.
        """
        h = self._handles.get(shard_i)
        if h is not None:
            self._handles.move_to_end(shard_i)
            return h
        import h5py
        h = h5py.File(self.shards[shard_i].path, "r")
        self._handles[shard_i] = h
        while len(self._handles) > self.max_open:
            _, evicted = self._handles.popitem(last=False)
            try:
                evicted.close()
            except Exception:                                  # noqa: BLE001
                pass
        return h

    def montage(self) -> Dict[str, torch.Tensor]:
        """``channel_ids`` and ``valid_channel_mask`` for this dataset.

        One montage per dataset, read once. Preprocessing guarantees it is
        constant across a dataset's shards -- every window sits on the route's
        canonical slots -- so the model can build the channel code once per
        step instead of once per sample.
        """
        if self._montage is None:
            h = self._handle(0)
            self._montage = {
                "channel_ids": torch.as_tensor(h["channel_ids"][...]).long(),
                "valid_channel_mask": torch.as_tensor(
                    h["valid_channel_mask"][...]).bool(),
            }
        return self._montage

    def locate(self, i: int) -> Tuple[int, int]:
        shard_i = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return shard_i, int(i - self.offsets[shard_i])

    def __getitem__(self, i: int) -> Dict[str, object]:
        shard_i, local = self.locate(int(i))
        h = self._handle(shard_i)
        try:
            x = torch.from_numpy(np.asarray(h["data"][local], dtype=np.float32))
        except OSError as exc:
            # h5py reports a bad gzip chunk as "inflate() failed" and says
            # nothing about which of the corpus's ~95,000 files it was, which
            # leaves a 16-rank job with an unactionable traceback. A shard whose
            # header is intact but whose chunks are not is what a preprocessing
            # task killed mid-write leaves behind.
            raise OSError(
                f"{self.shards[shard_i].path}: window {local} of "
                f"{self.shards[shard_i].n_windows} is unreadable ({exc}).\n"
                f"  The shard's header survived but its data did not -- almost "
                f"always a preprocessing task killed part-way through the write."
                f"\n  Find every such shard, and drop them, with:\n"
                f"    python scripts/build_eeg_c1_manifest.py "
                f"--corpus-root $DATA_ROOT --check-shards --jobs 16"
            ) from exc
        return {
            "x": x,
            "route_id": self.route_id,
            "dataset_id": self.dataset_id,
            "index": int(i),
            "subject_id": _decode(h["subject_id"][local]),
            "recording_id": _decode(h["recording_id"][local]),
        }

    def read_batch(self, indices: Sequence[int]) -> torch.Tensor:
        """``[len(indices), C, T]`` in the order given, each shard read once.

        What the training loader uses instead of one ``__getitem__`` per
        window. Same values, in the same order; what it saves is the number of
        HDF5 calls: the windows of a batch that live in one shard are fetched
        in a single read, and the per-window subject and recording strings --
        two more reads each, which no training step looks at -- are not
        fetched at all. At 1,024 single-lead windows a step that was most of
        the step's time.
        """
        locs = [self.locate(int(i)) for i in indices]
        by_shard: Dict[int, List[Tuple[int, int]]] = {}
        for pos, (shard_i, local) in enumerate(locs):
            by_shard.setdefault(shard_i, []).append((local, pos))
        out: Optional[np.ndarray] = None
        for shard_i, items in by_shard.items():
            h = self._handle(shard_i)
            uniq = sorted({local for local, _ in items})
            try:
                block = np.asarray(h["data"][uniq], dtype=np.float32)
            except OSError as exc:
                raise OSError(
                    f"{self.shards[shard_i].path}: windows {uniq[:4]}... of "
                    f"{self.shards[shard_i].n_windows} are unreadable ({exc}). "
                    f"Check the corpus with scripts/build_eeg_c1_manifest.py "
                    f"--check-shards.") from exc
            if out is None:
                out = np.empty((len(locs),) + block.shape[1:], dtype=np.float32)
            row = {local: k for k, local in enumerate(uniq)}
            for local, pos in items:
                out[pos] = block[row[local]]
        return torch.from_numpy(out)

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def collate_windows(items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Stack one route's windows. Every item must share route and dataset."""
    routes = {it["route_id"] for it in items}
    if len(routes) != 1:
        raise RuntimeError(f"a batch mixed routes: {sorted(routes)}")
    return {
        "x": torch.stack([it["x"] for it in items]),
        "route_id": items[0]["route_id"],
        "dataset_id": items[0]["dataset_id"],
        "indices": [int(it["index"]) for it in items],
        "subject_ids": [it["subject_id"] for it in items],
        "recording_ids": [it["recording_id"] for it in items],
    }


# --------------------------------------------------------------------------- #
# Route-aware, DDP-safe schedule
# --------------------------------------------------------------------------- #

#: Conservative per-route micro-batch sizes. Token count varies by 6.7x across
#: routes, so one batch size would either waste the small routes or blow up on
#: E128_512. These keep tokens-per-step roughly level; gradient accumulation
#: keeps the optimizer step comparable regardless of which route a step drew.
DEFAULT_BATCH_BY_ROUTE: Dict[str, int] = {
    "E19_256": 64,
    "E32_512": 48,
    "E64_256": 24,
    "E128_512": 12,
}


class RouteSchedule:
    """The (route, dataset) each global step uses. Identical on every rank.

    Drawn from a generator seeded by ``(seed, epoch)`` and nothing else -- not
    by rank, not by wall clock, not by how many windows a dataset happens to
    hold. Two ranks that call ``set_epoch`` with the same epoch therefore
    produce the same sequence without exchanging a byte, which is what keeps the
    backward pass from deadlocking on a frontend only one rank ran.
    """

    def __init__(self, index: CorpusIndex, weights: Optional[Dict[str, float]] = None,
                 steps_per_epoch: Optional[int] = None, seed: int = 42,
                 batch_by_route: Optional[Dict[str, int]] = None,
                 num_replicas: int = 1, rank: int = 0,
                 routes: Optional[Dict[str, Route]] = None,
                 datasets: Optional[Dict[str, object]] = None,
                 fixed_weights: Optional[Dict[str, float]] = None,
                 epoch_pass_of: Optional[str] = None):
        self.index = index
        self.fixed_weights = dict(fixed_weights or {})
        self.epoch_pass_of = epoch_pass_of or None
        # The registry this schedule draws from. Defaults to the EEG corpus;
        # physiowave.ecg_c1 passes its own routes and datasets, and nothing
        # below reads the module-level tables directly.
        self.routes = dict(ROUTES if routes is None else routes)
        self.datasets = dict(PRETRAIN_DATASETS if datasets is None else datasets)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.batch_by_route = dict(batch_by_route or DEFAULT_BATCH_BY_ROUTE)
        self.epoch = 0
        self.start_step = 0

        available = set(index.by_dataset())
        counts = index.window_counts()

        # `weights` may be an explicit {dataset: weight} map, or one of the
        # named policies. The default is proportional: an epoch is one pass over
        # the corpus and nothing is revisited to fill a quota.
        self.full_pass_steps: Optional[Dict[str, int]] = None
        if isinstance(weights, str) and weights.lower().replace("-", "_") in (
                "full_pass", "epoch_pass"):
            # EVERY WINDOW, EVERY EPOCH. The other policies draw each step's
            # batch independently, so an epoch of "one pass in expectation"
            # leaves ~37% of windows unseen and shows others twice. Here each
            # dataset is permuted once per epoch and read through in order: it
            # gets exactly ceil(n / global batch) steps, the last one topped
            # up from the start of the permutation, and the steps of all the
            # datasets are shuffled together. The mixture is therefore the
            # proportional one, exactly rather than in expectation.
            self.weight_policy = "full_pass"
            self.full_pass_steps = {}
            for d, n in counts.items():
                if n <= 0 or d not in self.datasets:
                    continue
                g = self.batch_by_route[self.datasets[d].route_id] * self.num_replicas
                self.full_pass_steps[d] = int(-(-int(n) // g))
            w = {d: float(k) for d, k in self.full_pass_steps.items()}
            full = int(sum(self.full_pass_steps.values()))
            if steps_per_epoch is not None and int(steps_per_epoch) < full:
                # A cap (--smoke-test, --max-steps, STEPS_PER_EPOCH=): the
                # epoch is the first N steps of the shuffled full-pass plan,
                # and is then NOT a read of every window.
                print(f"  full_pass: epoch capped at {int(steps_per_epoch)} of "
                      f"the {full} steps one read of every window takes",
                      flush=True)
            else:
                steps_per_epoch = full
        elif weights is None or (isinstance(weights, str) and
                               weights.lower() in ("balanced", "uniform")):
            self.weight_policy = "balanced"
            w = balanced_sampling_weights(self.routes, self.datasets)
        elif isinstance(weights, str) and \
                weights.lower() in ("proportional", "size", "auto"):
            self.weight_policy = "proportional"
            w = proportional_sampling_weights(counts, self.batch_by_route,
                                              self.datasets)
        elif isinstance(weights, str) and weights.lower().startswith("temperature"):
            # "temperature:0.5" -- the dial between the two.
            try:
                alpha = float(weights.split(":", 1)[1])
            except (IndexError, ValueError):
                raise SystemExit(
                    f"{weights!r}: temperature needs an exponent, as in "
                    f"'temperature:0.5'. 1.0 is proportional, 0.0 is equal "
                    f"shares regardless of size.") from None
            if not 0.0 <= alpha <= 1.0:
                raise SystemExit(
                    f"temperature exponent {alpha} is outside [0, 1]. Above 1 "
                    f"amplifies the largest corpus beyond its own share.")
            self.weight_policy = f"temperature:{alpha:g}"
            w = temperature_sampling_weights(counts, alpha, self.batch_by_route,
                                             self.datasets)
        elif isinstance(weights, str):
            raise SystemExit(
                f"unknown sampling policy {weights!r}. Use 'full_pass' (every "
                f"window once an epoch), 'balanced' (P(route)=1/4), "
                f"'proportional' (one pass over the corpus in expectation), "
                f"'temperature:0.5' (between the two), or an explicit "
                f"{{dataset: weight}} mapping.")
        else:
            self.weight_policy = "explicit"
            w = dict(weights)

        w = {k: v for k, v in w.items() if k in available and v > 0}
        if not w:
            hint = ""
            if isinstance(weights, dict) and set(weights) & {
                    "temperature", "balanced", "proportional"}:
                # `--set data.weights="temperature: 0.5"` is YAML for a mapping;
                # only the spelling WITHOUT the space stays a string and reaches
                # the policy branch above.
                hint = ("\n\n  This looks like a policy written with a space: "
                        "'temperature: 0.5' is\n  YAML for a mapping, and "
                        "'temperature:0.5' is the string the policy\n  branch "
                        "reads. Write it without the space.")
            raise SystemExit(
                "no configured dataset is present in the manifest. Configured: "
                f"{sorted(weights) if weights else sorted(counts)}; present: "
                f"{sorted(available)}{hint}")
        total = sum(w.values())
        self.weights = {k: v / total for k, v in w.items()}
        # Fixed step shares for named corpora (continued pretraining: Siena at
        # 0.3 of the steps); the policy above shares out what is left among the
        # rest in its own proportions.
        fixed = {k: float(v) for k, v in (self.fixed_weights or {}).items() if float(v) > 0}
        if fixed:
            missing = sorted(set(fixed) - set(available))
            if missing:
                raise SystemExit(f"weights_fixed names {missing}, which the manifest "
                                 f"does not hold; present: {sorted(available)}")
            if sum(fixed.values()) >= 1.0:
                raise SystemExit(f"weights_fixed sums to {sum(fixed.values()):g}; it must "
                                 "leave a share for the other corpora")
            rest = {k: v for k, v in self.weights.items() if k not in fixed}
            r_total = sum(rest.values())
            if r_total <= 0:
                raise SystemExit("weights_fixed leaves no other corpus to share the rest")
            left = 1.0 - sum(fixed.values())
            self.weights = {**{k: v / r_total * left for k, v in rest.items()}, **fixed}
            self.weight_policy = f"{self.weight_policy} + fixed {fixed}"
        self.dataset_ids = sorted(self.weights)
        self.probs = np.array([self.weights[d] for d in self.dataset_ids],
                              dtype=np.float64)

        if steps_per_epoch is None and self.epoch_pass_of:
            # An epoch is one read of the named corpus at its share of the steps.
            d = self.epoch_pass_of
            if d not in self.weights:
                raise SystemExit(f"epoch_pass_of={d!r} is not in the mixture "
                                 f"{sorted(self.weights)}")
            per = self.batch_by_route[self.datasets[d].route_id] * self.num_replicas
            steps_per_epoch = int(max(1, np.ceil(counts.get(d, 0) / (self.weights[d] * per))))
        if steps_per_epoch is None:
            # One pass over the corpus in expectation: the dataset whose share
            # of the mixture is smallest relative to its size sets the length,
            # so no dataset is silently repeated many times per epoch. Under the
            # proportional policy every dataset hits that bound at once, and the
            # min is the exact one-pass length rather than a floor set by
            # whichever corpus is most starved.
            per_step = np.array(
                [self.batch_by_route[self.datasets[d].route_id] * self.num_replicas
                 for d in self.dataset_ids], dtype=np.float64)
            n = np.array([counts.get(d, 0) for d in self.dataset_ids],
                         dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                need = np.where(self.probs > 0, n / (self.probs * per_step), np.inf)
            steps_per_epoch = int(max(1, np.min(need[np.isfinite(need)])))
        self.steps_per_epoch = int(steps_per_epoch)

    def windows_per_step(self) -> float:
        """Expected windows a global step reads under this mixture."""
        return float(sum(
            p * self.batch_by_route[self.datasets[d].route_id] * self.num_replicas
            for d, p in zip(self.dataset_ids, self.probs)))

    def size_for_run(self, passes: float, epochs: int) -> int:
        """Set the epoch so ``epochs`` of them read ``passes`` x the corpus.

        A budget for the WHOLE RUN, in windows: with the corpus at N windows,
        ``epochs * steps * windows_per_step`` comes to ``passes * N``. The
        mixture still decides which corpus those windows come from -- under
        temperature:0.5 the small ones are read more than once and the large
        ones less -- so this sizes the total, not each corpus's share.
        """
        total = sum(self.index.window_counts().get(d, 0) for d in self.dataset_ids)
        steps = int(np.ceil(passes * total
                            / (max(1, int(epochs)) * self.windows_per_step())))
        self.steps_per_epoch = max(1, steps)
        return self.steps_per_epoch

    # -- state ------------------------------------------------------------- #
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.start_step = 0

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "seed": self.seed,
                "steps_per_epoch": self.steps_per_epoch,
                "start_step": self.start_step}

    def load_state_dict(self, state: Dict[str, int]):
        self.epoch = int(state.get("epoch", 0))
        self.seed = int(state.get("seed", self.seed))
        self.steps_per_epoch = int(state.get("steps_per_epoch",
                                             self.steps_per_epoch))
        self.start_step = int(state.get("start_step", 0))

    # -- the schedule ------------------------------------------------------ #
    def _rng(self) -> np.random.Generator:
        # Epoch enters the seed, so epoch N is the same sequence on every rank
        # and on every resume, and a different one from epoch N+1.
        return np.random.default_rng([self.seed, self.epoch])

    def plan(self) -> List[str]:
        """The dataset each step of this epoch draws from. Rank-independent."""
        rng = self._rng()
        if self.full_pass_steps is not None:
            plan = [d for d in self.dataset_ids
                    for _ in range(self.full_pass_steps.get(d, 0))]
            return [plan[i] for i in rng.permutation(len(plan))
                    ][:self.steps_per_epoch]
        return [self.dataset_ids[i] for i in
                rng.choice(len(self.dataset_ids), size=self.steps_per_epoch,
                           p=self.probs)]

    def realised_mixture(self) -> Dict[str, Dict[str, float]]:
        """This epoch's actual mixture, reported two ways, because they differ.

        ``by_step`` is the share of optimizer steps a dataset gets. This is what
        ``P(route)=1/4`` governs and what the configured weights mean.

        ``by_window`` is the share of *windows*, which is by_step reweighted by
        each route's micro-batch size. The two cannot both equal the configured
        weights while routes have different batch sizes, and they must: a
        128-channel window is 6.7x the tokens of a 19-channel one, so equal
        batch sizes would either waste E19_256 or exhaust memory on E128_512.
        So E19_256 draws a quarter of the steps and rather more than a quarter
        of the windows. Both are logged every epoch; neither is a bug, and
        which one is "the mixture" is a choice this records rather than hides.
        """
        plan = self.plan()
        per_step = {d: self.batch_by_route[self.datasets[d].route_id]
                    * self.num_replicas for d in self.dataset_ids}
        steps: Dict[str, int] = {}
        windows: Dict[str, int] = {}
        for d in plan:
            steps[d] = steps.get(d, 0) + 1
            windows[d] = windows.get(d, 0) + per_step[d]
        n_steps = sum(steps.values()) or 1
        n_windows = sum(windows.values()) or 1
        return {
            "by_step": {d: steps.get(d, 0) / n_steps for d in self.dataset_ids},
            "by_window": {d: windows.get(d, 0) / n_windows
                          for d in self.dataset_ids},
        }

    def __len__(self) -> int:
        return max(0, self.steps_per_epoch - self.start_step)

    def steps(self, lengths: Dict[str, int]) -> Iterator[Tuple[str, str, List[int]]]:
        """Yield ``(route_id, dataset_id, indices for THIS rank)`` per step.

        ``lengths`` maps dataset_id to how many windows it has.

        Within a step the ``batch * world_size`` positions are drawn WITHOUT
        replacement, so no two ranks are handed the same window and the
        gradient a step averages is over that many distinct recordings rather
        than over a few of them counted twice. Across steps the draw is
        independent: sampling the whole epoch without replacement would make a
        dataset's share of the steps proportional to its size again, which is
        precisely the imbalance the mixture exists to remove.

        A dataset smaller than one global batch is the one case where distinct
        positions do not exist; it falls back to replacement rather than
        silently shrinking the batch on one rank and deadlocking the all-reduce.
        """
        plan = self.plan()
        if self.full_pass_steps is not None:
            yield from self._full_pass_steps(plan, lengths)
            return
        rng = self._rng()
        # Advance the index stream past the draws the plan itself consumed, so
        # index draws are not correlated with the dataset choice.
        rng.integers(0, 2 ** 31 - 1, size=self.steps_per_epoch)

        for step, dataset_id in enumerate(plan):
            route_id = self.datasets[dataset_id].route_id
            per_rank = self.batch_by_route[route_id]
            n_total = per_rank * self.num_replicas
            n_avail = max(1, int(lengths.get(dataset_id, 0)))
            if n_avail >= n_total:
                picks = rng.choice(n_avail, size=n_total, replace=False)
            else:
                picks = rng.integers(0, n_avail, size=n_total)
            if step < self.start_step:
                continue        # resume: the draw is consumed, the step is not
            lo = self.rank * per_rank
            yield route_id, dataset_id, [int(v) for v in picks[lo:lo + per_rank]]

    def _full_pass_steps(self, plan: List[str], lengths: Dict[str, int]
                         ) -> Iterator[Tuple[str, str, List[int]]]:
        """``full_pass``: each dataset read through one permutation per epoch.

        Seeded by (seed, epoch) only, like everything else here, so every rank
        holds the same permutations and takes its own slice of each step.
        """
        rng = np.random.default_rng([self.seed, self.epoch, 1])
        perms = {d: rng.permutation(max(1, int(lengths.get(d, 0))))
                 for d in self.dataset_ids}
        pos = {d: 0 for d in self.dataset_ids}
        for step, dataset_id in enumerate(plan):
            route_id = self.datasets[dataset_id].route_id
            per_rank = self.batch_by_route[route_id]
            n_total = per_rank * self.num_replicas
            perm, p = perms[dataset_id], pos[dataset_id]
            if p + n_total <= perm.size:
                picks = perm[p:p + n_total]
            else:
                # The last batch: the remainder, topped up from the start --
                # repeated windows only when the dataset is smaller than one
                # global batch.
                picks = np.resize(np.concatenate([perm[p:], perm[:p]]), n_total)
            pos[dataset_id] = p + n_total
            if step < self.start_step:
                continue        # resume: the draw is consumed, the step is not
            lo = self.rank * per_rank
            yield route_id, dataset_id, [int(v) for v in picks[lo:lo + per_rank]]


class RouteBatchLoader:
    """Iterates a RouteSchedule, materialising one route's batch per step.

    Deliberately not a torch DataLoader: the unit of work is a step whose route
    is decided by the schedule, not a fixed dataset the loader walks. Workers
    would have to be told which dataset to read from per step, which is the
    sampler's job here.
    """

    def __init__(self, index: CorpusIndex, schedule: RouteSchedule,
                 prefetch: Optional[bool] = None):
        self.index = index
        self.schedule = schedule
        self.datasets = {d: EEGWindowDataset(index, d, routes=schedule.routes)
                         for d in schedule.dataset_ids}
        self.lengths = {d: len(ds) for d, ds in self.datasets.items()}
        # Read the next step's windows while this one trains. PW_PREFETCH=0
        # turns it off. A thread, not a worker process: h5py serialises its
        # own calls, and the main thread spends a step on the GPU, not in
        # h5py, so one reader overlaps the two without a second copy of every
        # open shard.
        self.prefetch = (os.environ.get("PW_PREFETCH", "1") != "0"
                         if prefetch is None else bool(prefetch))

    def __len__(self) -> int:
        return len(self.schedule)

    def _batch(self, route_id, dataset_id, idx):
        ds = self.datasets[dataset_id]
        return {"x": ds.read_batch(idx), "route_id": route_id,
                "dataset_id": dataset_id, "indices": [int(i) for i in idx],
                "channel_meta": ds.montage()}

    def __iter__(self):
        steps = self.schedule.steps(self.lengths)
        if not self.prefetch:
            for route_id, dataset_id, idx in steps:
                yield self._batch(route_id, dataset_id, idx)
            return
        # The schedule is a deterministic generator, so reading one step ahead
        # draws exactly the batches the serial loop would, in the same order.
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=1) as pool:
            pending = None
            for step in steps:
                nxt = pool.submit(self._batch, *step)
                if pending is not None:
                    yield pending.result()
                pending = nxt
            if pending is not None:
                yield pending.result()

    def close(self):
        for ds in self.datasets.values():
            ds.close()
