"""Dataset objects, weighted multi-corpus sampling and dataloader construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .manifest import ManifestEntry
from .montages import positions_for
from .preprocess import PreprocessCache, PreprocessConfig, preprocess
from .registry import DatasetSpec
from .schema import Sample, collate_samples

logger = logging.getLogger(__name__)


class HDF5WindowDataset(Dataset):
    """Windows stored as ``[N, C, T]`` in an HDF5 file, one :class:`Sample` each.

    Files are opened lazily per worker: an ``h5py.File`` handle cannot be shared
    across forked processes.
    """

    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        spec: DatasetSpec,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        data_key: str = "data",
        label_key: str = "label",
    ) -> None:
        self.entries = list(entries)
        self.spec = spec
        self.pp = preprocess_cfg
        self.cache = PreprocessCache(preprocess_cfg) if preprocess_cfg else None
        self.data_key, self.label_key = data_key, label_key
        self._handles: Dict[str, Any] = {}

        names = self.entries[0].channel_names if self.entries else []
        if names:
            self.channel_names = names
            self.xyz, known = positions_for(names)
            if spec.modality == "eeg" and not bool(known.all()):
                logger.warning(
                    "Dataset %s: %d/%d channels have no template coordinates. The SSL "
                    "branch will be unavailable for those montages.",
                    spec.dataset_id, int((~known).sum()), len(names),
                )
        else:
            C = spec.num_channels or 0
            self.channel_names = [f"CH{i}" for i in range(C)]
            self.xyz = torch.zeros(C, 3)
            if spec.modality == "eeg":
                logger.warning(
                    "Dataset %s ships no channel names; EEG spatial modules will run "
                    "with the unknown-coordinate fallback and the SSL branch disabled.",
                    spec.dataset_id,
                )

    def _file(self, path: str):
        import h5py

        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Sample:
        e = self.entries[i]
        key = f"{e.path}:{e.index}"
        arr = self.cache.get(key) if self.cache else None
        fs = self.pp.target_sampling_rate if (self.pp and self.pp.target_sampling_rate) \
            else e.sampling_rate
        if arr is None:
            f = self._file(e.path)
            raw = np.asarray(f[self.data_key][e.index], dtype=np.float32)
            if self.pp:
                raw, fs = preprocess(raw, e.sampling_rate, self.pp)
                if self.cache:
                    self.cache.put(key, raw)
            arr = raw
        label = None
        if self.label_key in self._file(e.path):
            lv = self._file(e.path)[self.label_key][e.index]
            label = torch.as_tensor(np.asarray(lv))
        C = arr.shape[0]
        names = self.channel_names[:C] if len(self.channel_names) >= C \
            else [f"CH{j}" for j in range(C)]
        xyz = self.xyz[:C] if self.xyz.shape[0] >= C else torch.zeros(C, 3)
        return Sample(
            signal=torch.from_numpy(np.ascontiguousarray(arr)),
            modality=e.modality, sampling_rate=fs,
            subject_id=e.subject_id, recording_id=e.recording_id, dataset_id=e.dataset_id,
            channel_names=list(names), channel_xyz=xyz.clone(),
            channel_mask=torch.ones(C, dtype=torch.bool),
            montage_type=e.montage_type, reference_type=e.reference_type,
            derivation_type=e.derivation_type, label=label,
            window_start=e.window_start, window_end=e.window_end,
            emg_region=e.emg_region,
        ).validate()


@dataclass
class LoaderConfig:
    """Dataloader settings."""

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    persistent_workers: bool = False


def weighted_sampler(
    datasets: Sequence[Dataset], weights: Sequence[float], num_samples: Optional[int] = None
) -> WeightedRandomSampler:
    """Sampler that draws from several corpora at a configured mixture ratio.

    Each corpus's per-item weight is ``w_d / len(d)``, so the *corpus* mixture
    matches ``weights`` regardless of how many windows each one contributes.
    """
    assert len(datasets) == len(weights), "one weight per dataset"
    per_item: List[float] = []
    for ds, w in zip(datasets, weights, strict=True):
        n = len(ds)
        per_item.extend([float(w) / max(n, 1)] * n)
    total = num_samples or sum(len(d) for d in datasets)
    return WeightedRandomSampler(per_item, num_samples=total, replacement=True)


def build_dataloader(
    dataset: Dataset,
    cfg: LoaderConfig,
    distributed: bool = False,
    sampler: Optional[Any] = None,
    seed: int = 42,
) -> DataLoader:
    """Build a dataloader, using a DistributedSampler under DDP."""
    if distributed and sampler is None:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=cfg.shuffle, seed=seed)
    # MPS has no pinned host memory; asking for it only emits a warning.
    pin_memory = cfg.pin_memory and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(cfg.shuffle and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=collate_samples,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
