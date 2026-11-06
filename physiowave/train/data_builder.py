"""Turn a ``data:`` config block into train/val datasets.

Falls back to the synthetic corpus when no dataset roots are configured, which is
what makes ``run_tpami.sh smoke`` and every unit test work without any protected
data.  Real corpora are built from manifests; nothing is ever downloaded.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import ConcatDataset, Dataset

from ..config import instantiate
from ..data.datasets import HDF5WindowDataset
from ..data.manifest import build_manifest, load_manifest, manifest_stats, save_manifest
from ..data.preprocess import PreprocessConfig
from ..data.registry import assert_limb_semg
from ..data.registry import get as get_spec
from ..data.splits import split_records
from ..data.synthetic import SyntheticConfig, SyntheticDataset

logger = logging.getLogger(__name__)


def build_synthetic(cfg: Dict[str, Any], modality: str) -> Tuple[Dataset, Dataset]:
    """Synthetic train/val pair; validation uses a disjoint subject seed."""
    s = dict(cfg.get("synthetic", {}) or {})
    base = SyntheticConfig(
        modality=modality,
        num_samples=int(s.get("num_samples", 128)),
        num_subjects=int(s.get("num_subjects", 8)),
        window_samples=int(s.get("window_samples", 1024)),
        sampling_rate=float(s.get("sampling_rate", 256.0)),
        num_channels=s.get("num_channels"),
        montage_name=s.get("montage_name", "standard_1010_64"),
        num_classes=s.get("num_classes", 4),
        seed=int(s.get("seed", 0)),
        missing_channel_prob=float(s.get("missing_channel_prob", 0.0)),
    )
    val = SyntheticConfig(**{**base.__dict__, "num_samples": max(base.num_samples // 4, 4),
                             "seed": base.seed + 1000})
    return SyntheticDataset(base), SyntheticDataset(val)


def build_real(cfg: Dict[str, Any], modality: str) -> Tuple[Dataset, Optional[Dataset], Dict[str, Any]]:
    """Build train/val datasets from the registry entries listed in ``cfg``."""
    ids: Sequence[str] = cfg.get("datasets") or []
    roots: Dict[str, str] = cfg.get("roots", {}) or {}
    specs = [get_spec(i) for i in ids]
    if modality == "semg":
        assert_limb_semg(specs)

    pp = instantiate(PreprocessConfig, cfg.get("preprocess", {}) or {})
    manifest_dir = cfg.get("manifest_dir") or "./manifests"
    train_parts, val_parts, stats = [], [], {}
    for spec in specs:
        root = roots.get(spec.dataset_id) or spec.root
        if spec.requires_agreement and not root:
            raise ValueError(
                f"Dataset {spec.dataset_id} requires a data use agreement and is never "
                "downloaded automatically. Provide data.roots."
                f"{spec.dataset_id} pointing at your local copy."
            )
        mpath = os.path.join(manifest_dir, f"{spec.dataset_id}.json")
        if os.path.exists(mpath):
            entries = load_manifest(mpath)
        else:
            entries = build_manifest(spec, root)
            save_manifest(entries, mpath)
        stats[spec.dataset_id] = manifest_stats(entries)
        records = [{"subject_id": e.subject_id, "recording_id": e.recording_id, "entry": e}
                   for e in entries]
        parts = split_records(records, tuple(cfg.get("split_ratios", [0.8, 0.1, 0.1])))
        train_parts.append(HDF5WindowDataset([r["entry"] for r in parts["train"]], spec, pp))
        if parts["val"]:
            val_parts.append(HDF5WindowDataset([r["entry"] for r in parts["val"]], spec, pp))
    train = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val = None if not val_parts else (val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts))
    stats["_parts"] = train_parts          # kept so the weighted sampler can be built
    return train, val, stats


def build_datasets(cfg: Dict[str, Any], modality: str) -> Tuple[Dataset, Optional[Dataset], Dict[str, Any]]:
    """Dispatch to the real or synthetic builder and return dataset statistics."""
    if cfg.get("datasets"):
        return build_real(cfg, modality)
    logger.warning(
        "No datasets configured for modality %s; using the SYNTHETIC smoke corpus. "
        "Any number produced from this run is a smoke-test number, not a result.",
        modality,
    )
    train, val = build_synthetic(cfg, modality)
    return train, val, {"synthetic": {"count": len(train), "val_count": len(val)}}


def maybe_weighted_sampler(cfg: Dict[str, Any], stats: Dict[str, Any], distributed: bool):
    """Weighted multi-corpus sampler, or ``None`` when it does not apply.

    Each corpus's per-item weight is ``w_d / len(d)``, so the *corpus* mixture
    matches ``data.weights`` regardless of how many windows each one contributes --
    without that normalisation a 2 TB corpus silently drowns out a 5 GB one no
    matter what ratio the config asks for.

    Returns ``None`` under DDP: mixing a weighted sampler with a distributed
    sampler would either duplicate or drop data. Set the mixture through
    per-dataset window counts instead when training distributed.
    """
    from ..data.datasets import weighted_sampler

    parts = stats.get("_parts") or []
    if len(parts) < 2:
        return None
    if distributed:
        logger.warning(
            "data.weights is set but a DistributedSampler is in use; the weighted "
            "multi-corpus sampler is skipped for this run (the two cannot be "
            "composed without duplicating or dropping windows)."
        )
        return None
    weights = dataset_weights(cfg)
    logger.info("Using a weighted multi-corpus sampler with mixture %s", weights)
    return weighted_sampler(parts, weights)


def dataset_weights(cfg: Dict[str, Any]) -> List[float]:
    ids = cfg.get("datasets") or []
    w = cfg.get("weights") or []
    if not ids:
        return []
    if not w:
        return [1.0 / len(ids)] * len(ids)
    assert len(w) == len(ids), "data.weights must have one entry per dataset"
    total = sum(w)
    return [x / total for x in w]
