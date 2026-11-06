"""Manifest construction, dataset statistics and checksums.

A manifest is a JSON file listing every analysis window with the metadata needed
to interpret it and to split it safely.  Building it up front means the split
checker, the statistics and the weighted sampler all see the same view of the
data, and a run can be reproduced from the manifest alone.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .registry import DatasetSpec

logger = logging.getLogger(__name__)


def file_checksum(path: str, chunk: int = 1 << 20, max_bytes: Optional[int] = None) -> str:
    """SHA-1 of a file (optionally only its first ``max_bytes``, for huge files)."""
    h = hashlib.sha1()
    read = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
            read += len(block)
            if max_bytes and read >= max_bytes:
                break
    return h.hexdigest()


@dataclass
class ManifestEntry:
    """One window (or one file, when windows are cut on the fly)."""

    dataset_id: str
    modality: str
    path: str
    subject_id: str
    recording_id: str
    sampling_rate: float
    num_channels: int
    num_samples: int
    channel_names: List[str] = field(default_factory=list)
    montage_type: str = "unknown"
    reference_type: str = "unknown"
    derivation_type: str = "monopolar"
    emg_region: str = "unknown"
    has_coordinates: bool = True
    label: Optional[Any] = None
    window_start: float = 0.0
    window_end: float = 0.0
    checksum: Optional[str] = None
    index: int = 0


def build_manifest(
    spec: DatasetSpec,
    root: Optional[str] = None,
    data_key: str = "data",
    label_key: str = "label",
    compute_checksums: bool = True,
    max_files: Optional[int] = None,
) -> List[ManifestEntry]:
    """Scan an HDF5 corpus and produce manifest entries.

    Subject ids come from ``spec.subject_from_path`` when given; otherwise the
    file stem is used, which keeps a file-level split subject-safe only if one
    file is one subject.  A warning is emitted when that assumption is being made
    implicitly, because getting it wrong silently reintroduces leakage.
    """
    import h5py

    root = root or spec.root
    if not root:
        raise ValueError(
            f"Dataset {spec.dataset_id} has no root path. Set it in the config; "
            "datasets are never downloaded automatically."
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    files = sorted(glob.glob(os.path.join(root, "**", spec.file_glob), recursive=True))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {spec.file_glob!r} under {root}")

    if not spec.subject_from_path:
        logger.warning(
            "Dataset %s has no subject_from_path regex; falling back to the file "
            "stem as the subject id. If one file contains several subjects this "
            "will leak across splits -- set subject_from_path in the registry.",
            spec.dataset_id,
        )

    entries: List[ManifestEntry] = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        subject = stem
        if spec.subject_from_path:
            m = re.search(spec.subject_from_path, path)
            if m:
                subject = m.group(1)
        with h5py.File(path, "r") as f:
            if data_key not in f:
                logger.warning("Skipping %s: no '%s' dataset", path, data_key)
                continue
            d = f[data_key]
            n, C, T = (d.shape + (1, 1))[:3] if d.ndim == 3 else (1,) + d.shape[:2]
            names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in f["channel_names"][:]] if "channel_names" in f else []
            has_labels = label_key in f
        checksum = file_checksum(path, max_bytes=1 << 24) if compute_checksums else None
        dur = T / spec.sampling_rate
        for i in range(int(n)):
            entries.append(ManifestEntry(
                dataset_id=spec.dataset_id, modality=spec.modality, path=path,
                subject_id=str(subject), recording_id=f"{stem}", index=i,
                sampling_rate=spec.sampling_rate, num_channels=int(C), num_samples=int(T),
                channel_names=names, montage_type=spec.montage or "unknown",
                reference_type=spec.reference_type, derivation_type=spec.derivation_type,
                emg_region=spec.emg_region, has_coordinates=spec.has_coordinates,
                label=None if not has_labels else "in_file",
                window_start=i * dur, window_end=(i + 1) * dur, checksum=checksum,
            ))
    return entries


def save_manifest(entries: Sequence[ManifestEntry], path: str) -> str:
    """Write a manifest atomically."""
    payload = {"version": 1, "count": len(entries),
               "entries": [asdict(e) for e in entries]}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    return path


def load_manifest(path: str) -> List[ManifestEntry]:
    with open(path) as f:
        payload = json.load(f)
    return [ManifestEntry(**e) for e in payload["entries"]]


def manifest_stats(entries: Sequence[ManifestEntry]) -> Dict[str, Any]:
    """Summary statistics used in reports and in the ``--dry-run`` output."""
    if not entries:
        return {"count": 0}
    subjects = {e.subject_id for e in entries}
    recordings = {e.recording_id for e in entries}
    hours = sum(e.window_end - e.window_start for e in entries) / 3600.0
    return {
        "count": len(entries),
        "datasets": sorted({e.dataset_id for e in entries}),
        "modalities": sorted({e.modality for e in entries}),
        "num_subjects": len(subjects),
        "num_recordings": len(recordings),
        "channels": sorted({e.num_channels for e in entries}),
        "sampling_rates": sorted({e.sampling_rate for e in entries}),
        "total_hours": round(hours, 3),
        "without_coordinates": sorted({e.dataset_id for e in entries if not e.has_coordinates}),
    }
