"""Unified sample schema shared by every dataset in the registry.

Every dataset -- EEG, ECG or limb sEMG, synthetic or real -- yields a
:class:`Sample` with the same fields.  Downstream code (TARE, the SSL branch, the
reference augmentations, the split checker) reads metadata only through this
schema, so adding a dataset never requires touching the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

MODALITIES = ("eeg", "ecg", "semg")
EMG_REGIONS = ("limb", "facial", "trunk", "unknown")


@dataclass
class Sample:
    """One analysis window plus everything needed to interpret it."""

    signal: torch.Tensor                       # [C, T] float32
    modality: str
    sampling_rate: float
    subject_id: str
    recording_id: str
    dataset_id: str
    channel_names: List[str]
    channel_xyz: torch.Tensor                  # [C, 3]; all-zero row = unknown
    channel_mask: torch.Tensor                 # [C] bool; False = missing/bad
    montage_type: str = "unknown"
    reference_type: str = "unknown"
    reference_channel: Optional[str] = None
    derivation_type: str = "monopolar"
    bipolar_endpoints: Optional[List[List[str]]] = None
    channel_quality: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    window_start: float = 0.0                  # seconds from the recording start
    window_end: float = 0.0
    emg_region: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Sample:
        """Assert the internal invariants every consumer relies on."""
        assert self.modality in MODALITIES, f"unknown modality {self.modality!r}"
        assert self.signal.dim() == 2, f"signal must be [C, T], got {tuple(self.signal.shape)}"
        C = self.signal.shape[0]
        assert len(self.channel_names) == C, (
            f"{len(self.channel_names)} channel names for {C} channels"
        )
        assert self.channel_xyz.shape == (C, 3), (
            f"channel_xyz must be [C, 3], got {tuple(self.channel_xyz.shape)}"
        )
        assert self.channel_mask.shape == (C,), (
            f"channel_mask must be [C], got {tuple(self.channel_mask.shape)}"
        )
        assert self.sampling_rate > 0, "sampling_rate must be positive"
        if self.modality == "semg":
            assert self.emg_region in EMG_REGIONS, f"unknown emg_region {self.emg_region!r}"
        return self


def collate_samples(batch: Sequence[Sample]) -> Dict[str, Any]:
    """Collate a list of :class:`Sample` into batched tensors.

    Channel counts must agree inside a batch; mixing montages is handled by the
    sampler (one montage per batch) rather than by padding, because padding a
    channel axis with zeros would feed the model electrodes that do not exist.
    """
    assert batch, "empty batch"
    ref = batch[0]
    C = ref.signal.shape[0]
    for s in batch:
        assert s.signal.shape[0] == C, (
            "all samples in a batch must share a channel count; got "
            f"{s.signal.shape[0]} vs {C}. Use a montage-homogeneous sampler."
        )
    out: Dict[str, Any] = {
        "signal": torch.stack([s.signal for s in batch]),
        "modality": ref.modality,
        "sampling_rate": ref.sampling_rate,
        "subject_id": [s.subject_id for s in batch],
        "recording_id": [s.recording_id for s in batch],
        "dataset_id": [s.dataset_id for s in batch],
        "channel_names": ref.channel_names,
        "channel_xyz": ref.channel_xyz,
        "channel_mask": torch.stack([s.channel_mask for s in batch]).all(dim=0),
        "montage_type": ref.montage_type,
        "reference_type": ref.reference_type,
        "reference_channel": ref.reference_channel,
        "derivation_type": ref.derivation_type,
        "bipolar_endpoints": ref.bipolar_endpoints,
        "window_start": torch.tensor([s.window_start for s in batch]),
        "window_end": torch.tensor([s.window_end for s in batch]),
        "emg_region": ref.emg_region,
    }
    if ref.label is not None:
        out["label"] = torch.stack([s.label for s in batch])
    if ref.channel_quality is not None:
        out["channel_quality"] = torch.stack([s.channel_quality for s in batch]).mean(0)
    return out


def batch_to_meta(batch: Dict[str, Any]):
    """Build a :class:`physiowave.channels.tare.ChannelMeta` from a collated batch."""
    from ..channels.tare import ChannelMeta

    return ChannelMeta(
        channel_names=batch["channel_names"],
        channel_xyz=batch["channel_xyz"],
        channel_mask=batch.get("channel_mask"),
        channel_quality=batch.get("channel_quality"),
        montage_type=batch.get("montage_type", "unknown"),
        reference_type=batch.get("reference_type", "unknown"),
        reference_channel=batch.get("reference_channel"),
        derivation_type=batch.get("derivation_type", "monopolar"),
        bipolar_endpoints=batch.get("bipolar_endpoints"),
    )
