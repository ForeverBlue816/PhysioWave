"""The multimodal window dataset over the layout prep_common.py writes."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .prep_common import MODALITIES


class MultimodalWindows(Dataset):
    """One split's HDF5 file: per item the modalities' windows and their presence.

    Absence is a flag beside a zero window, never a zero window alone: the
    collate hands the flag to the fusion as a mask, and the zeros exist only
    so a batch stacks.
    """

    def __init__(self, path: str, modalities: Optional[Sequence[str]] = None):
        import h5py
        self.path = path
        with h5py.File(path, "r") as f:
            stored = json.loads(f.attrs["modalities"])
            self.modalities = [m for m in (modalities or stored) if m in stored]
            self.task = str(f.attrs["task"])
            self.class_names = json.loads(f.attrs["class_names"])
            self.window_seconds = float(f.attrs["window_seconds"])
            self.channel_names = {m: json.loads(f.attrs[f"{m}_channel_names"])
                                  for m in self.modalities}
            self.rates = {m: float(f.attrs[f"{m}_sampling_rate"]) for m in self.modalities}
            self.shapes = {m: tuple(f[m].shape[1:]) for m in self.modalities}
            self.n = int(f["label"].shape[0])
            self._labels = f["label"][:].astype(np.int64)
            self._present = {m: f[f"present_{m}"][:].astype(bool) for m in self.modalities}
            self._subjects = [s.decode() if isinstance(s, bytes) else str(s)
                              for s in f["subject"][:]]
        self._f = None

    def __len__(self) -> int:
        return self.n

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def present(self) -> Dict[str, np.ndarray]:
        return self._present

    @property
    def subjects(self) -> List[str]:
        return self._subjects

    def class_counts(self) -> np.ndarray:
        return np.bincount(self._labels, minlength=len(self.class_names))

    def _file(self):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.path, "r")       # opened in the worker that reads
        return self._f

    def __getitem__(self, i: int) -> dict:
        f = self._file()
        item = {"x": {}, "present": {}, "label": int(self._labels[i]),
                "subject": self._subjects[i],
                "segment": f["segment"][i].decode() if isinstance(f["segment"][i], bytes)
                else str(f["segment"][i]), "index": i}
        for m in self.modalities:
            item["x"][m] = torch.from_numpy(np.asarray(f[m][i], dtype=np.float32))
            item["present"][m] = bool(self._present[m][i])
        return item


def collate_multimodal(items: Sequence[dict]) -> dict:
    mods = list(items[0]["x"])
    return {
        "x": {m: torch.stack([it["x"][m] for it in items]) for m in mods},
        "present": {m: torch.tensor([it["present"][m] for it in items], dtype=torch.bool)
                    for m in mods},
        "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
        "subject": [it["subject"] for it in items],
        "segment": [it["segment"] for it in items],
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
    }


def batch_to(batch: dict, device) -> dict:
    out = dict(batch)
    out["x"] = {m: v.to(device, non_blocking=True) for m, v in batch["x"].items()}
    out["present"] = {m: v.to(device) for m, v in batch["present"].items()}
    out["label"] = batch["label"].to(device)
    return out


__all__ = ["MODALITIES", "MultimodalWindows", "collate_multimodal", "batch_to"]
