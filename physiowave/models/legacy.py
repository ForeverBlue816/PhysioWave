"""Legacy PhysioWave compatibility layer.

The original top-level modules (``model.py``, ``wavelet_modules.py``,
``transformer_modules.py``, ``head_modules.py``, ``dataset.py``, ``pretrain.py``,
``finetune.py``) are left untouched.  This module only makes them importable from
inside the ``physiowave`` package and exposes a builder so the legacy path can be
selected from a config like any other model.

``tests/test_legacy.py`` runs a forward regression: with every new module turned
off, the legacy entry point must still produce the same output shapes as the
original implementation.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_repo_on_path() -> None:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def legacy_available() -> bool:
    """True if the original top-level modules can be imported."""
    _ensure_repo_on_path()
    try:
        import model  # noqa: F401
        return True
    except Exception:
        return False


def build_legacy_model(**kwargs: Any):
    """Instantiate the original :class:`BERTWaveletTransformer` unchanged.

    Every keyword is forwarded verbatim, so the legacy CLI defaults keep working.
    """
    _ensure_repo_on_path()
    from model import BERTWaveletTransformer

    return BERTWaveletTransformer(**kwargs)


class LegacyForFinetuneMain(nn.Module):
    """The legacy model behind the interface ``finetune_main`` expects.

    Exists to make one experiment possible: run the original architecture through
    the *new* training loop, same data, same schedule, same metrics. When the two
    paths disagree on a benchmark that is the only way to tell an architecture
    difference from a pipeline difference -- optimizer groups, warmup, label
    smoothing, evaluation and checkpoint selection all differ between
    ``finetune.py`` and ``physiowave.train.finetune_main``, and any of them could
    account for a gap that looks architectural.

    The legacy ``forward`` takes a task name as its second positional argument,
    where this interface passes ``ChannelMeta``; the adapter swallows the
    metadata (the legacy model has no use for it) and returns the dict shape.
    """

    def __init__(self, **legacy_kwargs: Any) -> None:
        super().__init__()
        self.model = build_legacy_model(**legacy_kwargs)

    def forward(self, x: torch.Tensor, meta: Any = None, **_: Any) -> Dict[str, Any]:
        out = self.model(x, task="downstream", task_name="classification")
        logits = out["logits"] if isinstance(out, dict) else out
        return {"logits": logits, "pooled": logits}


def legacy_token_count(in_channels: int, max_level: int, T: int, patch_size) -> Dict[str, int]:
    """Token count of the legacy path, for the token-efficiency table.

    The legacy wavelet stage upsamples every band back to ``T`` and concatenates
    them on the channel axis, producing a ``[B, 1, (J+1)*C, T]`` "image" that is
    then cut into ``patch_size`` patches.  So the sequence length is
    ``((J+1)*C / p_f) * (T / p_t)`` -- linear in the number of decomposition
    levels, which is exactly the inflation WAST removes.
    """
    p_f, p_t = patch_size if isinstance(patch_size, (tuple, list)) else (1, patch_size)
    freq_rows = (max_level + 1) * in_channels
    assert freq_rows % p_f == 0, "legacy patch_size[0] must divide (J+1)*C"
    assert T % p_t == 0, "legacy patch_size[1] must divide T"
    return {
        "rows": freq_rows,
        "patches_per_freq": freq_rows // p_f,
        "patches_per_time": T // p_t,
        "num_tokens": (freq_rows // p_f) * (T // p_t),
    }


class LegacyWithChannelID(nn.Module):
    """Legacy model plus a learnable per-channel identity embedding.

    This is rung 2 of the tier-1 ablation ladder ("legacy + channel ID").  It is
    the weakest possible way to tell the model which electrode is which: a free
    vector per channel index, with no geometry and no metadata.  Because it is
    indexed by *position in the array*, it cannot generalise to a permuted or
    differently-sized montage -- which is exactly the limitation the rest of the
    ladder is measured against.

    The legacy module itself is not modified: the embedding is added to the input
    signal before it enters the unchanged legacy forward.
    """

    def __init__(self, in_channels: int, scale: float = 0.1, **legacy_kwargs: Any) -> None:
        super().__init__()
        self.backbone = build_legacy_model(in_channels=in_channels, **legacy_kwargs)
        self.channel_id = nn.Parameter(torch.zeros(in_channels, 1))
        self.scale = float(scale)
        nn.init.normal_(self.channel_id, std=0.02)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
        assert x.shape[1] == self.channel_id.shape[0], (
            f"channel-ID embedding was built for {self.channel_id.shape[0]} channels, "
            f"got {x.shape[1]}. This variant is montage-specific by construction."
        )
        return self.backbone(x + self.scale * self.channel_id.unsqueeze(0), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["backbone"], name)
