"""Physically legal EEG re-reference views.

Constraint C (``docs/terminology.md``) in code
----------------------------------------------
Re-referencing is a **linear transformation of the channel axis**:
``V' = (I - 1 w^T) V`` where ``w`` is a weight vector over the *recorded*
channels.  Three consequences shape this module:

1. Only views that can be written that way may be constructed.  A "reference"
   signal that is not a linear combination of recorded channels does not exist in
   the data and inventing one would fabricate a recording that was never made.
   Every view here returns its operator ``M`` alongside the signal, and
   ``tests/test_reference.py`` checks that ``view == M @ X`` exactly.
2. Reference invariance of the learned representation is therefore a
   well-posed objective: all these views span the same measured field.
3. The surface Laplacian annihilates the ``1`` direction, so it is reference
   invariant and can serve as the anchor of that objective (see
   :mod:`physiowave.spatial.spline_laplacian`).

Lateralisation
--------------
A single-sided reference (one ear, one mastoid, one arbitrary channel) subtracts
a signal recorded over one hemisphere from every channel, which injects a
systematic left/right asymmetry.  Views are therefore split into two tiers:

``standard_views``  ``original``, ``common_average``, ``linked_mastoids`` -- balanced
``hard_views``      ``left_ear``, ``right_ear``, ``left_mastoid``, ``right_mastoid``,
                    ``random_channel`` -- lateralised

Both are used during pretraining (hard views at a configurable probability), but
a hard view is **never** the anchor of the consistency loss, and downstream
supervised evaluation defaults to a standard view; hard views appear only in the
reference-robustness evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..channels.tare import ChannelMeta
from ..data.montages import canonical_name

logger = logging.getLogger(__name__)

#: Skip reasons already reported, so a per-batch skip is logged once, not N times.
_REPORTED_SKIPS: set = set()


def _warn_once(message: str) -> None:
    if message not in _REPORTED_SKIPS:
        _REPORTED_SKIPS.add(message)
        logger.warning("%s", message)

STANDARD_VIEWS = ("original", "common_average", "linked_mastoids")
HARD_VIEWS = ("left_ear", "right_ear", "left_mastoid", "right_mastoid", "random_channel")

_MASTOID_LABELS = {"left_mastoid": ("M1", "TP9"), "right_mastoid": ("M2", "TP10")}
_EAR_LABELS = {"left_ear": ("A1",), "right_ear": ("A2",)}


@dataclass
class ReferenceConfig:
    """Configuration of reference augmentation."""

    enabled: bool = True
    standard_views: Sequence[str] = field(default_factory=lambda: list(STANDARD_VIEWS))
    hard_views: Sequence[str] = field(default_factory=lambda: list(HARD_VIEWS))
    hard_view_prob: float = 0.2
    num_views: int = 2
    car_min_channels: int = 32       # below this, a common average is not representative


def _find(names: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    canon = [canonical_name(n) for n in names]
    for c in candidates:
        if c in canon:
            return canon.index(c)
    return None


def reference_operator(
    view: str,
    meta: ChannelMeta,
    cfg: ReferenceConfig,
    generator: Optional[torch.Generator] = None,
    device=None,
) -> Optional[Tuple[torch.Tensor, Dict[str, object]]]:
    """Build ``M`` such that ``X_view = M @ X``, or ``None`` if the view is illegal.

    Returns ``(M [C, C], info)``.  ``info['skipped']`` explains why a view was not
    produced (too few channels for a common average, no mastoid channel present,
    a bipolar montage, ...); the caller logs it rather than silently continuing.
    """
    names = list(meta.channel_names)
    C = len(names)
    device = device if device is not None else torch.device("cpu")
    eye = torch.eye(C, device=device)
    ones = torch.ones(C, 1, device=device)

    if meta.derivation_type and meta.derivation_type.lower().startswith("bipolar"):
        return None if view != "original" else (eye, {"view": "original", "tier": "standard"})

    good = torch.ones(C, dtype=torch.bool) if meta.channel_mask is None \
        else meta.channel_mask.to(torch.bool)

    if view == "original":
        return eye, {"view": view, "tier": "standard"}

    if view == "common_average":
        n_good = int(good.sum().item())
        if n_good < cfg.car_min_channels:
            return None, {"skipped": (
                f"common_average skipped: {n_good} good channels < "
                f"car_min_channels={cfg.car_min_channels}; an average over too few "
                "electrodes is not a neutral reference and biases the montage."
            )}
        w = good.to(torch.float32).to(device)
        w = (w / w.sum()).view(C, 1)
        return eye - ones @ w.transpose(0, 1), {"view": view, "tier": "standard"}

    if view == "linked_mastoids":
        i = _find(names, _MASTOID_LABELS["left_mastoid"])
        j = _find(names, _MASTOID_LABELS["right_mastoid"])
        if i is None or j is None:
            i = _find(names, _EAR_LABELS["left_ear"])
            j = _find(names, _EAR_LABELS["right_ear"])
        if i is None or j is None:
            return None, {"skipped": "linked_mastoids skipped: no mastoid/ear channel pair "
                                     "in this montage."}
        w = torch.zeros(C, 1, device=device)
        w[i] = w[j] = 0.5
        return eye - ones @ w.transpose(0, 1), {"view": view, "tier": "standard"}

    if view in _EAR_LABELS or view in _MASTOID_LABELS:
        labels = _EAR_LABELS.get(view) or _MASTOID_LABELS[view]
        i = _find(names, labels)
        if i is None:
            return None, {"skipped": f"{view} skipped: none of {labels} present."}
        w = torch.zeros(C, 1, device=device)
        w[i] = 1.0
        return eye - ones @ w.transpose(0, 1), {"view": view, "tier": "hard",
                                                "lateralised": True}

    if view == "random_channel":
        idx_pool = torch.nonzero(good, as_tuple=False).flatten()
        if idx_pool.numel() == 0:
            return None, {"skipped": "random_channel skipped: no good channels."}
        pick = idx_pool[torch.randint(idx_pool.numel(), (1,), generator=generator).item()]
        w = torch.zeros(C, 1, device=device)
        w[pick] = 1.0
        return eye - ones @ w.transpose(0, 1), {"view": view, "tier": "hard",
                                                "lateralised": True,
                                                "reference_channel": names[int(pick)]}

    raise ValueError(f"Unknown reference view {view!r}")


def apply_reference(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """``[B, C, T]`` re-referenced by the channel operator ``M [C, C]``."""
    assert x.dim() == 3, f"expected [B, C, T], got {tuple(x.shape)}"
    return torch.einsum("ij,bjt->bit", M.to(x.dtype), x)


def sample_views(
    cfg: ReferenceConfig, generator: Optional[torch.Generator] = None
) -> List[str]:
    """Sample ``cfg.num_views`` view names honouring ``hard_view_prob``."""
    views: List[str] = []
    std = list(cfg.standard_views)
    hard = list(cfg.hard_views)
    for _ in range(cfg.num_views):
        use_hard = hard and float(torch.rand(1, generator=generator).item()) < cfg.hard_view_prob
        pool = hard if use_hard else std
        idx = int(torch.randint(len(pool), (1,), generator=generator).item())
        views.append(pool[idx])
    return views


def build_views(
    x: torch.Tensor,
    meta: ChannelMeta,
    cfg: ReferenceConfig,
    view_names: Optional[Sequence[str]] = None,
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, object]]:
    """Materialise reference views of ``[B, C, T]``.

    Returns a list of ``{'name', 'tier', 'signal', 'operator', 'meta'}``.  Views
    that are not physically constructible for this montage are dropped, with the
    reason logged once.
    """
    if not cfg.enabled:
        return [{"name": "original", "tier": "standard", "signal": x,
                 "operator": torch.eye(x.shape[1], device=x.device), "meta": meta}]
    names = list(view_names) if view_names is not None else sample_views(cfg, generator)
    out: List[Dict[str, object]] = []
    for name in names:
        res = reference_operator(name, meta, cfg, generator, x.device)
        if res is None:
            continue
        M, info = res
        if M is None:
            _warn_once(str(info.get("skipped", f"{name} skipped")))
            continue
        vm = ChannelMeta(
            channel_names=meta.channel_names, channel_xyz=meta.channel_xyz,
            channel_mask=meta.channel_mask, channel_quality=meta.channel_quality,
            montage_type=meta.montage_type,
            reference_type=_view_to_reference_type(name),
            reference_channel=info.get("reference_channel"),
            derivation_type=meta.derivation_type,
            bipolar_endpoints=meta.bipolar_endpoints,
        )
        out.append({"name": name, "tier": info.get("tier", "standard"),
                    "signal": apply_reference(x, M), "operator": M, "meta": vm})
    if not out:
        out.append({"name": "original", "tier": "standard", "signal": x,
                    "operator": torch.eye(x.shape[1], device=x.device), "meta": meta})
    return out


def _view_to_reference_type(view: str) -> str:
    return {
        "original": "original",
        "common_average": "common_average",
        "linked_mastoids": "linked_mastoids",
        "left_ear": "left_ear", "right_ear": "right_ear",
        "left_mastoid": "left_mastoid", "right_mastoid": "right_mastoid",
        "random_channel": "single_channel",
    }.get(view, "unknown")
