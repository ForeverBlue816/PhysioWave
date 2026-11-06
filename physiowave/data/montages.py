"""Template EEG montages with 3-D scalp coordinates.

Coordinates are built from the *definition* of the 10-20 / 10-10 systems rather
than copied from a table: electrodes sit on a unit sphere with

* ``+x`` towards the right pre-auricular point,
* ``+y`` towards the nasion,
* ``+z`` towards the vertex (Cz),

circumferential landmarks are placed at their 10-20 arc fractions, and the
intermediate 10-10 electrodes are obtained by spherical interpolation (slerp)
along the great-circle arc between a circumferential anchor and the midline,
which is exactly how the 10-10 extension is specified.

These are **templates**.  Real digitised positions are always preferable and can
be supplied per-recording through the dataset schema; the template is what the
data layer falls back to when a dataset only ships channel names.  Any dataset
without coordinates disables the SSL branch (see
:mod:`physiowave.spatial.spline_laplacian`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

_DEG = math.pi / 180.0


def _ring_point(polar_deg: float, azim_deg: float) -> Tuple[float, float, float]:
    """Point on the unit sphere at ``polar`` from the vertex, ``azim`` from anterior.

    Positive azimuth rotates towards the **left** hemisphere, matching the odd
    electrode numbering of the 10-20 system.
    """
    th, ph = polar_deg * _DEG, azim_deg * _DEG
    return (-math.sin(ph) * math.sin(th), math.cos(ph) * math.sin(th), math.cos(th))


def _slerp(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float, float]:
    """Great-circle interpolation between two unit vectors."""
    av = torch.tensor(a, dtype=torch.float64)
    bv = torch.tensor(b, dtype=torch.float64)
    dot = float(torch.clamp((av * bv).sum(), -1.0, 1.0))
    omega = math.acos(dot)
    if abs(omega) < 1e-9:
        v = av
    else:
        v = (math.sin((1 - t) * omega) * av + math.sin(t * omega) * bv) / math.sin(omega)
    v = v / v.norm()
    return (float(v[0]), float(v[1]), float(v[2]))


# Circumference ring sits at 72 deg from the vertex; azimuths follow the 10-20
# arc fractions (T7/T8 land exactly on the pre-auricular points at +-90 deg).
_CIRC_POLAR = 72.0
_CIRC = {
    "Fpz": 0.0, "Fp1": 18.0, "AF7": 36.0, "F7": 54.0, "FT7": 72.0, "T7": 90.0,
    "TP7": 108.0, "P7": 126.0, "PO7": 144.0, "O1": 162.0, "Oz": 180.0,
}
# Midline electrodes as fractions of the nasion->inion sagittal arc.
_MIDLINE = {
    "Fpz": 0.10, "AFz": 0.20, "Fz": 0.30, "FCz": 0.40, "Cz": 0.50,
    "CPz": 0.60, "Pz": 0.70, "POz": 0.80, "Oz": 0.90,
}


def _midline_point(frac: float) -> Tuple[float, float, float]:
    a = math.pi * frac
    return (0.0, math.cos(a), math.sin(a))


# Each 10-10 row: (circumferential anchor, midline anchor, labels left->midline).
_ROWS: List[Tuple[str, str, List[str]]] = [
    ("AF7", "AFz", ["AF7", "AF3", "AFz"]),
    ("F7", "Fz", ["F7", "F5", "F3", "F1", "Fz"]),
    ("FT7", "FCz", ["FT7", "FC5", "FC3", "FC1", "FCz"]),
    ("T7", "Cz", ["T7", "C5", "C3", "C1", "Cz"]),
    ("TP7", "CPz", ["TP7", "CP5", "CP3", "CP1", "CPz"]),
    ("P7", "Pz", ["P7", "P5", "P3", "P1", "Pz"]),
    ("PO7", "POz", ["PO7", "PO3", "POz"]),
]

#: Right-hemisphere counterpart of every left-hemisphere label.
_MIRROR = {
    "Fp1": "Fp2", "AF7": "AF8", "AF3": "AF4", "F7": "F8", "F5": "F6", "F3": "F4",
    "F1": "F2", "FT7": "FT8", "FC5": "FC6", "FC3": "FC4", "FC1": "FC2",
    "T7": "T8", "C5": "C6", "C3": "C4", "C1": "C2", "TP7": "TP8", "CP5": "CP6",
    "CP3": "CP4", "CP1": "CP2", "P7": "P8", "P5": "P6", "P3": "P4", "P1": "P2",
    "PO7": "PO8", "PO3": "PO4", "O1": "O2", "A1": "A2", "M1": "M2",
}


def _build_positions() -> Dict[str, Tuple[float, float, float]]:
    pos: Dict[str, Tuple[float, float, float]] = {}
    for name, frac in _MIDLINE.items():
        pos[name] = _midline_point(frac)
    for name, azim in _CIRC.items():
        pos.setdefault(name, _ring_point(_CIRC_POLAR, azim))
    pos["Fp1"] = _ring_point(_CIRC_POLAR, _CIRC["Fp1"])
    pos["O1"] = _ring_point(_CIRC_POLAR, _CIRC["O1"])
    for circ, mid, labels in _ROWS:
        a, b = pos[circ], pos[mid]
        n = len(labels) - 1
        for i, lab in enumerate(labels):
            pos.setdefault(lab, _slerp(a, b, i / n))
    # Mirror the left hemisphere onto the right (x -> -x).
    for left, right in _MIRROR.items():
        if left in pos and right not in pos:
            x, y, z = pos[left]
            pos[right] = (-x, y, z)
    # Reference landmarks: earlobes and mastoids sit below the circumference ring.
    pos["A1"] = _ring_point(100.0, 90.0)
    pos["A2"] = _ring_point(100.0, -90.0)
    pos["M1"] = _ring_point(100.0, 108.0)
    pos["M2"] = _ring_point(100.0, -108.0)
    pos["Iz"] = _midline_point(1.0)
    pos["Nz"] = _midline_point(0.0)
    return pos


TEMPLATE_POSITIONS: Dict[str, Tuple[float, float, float]] = _build_positions()

#: Classic 19-channel 10-20 clinical montage.
STANDARD_1020_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

#: Full 10-10 label set produced by the construction above (61 scalp electrodes).
STANDARD_1010_61 = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

#: 64-channel set = 10-10 plus mastoids and the inion landmark.
STANDARD_1010_64 = STANDARD_1010_61 + ["M1", "M2", "Iz"]

#: Old / alternative names accepted on input.
ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
}

MONTAGES: Dict[str, List[str]] = {
    "standard_1020_19": STANDARD_1020_19,
    "standard_1010_61": STANDARD_1010_61,
    "standard_1010_64": STANDARD_1010_64,
}


def canonical_name(name: str) -> str:
    """Map an input channel label onto its canonical template name."""
    raw = name.strip()
    for suffix in ("-REF", "-LE", "-AVG", "-A1", "-A2", "-M1", "-M2"):
        if raw.upper().endswith(suffix):
            raw = raw[: -len(suffix)]
    raw = raw.replace("EEG ", "").replace("EEG", "").strip()
    if raw in TEMPLATE_POSITIONS:
        return raw
    upper = raw.upper()
    if upper in ALIASES:
        return ALIASES[upper]
    for key in TEMPLATE_POSITIONS:
        if key.upper() == upper:
            return key
    return raw


def positions_for(
    channel_names: Sequence[str], strict: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Template coordinates for ``channel_names``.

    Returns:
        ``(xyz [C, 3], known [C] bool)``.  Unknown labels get an all-zero row and
        ``known=False``; the caller must route those through the learnable
        unknown-coordinate fallback and log a warning.
    """
    xyz = torch.zeros(len(channel_names), 3)
    known = torch.zeros(len(channel_names), dtype=torch.bool)
    missing: List[str] = []
    for i, name in enumerate(channel_names):
        canon = canonical_name(name)
        p = TEMPLATE_POSITIONS.get(canon)
        if p is None:
            missing.append(name)
            continue
        xyz[i] = torch.tensor(p)
        known[i] = True
    if missing and strict:
        raise KeyError(f"No template coordinates for channels: {missing}")
    return xyz, known


def montage(name: str) -> Tuple[List[str], torch.Tensor]:
    """``(channel_names, xyz)`` for a registered template montage."""
    if name not in MONTAGES:
        raise KeyError(f"Unknown montage {name!r}; available: {sorted(MONTAGES)}")
    names = MONTAGES[name]
    xyz, known = positions_for(names, strict=True)
    assert bool(known.all()), "template montage must have complete coordinates"
    return list(names), xyz
