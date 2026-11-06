"""Subject-wise splitting and leakage checking.

A random split over *windows* is the single most common way a physiological
benchmark result becomes meaningless: consecutive windows from one recording are
near-duplicates, so a window-level split lets the model memorise the subject
rather than the phenomenon.  Every split produced here is subject-wise by
default, and :func:`assert_no_leakage` raises -- it does not warn -- if a subject
or recording appears in more than one split.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLITS = ("train", "val", "test")


class SplitLeakageError(RuntimeError):
    """Raised when the same subject or recording appears in two splits."""


def _stable_hash(key: str, salt: str = "") -> float:
    """Deterministic ``[0, 1)`` hash, independent of PYTHONHASHSEED."""
    digest = hashlib.sha256((salt + "::" + key).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def subject_wise_split(
    subject_ids: Sequence[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Assign whole subjects to train/val/test.

    Uses a stable hash of the subject id rather than a shuffle, so adding new
    subjects never reshuffles the existing assignment -- results stay comparable
    across dataset revisions.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1, got {ratios}"
    salt = str(seed)
    out: Dict[str, List[str]] = {s: [] for s in SPLITS}
    tr, va = ratios[0], ratios[0] + ratios[1]
    for sid in sorted(set(subject_ids)):
        h = _stable_hash(sid, salt)
        split = "train" if h < tr else ("val" if h < va else "test")
        out[split].append(sid)
    return out


def assert_no_leakage(
    split_subjects: Dict[str, Iterable[str]],
    split_recordings: Optional[Dict[str, Iterable[str]]] = None,
) -> None:
    """Raise :class:`SplitLeakageError` on any subject/recording overlap."""
    def _check(name: str, mapping: Dict[str, Iterable[str]]) -> None:
        sets = {k: set(v) for k, v in mapping.items()}
        for a in SPLITS:
            for b in SPLITS:
                if a >= b or a not in sets or b not in sets:
                    continue
                overlap = sets[a] & sets[b]
                if overlap:
                    sample = sorted(overlap)[:10]
                    raise SplitLeakageError(
                        f"{name} overlap between '{a}' and '{b}': {len(overlap)} shared "
                        f"ids, e.g. {sample}. Subject-level tasks must never use a "
                        "random window/segment split."
                    )
    _check("subject", split_subjects)
    if split_recordings:
        _check("recording", split_recordings)


def split_records(
    records: Sequence[Dict[str, str]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    subject_key: str = "subject_id",
    recording_key: str = "recording_id",
) -> Dict[str, List[Dict[str, str]]]:
    """Split manifest records subject-wise and verify no leakage."""
    assign = subject_wise_split([r[subject_key] for r in records], ratios, seed)
    lookup = {sid: split for split, sids in assign.items() for sid in sids}
    out: Dict[str, List[Dict[str, str]]] = {s: [] for s in SPLITS}
    for r in records:
        out[lookup[r[subject_key]]].append(r)
    assert_no_leakage(
        {s: [r[subject_key] for r in rs] for s, rs in out.items()},
        {s: [r[recording_key] for r in rs] for s, rs in out.items()},
    )
    return out
