"""Terminology guard (constraints A and B of ``docs/terminology.md``).

Constraint A: a channel-wise statistic computed from recorded signals may never be
called "connectivity", "functional connectivity" or "brain connectivity" in the
core source.  A scalp channel-relation matrix mixes genuine source correlation
with the reference montage and with instantaneous volume conduction, so the word
would assert something the data cannot support.

Constraint B: the strict spherical-spline surface Laplacian is the **SSL branch**
and is the only thing called CSD; the learnable graph-Laplacian branch is the
**GL branch** and may only be called *CSD-inspired*.

Occurrences that explain or forbid the term are whitelisted with an inline
``TERMINOLOGY-ALLOW`` marker, so the guard stays strict without banning the
discussion of why it exists.
"""

from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_PACKAGE = os.path.join(REPO_ROOT, "physiowave")
ALLOW_MARKER = "TERMINOLOGY-ALLOW"

BANNED = re.compile(r"\bconnectivit(y|ies)\b", re.IGNORECASE)
CSD_CLAIM = re.compile(r"\bCSD\b")


def _python_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_no_connectivity_language_in_core_modules():
    """No un-whitelisted use of "connectivity" anywhere in ``physiowave/``."""
    offenders = []
    for path in _python_files(CORE_PACKAGE):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if BANNED.search(line) and ALLOW_MARKER not in line:
                    # A whitelist marker may sit on the line above (long docstrings).
                    offenders.append((os.path.relpath(path, REPO_ROOT), i, line.strip()))
    filtered = []
    for path, lineno, line in offenders:
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            lines = f.readlines()
        window = "".join(lines[max(0, lineno - 3): lineno + 2])
        if ALLOW_MARKER not in window:
            filtered.append((path, lineno, line))
    assert not filtered, (
        "Constraint A violated -- 'connectivity' used in core source. A data-derived "
        "channel matrix is a spatial statistic / channel-relation graph, not "
        "connectivity. Offenders:\n"
        + "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in filtered)
    )


def test_graph_laplacian_branch_is_never_called_csd():
    """The GL branch may say "CSD-inspired" but must never claim to *be* CSD."""
    path = os.path.join(CORE_PACKAGE, "spatial", "graph_laplacian.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for match in CSD_CLAIM.finditer(text):
        start = max(0, match.start() - 40)
        context = text[start: match.end() + 40]
        assert ("CSD-inspired" in context or "not" in context.lower()
                or "spline_laplacian" in context), (
            f"Constraint B violated in graph_laplacian.py near: {context!r}"
        )


def test_spline_laplacian_is_the_named_ssl_branch():
    path = os.path.join(CORE_PACKAGE, "spatial", "spline_laplacian.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "SSL branch" in text or "SSL" in text
    assert "Perrin" in text, "the strict CSD implementation must cite its method"


def test_a_dyn_docstring_states_the_interpretation_limit():
    path = os.path.join(CORE_PACKAGE, "spatial", "spatial_stats.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "MUST NOT be interpreted" in text
    assert "volume conduction" in text
    assert ALLOW_MARKER in text, "the prohibition line itself must carry the whitelist marker"


def test_terminology_document_exists():
    doc = os.path.join(REPO_ROOT, "docs", "terminology.md")
    assert os.path.exists(doc), "docs/terminology.md is required by the project spec"
    with open(doc, encoding="utf-8") as f:
        text = f.read()
    for token in ("spatial statistics", "channel-relation graph", "SSL", "GL",
                  "CSD-inspired", "reference"):
        assert token.lower() in text.lower(), f"terminology.md does not define {token!r}"


def test_limb_semg_boundary_is_documented():
    from physiowave.data.registry import REGISTRY, assert_limb_semg

    semg = [s for s in REGISTRY.values() if s.modality == "semg"]
    assert semg and all(s.emg_region in ("limb", "facial", "trunk", "unknown") for s in semg)
    assert_limb_semg([s for s in semg if s.emg_region == "limb"])
    facial = [s for s in semg if s.emg_region == "limb"][0]
    import copy
    bad = copy.copy(facial)
    bad.emg_region = "facial"
    with pytest.raises(ValueError, match="Facial EMG"):
        assert_limb_semg([bad])
