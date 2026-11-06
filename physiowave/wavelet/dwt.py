"""Critically-sampled 1D discrete wavelet transform for patch-level tokenisation.

Design contract
---------------
For an input patch of length ``P`` the ``J``-level decomposition returns exactly
``P`` coefficients in total: ``P / 2**J`` approximation coefficients plus
``P / 2**j`` detail coefficients for ``j = J .. 1``.  This *critical sampling* is
what keeps the tokenizer token-efficient -- no scale is ever upsampled back to
``P`` and concatenated, so the sequence length seen by the backbone is
independent of the number of decomposition levels.  :func:`dwt` asserts the
length budget on every call.

How boundaries are handled
--------------------------
Naively padding a patch, filtering, and then cropping back to ``P/2`` coefficients
destroys information: the cropped boundary coefficients are exactly the ones that
carried the padded samples, and the resulting square operator is numerically
singular (condition number ~1e17 measured for db4).  This module therefore uses
the classical symmetric-extension filter bank instead, the same construction
JPEG2000 uses for the same reason:

1. extend the patch from ``P`` to ``2P`` samples with the requested boundary rule
   (``reflect``/``symmetric`` -> whole-sample mirror ``[x, flip(x)]``);
2. run the *periodic* (exactly critically sampled, perfect-reconstruction) filter
   bank on the ``2P`` extended signal;
3. keep the first half of every subband, which is exactly ``P`` coefficients.

The mirror makes the extended signal both continuous and periodic, so the wrap
point introduces no step discontinuity -- that is the boundary artefact this
whole construction exists to avoid.

Wavelet / boundary-mode compatibility
-------------------------------------
Step 3 is invertible only when the analysis filters are symmetric, i.e. for the
biorthogonal families (``bior*``, ``rbio*``).  Measured condition numbers of the
``[P, P]`` analysis operator at ``P=64``:

===========  ==============  ==============
wavelet      ``reflect``     ``periodization``
===========  ==============  ==============
``bior4.4``  3.2 - 42        1.3 - 1.5
``rbio4.4``  2.2 - 12        1.3 - 1.5
``db4``      ~1e18 (singular) 1.0
===========  ==============  ==============

So for an *orthogonal* wavelet (``db*``, ``sym*``, ``coif*``) the only boundary
rule that admits a critically-sampled inverse is ``periodization``.  When such a
combination is requested this module falls back to ``periodization`` and logs a
warning -- never silently.  The default wavelet is therefore ``bior4.4`` (CDF 9/7),
which supports the default ``reflect`` boundary rule.

``zero`` is supported but is *not* the default: zero extension asserts that the
signal drops to 0 immediately outside the patch, which for a stationary
physiological signal is a step discontinuity.  Because the DWT here is applied
per patch, that artefact recurs ``S`` times per channel instead of twice per
recording, and the learnable subband gates downstream cannot tell it from signal.

Shapes
------
Analysis input  : ``[N, P]`` (callers flatten any leading batch axes)
Analysis output : ``[cA_J, cD_J, ..., cD_1]``, lengths ``P/2**J, P/2**J, ..., P/2``
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Boundary rules understood by :func:`extend_patch`.
BOUNDARY_MODES = ("reflect", "symmetric", "periodization", "zero", "constant")

#: Default boundary rule.  Deliberately **not** ``zero`` (see module docstring).
DEFAULT_BOUNDARY_MODE = "reflect"

#: Default wavelet.  Symmetric/biorthogonal so that ``reflect`` stays invertible.
DEFAULT_WAVELET = "bior4.4"

#: Wavelet families whose analysis filters are symmetric.
_SYMMETRIC_FAMILIES = ("bior", "rbio")


def is_symmetric_wavelet(wavelet: str) -> bool:
    """True if ``wavelet`` has symmetric analysis filters (biorthogonal families)."""
    return wavelet.lower().startswith(_SYMMETRIC_FAMILIES)


def resolve_boundary_mode(wavelet: str, boundary_mode: str) -> str:
    """Return the boundary rule that will actually be used, warning on fallback."""
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"Unknown boundary mode {boundary_mode!r}; expected {BOUNDARY_MODES}")
    if boundary_mode == "periodization" or is_symmetric_wavelet(wavelet):
        return boundary_mode
    logger.warning(
        "Wavelet %r has non-symmetric (orthogonal) analysis filters, for which the "
        "critically-sampled %r boundary transform is singular. Falling back to "
        "boundary_mode='periodization'. Use a biorthogonal wavelet (e.g. 'bior4.4') "
        "to keep %r.",
        wavelet, boundary_mode, boundary_mode,
    )
    return "periodization"


def get_filters(wavelet: str, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Filter coefficients of ``wavelet`` as tensors.

    PyWavelets is used purely as a coefficient table; every transform below is
    implemented in torch so it stays differentiable and runs on GPU.
    """
    import pywt

    w = pywt.Wavelet(wavelet)
    return {
        "dec_lo": torch.tensor(w.dec_lo, dtype=dtype),
        "dec_hi": torch.tensor(w.dec_hi, dtype=dtype),
        "rec_lo": torch.tensor(w.rec_lo, dtype=dtype),
        "rec_hi": torch.tensor(w.rec_hi, dtype=dtype),
    }


def extend_patch(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Extend ``[N, P]`` to ``[N, 2P]`` with the given boundary rule."""
    P = x.shape[-1]
    if mode in ("reflect", "symmetric"):
        # Whole-sample mirror: continuous at the join *and* periodic at the wrap.
        return torch.cat([x, x.flip(-1)], dim=-1)
    if mode == "periodization":
        return torch.cat([x, x], dim=-1)
    if mode == "zero":
        return torch.cat([x, torch.zeros_like(x)], dim=-1)
    if mode == "constant":
        return torch.cat([x, x[..., -1:].expand(*x.shape[:-1], P)], dim=-1)
    raise ValueError(f"Unknown boundary mode {mode!r}")


def _circular_pad(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Circular padding of ``[N, T]`` that also works when the pad exceeds ``T``."""
    T = x.shape[-1]
    if left == 0 and right == 0:
        return x
    reps = int(math.ceil((left + T + right) / T)) + 2
    tiled = x.repeat(1, reps)
    start = (reps // 2) * T - left
    return tiled[:, start : start + left + T + right]


def periodic_dwt_level(
    x: torch.Tensor, dec_lo: torch.Tensor, dec_hi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One exactly critically-sampled periodic analysis level.

    Args:
        x: ``[N, T]``, ``T`` even.
        dec_lo/dec_hi: ``[L]`` analysis filters.

    Returns:
        ``(cA, cD)`` each ``[N, T // 2]``.
    """
    assert x.dim() == 2, f"expected [N, T], got {tuple(x.shape)}"
    N, T = x.shape
    assert T % 2 == 0, f"periodic DWT requires an even length, got T={T}"
    L = dec_lo.numel()
    pad = L - 2
    xe = _circular_pad(x, pad, pad)
    w = torch.stack([dec_lo, dec_hi], dim=0).unsqueeze(1).to(x.dtype)  # [2, 1, L]
    y = F.conv1d(xe.unsqueeze(1), w, stride=2)  # [N, 2, T/2 + (L-2)/2]
    half = T // 2
    off = (y.shape[-1] - half) // 2  # centred crop: coefficient n aligns with sample 2n
    y = y[..., off : off + half]
    assert y.shape[-1] == half, f"expected {half} coefficients, got {y.shape[-1]}"
    return y[:, 0], y[:, 1]


def coeff_lengths(patch_len: int, level: int) -> List[int]:
    """Lengths of ``[cA_J, cD_J, ..., cD_1]`` -- they sum to ``patch_len``."""
    if patch_len % (2 ** level) != 0:
        raise ValueError(
            f"patch_len={patch_len} must be divisible by 2**level={2 ** level}"
        )
    details = [patch_len // (2 ** j) for j in range(1, level + 1)]
    return [patch_len // (2 ** level)] + list(reversed(details))


def dwt(
    x: torch.Tensor,
    wavelet_filters: Dict[str, torch.Tensor],
    level: int,
    mode: str = DEFAULT_BOUNDARY_MODE,
) -> List[torch.Tensor]:
    """Critically-sampled multi-level analysis of ``[N, P]``.

    Returns ``[cA_J, cD_J, ..., cD_1]`` whose concatenated length is exactly ``P``.
    """
    assert x.dim() == 2, f"dwt expects [N, P], got {tuple(x.shape)}"
    P = x.shape[-1]
    dec_lo, dec_hi = wavelet_filters["dec_lo"], wavelet_filters["dec_hi"]
    xe = extend_patch(x, mode)  # [N, 2P]
    a = xe
    details: List[torch.Tensor] = []
    for _ in range(level):
        a, d = periodic_dwt_level(a, dec_lo, dec_hi)
        details.append(d)
    # Keep the first half of every subband: the extension doubled the support,
    # so the first half corresponds to the original patch.
    coeffs = [a[..., : a.shape[-1] // 2]]
    for d in reversed(details):
        coeffs.append(d[..., : d.shape[-1] // 2])
    total = sum(c.shape[-1] for c in coeffs)
    assert total == P, (
        f"critical sampling violated: sum(|coeffs|)={total} != P={P}. "
        "The wavelet tokenizer must never inflate the coefficient budget."
    )
    return coeffs


def _no_autocast(device: torch.device):
    """Disable autocast for a region.

    Every linear-algebra call in this module builds a small dense operator out
    of the learnable filters and then factorises it. Under an autocast region
    the ops that build the operator are cast to the autocast dtype regardless
    of the ``dtype=`` asked for, and ``linalg.inv``/``linalg.cond`` have no
    bf16 kernel at all -- so the failure is a hard error rather than a silent
    loss of precision. Even where a kernel exists, a dense inverse in bf16 is
    the last place to spend the mantissa.
    """
    return torch.autocast(device_type=device.type, enabled=False)


def analysis_matrix(
    patch_len: int,
    wavelet_filters: Dict[str, torch.Tensor],
    level: int,
    mode: str,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Materialise the square analysis operator ``A`` with ``c = A @ x``.

    ``A`` is ``[P, P]``; it is obtained by pushing the identity basis through
    :func:`dwt`, which makes it exact for whatever filters are currently held
    (including learned ones) and keeps the result differentiable.
    """
    device = torch.device(device) if device is not None else wavelet_filters["dec_lo"].device
    if device.type == "mps" and dtype == torch.float64:
        # MPS has no float64; the operator is small, so build it in float32 there.
        dtype = torch.float32
    eye = torch.eye(patch_len, device=device, dtype=dtype)
    # Move before casting: MPS refuses a direct float32 -> float64 conversion, so
    # `.to(device=..., dtype=...)` in one call raises when crossing off MPS.
    filt = {k: v.to(device).to(dtype) for k, v in wavelet_filters.items()}
    rows = dwt(eye, filt, level, mode)          # each row i holds A @ e_i
    return torch.cat(rows, dim=-1).transpose(0, 1).contiguous()


def condition_number(
    patch_len: int, wavelet: str, level: int, mode: str
) -> float:
    """Condition number of the analysis operator, for diagnostics and tests."""
    filt = get_filters(wavelet, torch.float64)
    A = analysis_matrix(patch_len, filt, level, mode, dtype=torch.float64)
    return float(torch.linalg.cond(A).item())


class WaveletTransform1D(nn.Module):
    """Critically-sampled analysis/synthesis pair with optionally learnable filters.

    Args:
        wavelet: PyWavelets name used to initialise the filters.
        level: number of decomposition levels ``J``.
        boundary_mode: see :data:`BOUNDARY_MODES`; default ``reflect``.
        learnable: if True the analysis filters are ``nn.Parameter`` and the whole
            transform (including the synthesis solve) is differentiable w.r.t. them.
        max_filter_drift: filters are clamped to stay within this L-inf distance of
            their initialisation, which keeps the analysis operator invertible when
            it is being learned.  ``None`` disables the constraint.
    """

    def __init__(
        self,
        wavelet: str = DEFAULT_WAVELET,
        level: int = 3,
        boundary_mode: str = DEFAULT_BOUNDARY_MODE,
        learnable: bool = True,
        max_filter_drift: float | None = 0.25,
    ) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.level = int(level)
        self.requested_boundary_mode = boundary_mode
        self.boundary_mode = resolve_boundary_mode(wavelet, boundary_mode)
        self.learnable = bool(learnable)
        self.max_filter_drift = max_filter_drift

        f = get_filters(wavelet)
        self.register_buffer("dec_lo_init", f["dec_lo"].clone())
        self.register_buffer("dec_hi_init", f["dec_hi"].clone())
        if learnable:
            self.dec_lo = nn.Parameter(f["dec_lo"].clone())
            self.dec_hi = nn.Parameter(f["dec_hi"].clone())
        else:
            self.register_buffer("dec_lo", f["dec_lo"].clone())
            self.register_buffer("dec_hi", f["dec_hi"].clone())
        self._inv_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    # -- filters ---------------------------------------------------------------
    @property
    def filters(self) -> Dict[str, torch.Tensor]:
        lo, hi = self.dec_lo, self.dec_hi
        if self.learnable and self.max_filter_drift is not None:
            d = self.max_filter_drift
            lo = self.dec_lo_init + (lo - self.dec_lo_init).clamp(-d, d)
            hi = self.dec_hi_init + (hi - self.dec_hi_init).clamp(-d, d)
        return {"dec_lo": lo, "dec_hi": hi}

    def coeff_lengths(self, patch_len: int) -> List[int]:
        return coeff_lengths(patch_len, self.level)

    # -- transforms ------------------------------------------------------------
    def analysis(self, x: torch.Tensor) -> List[torch.Tensor]:
        """``[N, P] -> [cA_J, cD_J, ..., cD_1]`` (sums to ``P``)."""
        return dwt(x, self.filters, self.level, self.boundary_mode)

    def synthesis(self, coeffs: Sequence[torch.Tensor], patch_len: int | None = None) -> torch.Tensor:
        """``[cA_J, cD_J, ..., cD_1] -> [N, P]``, the exact inverse of :meth:`analysis`.

        Because the critically-sampled analysis is a square ``[P, P]`` map, the
        synthesis is obtained by solving ``A x = c``.  This is the numerically
        obtained equivalent of classical boundary-corrected wavelet filters; it is
        exact for every supported boundary mode and stays differentiable w.r.t.
        learned filters.
        """
        flat = torch.cat(list(coeffs), dim=-1)  # [N, P]
        P = patch_len if patch_len is not None else flat.shape[-1]
        assert flat.shape[-1] == P, f"expected {P} coefficients, got {flat.shape[-1]}"
        if self.learnable and torch.is_grad_enabled() and self.training:
            return flat @ self._live_inverse(P, flat.device, flat.dtype)
        Ainv_t = self._cached_inverse(P, flat.device, flat.dtype)
        return flat @ Ainv_t

    @staticmethod
    def _operator_device(device: torch.device) -> torch.device:
        """Where to build and invert the ``[P, P]`` operator.

        MPS has no working backward for ``linalg.solve``/``lu_solve``, so the
        (tiny) operator is built and inverted on CPU there and moved back; the
        cross-device ``.to`` keeps the graph intact so filter gradients still flow.
        """
        return torch.device("cpu") if device.type == "mps" else device

    def _live_inverse(self, P: int, device, dtype) -> torch.Tensor:
        """Differentiable ``(A^{-1})^T`` recomputed from the current filters."""
        op_dev = self._operator_device(device)
        with _no_autocast(op_dev):
            A = analysis_matrix(P, self.filters, self.level, self.boundary_mode,
                                device=op_dev, dtype=torch.float32)
            inv_t = torch.linalg.inv(A.float()).transpose(0, 1)
        return inv_t.to(device=device, dtype=dtype)

    def _cached_inverse(self, P: int, device, dtype) -> torch.Tensor:
        """``(A^{-1})^T`` so that ``x = c @ (A^{-1})^T``; cached per ``(P, device, dtype)``."""
        key = (P, device, dtype)
        cached = self._inv_cache.get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            filt = {k: v.detach() for k, v in self.filters.items()}
            op_dev = self._operator_device(device)
            with _no_autocast(op_dev):
                A = analysis_matrix(P, filt, self.level, self.boundary_mode,
                                    device=op_dev, dtype=torch.float64)
                inv_t = torch.linalg.inv(A).transpose(0, 1).contiguous()
            inv_t = inv_t.to(device=device, dtype=dtype)
        self._inv_cache[key] = inv_t
        return inv_t

    def clear_cache(self) -> None:
        """Drop cached synthesis operators.  Call after mutating the filters."""
        self._inv_cache.clear()

    def train(self, mode: bool = True):  # noqa: D102 - keeps the cache honest
        if mode:
            self.clear_cache()
        return super().train(mode)

    def condition_number(self, patch_len: int) -> float:
        """Condition number of the current analysis operator."""
        dev = next(iter(self.filters.values())).device
        with torch.no_grad(), _no_autocast(dev):
            A = analysis_matrix(patch_len, {k: v.detach() for k, v in self.filters.items()},
                                self.level, self.boundary_mode, dtype=torch.float64)
            return float(torch.linalg.cond(A).item())

    def extra_repr(self) -> str:
        return (
            f"wavelet={self.wavelet}, level={self.level}, "
            f"boundary_mode={self.boundary_mode} (requested {self.requested_boundary_mode}), "
            f"learnable={self.learnable}"
        )
