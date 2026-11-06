"""SSL branch -- Spline Surface Laplacian (strict spherical-spline CSD).

This module implements the spherical spline surface Laplacian of Perrin et al.
(1989), *Electroencephalogr. Clin. Neurophysiol.* 72:184-187.  Following
constraint B of the project terminology, this -- and only this -- is referred to
as **strict CSD**.  The learnable graph-Laplacian branch in
:mod:`physiowave.spatial.graph_laplacian` is a separate, parallel branch and is
called *CSD-inspired*; the two names are never interchanged.

Why it can be a first-class branch rather than an offline preprocessing step
---------------------------------------------------------------------------
The ``G`` and ``H`` matrices depend only on the electrode coordinates and the
spline parameters ``(m, lambda, n_legendre)`` -- **not on the data**.  So the
whole surface Laplacian collapses to a single fixed linear operator
``L_ssl in R^{C x C}`` that can be precomputed once per montage, cached, and
applied in the forward pass as one ``[C, C] @ [C, T]`` matmul.  Its cost is
negligible next to the tokenizer.

Two physical facts drive how this branch is wired into the model
----------------------------------------------------------------
* **Reference invariance.**  Re-referencing is ``V -> V - r 1^T`` for some
  channel-linear combination ``r``; the spline fit assigns a constant potential
  to the spline constant term, which the Laplacian annihilates.  Hence
  ``L_ssl @ (V - r 1^T) == L_ssl @ V`` exactly.  This makes the SSL view a
  natural, physically-grounded anchor for the reference-consistency objective in
  :mod:`physiowave.pretrain`.  :func:`verify_reference_invariance` checks it
  numerically and ``tests/test_spatial.py`` asserts it.
* **The surface Laplacian is a spatial band-pass, not a high-pass.**  It
  sharpens local, superficial generators and attenuates deep or widely
  distributed ones.  Replacing the raw branch with SSL would therefore *discard*
  real signal.  The SSL output is always fused with the raw branch through a
  learnable gate initialised to a small value -- never used as a replacement.

Degradation rules (all logged, never silent)
--------------------------------------------
* fewer than ``min_channels`` (default 16) electrodes -> skip; spline CSD is
  unreliable at low spatial sampling;
* bipolar derivations -> skip; the surface Laplacian is defined on monopolar
  potentials;
* missing coordinates -> skip;
* bad/missing channels -> spherical-spline **interpolate them first**, then build
  the Laplacian.  A single bad electrode otherwise contaminates the entire
  Laplacian output, since every output channel is a weighted sum of all inputs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from .geometry import cos_angles, montage_hash, normalize_to_sphere

logger = logging.getLogger(__name__)

#: Default spline parameters (Perrin et al. 1989; same defaults as MNE-Python).
DEFAULT_STIFFNESS = 4          # spline order m
DEFAULT_LAMBDA = 1e-5          # Tikhonov regularisation
DEFAULT_N_LEGENDRE = 50        # Legendre series truncation


@dataclass
class SSLConfig:
    """Configuration for the SSL branch."""

    enabled: bool = True
    stiffness: int = DEFAULT_STIFFNESS
    lambda_reg: float = DEFAULT_LAMBDA
    n_legendre: int = DEFAULT_N_LEGENDRE
    min_channels: int = 16
    gate_init: float = 0.1
    cache_dir: Optional[str] = None


class SSLSkipped(Exception):
    """Raised internally when the SSL branch cannot be built for a montage."""


def legendre_matrix(cos: torch.Tensor, n_terms: int) -> torch.Tensor:
    """Legendre polynomials ``P_n(cos)`` for ``n = 1..n_terms`` -> ``[n_terms, C, C]``.

    Uses Bonnet's recurrence, which is stable for the ``n <= 50`` used here.
    """
    p_prev = torch.ones_like(cos)          # P_0
    p_curr = cos.clone()                   # P_1
    out = [p_curr]
    for n in range(1, n_terms):
        p_next = ((2 * n + 1) * cos * p_curr - n * p_prev) / (n + 1)
        out.append(p_next)
        p_prev, p_curr = p_curr, p_next
    return torch.stack(out, dim=0)         # [n_terms, C, C], index 0 == P_1


def _gh_matrices(
    cos: torch.Tensor, stiffness: int, n_legendre: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perrin ``G`` and ``H`` matrices for the given cosine-angle matrix."""
    m = stiffness
    P = legendre_matrix(cos, n_legendre)                     # [N, C, C]
    n = torch.arange(1, n_legendre + 1, device=cos.device, dtype=cos.dtype)
    fac = (2 * n + 1)
    g_coef = fac / ((n ** m) * ((n + 1) ** m))
    h_coef = -fac / ((n ** (m - 1)) * ((n + 1) ** (m - 1)))
    four_pi = 4.0 * torch.pi
    G = (g_coef.view(-1, 1, 1) * P).sum(dim=0) / four_pi
    H = (h_coef.view(-1, 1, 1) * P).sum(dim=0) / four_pi
    return G, H


def _spline_solver(G: torch.Tensor, lambda_reg: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Gi, w)`` for the constrained spline fit.

    The Perrin spline solves

    .. code-block:: text

        [G + lambda I   1] [c ]   [V]
        [     1^T       0] [c0] = [0]

    whose solution is ``c0 = w^T V`` with ``w = Gi 1 / (1^T Gi 1)`` and
    ``c = Gi (V - c0 1)``.
    """
    C = G.shape[0]
    Gl = G + lambda_reg * torch.eye(C, device=G.device, dtype=G.dtype)
    Gi = torch.linalg.inv(Gl)
    ones = torch.ones(C, 1, device=G.device, dtype=G.dtype)
    denom = (ones.transpose(0, 1) @ Gi @ ones).clamp_min(1e-12)
    w = (Gi @ ones) / denom                                  # [C, 1]
    return Gi, w


def spline_interpolation_matrix(
    sphere_xyz: torch.Tensor,
    good: torch.Tensor,
    stiffness: int = DEFAULT_STIFFNESS,
    lambda_reg: float = DEFAULT_LAMBDA,
    n_legendre: int = DEFAULT_N_LEGENDRE,
) -> torch.Tensor:
    """``[C, C_good]`` operator reconstructing *all* channels from the good ones.

    Rows for good channels are one-hot (the recorded value is kept); rows for bad
    channels hold the spherical-spline interpolation weights.  This is applied
    before the Laplacian so that a bad electrode cannot contaminate every output.
    """
    idx_good = torch.nonzero(good, as_tuple=False).squeeze(-1)
    C = sphere_xyz.shape[0]
    cos_gg = cos_angles(sphere_xyz[idx_good])
    G_gg, _ = _gh_matrices(cos_gg, stiffness, n_legendre)
    Gi, w = _spline_solver(G_gg, lambda_reg)
    # coefficients of the spline given the good-channel potentials:
    #   c  = Gi (V_g - (w^T V_g) 1)   ->   c = Cmat V_g
    ones = torch.ones(idx_good.numel(), 1, device=G_gg.device, dtype=G_gg.dtype)
    Cmat = Gi - (Gi @ ones) @ w.transpose(0, 1)
    cos_ag = sphere_xyz @ sphere_xyz[idx_good].transpose(0, 1)
    cos_ag = cos_ag.clamp(-1.0, 1.0)
    G_ag, _ = _gh_matrices(cos_ag, stiffness, n_legendre)     # [C, C_good]
    M = torch.ones(C, 1, device=G_gg.device, dtype=G_gg.dtype) @ w.transpose(0, 1) + G_ag @ Cmat
    # Keep recorded values exactly where they exist.
    M[idx_good] = 0.0
    M[idx_good, torch.arange(idx_good.numel(), device=M.device)] = 1.0
    return M


def build_ssl_operator(
    xyz: torch.Tensor,
    channel_mask: Optional[torch.Tensor] = None,
    stiffness: int = DEFAULT_STIFFNESS,
    lambda_reg: float = DEFAULT_LAMBDA,
    n_legendre: int = DEFAULT_N_LEGENDRE,
    min_channels: int = 16,
) -> torch.Tensor:
    """Build the fixed ``[C, C]`` surface-Laplacian operator for one montage.

    The returned operator maps *recorded* channel values (including bad ones,
    whose columns are zero) to CSD values at every electrode:
    ``L_ssl = H @ Cmat @ M_interp``.

    Raises:
        SSLSkipped: if the montage is too sparse or has no usable coordinates.
    """
    # The operator depends only on geometry, so it is always built on CPU in
    # float64 (the Legendre series and the G-matrix inverse need the precision,
    # and MPS has no float64 at all) and moved to the compute device afterwards.
    xyz = xyz.detach().cpu().double()
    good = (torch.ones(xyz.shape[0], dtype=torch.bool)
            if channel_mask is None else channel_mask.detach().cpu().to(torch.bool))
    # Channels without coordinates are never usable for a spline fit.
    has_coord = xyz.norm(dim=-1) > 1e-8
    good = good & has_coord
    n_good = int(good.sum().item())
    if n_good < min_channels:
        raise SSLSkipped(
            f"SSL branch skipped: {n_good} usable electrodes < min_channels={min_channels}. "
            "Spherical-spline CSD is unreliable at low spatial sampling."
        )

    sphere = normalize_to_sphere(xyz)
    idx_good = torch.nonzero(good, as_tuple=False).squeeze(-1)

    cos_gg = cos_angles(sphere[idx_good])
    G_gg, _ = _gh_matrices(cos_gg, stiffness, n_legendre)
    Gi, w = _spline_solver(G_gg, lambda_reg)
    ones = torch.ones(idx_good.numel(), 1, device=xyz.device, dtype=xyz.dtype)
    Cmat = Gi - (Gi @ ones) @ w.transpose(0, 1)     # V_good -> spline coefficients
    cos_ag = (sphere @ sphere[idx_good].transpose(0, 1)).clamp(-1.0, 1.0)
    _, H_ag = _gh_matrices(cos_ag, stiffness, n_legendre)      # [C, C_good]

    # `H_ag @ Cmat` maps the GOOD channel potentials to the CSD at EVERY electrode.
    # Interpolation of the bad channels is therefore already built in and does not
    # need a separate step: the spline is fitted through the good electrodes only,
    # and evaluating H at all positions with those coefficients is exactly "spline
    # interpolate the missing electrodes, then take the surface Laplacian".
    # (`spline_interpolation_matrix` exposes the potential-domain interpolation on
    # its own, for callers that want the imputed signal rather than its Laplacian.)
    L_good = H_ag @ Cmat                            # [C, C_good]
    # Widen to the full channel set with zero columns for the bad electrodes, so a
    # bad channel's recorded value can never reach any output.
    L = torch.zeros(xyz.shape[0], xyz.shape[0], device=xyz.device, dtype=xyz.dtype)
    L[:, idx_good] = L_good
    return L.to(torch.float32)


def verify_reference_invariance(
    L: torch.Tensor, tol: float = 1e-6
) -> Tuple[bool, float]:
    """Check ``L @ 1 == 0``, i.e. the operator is invariant to any monopolar reference.

    Re-referencing subtracts one channel-linear combination from every channel,
    which is exactly adding a multiple of the all-ones direction.  An operator
    that annihilates that direction is reference invariant by construction.

    Returns:
        ``(is_invariant, max_abs_row_sum)``
    """
    row_sums = L.sum(dim=-1)
    max_abs = float(row_sums.abs().max().item())
    scale = float(L.abs().max().item()) + 1e-12
    return (max_abs / scale) < tol, max_abs


class SSLOperatorCache:
    """Memory + optional disk cache of SSL operators keyed by montage and params.

    Key: ``(montage_hash, stiffness, lambda, n_legendre)`` as required by the
    project spec.  ``hits``/``misses`` are exposed so the smoke test can assert
    that the cache is actually being used.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir
        self._mem: Dict[str, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0
        self.skips: Dict[str, str] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def make_key(mhash: str, stiffness: int, lambda_reg: float, n_legendre: int) -> str:
        return f"ssl_{mhash}_m{stiffness}_l{lambda_reg:g}_n{n_legendre}"

    def get(
        self,
        channel_names: Sequence[str],
        xyz: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
        cfg: SSLConfig,
        derivation_type: str = "monopolar",
    ) -> Optional[torch.Tensor]:
        """Return the cached/-built operator, or ``None`` if the branch is skipped."""
        if not cfg.enabled:
            return None
        if derivation_type and derivation_type.lower().startswith("bipolar"):
            self._skip("bipolar", "SSL skipped: surface Laplacian is defined on "
                                  "monopolar potentials, input is a bipolar derivation.")
            return None
        mhash = montage_hash(channel_names, xyz, channel_mask)
        key = self.make_key(mhash, cfg.stiffness, cfg.lambda_reg, cfg.n_legendre)
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        if self.cache_dir:
            path = os.path.join(self.cache_dir, key + ".pt")
            if os.path.exists(path):
                L = torch.load(path, map_location="cpu")
                self._mem[key] = L
                self.hits += 1
                return L
        self.misses += 1
        try:
            L = build_ssl_operator(
                xyz, channel_mask, cfg.stiffness, cfg.lambda_reg,
                cfg.n_legendre, cfg.min_channels,
            )
        except SSLSkipped as exc:
            self._skip("low_density", str(exc))
            return None
        ok, resid = verify_reference_invariance(L)
        if not ok:
            logger.warning(
                "SSL operator for montage %s has residual row sum %.3e; reference "
                "invariance may be degraded (check coordinates and lambda).", mhash, resid
            )
        self._mem[key] = L
        if self.cache_dir:
            tmp = os.path.join(self.cache_dir, key + ".pt.tmp")
            torch.save(L, tmp)
            os.replace(tmp, os.path.join(self.cache_dir, key + ".pt"))
        return L

    def _skip(self, reason: str, message: str) -> None:
        if reason not in self.skips:
            logger.warning(message)
        self.skips[reason] = message

    def stats(self) -> Dict[str, object]:
        return {"hits": self.hits, "misses": self.misses, "cached": len(self._mem),
                "skips": dict(self.skips)}
