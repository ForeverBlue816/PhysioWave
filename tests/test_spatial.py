"""SSL branch, GL branch and A_dyn spatial-statistics tests."""

from __future__ import annotations

import math

import pytest
import torch

from physiowave.data.montages import montage
from physiowave.spatial.geometry import normalize_to_sphere
from physiowave.spatial.graph_laplacian import GLConfig, GraphLaplacianBranch
from physiowave.spatial.spatial_stats import (
    DEFAULT_BANDS,
    DynGraphConfig,
    SpatialStatGraph,
    band_fourier_coefficients,
    condition_number,
    imaginary_coherence,
    magnitude_coherence,
    shrunk_covariance,
    weighted_phase_lag_index,
)
from physiowave.spatial.spline_laplacian import (
    SSLConfig,
    SSLOperatorCache,
    SSLSkipped,
    build_ssl_operator,
    spline_interpolation_matrix,
    verify_reference_invariance,
)


# --------------------------------------------------------------------------- #
# SSL branch
# --------------------------------------------------------------------------- #
def test_ssl_operator_is_reference_invariant():
    """`L_ssl @ X` is unchanged by CAR and by a linked-mastoid re-reference.

    The surface Laplacian annihilates the all-ones channel direction, and a
    re-reference adds exactly a multiple of that direction, so this must hold to
    numerical precision.  It is the property the pretraining anchor relies on.
    """
    names, xyz = montage("standard_1010_64")
    L = build_ssl_operator(xyz)
    ok, resid = verify_reference_invariance(L)
    assert ok, f"L @ 1 is not zero; residual {resid:.3e}"

    x = torch.randn(3, len(names), 512)
    ref = torch.einsum("ij,bjt->bit", L, x)
    scale = ref.abs().max().item()

    car = x - x.mean(dim=1, keepdim=True)
    i, j = names.index("M1"), names.index("M2")
    lm = x - 0.5 * (x[:, i:i + 1] + x[:, j:j + 1])
    single = x - x[:, i:i + 1]

    results = {}
    for label, view in (("CAR", car), ("linked_mastoids", lm), ("single_mastoid", single)):
        d = (torch.einsum("ij,bjt->bit", L, view) - ref).abs().max().item()
        results[label] = d / scale
        assert d / scale < 1e-5, f"{label}: relative deviation {d / scale:.3e}"
    print("SSL reference-invariance relative deviations:", results)


def test_ssl_low_density_and_bipolar_skip(caplog):
    """Sparse montages and bipolar derivations disable SSL, with a logged reason."""
    names, xyz = montage("standard_1020_19")
    with pytest.raises(SSLSkipped):
        build_ssl_operator(xyz[:8])                       # 8 < min_channels=16

    cache = SSLOperatorCache()
    with caplog.at_level("WARNING"):
        assert cache.get(names[:8], xyz[:8], None, SSLConfig()) is None
        assert cache.get(names, xyz, None, SSLConfig(), derivation_type="bipolar") is None
    assert "low_density" in cache.stats()["skips"]
    assert "bipolar" in cache.stats()["skips"]


def test_ssl_cache_hits(tmp_path):
    """The operator is built once per montage and then served from cache."""
    names, xyz = montage("standard_1010_61")
    cfg = SSLConfig(cache_dir=str(tmp_path))
    cache = SSLOperatorCache(str(tmp_path))
    a = cache.get(names, xyz, None, cfg)
    b = cache.get(names, xyz, None, cfg)
    assert a is not None and torch.equal(a, b)
    assert cache.stats()["misses"] == 1 and cache.stats()["hits"] >= 1
    fresh = SSLOperatorCache(str(tmp_path))               # cold memory, warm disk
    c = fresh.get(names, xyz, None, cfg)
    assert torch.allclose(a, c) and fresh.stats()["hits"] == 1


def test_ssl_interpolates_bad_channels_first():
    """A bad electrode is spline-interpolated before the Laplacian is formed.

    Its column in the operator must be exactly zero: if the recorded (garbage)
    value could reach the output, one bad electrode would contaminate every
    output channel, since each CSD value is a weighted sum over all electrodes.
    """
    names, xyz = montage("standard_1010_64")
    mask = torch.ones(len(names), dtype=torch.bool)
    bad = names.index("C3")
    mask[bad] = False
    L = build_ssl_operator(xyz, mask)
    assert L[:, bad].abs().max() == 0.0, "a bad channel still feeds the SSL output"

    x = torch.randn(2, len(names), 256)
    x_corrupt = x.clone()
    x_corrupt[:, bad] = 1e4 * torch.randn(2, 256)
    a = torch.einsum("ij,bjt->bit", L, x)
    b = torch.einsum("ij,bjt->bit", L, x_corrupt)
    assert torch.allclose(a, b), "corrupting a masked channel changed the SSL output"


def test_spline_interpolation_recovers_a_smooth_field():
    """Interpolating a held-out electrode of a smooth field is accurate."""
    names, xyz = montage("standard_1010_64")
    sphere = normalize_to_sphere(xyz.double())
    field = (sphere[:, 2] * 2.0 + sphere[:, 0]).unsqueeze(-1)    # smooth on the sphere
    hold = names.index("Cz")
    good = torch.ones(len(names), dtype=torch.bool)
    good[hold] = False
    M = spline_interpolation_matrix(sphere, good)
    est = (M @ field[good])[hold, 0].item()
    assert abs(est - field[hold, 0].item()) < 0.1, f"interpolated {est}, true {field[hold, 0]}"


# --------------------------------------------------------------------------- #
# A_dyn -- spatial statistics
# --------------------------------------------------------------------------- #
def _band_noise(T: int, fs: float, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    """Unit-variance band-limited noise -- a realistic stand-in for an EEG rhythm."""
    X = torch.fft.rfft(torch.randn(T, generator=gen))
    f = torch.fft.rfftfreq(T, 1.0 / fs)
    X[(f < lo) | (f >= hi)] = 0
    y = torch.fft.irfft(X, n=T)
    return y / y.std()


def test_wpli_is_near_zero_for_a_zero_phase_shared_source():
    """Volume-conduction robustness, measured directly.

    Two channels are built from ONE shared alpha source with **zero** phase lag --
    exactly what instantaneous volume conduction produces -- plus independent
    noise.  wPLI and imaginary coherence must be near zero, while correlation and
    magnitude coherence stay large; that is precisely why neither of the latter
    two is offered as a volume-conduction-robust option anywhere in this codebase.

    The shared source is band-limited noise rather than a pure tone on purpose: a
    single-frequency source concentrates all its energy in one or two FFT bins, so
    the estimator genuinely has only a couple of independent observations and any
    phase-lag index becomes high-variance.  Real rhythms are broadband within
    their band, which is the regime these estimators are designed for.
    """
    fs, T = 256.0, 2048
    g = torch.Generator().manual_seed(7)
    src = _band_noise(T, fs, 8.0, 13.0, g)
    x = torch.stack([src + 0.3 * torch.randn(T, generator=g),
                     0.7 * src + 0.3 * torch.randn(T, generator=g)]).unsqueeze(0)
    z = band_fourier_coefficients(x, fs, [(8.0, 13.0)])[0]

    wpli = weighted_phase_lag_index(z)[0, 0, 1].item()
    imcoh = imaginary_coherence(z)[0, 0, 1].item()
    coh = magnitude_coherence(z)[0, 0, 1].item()
    corr = shrunk_covariance(x, DynGraphConfig())[0, 0, 1].item()
    print(f"zero-phase shared source: wPLI={wpli:.4f} imCoh={imcoh:.4f} "
          f"|coh|={coh:.4f} corr={corr:.4f}")

    assert wpli < 0.1, f"wPLI {wpli:.4f} should be near zero for a zero-phase pair"
    assert imcoh < 0.1, f"imCoh {imcoh:.4f} should be near zero for a zero-phase pair"
    assert coh > 0.8, f"magnitude coherence {coh:.4f} should be large (it is contaminated)"
    assert abs(corr) > 0.5, f"correlation {corr:.4f} should be large (it is contaminated)"


def test_wpli_detects_a_genuine_phase_lag():
    """The same estimator must still respond to a real phase-lagged relation."""
    fs, T = 256.0, 2048
    g = torch.Generator().manual_seed(8)
    src = _band_noise(T, fs, 8.0, 13.0, g)
    lag = int(0.025 * fs)
    x = torch.stack([src + 0.3 * torch.randn(T, generator=g),
                     0.7 * torch.roll(src, lag) + 0.3 * torch.randn(T, generator=g)]).unsqueeze(0)
    z = band_fourier_coefficients(x, fs, [(8.0, 13.0)])[0]
    wpli = weighted_phase_lag_index(z)[0, 0, 1].item()
    print(f"phase-lagged pair: wPLI={wpli:.4f}")
    assert wpli > 0.5, f"wPLI {wpli:.4f} should be large for a genuinely lagged pair"


def test_band_wise_covariance_is_per_band():
    """Band-wise A_dyn produces one matrix per band, each band-specific."""
    fs, T, C = 256.0, 1024, 8
    t = torch.arange(T) / fs
    x = torch.randn(2, C, T) * 0.1
    x[:, :4] += torch.sin(2 * math.pi * 10 * t)           # alpha only on the first half
    x[:, 4:] += torch.sin(2 * math.pi * 25 * t)           # beta only on the second half
    m = SpatialStatGraph(DynGraphConfig(band_wise=True))
    per_band = m.per_band_graphs(x, fs)
    assert per_band.shape == (2, len(DEFAULT_BANDS), C, C)
    names = list(DEFAULT_BANDS)
    alpha, beta = per_band[0, names.index("alpha")], per_band[0, names.index("beta")]
    assert alpha[0, 1] > alpha[0, 5], "alpha graph should link the alpha-carrying channels"
    assert beta[5, 6] > beta[0, 5], "beta graph should link the beta-carrying channels"


def test_shrinkage_is_numerically_stable():
    """Shrinkage keeps a rank-deficient sample covariance well conditioned."""
    C, T = 32, 8                                          # T << C: rank deficient
    x = torch.randn(4, C, T)
    raw = x - x.mean(-1, keepdim=True)
    S_raw = raw @ raw.transpose(-2, -1) / max(T - 1, 1)
    S = shrunk_covariance(x, DynGraphConfig(shrinkage="ledoit_wolf", correlation=False))
    assert torch.isfinite(S).all()
    assert condition_number(S).max() < 1e6, "shrunk covariance is still ill conditioned"
    assert condition_number(S).max() < condition_number(S_raw + 1e-12 *
                                                        torch.eye(C)).max()
    ev = torch.linalg.eigvalsh(S)
    assert (ev > 0).all(), "shrunk covariance must be positive definite"


def test_a_dyn_is_detached():
    """A_dyn enters the model as structure, never as a differentiable feature."""
    m = SpatialStatGraph(DynGraphConfig())
    x = torch.randn(2, 16, 512, requires_grad=True)
    A = m(x, 256.0)
    assert not A.requires_grad
    assert x.grad is None


@pytest.mark.parametrize("kind", ["cov", "wpli", "imcoh"])
def test_dyn_graph_types_run(kind):
    m = SpatialStatGraph(DynGraphConfig(dyn_graph_type=kind))
    A = m(torch.randn(2, 12, 512), 256.0)
    assert A.shape == (2, 12, 12) and torch.isfinite(A).all()


# --------------------------------------------------------------------------- #
# GL branch
# --------------------------------------------------------------------------- #
def test_gl_branch_shapes_and_gate():
    names, xyz = montage("standard_1020_19")
    gl = GraphLaplacianBranch(GLConfig(gate_init=0.1))
    x = torch.randn(2, len(names), 256)
    out = gl(x, xyz)
    assert out.shape == x.shape
    L = gl.laplacian(xyz)
    assert torch.allclose(L, L.transpose(0, 1), atol=1e-5), "graph Laplacian must be symmetric"
    out.sum().backward()
    assert gl.gate.grad is not None and gl.edge_delta.grad is not None
