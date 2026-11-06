"""WAST / DWT tests: perfect reconstruction, critical sampling, boundaries, gradients."""

from __future__ import annotations

import math

import pytest
import torch

from physiowave.wavelet.dwt import (
    WaveletTransform1D,
    coeff_lengths,
    condition_number,
    dwt,
    get_filters,
    resolve_boundary_mode,
)
from physiowave.wavelet.wast import WAST, WASTConfig


def _stationary_signal(P: int, n: int = 8, cycles: float = 2.5, offset: float = 0.0,
                       dtype=torch.float64) -> torch.Tensor:
    t = torch.arange(P, dtype=dtype) / P
    base = torch.sin(2 * math.pi * cycles * t) + 0.3 * torch.sin(2 * math.pi * 7 * t)
    return (base + offset).unsqueeze(0).repeat(n, 1)


@pytest.mark.parametrize("mode", ["reflect", "symmetric", "periodization", "zero", "constant"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_dwt_roundtrip(mode, level):
    """Analysis followed by synthesis reproduces the input to numerical precision."""
    P = 64
    x = _stationary_signal(P)
    wt = WaveletTransform1D("bior4.4", level, mode, learnable=False).double()
    rec = wt.synthesis(wt.analysis(x), P)
    err = (rec - x).abs().max().item()
    assert err < 1e-9, f"mode={mode} level={level} round-trip error {err:.3e}"


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("P", [32, 64, 128])
def test_critical_sampling_length_conservation(level, P):
    """The multi-level coefficient budget equals the patch length exactly."""
    lens = coeff_lengths(P, level)
    assert sum(lens) == P, f"coeff_lengths({P}, {level}) sums to {sum(lens)}"
    wt = WaveletTransform1D("bior4.4", level, "reflect", learnable=False)
    coeffs = wt.analysis(torch.randn(4, P))
    assert [c.shape[-1] for c in coeffs] == lens
    assert sum(c.shape[-1] for c in coeffs) == P


def test_orthogonal_wavelet_falls_back_loudly(caplog):
    """A non-symmetric wavelet cannot use `reflect`; the fallback is logged."""
    with caplog.at_level("WARNING"):
        mode = resolve_boundary_mode("db4", "reflect")
    assert mode == "periodization"
    assert "periodization" in caplog.text
    assert resolve_boundary_mode("bior4.4", "reflect") == "reflect"


def test_boundary_artifact_roundtrip():
    """Edge reconstruction error must not exceed 3x the centre error for `reflect`.

    `zero` is measured and reported but is allowed to fail: it is not the default
    precisely because it asserts the signal vanishes just outside the patch.
    """
    P, edge = 64, 8
    x = _stationary_signal(P, offset=1.5)                 # baseline drift, as in real EEG
    report = {}
    for mode in ("reflect", "zero"):
        wt = WaveletTransform1D("bior4.4", 3, mode, learnable=False).double()
        err = (wt.synthesis(wt.analysis(x), P) - x) ** 2
        e = 0.5 * (err[:, :edge].mean() + err[:, -edge:].mean()).item()
        c = err[:, P // 2 - edge: P // 2 + edge].mean().item()
        report[mode] = (e, c, condition_number(P, "bior4.4", 3, mode))
    e, c, _ = report["reflect"]
    scale = float((x ** 2).mean())
    assert e <= 3 * c + 1e-12 * scale, (
        f"reflect edge MSE {e:.3e} exceeds 3x centre MSE {c:.3e}; report={report}"
    )
    print("boundary round-trip (edge_mse, center_mse, cond):", report)


def test_boundary_analysis_artifact():
    """Patch-local coefficients: `reflect` is closer to an infinite-context DWT.

    This is the measurement that actually justifies the default.  A patch is cut
    out of a longer signal with baseline drift; its finest detail band is compared
    against the band a whole-signal transform produces.  Zero extension asserts a
    step at each patch border, so its edge coefficients deviate far more.
    """
    P, S, level, edge = 64, 8, 2, 3
    T = P * S
    t = torch.arange(T, dtype=torch.float64) / P
    sig = (torch.sin(2 * math.pi * 2.5 * t) + 1.5
           + 0.05 * torch.randn(T, generator=torch.Generator().manual_seed(3), dtype=torch.float64))
    filt = get_filters("bior4.4", torch.float64)
    ref = dwt(sig.unsqueeze(0), filt, level, "periodization")[-1][0].view(S, P // 2)
    patches = sig.view(S, P)

    errs = {}
    for mode in ("reflect", "zero"):
        d1 = dwt(patches, filt, level, mode)[-1]
        e = (d1 - ref).abs()
        errs[mode] = 0.5 * (e[:, :edge].mean() + e[:, -edge:].mean()).item()
    assert errs["reflect"] < errs["zero"], (
        f"reflect should distort patch-edge coefficients less than zero: {errs}"
    )
    print("analysis-side edge deviation:", errs)


def test_wavelet_parameters_receive_gradient():
    """The wavelet filters are on the gradient path, not decorative."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=3))
    m.train()
    out = m(torch.randn(2, 4, 256))
    out["tokens"].pow(2).mean().backward()
    for name in ("dec_lo", "dec_hi"):
        g = getattr(m.wt, name).grad
        assert g is not None and g.abs().max() > 0, f"{name} received no gradient"
    assert m.subbands[0].depthwise.weight.grad.abs().max() > 0


def test_wavelet_parameters_change_output():
    """Perturbing the filters changes the tokens materially."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=3)).eval()
    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        base = m(x)["tokens"].clone()
        m.wt.dec_lo.add_(0.05)
        m.wt.clear_cache()
        pert = m(x)["tokens"]
    rel = ((pert - base).norm() / base.norm()).item()
    assert rel > 1e-3, f"changing the wavelet changed the output by only {rel:.2e}"


def test_token_counts_and_compression():
    """WAST does not inflate tokens with decomposition level; report the ratio."""
    C, T, K = 19, 1024, 16
    prev = None
    for level in (1, 2, 3, 4):
        m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=level))
        rep = m.token_report(C, T, K)
        S = T // 64
        assert rep["N_wast"] == C * S, "WAST token count must not depend on the level"
        assert rep["N_new"] == K * S, "compressed token count must be K * S"
        assert rep["N_old_legacy"] == (level + 1) * C * S
        if prev is not None:
            assert rep["N_wast"] == prev, "token count grew with the decomposition level"
        prev = rep["N_wast"]
    print("compression at level 4:", rep)
    assert rep["compression_vs_legacy_with_compression"] > 1.0


def test_shapes_and_assertions():
    m = WAST(WASTConfig(patch_size=64, embed_dim=48, level=2))
    out = m(torch.randn(3, 7, 512))
    assert out["tokens"].shape == (3, 7, 8, 48)
    assert out["raw_patches"].shape == (3, 7, 8, 64)
    assert out["patch_scores"].shape == (3, 8)
    with pytest.raises(AssertionError):
        m(torch.randn(3, 7, 500))                          # T not a multiple of P
    with pytest.raises(AssertionError):
        m(torch.randn(3, 7))                               # not [B, C, T]


def test_fgm_scores_are_detached():
    """Masking decisions must never back-propagate into the wavelet filters."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=2))
    out = m(torch.randn(2, 4, 256))
    assert not out["patch_scores"].requires_grad


def test_fold_tokenizer_is_token_neutral_and_uses_every_band():
    """'fold' packs subbands into features, so the token count is the patch count.

    And every band must actually reach the token: a tokenizer that silently
    dropped the coarsest scale would still have the right shape.
    """
    from physiowave.wavelet.wast import WAST, WASTConfig, split_embedding

    cfg = WASTConfig(patch_size=64, level=3, embed_dim=64, tokenizer="fold")
    tok = WAST(cfg).eval()
    B, C, T = 2, 5, 256
    with torch.no_grad():
        out = tok(torch.randn(B, C, T))
    S = T // cfg.patch_size
    assert out["tokens"].shape == (B, C, S, 64)
    assert sum(tok.band_dims) == 64 and len(tok.band_dims) == cfg.level + 1

    # Zeroing one band's projection must change the token -- for every band.
    x = torch.randn(B, C, T)
    with torch.no_grad():
        base = tok(x)["tokens"].clone()
        for i, proj in enumerate(tok.band_projs):
            w = proj.weight.clone()
            proj.weight.zero_()
            moved = (tok(x)["tokens"] - base).abs().max().item()
            proj.weight.copy_(w)
            assert moved > 1e-6, f"band {i} does not reach the token"


def test_split_embedding_is_exact_and_gives_every_band_room():
    from physiowave.wavelet.wast import split_embedding

    for dim in (16, 64, 256, 257):
        for lens in ([8, 8, 16, 32], [4, 4, 8], [32, 32]):
            dims = split_embedding(dim, lens)
            assert sum(dims) == dim and all(d >= 1 for d in dims)
    with pytest.raises(ValueError):
        split_embedding(3, [8, 8, 16, 32])


def test_fold_gate_starts_where_the_config_says():
    import math

    from physiowave.wavelet.wast import WAST, WASTConfig

    tok = WAST(WASTConfig(patch_size=32, level=2, embed_dim=32, tokenizer="fold",
                          fold_gate_init=0.1))
    assert math.isclose(float(torch.tanh(tok.fold_gate.detach())), 0.1, abs_tol=1e-6)


def test_every_tokenizer_produces_the_same_token_count():
    from physiowave.wavelet.wast import WAST, WASTConfig

    B, C, T, P = 2, 4, 256, 64
    counts = {}
    for mode in ("raw", "synthesis", "fold"):
        tok = WAST(WASTConfig(patch_size=P, level=3, embed_dim=32, tokenizer=mode)).eval()
        with torch.no_grad():
            counts[mode] = tuple(tok(torch.randn(B, C, T))["tokens"].shape)
    assert len(set(counts.values())) == 1, counts
    assert counts["fold"] == (B, C, T // P, 32)


def test_fold_rejects_pre_patch_placement():
    from physiowave.wavelet.wast import WASTConfig

    with pytest.raises(ValueError, match="post_patch"):
        WASTConfig(tokenizer="fold", placement="pre_patch")


def test_per_channel_filters_match_the_shared_bank_at_init():
    """Identical initial filters must give identical tokens.

    This is the check that the channel bookkeeping is right. ``analyse`` flattens
    to ``((b*C + c)*S + s)`` and rebuilds by stacking on axis 1; getting that
    order wrong would pair each channel's signal with another channel's filters
    and still produce a plausible-looking tensor.
    """
    from physiowave.wavelet.wast import WAST, WASTConfig

    B, C, T, P = 2, 5, 256, 64
    torch.manual_seed(0)
    x = torch.randn(B, C, T)
    for mode in ("synthesis", "fold"):
        shared = WAST(WASTConfig(patch_size=P, level=3, embed_dim=32,
                                 tokenizer=mode)).eval()
        perch = WAST(WASTConfig(patch_size=P, level=3, embed_dim=32, tokenizer=mode,
                                channel_filters=True, max_channels=8)).eval()
        perch.load_state_dict({k: v for k, v in shared.state_dict().items()
                               if k in perch.state_dict()}, strict=False)
        with torch.no_grad():
            a, b = shared(x)["tokens"], perch(x)["tokens"]
        assert torch.equal(a, b), mode


def test_a_channels_filter_only_moves_its_own_tokens():
    from physiowave.wavelet.wast import WAST, WASTConfig

    tok = WAST(WASTConfig(patch_size=64, level=3, embed_dim=32, tokenizer="fold",
                          channel_filters=True, max_channels=8)).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 5, 256)
    with torch.no_grad():
        base = tok(x)["tokens"].clone()
        tok.channel_wt[3].dec_lo.add_(0.05)
        moved = (tok(x)["tokens"] - base).abs().amax(dim=(0, 2, 3))
    assert moved[3] > 1e-4, moved
    assert moved[[0, 1, 2, 4]].max() == 0.0, moved


def test_too_many_channels_for_the_bank_count_is_an_error():
    from physiowave.wavelet.wast import WAST, WASTConfig

    tok = WAST(WASTConfig(patch_size=64, level=2, embed_dim=32,
                          channel_filters=True, max_channels=2)).eval()
    with pytest.raises(ValueError, match="max_channels"):
        tok(torch.randn(1, 4, 128))


# --------------------------------------------------------------------------- #
# Wavelet initialisation
# --------------------------------------------------------------------------- #
def _power_complementary_error(lo, hi, n=1024):
    """max | |H_lo(w)|^2 + |H_hi(w)|^2 - 2 |. Zero for a valid orthogonal pair."""
    import numpy as np

    P = np.abs(np.fft.rfft(lo, n)) ** 2 + np.abs(np.fft.rfft(hi, n)) ** 2
    return float(np.abs(P - 2.0).max())


def test_interpolating_a_wavelet_to_a_longer_kernel_stops_being_a_filter_bank():
    """The failure the 'pad' mode exists to avoid, pinned so it cannot be lost.

    Stretching a filter in time compresses it in frequency, so the half-band
    cutoff moves and the lowpass/highpass pair no longer partitions the
    spectrum. The error below is against a theoretical maximum of 2.0.
    """
    from wavelet_modules import load_wavelet_kernel

    lo, hi = load_wavelet_kernel("sym4", 16, "interp")
    assert _power_complementary_error(lo.numpy(), hi.numpy()) > 1.9
    # Energy is not preserved either: a valid orthonormal lowpass has ||h||=1.
    assert float((lo ** 2).sum()) < 0.5


def test_padding_preserves_every_property_that_makes_a_wavelet_a_wavelet():
    import numpy as np

    from wavelet_modules import load_wavelet_kernel

    for name in ("sym4", "sym5", "db6", "db8", "sym8"):
        lo, hi = load_wavelet_kernel(name, 16, "pad")
        lo_np = lo.numpy()
        assert abs(float(lo.sum()) - 2 ** 0.5) < 1e-5, name          # sum h = sqrt 2
        assert abs(float(hi.sum())) < 1e-5, name                      # sum g = 0
        assert abs(float((lo ** 2).sum()) - 1.0) < 1e-5, name         # unit energy
        assert _power_complementary_error(lo_np, hi.numpy()) < 1e-4, name
        # Orthonormality: the filter is orthogonal to its even shifts.
        for k in range(1, 8):
            assert abs(float(np.dot(lo_np, np.roll(lo_np, 2 * k)))) < 1e-5, (name, k)


def test_padded_taps_are_centred_so_wavelets_share_a_group_delay():
    """AdaptiveWaveletSelector mixes several families in one layer.

    Filters of different native lengths must sit at the same delay, or the
    selector is blending versions of the signal that are shifted relative to
    each other.
    """
    import numpy as np

    from wavelet_modules import load_wavelet_kernel

    centres = []
    for name in ("sym4", "sym5", "db6", "db8"):
        lo = load_wavelet_kernel(name, 16, "pad")[0].numpy()
        nz = np.nonzero(np.abs(lo) > 1e-12)[0]
        centres.append((nz[0] + nz[-1]) / 2.0)
    assert max(centres) - min(centres) <= 0.5, centres


def test_a_wavelet_too_long_for_the_kernel_is_rejected_rather_than_squeezed():
    from wavelet_modules import load_wavelet_kernel

    with pytest.raises(ValueError, match="18 taps"):
        load_wavelet_kernel("coif3", 16, "pad")     # coif3 is 18 taps
    with pytest.raises(ValueError, match="interp' or 'pad"):
        load_wavelet_kernel("db6", 16, "nearest")


def test_wave_init_mode_reaches_the_filters():
    """A flag that does not change the weights is worse than no flag."""
    import torch

    from model import create_wavelet_classifier

    kw = dict(in_channels=2, max_level=2, embed_dim=32, depth=1, num_heads=4,
              num_classes=5, patch_size=(1, 50), wave_kernel_size=16,
              wavelet_names=["sym4"])
    a = create_wavelet_classifier(**kw, wave_init_mode="interp")
    b = create_wavelet_classifier(**kw, wave_init_mode="pad")
    wa = a.wavelet_decomp.selector.wavelet_filters[0].low_filter.weight.detach()
    wb = b.wavelet_decomp.selector.wavelet_filters[0].low_filter.weight.detach()
    assert not torch.allclose(wa, wb)
    assert abs(float((wb[0, 0] ** 2).sum()) - 1.0) < 1e-5     # the padded one is unit norm
    assert float((wa[0, 0] ** 2).sum()) < 0.5                 # the stretched one is not


def test_cross_scale_attention_does_not_materialise_its_score_matrix():
    """need_weights=True is a silent memory cliff, quadratic in window length.

    nn.MultiheadAttention's default returns the [B, heads, len(q), len(kv)]
    score matrix and takes the unfused path to build it. CrossScaleCAFFN
    discards those weights, but the allocation still happens: at a 30 s EEG
    epoch it is 17.2 GiB for one decomposition level at batch 64, which is
    where a Sleep-EDF run died. At an sEMG window it is 0.13 GiB and invisible,
    so nothing about sEMG testing protects this.
    """
    import torch

    from wavelet_modules import CrossScaleCAFFN

    class Saved:
        def __init__(self):
            self.seen, self.total = set(), 0

        def pack(self, t):
            st = t.untyped_storage()
            if st.data_ptr() not in self.seen:
                self.seen.add(st.data_ptr())
                self.total += st.nbytes()
            return t

        def unpack(self, t):
            return t

    torch.manual_seed(0)
    B, C, T = 2, 4, 2048
    m = CrossScaleCAFFN(C).train()
    x = torch.randn(B, C, 1, T)
    prev = [torch.randn(B, C, 1, T), torch.randn(B, C, 1, T)]

    s = Saved()
    with torch.autograd.graph.saved_tensors_hooks(s.pack, s.unpack):
        out = m(x, prev)
    assert out.shape == x.shape

    # A materialised matrix would be B * heads * T * 2T * 4 bytes on its own.
    matrix_bytes = B * 4 * T * (2 * T) * 4
    assert s.total < matrix_bytes / 10, (s.total, matrix_bytes)
    # And the cost must grow linearly in the window, not quadratically.
    s2 = Saved()
    with torch.autograd.graph.saved_tensors_hooks(s2.pack, s2.unpack):
        m(torch.randn(B, C, 1, 2 * T),
          [torch.randn(B, C, 1, 2 * T), torch.randn(B, C, 1, 2 * T)])
    assert s2.total < 3 * s.total, (s.total, s2.total)
