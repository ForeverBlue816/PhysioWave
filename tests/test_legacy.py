"""Legacy compatibility: the original model still runs and still has its old shapes."""

from __future__ import annotations

import pytest
import torch

from physiowave.config import load_config
from physiowave.models.build import build_model
from physiowave.models.legacy import build_legacy_model, legacy_available, legacy_token_count

pytestmark = pytest.mark.skipif(not legacy_available(),
                                reason="legacy top-level modules not importable")


def _legacy(**kw):
    params = dict(in_channels=8, max_level=2, wave_kernel_size=16,
                  wavelet_names=["db4", "sym4"], patch_size=(1, 32),
                  embed_dim=32, depth=1, num_heads=4)
    params.update(kw)
    return build_legacy_model(**params)


def test_legacy_forward_shapes_unchanged():
    """Regression: with every new module off, the legacy shapes are exactly as before.

    ``forward_features`` returns ``[B, (J+1)*C/p_f * T/p_t, D]`` -- the sequence
    length that grows with the decomposition level, which is the behaviour WAST
    replaces (and which this test pins down so the replacement stays honest).
    """
    B, C, T, J, D = 2, 8, 256, 2, 32
    model = _legacy().eval()
    x = torch.randn(B, C, T)
    feats = model(x, task="features")
    expected_tokens = (J + 1) * C * (T // 32)
    assert feats.shape == (B, expected_tokens, D), (
        f"legacy feature shape changed: {tuple(feats.shape)}"
    )
    assert legacy_token_count(C, J, T, (1, 32))["num_tokens"] == expected_tokens


def test_legacy_pretrain_forward():
    B, C, T = 2, 8, 256
    model = _legacy().eval()
    pred, mask, target = model(torch.randn(B, C, T), task="pretrain", mask_ratio=0.5)
    assert pred.shape == target.shape
    assert mask.shape == pred.shape[:2]
    assert 0.4 < mask.float().mean().item() < 0.6


def test_legacy_classification_head():
    model = _legacy(task_type="classification", num_classes=4).eval()
    logits = model(torch.randn(2, 8, 256), task="downstream", task_name="classification")
    assert logits.shape == (2, 4)


def test_legacy_config_builds():
    cfg = load_config("model/legacy")
    cfg["model"]["legacy"].update({"in_channels": 8, "embed_dim": 32, "depth": 1,
                                   "num_heads": 4, "max_level": 2,
                                   "wavelet_names": ["db4"], "patch_size": [1, 32]})
    model = build_model(cfg)
    assert model(torch.randn(1, 8, 128), task="features").shape[-1] == 32


def test_legacy_token_inflation_is_linear_in_level():
    """The legacy token count grows with J; this is the baseline WAST is measured against."""
    counts = [legacy_token_count(19, J, 1024, (1, 64))["num_tokens"] for J in (1, 2, 3, 4)]
    assert counts == [2 * 19 * 16, 3 * 19 * 16, 4 * 19 * 16, 5 * 19 * 16]


def test_legacy_with_channel_id():
    """Rung 2 of the ablation ladder: legacy plus a learnable per-channel vector."""
    from physiowave.models.legacy import LegacyWithChannelID

    model = LegacyWithChannelID(in_channels=8, max_level=2, wave_kernel_size=16,
                                wavelet_names=["db4"], patch_size=(1, 32),
                                embed_dim=32, depth=1, num_heads=4)
    x = torch.randn(2, 8, 128)
    out = model(x, task="features")
    assert out.shape == (2, (2 + 1) * 8 * (128 // 32), 32)
    out.sum().backward()
    assert model.channel_id.grad is not None and model.channel_id.grad.abs().max() > 0

    # It is index-based, so it cannot accept a different montage size at all.
    with pytest.raises(AssertionError, match="montage-specific"):
        model(torch.randn(2, 12, 128), task="features")


def test_default_block_is_the_unmodified_original():
    """norm/ffn/qk_norm defaults must reproduce the original block exactly.

    The switches exist so each modernisation is an ablation row. If a default
    ever flipped, every "legacy" number in the paper would quietly become a
    different architecture's.
    """
    import torch.nn as nn

    from transformer_modules import TransformerBlock

    blk = TransformerBlock(dim=32, num_heads=4)
    assert isinstance(blk.norm1, nn.LayerNorm) and isinstance(blk.norm2, nn.LayerNorm)
    assert isinstance(blk.attn.q_norm, nn.Identity) and isinstance(blk.attn.k_norm, nn.Identity)
    assert isinstance(blk.mlp, nn.Sequential)
    linears = [m for m in blk.mlp if isinstance(m, nn.Linear)]
    assert len(linears) == 2 and linears[0].out_features == int(32 * 4.0)


def test_modern_block_variants_build_and_stay_the_same_size():
    """SwiGLU is scaled to keep the comparison about gating, not parameter count."""
    from model import create_wavelet_classifier

    kw = dict(in_channels=8, max_level=2, embed_dim=64, depth=2, num_heads=4, num_classes=5)
    base = create_wavelet_classifier(**kw)
    modern = create_wavelet_classifier(norm="rmsnorm", ffn="swiglu", qk_norm=True, **kw)
    n = lambda m: sum(p.numel() for p in m.parameters())          # noqa: E731
    assert abs(n(modern) - n(base)) / n(base) < 0.02, (n(base), n(modern))

    x = torch.randn(2, 8, 256)
    for m in (base, modern):
        m.eval()
        with torch.no_grad():
            out = m(x)
        out = out if torch.is_tensor(out) else out["logits"]
        assert torch.isfinite(out).all()


def test_unknown_block_variants_are_rejected():
    from transformer_modules import TransformerBlock

    with pytest.raises(ValueError, match="ffn"):
        TransformerBlock(dim=32, num_heads=4, ffn="gelu-glu")
    with pytest.raises(ValueError, match="norm"):
        TransformerBlock(dim=32, num_heads=4, norm="batchnorm")


def test_legacy_runs_through_the_new_trainer_interface():
    """The legacy model must accept (x, meta) and return a logits dict.

    This is what lets the same training loop drive both architectures, which is
    the only way to tell an architecture gap from a pipeline gap.
    """
    cfg = load_config("pretrain/semg_legacy",
                      ["model.legacy.depth=1", "model.legacy.embed_dim=32",
                       "model.legacy.max_level=2", "model.legacy.in_channels=8"])
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()

    from physiowave.channels.tare import ChannelMeta

    meta = ChannelMeta([f"ch{i:02d}" for i in range(8)], torch.zeros(8, 3))
    with torch.no_grad():
        out = enc(torch.randn(2, 8, 256), meta)
    assert isinstance(out, dict) and out["logits"].shape == (2, 5)
    assert torch.isfinite(out["logits"]).all()


def test_the_two_dataset_classes_read_identical_tensors(tmp_path):
    """finetune.py and finetune_main must feed the model the same numbers.

    They have separate Dataset implementations over the same HDF5, and a
    difference here would look exactly like an architecture difference in a
    benchmark table.
    """
    import importlib.util

    import h5py
    import numpy as np

    from physiowave.train.finetune_main import LabelledWindows

    path = tmp_path / "train.h5"
    rng = np.random.default_rng(0)
    data = rng.normal(size=(32, 8, 128)).astype(np.float32)
    label = rng.integers(0, 4, 32).astype(np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("label", data=label)

    spec = importlib.util.spec_from_file_location("_ft_for_test", "finetune.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)

    new, old = LabelledWindows(str(path)), ft.TimeSeriesDataset(str(path))
    assert len(new) == len(old)
    for i in range(len(new)):
        xn, yn = new[i][:2]           # the third element is the window index
        xo, yo = old[i]
        assert torch.equal(xn, xo) and int(yn) == int(yo), i


def test_scale_fold_reduces_the_token_count_by_the_band_count():
    """(J+1)*C*S tokens become C*S, and nothing else about the model moves."""
    from model import create_wavelet_classifier

    kw = dict(in_channels=8, max_level=3, embed_dim=64, depth=1, num_heads=4,
              num_classes=5, patch_size=(1, 32), wavelet_names=["db4", "sym4"])
    x = torch.randn(2, 8, 128)
    counts, params = {}, {}
    for fold in ("none", "mean", "learned"):
        m = create_wavelet_classifier(**kw, scale_fold=fold).eval()
        with torch.no_grad():
            tok = m.prepare_tokens(m.fold_scales(m.wavelet_decomp(x)).unsqueeze(1))
        counts[fold] = tok.shape[1]
        params[fold] = sum(p.numel() for p in m.parameters())
    assert counts["none"] == (3 + 1) * counts["mean"], counts
    assert counts["mean"] == counts["learned"]
    # The fold weight is (J+1)*C scalars; the budget must not move meaningfully.
    assert abs(params["learned"] - params["none"]) == (3 + 1) * 8


def test_scale_fold_groups_bands_not_channels():
    """Spec(X) is scale-major; folding the wrong axis would still typecheck.

    A tensor whose bands differ but whose channels do not must survive folding
    with per-channel structure intact -- if the reshape grouped channels instead
    of bands, distinct channels would be averaged into each other.
    """
    from model import create_wavelet_classifier

    m = create_wavelet_classifier(in_channels=4, max_level=2, embed_dim=32, depth=1,
                                  num_heads=4, num_classes=3, patch_size=(1, 32),
                                  wavelet_names=["db4"], scale_fold="mean").eval()
    B, C, T, J1 = 1, 4, 64, 3
    # Band b, channel c carries the constant value c. Averaging over bands must
    # therefore return exactly [0, 1, 2, 3] per channel.
    spec = torch.zeros(B, J1 * C, T)
    for b in range(J1):
        for c in range(C):
            spec[0, b * C + c] = float(c)
    out = m.fold_scales(spec)
    assert out.shape == (B, C, T)
    assert torch.allclose(out[0, :, 0], torch.arange(C, dtype=torch.float32))


def test_unknown_scale_fold_is_rejected():
    from model import create_wavelet_classifier

    with pytest.raises(ValueError, match="scale_fold"):
        create_wavelet_classifier(in_channels=4, max_level=2, num_classes=3,
                                  scale_fold="idwt")


def test_every_fold_mode_reaches_the_same_token_count():
    """The ladder differs in how the bands combine, not in how many tokens result."""
    from model import create_wavelet_classifier

    kw = dict(in_channels=8, max_level=3, embed_dim=64, depth=1, num_heads=4,
              num_classes=5, patch_size=(1, 32), wavelet_names=["db4", "sym4"])
    x = torch.randn(2, 8, 128)
    counts = {}
    for fold in ("none", "mean", "learned", "softmax", "dynamic"):
        m = create_wavelet_classifier(**kw, scale_fold=fold).eval()
        with torch.no_grad():
            counts[fold] = m.prepare_tokens(
                m.fold_scales(m.wavelet_decomp(x)).unsqueeze(1)).shape[1]
    assert counts["none"] == 4 * counts["mean"], counts
    assert len(set(v for k, v in counts.items() if k != "none")) == 1, counts


def test_dynamic_fold_starts_exactly_at_the_mean_fold():
    """Every added mechanism is identity at step 0.

    This is the property that makes the dynamic fold an ablation *of* the mean
    fold rather than a different model: if it does not hold, a worse number
    cannot be attributed to the dynamics because the starting point moved too.
    """
    from wavelet_modules import ScaleFold

    torch.manual_seed(0)
    # Bands of very different magnitude, so an accidental renormalisation shows.
    x = torch.randn(4, 24, 128) * torch.tensor(
        [4.0, 1.0, 0.2]).repeat_interleave(8).view(1, -1, 1)
    ref = x.view(4, 3, 8, 128).mean(dim=1)
    for kwargs in ({}, {"synthesis_kernel": 3}, {"synthesis_kernel": 5},
                   {"scale_dropout": 0.2}):
        fold = ScaleFold("dynamic", num_scales=3, in_channels=8, patch_len=32,
                         **kwargs).eval()
        assert torch.allclose(fold(x), ref, atol=1e-5), kwargs
    # Shrinkage is deliberately *near* identity rather than identity: the
    # threshold starts at softplus(-6) sigma so that it has a gradient.
    fold = ScaleFold("dynamic", num_scales=3, in_channels=8, patch_len=32,
                     shrinkage=True).eval()
    assert torch.allclose(fold(x), ref, atol=1e-2)
    assert not torch.allclose(fold(x), ref, atol=1e-6)


def test_dynamic_fold_has_no_channel_shaped_parameter():
    """One MLP serves any channel count, which the static modes cannot do."""
    from wavelet_modules import ScaleFold

    sizes = {}
    for C in (4, 16, 64):
        fold = ScaleFold("dynamic", num_scales=4, in_channels=C, patch_len=16,
                         synthesis_kernel=3, shrinkage=True)
        sizes[C] = sum(p.numel() for p in fold.parameters())
        assert fold(torch.randn(2, 4 * C, 64)).shape == (2, C, 64)
    assert len(set(sizes.values())) == 1, sizes
    # ... and the static ones do, which is the trade being made.
    static = {C: sum(p.numel() for p in
                     ScaleFold("learned", 4, C).parameters()) for C in (4, 16, 64)}
    assert len(set(static.values())) == 3, static


def test_one_weight_is_decided_per_token_the_patcher_will_make():
    """The weighting block defaults to the patcher's time patch.

    A mismatch would hand the backbone tokens whose scale mixture changes
    partway through, which is invisible in the shapes and in the loss.
    """
    from model import create_wavelet_classifier

    m = create_wavelet_classifier(in_channels=4, max_level=2, embed_dim=32, depth=1,
                                  num_heads=4, num_classes=3, patch_size=(1, 32),
                                  wavelet_names=["db4"], scale_fold="dynamic")
    assert m.fold.patch_len == 32
    m.fold_scales(m.wavelet_decomp(torch.randn(2, 4, 128)))
    # 128 samples / 32 = 4 blocks, matching the 4 time patches per frequency row.
    assert m.fold.alpha_mean.shape == (3,)

    whole = create_wavelet_classifier(in_channels=4, max_level=2, embed_dim=32, depth=1,
                                      num_heads=4, num_classes=3, patch_size=(1, 32),
                                      wavelet_names=["db4"], scale_fold="dynamic",
                                      fold_patch_len=0)
    assert whole.fold.patch_len == 0


def test_softmax_fold_is_convex_and_learned_fold_is_not():
    from wavelet_modules import ScaleFold

    soft = ScaleFold("softmax", num_scales=4, in_channels=6)
    with torch.no_grad():
        soft.scale_weight.normal_(0, 3.0)
        w = soft.scale_weight.softmax(dim=0)
    assert torch.allclose(w.sum(dim=0), torch.ones(6), atol=1e-6)
    assert (w >= 0).all()
    # A constant signal must survive a convex fold untouched; the free version
    # has no such guarantee once its weights move.
    x = torch.ones(2, 24, 32)
    assert torch.allclose(soft(x), torch.ones(2, 6, 32), atol=1e-5)
    free = ScaleFold("learned", num_scales=4, in_channels=6)
    with torch.no_grad():
        free.scale_weight.normal_(0, 3.0)
    assert not torch.allclose(free(x), torch.ones(2, 6, 32), atol=1e-2)


def test_scale_dropout_never_empties_the_mixture():
    """Dropping every band would divide by zero; the guard must be exercised."""
    from wavelet_modules import ScaleFold

    torch.manual_seed(0)
    fold = ScaleFold("dynamic", num_scales=4, in_channels=8, patch_len=16,
                     scale_dropout=0.95).train()
    out = fold(torch.randn(8, 32, 64))
    assert torch.isfinite(out).all()


def test_fold_regulariser_is_zero_at_uniform_and_positive_when_collapsed():
    from wavelet_modules import ScaleFold

    fold = ScaleFold("dynamic", num_scales=4, in_channels=8, patch_len=16).eval()
    x = torch.randn(2, 32, 64)
    fold(x)
    assert fold.reg_loss.abs().item() < 1e-6
    with torch.no_grad():
        fold.scale_logits.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    fold(x)
    assert fold.reg_loss.item() > 1.0
    assert fold.alpha_mean.argmax().item() == 0


def test_legacy_checkpoints_keep_their_fold_weight():
    """The static weight used to live at the model root.

    Loading such a checkpoint under the current code must not leave the fold at
    its uniform init -- that would look like a successful reproduction while
    silently discarding what the run had learned.
    """
    from model import create_wavelet_classifier

    kw = dict(in_channels=4, max_level=2, embed_dim=32, depth=1, num_heads=4,
              num_classes=3, patch_size=(1, 32), wavelet_names=["db4"],
              scale_fold="learned")
    trained = create_wavelet_classifier(**kw)
    with torch.no_grad():
        trained.fold.scale_weight.normal_()
    old_style = dict(trained.state_dict())
    old_style["scale_weight"] = old_style.pop("fold.scale_weight")

    fresh = create_wavelet_classifier(**kw)
    missing, unexpected = fresh.load_state_dict(old_style, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert torch.equal(fresh.fold.scale_weight, trained.fold.scale_weight)


def test_synthesis_filter_is_available_to_every_fold_mode():
    """The filters must be separable from the mixture that follows them.

    `dynamic + synthesis` beating `learned` says nothing about which half did
    the work unless `mean + synthesis` can also be run.
    """
    from wavelet_modules import ScaleFold

    x = torch.randn(2, 12, 64)
    for mode in ("mean", "learned", "softmax", "dynamic"):
        fold = ScaleFold(mode, num_scales=3, in_channels=4, patch_len=16,
                         synthesis_kernel=3).eval()
        assert fold.synth is not None, mode
        assert fold(x).shape == (2, 4, 64)
        # Delta-initialised, so every mode still starts where it did without it.
        plain = ScaleFold(mode, num_scales=3, in_channels=4, patch_len=16).eval()
        assert torch.allclose(fold(x), plain(x), atol=1e-5), mode
    assert ScaleFold("none", 3, 4, synthesis_kernel=3).synth is None


def test_synthesis_filter_actually_filters_once_it_moves():
    """A smoothing kernel must change the fold, or the parameter is decorative."""
    from wavelet_modules import ScaleFold

    torch.manual_seed(0)
    x = torch.randn(2, 12, 64)
    fold = ScaleFold("mean", num_scales=3, in_channels=4, synthesis_kernel=3).eval()
    before = fold(x)
    with torch.no_grad():
        # The de-staircase kernel: what a zero-order-hold upsample needs.
        fold.synth.weight.copy_(torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
                                .expand(3, 1, 3).contiguous())
    after = fold(x)
    assert not torch.allclose(before, after, atol=1e-3)
    # Smoothing must reduce sample-to-sample variation, not merely perturb it.
    rough = lambda t: t.diff(dim=-1).abs().mean()
    assert rough(after) < rough(before)


def test_synthesis_norm_removes_gain_but_keeps_shape():
    """Unit-DC kernels are what separate "reshapes a band" from "rescales it"."""
    from wavelet_modules import ScaleFold

    torch.manual_seed(0)
    x = torch.randn(2, 12, 64)
    # The kernel the DB5 run actually learned on its finest detail band.
    trained = torch.tensor([0.1339, 1.1538, 0.1117])
    free = ScaleFold("mean", 3, 4, synthesis_kernel=3).eval()
    unit = ScaleFold("mean", 3, 4, synthesis_kernel=3, synthesis_norm=True).eval()
    for f in (free, unit):
        with torch.no_grad():
            f.synth.weight.copy_(trained.view(1, 1, 3).expand(3, 1, 3).contiguous())
    a, b = free(x), unit(x)
    # Same shape, different scale: the ratio is the kernel's DC gain.
    assert torch.allclose(a, b * trained.sum(), atol=1e-4)
    assert (a.std() / b.std() - trained.sum()).abs() < 1e-3


def test_share_channels_makes_the_static_folds_channel_independent():
    from wavelet_modules import ScaleFold

    shared = {C: sum(p.numel() for p in
                     ScaleFold("learned", 4, C, share_channels=True).parameters())
              for C in (8, 16, 64)}
    assert set(shared.values()) == {4}, shared
    fold = ScaleFold("learned", 4, 16, share_channels=True)
    assert fold(torch.randn(2, 64, 32)).shape == (2, 16, 32)
    # Still a plain average at init, like every other row of the ladder.
    x = torch.randn(2, 64, 32)
    assert torch.allclose(fold(x), x.view(2, 4, 16, 32).mean(1), atol=1e-6)
