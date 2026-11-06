"""Channel metadata, the encoder, and its two injection sites.

The property every one of these defends is the same: a channel-embedding
variant must differ from the baseline in the channel modules and in nothing
else. If the legacy parameters move, or the gates do not start at zero, or a
channel's code reaches another channel's tokens, the ablation stops measuring
what it claims to.

Deliberately absent: any test that swapping the two input channels leaves the
model unchanged. The wavelet front end has a filter bank per channel and a
cross-channel stage, so it is not channel-permutation equivariant and never
claimed to be. What is tested is the encoder's own pairing and the token
broadcast.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from channel_embedding import (                                    # noqa: E402
    CHANNEL_VOCAB, PAD_ID, UNK_ID, ChannelEncoder, channel_id, spherical_basis)
from model import BERTWaveletTransformer                           # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "sleep_edf_finetune", os.path.join(_HERE, "EEG", "sleep_edf_finetune.py"))
sleep = importlib.util.module_from_spec(_SPEC)
sys.modules["sleep_edf_finetune"] = sleep
_SPEC.loader.exec_module(sleep)

VARIANTS = [("C0", "none", "none"), ("C1", "id", "token"),
            ("C2", "signed", "token"), ("C3", "signed", "fold"),
            ("C4", "signed", "dual"), ("C5", "hybrid", "dual")]

# A stand-in for the standard_1020 positions, so these tests do not need MNE.
_XYZ = torch.tensor([[1e-4, 0.0882, -0.0017],
                     [4e-4, -0.0092, 0.1002],
                     [3e-4, -0.0811, 0.0826],
                     [1e-4, -0.1149, 0.0147]], dtype=torch.float32)


def _meta(reverse: bool = False):
    pos, neg = ([1, 3], [0, 2]) if reverse else ([0, 2], [1, 3])
    return {
        "channel_ids": torch.tensor([channel_id("Fpz-Cz"), channel_id("Pz-Oz")]),
        "electrode_xyz": _XYZ.clone(),
        "positive_electrode_index": torch.tensor(pos),
        "negative_electrode_index": torch.tensor(neg),
        "valid_channel_mask": torch.tensor([True, True]),
    }


def _model(encoding="none", injection="none", seed=42, embed_dim=96, depth=2,
           in_channels=2, **kw):
    torch.manual_seed(seed)
    return BERTWaveletTransformer(
        in_channels=in_channels, max_level=3, wave_kernel_size=16,
        wavelet_names=["sym4", "db6"], use_separate_channel=True,
        wave_init_mode="pad", patch_size=(1, 50), embed_dim=embed_dim,
        depth=depth, num_heads=4, mlp_ratio=4.0, dropout=0.0,
        norm="rmsnorm", ffn="swiglu", qk_norm=False,
        scale_fold=kw.pop("scale_fold", "dynamic"), fold_synthesis=3,
        fold_gamma=0.1, use_pos_embed=True, pos_embed_type="2d",
        channel_encoding=encoding, channel_injection=injection,
        channel_embed_dim=32, task_type="classification", num_classes=5,
        head_config={"hidden_dims": [64], "dropout": 0.0, "pooling": "mean"},
        pooling="mean", **kw)


def _legacy_hash(m):
    skip = set(m.channel_parameter_names())
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        if n not in skip:
            h.update(n.encode())
            h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1-4: the metadata written into the HDF5
# --------------------------------------------------------------------------- #
def test_channel_and_electrode_order_is_fixed_not_incidental():
    """Row 0 is Fpz-Cz and row 1 is Pz-Oz, and the electrodes are Fpz Cz Pz Oz.

    Every other field indexes into these two lists, so an order that came from
    whatever the EDF happened to contain would make the derivation matrix and
    the endpoint indices describe the wrong rows -- with no error anywhere.
    """
    assert sleep.CHANNELS == ["Fpz-Cz", "Pz-Oz"]
    assert sleep.ELECTRODES == ["Fpz", "Cz", "Pz", "Oz"]
    assert sleep.BIPOLAR_ENDPOINTS == [("Fpz", "Cz"), ("Pz", "Oz")]


def test_derivation_matrix_is_exactly_the_two_bipolar_rows():
    """It states what the channels mean. It is never multiplied into the data.

    Sleep-EDF arrives already re-referenced: the subtraction happened in the
    recording hardware. A matrix that drifted from the endpoint lists would
    describe a montage the data does not have.
    """
    np.testing.assert_array_equal(
        sleep.DERIVATION_MATRIX,
        np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]], dtype=np.float32))
    idx = {e: i for i, e in enumerate(sleep.ELECTRODES)}
    for c, (a, b) in enumerate(sleep.BIPOLAR_ENDPOINTS):
        row = sleep.DERIVATION_MATRIX[c]
        assert row[idx[a]] == 1.0 and row[idx[b]] == -1.0
        assert np.abs(row).sum() == 2.0, "a bipolar row touches exactly two electrodes"


def test_metadata_does_not_disturb_the_arrays_finetune_reads(tmp_path):
    """data, label and subject must be byte-identical with and without metadata."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("mne")
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    for s in (0, 2, 4):
        np.savez_compressed(
            cache / f"sub{s:02d}.npz",
            data=rng.standard_normal((20, 2, 3000)).astype("float32"),
            label=rng.integers(0, 5, 20))

    plain, rich = tmp_path / "plain", tmp_path / "rich"
    for out, bundle in ((plain, None), (rich, sleep.build_channel_metadata())):
        out.mkdir()
        sleep.write_split("train", [0, 2, 4], str(cache), str(out), bundle)

    with h5py.File(plain / "train.h5") as a, h5py.File(rich / "train.h5") as b:
        for key in ("data", "label", "subject"):
            assert a[key].shape == b[key].shape
            np.testing.assert_array_equal(a[key][:], b[key][:])
        assert "channel_ids" not in a and "channel_ids" in b
        np.testing.assert_array_equal(b["derivation_matrix"][:],
                                      sleep.DERIVATION_MATRIX)
        assert [c.decode() for c in b["channel_names"][:]] == sleep.CHANNELS
        assert [c.decode() for c in b["electrode_names"][:]] == sleep.ELECTRODES
        assert b["positive_electrode_index"][:].tolist() == [0, 2]
        assert b["negative_electrode_index"][:].tolist() == [1, 3]


# --------------------------------------------------------------------------- #
# 5-7: the baseline must not move
# --------------------------------------------------------------------------- #
def test_channel_encoding_none_reproduces_the_model_without_the_feature():
    """Built with the defaults, the model must not know this feature exists."""
    a, b = _model(), _model()
    a.eval(); b.eval()
    assert a.channel_encoder is None
    assert a.channel_parameter_names() == ()
    torch.manual_seed(0)
    x = torch.randn(3, 2, 3000)
    with torch.no_grad():
        assert torch.equal(a(x, task="classify"), b(x, task="classify"))


@pytest.mark.parametrize("tag,enc,inj", VARIANTS)
def test_zero_gates_leave_the_backbone_output_untouched(tag, enc, inj):
    """Every variant, at step 0, must give the baseline's logits exactly.

    Not approximately: the gates are exactly zero and tanh(0) is exactly zero,
    so the injected term is exactly zero and the arithmetic is the baseline's.
    A tolerance here would hide a gate that had been initialised to 1e-8.
    """
    base = _model(); base.eval()
    var = _model(enc, inj); var.eval()
    torch.manual_seed(0)
    x = torch.randn(3, 2, 3000)
    meta = None if enc == "none" else _meta()
    with torch.no_grad():
        y0 = base(x, task="classify")
        y1 = var(x, task="classify", channel_meta=meta)
    assert torch.max((y0 - y1).abs()).item() < 1e-6
    if enc != "none":
        f, t = var.channel_gate_values()
        assert (f is None or f == 0.0) and (t is None or t == 0.0)


@pytest.mark.parametrize("tag,enc,inj", VARIANTS)
def test_legacy_parameters_are_identical_across_every_variant(tag, enc, inj):
    """One seed, six variants, one backbone.

    Constructing a module draws from the global RNG, so channel modules built
    before the legacy ones would shift every legacy draw and the variants would
    no longer share an initialisation. They are built last for this reason, and
    this is the test that keeps them there.
    """
    assert _legacy_hash(_model(enc, inj)) == _legacy_hash(_model())


# --------------------------------------------------------------------------- #
# 8-9: the encoder and the broadcast
# --------------------------------------------------------------------------- #
def test_a_channel_code_reaches_only_its_own_time_patches():
    """Sentinel: give channel 0 a code and channel 1 none, and see where it lands.

    PatchEmbed emits the sequence channel-major and time-minor, so token
    c*P + p belongs to channel c. If that assumption were wrong -- if the
    flatten walked time first -- a code meant for one derivation would smear
    across both, and nothing else in the model would complain.
    """
    m = _model("id", "token"); m.eval()
    B, C, P, D = 2, 2, 60, 96
    tokens = torch.zeros(B, C * P, D)
    code = torch.zeros(C, m.channel_embed_dim)
    code[0] = 1.0                                   # only channel 0 is marked
    with torch.no_grad():
        m.channel_token_gate.fill_(1.0)             # open the gate
        out = m._inject_channel_tokens(tokens, code, n_rows=C, n_samples=P * 50)
    out = out.reshape(B, C, P, D)
    moved_c0 = out[:, 0].abs().sum().item()
    moved_c1 = out[:, 1].abs().sum().item()
    assert moved_c0 > 0.0, "the marked channel received nothing"
    # Channel 1's code is the zero vector, so only the projection's bias could
    # reach it -- and the bias is zero-initialised.
    assert moved_c1 == 0.0, "an unmarked channel was touched"
    # And within channel 0, every one of its P patches got the same vector.
    per_patch = out[0, 0]
    assert torch.allclose(per_patch, per_patch[0].expand_as(per_patch))


def test_signed_encoding_keeps_midpoint_and_negates_direction():
    """A-B and B-A share a midpoint and have opposite directions.

    This is the entire difference between the signed encoding and a positional
    one. A midpoint-only code cannot tell the two apart, which for a bipolar
    montage means it cannot tell a derivation from its own negation.
    """
    phi = spherical_basis(_XYZ)
    fwd, rev = _meta(), _meta(reverse=True)
    pa, pb = phi[fwd["positive_electrode_index"]], phi[fwd["negative_electrode_index"]]
    qa, qb = phi[rev["positive_electrode_index"]], phi[rev["negative_electrode_index"]]
    assert torch.equal(0.5 * (pa + pb), 0.5 * (qa + qb))
    assert torch.equal(pa - pb, -(qa - qb))
    # phi itself is a direction on the sphere, so scaling the coordinates does
    # not change it -- the code describes a montage, not a head radius.
    assert torch.allclose(spherical_basis(_XYZ * 7.3), phi, atol=1e-6)


def test_the_vocabulary_reserves_pad_and_unk_and_is_stable():
    assert CHANNEL_VOCAB[PAD_ID] == "<pad>" and CHANNEL_VOCAB[UNK_ID] == "<unk>"
    assert channel_id("Fpz-Cz") == 2 and channel_id("Pz-Oz") == 3
    assert channel_id("a channel that does not exist") == UNK_ID


# --------------------------------------------------------------------------- #
# 10-11: the fold
# --------------------------------------------------------------------------- #
def test_fold_alpha_is_unchanged_while_the_fold_gate_is_zero():
    base, var = _model(), _model("signed", "fold")
    base.eval(); var.eval()
    torch.manual_seed(0)
    x = torch.randn(3, 2, 3000)
    with torch.no_grad():
        base(x, task="classify")
        var(x, task="classify", channel_meta=_meta())
    assert torch.allclose(base.scale_fold_alpha(), var.scale_fold_alpha(), atol=1e-7)


def test_an_open_fold_gate_makes_different_channels_want_different_scales():
    """With the gate open, the prior must actually reach the mixing weights.

    Two derivations with different geometry should not receive identical scale
    weights once the prior is switched on; if they do, the bias is being
    computed and then dropped.
    """
    m = _model("signed", "fold"); m.eval()
    with torch.no_grad():
        m.channel_fold_gate.fill_(2.0)
        torch.nn.init.normal_(m.channel_to_scale.weight, std=1.0)
    torch.manual_seed(0)
    x = torch.randn(3, 2, 3000)
    with torch.no_grad():
        m(x, task="classify", channel_meta=_meta())
    per_channel = m.scale_fold_per_channel()                # [C, S]
    assert per_channel is not None and per_channel.shape[0] == 2
    assert not torch.allclose(per_channel[0], per_channel[1], atol=1e-4), \
        "both channels received the same scale weights with the prior open"


# --------------------------------------------------------------------------- #
# 12: gradients
# --------------------------------------------------------------------------- #
def test_gates_get_gradient_first_and_the_projections_follow():
    """Zero gate, non-zero projection -- and why it has to be that way round.

    d(loss)/d(gate) carries the projection's output, so it is non-zero at step
    0. d(loss)/d(projection) carries tanh(gate), so it is exactly zero until
    the gate moves. Initialising both at zero would put a zero on both sides of
    the product and the branch would never receive gradient at all.
    """
    m = _model("signed", "dual")
    x, y = torch.randn(2, 2, 3000), torch.randint(0, 5, (2,))
    opt = torch.optim.SGD(m.parameters(), lr=1.0)

    def grads():
        return {n: (0.0 if p.grad is None else float(p.grad.abs().sum()))
                for n, p in m.named_parameters()}

    opt.zero_grad()
    torch.nn.functional.cross_entropy(
        m(x, task="classify", channel_meta=_meta()), y).backward()
    g1 = grads()
    assert g1["channel_fold_gate"] > 0.0 and g1["channel_token_gate"] > 0.0
    assert g1["channel_to_token.weight"] == 0.0
    assert g1["channel_encoder.mid_proj.weight"] == 0.0
    opt.step()

    opt.zero_grad()
    torch.nn.functional.cross_entropy(
        m(x, task="classify", channel_meta=_meta()), y).backward()
    g2 = grads()
    assert g2["channel_to_token.weight"] > 0.0
    assert g2["channel_encoder.mid_proj.weight"] > 0.0


# --------------------------------------------------------------------------- #
# 13-14: compatibility and refusals
# --------------------------------------------------------------------------- #
def test_a_code_without_metadata_is_refused_rather_than_run_without_one():
    m = _model("signed", "token"); m.eval()
    with pytest.raises(ValueError, match="needs channel metadata"):
        m(torch.randn(1, 2, 3000), task="classify")


def test_metadata_without_a_code_is_refused_rather_than_ignored():
    m = _model(); m.eval()
    with pytest.raises(ValueError, match="silently ignored"):
        m(torch.randn(1, 2, 3000), task="classify", channel_meta=_meta())


def test_mismatched_encoding_and_injection_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        _model("signed", "none")
    with pytest.raises(ValueError, match="disagree"):
        _model("none", "token")


def test_fold_injection_needs_the_dynamic_fold():
    with pytest.raises(ValueError, match="dynamic"):
        _model("signed", "fold", scale_fold="mean")


def test_an_old_checkpoint_may_lack_only_the_channel_keys(tmp_path):
    """A pre-feature checkpoint loads; a genuinely wrong one does not.

    strict=False is needed -- a feature extractor has no task head -- but on its
    own it swallows an architecture mismatch without a word, leaving tensors at
    their random init while the run reports a fine-tune. The channel keys are
    the ones a checkpoint from before this feature is *expected* to lack.
    """
    import finetune

    old_model = _model()                       # no channel modules at all
    ckpt = tmp_path / "old.pth"
    torch.save({"model_state_dict": old_model.state_dict()}, ckpt)

    new_model = _model("signed", "dual")
    finetune.load_pretrained_feature_extractor(new_model, str(ckpt), rank=1)
    # The legacy half came across; the channel half kept its initialisation.
    assert _legacy_hash(new_model) == _legacy_hash(old_model)

    # Now break it: drop a legacy tensor and the load must refuse.
    sd = old_model.state_dict()
    victim = next(k for k in sd if k.startswith("encoder."))
    del sd[victim]
    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": sd}, bad)
    with pytest.raises(SystemExit, match="does not match"):
        finetune.load_pretrained_feature_extractor(_model("signed", "dual"),
                                                   str(bad), rank=1)


# --------------------------------------------------------------------------- #
# 15-16: every variant runs, in both precisions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,enc,inj", VARIANTS)
def test_every_variant_completes_a_forward_and_a_backward(tag, enc, inj):
    m = _model(enc, inj)
    x, y = torch.randn(2, 2, 3000), torch.randint(0, 5, (2,))
    meta = None if enc == "none" else _meta()
    logits = m(x, task="classify", channel_meta=meta)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, y)
    reg = m.scale_fold_reg()
    if reg is not None:
        loss = loss + 1e-3 * reg
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "nothing received gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("tag,enc,inj", VARIANTS)
def test_autocast_produces_no_nan(tag, enc, inj):
    """bf16 on CPU, since that is what is available here.

    The point is not the exact dtype: it is that the new arithmetic -- a tanh
    gate, a projection, a bias added to softmax logits -- survives reduced
    precision. A NaN here would only appear on the cluster otherwise, after
    the job had been allocated.
    """
    m = _model(enc, inj); m.eval()
    x = torch.randn(2, 2, 3000)
    meta = None if enc == "none" else _meta()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = m(x, task="classify", channel_meta=meta)
    assert torch.isfinite(logits.float()).all()


def test_the_scale_fold_refuses_a_channel_prior_it_cannot_use():
    """ScaleFold itself, not just the model, guards the dynamic-only path."""
    from wavelet_modules import ScaleFold
    fold = ScaleFold(mode="mean", num_scales=4, in_channels=2, patch_len=50)
    spec = torch.randn(2, 8, 300)
    with pytest.raises(ValueError, match="dynamic"):
        fold(spec, channel_scale_bias=torch.zeros(2, 4))
    fold_none = ScaleFold(mode="none", num_scales=4, in_channels=2, patch_len=50)
    with pytest.raises(ValueError, match="dynamic"):
        fold_none(spec, channel_scale_bias=torch.zeros(2, 4))
