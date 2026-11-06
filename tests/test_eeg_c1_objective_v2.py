"""
The EEG C1 objective: detached dual reconstruction with pre-frontend masking.

Three changes to the objective, and one test group each:

  * the reconstruction targets are stop-gradient
  * a second head predicts the preprocessed EEG, not only the folded wavelet
  * the masked patches are zeroed in the SIGNAL, before the wavelet frontend

The leakage test is the important one. Everything else here checks that the
pieces are wired the way they are described; that one checks the property the
whole rearrangement exists to get.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import PAD_ID, channel_ids_for              # noqa: E402
from physiowave.eeg_c1.model import (MultiRouteEEGPretrainer,      # noqa: E402
                                     apply_patch_mask_to_signal,
                                     masked_reconstruction_loss)
from physiowave.eeg_c1.routes import ROUTES                        # noqa: E402


def build(mask_before_frontend=True, **kw):
    kw.setdefault("embed_dim", 64)
    kw.setdefault("depth", 1)
    kw.setdefault("num_heads", 4)
    kw.setdefault("channel_embed_dim", 16)
    kw.setdefault("dropout", 0.0)
    torch.manual_seed(0)
    m = MultiRouteEEGPretrainer(mask_before_frontend=mask_before_frontend, **kw)
    return m.eval()


def meta_for(route_id, n_valid=None):
    r = ROUTES[route_id]
    ids, _ = channel_ids_for(r.slots)
    valid = torch.ones(r.n_channels, dtype=torch.bool)
    if n_valid is not None:
        valid[n_valid:] = False
        ids = [i if valid[k] else PAD_ID for k, i in enumerate(ids)]
    return {"channel_ids": torch.tensor(ids, dtype=torch.long),
            "valid_channel_mask": valid}


def one_token_mask(route_id, batch, channel, patch):
    """A mask selecting exactly ``(channel, patch)``, in channel-major order."""
    r = ROUTES[route_id]
    m = torch.zeros(batch, r.n_channels * r.patches_per_channel, dtype=torch.bool)
    m[:, channel * r.patches_per_channel + patch] = True
    return m


# --- 1. the leakage property ------------------------------------------------ #
def test_masked_raw_content_cannot_reach_the_online_representation():
    """Two windows differing ONLY inside a masked patch must encode identically.

    The wavelet frontend has temporal convolution and cross-scale attention, so
    when the mask is applied to TOKENS after it, the masked patch has already
    spread into its neighbours' features and the encoder can read what it is
    supposed to be predicting. Zeroing those samples before the frontend closes
    that path, and the way to prove it is closed is that the content becomes
    unobservable: change it by any amount, and nothing downstream moves.
    """
    route_id, channel, patch = "E19_256", 5, 3
    route = ROUTES[route_id]
    pt = route.patch_t
    meta = meta_for(route_id)
    mask = one_token_mask(route_id, 2, channel, patch)

    torch.manual_seed(11)
    x1 = torch.randn(2, route.n_channels, route.window_samples)
    x2 = x1.clone()
    x2[:, channel, patch * pt:(patch + 1) * pt] += 100.0     # only inside the mask
    assert not torch.equal(x1, x2)

    m = build(mask_before_frontend=True)
    with torch.no_grad():
        a = m(x1, route_id, channel_meta=meta, mask_override=mask)
        b = m(x2, route_id, channel_meta=meta, mask_override=mask)

    for key in ("online_spec", "pred_spec", "pred_raw"):
        assert torch.allclose(a[key], b[key], atol=1e-5), \
            f"{key} moved with content that was supposed to be masked out"

    # The TARGET must still differ -- it is built from the clean signal, and a
    # target that did not contain the masked content would be asking the model
    # to predict something nobody knows.
    assert not torch.allclose(a["target_raw"], b["target_raw"])
    assert not torch.allclose(a["target_spec"], b["target_spec"])


def test_the_leakage_test_is_sensitive_to_the_mechanism():
    """The same perturbation DOES move the old ordering -- so the test can fail.

    Without this, the test above would pass just as well against a model that
    ignored its input entirely.
    """
    route_id, channel, patch = "E19_256", 5, 3
    route = ROUTES[route_id]
    pt = route.patch_t
    meta = meta_for(route_id)
    mask = one_token_mask(route_id, 2, channel, patch)

    torch.manual_seed(11)
    x1 = torch.randn(2, route.n_channels, route.window_samples)
    x2 = x1.clone()
    x2[:, channel, patch * pt:(patch + 1) * pt] += 100.0

    m = build(mask_before_frontend=False)
    with torch.no_grad():
        a = m(x1, route_id, channel_meta=meta, mask_override=mask)
        b = m(x2, route_id, channel_meta=meta, mask_override=mask)

    assert not torch.allclose(a["online_spec"], b["online_spec"]), \
        "masking only the tokens should leave the frontend output dependent " \
        "on the masked samples"
    assert not torch.allclose(a["pred_spec"], b["pred_spec"], atol=1e-5)


# --- 2. the targets are stop-gradient --------------------------------------- #
def test_targets_are_detached_and_the_model_still_learns():
    route_id = "E32_512"
    route = ROUTES[route_id]
    meta = meta_for(route_id)
    m = build().train()
    # The channel gate starts at zero, and delta = tanh(gate) * proj(code) is
    # then zero -- so the projection and the encoder legitimately have no
    # gradient at step 0, by design. Open the gate so this test is about
    # whether the path is connected rather than about its initialisation.
    with torch.no_grad():
        m.channel_token_gate.fill_(0.5)
    out = m(torch.randn(2, route.n_channels, route.window_samples),
            route_id, channel_meta=meta, mask_ratio=0.5)

    assert out["target_spec"].requires_grad is False
    assert out["target_raw"].requires_grad is False
    assert out["pred_spec"].requires_grad is True
    assert out["pred_raw"].requires_grad is True

    loss, _ = masked_reconstruction_loss(out)
    loss.backward()

    def has_grad(mod):
        gs = [p.grad for p in mod.parameters() if p.requires_grad]
        return bool(gs) and any(g is not None and float(g.abs().sum()) > 0
                                for g in gs)

    assert has_grad(m.wavelet_frontends[route_id].decomp), "wavelet filters"
    assert has_grad(m.wavelet_frontends[route_id].fold), "ScaleFold"
    assert has_grad(m.patch_embed_by_rate[route.rate_key]), "PatchEmbed"
    assert has_grad(m.shared_transformer), "shared Transformer"
    assert has_grad(m.reconstruction_heads[route.rate_key]), "spec head"
    assert has_grad(m.raw_reconstruction_heads[route.rate_key]), "raw head"
    assert has_grad(m.channel_encoder), "channel encoder"
    assert has_grad(m.channel_to_token), "channel projection"
    assert m.channel_token_gate.grad is not None and \
        float(m.channel_token_gate.grad.abs()) > 0, "channel gate"


# --- 3. the raw auxiliary term ---------------------------------------------- #
def test_raw_loss_sees_masked_tokens_only():
    route_id = "E19_256"
    route = ROUTES[route_id]
    n_tok = route.n_channels * route.patches_per_channel
    mask = torch.zeros(1, n_tok, dtype=torch.bool)
    mask[0, 4] = True

    def make():
        return {
            "pred_spec": torch.zeros(1, n_tok, route.patch_t),
            "target_spec": torch.zeros(1, n_tok, route.patch_t),
            "pred_raw": torch.zeros(1, n_tok, route.patch_t),
            "target_raw": torch.zeros(1, n_tok, route.patch_t),
            "mask": mask, "valid_tokens": None, "fold_reg": None,
        }

    base_total, base = masked_reconstruction_loss(make())
    assert base["loss_masked_raw_smoothl1"] == pytest.approx(0.0)

    visible = make()
    visible["pred_raw"][0, 7] += 5.0            # not masked
    _, vis = masked_reconstruction_loss(visible)
    assert vis["loss_masked_raw_smoothl1"] == pytest.approx(0.0)

    hidden = make()
    hidden["pred_raw"][0, 4] += 5.0             # masked
    _, hid = masked_reconstruction_loss(hidden)
    assert hid["loss_masked_raw_smoothl1"] > 0.1
    assert torch.isfinite(torch.tensor(hid["loss_masked_raw_smoothl1"]))

    # raw_weight=0 must remove it from the total entirely.
    off_total, off = masked_reconstruction_loss(hidden, raw_weight=0.0)
    assert off["loss_masked_raw_smoothl1"] > 0.1     # still reported
    assert float(off_total) == pytest.approx(0.0, abs=1e-6)   # not in the total

    # SmoothL1, not MSE: a large error is penalised linearly.
    big = make()
    big["pred_raw"][0, 4] += 100.0
    _, b = masked_reconstruction_loss(big)
    mse_would_be = 100.0 ** 2
    assert b["loss_masked_raw_smoothl1"] < mse_would_be / 10


def test_padded_channels_never_enter_either_loss():
    route_id = "E32_512"
    route = ROUTES[route_id]
    meta = meta_for(route_id, n_valid=26)          # TDBRAIN's shape
    m = build()
    out = m(torch.randn(2, route.n_channels, route.window_samples),
            route_id, channel_meta=meta, mask_ratio=0.5)

    valid = out["valid_tokens"]
    assert valid is not None
    assert not bool((out["mask"] & ~valid).any()), "a padded slot was masked"

    # And an explicit attempt to mask one is refused rather than silently
    # dropped, which would quietly lower the mask ratio instead.
    bad = torch.zeros_like(out["mask"])
    bad[:, route.patches_per_channel * 30] = True          # channel 30 is padded
    with pytest.raises(ValueError, match="padded"):
        m(torch.randn(2, route.n_channels, route.window_samples),
          route_id, channel_meta=meta, mask_override=bad)


# --- 4. the spec term ------------------------------------------------------- #
def test_spec_loss_is_zero_on_a_perfect_masked_prediction():
    route = ROUTES["E19_256"]
    n_tok = route.n_channels * route.patches_per_channel
    mask = torch.zeros(1, n_tok, dtype=torch.bool)
    mask[0, 2] = True
    tgt = torch.randn(1, n_tok, route.patch_t)

    out = {"pred_spec": tgt.clone(), "target_spec": tgt,
           "pred_raw": torch.zeros_like(tgt), "target_raw": torch.zeros_like(tgt),
           "mask": mask, "valid_tokens": None, "fold_reg": None}
    _, m0 = masked_reconstruction_loss(out)
    assert m0["loss_masked_spec_mse"] == pytest.approx(0.0, abs=1e-12)

    out["pred_spec"] = tgt.clone()
    out["pred_spec"][0, 9] += 3.0                  # visible token
    _, m1 = masked_reconstruction_loss(out)
    assert m1["loss_masked_spec_mse"] == pytest.approx(0.0, abs=1e-12)

    out["pred_spec"] = tgt.clone()
    out["pred_spec"][0, 2] += 3.0                  # masked token
    _, m2 = masked_reconstruction_loss(out)
    assert m2["loss_masked_spec_mse"] > 1.0

    # The historical metric name still resolves to the spec term.
    assert m2["loss_masked_mse"] == m2["loss_masked_spec_mse"]


def test_zero_mask_gives_a_differentiable_zero():
    route = ROUTES["E19_256"]
    n_tok = route.n_channels * route.patches_per_channel
    pred = torch.randn(1, n_tok, route.patch_t, requires_grad=True)
    out = {"pred_spec": pred, "target_spec": torch.randn_like(pred),
           "pred_raw": pred, "target_raw": torch.randn_like(pred),
           "mask": torch.zeros(1, n_tok, dtype=torch.bool),
           "valid_tokens": None, "fold_reg": None}
    loss, mets = masked_reconstruction_loss(out)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    loss.backward()                                 # must not raise
    assert all(v == 0.0 for v in mets.values())


# --- 5. shapes, every route ------------------------------------------------- #
@pytest.mark.parametrize("route_id", ["E19_256", "E32_512", "E64_256",
                                      "E128_512"])
def test_shapes_on_every_route(route_id):
    route = ROUTES[route_id]
    C, P, pt = route.n_channels, route.patches_per_channel, route.patch_t
    meta = meta_for(route_id)
    m = build()
    out = m(torch.randn(2, C, route.window_samples), route_id,
            channel_meta=meta, mask_ratio=0.5)

    for key in ("pred_spec", "target_spec", "pred_raw", "target_raw"):
        assert out[key].shape == (2, C * P, pt), f"{key} on {route_id}"
    assert out["mask"].shape == (2, C * P)
    assert out["valid_tokens"].shape == (2, C * P)
    loss, mets = masked_reconstruction_loss(out)
    assert torch.isfinite(loss)
    assert 0.0 < mets["actual_mask_ratio"] < 1.0


# --- 6. the target comes from the CLEAN pass -------------------------------- #
def test_target_spec_is_the_clean_frontend_not_the_corrupted_one():
    """A regression guard: the target must never become patchify(online_spec).

    That substitution type-checks, runs, and trains -- against a target built
    from the same corrupted input the prediction saw.
    """
    route_id = "E19_256"
    route = ROUTES[route_id]
    meta = meta_for(route_id)
    m = build(mask_before_frontend=True)
    x = torch.randn(2, route.n_channels, route.window_samples)
    mask = one_token_mask(route_id, 2, 4, 2)

    with torch.no_grad():
        out = m(x, route_id, channel_meta=meta, mask_override=mask)
        clean_patches = m.patchify(out["clean_spec"], route.patch_t)
        online_patches = m.patchify(out["online_spec"], route.patch_t)

    # Normalised, so the comparison is against the normalised clean patches --
    # what must never happen is the ONLINE ones, normalised or not.
    assert torch.allclose(out["target_spec"],
                          m.normalize_patches(clean_patches), atol=1e-5)
    assert not torch.allclose(out["target_spec"],
                              m.normalize_patches(online_patches), atol=1e-4)
    # And the corruption actually reached the frontend.
    assert float(out["clean_online_spec_delta"]) > 0

    # The same guard with normalisation off, so this test still covers the
    # substitution it was written for on both settings.
    raw = build(mask_before_frontend=True, normalize_spec_target=False)
    with torch.no_grad():
        out = raw(x, route_id, channel_meta=meta, mask_override=mask)
    assert torch.allclose(out["target_spec"],
                          raw.patchify(out["clean_spec"], route.patch_t),
                          atol=1e-6)
    assert not torch.allclose(out["target_spec"],
                              raw.patchify(out["online_spec"], route.patch_t),
                              atol=1e-4)


def test_target_raw_is_the_uncorrupted_signal():
    route_id = "E19_256"
    route = ROUTES[route_id]
    meta = meta_for(route_id)
    m = build(mask_before_frontend=True)
    x = torch.randn(2, route.n_channels, route.window_samples)
    mask = one_token_mask(route_id, 2, 4, 2)
    with torch.no_grad():
        out = m(x, route_id, channel_meta=meta, mask_override=mask)
    assert torch.allclose(out["target_raw"], m.patchify(x, route.patch_t),
                          atol=1e-6)
    assert float(out["target_raw"][:, 4 * route.patches_per_channel + 2]
                 .abs().sum()) > 0, "the masked patch's target was zeroed too"


# --- 7. the signal-level mask helper ---------------------------------------- #
def test_patch_mask_is_per_channel_not_per_timestep():
    """Fp1's patch 4 masked and Fp2's patch 4 visible must mean exactly that."""
    B, C, P, pt = 2, 4, 5, 8
    x = torch.arange(B * C * P * pt, dtype=torch.float32).reshape(B, C, P * pt)
    mask = torch.zeros(B, C * P, dtype=torch.bool)
    mask[:, 1 * P + 3] = True                    # channel 1, patch 3

    y = apply_patch_mask_to_signal(x, mask, pt)
    assert y.shape == x.shape
    assert float(y[:, 1, 3 * pt:(3 + 1) * pt].abs().sum()) == 0.0
    for c in range(C):
        for p in range(P):
            if (c, p) == (1, 3):
                continue
            assert torch.equal(y[:, c, p * pt:(p + 1) * pt],
                               x[:, c, p * pt:(p + 1) * pt]), \
                f"channel {c} patch {p} was changed"

    with pytest.raises(ValueError, match=r"\[B, C\*P\]"):
        apply_patch_mask_to_signal(x, torch.zeros(B, C * P + 1, dtype=torch.bool), pt)


def test_forward_order_is_mask_then_frontend():
    """The ordering itself, read off the tensors rather than off the source.

    If the implementation regressed to running the frontend on the clean signal
    and only replacing tokens, online_spec would equal clean_spec.
    """
    route_id = "E19_256"
    route = ROUTES[route_id]
    meta = meta_for(route_id)
    x = torch.randn(2, route.n_channels, route.window_samples)
    mask = one_token_mask(route_id, 2, 6, 1)

    with torch.no_grad():
        before = build(mask_before_frontend=True)(
            x, route_id, channel_meta=meta, mask_override=mask)
        after = build(mask_before_frontend=False)(
            x, route_id, channel_meta=meta, mask_override=mask)

    assert not torch.allclose(before["clean_spec"], before["online_spec"])
    assert torch.equal(after["clean_spec"], after["online_spec"])
    assert before["mask_before_frontend"] is True
    assert after["mask_before_frontend"] is False


# --- 8. the raw decoder is pretraining-only --------------------------------- #
def test_raw_head_is_reported_as_pretraining_only():
    m = build()
    rep = m.parameter_report()
    assert any(k.startswith("raw_reconstruction_head.") for k in rep)
    assert rep["downstream_encoder"] + rep["pretraining_only"] == rep["total"]
    raw_params = sum(v for k, v in rep.items()
                     if k.startswith("raw_reconstruction_head."))
    assert raw_params > 0
    # Whatever the raw heads cost, it is not part of what fine-tuning carries.
    assert rep["downstream_encoder"] < rep["total"] - raw_params + 1


def test_export_carries_neither_decoder(tmp_path):
    """Fine-tuning gets the representation path and no reconstruction head."""
    import subprocess

    m = build()
    ck = tmp_path / "ck.pth"
    torch.save({"model": m.state_dict(), "config": {"model": {}},
                "epoch": 0, "global_step": 0}, ck)
    out = tmp_path / "enc.pth"
    r = subprocess.run(
        [sys.executable, "scripts/export_eeg_pretrained_encoder.py",
         "--checkpoint", str(ck), "--route", "E32_512", "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]

    keys = torch.load(out, map_location="cpu", weights_only=False)["model"]
    assert not [k for k in keys if "reconstruction_head" in k], \
        "a reconstruction head reached the exported encoder"
    assert any(k.startswith("shared_transformer.") for k in keys)
    assert any(k.startswith("wavelet_frontend.") for k in keys)


def test_an_old_checkpoint_is_not_a_resume(tmp_path):
    """A single-decoder checkpoint must not silently continue as this run."""
    m = build()
    old = {k: v for k, v in m.state_dict().items()
           if not k.startswith("raw_reconstruction_heads.")}
    with pytest.raises(RuntimeError) as exc:
        m.load_state_dict(old)
    assert "raw_reconstruction_heads" in str(exc.value)
    # And it loads cleanly once those keys are there.
    m.load_state_dict(m.state_dict())


# --- 9. the data the figures are made of ------------------------------------ #
def test_error_histogram_separates_masked_from_visible():
    """Counts add exactly; averaged per-batch quantiles would not."""
    from physiowave.eeg_c1.train import ERROR_BIN_EDGES, ErrorHistogram

    route = ROUTES["E19_256"]
    n_tok = route.n_channels * route.patches_per_channel
    mask = torch.zeros(1, n_tok, dtype=torch.bool)
    mask[0, :10] = True

    zeros = torch.zeros(1, n_tok, route.patch_t)
    pred = zeros.clone()
    pred[0, :10] = 1.0            # masked tokens are wrong by exactly 1
    out = {"pred_spec": pred, "target_spec": zeros,
           "pred_raw": zeros.clone(), "target_raw": zeros,
           "mask": mask, "valid_tokens": None}

    h = ErrorHistogram()
    h.add(out)
    p = h.payload()
    assert p["mean_abs_error"]["spec_masked"] == pytest.approx(1.0)
    assert p["mean_abs_error"]["spec_visible"] == pytest.approx(0.0)
    assert p["n"]["spec_masked"] == 10 * route.patch_t
    assert p["n"]["spec_visible"] == (n_tok - 10) * route.patch_t
    assert len(p["edges"]) == len(ERROR_BIN_EDGES)
    assert sum(p["counts"]["spec_masked"]) == p["n"]["spec_masked"]

    # Two batches: counts and the mean must accumulate, not overwrite.
    h.add(out)
    p2 = h.payload()
    assert p2["n"]["spec_masked"] == 2 * p["n"]["spec_masked"]
    assert p2["mean_abs_error"]["spec_masked"] == pytest.approx(1.0)


def test_gradient_norms_are_reported_per_branch():
    """A global norm cannot say which branch produced it."""
    from physiowave.eeg_c1.train import module_grad_norms

    route_id = "E19_256"
    route = ROUTES[route_id]
    m = build().train()
    with torch.no_grad():
        m.channel_token_gate.fill_(0.5)
    out = m(torch.randn(2, route.n_channels, route.window_samples), route_id,
            channel_meta=meta_for(route_id), mask_ratio=0.5)
    loss, _ = masked_reconstruction_loss(out)
    loss.backward()

    g = module_grad_norms(m)
    # The route and the rate are kept: E19's frontend and E128's are trained on
    # different steps, and one number for both says nothing about either.
    assert f"gradnorm/wavelet_frontends.{route_id}" in g
    assert f"gradnorm/reconstruction_heads.{route.rate_key}" in g
    assert f"gradnorm/raw_reconstruction_heads.{route.rate_key}" in g
    assert "gradnorm/shared_transformer" in g
    assert all(v >= 0 for v in g.values())
    assert g["gradnorm/shared_transformer"] > 0
    assert g[f"gradnorm/raw_reconstruction_heads.{route.rate_key}"] > 0
    # A route that took no step in this batch must not appear with a zero.
    assert "gradnorm/wavelet_frontends.E128_512" not in g


# --------------------------------------------------------------------------- #
# The target moves, and the loss must not follow it
# --------------------------------------------------------------------------- #

def test_spec_target_is_normalised_per_patch():
    from physiowave.eeg_c1.model import MultiRouteEEGPretrainer as M

    p = torch.randn(3, 7, 16) * 5.0 + 2.0
    n = M.normalize_patches(p)
    assert torch.allclose(n.mean(-1), torch.zeros(3, 7), atol=1e-5)
    assert torch.allclose(n.var(-1, unbiased=False), torch.ones(3, 7), atol=1e-3)
    # Per patch, not per tensor: two patches at wildly different scales both
    # come out unit.
    q = p.clone()
    q[0, 0] *= 1000.0
    assert torch.allclose(M.normalize_patches(q)[0, 0], n[0, 0], atol=1e-3)


def test_a_constant_patch_does_not_divide_by_zero():
    from physiowave.eeg_c1.model import MultiRouteEEGPretrainer as M
    flat = torch.full((2, 3, 8), 4.0)
    out = M.normalize_patches(flat)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-2)


def test_the_target_no_longer_grows_with_the_frontend():
    """The failure this exists to stop.

    Seven epochs in, the target's standard deviation had grown 67% while the
    correlation between prediction and target had not moved. The MSE nearly
    tripled and reported a model getting worse that was tracking exactly as
    well as before.
    """
    route_id = "E19_256"
    r = ROUTES[route_id]
    meta = meta_for(route_id)
    x = torch.randn(2, r.n_channels, r.window_samples)
    mask = one_token_mask(route_id, 2, 4, 2)

    spread = {}
    for normalise in (True, False):
        m = build(normalize_spec_target=normalise)
        stds = []
        for gain in (1.0, 4.0):
            with torch.no_grad():
                # Growing the fold's synthesis weights IS the drift: the same
                # frontend produces both the target and the online view, so a
                # larger frontend means a larger target.
                mm = build(normalize_spec_target=normalise)
                for prm in mm.wavelet_frontends[route_id].fold.parameters():
                    prm.mul_(gain)
                out = mm(x, route_id, channel_meta=meta, mask_override=mask)
                stds.append(float(out["target_spec"].std()))
        spread[normalise] = stds

    on, off = spread[True], spread[False]
    assert off[1] / off[0] > 1.5, (
        f"the drift was not reproduced: target std {off} for a 4x frontend")
    assert abs(on[1] / on[0] - 1.0) < 0.05, (
        f"a normalised target still grew with the frontend: {on}")


def test_normalisation_is_recorded_in_the_forward_output():
    r = ROUTES["E19_256"]
    meta = meta_for("E19_256")
    for flag in (True, False):
        m = build(normalize_spec_target=flag)
        out = m(torch.randn(1, r.n_channels, r.window_samples), "E19_256",
                channel_meta=meta, mask_ratio=0.5)
        assert out["normalize_spec_target"] is flag


# --------------------------------------------------------------------------- #
# No dropout means no dropout
# --------------------------------------------------------------------------- #

def test_dropout_zero_reaches_every_module_including_the_frontend():
    """`dropout: 0` used to leave the wavelet frontend's FFN dropping 10%.

    Masked reconstruction already corrupts its input; a second, uncontrolled
    corruption adds noise to a regression target. If the config says none, it
    has to mean none everywhere the gradient flows.
    """
    import torch.nn as nn

    m = build(dropout=0.0)
    live = [(n, mod.p) for n, mod in m.named_modules()
            if isinstance(mod, nn.Dropout) and mod.p > 0]
    assert not live, f"dropout still active at: {live}"

    # And the setting is honoured, not ignored in the other direction.
    m = build(dropout=0.3)
    fe = m.wavelet_frontends["E19_256"]
    ps = {mod.p for _, mod in fe.named_modules() if isinstance(mod, nn.Dropout)}
    assert ps == {0.3}, f"the frontend ignored the configured dropout: {ps}"


def test_the_config_asks_for_none():
    import yaml
    with open(os.path.join(ROOT, "configs", "pretrain", "eeg_c1_moe.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["dropout"] == 0.0
    assert cfg["model"]["mask_ratio"] == 0.70


# --------------------------------------------------------------------------- #
# What fine-tuning is handed
# --------------------------------------------------------------------------- #

def test_export_carries_every_parameter_the_encoder_uses(tmp_path):
    """Derived from the forward pass, not from reading the allowlist.

    channel_token_gate is a bare scalar, so it does not start with
    "channel_to_token." and a list of module prefixes dropped it. It gates the
    entire channel-identity contribution and initialises to zero, so the export
    handed fine-tuning an encoder whose C1 mechanism contributed exactly
    nothing -- in the experiment that exists to measure that mechanism.
    """
    import subprocess

    route_id = "E32_512"
    r = ROUTES[route_id]
    m = build()
    x = torch.randn(2, r.n_channels, r.window_samples)
    m.encode(x, route_id, channel_meta=meta_for(route_id)).sum().backward()

    # Every parameter the encoder path actually reaches.
    needed = {n for n, p in m.named_parameters() if p.grad is not None}
    # Minus the other routes' frontends and the other rate's patcher, which
    # this route does not use and the export deliberately leaves behind.
    needed = {n for n in needed
              if not (n.startswith("wavelet_frontends.")
                      and not n.startswith(f"wavelet_frontends.{route_id}."))
              and not (n.startswith("patch_embed_by_rate.")
                       and not n.startswith(f"patch_embed_by_rate.{r.rate_key}."))}
    assert "channel_token_gate" in needed, "the test's premise moved"

    ck = tmp_path / "ck.pth"
    torch.save({"model": m.state_dict(), "config": {"model": {}},
                "epoch": 0, "global_step": 0}, ck)
    out = tmp_path / "enc.pth"
    res = subprocess.run(
        [sys.executable, "scripts/export_eeg_pretrained_encoder.py",
         "--checkpoint", str(ck), "--route", route_id, "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    got = set(torch.load(out, map_location="cpu", weights_only=False)["model"])

    renamed = set()
    for n in needed:
        n = n.replace(f"wavelet_frontends.{route_id}.", "wavelet_frontend.")
        n = n.replace(f"patch_embed_by_rate.{r.rate_key}.", "patch_embed.")
        renamed.add(n)
    missing = sorted(renamed - got)
    assert not missing, f"the encoder needs these and the export drops them: {missing}"


def test_encode_accepts_a_downstream_window_length():
    """P300 epochs are 2 s against E64_256's 4 s, and nothing here is fixed
    to a length: conv frontend, conv patcher, generated position encoding, RoPE."""
    route_id = "E64_256"
    r = ROUTES[route_id]
    m = build()
    meta = meta_for(route_id)
    # The frontend preserves width exactly, so the patch count is T / patch_t.
    for samples in (r.window_samples, r.window_samples // 2, r.patch_t * 5):
        out = m.encode(torch.randn(2, r.n_channels, samples), route_id,
                       channel_meta=meta)
        assert out.shape == (2, r.n_channels * (samples // r.patch_t),
                             m.embed_dim), f"{samples} samples gave {tuple(out.shape)}"

    # A width that is not a whole number of patches dies inside ScaleFold's
    # synthesis convolution with a message about 4D padding. Refused here
    # instead, where the window is still in hand.
    with pytest.raises(ValueError, match="not a whole number"):
        m.encode(torch.randn(1, r.n_channels, r.window_samples + r.patch_t // 2),
                 route_id, channel_meta=meta)
