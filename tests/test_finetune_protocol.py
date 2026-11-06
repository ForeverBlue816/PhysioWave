# -*- coding: utf-8 -*-
"""The protocol knobs that decide whether a downstream number means anything.

Three of them, each of which used to be able to go wrong silently:

* the config's ``train:`` block. Until ``resolve_hparams`` existed nothing in
  ``finetune_main`` read it, so ``batch_size: 32`` in ``eeg_c1_sleep.yaml``
  described a run at 64 and said so in a config a reader would quote.
* a frozen encoder's dropout. ``model.train()`` puts the whole module in train
  mode, so a linear probe was reading a fresh dropout sample of the
  representation every step rather than the representation.
* pooling. ``mean`` averages over electrodes AND time, which for an ERP is the
  signal divided by the number of patches.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from physiowave.eeg_c1.downstream import EEGC1Downstream
from physiowave.eeg_c1.routes import ROUTES
from physiowave.train.finetune_main import HPARAM_FALLBACKS, resolve_hparams


SLOTS = list(ROUTES["E64_256"].slots)


def build(pool="mean", freeze=False, num_classes=2, **kw):
    return EEGC1Downstream(
        in_channels=8, window_samples=512, sampling_rate=256, patch_samples=128,
        num_classes=num_classes, channel_names=SLOTS[:8], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool=pool, freeze_encoder=freeze, **kw)


def namespace(**kw):
    base = {k: None for k in HPARAM_FALLBACKS}
    base["config"] = "finetune/eeg_c1_p300"
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# where a hyper-parameter comes from
# --------------------------------------------------------------------------- #
def test_the_config_train_block_is_read():
    args = namespace()
    rows = resolve_hparams(args, {"train": {"lr": 5e-4, "epochs": 100,
                                            "select_by": "auroc"}})
    assert args.lr == 5e-4 and args.epochs == 100 and args.select_by == "auroc"
    assert dict((k, s) for k, _, s in rows)["lr"].startswith("config:")


def test_the_command_line_beats_the_config():
    args = namespace(lr=2e-4)
    resolve_hparams(args, {"train": {"lr": 5e-4}})
    assert args.lr == 2e-4


def test_a_value_in_neither_falls_back_and_says_so():
    args = namespace()
    rows = resolve_hparams(args, {})
    assert args.weight_decay == HPARAM_FALLBACKS["weight_decay"]
    assert dict((k, s) for k, _, s in rows)["weight_decay"] == "builtin"


def test_an_unknown_selection_metric_is_refused():
    with pytest.raises(SystemExit):
        resolve_hparams(namespace(), {"train": {"select_by": "f_beta"}})


# --------------------------------------------------------------------------- #
# a frozen encoder is a frozen encoder
# --------------------------------------------------------------------------- #
def test_a_frozen_encoder_stays_in_eval_inside_model_train():
    model = build(freeze=True, dropout=0.5, head_dropout=0.5)
    model.train()
    assert not model.shared_transformer.training
    assert not model.wavelet_frontend.training
    # the head is what is being trained, and its dropout has to be live
    assert model.head_drop.training and model.head.training


def test_an_unfrozen_encoder_still_trains():
    model = build(freeze=False, dropout=0.5)
    model.train()
    assert model.shared_transformer.training


def test_a_frozen_probe_gives_the_same_features_twice():
    """The point of the eval override: a probe reads a fixed representation."""
    model = build(freeze=True, dropout=0.5, head_dropout=0.0)
    model.train()
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    with torch.no_grad():
        a = model(x, meta)["features"]
        b = model(x, meta)["features"]
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def test_time_pooling_keeps_the_time_axis():
    model = build(pool="time", probe_dim=16)
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    out = model(x, meta)
    # 512 / 128 = 4 patches, each projected to probe_dim and kept
    assert out["features"].shape == (2, 4 * 16)
    assert out["logits"].shape == (2, 2)


def test_mean_pooling_is_unchanged():
    model = build(pool="mean")
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    out = model(x, meta)
    assert out["features"].shape == (2, 32)


def test_time_pooling_is_sensitive_to_when_the_deflection_happens():
    """A mean over time cannot tell these apart; that is the whole objection."""
    model = build(pool="time", probe_dim=16).eval()
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    base = torch.zeros(1, 8, 512)
    early, late = base.clone(), base.clone()
    early[:, :, 0:128] = 3.0
    late[:, :, 384:512] = 3.0
    with torch.no_grad():
        a = model(early, meta)["features"]
        b = model(late, meta)["features"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError):
        build(pool="attention")


# --------------------------------------------------------------------------- #
# the adaptive spatial filter, and the head that reads the time axis
# --------------------------------------------------------------------------- #
SLEEP_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "Fpz", "Fz", "Cz", "CPz",
               "Pz", "POz", "Oz"]


def build_free(pool="mean", freeze=False, num_classes=5, **kw):
    """A montage that is on no route, so it gets its own frontend."""
    return EEGC1Downstream(
        in_channels=2, window_samples=600, sampling_rate=100, patch_samples=50,
        num_classes=num_classes, channel_names=["Fpz-Cz", "Pz-Oz"],
        embed_dim=32, depth=1, num_heads=2, channel_embed_dim=8, pool=pool,
        freeze_encoder=freeze, **kw)


def test_a_scale_filter_keeps_the_montage():
    model = build(pool="time", spatial_filter="scale")
    assert model.raw_channels == 8 and model.in_channels == 8
    assert model.spatial_filter.gain.shape == (1, 8, 1)


def test_a_mix_filter_replaces_the_montage_with_its_named_outputs():
    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES)
    # the file still hands it two derivations; the frontend is built for 13
    assert model.raw_channels == 2
    assert model.in_channels == 13
    assert model.model_channel_names == SLEEP_NAMES
    out = model(torch.randn(2, 2, 600))
    assert out["logits"].shape == (2, 5)


def test_a_mix_looks_up_its_own_names_not_the_files():
    """Fpz-Cz's vocabulary row must not end up attached to a virtual Fz."""
    from channel_embedding import channel_ids_for

    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES)
    ids = model._model_name_meta(torch.device("cpu"))["channel_ids"]
    expected, _ = channel_ids_for(SLEEP_NAMES)
    assert ids.tolist() == list(expected)


def test_the_spatial_filter_trains_under_a_frozen_encoder():
    """EEGPT's probe optimiser is [chan_scale] + the linear layers."""
    model = build(pool="time", spatial_filter="scale", freeze=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("spatial_filter") for n in trainable)
    assert any(n.startswith("head") for n in trainable)
    assert not any(n.startswith("shared_transformer") for n in trainable)


def test_the_filter_stays_in_train_mode_when_the_encoder_is_frozen():
    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES, freeze=True)
    model.train()
    assert model.spatial_filter.training
    assert not model.shared_transformer.training


def test_the_attn_head_does_not_grow_with_the_number_of_positions():
    """Why staging gets it and P300 does not: 60 positions cannot be flattened."""
    short = build_free(pool="attn", probe_dim=16, head_depth=1)
    long_ = EEGC1Downstream(
        in_channels=2, window_samples=3000, sampling_rate=100, patch_samples=50,
        num_classes=5, channel_names=["Fpz-Cz", "Pz-Oz"], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool="attn", probe_dim=16, head_depth=1)
    n = lambda m: sum(p.numel() for p in m.head.parameters())  # noqa: E731
    assert short.n_patches == 12 and long_.n_patches == 60
    assert n(short) == n(long_)


def test_the_flatten_head_does_grow_with_them():
    a = build(pool="time", probe_dim=16)
    b = EEGC1Downstream(
        in_channels=8, window_samples=1024, sampling_rate=256, patch_samples=128,
        num_classes=2, channel_names=SLOTS[:8], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool="time", probe_dim=16)
    n = lambda m: sum(p.numel() for p in m.head.parameters())  # noqa: E731
    assert n(b) > n(a)


def test_max_norm_actually_constrains_the_rows():
    from physiowave.eeg_c1.heads import MaxNormLinear

    layer = MaxNormLinear(8, 3, max_norm=0.25)
    with torch.no_grad():
        layer.weight.fill_(10.0)
    layer(torch.randn(2, 8))
    assert torch.all(layer.weight.norm(p=2, dim=1) <= 0.25 + 1e-5)


def test_max_norm_zero_leaves_the_weights_alone():
    from physiowave.eeg_c1.heads import MaxNormLinear

    layer = MaxNormLinear(8, 3, max_norm=0.0)
    with torch.no_grad():
        layer.weight.fill_(10.0)
    layer(torch.randn(2, 8))
    assert torch.all(layer.weight == 10.0)


def test_channel_attention_pooling_is_not_a_mean():
    from physiowave.eeg_c1.heads import ChannelPool

    tokens = torch.randn(2, 6, 4, 32)
    pool = ChannelPool("attn", 32)
    with torch.no_grad():
        pool.query.copy_(tokens[0, 0, 0] * 4.0)     # a query that picks one out
        assert not torch.allclose(pool(tokens), tokens.mean(dim=1), atol=1e-3)


def test_a_mix_without_output_names_is_refused():
    with pytest.raises(ValueError):
        build_free(spatial_filter="mix")


# --------------------------------------------------------------------------- #
# how much of the recorded montage reaches a pretrained route
# --------------------------------------------------------------------------- #
def _p300_module():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "EEG", "physio_p300_finetune.py")
    spec = importlib.util.spec_from_file_location("p300_conv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def on_route(names, aliases=None):
    return EEGC1Downstream(
        in_channels=len(names), window_samples=512, sampling_rate=256,
        patch_samples=128, num_classes=2, channel_names=names,
        route_id="E64_256", embed_dim=32, depth=1, num_heads=2,
        channel_embed_dim=8, pool="time", slot_aliases=aliases)


def test_the_62_set_is_exactly_the_recorded_electrodes_the_route_has():
    conv = _p300_module()
    from channel_embedding import normalize_channel_name as norm

    slots = {norm(s) for s in ROUTES["E64_256"].slots}
    recorded = {norm(c) for c in conv.EEG_64}
    assert {norm(c) for c in conv.CHANNELS_62} == slots & recorded
    assert len(conv.CHANNELS_62) == 62


def test_58_leaves_four_pretrained_slots_empty():
    conv = _p300_module()
    a, b = on_route(conv.CHANNELS_58), on_route(conv.CHANNELS_62)
    assert int((a.slot_index >= 0).sum()) == 58
    assert int((b.slot_index >= 0).sum()) == 62


def test_all_64_is_refused_without_an_explicit_alias():
    """P9 is not TP9, and reading it as one has to be someone's decision."""
    conv = _p300_module()
    with pytest.raises(ValueError, match="P9"):
        on_route(conv.CHANNELS_64)


def test_an_alias_fills_the_route():
    conv = _p300_module()
    model = on_route(conv.CHANNELS_64, {"P9": "TP9", "P10": "TP10"})
    assert int((model.slot_index >= 0).sum()) == 64
    out = model(torch.randn(2, 64, 512),
                {"channel_ids": torch.arange(1, 65),
                 "valid_channel_mask": torch.ones(64, dtype=torch.bool)})
    assert out["logits"].shape == (2, 2)


def test_an_alias_onto_a_slot_that_does_not_exist_is_refused():
    conv = _p300_module()
    with pytest.raises(ValueError, match="not.*slots"):
        on_route(conv.CHANNELS_64, {"P9": "Nz", "P10": "TP10"})


def test_aliases_mean_nothing_off_a_route():
    with pytest.raises(ValueError, match="only means something on a route"):
        EEGC1Downstream(
            in_channels=2, window_samples=600, sampling_rate=100,
            patch_samples=50, num_classes=5, channel_names=["Fpz-Cz", "Pz-Oz"],
            embed_dim=32, depth=1, num_heads=2, channel_embed_dim=8,
            slot_aliases={"Fpz-Cz": "Fpz"})


def test_the_split_stage_subsets_the_cache_by_name():
    """58 vs 62 is a split, not another pass over 245 runs of EDF."""
    import numpy as np

    conv = _p300_module()
    cached = conv.EEG_64
    X = np.arange(len(cached) * 4, dtype=np.float32).reshape(1, len(cached), 4)
    out = conv._take_channels(X, cached, conv.CHANNELS_58, "test")
    assert out.shape == (1, 58, 4)
    for j, name in enumerate(conv.CHANNELS_58):
        assert np.array_equal(out[0, j], X[0, cached.index(name)])


def test_asking_the_cache_for_a_channel_it_lacks_is_refused():
    import numpy as np

    conv = _p300_module()
    X = np.zeros((1, 3, 4), dtype=np.float32)
    with pytest.raises(SystemExit, match="Cz"):
        conv._take_channels(X, ["Fp1", "Fp2", "Oz"], ["Fp1", "Cz"], "test")


# --------------------------------------------------------------------------- #
# a cache that predates the channel names in it
# --------------------------------------------------------------------------- #
def _legacy_cache(tmp_path, n_channels=58, subjects=(1, 2)):
    import numpy as np

    conv = _p300_module()
    base = tmp_path / "cache"
    (base / f"c{n_channels}").mkdir(parents=True)
    for s in subjects:
        np.savez_compressed(base / f"c{n_channels}" / f"sub{s:02d}.npz",
                            data=np.zeros((4, n_channels, 512), dtype="float32"),
                            label=np.zeros(4, dtype="int64"))
    return conv, str(base)


def test_a_legacy_cache_is_adopted_rather_than_re_decoded(tmp_path):
    """The list is not a guess: the directory name is what the writer used."""
    conv, base = _legacy_cache(tmp_path)
    path, names = conv.find_subject_cache(base, 1)
    assert path.endswith("sub01.npz")
    assert names == conv.CHANNELS_58


def test_the_superset_cache_wins_over_a_legacy_one(tmp_path):
    import numpy as np

    conv, base = _legacy_cache(tmp_path)
    (tmp_path / "cache" / "c64").mkdir()
    np.savez_compressed(tmp_path / "cache" / "c64" / "sub01.npz",
                        data=np.zeros((4, 64, 512), dtype="float32"),
                        label=np.zeros(4, dtype="int64"),
                        channel_names=np.array(conv.EEG_64, dtype="U32"))
    path, names = conv.find_subject_cache(base, 1)
    assert path.endswith("c64/sub01.npz") and len(names) == 64


def test_a_legacy_cache_with_the_wrong_width_is_refused(tmp_path):
    conv, base = _legacy_cache(tmp_path, n_channels=58)
    import numpy as np

    np.savez_compressed(tmp_path / "cache" / "c58" / "sub03.npz",
                        data=np.zeros((4, 40, 512), dtype="float32"),
                        label=np.zeros(4, dtype="int64"))
    with pytest.raises(SystemExit, match="does not match"):
        conv.find_subject_cache(base, 3)


def test_58_still_comes_out_of_a_legacy_cache(tmp_path):
    import numpy as np

    conv, _ = _legacy_cache(tmp_path)
    X = np.arange(58 * 4, dtype="float32").reshape(1, 58, 4)
    assert conv._take_channels(X, conv.CHANNELS_58, conv.CHANNELS_58, "x").shape \
        == (1, 58, 4)


def test_62_out_of_a_58_cache_names_the_four_that_were_never_decoded(tmp_path):
    import numpy as np

    conv, _ = _legacy_cache(tmp_path)
    X = np.zeros((1, 58, 4), dtype="float32")
    with pytest.raises(SystemExit, match="AF7"):
        conv._take_channels(X, conv.CHANNELS_58, conv.CHANNELS_62, "x")


def test_a_missing_subject_is_reported_not_invented(tmp_path):
    conv, base = _legacy_cache(tmp_path, subjects=(1,))
    assert conv.find_subject_cache(base, 9) == (None, None)


# --------------------------------------------------------------------------- #
# the gate that scales the whole channel-identity path
# --------------------------------------------------------------------------- #
def _export(tmp_path, drop_gate: bool):
    """An exported encoder, optionally missing the key an old export dropped."""
    from physiowave.eeg_c1.downstream import ROUTE_BOUND, TRANSFERABLE
    from channel_embedding import vocab_payload

    full = EEGC1Downstream(
        in_channels=64, window_samples=1024, sampling_rate=256, patch_samples=128,
        num_classes=2, channel_names=SLOTS, route_id="E64_256", embed_dim=32,
        depth=1, num_heads=2, channel_embed_dim=8)
    keep = {k: v for k, v in full.state_dict().items()
            if k.startswith(TRANSFERABLE) or k.startswith(ROUTE_BOUND)}
    if drop_gate:
        keep.pop("channel_token_gate")
    path = tmp_path / ("no_gate.pth" if drop_gate else "with_gate.pth")
    torch.save({"model": keep, "route_id": "E64_256", **vocab_payload()}, path)
    return str(path)


def _on_route_64(**kw):
    return EEGC1Downstream(
        in_channels=64, window_samples=512, sampling_rate=256, patch_samples=128,
        num_classes=2, channel_names=SLOTS, route_id="E64_256", embed_dim=32,
        depth=1, num_heads=2, channel_embed_dim=8, **kw)


def test_a_gateless_export_is_refused(tmp_path):
    """tanh(0) = 0 multiplies the whole channel path away, silently."""
    path = _export(tmp_path, drop_gate=True)
    with pytest.raises(SystemExit, match="channel_token_gate"):
        _on_route_64().load_pretrained(path)


def test_the_refusal_names_the_re_export(tmp_path):
    path = _export(tmp_path, drop_gate=True)
    with pytest.raises(SystemExit, match="export_eeg_pretrained_encoder"):
        _on_route_64().load_pretrained(path)


def test_a_gateless_export_loads_when_the_ablation_is_deliberate(tmp_path):
    path = _export(tmp_path, drop_gate=True)
    report = _on_route_64().load_pretrained(path, allow_missing_gate=True)
    assert "channel_token_gate" not in report["taken"]


def test_a_complete_export_loads(tmp_path):
    path = _export(tmp_path, drop_gate=False)
    report = _on_route_64().load_pretrained(path)
    assert "channel_token_gate" in report["taken"]


def test_a_zero_gate_really_does_erase_the_channel_code():
    """Why the missing key matters, stated as behaviour rather than as a claim.

    Off a route, so the ids the caller passes are the ids the encoder reads:
    on a route `_slot_meta` replaces them with the route's own, which is
    correct there and would hide the effect being measured here.
    """
    model = build_free(pool="mean").eval()
    x = torch.randn(1, 2, 600)
    a = {"channel_ids": torch.tensor([2, 3]),
         "valid_channel_mask": torch.ones(2, dtype=torch.bool)}
    b = {"channel_ids": torch.tensor([32, 52]),
         "valid_channel_mask": torch.ones(2, dtype=torch.bool)}
    with torch.no_grad():
        model.channel_token_gate.fill_(0.0)
        # two completely different electrode identities, identical tokens
        assert torch.allclose(model.encode(x, a), model.encode(x, b))
        model.channel_token_gate.fill_(0.33)          # what pretraining reached
        assert not torch.allclose(model.encode(x, a), model.encode(x, b))


def test_a_frozen_probe_cannot_recover_a_dead_gate():
    model = _on_route_64(freeze_encoder=True)
    gate = dict(model.named_parameters())["channel_token_gate"]
    assert not gate.requires_grad


# --------------------------------------------------------------------------- #
# attentive statistics pooling
# --------------------------------------------------------------------------- #
def test_zero_queries_reproduce_the_plain_mean_and_std():
    """The property that makes this a safe replacement for the mean head."""
    from physiowave.eeg_c1.heads import AttentiveStatsPool

    pool = AttentiveStatsPool(8, heads=3).eval()
    t = torch.randn(4, 21, 8)
    out = pool(t)
    h = torch.nn.functional.layer_norm(t, (8,))
    mu, sd = h.mean(1), h.std(1, unbiased=False)
    for k in range(3):
        head = out[:, k * 16:(k + 1) * 16]
        assert torch.allclose(head[:, :8], mu, atol=1e-5)
        assert torch.allclose(head[:, 8:], sd, atol=1e-4)


def test_a_trained_query_stops_being_a_mean():
    from physiowave.eeg_c1.heads import AttentiveStatsPool

    pool = AttentiveStatsPool(8, heads=1).eval()
    t = torch.randn(2, 21, 8)
    flat = pool(t)
    with torch.no_grad():
        pool.query.copy_(t[0, 0] * 8.0)          # a query that picks one cell out
    assert not torch.allclose(pool(t), flat, atol=1e-3)


def test_sigma_sees_variation_a_mean_cannot():
    """Zero spread across tokens vs real spread, at the same token norm.

    The comparison has to be made AFTER the LayerNorm the pool applies, which
    is per token across D -- so a token set that differs only by a constant
    offset is identical to it, and would test nothing.
    """
    from physiowave.eeg_c1.heads import AttentiveStatsPool

    pool = AttentiveStatsPool(4, heads=1).eval()
    v = torch.tensor([1.0, -1.0, 2.0, -2.0])
    still = v.repeat(1, 12, 1)                   # every token the same
    moving = torch.randn(1, 12, 4)
    assert pool(still)[:, 4:].abs().max() < 1e-3          # no spread, no sigma
    assert pool(moving)[:, 4:].abs().max() > 0.1          # spread, sigma


def test_the_stat_head_runs_on_both_task_shapes():
    p300 = build(pool="stat", pool_heads=4)                   # 8 ch x 4 patches
    out = p300(torch.randn(2, 8, 512),
               {"channel_ids": torch.arange(1, 9),
                "valid_channel_mask": torch.ones(8, dtype=torch.bool)})
    assert out["logits"].shape == (2, 2)
    assert out["features"].shape == (2, 4 * 2 * 32)

    sleep = build_free(pool="stat", pool_heads=4)              # 2 ch x 12 patches
    out = sleep(torch.randn(2, 2, 600),
                {"channel_ids": torch.tensor([2, 3]),
                 "valid_channel_mask": torch.ones(2, dtype=torch.bool)})
    assert out["logits"].shape == (2, 5)


def test_the_stat_head_is_the_only_thing_a_probe_trains():
    model = build(pool="stat", pool_heads=4, freeze=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head_pool.norm.weight", "head_pool.norm.bias",
                         "head_pool.query", "head.weight", "head.bias"}


def test_pooling_survives_bf16_tokens():
    """sigma is E[h^2] - mu^2, which is where bf16 loses its digits."""
    from physiowave.eeg_c1.heads import AttentiveStatsPool

    pool = AttentiveStatsPool(16, heads=2).eval()
    t = torch.randn(2, 30, 16)
    ref = pool(t)
    got = pool(t.to(torch.bfloat16))
    assert got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), ref, atol=2e-2)


def test_the_shipped_configs_pool_by_mean():
    """The shipped default, pinned. The other shapes stay reachable by --set.

    Pinned rather than assumed because the default has moved twice and both
    times something downstream still described the old one.
    """
    import yaml

    for name in ("eeg_c1_p300", "eeg_c1_sleep"):
        with open(f"configs/finetune/{name}.yaml") as fh:
            c1 = yaml.safe_load(fh)["model"]["eeg_c1"]
        assert c1["pool"] == "mean", f"{name} ships pool={c1['pool']!r}"
        assert "route_id" not in c1, f"{name} places its montage on a route"


def test_patience_is_a_resolved_hyperparameter():
    """It is logged, and the config can set it.

    Left out of the table it was a flag whose absence looked exactly like its
    presence: a run given no --patience printed nothing about early stopping
    and then did not stop.
    """
    args = namespace()
    rows = resolve_hparams(args, {"train": {"patience": 15, "min_delta": 1e-4}})
    assert args.patience == 15 and args.min_delta == 1e-4
    sources = dict((k, s) for k, _, s in rows)
    assert sources["patience"].startswith("config:")


def test_patience_defaults_to_off_and_says_so():
    args = namespace()
    rows = resolve_hparams(args, {})
    assert args.patience == 0
    assert dict((k, s) for k, _, s in rows)["patience"] == "builtin"


def test_the_command_line_beats_a_config_patience():
    args = namespace(patience=5)
    resolve_hparams(args, {"train": {"patience": 15}})
    assert args.patience == 5
