"""Reference augmentation: physical legality, tiering and consistency losses."""

from __future__ import annotations

import pytest
import torch

from physiowave.channels.tare import ChannelMeta
from physiowave.data.montages import montage
from physiowave.pretrain.losses import reference_consistency_loss
from physiowave.pretrain.reference import (
    HARD_VIEWS,
    STANDARD_VIEWS,
    ReferenceConfig,
    apply_reference,
    build_views,
    reference_operator,
    sample_views,
)


@pytest.fixture
def meta64():
    names, xyz = montage("standard_1010_64")
    return ChannelMeta(names, xyz, montage_type="standard_1010", reference_type="original")


def test_every_view_is_a_channel_linear_map(meta64):
    """Constraint C: a legal reference view is exactly ``M @ X``.

    Re-referencing is a linear transformation of the channel axis, so a view that
    cannot be written that way corresponds to a signal that was never recorded.
    """
    cfg = ReferenceConfig(num_views=1)
    x = torch.randn(3, meta64.num_channels(), 256)
    checked = 0
    for name in list(STANDARD_VIEWS) + list(HARD_VIEWS):
        res = reference_operator(name, meta64, cfg)
        if res is None or res[0] is None:
            continue
        M, _ = res
        assert M.shape == (x.shape[1], x.shape[1])
        view = apply_reference(x, M)
        assert torch.allclose(view, torch.einsum("ij,bjt->bit", M, x), atol=1e-6)
        checked += 1
    assert checked >= 3, "too few reference views were constructible"


def test_car_operator_matches_the_definition(meta64):
    cfg = ReferenceConfig(car_min_channels=8)
    M, _ = reference_operator("common_average", meta64, cfg)
    x = torch.randn(2, meta64.num_channels(), 128)
    assert torch.allclose(apply_reference(x, M), x - x.mean(dim=1, keepdim=True), atol=1e-5)


def test_car_skipped_on_a_sparse_montage(caplog):
    """A common average over too few electrodes is not a neutral reference."""
    names, xyz = montage("standard_1020_19")
    meta = ChannelMeta(names, xyz)
    res = reference_operator("common_average", meta, ReferenceConfig(car_min_channels=32))
    assert res[0] is None and "car_min_channels" in res[1]["skipped"]


def test_linked_mastoids_needs_mastoid_channels(meta64):
    M, info = reference_operator("linked_mastoids", meta64, ReferenceConfig())
    assert M is not None and info["tier"] == "standard"
    names, xyz = montage("standard_1010_61")           # no M1/M2
    res = reference_operator("linked_mastoids", ChannelMeta(names, xyz), ReferenceConfig())
    assert res[0] is None and "mastoid" in res[1]["skipped"]


def test_hard_views_are_marked_lateralised(meta64):
    """Single-sided references carry a lateralisation flag and the `hard` tier."""
    for name in ("left_mastoid", "right_mastoid", "random_channel"):
        res = reference_operator(name, meta64, ReferenceConfig())
        if res is None or res[0] is None:
            continue
        _, info = res
        assert info["tier"] == "hard" and info.get("lateralised") is True


def test_bipolar_montage_only_yields_the_original_view():
    """A bipolar derivation cannot be re-referenced; only `original` is legal."""
    names, xyz = montage("standard_1020_19")
    meta = ChannelMeta(names, xyz, derivation_type="bipolar")
    for name in ("common_average", "linked_mastoids", "left_ear"):
        assert reference_operator(name, meta, ReferenceConfig()) is None
    M, info = reference_operator("original", meta, ReferenceConfig())
    assert torch.allclose(M, torch.eye(len(names)))


def test_hard_view_probability_is_respected():
    cfg = ReferenceConfig(num_views=1, hard_view_prob=0.0)
    g = torch.Generator().manual_seed(0)
    assert all(v in STANDARD_VIEWS for v in
               [sample_views(cfg, g)[0] for _ in range(30)])
    cfg = ReferenceConfig(num_views=1, hard_view_prob=1.0)
    assert all(v in HARD_VIEWS for v in [sample_views(cfg, g)[0] for _ in range(30)])


def test_build_views_tags_tiers(meta64):
    x = torch.randn(2, meta64.num_channels(), 256)
    views = build_views(x, meta64, ReferenceConfig(car_min_channels=8),
                        view_names=["original", "common_average", "left_mastoid"])
    tiers = {v["name"]: v["tier"] for v in views}
    assert tiers["original"] == "standard" and tiers["common_average"] == "standard"
    assert tiers.get("left_mastoid") == "hard"
    for v in views:
        assert v["signal"].shape == x.shape


def test_reference_consistency_anchor_and_pairwise():
    """The anchored form pulls views to a stop-gradient target; pairwise is symmetric."""
    a = torch.randn(4, 16, requires_grad=True)
    b = torch.randn(4, 16, requires_grad=True)
    anchor = torch.randn(4, 16, requires_grad=True)
    l_anchor = reference_consistency_loss([a, b], anchor)
    l_anchor.backward()
    assert anchor.grad is None, "the anchor must be stop-gradient"
    assert a.grad is not None and b.grad is not None

    a2 = torch.randn(4, 16, requires_grad=True)
    b2 = torch.randn(4, 16, requires_grad=True)
    reference_consistency_loss([a2, b2]).backward()
    assert a2.grad is not None and b2.grad is not None

    same = torch.randn(4, 16)
    assert reference_consistency_loss([same, same.clone()], same.clone()).item() < 1e-6
