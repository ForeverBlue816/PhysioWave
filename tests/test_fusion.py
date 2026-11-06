"""RALF: arbitrary modality subsets, reliability, corruption and consistency."""

from __future__ import annotations

import pytest
import torch

from physiowave.models.encoder import EncoderConfig, PhysioWaveEncoder
from physiowave.models.fusion import MultimodalPhysioWave, RALFConfig, ReliabilityAwareLatentFusion
from physiowave.pretrain.corruption import CORRUPTIONS, CorruptionConfig, SignalCorruptor
from physiowave.wavelet.wast import WASTConfig


def _fusion(D=32, K=4):
    cfg = RALFConfig(embed_dim=D, num_fusion_tokens=4, num_heads=2, depth=1, num_classes=3)
    return ReliabilityAwareLatentFusion(cfg).eval()


def _feats(B=2, D=32, counts=(4, 6, 3)):
    """Different token counts per modality on purpose: the interface must not care."""
    return {m: torch.randn(B, n, D) for m, n in zip(("eeg", "ecg", "semg"), counts, strict=True)}


@pytest.mark.parametrize("subset", [
    ("eeg",), ("ecg",), ("semg",),
    ("eeg", "ecg"), ("eeg", "semg"), ("ecg", "semg"),
    ("eeg", "ecg", "semg"),
])
def test_every_modality_subset_infers(subset):
    m = _fusion()
    feats = {k: v for k, v in _feats().items() if k in subset}
    out = m(feats)
    assert out["logits"].shape == (2, 3) and torch.isfinite(out["logits"]).all()
    assert set(out["reliability"]) == set(subset)


def test_per_item_modality_mask():
    """Availability may differ per batch item, not just per batch."""
    m = _fusion()
    feats = _feats(B=4)
    mask = {"eeg": torch.tensor([True, True, False, True]),
            "ecg": torch.tensor([True, False, True, False]),
            "semg": torch.tensor([False, True, True, True])}
    out = m(feats, mask)
    assert out["logits"].shape == (4, 3) and torch.isfinite(out["logits"]).all()
    assert out["reliability"]["eeg"][2].item() == 0.0


def test_all_modalities_missing_raises():
    """An item with no modality at all is an error, never a silent prediction."""
    m = _fusion()
    feats = _feats(B=2)
    mask = {k: torch.zeros(2, dtype=torch.bool) for k in feats}
    with pytest.raises(ValueError, match="no available modality"):
        m(feats, mask)
    with pytest.raises(AssertionError):
        m({})


def test_token_counts_do_not_change_the_interface():
    m = _fusion()
    a = m(_feats(counts=(4, 6, 3)))["logits"]
    b = m(_feats(counts=(8, 2, 9)))["logits"]
    assert a.shape == b.shape


def test_reliability_supervision_lowers_corrupted_scores():
    """After training against known corruption levels, corrupted input scores lower."""
    torch.manual_seed(0)
    m = ReliabilityAwareLatentFusion(
        RALFConfig(embed_dim=16, num_fusion_tokens=2, num_heads=2, depth=1, num_classes=2))
    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    B = 16
    clean = {mm: torch.randn(B, 3, 16) * 0.5 for mm in ("eeg", "ecg", "semg")}
    noisy = {mm: v + torch.randn_like(v) * 3.0 for mm, v in clean.items()}
    present = {mm: torch.ones(B) for mm in clean}
    for _ in range(80):
        opt.zero_grad()
        oc = m(clean)
        on = m(noisy)
        loss = (m.reliability_loss(oc["reliability"], {k: torch.ones(B) for k in clean}, present)
                + m.reliability_loss(on["reliability"], {k: torch.zeros(B) for k in clean}, present))
        loss.backward()
        opt.step()
    with torch.no_grad():
        rc = m(clean)["reliability"]["eeg"].mean().item()
        rn = m(noisy)["reliability"]["eeg"].mean().item()
    assert rn < rc - 0.1, f"corrupted reliability {rn:.3f} not below clean {rc:.3f}"


def test_consistency_loss_is_zero_for_identical_predictions():
    m = _fusion()
    logits = torch.randn(4, 3)
    assert m.consistency_loss(logits, logits.clone()).abs().item() < 1e-6
    assert m.consistency_loss(logits, torch.randn(4, 3)).item() > 0


@pytest.mark.parametrize("kind", list(CORRUPTIONS))
def test_corruption_reports_its_severity(kind):
    c = SignalCorruptor(CorruptionConfig(prob=1.0, kinds=[kind], max_level=1.0),
                        torch.Generator().manual_seed(0))
    x = torch.randn(4, 8, 512)
    y, level, kinds = c(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (level > 0).all() and set(kinds) == {kind}
    assert not torch.allclose(x, y), f"{kind} did not change the signal"


def test_corruption_disabled_is_a_noop():
    c = SignalCorruptor(CorruptionConfig(enabled=False))
    x = torch.randn(2, 4, 128)
    y, level, kinds = c(x)
    assert torch.allclose(x, y) and level.sum() == 0 and set(kinds) == {"none"}


def test_multimodal_wrapper_end_to_end():
    """Encoders + fusion, with one modality genuinely absent."""
    def enc(modality, C):
        cfg = EncoderConfig(modality=modality, embed_dim=32, sampling_rate=256.0,
                            wast=WASTConfig(patch_size=32, embed_dim=32, level=2))
        cfg.backbone.depth = 1
        cfg.backbone.num_heads = 4
        cfg.backbone.slot_heads = 2
        cfg.compression.num_queries = 4
        cfg.compression.num_heads = 2
        return PhysioWaveEncoder(cfg)

    model = MultimodalPhysioWave(
        {"eeg": enc("eeg", 19), "ecg": enc("ecg", 12), "semg": enc("semg", 8)},
        RALFConfig(embed_dim=32, num_fusion_tokens=4, num_heads=2, depth=1, num_classes=3),
    ).eval()
    inputs = {"eeg": torch.randn(2, 19, 256), "semg": torch.randn(2, 8, 256)}
    with torch.no_grad():
        out = model(inputs)
    assert out["logits"].shape == (2, 3)
    assert set(out["reliability"]) == {"eeg", "semg"}


def test_concat_fusion_baseline_shares_the_interface():
    """The `original fusion vs RALF` baseline is drop-in comparable."""
    from physiowave.models.fusion import ConcatFusionBaseline

    cfg = RALFConfig(embed_dim=32, num_fusion_tokens=4, num_heads=2, depth=1, num_classes=3)
    base = ConcatFusionBaseline(cfg).eval()
    out = base(_feats())
    assert out["logits"].shape == (2, 3)
    assert set(out["reliability"]) == {"eeg", "ecg", "semg"}

    # A missing modality is zero-filled, which is exactly the weakness RALF fixes.
    partial = base({k: v for k, v in _feats().items() if k != "ecg"})
    assert partial["logits"].shape == (2, 3)
    assert partial["reliability"]["ecg"].sum().item() == 0.0


def test_build_selects_the_fusion_variant():
    from physiowave.config import load_config
    from physiowave.models.build import build_model
    from physiowave.models.fusion import ConcatFusionBaseline, ReliabilityAwareLatentFusion

    small = [f"model.encoders.{m}.{k}" for m in ("eeg", "ecg", "semg")
             for k in ("backbone.depth=1",)]
    ralf = build_model(load_config("fusion/ralf", small))
    assert isinstance(ralf.fusion, ReliabilityAwareLatentFusion)
    concat = build_model(load_config("fusion/ralf", small + ["model.name=concat_fusion"]))
    assert isinstance(concat.fusion, ConcatFusionBaseline)
