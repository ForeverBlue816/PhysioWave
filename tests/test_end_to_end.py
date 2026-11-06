"""End-to-end: encoder forward/backward, the pretraining objective, and the smoke run."""

from __future__ import annotations

import json
import os

import pytest
import torch

from physiowave.channels.tare import ChannelMeta
from physiowave.config import instantiate, load_config
from physiowave.data.montages import montage
from physiowave.models.build import build_model, count_parameters
from physiowave.models.encoder import EncoderConfig, PhysioWaveEncoder
from physiowave.pretrain.objectives import PretrainObjective, PretrainObjectiveConfig
from physiowave.wavelet.wast import WASTConfig


def _cfg(modality="eeg", D=32, K=6, P=32, level=2, fs=256.0):
    cfg = EncoderConfig(modality=modality, embed_dim=D, sampling_rate=fs,
                        wast=WASTConfig(patch_size=P, embed_dim=D, level=level))
    cfg.backbone.depth = 1
    cfg.backbone.num_heads = 4
    cfg.backbone.slot_heads = 2
    cfg.compression.num_queries = K
    cfg.compression.num_heads = 2
    return cfg


def test_encoder_outputs_the_full_contract(meta_64):
    enc = PhysioWaveEncoder(_cfg()).eval()
    with torch.no_grad():
        out = enc(torch.randn(2, 64, 256), meta_64)
    assert out["summary_tokens"].shape == (2, 4, 32)
    assert out["pooled"].shape == (2, 32)
    assert out["quality"].shape == (2,)
    assert 0.0 <= out["quality"].min() and out["quality"].max() <= 1.0
    assert out["tokens"].shape == (2, 6, 8, 32)
    ts = out["token_stats"]
    assert ts["N_old_legacy"] == 3 * 64 * 8 and ts["N_new"] == 6 * 8
    assert out["spatial_info"]["ssl"] is True and out["spatial_info"]["gl"] is True


@pytest.mark.parametrize("modality,C,fs", [("eeg", 19, 256.0), ("ecg", 12, 500.0),
                                           ("semg", 8, 1000.0)])
def test_three_encoders_train_independently(modality, C, fs):
    """Shared macro interface, separate parameters, per-modality SSL policy."""
    enc = PhysioWaveEncoder(_cfg(modality, fs=fs))
    if modality != "eeg":
        assert enc.cfg.spatial.ssl.enabled is False, (
            f"{modality} must not use the spline surface Laplacian"
        )
    x = torch.randn(2, C, 256)
    out = enc(x)
    out["pooled"].pow(2).mean().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and max(g.abs().max().item() for g in grads) > 0


def test_encoders_do_not_share_parameters():
    a, b = PhysioWaveEncoder(_cfg("eeg")), PhysioWaveEncoder(_cfg("ecg"))
    ids_a = {id(p) for p in a.parameters()}
    assert not (ids_a & {id(p) for p in b.parameters()})


def test_variable_channel_counts_and_missing_channels():
    """One encoder handles 19, 32 and 64 electrodes, with or without bad channels."""
    enc = PhysioWaveEncoder(_cfg()).eval()
    outs = {}
    for name in ("standard_1020_19", "standard_1010_61", "standard_1010_64"):
        names, xyz = montage(name)
        meta = ChannelMeta(names, xyz, montage_type="standard_1010")
        with torch.no_grad():
            outs[name] = enc(torch.randn(2, len(names), 256), meta)["pooled"]
        assert outs[name].shape == (2, 32)

    names, xyz = montage("standard_1010_64")
    mask = torch.ones(len(names), dtype=torch.bool)
    mask[[3, 11, 40]] = False
    meta = ChannelMeta(names, xyz, channel_mask=mask)
    x = torch.randn(2, len(names), 256)
    with torch.no_grad():
        a = enc(x, meta)["pooled"]
        x2 = x.clone()
        x2[:, [3, 11, 40]] = 1e3
        b = enc(x2, meta)["pooled"]
    assert torch.allclose(a, b, atol=1e-3), "masked channels leaked into the representation"


def test_channel_permutation_leaves_the_representation_almost_unchanged(meta_64):
    """Channel order carries no meaning: permuting signal + metadata is near-invariant."""
    enc = PhysioWaveEncoder(_cfg()).eval()
    x = torch.randn(2, 64, 256)
    p = torch.randperm(64)
    pmeta = ChannelMeta([meta_64.channel_names[i] for i in p], meta_64.channel_xyz[p],
                        montage_type=meta_64.montage_type,
                        reference_type=meta_64.reference_type)
    with torch.no_grad():
        a = enc(x, meta_64)["pooled"]
        b = enc(x[:, p], pmeta)["pooled"]
    rel = ((a - b).norm(dim=-1) / a.norm(dim=-1)).max().item()
    assert rel < 5e-2, f"permutation changed the pooled representation by {rel:.3e}"


def test_semg_encoder_is_not_channel_permutation_invariant():
    """The sEMG path must be able to tell one electrode from another.

    WAST shares its parameters across channels, so the per-channel embedding is
    the *only* thing that breaks the symmetry. On a forearm ring TARE has no
    coordinate and no label it recognises and hands back the same vector for
    every channel, which leaves the whole encoder invariant to permuting the
    channel axis -- and on an array where "which electrode fired" is the label,
    that discards the signal rather than a nuisance. ``channel_embedding:
    channel_id`` is what the sEMG config uses instead.
    """
    cfg = load_config("pretrain/semg")
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()
    assert enc.tare is None and enc.channel_id is not None

    C, T = 16, 256
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    torch.manual_seed(0)
    x = torch.randn(2, C, T)
    p = torch.randperm(C)
    with torch.no_grad():
        a = enc(x, meta)["logits"]
        b = enc(x[:, p], meta)["logits"]
    spread = (a - b).abs().max().item()
    assert spread > 1e-3, f"permuting the channels changed the logits by only {spread:.2e}"


def test_channel_embedding_none_is_permutation_invariant():
    """The counterpart: with no channel embedding the symmetry is exact.

    This is what the sEMG path used to do by accident, and it is what makes the
    test above worth having.
    """
    cfg = EncoderConfig(modality="semg", embed_dim=32, num_summary_tokens=2,
                        channel_embedding="none", use_spatial_frontend=False,
                        num_classes=5, wast=WASTConfig(embed_dim=32, patch_size=32, level=2))
    enc = PhysioWaveEncoder(cfg).eval()
    C, T = 8, 128
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    torch.manual_seed(0)
    x = torch.randn(2, C, T)
    p = torch.randperm(C)
    with torch.no_grad():
        a = enc(x, meta)["logits"]
        b = enc(x[:, p], meta)["logits"]
    assert (a - b).abs().max().item() < 1e-4


def test_use_tare_false_still_means_no_channel_embedding():
    """The tokenizer-only ablation keeps its meaning; model/wast.yaml relies on it."""
    cfg = EncoderConfig(modality="eeg", use_tare=False)
    assert cfg.channel_embedding == "none"


def test_lattice_config_keeps_every_channel_time_token():
    """The variant's whole claim: only the scale axis is compressed.

    N_out must equal the number of patches the tokenizer was handed, and the
    reduction against legacy must be exactly J+1 -- the per-scale copies -- with
    nothing coming from the channel axis.
    """
    cfg = load_config("pretrain/semg_lattice")
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()
    assert enc.tare is None and enc.compressor is None and enc.spatial is None

    C, T = 16, 512
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    with torch.no_grad():
        out = enc(torch.randn(2, C, T), meta)
    st = out["token_stats"]
    assert st["N_out"] == st["N_patch_in"] == C * st["S"]
    assert st["scale_token_reduction"] == st["J"] + 1
    assert out["tokens"].shape[1] == C, "the channel axis must survive the backbone"


def test_lattice_config_still_tells_channels_apart():
    """No TARE and no channel_id here -- the backbone's slot embedding is the identity."""
    cfg = load_config("pretrain/semg_lattice")
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()
    C, T = 16, 512
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    torch.manual_seed(0)
    x = torch.randn(2, C, T)
    p = torch.randperm(C)
    with torch.no_grad():
        a, b = enc(x, meta)["logits"], enc(x[:, p], meta)["logits"]
    # Not just "nonzero": at the default pos_embed_init of 0.02 the slot
    # embedding is ~3% of the token magnitude and the model provably fails to
    # find it (see the table in configs/pretrain/semg_lattice.yaml). The
    # threshold is a floor on the config actually carrying channel identity,
    # not on the mechanism existing.
    moved = (a - b).abs().max().item() / a.std().item()
    assert moved > 0.05, f"channel identity is only {moved:.1%} of the logit scale"


def test_channel_reduction_mean_is_the_old_lossy_fallback():
    cfg = EncoderConfig(modality="semg", embed_dim=32, use_compression=False,
                        channel_reduction="mean", use_spatial_frontend=False,
                        channel_embedding="none", num_classes=4,
                        wast=WASTConfig(embed_dim=32, patch_size=32, level=2))
    enc = PhysioWaveEncoder(cfg).eval()
    with torch.no_grad():
        out = enc(torch.randn(2, 8, 128),
                  ChannelMeta([f"ch{i}" for i in range(8)], torch.zeros(8, 3)))
    assert out["tokens"].shape[1] == 1, "'mean' collapses the channel axis"


def test_wavelet_kv_shrinks_the_keys_and_not_the_tokens():
    from physiowave.models.backbone import BackboneConfig, FactorizedBackbone, WaveletKV

    x = torch.randn(2, 6, 16, 32)
    for level in (0, 1, 2):
        bb = FactorizedBackbone(BackboneConfig(embed_dim=32, depth=1, num_heads=4,
                                               slot_heads=2, kv_wavelet_level=level)).eval()
        with torch.no_grad():
            y = bb(x)
        assert y.shape == x.shape, f"level {level} changed the token lattice"

    kv = WaveletKV(32, 2).eval()
    with torch.no_grad():
        out, stride = kv(torch.randn(1, 16, 32))
    assert out.shape[1] == 4 and stride == 4.0


def test_wavelet_kv_reports_the_stride_it_achieved():
    """An odd length stops the halving early; the stride must follow, not assume."""
    from physiowave.models.backbone import WaveletKV

    kv = WaveletKV(16, 2).eval()
    with torch.no_grad():
        out, stride = kv(torch.randn(1, 6, 16))       # 6 -> 3, then 3 is odd
    assert out.shape[1] == 3 and stride == 2.0
    with torch.no_grad():
        out, stride = kv(torch.randn(1, 5, 16))       # cannot halve at all
    assert out.shape[1] == 5 and stride == 1.0


def test_pretrain_objective_uses_the_ssl_anchor(meta_64):
    enc = PhysioWaveEncoder(_cfg())
    obj = PretrainObjective(PretrainObjectiveConfig())
    out = obj(enc, torch.randn(2, 64, 256), meta_64)
    assert out["ref_anchor"] == "ssl", "a dense montage should use the SSL anchor"
    for term in ("loss_masked_raw", "loss_wavelet", "loss_reference_consistency",
                 "loss_query_specialization", "loss_covariance", "loss_total"):
        assert term in out["logs"] and torch.isfinite(torch.tensor(out["logs"][term]))
    out["loss"].backward()
    assert enc.wast.wt.dec_lo.grad.abs().max() > 0


def test_pretrain_objective_falls_back_when_ssl_is_unavailable(caplog):
    """A sparse montage disables SSL, so the anchor must fall back to pairwise."""
    names, xyz = montage("standard_1020_19")
    meta = ChannelMeta(names[:12], xyz[:12])              # 12 < min_channels = 16
    enc = PhysioWaveEncoder(_cfg())
    obj = PretrainObjective(PretrainObjectiveConfig())
    with caplog.at_level("WARNING"):
        out = obj(enc, torch.randn(2, 12, 256), meta)
    assert out["ref_anchor"] == "pairwise_fallback"
    assert out["logs"].get("ref_anchor_fallback") == 1.0
    assert "falling back to pairwise" in caplog.text


def test_loss_terms_can_be_switched_off_individually(meta_64):
    enc = PhysioWaveEncoder(_cfg())
    cfg = PretrainObjectiveConfig(use_wavelet=False, use_covariance=False,
                                  use_reference_consistency=False)
    out = PretrainObjective(cfg)(enc, torch.randn(2, 64, 256), meta_64)
    assert "loss_wavelet" not in out["logs"]
    assert "loss_covariance" not in out["logs"]
    assert "loss_reference_consistency" not in out["logs"]
    assert "loss_masked_raw" in out["logs"]


def test_config_system_composes_and_rejects_typos():
    cfg = load_config("pretrain/eeg")
    assert cfg["model"]["name"] == "wast_tare"
    assert cfg["model"]["spatial"]["ssl"]["enabled"] is True
    assert cfg["pretrain"]["ref_consistency"]["anchor"] == "ssl"
    cfg2 = load_config("pretrain/eeg", ["model.compression.num_queries=8",
                                        "train.epochs=3"])
    assert cfg2["model"]["compression"]["num_queries"] == 8 and cfg2["train"]["epochs"] == 3
    with pytest.raises(ValueError, match="Unknown config keys"):
        instantiate(WASTConfig, {"patch_size": 64, "not_a_key": 1})


@pytest.mark.parametrize("name", ["model/wast", "model/wast_tare", "pretrain/eeg",
                                  "pretrain/ecg", "pretrain/semg",
                                  "pretrain/semg_lattice"])
def test_every_config_builds_and_runs(name):
    cfg = load_config(name, ["model.backbone.depth=1", "model.embed_dim=32",
                             "model.wast.patch_size=32", "model.wast.embed_dim=32",
                             "model.backbone.num_heads=4", "model.backbone.slot_heads=2",
                             "model.compression.num_heads=2"])
    model = build_model(cfg).eval()
    C = 12 if cfg["model"].get("modality") == "ecg" else (
        8 if cfg["model"].get("modality") == "semg" else 19)
    with torch.no_grad():
        out = model(torch.randn(2, C, 256))
    assert out["pooled"].shape == (2, 32)
    assert count_parameters(model)["total"] > 0


def test_smoke_pretrain_and_resume(tmp_path):
    """The CPU smoke path: train, checkpoint, resume, and keep improving."""
    from physiowave.train.pretrain_main import main as pretrain_main

    out_dir = str(tmp_path / "run")
    common = ["--config", "pretrain/eeg", "--output-dir", out_dir, "--set",
              "train.batch_size=2", "train.num_workers=0",
              "data.synthetic.num_samples=8", "data.synthetic.window_samples=256",
              "data.synthetic.montage_name=standard_1020_19",
              "model.backbone.depth=1", "model.embed_dim=32",
              "model.wast.patch_size=32", "model.wast.embed_dim=32",
              "model.backbone.num_heads=4", "model.backbone.slot_heads=2",
              "model.compression.num_heads=2", "train.log_every=100"]
    assert pretrain_main(common + ["train.epochs=1"]) == 0
    assert os.path.exists(os.path.join(out_dir, "latest.pth"))
    assert os.path.exists(os.path.join(out_dir, "config_resolved.yaml"))
    assert os.path.exists(os.path.join(out_dir, "environment.json"))

    assert pretrain_main(common + ["train.epochs=2"] + ["--resume", "auto"]) == 0
    with open(os.path.join(out_dir, "history.json")) as f:
        history = json.load(f)
    assert [h["epoch"] for h in history] == [1], "resume did not continue from epoch 1"


def test_dry_run_validates_without_training(tmp_path, capsys):
    from physiowave.train.pretrain_main import main as pretrain_main

    assert pretrain_main(["--config", "pretrain/eeg", "--dry-run",
                          "--output-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "[dry-run] model=" in printed and "git_commit" in printed


def test_token_benchmark_runs(tmp_path):
    from physiowave.train.benchmark import benchmark_variant

    cfg = load_config("model/wast_tare", ["model.backbone.depth=1", "model.embed_dim=32",
                                          "model.wast.patch_size=32", "model.wast.embed_dim=32",
                                          "model.backbone.num_heads=4",
                                          "model.backbone.slot_heads=2",
                                          "model.compression.num_heads=2",
                                          "model.compression.num_queries=8"])
    row = benchmark_variant(cfg, C=19, T=256, batch_size=2, warmup=1, iters=2)
    assert row["tokens"] == 8 * 8 and row["token_compression_ratio"] > 1
    assert row["params"] > 0 and row["samples_per_sec"] > 0


def test_lattice_mean_pooling_bypasses_the_summary_attention():
    """Summary tokens are still produced, but they must not reach the logits.

    The point of the option is that the head sees an unweighted average over the
    lattice, not a learned selection from it -- so if perturbing the summary
    query moved the logits, the option would not be doing its job.
    """
    cfg = load_config("pretrain/semg_lattice", ["model.pooling=lattice_mean"])
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()
    C, T = 16, 512
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    torch.manual_seed(0)
    x = torch.randn(2, C, T)
    with torch.no_grad():
        before = enc(x, meta)["logits"].clone()
        enc.summary_query.add_(torch.randn_like(enc.summary_query))
        after = enc(x, meta)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


def test_summary_pooling_does_depend_on_the_query():
    cfg = load_config("pretrain/semg_lattice", ["model.pooling=mean"])
    cfg["model"]["num_classes"] = 5
    enc = build_model(cfg).eval()
    C, T = 16, 512
    meta = ChannelMeta([f"ch{i:02d}" for i in range(C)], torch.zeros(C, 3))
    torch.manual_seed(0)
    x = torch.randn(2, C, T)
    with torch.no_grad():
        before = enc(x, meta)["logits"].clone()
        enc.summary_query.add_(torch.randn_like(enc.summary_query))
        after = enc(x, meta)["logits"]
    assert not torch.allclose(before, after, atol=1e-5)


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        EncoderConfig(pooling="attention")
