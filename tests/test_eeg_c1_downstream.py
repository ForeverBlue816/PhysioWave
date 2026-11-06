"""Fine-tuning a C1 pretrained encoder.

The thing being guarded is that a downstream number means what it says. Three
ways it silently would not have:

  * the export dropped channel_token_gate, so the C1 mechanism was switched off
    in the experiment that exists to measure it;
  * a shape disagreement loaded nothing, trained from scratch, and logged
    "pretrained";
  * the legacy key migration renamed patch_embed out of existence on every
    save/load round trip, so the test metrics were computed with a freshly
    initialised patcher.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import channel_ids_for, vocab_payload   # noqa: E402
from physiowave.eeg_c1.downstream import (DEFAULT_WAVELETS,    # noqa: E402
                                          EEGC1Downstream)
from physiowave.eeg_c1.model import MultiRouteEEGPretrainer    # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                    # noqa: E402

SMALL = dict(embed_dim=64, depth=2, num_heads=4, channel_embed_dim=16,
             dropout=0.0, wavelet_names=list(DEFAULT_WAVELETS))


def _channel_set(name):
    src = open(os.path.join(ROOT, "EEG", "physio_p300_finetune.py")).read()
    body = re.search(rf"^{name} = \[(.*?)\]", src, re.S | re.M).group(1)
    return re.findall(r"['\"]([^'\"]+)['\"]", body)


def p300_channels():
    # CHANNELS_62, not CHANNELS_58: 62 is the converter's default and is the
    # montage these runs are actually handed. It is also the stronger test --
    # it fills every E64_256 slot the EDF can fill, so a placement bug has
    # only TP9/TP10 left to hide in.
    return _channel_set("CHANNELS_62")


class _Meta:
    def __init__(self, names):
        self.channel_names = names
        self.channel_mask = None


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("pre")
    pre = MultiRouteEEGPretrainer(**SMALL)
    with torch.no_grad():
        pre.channel_token_gate.fill_(0.33)
    ck = d / "pre.pth"
    torch.save({"model": pre.state_dict(), "config": {"model": {}},
                "epoch": 4, "global_step": 1920}, ck)
    out = {}
    for route in ("E64_256", "E19_256"):
        p = d / f"enc_{route}.pth"
        r = subprocess.run(
            [sys.executable, "scripts/export_eeg_pretrained_encoder.py",
             "--checkpoint", str(ck), "--route", route, "--output", str(p)],
            cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        out[route] = str(p)
    # The same checkpoint with no --route: what both downstream tasks actually
    # consume, since neither declares a route_id.
    p = d / "enc_noroute.pth"
    r = subprocess.run(
        [sys.executable, "scripts/export_eeg_pretrained_encoder.py",
         "--checkpoint", str(ck), "--output", str(p)],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out["none"] = str(p)
    out["gate"] = 0.33
    return out


# --- placement -------------------------------------------------------------- #
def test_the_default_p300_montage_is_62_slots_of_e64_256():
    """Every electrode erpbci records that the route has a slot for.

    The two it leaves empty are TP9 and TP10, and that is not an oversight:
    the EDF has P9/P10, one position higher in the same inferior chain. If
    this list ever drifts off the route the placement below stops meaning
    anything, because a name that is not a slot is refused, not dropped.
    """
    ch = p300_channels()
    slots = list(ROUTES["E64_256"].slots)
    assert len(ch) == 62
    assert [c for c in ch if c not in slots] == []
    assert [c for c in slots if c not in ch] == ["TP9", "TP10"]
    # In the ROUTE's order, so the HDF5 and the slot array read alike.
    assert [c for c in slots if c in ch] == ch
    # And all 62 are electrodes the EDF actually records.
    assert [c for c in ch if c not in _channel_set("EEG_64")] == []


def test_a_montage_that_is_a_subset_of_a_route_goes_in_its_slots():
    ch = p300_channels()
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **SMALL)
    r = ROUTES["E64_256"]
    assert int((m.slot_index >= 0).sum()) == len(ch)

    # Every channel lands on the slot with its own name, and nowhere else.
    from channel_embedding import normalize_channel_name
    for slot_i, src_i in enumerate(m.slot_index.tolist()):
        if src_i >= 0:
            assert normalize_channel_name(ch[src_i]) == \
                normalize_channel_name(r.slots[slot_i])

    # The absent slots carry zero, not whatever was next in the array.
    x = torch.randn(2, len(ch), 512)
    placed = m._to_slots(x)
    assert placed.shape == (2, r.n_channels, 512)
    absent = (m.slot_index < 0)
    assert float(placed[:, absent].abs().sum()) == 0.0
    for slot_i, src_i in enumerate(m.slot_index.tolist()):
        if src_i >= 0:
            assert torch.equal(placed[:, slot_i], x[:, src_i])


def test_a_channel_the_route_does_not_have_is_refused_not_dropped():
    with pytest.raises(ValueError, match="not slots of"):
        EEGC1Downstream(in_channels=2, window_samples=512, sampling_rate=256,
                        patch_samples=128, num_classes=2,
                        channel_names=["Fpz-Cz", "Pz-Oz"],
                        route_id="E64_256", **SMALL)


def test_a_montage_off_every_route_gets_its_own_frontend():
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    assert m.slot_index is None
    assert m.route.n_channels == 2 and m.route.patch_t == 50
    out = m(torch.randn(2, 2, 3000), _Meta(["Fpz-Cz", "Pz-Oz"]))
    assert out["logits"].shape == (2, 5)
    assert out["features"].shape == (2, SMALL["embed_dim"])


# --- what transfers --------------------------------------------------------- #
def test_a_matching_route_transfers_the_frontend_and_the_patcher(exported):
    ch = p300_channels()
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **SMALL)
    rep = m.load_pretrained(exported["E64_256"])
    for prefix in ("shared_transformer.", "wavelet_frontend.", "patch_embed.",
                   "channel_encoder."):
        assert any(k.startswith(prefix) for k in rep["taken"]), prefix
    assert "channel_token_gate" in rep["taken"], \
        "the gate scales the whole C1 contribution and initialises to zero"
    assert float(m.channel_token_gate.detach()) == pytest.approx(exported["gate"])


def test_a_different_montage_transfers_the_transformer_and_says_so(exported):
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    rep = m.load_pretrained(exported["E19_256"])
    assert any(k.startswith("shared_transformer.") for k in rep["taken"])
    assert "channel_token_gate" in rep["taken"]
    # The frontend is per-electrode and the patcher per patch length: neither
    # is transferable here, and pretending otherwise is the failure.
    assert not any(k.startswith("wavelet_frontend.") for k in rep["taken"])
    assert not any(k.startswith("patch_embed.") for k in rep["taken"])
    assert "loaded: transformer" in m.describe_transfer(rep)


def test_an_export_without_a_route_carries_only_what_transfers(exported):
    """--route is optional, and omitting it is the case both tasks are in.

    A route-bound export fed to a task that builds its own frontend ships a
    frontend and a patcher that load_pretrained then skips. The file is bigger
    for nothing, and it names a route the weights will not be used at, which
    reads as a constraint on the montage that does not exist.
    """
    ck = torch.load(exported["none"], map_location="cpu", weights_only=False)
    assert ck["route_id"] is None
    assert ck["route"] is None
    keys = set(ck["model"])
    assert any(k.startswith("shared_transformer.") for k in keys)
    assert any(k.startswith("channel_encoder.") for k in keys)
    assert "channel_token_gate" in keys
    assert not any(k.startswith("wavelet_frontend.") for k in keys)
    assert not any(k.startswith("patch_embed.") for k in keys)
    # And it is genuinely smaller, which is the other half of the point.
    assert os.path.getsize(exported["none"]) < \
        os.path.getsize(exported["E64_256"])


def test_a_routeless_export_transfers_exactly_what_a_routed_one_does(exported):
    """The bytes a montage-mismatched task actually receives are identical.

    If dropping the frontend changed what lands, the smaller file would be a
    different model rather than the same one without its unused half.
    """
    kw = dict(in_channels=2, window_samples=3000, sampling_rate=100,
              patch_samples=50, num_classes=5,
              channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    torch.manual_seed(0)
    a = EEGC1Downstream(**kw)
    torch.manual_seed(0)
    b = EEGC1Downstream(**kw)
    rep_routed = a.load_pretrained(exported["E19_256"])
    rep_bare = b.load_pretrained(exported["none"])

    assert set(rep_routed["taken"]) == set(rep_bare["taken"])
    assert any(k.startswith("shared_transformer.") for k in rep_bare["taken"])
    sd_a, sd_b = a.state_dict(), b.state_dict()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k


def _routeless(**kw):
    return EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                           patch_samples=50, num_classes=5,
                           channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL, **kw)


def test_a_pretrained_scope_probe_trains_the_fresh_frontend_not_the_encoder(exported):
    """freeze_scope=pretrained freezes what was LOADED, nothing more.

    With no route the frontend and patcher are built fresh. The original
    freeze_encoder froze them too, at their initialisation, which puts a random
    projection in front of the pretrained transformer -- a probe of that
    projection as much as of the representation.
    """
    m = _routeless(freeze_encoder=True, freeze_scope="pretrained")
    m.load_pretrained(exported["none"])
    grad = {n: p.requires_grad for n, p in m.named_parameters()}
    assert all(v for n, v in grad.items() if n.startswith("wavelet_frontend."))
    assert all(v for n, v in grad.items() if n.startswith("patch_embed."))
    assert all(v for n, v in grad.items() if n.startswith("head"))
    assert not any(v for n, v in grad.items() if n.startswith("shared_transformer."))
    assert not any(v for n, v in grad.items() if n.startswith("channel_encoder."))
    assert not grad["channel_token_gate"]

    m.train()
    assert not m.shared_transformer.training, "frozen encoder must stay in eval"
    assert m.wavelet_frontend.training, "the fresh frontend is being trained"


def test_the_default_scope_still_freezes_everything_but_the_head(exported):
    """P300 and Sleep-EDF probes were run under this; it must not move."""
    m = _routeless(freeze_encoder=True)
    m.load_pretrained(exported["none"])
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("head", "spatial_filter"))
                             for n in trainable)


def test_a_pretrained_scope_freezes_the_frontend_when_it_was_loaded(exported):
    """On the matching route the frontend IS pretrained, so it is frozen too."""
    ch = p300_channels()
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256",
                        freeze_encoder=True, freeze_scope="pretrained", **SMALL)
    rep = m.load_pretrained(exported["E64_256"])
    assert any(k.startswith("wavelet_frontend.") for k in rep["taken"])
    assert not any(p.requires_grad for n, p in m.named_parameters()
                   if n.startswith(("wavelet_frontend.", "patch_embed.")))


def test_one_probe_step_moves_the_frontend_and_leaves_the_encoder(exported):
    """Behaviour, not flags: after an optimiser step, compare the tensors."""
    torch.manual_seed(0)
    m = _routeless(freeze_encoder=True, freeze_scope="pretrained")
    m.load_pretrained(exported["none"])
    before = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    m.train()
    out = m(torch.randn(3, 2, 3000), _Meta(["Fpz-Cz", "Pz-Oz"]))
    torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1, 2])).backward()
    opt.step()
    after = m.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert any(k.startswith("wavelet_frontend.") for k in moved)
    assert any(k.startswith("head") for k in moved)
    assert not any(k.startswith(("shared_transformer.", "channel_encoder."))
                   for k in moved)
    assert "channel_token_gate" not in moved


def test_an_unknown_freeze_scope_is_refused():
    with pytest.raises(ValueError, match="freeze_scope"):
        _routeless(freeze_encoder=True, freeze_scope="everything")


def test_the_weights_actually_land(exported):
    """A report that says "taken" and a tensor that did not move is the bug."""
    ch = p300_channels()
    kw = dict(in_channels=len(ch), window_samples=512, sampling_rate=256,
              patch_samples=128, num_classes=2, channel_names=ch,
              route_id="E64_256", **SMALL)
    torch.manual_seed(0)
    fresh = EEGC1Downstream(**kw)
    torch.manual_seed(0)
    loaded = EEGC1Downstream(**kw)
    loaded.load_pretrained(exported["E64_256"])

    sd_f, sd_l = fresh.state_dict(), loaded.state_dict()
    moved = [k for k in sd_f
             if sd_f[k].shape == sd_l[k].shape and not torch.equal(sd_f[k], sd_l[k])]
    assert any(k.startswith("shared_transformer.") for k in moved)
    assert "channel_token_gate" in moved
    # And the head, which nothing supplies, did not.
    assert not any(k.startswith("head.") for k in moved)


# --- the ways it could lie --------------------------------------------------- #
def test_a_shape_disagreement_is_refused_not_ignored(exported, tmp_path):
    ch = p300_channels()
    wrong = dict(SMALL, embed_dim=128, num_heads=4)
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **wrong)
    with pytest.raises(SystemExit, match="do not fit"):
        m.load_pretrained(exported["E64_256"])


def test_a_checkpoint_with_no_transformer_is_refused(tmp_path):
    bad = tmp_path / "bad.pth"
    torch.save({"model": {"head.weight": torch.zeros(2, 64)}}, bad)
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    with pytest.raises(SystemExit, match="no transformer weights"):
        m.load_pretrained(str(bad))


def test_a_different_channel_vocabulary_is_refused(exported, tmp_path):
    ck = torch.load(exported["E19_256"], map_location="cpu", weights_only=False)
    ck["channel_vocab_sha256"] = "0" * 64
    p = tmp_path / "relabelled.pth"
    torch.save(ck, p)
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    with pytest.raises(SystemExit, match="different electrode"):
        m.load_pretrained(str(p))


def test_the_legacy_migration_does_not_rename_a_patcher_this_model_owns():
    """patch_embed. -> legacy_patch_embed. was unconditional.

    So the C1 downstream encoder's patcher was renamed out of existence on
    every save/load round trip: reported as remapped, loaded as missing, and
    the test metrics computed with a freshly initialised patcher while the log
    said the checkpoint had loaded.
    """
    from physiowave.models.checkpoint import migrate_state_dict

    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    out, report = migrate_state_dict(m.state_dict(), m)
    assert not report.missing, f"a round trip lost {report.missing[:6]}"
    assert not report.remapped, f"a round trip renamed {report.remapped[:6]}"
    assert any(k.startswith("patch_embed.") for k in out)


# --- encoder_lr_scale -------------------------------------------------------- #
def test_encoder_lr_scale_gives_pretrained_parameters_their_own_rate():
    """The TRANSFERABLE parameters get lr x scale, everything else lr, and the
    weight-decay split is the same as without the scale."""
    from physiowave.eeg_c1.downstream import TRANSFERABLE
    from physiowave.train.utils import build_optimizer, build_scheduler

    m = _routeless()
    base = build_optimizer(m, 1e-3, 0.05)
    opt = build_optimizer(m, 1e-3, 0.05,
                          lr_scale=lambda n: 0.1 if n.startswith(TRANSFERABLE) else 1.0)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    wd_of = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    wd_base = {id(p): g["weight_decay"] for g in base.param_groups for p in g["params"]}
    for name, p in m.named_parameters():
        want = 1e-4 if name.startswith(TRANSFERABLE) else 1e-3
        assert lr_of[id(p)] == pytest.approx(want), name
        assert wd_of[id(p)] == wd_base[id(p)], name        # decay rules unchanged

    # The schedule scales every group by the same factor: halfway through a
    # cosine with no warmup, both are at the same fraction of their own peak.
    sched = build_scheduler(opt, 0, 100, 0.0)
    for _ in range(50):
        opt.step(); sched.step()
    frac = {round(g["lr"] / g["initial_lr"], 6) for g in opt.param_groups}
    assert len(frac) == 1 and abs(frac.pop() - 0.5) < 0.02
