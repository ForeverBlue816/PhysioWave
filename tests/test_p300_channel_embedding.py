"""The channel embedding on erpbci's monopolar montage.

Sleep-EDF is two bipolar derivations; erpbci is 62 electrodes against a common
reference (the converter's default; --channels 58 writes EEGPT's set instead). The same encoder has to describe both without pretending either is
the other, and the property that separates them is the one tested hardest here:
a monopolar channel has a position and NO direction, and the code must say so
rather than invent a reference electrode to subtract.
"""

from __future__ import annotations

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
    CHANNEL_TO_ID, CHANNEL_VOCAB, PAD_ID, UNK_ID, ChannelEncoder, channel_id)
from model import BERTWaveletTransformer                           # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "physio_p300_finetune", os.path.join(_HERE, "EEG", "physio_p300_finetune.py"))
p300 = importlib.util.module_from_spec(_SPEC)
sys.modules["physio_p300_finetune"] = p300
_SPEC.loader.exec_module(p300)

mne = pytest.importorskip("mne", reason="coordinates come from MNE at prep time")

VARIANTS = [("C0", "none", "none"), ("C1", "id", "token"),
            ("C2", "signed", "token"), ("C3", "signed", "fold"),
            ("C4", "signed", "dual"), ("C5", "hybrid", "dual")]


# --------------------------------------------------------------------------- #
# 1-3: the vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attr", ["CHANNELS_58", "CHANNELS_62", "CHANNELS_64"])
def test_every_p300_channel_has_its_own_id(attr):
    """One distinct id per electrode, none of them UNK.

    Left as UNK they would all share one embedding row, and `id` would encode
    "some channel" 62 times over -- a null result that looked like a
    measurement. Every set the converter can write is checked, not just the
    default: --channels picks between them at prep time.
    """
    chans = getattr(p300, attr)
    ids = [channel_id(c) for c in chans]
    assert UNK_ID not in ids, [c for c, i in zip(chans, ids) if i == UNK_ID]
    assert len(set(ids)) == len(chans)


def test_sleep_ids_did_not_move_when_p300_was_appended():
    """The vocabulary is append-only: ids live in every HDF5 and checkpoint.

    Inserting the monopolar names anywhere but the end would silently relabel
    every channel of every file written before.
    """
    assert CHANNEL_VOCAB[PAD_ID] == "<pad>" and CHANNEL_VOCAB[UNK_ID] == "<unk>"
    assert channel_id("Fpz-Cz") == 2 and channel_id("Pz-Oz") == 3


def test_an_electrode_and_a_derivation_are_different_words():
    """"Cz" and "Fz-Cz" must not collide.

    One is a potential at a site, the other a difference of two. A shared id
    would tell the model they are the same measurement.
    """
    assert channel_id("Cz") != channel_id("Fz-Cz")
    assert channel_id("Cz") != UNK_ID and channel_id("Fz-Cz") != UNK_ID
    assert len(CHANNEL_VOCAB) == len(set(CHANNEL_VOCAB))


# --------------------------------------------------------------------------- #
# 4-7: the metadata the preparation writes
# --------------------------------------------------------------------------- #
def test_metadata_describes_a_monopolar_montage():
    chans = p300.CHANNELS_62
    n = len(chans)
    b = p300.build_channel_metadata(chans)
    d, a = b["datasets"], b["attrs"]
    assert a["derivation_type"] == "monopolar_common_reference"
    # Derived from the montage it was handed, not from a constant that would
    # go on saying 58 after --channels changed.
    assert a["montage_type"] == f"erpbci_{n}"
    assert d["electrode_xyz"].shape == (n, 3)
    # Every channel is its own electrode: the encoder reads equal endpoints as
    # "position, no direction".
    assert (d["positive_electrode_index"] == d["negative_electrode_index"]).all()
    assert (d["positive_electrode_index"] == np.arange(n)).all()
    assert np.array_equal(d["derivation_matrix"], np.eye(n, dtype=np.float32))
    assert [c.decode() for c in d["channel_names"]] == chans


def test_metadata_carries_no_bipolar_endpoints():
    """Absence is the honest value for a montage that has no electrode pairs."""
    b = p300.build_channel_metadata(p300.CHANNELS_62)
    assert "bipolar_endpoints" not in b["datasets"]


def test_metadata_hash_is_deterministic_and_montage_specific():
    a = p300.build_channel_metadata(p300.CHANNELS_62)["attrs"]["metadata_hash"]
    b = p300.build_channel_metadata(p300.CHANNELS_62)["attrs"]["metadata_hash"]
    assert a == b
    # A different montage is a different file, so it must hash differently.
    assert p300.build_channel_metadata(
        p300.CHANNELS_58)["attrs"]["metadata_hash"] != a
    shuffled = list(p300.CHANNELS_62)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    c = p300.build_channel_metadata(shuffled)["attrs"]["metadata_hash"]
    assert c != a, "swapping two rows must change the hash, or a reordered file passes"


def test_an_unknown_channel_is_refused_not_silently_unked():
    with pytest.raises(SystemExit, match="not in the vocabulary"):
        p300.build_channel_metadata(["Fp1", "NotAnElectrode"])


# --------------------------------------------------------------------------- #
# 8-11: the encoder on this montage
# --------------------------------------------------------------------------- #
N_DEFAULT = 62          # p300.CHANNELS_62, the converter's default montage


def _meta(n=N_DEFAULT):
    b = p300.build_channel_metadata(p300.CHANNELS_62[:n])["datasets"]
    return {
        "channel_ids": torch.as_tensor(b["channel_ids"]).long(),
        "electrode_xyz": torch.as_tensor(b["electrode_xyz"]).float(),
        "positive_electrode_index": torch.as_tensor(b["positive_electrode_index"]).long(),
        "negative_electrode_index": torch.as_tensor(b["negative_electrode_index"]).long(),
        "valid_channel_mask": torch.as_tensor(b["valid_channel_mask"]),
    }


def test_direction_branch_contributes_exactly_zero_on_a_monopolar_montage():
    """Not approximately: pa - pb with pa is pb is an exact zero.

    So dir_proj cannot influence the code however it is weighted, which is what
    "this montage has no direction" has to mean in the arithmetic.
    """
    enc = ChannelEncoder("signed", 32)
    enc.reset_channel_parameters()
    meta = _meta()
    with torch.no_grad():
        before = enc(meta).clone()
        enc.dir_proj.weight.mul_(1000.0)          # would dominate, if it applied
        after = enc(meta)
    assert torch.equal(before, after)


def test_the_monopolar_marker_is_the_one_used():
    """A monopolar channel takes monopolar_token; a bipolar one takes the other."""
    enc = ChannelEncoder("signed", 16)
    enc.reset_channel_parameters()
    with torch.no_grad():
        enc.monopolar_token.fill_(50.0)
        enc.bipolar_token.fill_(-50.0)
    mono = enc(_meta(4))
    bip = dict(_meta(4))
    bip["negative_electrode_index"] = torch.tensor([1, 0, 3, 2])   # pairs, not sites
    assert (mono.mean(dim=-1) > 0).all(), "monopolar channels took the wrong marker"
    assert (enc(bip).mean(dim=-1) < 0).all(), "bipolar channels took the wrong marker"


def test_both_markers_start_at_zero():
    """Which is why adding monopolar_token left every bipolar model unchanged.

    Neither draws from the global RNG either, so the legacy initialisation of
    the models already trained is untouched.
    """
    enc = ChannelEncoder("signed", 16)
    enc.reset_channel_parameters()
    assert float(enc.bipolar_token.detach().abs().max()) == 0.0
    assert float(enc.monopolar_token.detach().abs().max()) == 0.0


def test_distinct_electrodes_get_distinct_codes():
    """A position encoding that mapped two sites to one code would encode nothing."""
    enc = ChannelEncoder("signed", 32)
    enc.reset_channel_parameters()
    code = enc(_meta())
    pair = code @ code.T
    off = pair - torch.eye(N_DEFAULT) * pair.diag()
    assert not torch.isclose(code[0], code[1], atol=1e-6).all()
    assert torch.isfinite(off).all()


# --------------------------------------------------------------------------- #
# 12-13: the model. n=12, not the montage's 62: this is a forward/backward
# smoke test and the code path does not depend on the count.
# --------------------------------------------------------------------------- #
def _model(encoding, injection, seed=0, n=12):
    torch.manual_seed(seed)
    return BERTWaveletTransformer(
        in_channels=n, max_level=2, wave_kernel_size=16,
        wavelet_names=["sym4"], use_separate_channel=True, wave_init_mode="pad",
        patch_size=(1, 64), embed_dim=64, depth=2, num_heads=4, mlp_ratio=4.0,
        dropout=0.0, norm="rmsnorm", ffn="swiglu", qk_norm=True,
        scale_fold="dynamic", fold_synthesis=3, fold_gamma=0.1,
        use_pos_embed=True, pos_embed_type="2d",
        channel_encoding=encoding, channel_injection=injection,
        channel_embed_dim=32, channel_vocab_size=len(CHANNEL_VOCAB),
        task_type="classification", num_classes=2,
        head_config={"hidden_dims": [32], "dropout": 0.0, "pooling": "mean"},
        pooling="mean")


@pytest.mark.parametrize("name,enc,inj", VARIANTS)
def test_forward_and_backward_on_a_monopolar_montage(name, enc, inj):
    n = 12                                # the real montage is slow and no different
    model = _model(enc, inj, n=n)
    meta = None if enc == "none" else _meta(n)
    x = torch.randn(2, n, 256)
    logits = model(x, task="classify", channel_meta=meta)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name,enc,inj", VARIANTS)
def test_gates_start_closed_so_every_variant_starts_as_the_baseline(name, enc, inj):
    """|Delta logits| = 0 against C0, exactly, not to a tolerance."""
    n = 12
    x = torch.randn(2, n, 256)
    base = _model("none", "none", n=n).eval()
    var = _model(enc, inj, n=n).eval()
    meta = None if enc == "none" else _meta(n)
    with torch.no_grad():
        a = base(x, task="classify")
        b = var(x, task="classify", channel_meta=meta)
    assert float((a - b).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 14: the HDF5 round trip, which is where the schema mismatch actually showed
# --------------------------------------------------------------------------- #
def test_written_hdf5_is_readable_by_the_trainer(tmp_path):
    """Write a split the way the preparation does, read it the way finetune does.

    This is the test that catches a schema disagreement between the two ends.
    It caught one: the reader required `bipolar_endpoints`, which a monopolar
    montage has no honest value for, so every P300 file would have raised
    KeyError on a field that should not have been there in the first place.
    """
    import h5py

    _spec = importlib.util.spec_from_file_location(
        "finetune", os.path.join(_HERE, "finetune.py"))
    ft = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("finetune", ft)
    _spec.loader.exec_module(ft)

    channels = p300.CHANNELS_62
    bundle = p300.build_channel_metadata(channels)
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((4, len(channels), 512), np.float32))
        f.create_dataset("label", data=np.array([0, 1, 0, 1], np.int64))
        f.create_dataset("channel_names",
                         data=np.array([c.encode() for c in channels], dtype="S32"))
        for k, v in bundle["datasets"].items():
            if k != "channel_names":
                f.create_dataset(k, data=v)
        for k, v in bundle["attrs"].items():
            f.attrs[k] = v

    meta = ft.read_channel_metadata(str(path))
    assert meta is not None
    assert meta["_channel_names"] == channels
    assert meta["_attrs"]["metadata_hash"] == bundle["attrs"]["metadata_hash"]
    assert meta["_attrs"]["derivation_type"] == "monopolar_common_reference"
    assert (meta["positive_electrode_index"] == meta["negative_electrode_index"]).all()
    assert "bipolar_endpoints" not in meta

    # Two files written from the same montage must compare equal, and a file
    # with a different montage must not.
    other = tmp_path / "val.h5"
    with h5py.File(path, "r") as src, h5py.File(other, "w") as dst:
        for k in src:
            dst.create_dataset(k, data=src[k][:])
        for k in src.attrs:
            dst.attrs[k] = src.attrs[k]
    assert ft._meta_signature(meta) == ft._meta_signature(
        ft.read_channel_metadata(str(other)))

    swapped = list(channels)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    other2 = tmp_path / "test.h5"
    b2 = p300.build_channel_metadata(swapped)
    with h5py.File(other2, "w") as f:
        f.create_dataset("channel_names",
                         data=np.array([c.encode() for c in swapped], dtype="S32"))
        for k, v in b2["datasets"].items():
            if k != "channel_names":
                f.create_dataset(k, data=v)
        for k, v in b2["attrs"].items():
            f.attrs[k] = v
    assert ft._meta_signature(meta) != ft._meta_signature(
        ft.read_channel_metadata(str(other2)))
