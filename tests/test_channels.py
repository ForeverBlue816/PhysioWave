"""TARE channel encoding and topology-aware channel compression."""

from __future__ import annotations

import os

import pytest
import torch

from physiowave.channels.compression import ChannelCompressor, CompressionConfig
from physiowave.channels.tare import ChannelEncoder, ChannelMeta, TAREConfig


def _encoder(fusion="concat_mlp", dim=48):
    return ChannelEncoder(TAREConfig(embed_dim=dim, fusion_mode=fusion)).eval()


@pytest.mark.parametrize("fusion", ["concat_mlp", "film"])
def test_channel_embedding_shapes(fusion, montage_19):
    names, xyz = montage_19
    e = _encoder(fusion)(ChannelMeta(names, xyz))
    assert e.shape == (len(names), 48) and torch.isfinite(e).all()


def test_coordinates_only(montage_19, caplog):
    """A montage with coordinates but no recognisable names still works.

    Names are auxiliary by design; the coordinate branch carries the spatial
    information, and unknown labels fall back to the learnable unknown-name slot.
    """
    names, xyz = montage_19
    enc = _encoder()
    named = enc(ChannelMeta(names, xyz))
    anon = enc(ChannelMeta(["???"] * len(names), xyz))
    assert torch.isfinite(anon).all()
    assert (named - anon).abs().max() > 1e-6, "names had no effect at all"
    # ...and the anonymous encoding must still distinguish electrodes by position.
    assert (anon[0] - anon[5]).abs().max() > 1e-4


def test_unknown_coordinates_fallback_warns(montage_19, caplog):
    names, xyz = montage_19
    bad = xyz.clone()
    bad[3] = 0.0
    enc = _encoder()
    with caplog.at_level("WARNING"):
        e = enc(ChannelMeta(names, bad))
    assert torch.isfinite(e).all()
    assert "unknown-coordinate fallback" in caplog.text


def test_unknown_names_stay_distinct():
    """A montage the template does not know must still have distinguishable channels.

    An sEMG ring has no template coordinates and no template labels, so the name
    branch is the only thing left that can tell one electrode from another. When
    every unrecognised label shared slot 0 the encoder returned sixteen identical
    embeddings and the model as a whole became invariant to permuting the channel
    axis -- which on a forearm array discards the signal, not a nuisance.
    """
    names = [f"ch{i:02d}" for i in range(16)]
    enc = _encoder()
    e = enc(ChannelMeta(names, torch.zeros(16, 3)))
    assert torch.isfinite(e).all()
    rows = {tuple(round(v, 6) for v in r.tolist()) for r in e}
    assert len(rows) == len(names), f"only {len(rows)} distinct embeddings for {len(names)} channels"


def test_unknown_name_slots_are_process_stable():
    """The hash must not move between processes: DDP ranks share one embedding table."""
    import subprocess
    import sys

    from physiowave.channels.tare import _name_index, build_name_vocab

    vocab = build_name_vocab()
    here = [_name_index(f"ch{i:02d}", vocab, 512) for i in range(8)]
    # A different PYTHONHASHSEED is what would move a built-in hash().
    out = subprocess.run(
        [sys.executable, "-c",
         "from physiowave.channels.tare import build_name_vocab, _name_index;"
         "v = build_name_vocab();"
         "print([_name_index(f'ch{i:02d}', v, 512) for i in range(8)])"],
        capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "12345"},
        check=True,
    )
    assert eval(out.stdout.strip()) == here


def test_known_labels_keep_their_template_slot(montage_19):
    """Hashing the unknowns must not disturb the labels the template does know."""
    from physiowave.channels.tare import _name_index, build_name_vocab

    vocab = build_name_vocab()
    names, _ = montage_19
    for n in names:
        assert _name_index(n, vocab, 512) == vocab[n], n


def test_permutation_equivariance(montage_19):
    """Permuting signal and metadata together permutes the embeddings exactly."""
    names, xyz = montage_19
    enc = _encoder()
    base = enc(ChannelMeta(names, xyz))
    p = torch.randperm(len(names))
    perm = enc(ChannelMeta([names[i] for i in p], xyz[p]))
    assert torch.allclose(perm, base[p], atol=1e-6)


def test_reference_schemes_are_distinguishable(montage_19):
    names, xyz = montage_19
    enc = _encoder()
    refs = ["original", "common_average", "linked_mastoids", "left_ear", "single_channel"]
    embs = [enc(ChannelMeta(names, xyz, reference_type=r)) for r in refs]
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            assert (embs[i] - embs[j]).abs().max() > 1e-5, f"{refs[i]} == {refs[j]}"


def test_reference_channel_position_matters(montage_19):
    """A left-mastoid and a right-mastoid reference must not encode identically."""
    names, xyz = montage_19
    enc = _encoder()
    left = enc(ChannelMeta(names, xyz, reference_type="single_channel", reference_channel="M1"))
    right = enc(ChannelMeta(names, xyz, reference_type="single_channel", reference_channel="M2"))
    assert (left - right).abs().max() > 1e-5


def test_bipolar_endpoint_order_matters():
    """`a-b` and `b-a` are opposite derivations and must encode differently."""
    enc = _encoder()
    xyz = torch.zeros(1, 3)
    ab = enc(ChannelMeta(["F7-T7"], xyz, derivation_type="bipolar",
                         bipolar_endpoints=[["F7", "T7"]]))
    ba = enc(ChannelMeta(["T7-F7"], xyz, derivation_type="bipolar",
                         bipolar_endpoints=[["T7", "F7"]]))
    assert (ab - ba).abs().max() > 1e-3, "swapping bipolar endpoints changed nothing"


def test_bipolar_is_not_the_midpoint():
    """Two different pairs sharing a midpoint must encode differently."""
    enc = _encoder()
    xyz = torch.zeros(1, 3)
    a = enc(ChannelMeta(["F7-P7"], xyz, derivation_type="bipolar",
                        bipolar_endpoints=[["F7", "P7"]]))
    b = enc(ChannelMeta(["FT7-TP7"], xyz, derivation_type="bipolar",
                        bipolar_endpoints=[["FT7", "TP7"]]))
    assert (a - b).abs().max() > 1e-3


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #
def _compressor(K=8, D=32):
    return ChannelCompressor(CompressionConfig(num_queries=K, embed_dim=D, num_heads=2))


def test_compression_shape_and_complexity(montage_19):
    names, xyz = montage_19
    C, S, D, K = len(names), 6, 32, 8
    m = _compressor(K, D)
    out = m(torch.randn(2, C, S, D), torch.randn(C, D), xyz)
    assert out["tokens"].shape == (2, K, S, D)
    assert out["attn"].shape == (2, S, K, C)


def test_missing_channels_get_zero_attention(montage_19):
    names, xyz = montage_19
    C, S, D, K = len(names), 5, 32, 8
    m = _compressor(K, D)
    mask = torch.ones(C, dtype=torch.bool)
    mask[[2, 9, 14]] = False
    out = m(torch.randn(2, C, S, D), torch.randn(C, D), xyz, channel_mask=mask)
    assert out["attn"][..., [2, 9, 14]].abs().max() == 0.0
    assert torch.allclose(out["attn"].sum(-1), torch.ones(2, S, K), atol=1e-5)


def test_missing_channel_values_do_not_leak(montage_19):
    """A masked channel's *values* must not influence the compressed tokens."""
    names, xyz = montage_19
    C, S, D = len(names), 4, 32
    m = _compressor(8, D).eval()
    tokens = torch.randn(2, C, S, D)
    mask = torch.ones(C, dtype=torch.bool)
    mask[4] = False
    ce = torch.randn(C, D)
    with torch.no_grad():
        a = m(tokens, ce, xyz, channel_mask=mask)["tokens"]
        tokens2 = tokens.clone()
        tokens2[:, 4] = 1e3 * torch.randn(2, S, D)
        b = m(tokens2, ce, xyz, channel_mask=mask)["tokens"]
    assert torch.allclose(a, b, atol=1e-5)


def test_graph_bias_is_detached(montage_19):
    names, xyz = montage_19
    C, S, D = len(names), 4, 32
    m = _compressor(8, D)
    A = torch.rand(2, C, C, requires_grad=True)
    out = m(torch.randn(2, C, S, D), torch.randn(C, D), xyz, relation_graph=A)
    out["tokens"].sum().backward()
    assert A.grad is None, "the statistics graph must not receive gradient"


def test_query_specialization_loss_penalises_collapse(montage_19):
    names, xyz = montage_19
    m = _compressor(8, 32)
    with torch.no_grad():
        m.anchors.copy_(torch.zeros_like(m.anchors))      # fully collapsed anchors
    collapsed = m.query_specialization_loss(torch.ones(2, 4, 8, len(names)) / len(names))
    with torch.no_grad():
        spread = torch.randn_like(m.anchors)
        m.anchors.copy_(spread / spread.norm(dim=-1, keepdim=True) * 2.0)
    attn = torch.eye(8, len(names)).unsqueeze(0).unsqueeze(0).expand(2, 4, -1, -1)
    diverse = m.query_specialization_loss(attn)
    assert collapsed > diverse, f"collapse loss {collapsed} <= diverse loss {diverse}"


@pytest.mark.parametrize("K", [4, 8, 16, 32])
def test_all_configured_K(K, montage_19):
    names, xyz = montage_19
    m = _compressor(K, 32)
    out = m(torch.randn(2, len(names), 4, 32), torch.randn(len(names), 32), xyz)
    assert out["tokens"].shape == (2, K, 4, 32)


def test_tare_component_switches(montage_19):
    """Each ablation rung turns exactly one component on.

    With coordinates off, channels may only be told apart by name; with names and
    reference metadata off, only geometry remains.  Both must still produce
    finite, channel-varying embeddings so the rung is a fair comparison rather
    than a degenerate model.
    """
    names, xyz = montage_19
    meta = ChannelMeta(names, xyz, reference_type="common_average")

    coords_only = ChannelEncoder(TAREConfig(embed_dim=32, use_name_embedding=False,
                                            use_reference_metadata=False))(meta)
    names_only = ChannelEncoder(TAREConfig(embed_dim=32, use_coordinates=False))(meta)
    for e in (coords_only, names_only):
        assert torch.isfinite(e).all()
        assert (e - e.mean(0)).abs().max() > 1e-3, "the encoder cannot tell channels apart"

    # Coordinates off => permuting only the coordinates changes nothing.
    enc = ChannelEncoder(TAREConfig(embed_dim=32, use_coordinates=False)).eval()
    p = torch.randperm(len(names))
    with torch.no_grad():
        a = enc(ChannelMeta(names, xyz))
        b = enc(ChannelMeta(names, xyz[p]))
    assert torch.allclose(a, b, atol=1e-6)

    # Reference metadata off => the reference scheme no longer changes anything.
    enc = ChannelEncoder(TAREConfig(embed_dim=32, use_reference_metadata=False)).eval()
    with torch.no_grad():
        a = enc(ChannelMeta(names, xyz, reference_type="original"))
        b = enc(ChannelMeta(names, xyz, reference_type="linked_mastoids"))
    assert torch.allclose(a, b, atol=1e-6)
