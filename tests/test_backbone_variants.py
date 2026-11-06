"""The backbone's post-BERT components, and the switches that turn them off.

Each is a config option rather than a rewrite so the paper can ablate them.
The properties worth pinning are that the switches actually reach the modules,
that SwiGLU's 2/3 hidden scaling keeps the parameter count level with the GELU
MLP it replaces -- otherwise "better FFN" is really "bigger FFN" -- and that
RMSNorm reduces in fp32 under a half-precision input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physiowave.models.backbone import (        # noqa: E402
    BackboneConfig,
    FactorizedBackbone,
    RMSNorm,
    SwiGLU,
    make_ffn,
    make_norm,
)


def cfg(**kw) -> BackboneConfig:
    base = BackboneConfig(embed_dim=128, depth=2, num_heads=4, slot_heads=2, mlp_ratio=4.0)
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def test_defaults_are_the_modern_block():
    c = cfg()
    assert (c.norm, c.ffn, c.qk_norm) == ("rmsnorm", "swiglu", True)


@pytest.mark.parametrize("norm,ffn,qk", [
    ("layernorm", "mlp", False),          # the original block
    ("rmsnorm", "mlp", False),
    ("layernorm", "swiglu", False),
    ("layernorm", "mlp", True),
    ("rmsnorm", "swiglu", True),          # the default
])
def test_every_combination_runs(norm, ffn, qk):
    m = FactorizedBackbone(cfg(norm=norm, ffn=ffn, qk_norm=qk)).eval()
    x = torch.randn(2, 4, 8, 128)
    with torch.no_grad():
        assert m(x).shape == x.shape


def test_switches_reach_the_modules():
    modern = FactorizedBackbone(cfg())
    legacy = FactorizedBackbone(cfg(norm="layernorm", ffn="mlp", qk_norm=False))
    assert isinstance(modern.blocks[0].n1, RMSNorm)
    assert isinstance(modern.blocks[0].mlp, SwiGLU)
    assert isinstance(modern.blocks[0].time_attn.q_norm, RMSNorm)
    assert isinstance(legacy.blocks[0].n1, torch.nn.LayerNorm)
    assert isinstance(legacy.blocks[0].mlp, torch.nn.Sequential)
    assert isinstance(legacy.blocks[0].time_attn.q_norm, torch.nn.Identity)


def test_swiglu_does_not_smuggle_in_extra_capacity():
    c = cfg()
    swiglu = sum(p.numel() for p in make_ffn(c).parameters())
    c.ffn = "mlp"
    mlp = sum(p.numel() for p in make_ffn(c).parameters())
    assert abs(swiglu - mlp) / mlp < 0.05, (swiglu, mlp)


def test_rmsnorm_matches_the_definition():
    n = RMSNorm(32)
    x = torch.randn(4, 7, 32) * 6
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + n.eps)
    assert torch.allclose(n(x), expected, atol=1e-6)


def test_rmsnorm_reduces_in_fp32_and_returns_the_input_dtype():
    n = RMSNorm(64)
    x = (torch.randn(2, 5, 64) * 100).to(torch.bfloat16)
    out = n(x)
    assert out.dtype == torch.bfloat16
    # bf16 has ~8 mantissa bits; reducing in it would drift far more than this
    ref = n(x.float())
    assert (out.float() - ref).abs().max() < 0.5


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="unknown norm"):
        make_norm("batchnorm", 8)
    with pytest.raises(ValueError, match="unknown ffn"):
        make_ffn(cfg(ffn="glu"))


def test_gradients_reach_every_parameter():
    m = FactorizedBackbone(cfg()).train()
    m(torch.randn(2, 4, 8, 128)).square().mean().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, dead


def test_both_mixings_reach_a_cross_channel_cross_time_pair():
    """Records what the factorization does and does not cost.

    It is tempting to say the factorized block cannot relate (channel i, time t)
    to (channel j, time t'). It can, inside a single block: temporal attention
    spreads channel i along time, and the slot attention that follows spreads
    that across channels at t'. The two compose. What the factorization actually
    costs is that the path is forced through an intermediate already averaged
    along one axis, where 'full' scores the pair directly -- a difference in how
    the pattern is represented, not in whether it is reachable.

    This test exists so that claim stays honest: an earlier version of it
    asserted unreachability and was false.
    """
    from physiowave.models.backbone import BackboneConfig, FactorizedBlock

    B, K, S, D = 1, 4, 6, 16
    torch.manual_seed(0)
    x = torch.randn(B, K, S, D)
    bumped = x.clone()
    bumped[0, 0, 0] += 5.0                     # channel 0, time 0

    moved = {}
    for mixing in ("factorized", "full"):
        blk = FactorizedBlock(BackboneConfig(embed_dim=D, num_heads=2, slot_heads=2,
                                             dropout=0.0, mixing=mixing)).eval()
        with torch.no_grad():
            a, b = blk(x), blk(bumped)
        moved[mixing] = (a[0, 2, 4] - b[0, 2, 4]).abs().max().item()

    assert moved["factorized"] > 1e-4, moved
    assert moved["full"] > 1e-4, moved


def test_the_two_mixings_are_different_functions():
    from physiowave.models.backbone import BackboneConfig, FactorizedBlock

    torch.manual_seed(0)
    x = torch.randn(1, 4, 6, 16)
    outs = []
    for mixing in ("factorized", "full"):
        torch.manual_seed(0)
        blk = FactorizedBlock(BackboneConfig(embed_dim=16, num_heads=2, slot_heads=2,
                                             dropout=0.0, mixing=mixing)).eval()
        with torch.no_grad():
            outs.append(blk(x))
    assert not torch.allclose(outs[0], outs[1], atol=1e-4)


def test_full_mixing_drops_the_slot_attention():
    from physiowave.models.backbone import BackboneConfig, FactorizedBlock

    full = FactorizedBlock(BackboneConfig(embed_dim=16, num_heads=2, slot_heads=2,
                                          mixing="full"))
    fact = FactorizedBlock(BackboneConfig(embed_dim=16, num_heads=2, slot_heads=2,
                                          mixing="factorized"))
    assert full.slot_attn is None and fact.slot_attn is not None
    n = lambda m: sum(p.numel() for p in m.parameters())      # noqa: E731
    assert n(full) < n(fact), "an unused module must not sit in the budget"


def test_full_mixing_keeps_the_lattice_shape():
    from physiowave.models.backbone import BackboneConfig, FactorizedBackbone

    bb = FactorizedBackbone(BackboneConfig(embed_dim=32, depth=2, num_heads=4,
                                           slot_heads=2, mixing="full")).eval()
    x = torch.randn(2, 8, 12, 32)
    with torch.no_grad():
        assert bb(x).shape == x.shape
