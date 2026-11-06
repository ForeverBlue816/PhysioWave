"""Guards against APIs that only exist on a torch newer than the cluster's.

Development happens on torch 2.11; Leonardo's ``cineca-ai/4.3.0`` ships torch
2.2.0.  Anything added between the two imports fine here and raises
``AttributeError`` on the first batch of a 16-GPU job, which is an expensive
place to find out.  These tests pin the two spellings that differ.
"""

import torch

from physiowave.train.utils import make_grad_scaler


def _without_new_gradscaler(monkeypatch):
    """Make ``torch.amp`` look like torch 2.2, which has no ``GradScaler``."""
    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)


def test_make_grad_scaler_falls_back_when_torch_amp_has_none(monkeypatch):
    _without_new_gradscaler(monkeypatch)
    assert not hasattr(torch.amp, "GradScaler")
    scaler = make_grad_scaler("cuda", enabled=False)
    assert isinstance(scaler, torch.cuda.amp.GradScaler)
    assert not scaler.is_enabled()


def test_make_grad_scaler_uses_new_api_when_available():
    if not hasattr(torch.amp, "GradScaler"):                  # torch < 2.4
        return
    scaler = make_grad_scaler("cpu", enabled=True)
    assert not scaler.is_enabled()          # never enabled off CUDA


def test_make_grad_scaler_disables_itself_off_cuda(monkeypatch):
    _without_new_gradscaler(monkeypatch)
    # The old class takes no device argument, so passing a non-CUDA device
    # must not reach it as an enabled scaler.
    assert not make_grad_scaler("cpu", enabled=True).is_enabled()
    assert not make_grad_scaler("mps", enabled=True).is_enabled()


def test_no_module_calls_torch_amp_gradscaler_directly():
    """The new spelling must not creep back into the training modules."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        if ".git" in path.parts or path.name == "test_torch_version_compat.py":
            continue
        if path.name == "utils.py" and path.parent.name == "train":
            continue                                          # the shim itself
        if "torch.amp.GradScaler" in path.read_text():
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"use make_grad_scaler instead: {offenders}"


def test_nn_rmsnorm_is_not_required():
    """``nn.RMSNorm`` landed in torch 2.4; the C1 encoder must not need it."""
    import torch.nn as nn
    from channel_embedding import ChannelEncoder
    has_new = hasattr(nn, "RMSNorm")
    enc = ChannelEncoder(mode="id", embed_dim=32, norm="rmsnorm")
    expected = nn.RMSNorm if has_new else nn.LayerNorm
    assert isinstance(enc.norm, expected)


def test_channel_encoder_loads_either_norm():
    """A checkpoint trained with LayerNorm (torch < 2.4) loads where RMSNorm
    exists, and one trained with RMSNorm loads where it does not."""
    import torch
    import torch.nn as nn
    from channel_embedding import ChannelEncoder
    enc = ChannelEncoder(mode="id", embed_dim=16, norm="rmsnorm")
    ln_state = {k: v for k, v in enc.state_dict().items()
                if not k.startswith("norm.")}
    ln_state["norm.weight"] = torch.full((16,), 2.0)
    ln_state["norm.bias"] = torch.full((16,), 0.5)
    a = ChannelEncoder(mode="id", embed_dim=16, norm="rmsnorm")
    a.load_state_dict(ln_state)
    assert isinstance(a.norm, nn.LayerNorm)
    assert torch.allclose(a.norm.bias, torch.full((16,), 0.5))

    rms_state = {k: v for k, v in ln_state.items() if k != "norm.bias"}
    b = ChannelEncoder(mode="id", embed_dim=16, norm="layernorm")
    b.load_state_dict(rms_state)
    assert not isinstance(b.norm, nn.LayerNorm)
    x = torch.randn(3, 16)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                          + torch.finfo(x.dtype).eps) * 2.0
    assert torch.allclose(b.norm(x), ref, atol=1e-5)
