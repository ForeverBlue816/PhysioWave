"""The training loop itself, called the way main_worker calls it.

Every other channel-embedding test builds a model and calls it directly. That
is why `train_one_epoch(..., channel_meta=...)` reached the cluster with no
`channel_meta` parameter in its signature: the call site was updated, the
function was not, and nothing between here and the GPU nodes ever ran the loop.

These tests run the real `train_one_epoch` and `eval_one_epoch` on a handful of
synthetic windows, on CPU. They assert almost nothing about the numbers -- they
exist so that a keyword the loop is called with is a keyword the loop takes.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from channel_embedding import channel_id                           # noqa: E402
from model import BERTWaveletTransformer                           # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "finetune", os.path.join(_HERE, "finetune.py"))
finetune = importlib.util.module_from_spec(_SPEC)
sys.modules["finetune"] = finetune
_SPEC.loader.exec_module(finetune)

_XYZ = torch.tensor([[1e-4, 0.0882, -0.0017],
                     [4e-4, -0.0092, 0.1002],
                     [3e-4, -0.0811, 0.0826],
                     [1e-4, -0.1149, 0.0147]], dtype=torch.float32)

VARIANTS = [("C0", "none", "none"), ("C1", "id", "token"),
            ("C2", "signed", "token"), ("C3", "signed", "fold"),
            ("C4", "signed", "dual"), ("C5", "hybrid", "dual")]


def _meta():
    return {
        "channel_ids": torch.tensor([channel_id("Fpz-Cz"), channel_id("Pz-Oz")]),
        "electrode_xyz": _XYZ.clone(),
        "positive_electrode_index": torch.tensor([0, 2]),
        "negative_electrode_index": torch.tensor([1, 3]),
        "valid_channel_mask": torch.tensor([True, True]),
    }


def _model(encoding, injection, seed=0):
    torch.manual_seed(seed)
    return BERTWaveletTransformer(
        in_channels=2, max_level=3, wave_kernel_size=16,
        wavelet_names=["sym4", "db6"], use_separate_channel=True,
        wave_init_mode="pad", patch_size=(1, 50), embed_dim=64, depth=2,
        num_heads=4, mlp_ratio=4.0, dropout=0.0, norm="rmsnorm", ffn="swiglu",
        qk_norm=True, scale_fold="dynamic", fold_synthesis=3, fold_gamma=0.1,
        use_pos_embed=True, pos_embed_type="2d",
        channel_encoding=encoding, channel_injection=injection,
        channel_embed_dim=32, task_type="classification", num_classes=5,
        head_config={"hidden_dims": [32], "dropout": 0.0, "pooling": "mean"},
        pooling="mean")


def _loader(n=8, batch=4, samples=200):
    torch.manual_seed(1234)
    x = torch.randn(n, 2, samples)
    y = torch.randint(0, 5, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,enc,inj", VARIANTS)
def test_train_and_eval_accept_channel_meta(name, enc, inj):
    """main_worker passes channel_meta to both; both must take it, for every row.

    Parametrised over all six variants rather than one, because the argument is
    passed unconditionally -- C0 is called with channel_meta=None and must not
    treat that as "a keyword I have never heard of" either.
    """
    model = _model(enc, inj)
    meta = None if enc == "none" else _meta()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    # rank=0 on purpose: it is the branch that prints alpha, the per-channel
    # alpha and the gates, so the logging added with this feature runs too.
    loss, acc, lr = finetune.train_one_epoch(
        0, 0, model, opt, _loader(), torch.device("cpu"), crit,
        scaler=None, grad_clip=1.0, scheduler=None, scheduler_per_batch=False,
        fold_kl=1e-3, channel_meta=meta)
    assert loss == loss and 0.0 <= acc <= 1.0 and lr > 0.0        # not NaN

    out = finetune.eval_one_epoch(
        0, 0, model, _loader(), torch.device("cpu"), crit,
        desc_prefix="Val", channel_meta=meta)
    assert len(out) == 6 and out[0] == out[0]


def test_train_one_epoch_is_called_with_exactly_what_it_declares():
    """The signature and the call site, checked against each other by name.

    The parametrised test above would catch a missing parameter, but only for
    the arguments it happens to pass. This reads main_worker's own call.
    """
    import ast
    import inspect

    src = open(os.path.join(_HERE, "finetune.py")).read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None)
             in ("train_one_epoch", "eval_one_epoch")]
    assert calls, "no call to the training loop found -- did it get renamed?"

    for call in calls:
        fn = getattr(finetune, call.func.id)
        accepted = set(inspect.signature(fn).parameters)
        passed = {kw.arg for kw in call.keywords if kw.arg is not None}
        missing = passed - accepted
        assert not missing, (
            f"{call.func.id} is called at line {call.lineno} with "
            f"{sorted(missing)}, which it does not accept")


def test_channel_gate_and_alpha_logging_survives_a_step():
    """The accessors the log line calls must exist and stay finite after a step.

    They are read inside the rank-0 branch of the loop, so a rename or a None
    would take down training on rank 0 only, several minutes in.
    """
    model = _model("signed", "dual")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    finetune.train_one_epoch(
        0, 0, model, opt, _loader(), torch.device("cpu"), nn.CrossEntropyLoss(),
        fold_kl=1e-3, channel_meta=_meta())

    g_f, g_t = model.channel_gate_values()
    assert g_f is not None and g_t is not None
    assert torch.isfinite(torch.tensor([g_f, g_t])).all()
    per_c = model.scale_fold_per_channel()
    assert per_c.shape[0] == 2 and torch.isfinite(per_c).all()


def test_only_training_is_sharded_across_ranks():
    """val and test must see every window on every rank.

    A DistributedSampler on the eval loaders hands each rank 1/world_size of
    the set, and nothing in finetune.py gathers the shards. The metrics are
    then computed per rank and only rank 0's are printed and saved, so the
    reported test score silently becomes a score on a quarter of the test set --
    and the confusion matrix in test_results.json sums to a quarter of it,
    which is the only visible trace.

    Replication rather than an all_gather, so this is a structural check: the
    only DistributedSampler in the file belongs to training.
    """
    import ast

    src = open(os.path.join(_HERE, "finetune.py")).read()
    tree = ast.parse(src)

    sharded = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "DistributedSampler"):
            continue
        # DistributedSampler(<dataset>, ...) -- the first argument names it.
        ds = node.args[0] if node.args else None
        sharded.append(getattr(ds, "id", ast.dump(ds) if ds else "?"))

    assert sharded == ["train_ds"], (
        f"DistributedSampler is applied to {sharded}; only train_ds may be "
        f"sharded, because eval_one_epoch never gathers across ranks")

    # And the eval loaders must not be handed a sampler by any other route.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "DataLoader"):
            continue
        target = getattr(node.targets[0], "id", "")
        if target in ("val_loader", "test_loader"):
            kws = {k.arg for k in node.value.keywords}
            assert "sampler" not in kws, (
                f"{target} at line {node.lineno} takes a sampler; eval metrics "
                f"would be computed per shard and never gathered")
