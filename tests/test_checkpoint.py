"""Checkpoint save/resume, atomic writes and explicit key migration."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from physiowave.models.checkpoint import (
    load_checkpoint,
    load_model_state,
    migrate_state_dict,
    save_checkpoint,
)


class Tiny(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.lin(x))


def test_save_and_resume_roundtrip(tmp_path):
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    x = torch.randn(4, 8)
    model(x).sum().backward()
    opt.step()

    path = str(tmp_path / "ck.pth")
    save_checkpoint(path, model, opt, sched, None, epoch=3, step=42,
                    metrics={"loss": 0.5}, config={"a": 1})
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")

    fresh = Tiny()
    fopt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fsched = torch.optim.lr_scheduler.LambdaLR(fopt, lambda s: 1.0)
    payload = load_checkpoint(path, fresh, fopt, fsched, strict=True)
    assert payload["epoch"] == 3 and payload["step"] == 42
    assert payload["config"] == {"a": 1}
    assert "git_commit" in payload["env"] and "torch" in payload["env"]
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.allclose(a, b)
    with torch.no_grad():
        assert torch.allclose(model(x), fresh(x))


def test_rng_state_is_restored(tmp_path):
    model = Tiny()
    torch.manual_seed(1234)
    path = str(tmp_path / "rng.pth")
    save_checkpoint(path, model)
    expected = torch.randn(5)
    load_checkpoint(path, model, restore_rng=True)
    assert torch.allclose(torch.randn(5), expected)


def test_key_mismatch_is_explicit_not_silent(tmp_path):
    """A mismatched checkpoint must raise with a description, never load partially."""
    model = Tiny()
    path = str(tmp_path / "ck.pth")
    save_checkpoint(path, model)
    payload = torch.load(path, weights_only=False)
    payload["model"]["lin.weight"] = torch.randn(16, 16)      # wrong shape
    payload["model"]["obsolete.weight"] = torch.randn(3)      # gone from the model
    torch.save(payload, path)

    with pytest.raises(RuntimeError) as exc:
        load_model_state(Tiny(), payload["model"], strict=True)
    msg = str(exc.value)
    assert "MISSING" in msg and "UNEXPECTED" in msg
    assert "lin.weight" in msg

    report = load_model_state(Tiny(), payload["model"], strict=False)
    assert not report.clean and report.missing and report.unexpected


def test_migration_remaps_known_prefixes():
    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.legacy_patch_embed = nn.Linear(4, 4)

    old = {"patch_embed.weight": torch.randn(4, 4), "patch_embed.bias": torch.randn(4)}
    migrated, report = migrate_state_dict(old, Target())
    assert set(migrated) == {"legacy_patch_embed.weight", "legacy_patch_embed.bias"}
    assert len(report.remapped) == 2 and report.clean
