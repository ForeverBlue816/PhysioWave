"""EEG downstream variants: class-balanced loss, a route-placed montage's lr
scaling, and the table's mean over seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_balanced_class_weights():
    from physiowave.train.finetune_main import balanced_class_weights
    w = balanced_class_weights(np.array([0] * 50 + [1] * 10), 2)
    assert torch.allclose(w, torch.tensor([0.6, 3.0]))
    with pytest.raises(SystemExit):                    # a class with no window
        balanced_class_weights(np.array([0, 0, 2]), 3)


def test_route_frontend_learns_with_the_encoder_when_it_was_loaded():
    from physiowave.eeg_c1.downstream import ROUTE_BOUND, TRANSFERABLE
    from physiowave.train.finetune_main import c1_lr_scale
    front = "wavelet_frontend.decomp.x"
    own = c1_lr_scale(4, 0.1, 0.75)
    assert own(front) == 1.0                           # fresh: learns at lr
    routed = c1_lr_scale(4, 0.1, 0.75, TRANSFERABLE + ROUTE_BOUND)
    assert routed(front) == pytest.approx(0.1 * 0.75 ** 5)
    assert routed("shared_transformer.blocks.3.attn.w") == pytest.approx(0.1 * 0.75)
    assert routed("head.weight") == 1.0


def _run(root, name, val, test, epoch=3):
    d = root / name
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "best_val": val, "best_epoch": epoch, "select_by": "auroc",
        "hparams": {"lr": 1e-3}, "test": test}))


def test_table_averages_seeds_and_selects_on_their_mean(tmp_path):
    t = lambda b, a: {"balanced_acc": b, "kappa": 0.2, "auroc": a, "weighted_f1": 0.8}
    _run(tmp_path, "physio_p300_probe_lr1e-3_route_cw", 0.80, t(0.70, 0.76))
    _run(tmp_path, "physio_p300_probe_lr1e-3_route_cw_s43", 0.70, t(0.60, 0.74))
    _run(tmp_path, "physio_p300_probe_lr1e-3", 0.77, t(0.62, 0.75))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "collect_eeg_downstream.py"),
                          "--root", str(tmp_path), "--json", str(tmp_path / "s.json")],
                         capture_output=True, text=True, check=True).stdout
    route = next(line for line in out.splitlines() if "[route_cw]" in line)
    assert "0.6500" in route and "n=2" in route        # mean of 0.70 and 0.60
    assert route.lstrip().startswith("✓") is False  # mean val 0.75 < 0.77
    plain = next(line for line in out.splitlines()
                 if "linear probe" in line and "[" not in line)
    assert plain.lstrip().startswith("✓")
    assert "0.0707" in out                              # std of the two BalAccs
