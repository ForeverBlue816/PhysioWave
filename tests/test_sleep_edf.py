"""Sleep-EDF preparation: splits and the shape contract with finetune.py."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "sleep_edf_finetune", os.path.join(_HERE, "EEG", "sleep_edf_finetune.py"))
sleep = importlib.util.module_from_spec(_SPEC)
sys.modules["sleep_edf_finetune"] = sleep
_SPEC.loader.exec_module(sleep)


def test_no_subject_appears_in_two_splits():
    """Consecutive 30 s epochs of one night are near-duplicates.

    A window-level split would put a sleeper's own neighbouring epochs on both
    sides of the boundary and report a number that says nothing about a new
    sleeper. Every split here must therefore be a partition of subjects.
    """
    splits = [sleep.holdout_split(sleep.EEGPT_SUBJECTS)]
    splits += [sleep.eegpt_fold_split(f) for f in range(10)]
    for s in splits:
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr and va and te


def test_the_eegpt_folds_cover_their_subject_list_without_overlap():
    held = [set(sleep.eegpt_fold_split(f)["test"]) for f in range(10)]
    for i in range(10):
        for j in range(i + 1, 10):
            assert not (held[i] & held[j]), (i, j)
    union = set().union(*held)
    assert union <= set(sleep.EEGPT_SUBJECTS)
    assert len(union) == 60      # 64 subjects, 10 folds of 6; the last 4 are unused


def test_the_eegpt_folds_never_validate_on_what_they_score():
    """EEGPT selects on the fold it reports; this must not."""
    for f in range(10):
        s = sleep.eegpt_fold_split(f)
        assert not (set(s["val"]) & set(s["test"])), f


def test_split_is_deterministic():
    assert sleep.holdout_split(sleep.EEGPT_SUBJECTS) == \
           sleep.holdout_split(sleep.EEGPT_SUBJECTS)
    assert sleep.eegpt_fold_split(3) == sleep.eegpt_fold_split(3)


def test_written_files_match_what_finetune_reads(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    subjects = [0, 2, 4]
    rng = np.random.default_rng(0)
    for s in subjects:
        np.savez_compressed(
            cache / f"sub{s:02d}.npz",
            data=rng.standard_normal((7, 2, sleep.WINDOW_SAMPLES)).astype("float32"),
            label=rng.integers(0, 5, 7).astype("int64"))
    info = sleep.write_split("train", subjects, str(cache), str(tmp_path))
    assert info["windows"] == 21

    import h5py

    with h5py.File(tmp_path / "train.h5") as f:
        assert f["data"].shape == (21, 2, 3000)
        assert f["data"].dtype == np.float32
        assert f["label"].dtype == np.int64
        assert [c.decode() for c in f["channel_names"][:]] == sleep.CHANNELS
        # Provenance, so a result can be traced back to the sleepers behind it.
        assert sorted(set(f["subject"][:].tolist())) == subjects


def test_a_missing_subject_cache_is_reported_not_silently_dropped(tmp_path, capsys):
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(cache / "sub00.npz",
                        data=np.zeros((3, 2, sleep.WINDOW_SAMPLES), "float32"),
                        label=np.zeros(3, "int64"))
    info = sleep.write_split("train", [0, 99], str(cache), str(tmp_path))
    assert info["windows"] == 3
    assert "subject 99" in capsys.readouterr().err


def test_the_configured_eeg_model_is_the_size_and_shape_it_claims():
    """11M parameters, 120 tokens, five classes, from a (2, 3000) window."""
    from model import create_wavelet_classifier

    m = create_wavelet_classifier(
        in_channels=2, max_level=3, wave_kernel_size=16,
        wavelet_names=["sym4", "sym5", "db6", "sym8", "db8"], wave_init_mode="pad",
        use_separate_channel=True, patch_size=(1, 50), embed_dim=384, depth=6,
        num_heads=6, mlp_ratio=4.0, dropout=0.1, norm="rmsnorm", ffn="swiglu",
        qk_norm=True, scale_fold="dynamic", fold_synthesis=3, use_pos_embed=True,
        pos_embed_type="2d", num_classes=5, pooling="mean",
        head_config={"hidden_dims": [512], "dropout": 0.1, "pooling": "mean"}).eval()
    n = sum(p.numel() for p in m.parameters())
    assert 10e6 < n < 12e6, n
    x = torch.randn(2, 2, sleep.WINDOW_SAMPLES)
    with torch.no_grad():
        tok = m.prepare_tokens(m.fold_scales(m.wavelet_decomp(x)).unsqueeze(1))
        assert m(x, task="classify").shape == (2, 5)
    # 2 channels x (3000 / 50) time patches; unfolded it would be 4x that.
    assert tok.shape[1] == 2 * (sleep.WINDOW_SAMPLES // 50) == 120
    # The fold carries no channel-shaped parameter, so this transfers to a
    # montage with a different electrode count.
    assert sum(p.numel() for p in m.fold.parameters()) == 301


@pytest.mark.parametrize("n", [3, 4, 5, 6, 10, 20, 64])
def test_no_split_is_ever_empty(n):
    """An empty validation split does not raise -- it silently disables
    model selection, and the run looks normal the whole way through."""
    s = sleep.holdout_split(list(range(n)))
    assert all(len(v) >= 1 for v in s.values()), (n, s)
    assert sum(len(v) for v in s.values()) == n


def test_too_few_subjects_is_refused_with_a_usable_message():
    with pytest.raises(SystemExit, match="at least 3 subjects"):
        sleep.holdout_split([0, 2])
