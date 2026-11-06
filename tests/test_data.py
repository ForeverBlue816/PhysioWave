"""Data layer: schema, splits, preprocessing, registry and montage geometry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physiowave.data.montages import (
    canonical_name,
    montage,
    positions_for,
)
from physiowave.data.preprocess import (
    PreprocessCache,
    PreprocessConfig,
    normalize_signal,
    preprocess,
)
from physiowave.data.registry import REGISTRY, DatasetSpec, assert_limb_semg, get
from physiowave.data.schema import batch_to_meta, collate_samples
from physiowave.data.splits import (
    SplitLeakageError,
    assert_no_leakage,
    split_records,
    subject_wise_split,
)
from physiowave.data.synthetic import SyntheticConfig, SyntheticDataset


# --------------------------------------------------------------------------- #
# Montages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["standard_1020_19", "standard_1010_61", "standard_1010_64"])
def test_template_montages_are_on_the_sphere(name):
    names, xyz = montage(name)
    assert len(names) == len(set(names)), "duplicate channel labels"
    r = xyz.norm(dim=-1)
    assert (r > 0.5).all(), "electrode at the origin"
    assert float((r - r.mean()).abs().max()) < 0.3, "positions are not near-spherical"


def test_montage_landmarks_are_where_they_should_be():
    """Cz at the vertex, T7/T8 at the pre-auricular points, odd labels on the left."""
    names, xyz = montage("standard_1010_64")
    pos = dict(zip(names, xyz, strict=True))
    assert pos["Cz"][2] > 0.99, "Cz is not at the vertex"
    assert abs(pos["T7"][2]) < 0.35 and pos["T7"][0] < -0.9, "T7 is not at the left ear"
    assert abs(pos["T8"][2]) < 0.35 and pos["T8"][0] > 0.9, "T8 is not at the right ear"
    assert pos["Fp1"][1] > 0.8 and pos["O1"][1] < -0.8, "anterior/posterior axis is wrong"
    for left, right in (("C3", "C4"), ("F7", "F8"), ("P3", "P4")):
        assert pos[left][0] < 0 < pos[right][0], f"{left}/{right} are not left/right"
        assert torch.allclose(pos[left][1:], pos[right][1:], atol=1e-6), "not mirror symmetric"


def test_channel_name_aliases():
    assert canonical_name("T3") == "T7" and canonical_name("T5") == "P7"
    assert canonical_name("EEG FP1-REF") == "Fp1"
    assert canonical_name("Cz-LE") == "Cz"


def test_unknown_channels_get_zero_coordinates():
    xyz, known = positions_for(["Cz", "NOT_AN_ELECTRODE", "Fp1"])
    assert known.tolist() == [True, False, True]
    assert xyz[1].abs().sum() == 0.0


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_sample_validation_catches_mismatches():
    ok = SyntheticDataset(SyntheticConfig("eeg", 2))[0]
    ok.validate()
    bad = SyntheticDataset(SyntheticConfig("eeg", 2))[0]
    bad.channel_names = bad.channel_names[:-1]
    with pytest.raises(AssertionError):
        bad.validate()


def test_collate_and_meta_roundtrip():
    ds = SyntheticDataset(SyntheticConfig("eeg", 8, window_samples=256))
    batch = collate_samples([ds[i] for i in range(4)])
    assert batch["signal"].shape == (4, 19, 256)
    meta = batch_to_meta(batch)
    assert meta.num_channels() == 19
    assert meta.channel_xyz.shape == (19, 3)


def test_collate_rejects_mixed_channel_counts():
    a = SyntheticDataset(SyntheticConfig("eeg", 2))[0]
    b = SyntheticDataset(SyntheticConfig("ecg", 2, num_channels=12))[0]
    with pytest.raises(AssertionError, match="channel count"):
        collate_samples([a, b])


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_subject_wise_split_is_disjoint_and_stable():
    subjects = [f"S{i:03d}" for i in range(200)]
    a = subject_wise_split(subjects)
    b = subject_wise_split(subjects + [f"S{i:03d}" for i in range(200, 220)])
    for split in ("train", "val", "test"):
        assert set(a[split]).issubset(set(b[split])), "adding subjects reshuffled the split"
    assert not (set(a["train"]) & set(a["val"]))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["val"]) & set(a["test"]))
    assert sum(len(v) for v in a.values()) == 200


def test_leakage_check_raises():
    with pytest.raises(SplitLeakageError, match="subject overlap"):
        assert_no_leakage({"train": ["a", "b"], "val": ["b"], "test": ["c"]})
    with pytest.raises(SplitLeakageError, match="recording overlap"):
        assert_no_leakage({"train": ["a"], "val": ["b"], "test": ["c"]},
                          {"train": ["r1"], "val": ["r1"], "test": ["r2"]})


def test_split_records_never_leaks_a_subject():
    """Windows of one subject all land in the same split -- no segment-level split."""
    records = [{"subject_id": f"S{i % 12}", "recording_id": f"S{i % 12}_R{i % 3}"}
               for i in range(240)]
    parts = split_records(records)
    owner = {}
    for split, rs in parts.items():
        for r in rs:
            assert owner.setdefault(r["subject_id"], split) == split


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_chain_changes_rate_and_normalises():
    fs, T = 500.0, 2000
    t = np.arange(T) / fs
    x = np.stack([np.sin(2 * np.pi * 10 * t) + 5.0 + 0.5 * np.sin(2 * np.pi * 50 * t)
                  for _ in range(4)])
    cfg = PreprocessConfig(target_sampling_rate=250.0, notch_freq=50.0,
                           bandpass=(1.0, 45.0), normalize="zscore")
    y, new_fs = preprocess(x, fs, cfg)
    assert new_fs == 250.0 and y.shape == (4, 1000) and y.dtype == np.float32
    assert abs(float(y.mean())) < 1e-3 and abs(float(y.std()) - 1.0) < 0.1
    spec = np.abs(np.fft.rfft(y, axis=-1)).mean(0)
    freqs = np.fft.rfftfreq(y.shape[-1], 1 / new_fs)
    line = spec[(freqs > 48) & (freqs < 52)].max()
    alpha = spec[(freqs > 8) & (freqs < 12)].max()
    assert line < 0.2 * alpha, "the 50 Hz notch did not attenuate line noise"


@pytest.mark.parametrize("mode", ["zscore", "minmax", "maxabs", "none"])
def test_normalisation_modes(mode):
    x = np.random.randn(3, 100) * 5 + 2
    y = normalize_signal(x, mode)
    assert y.shape == x.shape and np.isfinite(y).all()
    if mode == "minmax":
        assert y.min() >= -1.0001 and y.max() <= 1.0001
    if mode == "maxabs":
        assert np.abs(y).max() <= 1.0001


def test_preprocess_cache_hits(tmp_path):
    cfg = PreprocessConfig(cache_dir=str(tmp_path), normalize="zscore")
    cache = PreprocessCache(cfg)
    arr = np.random.randn(4, 64).astype(np.float32)
    assert cache.get("k") is None
    cache.put("k", arr)
    got = cache.get("k")
    assert got is not None and np.allclose(got, arr)
    assert cache.hits == 1 and cache.misses == 1


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_covers_the_planned_corpora():
    for ds in ("tueg", "siena", "tuab", "tuar", "tusl", "bci_iv_2a", "seizeit2", "mpdb",
               "mimic_iv_ecg", "ptbxl", "cpsc2018", "shaoxing", "ninapro_db6", "epn612"):
        assert ds in REGISTRY, f"{ds} missing from the registry"


def test_protected_datasets_are_flagged():
    for ds in ("tueg", "tuab", "tuar", "tusl", "seizeit2"):
        assert get(ds).requires_agreement, f"{ds} must be marked as requiring an agreement"


def test_ssl_admissibility_follows_the_dataset(caplog):
    assert get("siena").uses_ssl() is True
    assert get("ptbxl").uses_ssl() is False           # ECG: no scalp sphere
    assert get("epn612").uses_ssl() is False          # sEMG: not a sphere either
    with caplog.at_level("WARNING"):
        spec = DatasetSpec("nocoord", modality="eeg", has_coordinates=False)
        assert spec.uses_ssl() is False
    assert "no electrode coordinates" in caplog.text


def test_facial_emg_is_rejected_from_the_limb_corpus():
    assert_limb_semg([get("ninapro_db6"), get("epn612")])
    facial = DatasetSpec("some_face_emg", modality="semg", emg_region="facial")
    with pytest.raises(ValueError, match="Facial EMG"):
        assert_limb_semg([facial])
    unknown = DatasetSpec("unlabelled", modality="semg", emg_region="unknown")
    with pytest.raises(ValueError):
        assert_limb_semg([unknown])
