"""
The sEMG C1 multi-route pretraining path. What is shared with EEG/ECG is tested
there; this covers the sEMG registry, model size, readers and the run itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "ECG"), os.path.join(ROOT, "EMG")):
    if p not in sys.path:
        sys.path.insert(0, p)

from channel_embedding import vocab_sha256                           # noqa: E402
from physiowave.ecg_c1.leads import ecg_vocab_sha256                  # noqa: E402
from physiowave.ecg_c1.preprocess import ECGPreprocessConfig          # noqa: E402
from physiowave.emg_c1.electrodes import (BAND_16,  # noqa: E402
                                          CEMHSEY_5X13_BY_CHANNEL,
                                          CEMHSEY_8X8_BY_CHANNEL,
                                          CEMHSEY_GRID, CEMHSEY_GRID_5X13,
                                          EMG_ELECTRODE_VOCAB, UNK_ID,
                                          electrode_ids_for, emg_vocab_sha256)
from physiowave.emg_c1.model import EMG_WAVELETS, MultiRouteEMGPretrainer  # noqa: E402
from physiowave.emg_c1.routes import (DOWNSTREAM_ONLY, PRETRAIN_DATASETS,  # noqa: E402
                                      ROUTES)

PY = sys.executable
PREP = os.path.join(ROOT, "EMG", "preprocess_emg_corpus.py")


def test_vocab_is_its_own():
    assert emg_vocab_sha256() not in (vocab_sha256(), ecg_vocab_sha256())
    ids, unknown = electrode_ids_for(["band1", "BAND16", "hyser_r8c8", "Fp1"])
    assert ids[:3] != [UNK_ID] * 3 and unknown == ["Fp1"]
    assert len(EMG_ELECTRODE_VOCAB) == 2 + 16 + 24 + 64 + 64 + 64


def test_cemhsey_channel_maps():
    # PreProcess.m's maps: an 8x8 grid numbers its channels up each column
    # from the bottom left; a 5x13 grid has no electrode at r5c1.
    assert CEMHSEY_8X8_BY_CHANNEL[0] == "cemhsey_r8c1"
    assert CEMHSEY_8X8_BY_CHANNEL[63] == "cemhsey_r1c8"
    assert CEMHSEY_5X13_BY_CHANNEL[0] == "cemhsey13_r4c1"
    assert "cemhsey13_r5c1" not in CEMHSEY_GRID_5X13
    assert sorted(CEMHSEY_8X8_BY_CHANNEL) == sorted(CEMHSEY_GRID)
    assert sorted(CEMHSEY_5X13_BY_CHANNEL) == sorted(CEMHSEY_GRID_5X13)


def test_routes_and_registry():
    for r in ROUTES.values():
        assert (r.sampling_rate, r.window_samples, r.patch_t,
                r.patches_per_channel) == (2000, 2000, 250, 8)
    assert {r.n_channels for r in ROUTES.values()} == {16, 24, 64}
    for spec in PRETRAIN_DATASETS.values():
        assert len(spec.lead_slots) == ROUTES[spec.route_id].n_channels
    assert not set(DOWNSTREAM_ONLY) & set(PRETRAIN_DATASETS)
    for gone in ("db6", "db7", "db8", "ninapro_db6"):
        assert gone not in PRETRAIN_DATASETS


def test_standard_model_is_an_11m_backbone():
    m = MultiRouteEMGPretrainer(embed_dim=384, depth=6, num_heads=6, dropout=0.0)
    rep = m.parameter_report()
    assert 10.5e6 < rep["shared_transformer"] < 10.8e6
    assert 11.5e6 < rep["total"] < 12.5e6
    # one grid per sample keeps the HD frontend small
    assert rep["wavelet_frontend.G64_2000"] < 1.0e6
    bank = m.wavelet_frontends["W16_2000"].decomp.selector.wavelet_filters
    assert [f.wave_init for f in bank] == list(EMG_WAVELETS)


def test_pipeline_resamples_filters_and_keeps_channel_ratios():
    sys.path.insert(0, os.path.join(ROOT, "EMG"))
    from preprocess_emg_corpus import (EMG_SETTINGS, process_emg_record,
                                       synthetic_semg)
    cfg = ECGPreprocessConfig(**EMG_SETTINGS)
    x = synthetic_semg(64, 2048.0, 3.0, np.random.default_rng(0))
    x[5] *= 3.0
    out = process_emg_record(x, list(PRETRAIN_DATASETS["hyser"].lead_slots),
                             2048.0, "mV", ROUTES["G64_2000"], 50.0, cfg,
                             slots=PRETRAIN_DATASETS["hyser"].lead_slots)
    assert out.windows.shape == (3, 64, 2000)
    sd = out.windows[0].std(axis=-1)
    assert sd[5] > 2.0 * np.median(sd)                 # relative amplitude kept
    assert out.channel_ids[0] != UNK_ID


@pytest.mark.slow
def test_trainer_smoke_end_to_end(tmp_path):
    from physiowave.train.pretrain_main import main
    out = str(tmp_path / "run")
    assert main(["--config", "pretrain/emg_c1_moe", "--smoke-test",
                 "--max-steps", "20", "--output-dir", out]) == 0
    ck = torch.load(os.path.join(out, "latest.pth"), map_location="cpu",
                    weights_only=False)
    assert ck["channel_vocab_sha256"] == emg_vocab_sha256()
    row = json.loads(open(os.path.join(out, "metrics_epoch.jsonl")).readline())
    assert any(k.startswith("val/route/G64_2000/") for k in row)


@pytest.mark.slow
def test_nonfinite_step_is_skipped_and_nan_resume_refused(tmp_path, monkeypatch,
                                                          capsys):
    from physiowave.train.pretrain_main import main
    out = str(tmp_path / "run")
    monkeypatch.setenv("PW_DEBUG_NONFINITE_STEP", "2")      # rank 0's loss -> inf
    assert main(["--config", "pretrain/emg_c1_moe", "--smoke-test",
                 "--max-steps", "6", "--output-dir", out]) == 0
    assert "SKIPPED step 2" in capsys.readouterr().out
    ck = torch.load(os.path.join(out, "latest.pth"), map_location="cpu",
                    weights_only=False)
    assert all(torch.isfinite(v).all() for v in ck["model"].values())
    row = json.loads(open(os.path.join(out, "metrics_epoch.jsonl")).readline())
    assert row["train/skipped_steps"] == 1.0 and row["train/max_grad_norm"] > 0
    assert ck["global_step"] == 6                           # the schedule advanced
    # a checkpoint whose weights already went NaN is refused as a resume
    k = next(k for k, v in ck["model"].items() if v.dtype.is_floating_point)
    ck["model"][k] = torch.full_like(ck["model"][k], float("nan"))
    torch.save(ck, os.path.join(out, "latest.pth"))
    monkeypatch.delenv("PW_DEBUG_NONFINITE_STEP")
    with pytest.raises(SystemExit, match="non-finite weights"):
        main(["--config", "pretrain/emg_c1_moe", "--smoke-test", "--max-steps",
              "6", "--output-dir", out, "--resume", "auto"])


def test_downstream_sets_are_refused():
    r = subprocess.run([PY, PREP, "--dataset", "db5", "--smoke-test",
                        "--out-dir", "/nonexistent"], capture_output=True,
                       text=True)
    assert r.returncode == 1 and "downstream" in r.stderr


# --------------------------------------------------------------------------- #
# Readers, on small files written in each corpus's real format
# --------------------------------------------------------------------------- #

def _adapters():
    from preprocess_emg_corpus import ADAPTERS
    return ADAPTERS


def _read_all(dataset_id, root):
    ad = _adapters()[dataset_id]
    keys = ad.list_keys(str(root))
    ctx = {}
    return keys, [r for k in keys for r in ad.read_records(str(root), k, ctx)]


def _check(dataset_id, recs):
    """Every record maps onto its corpus's slots and survives the pipeline."""
    from preprocess_emg_corpus import EMG_SETTINGS, process_emg_record
    spec = PRETRAIN_DATASETS[dataset_id]
    cfg = ECGPreprocessConfig(**EMG_SETTINGS)
    for r in recs:
        out = process_emg_record(r.data, r.lead_names, r.fs, r.unit,
                                 ROUTES[spec.route_id], spec.mains_hz, cfg,
                                 slots=spec.lead_slots)
        assert out.windows.shape[1:] == (len(spec.lead_slots), 2000)
        assert out.windows.shape[0] >= 1 and UNK_ID not in out.channel_ids


def test_reader_emg2pose_from_tar_and_tree(tmp_path):
    import h5py
    import tarfile
    tree = tmp_path / "tree" / "emg2pose_data"
    tree.mkdir(parents=True)
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)),
                   ("emg", "<f4", (16,))])
    for side, user in (("left", "u1"), ("right", "u1")):
        ts = np.zeros(5000, dtype=dt)
        ts["emg"] = (np.random.default_rng(0).normal(0, 50, (5000, 16)))
        with h5py.File(tree / f"s-recording-1_{side}.hdf5", "w") as h:
            g = h.create_group("emg2pose")
            g.create_dataset("timeseries", data=ts)
            g.attrs.update({"user": user, "side": side, "sample_rate": 2000.0})
    tar = tmp_path / "emg2pose_dataset.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(tree, arcname="emg2pose_data")
    for root in (tar, tmp_path / "tree"):
        keys, recs = _read_all("emg2pose", root)
        assert len(recs) == 2 and {r.subject_id for r in recs} == {"user_u1"}
        assert recs[0].unit == "uV" and recs[0].data.shape == (16, 5000)
        _check("emg2pose", recs)
    assert all("|" in k for k in _adapters()["emg2pose"].list_keys(str(tar)))


def test_reader_emg2qwerty_two_wrists(tmp_path):
    import h5py
    dt = np.dtype([("emg_right", "<f4", (16,)), ("time", "<f8"),
                   ("emg_left", "<f4", (16,))])
    ts = np.zeros(4400, dtype=dt)
    rng = np.random.default_rng(1)
    ts["emg_left"] = rng.normal(0, 20, (4400, 16))
    ts["emg_right"] = rng.normal(0, 40, (4400, 16))
    (tmp_path / "d").mkdir()
    with h5py.File(tmp_path / "d" / "2020-01-01-1-keystrokes.hdf5", "w") as h:
        g = h.create_group("emg2qwerty")
        g.create_dataset("timeseries", data=ts)
        g.attrs.update({"user": "0042", "daq_sample_rate": "2000.0"})
    _, recs = _read_all("emg2qwerty", tmp_path)
    assert [r.notes["side"] for r in recs] == ["left", "right"]
    assert recs[1].data.std() > 1.5 * recs[0].data.std()      # not swapped
    assert recs[0].subject_id == "user_0042"
    _check("emg2qwerty", recs)


def test_reader_hyser_splits_grids_and_forces_mv(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    d = tmp_path / "1dof_dataset" / "subject07_session2"
    d.mkdir(parents=True)
    names = [f"{g}-{r}-{c}" for g in ("ED", "EP", "FD", "FP")
             for r in range(8, 0, -1) for c in range(8, 0, -1)]
    x = np.random.default_rng(2).normal(0, 0.03, (2 * 2048, 256))
    x[:, names.index("FD-3-5")] *= 4
    wfdb.wrsamp("1dof_raw_finger1_sample1", fs=2048, units=["V"] * 256,
                sig_name=names, p_signal=x, fmt=["16"] * 256,
                write_dir=str(d))
    wfdb.wrsamp("1dof_preprocess_finger1_sample1", fs=2048, units=["V"] * 256,
                sig_name=names, p_signal=x, fmt=["16"] * 256,
                write_dir=str(d))
    keys, recs = _read_all("hyser", tmp_path)
    assert len(keys) == 1 and "_raw_" in keys[0]           # preprocess skipped
    assert [r.notes["grid"] for r in recs] == ["ED", "EP", "FD", "FP"]
    assert {r.subject_id for r in recs} == {"subject07"} and recs[0].unit == "mV"
    fd = recs[2]
    assert fd.data[fd.lead_names.index("hyser_r3c5")].std() > 3 * 0.03
    _check("hyser", recs)


def test_reader_cemhsey_zip_and_tree(tmp_path):
    import scipy.io
    import zipfile
    x = np.random.default_rng(3).normal(0, 0.03, (320, 3 * 2048))
    x[64 + 5] *= 5                                    # grid 2, channel 6
    stage = tmp_path / "stage"
    (stage / "S2" / "D1").mkdir(parents=True)
    for name in ("S2_Day1_Session1_Task1_Trial1.mat",
                 "S4_Day1_Session1_Task1_Trial1.mat"):        # 2nd: failed trial
        scipy.io.savemat(stage / "S2" / "D1" / name, {"data_sEMG": x},
                         do_compression=True)
    root = tmp_path / "raw"
    root.mkdir()
    with zipfile.ZipFile(root / "GRASP_S2.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((stage / "S2" / "D1").iterdir()):
            z.write(f, f"S2/D1/{f.name}")
    gest = root / "GESTURE_S2" / "S2" / "D1"
    gest.mkdir(parents=True)
    scipy.io.savemat(gest / "S2_Day1_Trial1.mat", {"data_sEMG": x})
    keys, recs = _read_all("cemhsey_8x8", root)
    assert len(keys) == 2                              # failed trial left out
    assert {r.subject_id for r in recs} == {"grasp_S2", "gesture_S2"}
    assert len(recs) == 6 and all(r.data.shape == (64, 3 * 2048) for r in recs)
    g2 = [r for r in recs if r.record_id.endswith("#g2")][0]
    assert g2.data[5].std() > 3 * 0.03
    assert g2.lead_names[5] == CEMHSEY_8X8_BY_CHANNEL[5]
    _check("cemhsey_8x8", recs)
    _, recs = _read_all("cemhsey_5x13", root)
    assert sorted({r.notes["grid"] for r in recs}) == [4, 5]
    _check("cemhsey_5x13", recs)


def test_reader_putemg_frame_table(tmp_path):
    import h5py
    import pickle
    cols = [f"EMG_{k}" for k in range(1, 25)] + ["TRAJ_1", "subject"]
    n = 3 * 5120
    counts = np.round(np.random.default_rng(4).normal(0, 8, (n, len(cols))))
    counts[:, 9] *= 3                                  # EMG_10 -> r2e2
    dt = np.dtype([("index", "<f8"), ("values_block_3", "<f8", (len(cols),))])
    tab = np.zeros(n, dtype=dt)
    tab["values_block_3"] = counts
    (tmp_path / "Data-HDF5").mkdir()
    path = tmp_path / "Data-HDF5" / "emg_gestures-07-repeats_long-2018.hdf5"
    with h5py.File(path, "w") as h:
        t = h.create_dataset("data/table", data=tab)
        t.attrs["values_block_3_kind"] = np.bytes_(pickle.dumps(cols, protocol=0))
    _, recs = _read_all("putemg", tmp_path)
    r = recs[0]
    assert r.subject_id == "putemg_07" and r.fs == 5120.0 and r.unit == "mV"
    assert np.isclose(r.data.std(), 8 * 5 / 4096 * 1000 / 200, rtol=0.6)
    assert r.data[r.lead_names.index("putemg_r2e2")].std() > 2 * r.data[0].std()
    _check("putemg", recs)


# --------------------------------------------------------------------------- #
# full_pass: every window, every epoch
# --------------------------------------------------------------------------- #

class _Index:
    def __init__(self, counts):
        self.counts = counts

    def by_dataset(self):
        return {d: None for d in self.counts}

    def window_counts(self):
        return dict(self.counts)


def _schedule(counts, rank=0, world=2):
    from physiowave.eeg_c1.data import RouteSchedule
    return RouteSchedule(_Index(counts), weights="full_pass", seed=3,
                         batch_by_route={"W16_2000": 8, "A24_2000": 4,
                                         "G64_2000": 2},
                         num_replicas=world, rank=rank, routes=ROUTES,
                         datasets=PRETRAIN_DATASETS)


def test_full_pass_reads_every_window_once_per_epoch():
    counts = {"emg2pose": 1000, "hyser": 101, "putemg": 5}
    seen = {d: [] for d in counts}
    plans = []
    for rank in (0, 1):
        s = _schedule(counts, rank)
        s.set_epoch(4)
        plans.append(s.plan())
        steps = list(s.steps(counts))
        for (_, d, idx) in steps:
            seen[d].append(idx)
    assert plans[0] == plans[1]                          # rank-independent
    # ceil(n / global batch) steps each: 1000/16, 101/4, 5/8
    assert len(plans[0]) == 63 + 26 + 1
    for d, n in counts.items():
        flat = [i for b in seen[d] for i in b]
        assert set(flat) == set(range(n)), d             # nothing left unseen
        # only the topped-up last batch repeats anything
        assert len(flat) - n < 2 * 8 * 2
    # the two ranks never share a window within a step
    s0, s1 = _schedule(counts, 0), _schedule(counts, 1)
    for a, b in zip(s0.steps(counts), s1.steps(counts)):
        if a[1] != "putemg":                             # 5 < one global batch
            assert not set(a[2]) & set(b[2])


def test_full_pass_resume_and_epochs():
    counts = {"emg2qwerty": 640, "cemhsey_8x8": 90}
    s = _schedule(counts)
    s.set_epoch(1)
    full = list(s.steps(counts))
    s.start_step = 17
    assert list(s.steps(counts)) == full[17:]
    s.set_epoch(2)
    assert [x[2] for x in s.steps(counts)] != [x[2] for x in full]
    # an explicit epoch length caps the plan (smoke tests, --max-steps)
    from physiowave.eeg_c1.data import RouteSchedule
    capped = RouteSchedule(_Index(counts), weights="full_pass",
                           steps_per_epoch=10, routes=ROUTES,
                           datasets=PRETRAIN_DATASETS,
                           batch_by_route={"W16_2000": 8, "G64_2000": 2,
                                           "A24_2000": 4})
    assert capped.steps_per_epoch == 10 and len(list(capped.steps(counts))) == 10


def test_size_for_run_reads_the_corpus_once_over_the_run():
    from physiowave.eeg_c1.data import RouteSchedule
    counts = {"emg2qwerty": 20000, "hyser": 3000, "putemg": 500}
    s = RouteSchedule(_Index(counts), weights="temperature:0.5",
                      batch_by_route={"W16_2000": 8, "A24_2000": 4,
                                      "G64_2000": 2},
                      num_replicas=4, routes=ROUTES,
                      datasets=PRETRAIN_DATASETS)
    steps = s.size_for_run(1.0, 40)
    read = steps * 40 * s.windows_per_step()
    assert sum(counts.values()) <= read < sum(counts.values()) + 40 * 32
    assert s.size_for_run(2.0, 40) >= 2 * steps - 1
