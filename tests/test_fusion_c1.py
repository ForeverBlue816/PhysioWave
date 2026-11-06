"""Multimodal fusion on the C1 encoders: the two preparations, the cores, a run."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "Fusion"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from physiowave.fusion.prep_common import read_edf, write_edf  # noqa: E402

EEG8 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]        # inside E19_256
EEG9 = EEG8 + ["PO5"]                                            # PO5 is on no route


# --------------------------------------------------------------------------- #
# EDF and SeizeIT2
# --------------------------------------------------------------------------- #

def test_edf_round_trip(tmp_path):
    t = np.arange(256 * 10) / 256.0
    sig = {"BTEleft SD": 100 * np.sin(2 * np.pi * 3 * t), "CROSStop SD": 50 * np.cos(2 * np.pi * 7 * t)}
    write_edf(tmp_path / "a.edf", sig, 256.0)
    out, rates, dur = read_edf(str(tmp_path / "a.edf"))
    assert dur == 10.0 and rates["BTEleft SD"] == 256.0
    for k in sig:
        assert np.abs(out[k] - sig[k]).max() < 0.3           # 16-bit over +-5 mV


def _bids(root, subjects=("sub-097", "sub-098", "sub-099"), seconds=600, fs=256):
    rng = np.random.default_rng(0)
    for si, sub in enumerate(subjects):
        for run in (1, 2):
            d = root / sub / "ses-01"
            for m in ("eeg", "ecg", "emg"):
                (d / m).mkdir(parents=True, exist_ok=True)
            stem = f"{sub}_ses-01_task-szMonitoring_run-{run:02d}"
            T = seconds * fs
            write_edf(d / "eeg" / f"{stem}_eeg.edf",
                      {"BTEleft SD": rng.normal(0, 20, T), "CROSStop SD": rng.normal(0, 20, T)}, fs)
            if not (si == 2 and run == 1):                       # one run without ECG/EMG
                write_edf(d / "ecg" / f"{stem}_ecg.edf", {"ECG SD": rng.normal(0, 200, T)}, fs)
                write_edf(d / "emg" / f"{stem}_emg.edf", {"EMG SD": rng.normal(0, 30, T)}, fs)
            with open(d / "eeg" / f"{stem}_events.tsv", "w") as f:
                f.write("onset\tduration\teventType\n")
                f.write(f"0.00\t{seconds:.2f}\tbckg\n200.00\t30.00\tsz_foc_a\n500.00\t10.00\timpd\n")


def test_seizeit2_prep_windows_labels_presence_and_split(tmp_path):
    from seizeit2_prep import main
    raw = tmp_path / "raw"
    _bids(raw)
    out = tmp_path / "out"
    assert main(["--raw-dir", str(raw), "--out-dir", str(out), "--jobs", "1", "--split", "random",
                 "--bg-per-hour", "60", "--val-subjects", "1", "--test-subjects", "1"]) == 0
    import h5py
    f = {s: h5py.File(out / f"{s}.h5", "r") for s in ("train", "val", "test")}
    info = json.load(open(out / "split.json"))
    tr = f["train"]
    assert tr["eeg"].shape[1:] == (2, 1024) and tr["ecg"].shape[1:] == (1, 1000)
    assert tr["emg"].shape[1:] == (1, 8000)
    subs = {s: {x.decode() for x in f[s]["subject"][:]} for s in f}
    assert not (subs["train"] & subs["test"]) and not (subs["val"] & subs["test"])
    allf = [f[s] for s in f]
    labels = np.concatenate([g["label"][:] for g in allf])
    assert set(labels.tolist()) == {0, 1}
    # each run: seizure of 30 s, 4 s windows every 1 s with >= 2 s overlap -> 31 windows
    assert int((labels == 1).sum()) == 31 * 6
    # background: 600 s at 60/h -> 10 per run, none inside the guard band
    assert int((labels == 0).sum()) == 10 * 6
    for g in allf:
        for seg, y in zip(g["segment"][:], g["label"][:]):
            start = float(seg.decode().rsplit("/", 1)[1])
            if y == 0:
                assert not (170 - 4 < start < 260) and not (500 - 4 < start < 510)
    present = np.concatenate([g["present_ecg"][:] for g in allf])
    assert present.sum() == len(present) - 41               # the one run without ECG
    assert info["splits"]["train"]["present"]["eeg"] == info["splits"]["train"]["windows"]
    # the stored windows are normalised the way each encoder expects
    assert abs(float(tr["eeg"][:8].std()) - 1.0) < 0.1
    for g in f.values():
        g.close()


def test_seizeit2_official_split():
    from seizeit2_prep import official_split
    subs = [f"sub-{i:03d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 96, 97, 125)]
    side = official_split(subs, 0.12, 42)
    assert side["sub-097"] == side["sub-125"] == "test"
    assert all(side[s] in ("train", "val") for s in subs[:9])
    assert sum(v == "val" for v in side.values()) == 1                 # 12% of nine


def test_event_scoring_matches_the_reference_rules():
    from physiowave.fusion.seizure_events import evaluate, post_process, score_recording
    n = 3600
    y = np.zeros(n, bool); y[1000:1060] = True; y[2000:2030] = True
    p = np.zeros(n)
    p[1005:1020] = 0.9; p[1021:1050] = 0.9           # a 1 s gap: merged into one event
    p[2500:2520] = 0.8                               # 20 s, no seizure: one false alarm
    p[3000:3005] = 0.9                               # 5 s: shorter than 8 s, dropped
    assert post_process(p, 0.5)[1020] and not post_process(p, 0.5)[3002]
    r = score_recording(p, y, 0.5)
    assert (r["hit"], r["false_events"], r["n_seizures"]) == (1, 1, 2)
    assert r["sens_ovlp"] == 0.5 and r["fa_ovlp_h"] == 1.0
    out = evaluate([(p, y), (np.zeros(1800), np.zeros(1800, bool))])
    ev = out["at_0.5"]["events"]
    assert ev["sens"] == 0.5 and abs(ev["fa_24h"] - 16.0) < 1e-9          # 1 FA in 1.5 h
    assert out["n_seizures"] == 2 and len(out["curve"]) == 51
    # epoch precision after post-processing: 45 s inside the seizure, 20 s outside
    assert abs(r["prec_epoch"] - 45 / 65) < 1e-12
    # the paper's areas: a perfect detector scores 1, one that never fires 0.5 / nan
    yy = np.zeros(600, bool); yy[100:160] = True
    perfect = evaluate([(yy * 0.99, yy)])
    assert abs(perfect["auroc_paper"] - 1.0) < 1e-9 and abs(perfect["aupr_paper"] - 1.0) < 1e-9
    silent = evaluate([(np.zeros(600), yy)])
    assert abs(silent["auroc_paper"] - 0.5) < 1e-9 and silent["aupr_paper"] != silent["aupr_paper"]


# --------------------------------------------------------------------------- #
# MPDB (the EEGLAB reader is swapped for synthetic epochs)
# --------------------------------------------------------------------------- #

def _fake_eeglab(tmp_path):
    """subjects 1 and 4 identical, 2 with a pruned ECG, 3 refused."""
    import mpdb_prep
    rng = np.random.default_rng(1)
    seq = np.array([133, 135, 139, 129, 125, 143, 141, 131, 127, 145] * 2)
    plan = {1: {"eeg": seq, "ecg": seq, "emg": seq},
            2: {"eeg": seq, "ecg": np.delete(seq, [3, 9, 14]), "emg": seq},
            3: {"eeg": seq, "ecg": np.append(seq[:-1], 999), "emg": seq},
            4: {"eeg": seq, "ecg": seq, "emg": seq}}
    names = {"eeg": EEG9, "ecg": ["ECG"], "emg": ["ch1", "ch2", "ch3", "ch4"]}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for s in plan:
        for m in names:
            (tmp_path / f"{m.upper()}_{s}.set").write_bytes(b"")

    def fake(path):
        base = os.path.basename(path)
        m = base.split("_")[0].lower()
        s = int(base.split("_")[1].split(".")[0])
        codes = plan[s][m]
        data = rng.normal(0, 1, (len(codes), len(names[m]), 2000))
        return data, codes, names[m], 1000.0
    return mpdb_prep, fake


def test_mpdb_prep_aligns_epochs_and_splits_by_subject(tmp_path, monkeypatch):
    mpdb_prep, fake = _fake_eeglab(tmp_path)
    monkeypatch.setattr(mpdb_prep, "load_eeglab_epochs", fake)
    out = tmp_path / "out"
    assert mpdb_prep.main(["--raw-dir", str(tmp_path), "--out-dir", str(out),
                           "--val-subjects", "1", "--test-subjects", "1"]) == 0
    import h5py
    info = json.load(open(out / "split.json"))
    dropped = [n for n in info["subjects"] if n.get("dropped")]
    assert [n["subject"] for n in dropped] == [3]
    tot = sum(v["windows"] for v in info["splits"].values())
    assert tot == 57                                          # 20 + 17 (ECG pruned of 3) + 20
    with h5py.File(out / "train.h5", "r") as f:
        assert f["eeg"].shape[1:] == (9, 512) and f["ecg"].shape[1:] == (1, 500)
        assert f["emg"].shape[1:] == (4, 4000)
        assert set(np.unique(f["label"][:]).tolist()) <= {0, 1, 2, 3, 4}
        assert json.loads(f.attrs["eeg_channel_names"]) == EEG9
        assert json.loads(f.attrs["ecg_channel_names"]) == ["patch1"]


def test_mpdb_raw_pairs_events_across_devices(tmp_path, monkeypatch):
    """Continuous files: EEG&ECG and EMG on different clocks, one trigger missed."""
    import mpdb_raw_prep
    rng = np.random.default_rng(7)
    fs = 1000.0
    base = tmp_path / "raw"
    for subj in (1, 2, 3):
        for exp in mpdb_raw_prep.EXPERIMENTS:
            for kind in ("EEG&ECG", "EMG"):
                d = base / str(subj) / kind
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{subj}_{exp}.set").write_bytes(b"")
    codes = {"brake": [139, 141, 145] * 4, "throttle": [137, 143] * 6, "turn": [125, 127] * 5,
             "change": [129, 131] * 5 + [133] * 4}

    def fake(path):
        exp = os.path.basename(path).split("_")[1].split(".")[0]
        kind = os.path.basename(os.path.dirname(path))
        ev_codes = codes[exp]
        times = 10.0 + 20.0 * np.arange(len(ev_codes))          # 20 s apart
        T = int((times[-1] + 20) * fs)
        if kind == "EMG":
            # the EMG device started 3.21 s later, drifts 30 ppm, missed the fifth
            # trigger and logged three simulator codes the other device did not
            ev = [(t * (1 + 30e-6) - 3.21, c) for k, (t, c) in enumerate(zip(times, ev_codes))
                  if k != 4]
            ev += [(5.0, 77), (times[2] * (1 + 30e-6) - 3.21 + 0.4, 79), (times[-1] + 5, 27)]
            ev.sort()
            return rng.normal(0, 30, (4, T)), ["S01", "S05", "S09", "S13"], fs, ev
        ev = list(zip(times.tolist(), ev_codes))
        names = EEG8 + ["ECG"]
        return rng.normal(0, 20, (len(names), T)), names, fs, ev

    monkeypatch.setattr(mpdb_raw_prep, "load_eeglab_raw", fake)
    out = tmp_path / "out"
    assert mpdb_raw_prep.main(["--raw-dir", str(base), "--out-dir", str(out),
                               "--val-subjects", "1", "--test-subjects", "1"]) == 0
    import h5py
    info = json.load(open(out / "split.json"))
    n_events = sum(len(v) for v in codes.values())
    tot = sum(v["windows"] for v in info["splits"].values())
    assert tot == 3 * n_events                              # every EEG event is a window
    present_emg = sum(v["present"]["emg"] for v in info["splits"].values())
    assert present_emg == 3 * (n_events - 4)                # the missed trigger per experiment
    pairing = info["subjects"][0]["experiments"]["brake"]["emg_pairing"]
    assert pairing["paired"] == 11 and pairing["unpaired_eeg"] == 1 and pairing["unused_emg"] == 3
    assert abs(pairing["clock_offset_s"] + 3.21) < 0.01 and abs(pairing["clock_drift_ppm"] - 30) < 5
    assert pairing["max_residual_ms"] < 5
    with h5py.File(out / "train.h5", "r") as f:
        assert f["eeg"].shape[1:] == (8, 1024) and f["ecg"].shape[1:] == (1, 1000)
        assert f["emg"].shape[1:] == (4, 8000)
        assert json.loads(f.attrs["eeg_channel_names"]) == EEG8
        assert set(np.unique(f["label"][:]).tolist()) <= {0, 1, 2, 3, 4}


# --------------------------------------------------------------------------- #
# The cores
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fusion", ["ralf", "concat", "weighted_logit"])
def test_cores_handle_two_tasks_and_missing_modalities(fusion):
    from physiowave.fusion.model import CORES
    torch.manual_seed(0)
    dims = {"eeg": 96, "ecg": 96, "emg": 64}
    core = CORES[fusion](["eeg", "ecg", "emg"], dims, {"a": 5, "b": 2}, embed_dim=32,
                         num_fusion_tokens=4, num_heads=4, depth=1, dropout=0.0).eval()
    feats = {m: torch.randn(3, 4, d) for m, d in dims.items()}
    out = core(feats, "a")
    assert out["logits"].shape == (3, 5) and torch.isfinite(out["logits"]).all()
    assert core(feats, "b")["logits"].shape == (3, 2)
    # a modality absent from the batch, and one absent per item
    assert core({k: v for k, v in feats.items() if k != "ecg"}, "a")["logits"].shape == (3, 5)
    mask = {"eeg": torch.tensor([True, False, True]), "ecg": torch.tensor([False, True, True]),
            "emg": torch.tensor([True, True, False])}
    o = core(feats, "a", mask)
    assert torch.isfinite(o["logits"]).all()
    if fusion == "ralf":
        assert float(o["reliability"]["eeg"][1]) == 0.0
        l = core.reliability_loss(o["reliability"], {m: torch.ones(3) for m in dims},
                                  o["present"])
        assert torch.isfinite(l)
    with pytest.raises(ValueError, match="no available modality"):
        core(feats, "a", {m: torch.zeros(3, dtype=torch.bool) for m in dims})
    assert core.consistency_loss(o["logits"], o["logits"].clone()).abs() < 1e-6


def test_ralf_gate_floor_and_detach():
    from physiowave.fusion.model import RALFCore
    dims = {"eeg": 16, "ecg": 16, "emg": 16}
    torch.manual_seed(0)
    feats = {m: torch.randn(3, 4, 16, requires_grad=True) for m in dims}
    plain = RALFCore(list(dims), dims, {"a": 2}, embed_dim=32, num_heads=4)
    floored = RALFCore(list(dims), dims, {"a": 2}, embed_dim=32, num_heads=4, gate_floor=0.5)
    floored.load_state_dict(plain.state_dict())
    # the floor only changes the gate: reliabilities and unimodal logits agree
    o1, o2 = plain(feats, "a"), floored(feats, "a")
    for m in dims:
        assert torch.allclose(o1["reliability"][m], o2["reliability"][m])
        assert torch.allclose(o1["unimodal_logits"][m], o2["unimodal_logits"][m])
    assert not torch.allclose(o1["fusion_logits"], o2["fusion_logits"])
    with pytest.raises(ValueError, match="gate_floor"):
        RALFCore(list(dims), dims, {"a": 2}, embed_dim=32, num_heads=4, gate_floor=1.0)
    # detached: the fusion logits send no gradient into the reliability head
    det = RALFCore(list(dims), dims, {"a": 2}, embed_dim=32, num_heads=4, gate_detach=True)
    det.load_state_dict(plain.state_dict())
    for core in (plain, det):
        core.zero_grad()
        core(feats, "a")["fusion_logits"].sum().backward()
    g = lambda core: core.reliability_head["eeg"][1].weight.grad
    assert g(plain) is not None and g(plain).abs().sum() > 0
    assert g(det) is None or g(det).abs().sum() == 0


# --------------------------------------------------------------------------- #
# End to end on smoke encoders
# --------------------------------------------------------------------------- #

def _smoke_encoders(tmp_path):
    from physiowave.train.pretrain_main import main as pretrain
    paths = {}
    for m, cfg in (("eeg", "pretrain/eeg_c1_moe"), ("ecg", "pretrain/ecg_c1_moe"),
                   ("emg", "pretrain/emg_c1_moe")):
        out = tmp_path / f"pre_{m}"
        assert pretrain(["--config", cfg, "--smoke-test", "--max-steps", "2",
                         "--output-dir", str(out)]) == 0
        paths[m] = str(out / "latest.pth")
    return paths


@pytest.mark.slow
def test_fusion_trains_evaluates_and_tabulates(tmp_path, monkeypatch):
    import subprocess
    from seizeit2_prep import main as sz_main
    mpdb_prep, fake = _fake_eeglab(tmp_path / "mpdb_raw")
    monkeypatch.setattr(mpdb_prep, "load_eeglab_epochs", fake)
    mp = tmp_path / "c1" / "mpdb"
    assert mpdb_prep.main(["--raw-dir", str(tmp_path / "mpdb_raw"), "--out-dir", str(mp),
                           "--val-subjects", "1", "--test-subjects", "1"]) == 0
    raw = tmp_path / "sz_raw"
    _bids(raw, seconds=300)
    sz = tmp_path / "c1" / "seizeit2"
    assert sz_main(["--raw-dir", str(raw), "--out-dir", str(sz), "--jobs", "1", "--split", "random",
                    "--bg-per-hour", "120", "--val-subjects", "1", "--test-subjects", "1"]) == 0
    enc = _smoke_encoders(tmp_path)

    from physiowave.fusion.train import main as train
    common = ["--config", "fusion/c1_ralf", "--device", "cpu", "--num-workers", "0",
              "--progress", "none", "--epochs", "1", "--max-steps", "2"]
    sets = [f"data.tasks.mpdb.dir={mp}", f"data.tasks.seizeit2.dir={sz}",
            f"encoders.eeg={enc['eeg']}", f"encoders.ecg={enc['ecg']}",
            f"encoders.emg={enc['emg']}", "train.batch_size=4", "train.precision=fp32",
            "model.embed_dim=32", "model.num_heads=4", "model.summary_heads=4",
            "model.route_drop_max=2"]
    runs = tmp_path / "runs"
    for fusion, mode in (("ralf", "frozen"), ("concat", "frozen"), ("weighted_logit", "frozen"),
                         ("ralf", "finetune"), ("ralf", "scratch")):
        out = runs / f"{fusion}_{mode}"
        assert train(common + ["--output-dir", str(out), "--set", *sets,
                               f"model.fusion={fusion}", f"model.encoder_mode={mode}"]) == 0
        res = json.load(open(out / "results.json"))
        assert set(res["tasks"]) == {"mpdb", "seizeit2"}
        for t in res["tasks"]:
            assert "complete" in res["tasks"][t]["test"] and "eeg_only" in res["tasks"][t]["test"]
            assert "balanced_acc" in res["tasks"][t]["test"]["complete"]
        assert "auprc" in res["tasks"]["seizeit2"]["test"]["complete"]
        rep = json.load(open(out / "encoders.json"))
        if mode != "scratch":
            assert rep["mpdb"]["eeg"]["route"] == "E19_256"          # 8 of the 19 slots ...
            assert rep["mpdb"]["eeg"]["dropped_channels"] == ["PO5"]  # ... after leaving PO5 out
            assert rep["seizeit2"]["eeg"]["route"] is None            # bte names: no route
            assert rep["mpdb"]["ecg"]["route"] == "L1_250"            # the chest patch
            assert rep["seizeit2"]["eeg"]["shares_core_with"] == "mpdb"
            assert any(k.startswith("shared_transformer.") for k in rep["mpdb"]["emg"]["taken"])
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "collect_fusion.py"),
                        "--root", str(runs)], capture_output=True, text=True)
    assert r.returncode == 0 and "ralf/frozen" in r.stdout and "seizeit2" in r.stdout


@pytest.mark.slow
def test_seizeit2_official_split_train_and_event_evaluation(tmp_path):
    """Official split -> a fusion run -> continuous event-level evaluation."""
    import subprocess
    from seizeit2_prep import main as sz_main
    raw = tmp_path / "sz_raw"
    _bids(raw, subjects=("sub-001", "sub-002", "sub-003", "sub-004", "sub-097", "sub-098"),
          seconds=300)
    sz = tmp_path / "c1" / "seizeit2"
    assert sz_main(["--raw-dir", str(raw), "--out-dir", str(sz), "--jobs", "1",
                    "--bg-per-hour", "120"]) == 0
    info = json.load(open(sz / "split.json"))
    assert info["split"] == "official"
    assert info["subject_split"]["sub-097"] == info["subject_split"]["sub-098"] == "test"
    enc = _smoke_encoders(tmp_path)
    from physiowave.fusion.train import main as train
    run = tmp_path / "runs" / "ralf_frozen"
    assert train(["--config", "fusion/c1_ralf", "--device", "cpu", "--num-workers", "0",
                  "--progress", "none", "--epochs", "1", "--max-steps", "2",
                  "--output-dir", str(run), "--set", f"data.tasks.seizeit2.dir={sz}",
                  f"encoders.eeg={enc['eeg']}", f"encoders.ecg={enc['ecg']}",
                  f"encoders.emg={enc['emg']}", "train.batch_size=4", "train.precision=fp32",
                  "model.embed_dim=32", "model.num_heads=4", "model.summary_heads=4"]) == 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "Fusion", "seizeit2_events.py"),
                        "--run-dir", str(run), "--raw-dir", str(raw), "--device", "cpu",
                        "--step", "2", "--batch", "64"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    ev = json.load(open(run / "events_cross.json"))
    assert ev["n_runs"] == 4                                   # sub-097, sub-098 x 2 runs
    c = ev["conditions"]["complete"]
    assert c["n_seizures"] == 4 and "sens_at_fa_events" in c and len(c["curve"]) == 51
    # the run without ECG/EMG has no ecg_only condition; the others do
    assert "ecg_only" in ev["conditions"]
    z = np.load(run / "predictions" / "sub-097_run-01.npz")
    assert len(z["label"]) == 300 and z["label"].sum() == 30
    # a cut-off run resumes: the same directory, asked for two epochs, trains one more
    assert not (run / "latest.pth").exists()                   # removed with results.json
    h1 = json.load(open(run / "history.json"))
    (run / "results.json").unlink()
    assert train(["--config", "fusion/c1_ralf", "--device", "cpu", "--num-workers", "0",
                  "--progress", "none", "--epochs", "1", "--max-steps", "2", "--keep-latest",
                  "--output-dir", str(run), "--set", f"data.tasks.seizeit2.dir={sz}",
                  f"encoders.eeg={enc['eeg']}", f"encoders.ecg={enc['ecg']}",
                  f"encoders.emg={enc['emg']}", "train.batch_size=4", "train.precision=fp32",
                  "model.embed_dim=32", "model.num_heads=4", "model.summary_heads=4"]) == 0
    assert (run / "latest.pth").exists()
    assert train(["--config", "fusion/c1_ralf", "--device", "cpu", "--num-workers", "0",
                  "--progress", "none", "--epochs", "2", "--max-steps", "2",
                  "--output-dir", str(run), "--set", f"data.tasks.seizeit2.dir={sz}",
                  f"encoders.eeg={enc['eeg']}", f"encoders.ecg={enc['ecg']}",
                  f"encoders.emg={enc['emg']}", "train.batch_size=4", "train.precision=fp32",
                  "model.embed_dim=32", "model.num_heads=4", "model.summary_heads=4"]) == 0
    h2 = json.load(open(run / "history.json"))
    assert len(h2) == 2 and h2[0]["train"] == h1[0]["train"]      # epoch 0 kept, epoch 1 added
    assert json.load(open(run / "results.json"))["epochs_run"] == 2


# --------------------------------------------------------------------------- #
# EEG baselines: EEGPT and LaBraM under the same protocol
# --------------------------------------------------------------------------- #

def _write_safetensors(path, sd):
    import struct
    header, blobs, off = {}, [], 0
    for k, v in sd.items():
        b = v.detach().contiguous().float().numpy().tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _fake_baseline_checkpoints(tmp_path):
    """Random backbones saved in the released files' layouts: EEGPT's mcae
    checkpoint as safetensors (target_encoder.* beside the other three nets),
    LaBraM's labram-base.pth ({'model': {'student.*', 'lm_head.*', ...}})."""
    from physiowave.fusion.baselines import EEGPTBackbone, LaBraMBackbone
    torch.manual_seed(3)
    e = EEGPTBackbone()
    sd = {f"target_encoder.{k}": v for k, v in e.state_dict().items()}
    sd["encoder.summary_token"] = torch.zeros(1, 4, 512)
    eegpt = tmp_path / "eegpt.safetensors"
    _write_safetensors(eegpt, sd)
    lb = LaBraMBackbone()
    sd = {f"student.{k}": v for k, v in lb.state_dict().items()}
    sd["student.mask_token"] = torch.zeros(1, 1, 200)
    sd["student.lm_head.weight"] = torch.zeros(8192, 200)
    sd["lm_head.weight"] = torch.zeros(8192, 200)
    labram = tmp_path / "labram-base.pth"
    torch.save({"model": sd, "epoch": 0}, labram)
    from physiowave.fusion.baselines import LUNA_SIZES, LUNABackbone
    lu = LUNABackbone(**LUNA_SIZES["luna_base"])
    sd = dict(lu.state_dict())                       # LUNA_base.safetensors: no prefix,
    sd["decoder_head.norm.weight"] = torch.ones(64)  # plus the pretraining-only parts
    sd["channel_emb.embeddings.weight"] = torch.zeros(90, 64)
    sd["mask_token"] = torch.zeros(1, 1, 64)
    sd["cross_attn.temparature"] = torch.tensor(1.0)
    luna = tmp_path / "LUNA_base.safetensors"
    _write_safetensors(luna, sd)
    return {"eegpt": (str(eegpt), e), "labram": (str(labram), lb), "luna_base": (str(luna), lu)}


def test_eeg_baselines_load_freeze_and_decay(tmp_path):
    from physiowave.fusion.baselines import ChannelAdapter, build_baseline
    fake = _fake_baseline_checkpoints(tmp_path)
    x = torch.randn(3, 2, 1024)
    for arch, (path, ref) in fake.items():
        enc, rep = build_baseline(arch, ["bte_sameside", "bte_other"], 256.0, 1024, path,
                                  freeze=True)
        for k, v in ref.state_dict().items():                 # the weights are the file's
            assert torch.equal(enc.backbone.state_dict()[k], v), (arch, k)
        tok = enc.encode(x)
        n = {"eegpt": 16 * 4, "labram": 6 * 4,                # 16 patches x 4 summary / 6 chans x 4 s
             "luna_base": 1024 // 40}[arch]                    # 25 patches of the centred 1000 samples
        assert tok.shape == (3, n, enc.embed_dim)
        tok.sum().backward()
        assert all(p.grad is None for p in enc.backbone.parameters())   # frozen
        assert enc.adapter.weight.grad is not None                     # the adapter trains
        assert rep["adapter"]["to"] == ["F7", "T7", "P7", "F8", "T8", "P8"]
        d, depth = 0.5, enc.depth
        assert enc.layer_lr_scale("adapter.weight", 1.0, d) == 1.0
        assert enc.layer_lr_scale(f"backbone.blocks.{depth - 1}.mlp.fc1.weight", 1.0, d) == d
        assert enc.layer_lr_scale("backbone.blocks.0.norm1.weight", 1.0, d) == d ** depth
        assert enc.layer_lr_scale("backbone.norm.weight", 1.0, d) == 1.0
        first = {"labram": "backbone.patch_embed.conv1.weight", "eegpt": "backbone.chan_embed.weight",
                 "luna_base": "backbone.cross_attn.query_embed"}[arch]
        assert enc.layer_lr_scale(first, 1.0, d) == d ** (depth + 1)
        enc.train()
        assert not enc.backbone.training and enc.adapter.training   # a frozen backbone stays in eval
    a = ChannelAdapter(2, 6)
    with torch.no_grad():
        a.weight.mul_(100)
    a(torch.randn(1, 2, 8))
    assert float(a.weight.detach().flatten(1).norm(dim=1).max()) <= 1.0 + 1e-5   # max-norm 1
    # a checkpoint of the wrong model is refused, not half-loaded
    with pytest.raises(SystemExit):
        build_baseline("eegpt", ["a", "b"], 256.0, 1024, fake["labram"][0], freeze=True)
    with pytest.raises(SystemExit):
        build_baseline("luna_large", ["a", "b"], 256.0, 1024, fake["luna_base"][0], freeze=True)
    # LUNA's positions are LUNA's: MNE's head frame, a bipolar name at the midpoint
    from physiowave.fusion.baselines import electrode_positions
    p = electrode_positions(["T7", "T8", "T7-T8"])
    assert abs(p[0, 0] - (-0.0845)) < 2e-3 and np.allclose(p[2], (p[0] + p[1]) / 2, atol=1e-7)


@pytest.mark.slow
def test_seizeit2_eeg_only_c1_eegpt_labram(tmp_path):
    """Official split -> EEG-only runs of C1 (adapter / native), EEGPT and LaBraM
    -> continuous event-level evaluation of one of them."""
    import subprocess
    from seizeit2_prep import main as sz_main
    from physiowave.train.pretrain_main import main as pretrain
    raw = tmp_path / "sz_raw"
    _bids(raw, subjects=("sub-001", "sub-002", "sub-003", "sub-004", "sub-097", "sub-098"),
          seconds=300)
    sz = tmp_path / "c1" / "seizeit2"
    assert sz_main(["--raw-dir", str(raw), "--out-dir", str(sz), "--jobs", "1",
                    "--bg-per-hour", "120"]) == 0
    pre = tmp_path / "pre_eeg"
    assert pretrain(["--config", "pretrain/eeg_c1_moe", "--smoke-test", "--max-steps", "2",
                     "--output-dir", str(pre)]) == 0
    fake = _fake_baseline_checkpoints(tmp_path)
    from physiowave.fusion.train import main as train
    common = ["--config", "fusion/seizeit2_eeg", "--device", "cpu", "--num-workers", "0",
              "--progress", "none", "--epochs", "1", "--max-steps", "2"]
    small = ["train.batch_size=4", "train.precision=fp32", "model.embed_dim=32",
             "model.summary_heads=4", f"data.tasks.seizeit2.dir={sz}"]
    runs = {
        "c1_frozen": [f"encoders.eeg={pre / 'latest.pth'}", "model.encoder_mode=frozen"],
        "c1native_frozen": [f"encoders.eeg={pre / 'latest.pth'}", "model.encoder_mode=frozen",
                            "model.eeg_adapter=null"],
        "eegpt_frozen": [f"encoders.eeg={fake['eegpt'][0]}", "encoders.eeg_arch=eegpt",
                         "model.encoder_mode=frozen"],
        "labram_finetune": [f"encoders.eeg={fake['labram'][0]}", "encoders.eeg_arch=labram",
                            "model.encoder_mode=finetune"],
        "luna_base_frozen": [f"encoders.eeg={fake['luna_base'][0]}", "encoders.eeg_arch=luna_base",
                             "model.encoder_mode=frozen"],
    }
    for name, sets in runs.items():
        out = tmp_path / "runs" / name
        assert train(common + ["--output-dir", str(out), "--set", *small, *sets]) == 0, name
        res = json.load(open(out / "results.json"))
        test = res["tasks"]["seizeit2"]["test"]
        # one modality: complete is eeg alone, and no subset repeats it
        assert "complete" in test and not {"eeg_only", "eeg_ecg", "ecg_only"} & set(test)
        assert "auprc" in test["complete"]
        rep = json.load(open(out / "encoders.json"))["seizeit2"]["eeg"]
        if name == "c1_frozen":
            assert rep["route"] == "E19_256" and rep["adapter"]["to"][1] == "T7"
        if name == "c1native_frozen":
            assert rep["route"] is None and "adapter" not in rep
        if name.startswith(("eegpt", "labram", "luna")):
            assert rep["arch"] == name.rsplit("_", 1)[0]
    r = subprocess.run([sys.executable, os.path.join(ROOT, "Fusion", "seizeit2_events.py"),
                        "--run-dir", str(tmp_path / "runs" / "eegpt_frozen"),
                        "--raw-dir", str(raw), "--device", "cpu", "--step", "4", "--batch", "64"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    ev = json.load(open(tmp_path / "runs" / "eegpt_frozen" / "events_cross.json"))
    assert ev["n_runs"] == 4 and ev["conditions"]["complete"]["n_seizures"] == 4
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "collect_seizeit2_events.py"),
                        "--root", str(tmp_path / "runs"), "--condition", "complete"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "eegpt_frozen / complete" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# SeizeIT2 figures
# --------------------------------------------------------------------------- #

def _fake_fusion_run(path, raw, fusion="ralf", seed=0):
    """A run directory as train.py + seizeit2_events.py leave it, from synthetic
    recordings: predictions/*.npz, events_cross.json, results.json, history.json."""
    from seizeit2_events import SUBSETS, score, second_labels
    from seizeit2_prep import find_runs, read_events
    rng = np.random.default_rng(seed)
    pred = path / "predictions"
    pred.mkdir(parents=True)
    for r in find_runs(str(raw), "sub-09[78]"):
        n = 300
        y = second_labels(read_events(r["events"]), n)
        base = np.clip(0.1 + 0.7 * y + rng.normal(0, 0.1, n), 0, 1).astype(np.float32)
        d = {"label": y, "gate_cross": np.ones(n, bool), "gate_both": np.ones(n, bool),
             "present": np.array([True, True, True])}
        for c, keep in SUBSETS:
            d[f"prob_{c}"] = np.clip(base * (0.5 + 0.5 * len(keep) / 3), 0, 1).astype(np.float32)
        np.savez_compressed(pred / f"{r['subject']}_run-{r['run']}.npz", **d)
    ev = score(str(pred), "cross")
    json.dump(ev, open(path / "events_cross.json", "w"), default=float)
    conds = ["complete", "eeg_only", "ecg_only", "emg_only", "eeg_ecg", "eeg_emg", "ecg_emg",
             "eeg_noise", "eeg_bad_channel", "eeg_drift", "ecg_saturation", "ecg_noise",
             "emg_dropout", "emg_noise"]
    test = {}
    for i, c in enumerate(conds):
        row = {"auprc": 0.4 - 0.01 * i, "auroc": 0.75, "n": 100}
        for m in ("eeg", "ecg", "emg"):
            corrupted = c.startswith(m + "_") and c != f"{m}_only" and c[len(m) + 1:] not in (
                "ecg", "emg")                                   # eeg_noise yes, eeg_ecg no
            row[f"reliability_{m}"] = 0.3 if corrupted else 0.9
        test[c] = row
    json.dump({"fusion": fusion, "encoder_mode": "frozen", "best_epoch": 1, "seed": 42,
               "corruption_severity": 0.7,
               "tasks": {"seizeit2": {"classes": ["bckg", "sz"], "test": test}}},
              open(path / "results.json", "w"))
    json.dump([{"epoch": e, "val": {"seizeit2": {"auprc": 0.2 + 0.1 * e}}} for e in range(3)],
              open(path / "history.json", "w"))


def test_seizeit2_figures_from_run_artefacts(tmp_path):
    import visualize_seizeit2 as viz
    raw = tmp_path / "raw"
    _bids(raw, subjects=("sub-097", "sub-098"), seconds=300)
    main = tmp_path / "ralf_frozen_x"
    other = tmp_path / "concat_frozen_x"
    _fake_fusion_run(main, raw)
    _fake_fusion_run(other, raw, fusion="concat", seed=1)
    out = tmp_path / "figs"
    assert viz.main(["--raw-dir", str(raw), "--run-dir", str(main), "--compare", str(other),
                     "--eeg-runs", str(other), "--out-dir", str(out), "--format", "png",
                     "--margin", "30"]) == 0
    names = ["fig_seizure_example", "fig_sensitivity_fa", "fig_modality_subsets",
             "fig_reliability", "fig_per_patient", "fig_learning_curves",
             "fig_signal_eeg", "fig_signal_ecg", "fig_signal_emg",
             "fig_window_eeg", "fig_window_ecg", "fig_window_emg"]
    for n in names:
        assert (out / "figures" / f"{n}.png").stat().st_size > 1000, n
        assert (out / "figure_metadata" / f"{n}.json").exists(), n
    meta = json.load(open(out / "figure_metadata" / "fig_seizure_example.json"))
    assert meta["n_candidates"] == 4 and meta["chosen"]["rank"] == 1 and meta["chosen"]["detected"]
    assert meta["chosen"]["duration_s"] == 30
    assert "fig_seizure_example" in (out / "README.md").read_text()
    wm = json.load(open(out / "figure_metadata" / "fig_window_emg.json"))
    assert wm["samples"] == 4 * 256 and wm["seconds"] == 4.0        # the synthetic EMG is 256 Hz
    assert 0 <= wm["window_start_from_onset_s"] <= 30 - 4            # inside the 30 s seizure
    # a named recording is honoured, and an unknown one refused
    assert viz.main(["--raw-dir", str(raw), "--run-dir", str(main), "--out-dir", str(out),
                     "--format", "png", "--only", "fig_seizure_example",
                     "--example", "sub-098_run-02"]) == 0
    assert json.load(open(out / "figure_metadata" / "fig_seizure_example.json"))["recording"] == "sub-098_run-02"
    with pytest.raises(SystemExit):
        viz.main(["--raw-dir", str(raw), "--run-dir", str(main), "--out-dir", str(out),
                  "--format", "png", "--only", "fig_seizure_example", "--example", "sub-001_run-01"])
