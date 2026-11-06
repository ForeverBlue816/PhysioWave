"""
ECG C1 downstream: the benchmark converters, multi-label fine-tuning, the
ECG encoder transfer and the results table.
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
for p in (ROOT, os.path.join(ROOT, "ECG"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from physiowave.ecg_c1.leads import LEADS_12                      # noqa: E402

PY = sys.executable
LEADS = list(LEADS_12)


def _ecg(seconds, seed, fs=500):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    beat = np.exp(-((t % 0.8) - 0.3) ** 2 / 0.0005)           # R peaks, 75 bpm
    x = np.stack([beat * rng.uniform(0.5, 1.5) + rng.normal(0, 0.02, t.size)
                  for _ in range(12)])
    return x.T                                                  # [T, 12] mV


def _write(dirpath, name, seconds, seed, dx=None):
    wfdb = pytest.importorskip("wfdb")
    os.makedirs(dirpath, exist_ok=True)
    wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=LEADS,
                p_signal=_ecg(seconds, seed), fmt=["16"] * 12,
                comments=[f"Dx: {dx}"] if dx else [], write_dir=str(dirpath))


def _run(task, raw, out):
    from ecg_downstream_prep import main
    assert main(["--task", task, "--raw-dir", str(raw), "--out-dir", str(out),
                 "--jobs", "1"]) == 0
    import h5py
    files = {s: h5py.File(out / f"{s}.h5", "r") for s in ("train", "val", "test")}
    return files, json.load(open(out / "split.json"))


def test_ptbxl_labels_folds_and_layout(tmp_path):
    import pandas as pd
    raw = tmp_path / "raw"
    rows = []
    scp = [("NORM", "NORM"), ("IMI", "MI"), ("NDT", "STTC"), ("LAFB", "CD"),
           ("LVH", "HYP")]
    for i in range(1, 21):
        stem = f"records500/00000/{i:05d}_hr"
        _write(raw / "records500/00000", f"{i:05d}_hr", 10, i)
        codes = {"NORM": 100.0} if i % 3 else {"NORM": 100.0, "LVH": 80.0, "IMI": 50.0}
        rows.append({"ecg_id": i, "patient_id": i, "scp_codes": str(codes),
                     "strat_fold": 1 + (i - 1) % 10, "filename_hr": stem})
    pd.DataFrame(rows).to_csv(raw / "ptbxl_database.csv", index=False)
    pd.DataFrame({"code": [c for c, _ in scp], "diagnostic": [1] * 5,
                  "diagnostic_class": [k for _, k in scp]}).set_index("code") \
        .to_csv(raw / "scp_statements.csv")

    f, info = _run("ptbxl", raw, tmp_path / "single")
    assert f["train"]["data"].shape[1:] == (12, 5000)
    assert [c.decode() for c in f["train"]["channel_names"][:]] == LEADS
    assert f["train"].attrs["sampling_rate"] == 500.0
    assert info["splits"]["test"]["records"] == 2          # folds 10 of 1..20
    assert info["splits"]["val"]["records"] == 2
    labels = np.concatenate([f[s]["label"][:] for s in f])
    # NORM+LVH(80) -> HYP (latest in NORM<MI<STTC<CD<HYP); IMI at 50 ignored
    assert set(labels.tolist()) == {0, 4}
    w = f["train"]["data"][0]
    assert abs(float(w.std()) - 1.0) < 0.2                  # window_shared scale

    f2, info2 = _run("ptbxl_super", raw, tmp_path / "multi")
    y = np.concatenate([f2[s]["label"][:] for s in f2])
    assert y.shape[1] == 5 and y[:, 1].sum() > 0           # IMI counts at any likelihood


def test_cpsc_windows_records_and_short_padding(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "cpsc_2018/g1", "A0001", 25, 1, "164889003")          # AF, 25 s
    _write(raw / "cpsc_2018/g1", "A0002", 6, 2, "59118001,164884008")  # RBBB+PVC, 6 s
    _write(raw / "cpsc_2018/g1", "A0003", 10, 3, "999999")             # none of the 9
    for i in range(4, 14):
        _write(raw / "cpsc_2018_extra/g1", f"Q{i:04d}", 10, i, "426783006")
    f, info = _run("cpsc2018", raw, tmp_path / "out")
    total = sum(info["splits"][s]["records"] for s in info["splits"])
    assert total == 12                                     # A0003 left out
    recs = np.concatenate([[r.decode() if isinstance(r, bytes) else r
                            for r in f[s]["record"][:]] for s in f])
    labels = np.concatenate([f[s]["label"][:] for s in f])
    idx = [i for i, r in enumerate(recs) if r == "cpsc_A0001"]
    assert len(idx) == 4                                    # 25 s, 10 s windows, 5 s stride
    assert labels[idx[0]].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0]
    j = [i for i, r in enumerate(recs) if r == "cpsc_A0002"]
    assert len(j) == 1 and labels[j[0]].tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 0]
    assert not any(info["subject_overlap"].values())


def test_chapman_merged_classes(tmp_path):
    raw = tmp_path / "raw" / "WFDBRecords" / "01" / "010"
    dx = ["426177001", "164890007", "713422000", "427084000,426783006", "251146004"]
    for i, d in enumerate(dx * 3):
        _write(raw, f"JS{i:05d}", 10, i, d)
    f, info = _run("chapman", tmp_path / "raw", tmp_path / "out")
    assert sum(v["records"] for v in info["splits"].values()) == 12   # 3 unlabelled out
    y = np.concatenate([f[s]["label"][:] for s in f])
    assert y.shape[1] == 4 and set(map(tuple, y.astype(int).tolist())) == {
        (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)}


def test_multilabel_metrics_and_thresholds():
    from physiowave.train.finetune_main import (by_record, multilabel_metrics,
                                                tune_thresholds)
    rng = np.random.default_rng(0)
    y = (rng.random((200, 3)) < 0.3).astype(np.float32)
    p = np.clip(y * 0.4 + rng.random((200, 3)) * 0.5, 0, 1)     # informative, < 0.5 often
    th = tune_thresholds(p, y)
    m = multilabel_metrics(p, y, th)
    assert m["auroc"] > 0.8
    assert m["micro_f1_tuned"] >= m["micro_f1"]                 # tuned on the same data
    P, Y = by_record(np.array([[0.2], [0.4], [0.9]]), np.array([[0.], [0.], [1.]]),
                     ["a", "a", "b"])
    assert np.allclose(P[:, 0], [0.3, 0.9]) and Y[:, 0].tolist() == [0.0, 1.0]


def _smoke_encoder(tmp_path):
    from physiowave.train.pretrain_main import main as pretrain
    out = tmp_path / "pre"
    assert pretrain(["--config", "pretrain/ecg_c1_moe", "--smoke-test",
                     "--max-steps", "2", "--output-dir", str(out)]) == 0
    import yaml
    m = yaml.safe_load(open(out / "config_resolved.yaml"))["model"]
    enc = tmp_path / "enc.pth"
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "export_eeg_pretrained_encoder.py"),
                        "--checkpoint", str(out / "latest.pth"), "--route", "L12_500",
                        "--output", str(enc)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return enc, m


@pytest.mark.slow
def test_ecg_encoder_transfers_whole_and_eeg_is_refused(tmp_path):
    from physiowave.ecg_c1.downstream import ECGC1Downstream
    enc, m = _smoke_encoder(tmp_path)
    kw = dict(in_channels=12, window_samples=5000, sampling_rate=500,
              patch_samples=250, num_classes=4, channel_names=LEADS,
              route_id="L12_500", embed_dim=m["embed_dim"], depth=m["depth"],
              num_heads=m["num_heads"], channel_embed_dim=m["channel_embed_dim"])
    model = ECGC1Downstream(**kw)
    rep = model.load_pretrained(str(enc))
    desc = model.describe_transfer(rep)
    assert rep["route_reused"] == ["L12_500"] and "gate yes" in desc
    assert any(k.startswith("wavelet_frontend.") for k in rep["taken"])
    assert any(k.startswith("patch_embed.") for k in rep["taken"])
    out = model(torch.randn(2, 12, 5000),
                type("M", (), {"channel_names": LEADS, "channel_mask": None})())
    assert out["logits"].shape == (2, 4)

    probe = ECGC1Downstream(**kw, freeze_encoder=True)
    probe.load_pretrained(str(enc))
    trainable = {n.split(".")[0] for n, p in probe.named_parameters() if p.requires_grad}
    assert trainable == {"head", "head_norm"} or trainable <= {"head", "head_norm"}

    # an EEG-vocabulary checkpoint must not load into the ECG model
    ck = torch.load(enc, map_location="cpu", weights_only=False)
    from channel_embedding import vocab_payload
    ck.update(vocab_payload())
    torch.save(ck, tmp_path / "eeg_like.pth")
    with pytest.raises(SystemExit):
        ECGC1Downstream(**kw).load_pretrained(str(tmp_path / "eeg_like.pth"))


@pytest.mark.slow
def test_multilabel_finetune_end_to_end_and_table(tmp_path):
    raw = tmp_path / "raw"
    codes = ["164889003", "59118001", "426783006", "164884008"]
    for i in range(40):
        _write(raw / "cpsc_2018/g1", f"A{i:04d}", 12, i, codes[i % 4])
    data = tmp_path / "c1" / "cpsc2018"
    from ecg_downstream_prep import main as prep
    assert prep(["--task", "cpsc2018", "--raw-dir", str(raw), "--out-dir", str(data),
                 "--jobs", "1"]) == 0
    enc, m = _smoke_encoder(tmp_path)
    from physiowave.train.finetune_main import main as finetune
    run = tmp_path / "runs" / "cpsc2018_ft"
    assert finetune(["--config", "finetune/ecg_c1_cpsc2018", "--data-dir", str(data),
                     "--num-classes", "9", "--output-dir", str(run), "--epochs", "1",
                     "--batch-size", "8", "--precision", "fp32", "--device", "cpu",
                     "--num-workers", "0", "--progress", "none",
                     "--set", f"model.ecg_c1.pretrained={enc}",
                     f"model.embed_dim={m['embed_dim']}", f"model.depth={m['depth']}",
                     f"model.num_heads={m['num_heads']}",
                     f"model.channel_embed_dim={m['channel_embed_dim']}"]) == 0
    res = json.load(open(run / "results.json"))
    assert res["multilabel"] and res["pretrained"] and len(res["thresholds"]) == 9
    for k in ("auroc", "micro_f1", "macro_f1_tuned", "n_records", "window_auroc"):
        assert k in res["test"], k
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "collect_ecg_downstream.py"),
                        "--root", str(tmp_path / "runs")], capture_output=True, text=True)
    assert r.returncode == 0 and "CPSC 2018" in r.stdout and "0.7709" in r.stdout


def test_cpsc_kaggle_layout_and_abbreviated_labels(tmp_path):
    # PhysioNet's Kaggle copies: Training_WFDB (CPSC 2018) and Training_2
    # (CPSC-Extra); early releases write "AF", "I-AVB", "Normal" on the Dx line.
    raw = tmp_path / "raw"
    _write(raw / "Training_WFDB", "A0001", 10, 1, "AF")
    _write(raw / "Training_WFDB", "A0002", 10, 2, "I-AVB,PVC")
    _write(raw / "Training_WFDB", "A0003", 10, 3, "Normal")
    for i in range(1, 8):
        _write(raw / "Training_2", f"Q{i:04d}", 10, 10 + i, "426783006")
    from ecg_downstream_prep import CPSC_CLASSES, list_cpsc
    recs = {r["record"]: r["label"] for r in list_cpsc(str(raw))}
    names = [c for c, _ in CPSC_CLASSES]
    assert len(recs) == 10
    assert recs["cpsc_A0001"][names.index("AF")] == 1
    assert recs["cpsc_A0002"][[names.index("IAVB"), names.index("PVC")]].tolist() == [1, 1]
    assert recs["cpsc_A0003"][names.index("SNR")] == 1 and recs["cpsc_A0003"].sum() == 1
