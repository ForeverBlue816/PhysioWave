"""Continued pretraining of the EEG C1 encoder on Siena mixed with the original corpora."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.eeg_c1.data import CorpusIndex, RouteSchedule, ShardInfo  # noqa: E402
from physiowave.eeg_c1.routes import (CONTINUATION_DATASETS, EEG_DATASETS,  # noqa: E402
                                      PRETRAIN_DATASETS, datasets_with)

SIENA_LABELS = ["EEG Fp1", "EEG F3", "EEG C3", "EEG P3", "EEG O1", "EEG F7", "EEG T3",
                "EEG T5", "EEG Fc1", "EEG Fz", "EEG Cz", "EEG Pz", "EEG Fp2", "EEG F4",
                "EEG C4", "EEG P4", "EEG O2", "EEG F8", "EEG T4", "EEG T6", "EKG EKG", "SPO2"]


def test_siena_is_a_continuation_corpus_only():
    assert "siena" in CONTINUATION_DATASETS and "siena" not in PRETRAIN_DATASETS
    assert set(EEG_DATASETS) == set(PRETRAIN_DATASETS) | {"siena"}
    assert "siena" in datasets_with(["siena"]) and "siena" not in datasets_with()
    with pytest.raises(SystemExit):
        datasets_with(["deap"])
    from physiowave.eeg_c1.preprocess import map_to_slots
    m = map_to_slots(SIENA_LABELS, CONTINUATION_DATASETS["siena"].slots)
    assert int(m.valid.sum()) == 19                      # every 10-20 slot from Siena's labels


def test_fixed_share_and_epoch_sized_by_one_corpus():
    counts = {"tueg": 1_000_000, "siena": 120_000, "hbn": 400_000, "physionet_mi": 39_000}
    table = datasets_with(["siena"])
    index = CorpusIndex([ShardInfo(f"/x/{d}.h5", d, table[d].route_id, n)
                         for d, n in counts.items()])
    s = RouteSchedule(index, weights="temperature:0.5", seed=0, num_replicas=4,
                      datasets=table, fixed_weights={"siena": 0.3}, epoch_pass_of="siena")
    assert abs(s.weights["siena"] - 0.3) < 1e-12 and abs(sum(s.weights.values()) - 1) < 1e-12
    # the other three keep temperature:0.5's proportions among themselves
    plain = RouteSchedule(index, weights="temperature:0.5", seed=0, num_replicas=4,
                          datasets={k: v for k, v in table.items() if k != "siena"})
    for d in ("tueg", "hbn", "physionet_mi"):
        assert abs(s.weights[d] / 0.7 - plain.weights[d]) < 1e-9
    # an epoch is one read of Siena at its share: steps * 0.3 * batch * replicas = n
    b = s.batch_by_route["E19_256"] * 4
    assert abs(s.steps_per_epoch * 0.3 * b / counts["siena"] - 1) < 0.01
    with pytest.raises(SystemExit):                       # no room left for the others
        RouteSchedule(index, weights="temperature:0.5", datasets=table,
                      fixed_weights={"siena": 1.0})
    with pytest.raises(SystemExit):                       # a corpus the manifest lacks
        RouteSchedule(index, weights="temperature:0.5", datasets=table,
                      fixed_weights={"faced": 0.2})


@pytest.mark.slow
def test_siena_preprocess_merge_and_continue_from_a_checkpoint(tmp_path):
    """Siena-layout EDFs -> the siena adapter -> merged with a (smoke) TUEG corpus
    -> continued pretraining from a checkpoint with configs/pretrain/eeg_c1_siena."""
    sys.path.insert(0, os.path.join(ROOT, "Fusion"))
    from physiowave.fusion.prep_common import write_edf
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    for k in range(6):
        d = raw / f"PN{k:02d}"
        d.mkdir(parents=True)
        write_edf(d / f"PN{k:02d}-1.edf", {n: rng.normal(0, 30, 512 * 120) for n in SIENA_LABELS},
                  512.0)
    corpus = tmp_path / "corpus"
    py = sys.executable

    def run(*cmd):
        r = subprocess.run([py, *cmd], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout + r.stderr
    out = run("EEG/preprocess_pretrain_corpus.py", "--dataset", "siena", "--root", str(raw),
              "--out-dir", str(corpus / "siena"), "--val-fraction", "0.2")
    assert "6 recordings, 180 windows, 6 subjects" in out
    rec = json.loads(open(corpus / "siena" / "manifest_train.jsonl").readline())
    assert rec["dataset_id"] == "siena" and rec["route_id"] == "E19_256" and rec["empty_slots"] == []
    run("EEG/preprocess_pretrain_corpus.py", "--dataset", "smoke", "--smoke-test", "--smoke-as",
        "tueg", "--out-dir", str(corpus / "tueg"), "--smoke-subjects", "6", "--mains-hz", "50")
    # the verified original corpus first, then Siena added on top of it: the base
    # rows are kept as they are and only Siena's shards are checked
    run("scripts/build_eeg_c1_manifest.py", "--corpus-root", str(corpus), "--datasets",
        "tueg", "--out-dir", str(corpus / "merged"), "--check-shards")
    out = run("scripts/build_eeg_c1_manifest.py", "--corpus-root", str(corpus), "--base-dir",
              str(corpus / "merged"), "--datasets", "siena", "--out-dir",
              str(corpus / "merged_siena"), "--check-shards", "--drop-unreadable")
    n_siena = sum(1 for s in ("train", "val")
                  for _ in open(corpus / "siena" / f"manifest_{s}.jsonl"))
    assert f"checking {n_siena} shard(s)" in out                 # Siena only, not the base
    ids = [json.loads(line)["dataset_id"]
           for line in open(corpus / "merged_siena" / "manifest_train.jsonl")]
    assert set(ids) == {"tueg", "siena"}
    r = subprocess.run([py, "scripts/build_eeg_c1_manifest.py", "--corpus-root", str(corpus),
                        "--base-dir", str(corpus / "merged"), "--datasets", "tueg",
                        "--out-dir", str(tmp_path / "x")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "both in --base-dir" in r.stderr
    run("-m", "physiowave.train.pretrain_main", "--config", "pretrain/eeg_c1_moe", "--smoke-test",
        "--max-steps", "2", "--output-dir", str(tmp_path / "init"))
    out = run("-m", "physiowave.train.pretrain_main", "--config", "pretrain/eeg_c1_siena",
              "--init-from", str(tmp_path / "init" / "latest.pth"),
              "--output-dir", str(tmp_path / "cont"), "--set",
              f"data.manifest_train={corpus / 'merged_siena' / 'manifest_train.jsonl'}",
              f"data.manifest_val={corpus / 'merged_siena' / 'manifest_val.jsonl'}",
              "model.embed_dim=96", "model.depth=2", "model.num_heads=4",
              "model.channel_embed_dim=32", "train.precision=fp32",
              "train.batch_size_by_route.E19_256=4", "train.val_max_batches_per_dataset=2")
    assert "initialised from" in out and "weights only" in out
    assert "fixed {'siena': 0.3}" in out
    assert "epoch 1 done (2/2)" in out
    hist = json.load(open(tmp_path / "cont" / "history.json"))
    assert len(hist) == 2
    assert (tmp_path / "cont" / "latest.pth").exists()
