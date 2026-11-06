"""EEGPT benchmark converters: BCIC-IV-2a/2b and KaggleERN.

The failures worth testing are the silent ones -- an epoch cut at the wrong
offset, a label joined to the wrong trial, an alignment computed on a
rank-deficient covariance -- because each yields a file that trains, scores
near chance, and says nothing about why. So the synthetic data here PLANTS a
signal at a known latency on known trials, and the tests check it comes out
where and on what it should.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "EEG"))
import eegpt_bench_common as bc                                  # noqa: E402

ERN_NAMES = ("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,"
             "FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,"
             "CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,"
             "O1,O2").split(",")


# --------------------------------------------------------------------------- #
# Shared steps
# --------------------------------------------------------------------------- #

def test_the_pipeline_is_the_pretraining_one_call_for_call():
    """Same functions, same defaults, same order as process_recording.

    Checked against a direct sequence of physiowave.eeg_c1.preprocess calls
    with PreprocessConfig's defaults, so a later edit to either side that makes
    them disagree fails here rather than in a transfer result.
    """
    from physiowave.eeg_c1 import preprocess as pp
    cfg = pp.PreprocessConfig(notch_hz=50.0)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, (4, 2500)) + np.linspace(0, 50, 2500)   # with a drift
    ref = pp.to_microvolts(x, "uV")
    ref = pp.detrend(ref, cfg.detrend)
    ref = pp.notch(ref, 250.0, 50.0, cfg.notch_harmonics, cfg.notch_quality)
    ref = pp.highpass(ref, 250.0, cfg.highpass_hz)
    ref = pp.resample_to(ref, 250.0, 256)
    assert np.allclose(bc.pretrain_pipeline(x, 250.0), ref)
    assert bc.CFG.highpass_hz == 0.5 and bc.CFG.clip_sigma == 20.0


def test_windows_are_z_scored_per_window_and_channel_and_clipped():
    rng = np.random.default_rng(1)
    w = rng.normal(3.0, 7.0, (5, 4, 256))
    w[0, 0, 10] = 1e4                                  # an artefact
    z = bc.normalise_windows(w)
    assert np.allclose(z[1:].mean(-1), 0, atol=1e-5)
    assert np.allclose(z[1:].std(-1), 1, atol=1e-4)
    assert z.max() <= 20.0                             # pretraining's clip


def test_ea_whitens_the_session_covariance():
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(6, 6))
    x = np.einsum("dc,nct->ndt", mix, rng.normal(size=(40, 6, 200)))
    y = bc.euclidean_alignment(x)
    R = np.einsum("nct,ndt->cd", y, y) / len(y)
    assert np.allclose(R, np.eye(6), atol=1e-6)


def test_ea_refuses_a_common_average_applied_first():
    """CAR makes the channels sum to zero, so R loses a rank -- singular on
    3 channels, singular to rounding on 22 where R^{-1/2} still looks finite."""
    rng = np.random.default_rng(1)
    for C in (3, 22):
        x = bc.common_average(rng.normal(size=(30, C, 200)))
        with pytest.raises(ValueError, match="rank-deficient"):
            bc.euclidean_alignment(x)


def test_the_eegpt_session_step_aligns_then_references():
    """EA first, CAR second: the output sums to zero over channels."""
    rng = np.random.default_rng(2)
    out = bc.eegpt_session(rng.normal(size=(20, 5, 256)) * 30)
    assert np.allclose(out.sum(axis=1), 0, atol=1e-4)


def test_an_epoch_running_off_the_recording_is_refused():
    with pytest.raises(ValueError, match="runs off"):
        bc.epoch_at(np.zeros((2, 100)), [90], 20)


def test_the_split_holds_out_one_test_subject_and_one_validation_subject():
    sp = bc.loso_split(list(range(1, 10)), fold=3)
    assert sp["test"] == [4]
    assert len(sp["val"]) == 1 and sp["val"][0] != 4
    assert sorted(sp["train"] + sp["val"] + sp["test"]) == list(range(1, 10))
    assert bc.loso_split(list(range(1, 10)), fold=3) == sp        # deterministic


# --------------------------------------------------------------------------- #
# BCIC-IV-2a / 2b, from BNCI-shaped .mat files
# --------------------------------------------------------------------------- #

def _write_bnci(path, n_eeg, n_eog, cue_s, rng, runs=2, trials=8, n_cls=4,
                calib_runs=0):
    """A BNCI 001/004-2014-shaped file.

    Class c gets a 10 Hz burst on channel c, starting exactly at the CUE --
    trial start + cue_s -- and lasting 1 s. Nothing happens before the cue.
    """
    fs, trial_len = 250, int(8 * 250)
    data = []
    for _ in range(calib_runs):                      # 2a's EOG calibration runs
        data.append({"X": rng.normal(0, 5, (3000, n_eeg + n_eog)),
                     "trial": np.zeros(0), "y": np.zeros(0), "fs": fs,
                     "classes": np.array(["a"], dtype=object),
                     "artifacts": np.zeros(0)})
    for _ in range(runs):
        T = trials * trial_len + 500
        X = rng.normal(0, 5, (T, n_eeg + n_eog))
        starts = 1 + 100 + np.arange(trials) * trial_len          # 1-based
        y = 1 + rng.integers(0, n_cls, trials)
        t = np.arange(fs) / fs
        for s, lab in zip(starts, y):
            on = s - 1 + int(cue_s * fs)
            X[on:on + fs, lab - 1] += 60 * np.sin(2 * np.pi * 10 * t)
        data.append({"X": X, "trial": starts, "y": y, "fs": fs,
                     "classes": np.array(["c"] * n_cls, dtype=object),
                     "artifacts": np.zeros(trials)})
    import scipy.io as sio
    sio.savemat(path, {"data": np.array(data, dtype=object)})


@pytest.mark.parametrize("prep", ["pretrain", "eegpt"])
@pytest.mark.parametrize("dataset,n_eeg,n_eog,cue,n_cls,calib", [
    ("2a", 22, 3, 2.0, 4, 3), ("2b", 3, 3, 3.0, 2, 0)])
def test_bcic_epochs_start_at_the_cue_and_carry_their_labels(
        tmp_path, dataset, n_eeg, n_eog, cue, n_cls, calib, prep):
    rng = np.random.default_rng(7)
    raw = tmp_path / "raw"
    raw.mkdir()
    pre = "A" if dataset == "2a" else "B"
    for s in (1, 2, 3):
        for part in "TE":
            _write_bnci(str(raw / f"{pre}{s:02d}{part}.mat"), n_eeg, n_eog, cue,
                        rng, n_cls=n_cls, calib_runs=calib)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "EEG/bcic_iv2_finetune.py", "--dataset", dataset,
         "--raw-dir", str(raw), "--out-dir", str(out), "--fold", "0",
         "--subjects", "1,2,3", "--prep", prep],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]

    with h5py.File(out / "test.h5") as f:
        X, y = f["data"][:], f["label"][:]
        assert f.attrs["sampling_rate"] == 256
        assert f.attrs["prep_version"] == bc.PREPS[prep]
        assert X.shape[1:] == (n_eeg, 1024)
        assert set(np.unique(f["subject"][:])) == {1}          # fold 0 -> subject 1
        assert set(np.unique(f["session"][:])) == {0, 1}       # both T and E
    # 2 files x 2 runs x 8 trials; the calibration runs contribute nothing.
    assert len(y) == 32 and set(np.unique(y)) <= set(range(n_cls))
    # The burst sits on the labelled channel in the FIRST second after the cue
    # and nowhere else: the epoch starts at the cue, and label c is channel c.
    power = (X ** 2)
    first, rest = power[:, :, :256].mean(-1), power[:, :, 300:].mean(-1)
    for i, lab in enumerate(y):
        # The burst shows as power concentrated in the first second after the
        # cue. Under the pretraining prep (per-window z-score) it also stays on
        # its own channel; EA + CAR mix channels, so only timing is checked.
        if prep == "pretrain":
            assert first[i].argmax() == lab, f"epoch {i}: label {lab}"
            assert first[i, lab] > 3 * rest[i, lab]
        else:
            # EA whitens: the burst channels dominate the session covariance,
            # so R^{-1/2} scales them down and total power says little. The
            # burst's OWN channel still carries it in the first second.
            assert first[i, lab] > 3 * rest[i, lab], f"epoch {i}: label {lab}"


# --------------------------------------------------------------------------- #
# KaggleERN, from competition-shaped CSVs
# --------------------------------------------------------------------------- #

def _write_ern(root, rng, train=(2, 6, 7), test=(1,), break_session=None,
               test_ids=True):
    import pandas as pd
    tr_rows, te_rows = [], []

    def session(folder, s, k, rows):
        n_fb = 10
        T = 200 * (4 + 4 * n_fb)
        x = rng.normal(0, 20, (T, 56)) + 50
        fb = np.zeros(T, int)
        pos = (np.arange(n_fb) * 800 + 400).astype(int)
        fb[pos] = 1
        y = rng.random(n_fb) < 0.6
        for i, p in enumerate(pos):          # errors: a dip 0.2-0.5 s after
            if not y[i]:
                x[p + 40:p + 100, 18:22] -= 40
        # As the real files spell it: PO8 as "P08", digit zero.
        df = pd.DataFrame(x, columns=["P08" if c == "PO8" else c for c in ERN_NAMES])
        df.insert(0, "Time", np.arange(T) / 200)
        df["EOG"] = 0.0
        df["FeedBackEvent"] = fb
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        df.to_csv(os.path.join(root, folder, f"Data_S{s:02d}_Sess{k:02d}.csv"),
                  index=False)
        rows += [(f"S{s:02d}_Sess{k:02d}_FB{i + 1:03d}", int(y[i]))
                 for i in range(n_fb)]

    for s in train:
        for k in range(1, 6):
            session("train", s, k, tr_rows)
    for s in test:
        for k in range(1, 6):
            session("test", s, k, te_rows)
    if break_session:
        tr_rows = [r for r in tr_rows if not r[0].startswith(break_session)
                   or not r[0].endswith("FB010")]
    pd.DataFrame(tr_rows, columns=["IdFeedBack", "Prediction"]).to_csv(
        os.path.join(root, "TrainLabels.csv"), index=False)
    pd.DataFrame({"label": [r[1] for r in te_rows]}).to_csv(
        os.path.join(root, "true_labels.csv"), index=False)
    if test_ids:
        pd.DataFrame({"IdFeedBack": [r[0] for r in te_rows], "Prediction": 0}
                     ).to_csv(os.path.join(root, "SampleSubmission.csv"), index=False)


def _ern(raw, out, *extra):
    return subprocess.run(
        [sys.executable, "EEG/kaggle_ern_finetune.py", "--raw-dir", str(raw),
         "--out-dir", str(out), "--train-subjects", "2,6,7",
         "--test-subjects", "1", *extra],
        cwd=ROOT, capture_output=True, text=True)


def test_ern_epochs_are_timed_to_feedback_and_labels_follow_them(tmp_path):
    _write_ern(str(tmp_path / "raw"), np.random.default_rng(3))
    r = _ern(tmp_path / "raw", tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    for split in ("train", "test"):
        with h5py.File(tmp_path / "out" / f"{split}.h5") as f:
            X, y = f["data"][:], f["label"][:]
            names = [c.decode() for c in f["channel_names"][:]]
        assert X.shape[1:] == (56, 512)
        assert "PO8" in names and "P08" not in names
        ch = [names.index(c) for c in ("FC3", "FC1", "FCz", "FC2")]
        onset = round(0.7 * 256)
        win = X[:, ch, onset + 51:onset + 128].mean(axis=(1, 2))
        # The planted dip is on the error trials, after the feedback -- if the
        # epochs or the labels were shifted, it would land on the wrong class.
        assert win[y == 0].mean() < -0.5
        assert abs(win[y == 1].mean()) < 0.2


def test_ern_skips_and_names_a_session_whose_labels_do_not_match(tmp_path):
    _write_ern(str(tmp_path / "raw"), np.random.default_rng(4),
               break_session="S07_Sess05")
    r = _ern(tmp_path / "raw", tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "SKIPPED Data_S07_Sess05.csv" in r.stderr
    split = json.loads((tmp_path / "out" / "split.json").read_text())
    assert any("S07_Sess05" in s[0] for s in split["skipped_sessions"])


def test_ern_eegpt19_selects_their_electrodes_in_their_order(tmp_path):
    _write_ern(str(tmp_path / "raw"), np.random.default_rng(5))
    r = _ern(tmp_path / "raw", tmp_path / "out", "--channels", "eegpt19")
    assert r.returncode == 0, r.stderr[-2000:]
    with h5py.File(tmp_path / "out" / "test.h5") as f:
        names = [c.decode() for c in f["channel_names"][:]]
    assert names[:3] == ["Fp1", "Fp2", "F7"] and len(names) == 19


def test_ern_refuses_the_kaggle_test_set_without_its_labels(tmp_path):
    _write_ern(str(tmp_path / "raw"), np.random.default_rng(6))
    os.remove(tmp_path / "raw" / "true_labels.csv")
    r = _ern(tmp_path / "raw", tmp_path / "out", "--test-split", "kaggle")
    assert r.returncode != 0 and "true_labels.csv" in (r.stdout + r.stderr)


def test_ern_labelled_split_is_eegpts_fold_shape_and_subject_disjoint():
    sys.path.insert(0, os.path.join(ROOT, "EEG"))
    import kaggle_ern_finetune as k
    for fold in range(4):
        tr, va, te = k.labelled_split(fold)
        assert (len(tr), len(va), len(te)) == (10, 2, 4)
        assert set(tr) | set(va) | set(te) == set(k.TRAIN_SUBJECTS)
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        # the test set is exactly what EEGPT's fold leaves out of training
        assert set(te) == set(k.TRAIN_SUBJECTS) - set(k.EEGPT_FOLDS[fold])


def test_ern_without_true_labels_builds_from_the_labelled_subjects(tmp_path):
    """The Kaggle download has no true_labels.csv, so this is the usual case."""
    import kaggle_ern_finetune as k
    _write_ern(str(tmp_path / "raw"), np.random.default_rng(8),
               train=tuple(k.TRAIN_SUBJECTS), test=())
    os.remove(tmp_path / "raw" / "true_labels.csv")
    r = subprocess.run(
        [sys.executable, "EEG/kaggle_ern_finetune.py", "--raw-dir",
         str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out"), "--fold", "0"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "LABELLED subjects" in r.stderr
    split = json.loads((tmp_path / "out" / "split.json").read_text())
    assert split["provenance"]["test_split"] == "labelled"
    with h5py.File(tmp_path / "out" / "test.h5") as f:
        assert set(np.unique(f["subject"][:])) == {2, 6, 7, 11}
    with h5py.File(tmp_path / "out" / "train.h5") as f:
        assert len(np.unique(f["subject"][:])) == 10


# --------------------------------------------------------------------------- #
# The collector
# --------------------------------------------------------------------------- #

def test_the_collector_prints_eegpts_row_and_the_pretraining_delta(tmp_path):
    for mode, auroc in (("ft", 0.70), ("scratch", 0.65)):
        d = tmp_path / f"bcic2b_f0_{mode}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps({
            "best_epoch": 3, "select_by": "auroc", "trainable_params": 10,
            "test": {"balanced_acc": 0.6, "kappa": 0.2, "weighted_f1": 0.6,
                     "auroc": auroc}}))
    r = subprocess.run([sys.executable, "scripts/collect_eegpt_bench.py",
                        "--root", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "0.8059" in r.stdout                     # EEGPT Table 4, BCIC-2B AUROC
    assert "pretrained - scratch auroc = +0.0500" in r.stdout
