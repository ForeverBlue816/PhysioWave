#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCI Competition IV 2a / 2b (motor imagery) -> labelled HDF5.
------------------------------------------------------------
    python EEG/bcic_iv2_finetune.py --dataset 2a \\
        --raw-dir $PW_DATA_EEG/bcic_iv2a --out-dir <split dir> --fold 0

SOURCE. BNCI Horizon 2020's copies, data sets 001-2014 (= IV-2a) and 004-2014
(= IV-2b), fetched by EEG/download_eegpt_benchmarks.py: the competition
recordings, with the evaluation sessions' labels inside the files.

PREPROCESSING, two choices (EEG/eegpt_bench_common.py):

    --prep pretrain  (default) the pretraining pipeline, on each continuous
                     run: microvolts, linear detrend, 50 Hz notch, 0.5 Hz
                     high-pass, 256 Hz, then a per-window z-score
    --prep eegpt     EEGPT's: 0-38 Hz low-pass and 256 Hz on each run, then
                     per session file Euclidean alignment, common average and
                     the ±20 clip

THE TASK, as EEGPT defines it:

    epoch       0 to 4 s after the cue: 1024 samples at 256 Hz -- the same
                4 s as a pretraining window. The BNCI `trial` field marks the
                start of a trial; the cue follows 2 s later on 2a and 3 s later
                on 2b. Checked on the recordings: on 2b the 8-30 Hz power over
                C3/C4 is flat until +3.0 s and falls 37% by +3.5 s; on 2a it
                recovers at +6.5 s, where imagery begun at a +2 s cue ends.
    channels    2a: the 22 EEG electrodes (EOG dropped). 2b: C3, Cz, C4 --
                bipolar derivations centred on those sites, named for where
                they sit, which is what the channel-name embedding keys on.
    classes     2a: left hand, right hand, feet, tongue.  2b: left, right.
    trials      all of them, artefact-flagged ones included, as EEGPT keeps them
    folds       leave-one-subject-out over nine subjects; the held-out
                subject's two sessions are its TEST set, one more subject
                validates, seven train (EEGPT: 8/1, validating on the test)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eegpt_bench_common as bc                                  # noqa: E402

DATASETS = {
    "2a": {
        "prefix": "A0{s}{part}.mat",
        "parts": ("T", "E"),
        "n_eeg": 22,
        "channels": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                     "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                     "CP3", "CP1", "CPz", "CP2", "CP4",
                     "P1", "Pz", "P2", "POz"],
        "classes": ["left_hand", "right_hand", "feet", "tongue"],
        "cue_s": 2.0,
    },
    "2b": {
        "prefix": "B0{s}{part}.mat",
        "parts": ("T", "E"),
        "n_eeg": 3,
        "channels": ["C3", "Cz", "C4"],
        "classes": ["left_hand", "right_hand"],
        "cue_s": 3.0,
    },
}
SUBJECTS = list(range(1, 10))
FS_IN = 250.0
EPOCH_S = 4.0


def load_file(path: str, spec: dict, prep: str = "pretrain"):
    """One BNCI session file -> ``(epochs [N, C, 1024], labels [N], run [N])``.

    Each run goes through the pretraining pipeline continuous, then is epoched
    and z-scored per window. Runs with no trials -- 2a's three EOG calibration
    runs -- are skipped.
    """
    import scipy.io as sio

    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = m["data"]
    runs = list(runs) if isinstance(runs, np.ndarray) else [runs]
    length = int(round(EPOCH_S * bc.FS_OUT))
    X_all, y_all, r_all = [], [], []
    for ri, run in enumerate(runs):
        trial = np.atleast_1d(run.trial).astype(np.int64)
        y = np.atleast_1d(run.y).astype(np.int64)
        if trial.size == 0 or y.size == 0:
            continue
        if float(run.fs) != FS_IN:
            raise SystemExit(f"{path} run {ri}: fs {run.fs}, expected {FS_IN}")
        if len(trial) != len(y):
            raise SystemExit(f"{path} run {ri}: {len(trial)} trials, {len(y)} labels")
        x = np.asarray(run.X, dtype=np.float64)[:, :spec["n_eeg"]].T   # [C, T], µV
        if prep == "pretrain":
            x = bc.pretrain_pipeline(x, FS_IN, unit="uV")
        else:
            x = bc.eegpt_continuous(x, FS_IN)
        # `trial` is 1-based MATLAB indexing into X.
        onsets = np.round((trial - 1 + spec["cue_s"] * FS_IN)
                          * bc.FS_OUT / FS_IN).astype(np.int64)
        ep = bc.epoch_at(x, onsets, length)
        # pretrain normalises each window; eegpt normalises the session file,
        # below, once all its runs are in.
        X_all.append(bc.normalise_windows(ep) if prep == "pretrain" else ep)
        y_all.append(y - 1)
        r_all.append(np.full(len(y), ri, dtype=np.int64))
    if not X_all:
        raise SystemExit(f"{path}: no runs with trials")
    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(r_all)


def prepare_subject(raw_dir: str, dataset: str, subject: int,
                    prep: str = "pretrain"):
    """Both session files of one subject."""
    spec = DATASETS[dataset]
    out = []
    for si, part in enumerate(spec["parts"]):
        path = os.path.join(raw_dir, spec["prefix"].format(s=subject, part=part))
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}\n  fetch it with: python "
                             f"EEG/download_eegpt_benchmarks.py --dataset {dataset} "
                             f"--dest {raw_dir}")
        X, y, _ = load_file(path, spec, prep)
        if prep == "eegpt":
            # Per session FILE, as EEGPT's get_data applies EA(session_1) and
            # EA(session_2): alignment, then the common average, then clip.
            X = bc.eegpt_session(X)
        n_cls = len(spec["classes"])
        if y.min() < 0 or y.max() >= n_cls:
            raise SystemExit(f"{path}: labels {np.unique(y)} outside 0..{n_cls - 1}")
        out.append((X.astype(np.float32), y, si))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--raw-dir", required=True,
                   help="directory holding A0?T.mat / A0?E.mat (or B0?...)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0,
                   help="index into --subjects of the held-out TEST subject")
    p.add_argument("--subjects", default=",".join(map(str, SUBJECTS)),
                   help="comma-separated subject numbers (default: all nine)")
    p.add_argument("--prep", default="pretrain", choices=sorted(bc.PREPS),
                   help="pretrain: the pretraining pipeline (default). eegpt: "
                        "EEGPT's 0-38 Hz low-pass + Euclidean alignment + "
                        "common average, per session")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    spec = DATASETS[args.dataset]
    bc.check_vocabulary(spec["channels"])
    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    split = bc.loso_split(subjects, args.fold, args.seed)

    print(f"BCIC-IV-{args.dataset}: fold {args.fold}  "
          f"train {split['train']}  val {split['val']}  test {split['test']}")
    if args.prep == "pretrain":
        how = (f"pretraining pipeline ({bc.PREPS['pretrain']}): detrend, "
               f"{bc.MAINS_HZ:g} Hz notch, {bc.CFG.highpass_hz:g} Hz high-pass, "
               f"{bc.FS_OUT} Hz, window z-score")
    else:
        how = (f"EEGPT pipeline ({bc.PREPS['eegpt']}): {bc.EEGPT_LOWPASS_HZ:g} Hz "
               f"low-pass, {bc.FS_OUT} Hz, EA per session, common average")
    print(f"  {len(spec['channels'])} channels, cue+0..{EPOCH_S:g} s, {how}")

    prov = bc.provenance({
        "dataset": f"BCIC-IV-{args.dataset}",
        "source": "BNCI Horizon 2020 " + ("001-2014" if args.dataset == "2a" else "004-2014"),
        "source_sampling_rate": FS_IN,
        "epoch": f"cue + [0, {EPOCH_S}) s, cue at trial + {spec['cue_s']} s",
        "fold": args.fold, "split": split,
        "protocol": "leave-one-subject-out, test selects nothing",
    }, prep=args.prep)

    rows = []
    for name, subs in split.items():
        X, Y, S, K = [], [], [], []
        for s in subs:
            for Xs, ys, si in prepare_subject(args.raw_dir, args.dataset, s,
                                              args.prep):
                X.append(Xs); Y.append(ys)
                S.append(np.full(len(ys), s)); K.append(np.full(len(ys), si))
        rows.append(bc.write_split(os.path.join(args.out_dir, f"{name}.h5"),
                                   X, Y, S, K, spec["channels"], spec["classes"],
                                   prov))
    bc.print_summary(rows)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"provenance": prov, "files": rows}, f, indent=2)
    print(f"  wrote {args.out_dir}/{{train,val,test}}.h5  "
          f"(--num-classes {len(spec['classes'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
