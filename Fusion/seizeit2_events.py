#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Event-level SeizeIT2 evaluation of a trained fusion run, on the test patients'
complete continuous recordings.

    python Fusion/seizeit2_events.py --run-dir $PW_CKPT_ROOT/fusion/ralf_frozen_sz \\
        --raw-dir $R/seizeit2 [--subjects 'sub-(09[7-9]|1[0-2][0-9])'] [--step 1]

WHAT IT DOES. For every run of every test patient (default: the dataset
paper's validation patients, sub-097..sub-125, our test set) the EEG, ECG and
EMG EDFs are read whole, filtered with each modality's pretraining pipeline,
resampled to the encoder rates, and slid over in --window (4 s) windows every
--step (1 s) seconds; the window of second k is centred on it. Each window is
normalised exactly as the training windows were and goes through the run's
selected checkpoint (best.pth) once; the fusion core is then evaluated under
every modality subset from the same encodings. A run without ECG/EMG (3% of
the data) keeps them absent, as in training.

Per second the script stores the seizure probability for every condition,
the label (1 when the second's centre lies inside an annotated sz* event), and
the reference RMS gate (13 < RMS < 150 uV on the 2 s cross-head EEG window,
see physiowave/fusion/seizure_events.py). predictions/<sub>_<run>.npz holds
them, so the scoring can be redone without the GPU (--score-only).

SCORING (physiowave/fusion/seizure_events.py): the reference post-processing
(merge events < 2 s apart, drop events < 8 s) and any-overlap; the dataset
paper's per-recording means, the pooled event sensitivity and FA/24h, the
sensitivity -- false-alarm curve over thresholds 0..1, the sensitivity at
false-alarm budgets, and the pooled epoch AUROC / AUPRC. Written to
events.json in the run directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from seizeit2_prep import (EMG_CHANNELS, ECG_CHANNELS, MAINS_HZ,  # noqa: E402
                           OFFICIAL_TEST, find_runs, read_events)
from physiowave.fusion.prep_common import (MODALITIES, RATES, edf_units,  # noqa: E402
                                           filter_continuous, normalise_batch,
                                           read_edf, to_encoder_rate, to_microvolts,
                                           window_samples)
from physiowave.fusion.seizure_events import THRESHOLDS, evaluate, rms_gate  # noqa: E402

SUBSETS: List[Tuple[str, Tuple[str, ...]]] = [
    ("complete", ("eeg", "ecg", "emg")), ("eeg_only", ("eeg",)), ("ecg_only", ("ecg",)),
    ("emg_only", ("emg",)), ("eeg_ecg", ("eeg", "ecg")), ("eeg_emg", ("eeg", "emg")),
    ("ecg_emg", ("ecg", "emg")),
]


def load_streams(run: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, int]:
    """``({modality: stream at the encoder rate}, {eeg label: uV at native rate},
    native eeg rate, seconds)`` for one run; absent modalities are left out."""
    sig, rates, T = read_edf(run["eeg"])
    units = edf_units(run["eeg"])
    names = sorted(sig, key=lambda n: (0 if n.upper().startswith("BTE") else 1, n))[:2]
    fs = float(rates[names[0]])
    eeg_uv = {n: to_microvolts(np.asarray(sig[n], np.float64), units.get(n, "uV")) for n in names}
    eeg = np.stack([sig[n] for n in names]).astype(np.float64)
    if eeg.shape[0] == 1:
        eeg = np.concatenate([eeg, np.zeros_like(eeg)])
    streams = {"eeg": to_encoder_rate(filter_continuous(eeg, fs, "eeg", MAINS_HZ), fs, "eeg")
               .astype(np.float32)}
    for m in ("ecg", "emg"):
        if run.get(m) is None:
            continue
        try:
            s2, r2, _ = read_edf(run[m])
        except (OSError, ValueError):
            continue
        lab = next(iter(s2), None)
        if lab is None:
            continue
        x = np.asarray(s2[lab], np.float64)[None]
        streams[m] = to_encoder_rate(filter_continuous(x, float(r2[lab]), m, MAINS_HZ),
                                     float(r2[lab]), m).astype(np.float32)
    return streams, eeg_uv, fs, int(np.floor(T))


def second_labels(events, n_seconds: int) -> np.ndarray:
    y = np.zeros(n_seconds, bool)
    centres = np.arange(n_seconds) + 0.5
    for onset, dur, kind in events:
        if kind.startswith("sz"):
            y |= (centres >= onset) & (centres < onset + dur)
    return y


def predict_run(model, task: str, run: dict, window: float, step: float, batch: int,
                device, amp) -> Dict[str, np.ndarray]:
    import torch
    streams, eeg_uv, fs_eeg, T = load_streams(run)
    # what the model has an encoder for (an EEG-only model scores eeg subsets only)
    streams = {m: v for m, v in streams.items() if m in model.modalities}
    n_sec = int(T)
    centres = np.arange(0, n_sec, step) + 0.5
    starts = centres - window / 2.0
    ok = (starts >= 0) & (starts + window <= T)
    probs = {c: np.zeros(len(centres), np.float32) for c, _ in SUBSETS}
    present = {m: (m in streams) for m in MODALITIES}
    W = {m: window_samples(m, window) for m in MODALITIES}
    idx = np.where(ok)[0]
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(idx), batch):
            sel = idx[b0:b0 + batch]
            x = {}
            for m, st in streams.items():
                a = np.round(starts[sel] * RATES[m]).astype(np.int64)
                a = np.clip(a, 0, st.shape[1] - W[m])
                win = np.stack([st[:, s:s + W[m]] for s in a])
                x[m] = torch.from_numpy(normalise_batch(win, m)).to(device)
            B = len(sel)
            with amp:
                feats = model.encode(x, task)
                for cond, keep in SUBSETS:
                    if not any(present[m] for m in keep):
                        continue
                    mask = {m: torch.full((B,), bool(present[m] and m in keep),
                                          dtype=torch.bool, device=device)
                            for m in model.modalities}
                    out = model.core({m: f for m, f in feats.items()}, task, mask)
                    probs[cond][sel] = torch.softmax(out["logits"].float(), -1)[:, 1].cpu().numpy()
    for cond, keep in SUBSETS:
        if not any(present[m] for m in keep):
            probs[cond][:] = np.nan                      # condition unavailable on this run
    # per second (step 1 is the reference grid; a coarser step is repeated)
    rep = int(round(step))
    sec_prob = {c: np.repeat(v, rep)[:n_sec] if rep > 1 else v[:n_sec] for c, v in probs.items()}
    label = second_labels(read_events(run["events"]), n_sec)
    gates = {mode: rms_gate(eeg_uv, fs_eeg, n_sec, mode) for mode in ("cross", "both")}
    return {"label": label, "gate_cross": gates["cross"], "gate_both": gates["both"],
            "present": np.array([present[m] for m in MODALITIES]),
            **{f"prob_{c}": v.astype(np.float32) for c, v in sec_prob.items()}}


def score(pred_dir: str, gate: str) -> Dict[str, object]:
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    out = {"gate": gate, "n_runs": len(files), "conditions": {}}
    for cond, _ in SUBSETS:
        recs = []
        for f in files:
            z = np.load(f)
            p = z[f"prob_{cond}"].astype(np.float64)
            if np.isnan(p).all():
                continue
            g = np.ones_like(p, bool) if gate == "none" else z[f"gate_{gate}"][:len(p)]
            recs.append((np.where(g, np.nan_to_num(p), 0.0), z["label"][:len(p)]))
        if recs:
            out["conditions"][cond] = evaluate(recs, THRESHOLDS)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="a physiowave.fusion.train output dir")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--subjects", default=None,
                    help="regex over sub-XXX; default the official test patients 097-125")
    ap.add_argument("--task", default="seizeit2")
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--gate", choices=["cross", "both", "none"], default="cross",
                    help="RMS gate: cross = the reference code's behaviour")
    ap.add_argument("--device", default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args(argv)
    pred_dir = os.path.join(args.run_dir, "predictions")
    t0 = time.time()

    if not args.score_only:
        import torch
        from physiowave.config import load_config
        from physiowave.fusion.train import build_model, load_tasks
        cfg = load_config(os.path.join(args.run_dir, "config_resolved.yaml"))
        modalities = list(cfg["data"].get("modalities") or MODALITIES)
        tasks = load_tasks(cfg, modalities)
        if args.task not in tasks:
            raise SystemExit(f"the run has no task {args.task!r}: {sorted(tasks)}")
        model, _ = build_model(cfg, tasks, args.run_dir)
        ck = torch.load(os.path.join(args.run_dir, "best.pth"), map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["model"])
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        runs = find_runs(args.raw_dir, args.subjects)
        if args.subjects is None:
            runs = [r for r in runs if r["subject"] in OFFICIAL_TEST]
        if args.max_runs:
            runs = runs[: args.max_runs]
        if not runs:
            raise SystemExit("no test runs found (are sub-097..sub-125 downloaded?)")
        os.makedirs(pred_dir, exist_ok=True)
        print(f"{len(runs)} run(s) of {len({r['subject'] for r in runs})} patient(s); "
              f"checkpoint epoch {ck.get('epoch')}", flush=True)
        for i, run in enumerate(runs):
            dest = os.path.join(pred_dir, f"{run['subject']}_run-{run['run']}.npz")
            if os.path.isfile(dest):
                continue
            t1 = time.time()
            res = predict_run(model, args.task, run, args.window, args.step, args.batch,
                              device, amp)
            np.savez_compressed(dest, **res)
            print(f"  [{i + 1}/{len(runs)}] {run['subject']} run-{run['run']}: "
                  f"{len(res['label']) / 3600:.1f} h, {int(res['label'].sum())} seizure s, "
                  f"present {res['present'].tolist()}  ({time.time() - t1:.0f}s)", flush=True)

    result = score(pred_dir, args.gate)
    result["window_seconds"], result["step_seconds"] = args.window, args.step
    with open(os.path.join(args.run_dir, f"events_{args.gate}.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    for cond, r in result["conditions"].items():
        a = r["at_0.5"]
        print(f"  {cond:9s} th0.5: sens_ovlp {a['official']['sens_ovlp']:.3f}  FA/h "
              f"{a['official']['fa_ovlp_h']:.2f} (epoch {a['official']['fa_epoch_h']:.1f})  "
              f"| pooled sens {a['events']['sens']:.3f} FA/24h {a['events']['fa_24h']:.1f}  "
              f"| AUROC {r['auroc_epoch']:.3f} AUPRC {r['auprc_epoch']:.4f}  "
              f"| sens@1FA/h {r['sens_at_fa_events']['1.0']:.3f}", flush=True)
    print(f"wrote {args.run_dir}/events_{args.gate}.json in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
