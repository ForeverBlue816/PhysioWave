#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper figures for the SeizeIT2 multimodal runs, from their own artefacts.

    python scripts/visualize_seizeit2.py --raw-dir $FUSION_ROOT/raw/seizeit2 \\
        --run-dir $PW_CKPT_ROOT/fusion/ralf_frozen_szo \\
        --compare $PW_CKPT_ROOT/fusion/{concat_frozen,weighted_logit_frozen,ralf_finetune,ralf_scratch}_szo \\
        --eeg-runs $PW_CKPT_ROOT/fusion/eeg_{c1,eegpt,labram}_finetune_sze \\
        --out-dir docs/runs/seizeit2_fusion --format svg,pdf --paper

Nothing is illustrative. The traces are the test patients' recordings, the
probabilities are the ones the event-level evaluation cached
(<run>/predictions/*.npz, Fusion/seizeit2_events.py), the curves are
events_cross.json, the reliabilities and AUPRCs are results.json, the
learning curves are history.json.

  fig_seizure_example      one test recording around one seizure: the two
                           behind-the-ear EEG channels, ECG and EMG as recorded;
                           the model's per-second probability under every
                           modality subset; the RMS gate; the detection after
                           post-processing. The recording is CHOSEN: the
                           candidates are ranked and the ranking is kept in
                           figure_metadata, so the caption can say so.
  fig_main                 the four-panel summary: (a) sensitivity against
                           FA/h, the fusion runs against the EEG-alone
                           encoders and the published detectors; (b) what each
                           modality subset of the fusion runs reaches at <= 11
                           FA/h; (c) the main run's reliability head under
                           each corruption; (d) per test patient, EEG alone
                           against the main run at each one's <= 11 FA/h
                           threshold.
  fig_sensitivity_fa       sensitivity against false alarms per hour (the
                           dataset paper's scoring) over the threshold curve:
                           every run, every modality subset of the main run,
                           the published SVM and ChronoNet as points.
  fig_modality_subsets     what each modality subset reaches at <= 11 FA/h
                           (the SVM's operating point) and its window AUPRC.
  fig_reliability          RALF's per-modality reliability on the clean test
                           set and under each corruption; the AUPRC every run
                           keeps under the same corruptions.
  fig_per_patient          per test patient: seizures, sensitivity at
                           threshold 0.5, FA/h.
  fig_learning_curves      validation AUPRC per epoch, the selected epoch marked.
  fig_signal_<m>, fig_window_<m>   (fig_signal) per modality, the example's
                           recorded trace over the same span with scale bars,
                           and one 4 s encoder input window: bare signal in the
                           modality's colour, for the framework figure and for
                           single-modality panels.

Style and plumbing (Times New Roman bold, Okabe-Ito, svg+pdf with editable
text, figure_data / figure_metadata beside every figure) are those of
scripts/visualize_eeg_pretraining.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "Fusion"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402

from visualize_eeg_pretraining import (OKABE_ITO, FigureWriter,  # noqa: E402
                                       apply_style, git_commit)
from physiowave.fusion.prep_common import (MODALITIES, edf_units, filter_continuous,  # noqa: E402
                                           read_edf, to_microvolts)
from physiowave.fusion.seizure_events import post_process, score_recording  # noqa: E402
from seizeit2_events import SUBSETS                            # noqa: E402
from seizeit2_prep import MAINS_HZ, find_runs, read_events     # noqa: E402

MOD_COLOR = {"eeg": OKABE_ITO[0], "ecg": OKABE_ITO[1], "emg": OKABE_ITO[2]}
SUBSET_COLOR = {"complete": "#000000", "eeg_only": MOD_COLOR["eeg"], "ecg_only": MOD_COLOR["ecg"],
                "emg_only": MOD_COLOR["emg"], "eeg_ecg": OKABE_ITO[3], "eeg_emg": OKABE_ITO[4],
                "ecg_emg": OKABE_ITO[5]}
SUBSET_LABEL = {"complete": "EEG+ECG+EMG", "eeg_only": "EEG", "ecg_only": "ECG",
                "emg_only": "EMG", "eeg_ecg": "EEG+ECG", "eeg_emg": "EEG+EMG",
                "ecg_emg": "ECG+EMG"}
MOD_LABEL = {"eeg": "bte-EEG", "ecg": "ECG", "emg": "EMG"}
#: (condition id in results.json, modality, label)
CORRUPTIONS = [("eeg_noise", "eeg", "noise"), ("eeg_bad_channel", "eeg", "bad channel"),
               ("eeg_drift", "eeg", "drift"), ("ecg_saturation", "ecg", "saturation"),
               ("ecg_noise", "ecg", "noise"), ("emg_dropout", "emg", "dropout"),
               ("emg_noise", "emg", "noise")]
PUBLISHED = [("SVM (Bhagubai et al.)", 71.1, 11.0, "s"), ("ChronoNet (Bhagubai et al.)", 84.2, 100.5, "D")]
FA_BUDGET = 11.0
#: Okabe-Ito without the yellow (#F0E442), which vanishes on white as a line.
LINE_COLORS = [c for c in OKABE_ITO if c != "#F0E442"]
WINDOW_S = 4.0


def unit_label(u: str) -> str:
    """EDF physical dimensions as printed: uV -> µV, the rest as written."""
    return {"uV": "µV", "uv": "µV", "UV": "µV", "mv": "mV", "MV": "mV"}.get(u.strip(), u.strip())


def nice(v: float) -> float:
    """A round scale-bar length not above v: 1, 2, 5 x 10^k."""
    if v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (5, 2, 1):
        if m * e <= v:
            return float(m * e)
    return float(e)


# --------------------------------------------------------------------------- #
# Run artefacts
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, path: str):
        self.path = path.rstrip("/")
        self.name = os.path.basename(self.path)
        self.results = self._json("results.json")
        self.history = self._json("history.json") or []
        self.events = self._json("events_cross.json")
        self.pred_dir = os.path.join(self.path, "predictions")
        self.fusion = (self.results or {}).get("fusion", "?")
        self.mode = (self.results or {}).get("encoder_mode", "?")

    def _json(self, name):
        f = os.path.join(self.path, name)
        return json.load(open(f)) if os.path.isfile(f) else None

    @property
    def label(self) -> str:
        n = self.name
        for suf in ("_szo", "_sze"):
            n = n.replace(suf, "")
        if n.startswith("eeg_"):
            enc = {"c1": "PhysioWave-T", "c1native": "PhysioWave-T (native)",
                   "c1siena": "PhysioWave-T + Siena", "eegpt": "EEGPT",
                   "labram": "LaBraM", "luna_base": "LUNA-B", "luna_large": "LUNA-L",
                   "luna_huge": "LUNA-H"}
            import re as _re                                  # eeg_<encoder>_<mode>[_lr<lr>]
            m = _re.match(r"eeg_(.+?)_(frozen|finetune|scratch)(?:_lr(.+))?$", n)
            if m:
                key, mode, lr = m.groups()
                return f"{enc.get(key, key)} {mode}{f' lr {lr}' if lr else ''}, EEG"
            return n
        return f"{self.fusion.replace('weighted_logit', 'weighted logit')} / {self.mode}"

    def test(self) -> Dict[str, Dict]:
        return ((self.results or {}).get("tasks", {}).get("seizeit2", {}) or {}).get("test", {})

    def curve(self, cond: str = "complete") -> Optional[List[Dict]]:
        c = (self.events or {}).get("conditions", {}).get(cond)
        return c["curve"] if c else None

    def sens_at(self, cond: str, budget: float) -> Optional[float]:
        cur = self.curve(cond)
        if not cur:
            return None
        ok = [r["official"]["sens_ovlp"] for r in cur
              if r["official"]["fa_ovlp_h"] == r["official"]["fa_ovlp_h"]
              and r["official"]["fa_ovlp_h"] <= budget]
        return max(ok) if ok else 0.0

    def th_at(self, cond: str, budget: float) -> Optional[float]:
        """The threshold of the best sensitivity reachable within ``budget`` FA/h."""
        cur = self.curve(cond)
        if not cur:
            return None
        ok = [(r["official"]["sens_ovlp"], r["th"]) for r in cur
              if r["official"]["fa_ovlp_h"] == r["official"]["fa_ovlp_h"]
              and r["official"]["fa_ovlp_h"] <= budget]
        return max(ok)[1] if ok else None

    @property
    def short_label(self) -> str:
        """``RALF``, ``weighted logit``, ``concat`` for a fusion run; the encoder
        and mode for an EEG-alone run."""
        if self.name.startswith("eeg_"):          # the encoder alone; the mode is metadata
            import re as _re
            return _re.sub(r" (frozen|finetune|scratch)(?: lr .+)?, EEG$", "", self.label)
        name = {"ralf": "RALF", "weighted_logit": "weighted logit",
                "concat": "concat"}.get(self.fusion, self.fusion)
        return name if self.mode == "frozen" else f"{name} ({self.mode})"

    def predictions(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.pred_dir, "*.npz")))

    def per_patient(self, th: float, cond: str = "complete") -> Dict[str, Dict[str, float]]:
        """Pooled event counts per test patient at one threshold, from the
        cached per-second probabilities."""
        per: Dict[str, Dict[str, float]] = {}
        for f in self.predictions():
            z = np.load(f)
            sub = os.path.basename(f).split("_run-")[0]
            r = score_recording(gated(z, cond), z["label"], th)
            d = per.setdefault(sub, {"n_seizures": 0, "hit": 0, "false_events": 0, "hours": 0.0})
            for key in d:
                d[key] += r[key]
        return per


def gated(z, cond: str) -> np.ndarray:
    p = z[f"prob_{cond}"].astype(np.float64)
    g = z["gate_cross"][: len(p)]
    return np.where(g, np.nan_to_num(p), 0.0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def choose_example(run: Run, preferred: Optional[str]) -> Tuple[str, List[Dict]]:
    """Rank the recordings with a seizure and all three modalities by how far the
    complete model's probability inside the seizure sits above its surroundings;
    the best detected one is drawn. The ranking is returned for the metadata."""
    rows = []
    for f in run.predictions():
        z = np.load(f)
        y = z["label"].astype(bool)
        if not y.any() or not bool(z["present"].all()):
            continue
        p = gated(z, "complete")
        det = post_process(p, 0.5)
        from physiowave.fusion.seizure_events import mask_to_events
        for a, b in mask_to_events(y):
            a, b = int(a), int(b)
            lo, hi = max(0, a - 120), min(len(p), b + 120)
            around = np.concatenate([p[lo:a], p[b:hi]])
            contrast = float(p[a:b].mean() - (around.mean() if around.size else 0.0))
            rows.append({"recording": os.path.basename(f)[:-4], "onset_s": a, "offset_s": b,
                         "duration_s": b - a, "contrast": contrast,
                         "detected": bool(det[a:b].any()), "n_seizures": len(mask_to_events(y))})
    rows.sort(key=lambda r: (not r["detected"], -r["contrast"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    if preferred:
        pick = next((r for r in rows if r["recording"] == preferred), None)
        if pick is None:
            raise SystemExit(f"{preferred}: not among the recordings with a seizure and all "
                             f"three modalities ({len(rows)} candidates)")
    else:
        if not rows:
            raise SystemExit("no recording with a seizure and all three modalities")
        pick = rows[0]
    pick["chosen"] = True
    return pick["recording"], rows


def load_example(run: Run, raw_dir: str, preferred: Optional[str], margin: float) -> Dict:
    """The chosen recording's traces as recorded (filtered as the pipeline
    filters, at the native rate) from ``margin`` s before the seizure to
    ``margin`` s after, and the cached per-second predictions."""
    rec, ranking = choose_example(run, preferred)
    subject, runid = rec.split("_run-")
    cand = [r for r in find_runs(raw_dir, subject + "$") if r["run"] == runid]
    if not cand:
        raise SystemExit(f"{rec}: not under {raw_dir}")
    r = cand[0]
    z = np.load(os.path.join(run.pred_dir, rec + ".npz"))
    pick = next(x for x in ranking if x.get("chosen"))
    t0, t1 = pick["onset_s"] - margin, pick["offset_s"] + margin
    t0 = max(0.0, t0)
    sig, rates, T = read_edf(r["eeg"])
    units = edf_units(r["eeg"])
    t1 = min(float(T), t1)
    names = sorted(sig, key=lambda n: (0 if n.upper().startswith("BTE") else 1, n))[:2]
    traces = []                   # dicts: modality, label, fs, t (s), y, unit, colour
    fs = float(rates[names[0]])
    eeg = np.stack([to_microvolts(np.asarray(sig[n], np.float64), units.get(n, "uV"))
                    for n in names])
    eeg = filter_continuous(eeg, fs, "eeg", MAINS_HZ)
    a, b = int(t0 * fs), int(t1 * fs)
    for n, row in zip(names, eeg):
        traces.append({"modality": "eeg", "label": n.replace(" SD", ""), "fs": fs,
                       "t": np.arange(a, b) / fs, "y": row[a:b], "unit": "µV",
                       "color": MOD_COLOR["eeg"]})
    for m in ("ecg", "emg"):
        if not r.get(m) or not os.path.isfile(r[m]):
            continue
        s2, r2, _ = read_edf(r[m])
        u2 = edf_units(r[m])
        lab = next(iter(s2))
        f2 = float(r2[lab])
        x = filter_continuous(np.asarray(s2[lab], np.float64)[None], f2, m, MAINS_HZ)[0]
        a2, b2 = int(t0 * f2), int(t1 * f2)
        traces.append({"modality": m, "label": MOD_LABEL[m], "fs": f2, "t": np.arange(a2, b2) / f2,
                       "y": x[a2:b2], "unit": unit_label(u2.get(lab, "")), "color": MOD_COLOR[m]})
    return {"rec": rec, "subject": subject, "run": runid, "ranking": ranking, "pick": pick,
            "z": z, "t0": t0, "t1": t1, "traces": traces, "eeg_file": r["eeg"]}


def fig_seizure_example(w: FigureWriter, run: Run, raw_dir: str, preferred: Optional[str],
                        margin: float, ex: Optional[Dict] = None):
    ex = ex or load_example(run, raw_dir, preferred, margin)
    rec, ranking, pick, z = ex["rec"], ex["ranking"], ex["pick"], ex["z"]
    subject, runid, t0, t1 = ex["subject"], ex["run"], ex["t0"], ex["t1"]
    traces = [(d["label"], d["t"], d["y"], d["unit"], d["color"]) for d in ex["traces"]]
    r = {"eeg": ex["eeg_file"]}
    conds = [c for c, _ in SUBSETS if c in ("complete", "eeg_only", "ecg_only", "emg_only")]
    n_rows = len(traces) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 1.0 * len(traces) + 1.9), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(traces) + [1.9]})
    onset, offset = pick["onset_s"], pick["offset_s"]
    fig.subplots_adjust(right=0.84, hspace=0.25)
    for ax, (lab, t, y, unit, col) in zip(axes, traces):
        ax.plot(t - onset, y, color=col, lw=0.5)
        ax.axvspan(0, offset - onset, color="0.85", alpha=0.6, lw=0, zorder=0)
        amp = float(np.percentile(np.abs(y), 99.5)) if y.size else 1.0
        ax.set_ylabel(f"{lab}\n±{amp:.0f} {unit}".rstrip(), rotation=0, ha="right", va="center",
                      labelpad=6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_ylim(-amp * 1.1, amp * 1.1)
    ax = axes[-1]
    sec = np.arange(int(t0), int(t1))
    for c in conds:
        p = gated(z, c)
        if np.isnan(z[f"prob_{c}"]).all():
            continue
        ax.step(sec - onset + 0.5, p[int(t0):int(t1)], where="mid", color=SUBSET_COLOR[c],
                lw=1.4 if c == "complete" else 0.9, label=SUBSET_LABEL[c],
                alpha=1.0 if c == "complete" else 0.9)
    gate = z["gate_cross"][int(t0):int(t1)]
    for s, g in zip(sec, gate):
        if not g:
            ax.axvspan(s - onset, s - onset + 1, color="0.6", alpha=0.35, lw=0, hatch="///")
    det = post_process(gated(z, "complete"), 0.5)[int(t0):int(t1)]
    ax.fill_between(sec - onset + 0.5, 1.02, 1.10, where=det, step="mid", color="black", lw=0)
    ax.text(1.005, 1.06 / 1.12, "detected", fontsize=6, ha="left", va="center",
            transform=ax.transAxes)
    ax.axhline(0.5, color="0.5", lw=0.6, ls="--")
    ax.axvspan(0, offset - onset, color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("P(seizure)")
    ax.set_xlabel("time from annotated onset (s)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(1.0, 0.92), handlelength=1.5,
              borderaxespad=0.2)
    ax.set_xlim(t0 - onset, t1 - onset)
    fig.suptitle(f"{run.label}: {subject} run-{runid}, seizure {pick['rank']} of {len(ranking)} "
                 f"ranked candidates ({pick['duration_s']} s)", y=0.995)
    fig.text(0.5, 0.03, "grey band: annotated seizure; hatched: seconds the 13-150 µV RMS gate "
                        "forces to 0; black bar: detection after post-processing at threshold 0.5",
             ha="center", fontsize=6.5, color="0.35")
    w.save(fig, "fig_seizure_example",
           data={"t0": t0, "t1": t1, "onset": onset, "offset": offset,
                 **{f"prob_{c}": gated(z, c)[int(t0):int(t1)] for c in conds},
                 "gate": gate, "label": z["label"][int(t0):int(t1)],
                 **{f"trace_{i}": y for i, (_, _, y, _, _) in enumerate(traces)}},
           meta={"recording": rec, "eeg_file": r["eeg"], "chosen": pick, "ranking": ranking[:20],
                 "n_candidates": len(ranking), "margin_s": margin})


def _bare(ax):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _scale_bars(ax, x0, y0, dx, dy, xlab, ylab, color="black"):
    ax.plot([x0, x0 + dx], [y0, y0], color=color, lw=1.0, solid_capstyle="butt")
    ax.plot([x0, x0], [y0, y0 + dy], color=color, lw=1.0, solid_capstyle="butt")
    ax.text(x0 + dx / 2, y0, xlab, ha="center", va="top", fontsize=6)
    ax.text(x0, y0 + dy / 2, ylab + " ", ha="right", va="center", fontsize=6)


def fig_signal_strips(w: FigureWriter, run: Run, ex: Dict, window_start: Optional[float]):
    """Per modality: the recorded trace(s) over the example span with scale bars
    and the seizure shaded (fig_signal_<m>), and one encoder input window, the
    exact samples a 4 s window holds (fig_window_<m>). For the framework figure
    and for single-modality panels; nothing but the signal, in its colour."""
    pick, t0, t1 = ex["pick"], ex["t0"], ex["t1"]
    onset, offset = pick["onset_s"], pick["offset_s"]
    stamp, w.stamp = w.stamp, False          # a bare signal carries no corner stamp
    ws = (onset + (offset - onset) / 2 - WINDOW_S / 2) if window_start is None else onset + window_start
    ws = min(max(ws, t0), t1 - WINDOW_S)
    for m in MODALITIES:
        rows = [d for d in ex["traces"] if d["modality"] == m]
        if not rows:
            continue
        unit = rows[0]["unit"]
        # -- the span
        fig, axes = plt.subplots(len(rows), 1, figsize=(3.6, 0.6 * len(rows) + 0.45), sharex=True,
                                 squeeze=False, gridspec_kw={"height_ratios": [1] * (len(rows) - 1) + [1.6]})
        fig.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.02, hspace=0.05)
        amp = max(float(np.percentile(np.abs(d["y"]), 99.5)) for d in rows) or 1.0
        for i, (ax, d) in enumerate(zip(axes[:, 0], rows)):
            ax.plot(d["t"] - onset, d["y"], color=d["color"], lw=0.45)
            ax.axvspan(0, offset - onset, color="0.88", lw=0, zorder=0)
            # the last row keeps room under the trace for the scale bars
            ax.set_ylim(-amp * (2.0 if i == len(rows) - 1 else 1.15), amp * 1.15)
            ax.set_xlim(t0 - onset, t1 - onset)
            _bare(ax)
            ax.text(-0.01, 0.5 if i < len(rows) - 1 else 0.68, d["label"], transform=ax.transAxes,
                    fontsize=6, ha="right", va="center", color=d["color"])
        bar_t, bar_a = nice((t1 - t0) / 6), nice(amp)
        _scale_bars(axes[-1, 0], t1 - onset - bar_t - 1, -amp * 1.95, bar_t, bar_a,
                    f"{bar_t:g} s", f"{bar_a:g} {unit}")
        w.save(fig, f"fig_signal_{m}",
               data={"t": rows[0]["t"] - onset, **{f"y{i}": d["y"] for i, d in enumerate(rows)},
                     "onset": 0.0, "offset": offset - onset},
               meta={"recording": ex["rec"], "modality": m, "labels": [d["label"] for d in rows],
                     "unit": unit, "fs": rows[0]["fs"], "span_s": [t0 - onset, t1 - onset],
                     "scale_bar": {"s": bar_t, "amplitude": bar_a}})
        # -- one input window
        fig, axes = plt.subplots(len(rows), 1, figsize=(1.8, 0.5 * len(rows) + 0.3), sharex=True,
                                 squeeze=False, gridspec_kw={"height_ratios": [1] * (len(rows) - 1) + [1.5]})
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.05)
        seg = []
        for i, (ax, d) in enumerate(zip(axes[:, 0], rows)):
            k = (d["t"] >= ws) & (d["t"] < ws + WINDOW_S)
            y = d["y"][k]
            seg.append(y)
            a = float(np.abs(y).max()) if y.size else 1.0
            ax.plot(d["t"][k] - ws, y, color=d["color"], lw=0.6)
            ax.set_ylim(-a * (1.9 if i == len(rows) - 1 else 1.1), a * 1.1)
            ax.set_xlim(0, WINDOW_S)
            _bare(ax)
        a_last = float(np.abs(seg[-1]).max() or 1.0)
        _scale_bars(axes[-1, 0], WINDOW_S - 1.05, -a_last * 1.6, 1.0, 0.0, "1 s", "")
        w.save(fig, f"fig_window_{m}", data={f"y{i}": y for i, y in enumerate(seg)},
               meta={"recording": ex["rec"], "modality": m, "window_start_s": ws,
                     "window_start_from_onset_s": ws - onset, "seconds": WINDOW_S,
                     "fs": rows[0]["fs"], "samples": int(len(seg[0])), "unit": unit,
                     "labels": [d["label"] for d in rows]})
    w.stamp = stamp


def _plot_curve(ax, cur, color, label, lw=1.2, ls="-"):
    fa = np.array([c["official"]["fa_ovlp_h"] for c in cur])
    se = np.array([c["official"]["sens_ovlp"] for c in cur]) * 100
    ok = np.isfinite(fa) & np.isfinite(se)
    o = np.argsort(fa[ok])
    ax.plot(np.maximum(fa[ok][o], 0.05), se[ok][o], color=color, lw=lw, ls=ls, label=label)


def fig_sensitivity_fa(w: FigureWriter, main: Run, others: List[Run], eeg_runs: List[Run]):
    runs = [main] + others
    panels = [("(a) fusion runs, all modalities present", runs, None)]
    if eeg_runs:
        panels.append(("(b) EEG alone: one encoder, one protocol", eeg_runs, None))
    panels.append((f"({'abc'[len(panels)]}) {main.label}: modality subsets", None, main))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels), 2.7), squeeze=False)
    for ax, (title, rs, sub) in zip(axes[0], panels):
        if rs is not None:
            for i, r in enumerate(rs):
                cur = r.curve("complete")
                if cur:
                    _plot_curve(ax, cur, LINE_COLORS[i % len(LINE_COLORS)], r.label,
                                lw=1.6 if r is main else 1.0)
            for lab, se, fa, mk in PUBLISHED:
                ax.plot(fa, se, marker=mk, color="black", ms=4.5, ls="none", label=lab, zorder=5)
        else:
            for c, _ in SUBSETS:
                cur = sub.curve(c)
                if cur:
                    _plot_curve(ax, cur, SUBSET_COLOR[c], SUBSET_LABEL[c],
                                lw=1.6 if c == "complete" else 0.9)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 150)
        ax.set_ylim(0, 100)
        ax.axvline(FA_BUDGET, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("false alarms / h")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.set_title(title, fontsize=7.5)
        ax.legend(frameon=False, fontsize=5.2, loc="lower right", handlelength=1.6)
    axes[0][0].set_ylabel("sensitivity (%)")
    fig.suptitle("SeizeIT2 test patients (sub-097..125), the dataset paper's scoring (means over "
                 f"recordings, any-overlap); dotted: {FA_BUDGET:g} FA/h, the SVM's operating point",
                 y=1.04, fontsize=7.5)
    fig.subplots_adjust(wspace=0.25)
    w.save(fig, "fig_sensitivity_fa",
           data={f"{r.name}_{c}": np.array([[x["th"], x["official"]["fa_ovlp_h"],
                                             x["official"]["sens_ovlp"]] for x in r.curve(c)])
                 for r in runs + eeg_runs for c, _ in SUBSETS if r.curve(c)},
           meta={"runs": [r.name for r in runs], "eeg_runs": [r.name for r in eeg_runs],
                 "published": PUBLISHED})


def fig_modality_subsets(w: FigureWriter, main: Run, others: List[Run]):
    runs = [main] + others
    subsets = [c for c, _ in SUBSETS if any(r.curve(c) for r in runs)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    width = 0.8 / len(runs)
    x = np.arange(len(subsets))
    sens, auprc = {}, {}
    for i, r in enumerate(runs):
        s = [100 * (r.sens_at(c, FA_BUDGET) or 0) for c in subsets]
        p = [r.test().get(c, {}).get("auprc", np.nan) for c in subsets]
        sens[r.name], auprc[r.name] = s, p
        col = LINE_COLORS[i % len(LINE_COLORS)]
        a.bar(x + (i - (len(runs) - 1) / 2) * width, s, width, color=col)
        b.bar(x + (i - (len(runs) - 1) / 2) * width, p, width, color=col, label=r.label)
    a.axhline(71.1, color="black", lw=0.7, ls="--")
    a.legend(handles=[Line2D([], [], color="black", lw=0.7, ls="--",
                             label="SVM, bte-EEG (Bhagubai et al.): 71.1")],
             frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0, 1.0),
             borderaxespad=0)
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels([SUBSET_LABEL[c] for c in subsets], rotation=25, ha="right")
    a.set_ylabel(f"sensitivity at ≤ {FA_BUDGET:g} FA/h (%)")
    a.set_title("(a) event-level, continuous recordings", pad=14)
    a.set_ylim(0, 100)
    b.set_ylabel("window-level AUPRC")
    b.set_title("(b) window-level test (background sub-sampled)")
    top = np.nanmax([v for vv in auprc.values() for v in vv] + [0.1])
    b.set_ylim(0, top * 1.35)
    b.legend(frameon=False, fontsize=5.5, ncol=1, loc="upper right")
    fig.suptitle("Modality subsets: which signals carry the detection", y=1.1)
    w.save(fig, "fig_modality_subsets", data={"subsets": np.array(subsets),
                                              **{f"sens_{k}": v for k, v in sens.items()},
                                              **{f"auprc_{k}": v for k, v in auprc.items()}},
           meta={"fa_budget": FA_BUDGET, "runs": [r.name for r in runs]})


def fig_reliability(w: FigureWriter, main: Run, others: List[Run]):
    t = main.test()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.0, 1.1]})
    fig.subplots_adjust(wspace=0.35, right=0.82)
    labels, ticks, vals, cols, edges, pos, groups, k = [], [], [], [], [], [], [], 0
    for m in MODALITIES:
        items = [("complete", "clean")] + [(cid, lab) for cid, mm, lab in CORRUPTIONS if mm == m]
        start = k
        for cid, lab in items:
            v = t.get(cid, {}).get(f"reliability_{m}")
            if v is None:
                continue
            labels.append(f"{MOD_LABEL[m]} {lab}")
            ticks.append(lab.replace("bad channel", "bad ch."))
            vals.append(v)
            cols.append(MOD_COLOR[m] if lab == "clean" else matplotlib.colors.to_rgba(MOD_COLOR[m], 0.45))
            edges.append(MOD_COLOR[m])
            pos.append(k)
            k += 1
        if k > start:
            groups.append((MOD_LABEL[m], (start + k - 1) / 2, MOD_COLOR[m]))
        k += 0.7
    a.bar(pos, vals, 0.8, color=cols, edgecolor=edges, lw=0.6)
    a.set_xticks(pos)
    a.set_xticklabels(ticks, fontsize=6, rotation=45, ha="right")
    for name, centre, col in groups:
        a.text(centre, 1.02, name, ha="center", va="bottom", fontsize=7, color=col)
    a.set_ylim(0, 1)
    a.set_ylabel("predicted reliability (test mean)")
    a.set_title(f"(a) {main.label}: the reliability head", pad=16)
    runs = [main] + others
    conds = [cid for cid, _, _ in CORRUPTIONS if any(cid in r.test() for r in runs)]
    width = 0.8 / len(runs)
    x = np.arange(len(conds))
    au, clean = {}, {}
    for i, r in enumerate(runs):
        c0 = r.test().get("complete", {}).get("auprc", np.nan)
        clean[r.name] = c0
        v = [c0 - r.test().get(c, {}).get("auprc", np.nan) for c in conds]
        au[r.name] = v
        b.bar(x + (i - (len(runs) - 1) / 2) * width, v, width,
              color=LINE_COLORS[i % len(LINE_COLORS)], label=f"{r.label} (clean {c0:.3f})")
    b.axhline(0, color="black", lw=0.6)
    b.set_xticks(x)
    b.set_xticklabels([f"{MOD_LABEL[m]} {lab}" for cid, m, lab in CORRUPTIONS if cid in conds],
                      rotation=30, ha="right", fontsize=5.5)
    b.set_ylabel("window AUPRC lost (clean − corrupted)")
    b.set_title("(b) what each run loses under the same corruption (severity 0.7)", fontsize=7.5)
    b.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
             borderaxespad=0)
    fig.suptitle("Corrupted inputs: the reliability head notices, and what each fusion keeps", y=1.1)
    w.save(fig, "fig_reliability", data={"reliability_labels": np.array(labels), "reliability": vals,
                                         "conditions": np.array(conds),
                                         **{f"auprc_lost_{k}": v for k, v in au.items()},
                                         **{f"auprc_clean_{k}": v for k, v in clean.items()}},
           meta={"severity": (main.results or {}).get("corruption_severity")})


def fig_per_patient(w: FigureWriter, main: Run):
    per = main.per_patient(0.5)
    subs = sorted(per, key=lambda s: (per[s]["n_seizures"] == 0,
                                      -(per[s]["hit"] / per[s]["n_seizures"]) if per[s]["n_seizures"] else 0))
    sens = [100 * per[s]["hit"] / per[s]["n_seizures"] if per[s]["n_seizures"] else np.nan for s in subs]
    fah = [per[s]["false_events"] / per[s]["hours"] if per[s]["hours"] else np.nan for s in subs]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})
    x = np.arange(len(subs))
    a.bar(x, sens, color=OKABE_ITO[0])
    for i, s in enumerate(subs):
        a.text(i, (sens[i] if sens[i] == sens[i] else 0) + 2, str(per[s]["n_seizures"]), ha="center",
               va="bottom", fontsize=5.5)
    a.set_ylabel("sensitivity at 0.5 (%)")
    a.set_ylim(0, 112)
    a.set_title(f"{main.label}: per test patient (number above each bar: annotated seizures)")
    b.bar(x, fah, color=OKABE_ITO[1])
    b.set_ylim(bottom=0)
    b.set_ylabel("FA / h")
    b.set_xticks(x)
    b.set_xticklabels([s.replace("sub-", "") for s in subs], rotation=90, fontsize=6)
    b.set_xlabel("patient")
    w.save(fig, "fig_per_patient", data={"patients": np.array(subs), "sens": sens, "fa_h": fah,
                                         "hours": [per[s]["hours"] for s in subs],
                                         "n_seizures": [per[s]["n_seizures"] for s in subs]},
           meta={"threshold": 0.5, "gate": "cross"})


def fig_learning_curves(w: FigureWriter, main: Run, others: List[Run], eeg_runs: List[Run]):
    groups = [("(a) three modalities", [main] + others), ("(b) EEG alone", eeg_runs)]
    groups = [(t, rs) for t, rs in groups if rs]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 2.6), squeeze=False)
    from matplotlib.ticker import MaxNLocator
    data = {}
    for ax, (title, rs) in zip(axes[0], groups):
        for i, r in enumerate(rs):
            v = [e.get("val", {}).get("seizeit2", {}).get("auprc", np.nan) for e in r.history]
            if not v:
                continue
            data[r.name] = v
            col = LINE_COLORS[i % len(LINE_COLORS)]
            ax.plot(range(len(v)), v, color=col, lw=1.1, label=r.label)
            be = (r.results or {}).get("best_epoch")
            if be is not None and be < len(v):
                ax.plot(be, v[be], "o", color=col, ms=4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3),
                  ncol=2, borderaxespad=0)
    axes[0][0].set_ylabel("validation AUPRC (windows)")
    fig.suptitle("Learning curves; the dot is the epoch the test figures come from", y=1.03)
    w.save(fig, "fig_learning_curves", data=data,
           meta={"runs": [r.name for r in [main] + others + eeg_runs]})


CORRUPTION_TICK = {"noise": "noise", "bad channel": "bad ch.", "drift": "drift",
                   "saturation": "satur.", "dropout": "dropout"}
#: fig_main's palette: one hue per encoder (PhysioWave-T navy, the others
#: teal, violet, red), greys for the fusion rules that are not ours. A frozen
#: encoder is its hue lightened.
ENC_COLOR = {"c1": "#0F4D92", "luna_large": "#42949E", "labram": "#9A4D8E", "eegpt": "#B64342",
             "luna_huge": "#2E7A84", "luna_base": "#7FC1C8", "c1native": "#3775BA",
             "c1siena": "#5B8FD0"}
ENC_NAME = {"c1": "PhysioWave-T", "luna_large": "LUNA-L", "luna_huge": "LUNA-H",
            "luna_base": "LUNA-B", "labram": "LaBraM", "eegpt": "EEGPT",
            "c1native": "PhysioWave-T (native)", "c1siena": "PhysioWave-T + Siena"}
#: Encoder parameters in millions, the encoder alone (every run adds the same
#: channel adapter and head): the architectures as physiowave/fusion/baselines.py
#: builds them, and PhysioWave-T on its E19_256 route (EEGC1Downstream without
#: the head). Recount with the snippet in docs/fusion.md when one changes.
ENC_PARAMS_M = {"c1": 25.4, "eegpt": 25.3, "labram": 5.8, "luna_large": 40.4, "luna_huge": 303.4}
MAIN = {"ours": "#0F4D92", "neutral": "#8C8C8C", "neutral_dark": "#4D4D4D", "teal": "#42949E"}
MAIN_MOD = {"eeg": "#0F4D92", "ecg": "#B64342", "emg": "#42949E"}
MAIN_FUSION = {"ralf": MAIN["ours"], "weighted_logit": MAIN["teal"], "concat": MAIN["neutral"]}


def lighten(color: str, amount: float = 0.55) -> tuple:
    """``color`` blended towards white by ``amount``."""
    r, g, b, _ = matplotlib.colors.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, 1.0)


def _alignment_audit(fig, out_json: str):
    """The nature-figure skill's render-time panel-alignment gate, when its
    scripts directory is named in $PW_FIGURE_AUDIT; otherwise nothing."""
    d = os.environ.get("PW_FIGURE_AUDIT")
    if not d:
        return
    if d not in sys.path:
        sys.path.insert(0, d)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    rep = require_matplotlib_panel_alignment(fig, json_out=out_json, tolerance_pt=1.5,
                                             gutter_tolerance_pt=1.5, strict=False)
    print(f"  panel alignment: {rep.get('status', rep.get('verdict', '?'))} -> {out_json}")


def _encoder_key(run: Run) -> Optional[str]:
    import re as _re
    m = _re.match(r"eeg_(.+?)_(frozen|finetune|scratch)(?:_lr(.+))?$",
                  run.name.replace("_sze", ""))
    return m.group(1) if m else None


def _curve_data(r: Run, cond: str = "complete"):
    return np.array([[x["th"], x["official"]["fa_ovlp_h"], x["official"]["sens_ovlp"]]
                     for x in r.curve(cond)])


def fig_main(w: FigureWriter, main: Run, others: List[Run], eeg_runs: List[Run]):
    """The four-panel summary figure, SeizeIT2 only.

    (a) the hero: the main fusion run against the EEG-alone encoders (each at
    its better mode); (b) those encoders frozen and fine-tuned at the budget;
    (c) the mechanism RALF adds, its reliability head under corruption; (d) the
    window-level AUPRC -- the metric the runs are selected on -- of every
    modality subset of the fusion runs. Fusion runs of the main run's mode only.
    """
    others = [r for r in others if r.mode == main.mode]
    fusion = [main] + others
    ralf = next((r for r in fusion if r.fusion == "ralf"), main)
    # Several runs of one fusion rule (the RALF schedule sweep): the rule is
    # drawn as its main run's line with a min-max band over all its runs, in
    # (a) and (d), so the reader sees the spread of the rule, not one draw.
    by_rule: Dict[str, List[Run]] = {}
    for r in fusion:
        by_rule.setdefault(r.fusion, []).append(r)
    rule_main = {rule: (main if main in rs else rs[0]) for rule, rs in by_rule.items()}

    def fa_band(runs: List[Run], grid: np.ndarray):
        """Min and max sensitivity (%) over ``runs`` on a FA/h grid."""
        S, lo_fa, hi_fa = [], 0.05, np.inf
        for r in runs:
            cur = r.curve("complete")
            fa = np.array([c_["official"]["fa_ovlp_h"] for c_ in cur])
            se = np.array([c_["official"]["sens_ovlp"] for c_ in cur]) * 100
            ok = np.isfinite(fa) & np.isfinite(se)
            fa, se = np.maximum(fa[ok], 0.05), se[ok]
            ufa = np.unique(fa)                       # the best sensitivity at each FA/h
            use = np.array([se[fa == u].max() for u in ufa])
            S.append(np.interp(np.log(grid), np.log(ufa), use))
            lo_fa, hi_fa = max(lo_fa, ufa[0]), min(hi_fa, ufa[-1])
        S = np.array(S)
        inside = (grid >= lo_fa) & (grid <= hi_fa)     # no band where a run has no curve
        lo, hi = S.min(0), S.max(0)
        lo[~inside] = np.nan
        hi[~inside] = np.nan
        return lo, hi
    fus_col = {r.name: MAIN_FUSION.get(r.fusion, MAIN["neutral_dark"]) for r in fusion}
    fus_col[main.name] = MAIN["ours"]
    # the EEG-alone encoders: per encoder, its runs by mode
    enc: Dict[str, Dict[str, Run]] = {}
    for r in eeg_runs:
        k = _encoder_key(r)
        if k and "_lr" not in r.name and r.curve("complete"):
            enc.setdefault(k, {})[r.mode] = r
    keys = sorted(enc, key=lambda k: (k != "c1", -max(
        r.sens_at("complete", FA_BUDGET) or 0 for r in enc[k].values())))

    def best_mode(k: str) -> Run:
        return max(enc[k].values(), key=lambda r: r.sens_at("complete", FA_BUDGET) or 0)

    rc = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
          "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
          "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.2, 5.0))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.5,
                                 left=0.07, right=0.985, top=0.93, bottom=0.1)
        top = outer[0].subgridspec(1, 2, width_ratios=[1.75, 1.25], wspace=0.34)
        bot = outer[1].subgridspec(1, 2, width_ratios=[1.35, 1.65], wspace=0.3)
        a, b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
        c, d = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])
        data, meta = {}, {"fa_budget": FA_BUDGET, "fusion_runs": [r.name for r in fusion],
                          "eeg_runs": [r.name for r in eeg_runs], "reliability_run": ralf.name}

        def letter(ax, ch, title):
            # the letter and the title on one baseline
            ax.text(-0.015, 1.04, ch, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
                    ha="right", va="baseline")
            ax.text(0.01, 1.04, title, transform=ax.transAxes, fontsize=7.5, ha="left",
                    va="baseline")

        # (a) sensitivity against FA/h: the main fusion run and every EEG-alone
        # encoder at its better mode
        _plot_curve(a, main.curve("complete"), MAIN["ours"],
                    f"PhysioWave-T + {main.short_label} (EEG, ECG, EMG)", lw=2.0)
        data[f"curve_{main.name}"] = _curve_data(main)
        band_runs = by_rule[main.fusion]
        if len(band_runs) > 1:
            grid = np.geomspace(0.05, 100, 300)
            lo_b, hi_b = fa_band(band_runs, grid)
            a.fill_between(grid, lo_b, hi_b, color=MAIN["ours"], alpha=0.14, lw=0, zorder=1,
                           label=f"{main.short_label}, {len(band_runs)} training variants")
            data[f"band_{main.fusion}_fa"] = grid
            data[f"band_{main.fusion}_lo"] = lo_b
            data[f"band_{main.fusion}_hi"] = hi_b
        for rule, r in rule_main.items():          # the other fusion rules, same encoders
            if r is main:
                continue
            _plot_curve(a, r.curve("complete"), MAIN_FUSION.get(rule, MAIN["neutral"]),
                        f"PhysioWave-T + {r.short_label} (EEG, ECG, EMG)", lw=1.1)
            data[f"curve_{r.name}"] = _curve_data(r)
        a_runs = []
        for k in keys:
            r = best_mode(k)
            a_runs.append(r.name)
            _plot_curve(a, r.curve("complete"), ENC_COLOR.get(k, MAIN["neutral"]),
                        f"{ENC_NAME.get(k, k)} (EEG)", lw=1.0, ls="--")
            data[f"curve_{r.name}"] = _curve_data(r)
        a.axvline(FA_BUDGET, color=MAIN["neutral"], lw=0.6, ls=":")
        a.text(FA_BUDGET * 1.12, 2, f"{FA_BUDGET:g} FA/h", fontsize=6, color=MAIN["neutral_dark"],
               ha="left", va="bottom")
        a.set_xscale("log")
        a.set_xlim(0.05, 100)
        a.set_ylim(0, 100)
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        a.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel("False alarms per hour")
        a.set_ylabel("Sensitivity (%)")
        letter(a, "a", "Event-level seizure detection")
        a.legend(frameon=False, loc="upper left", handlelength=2.2, labelspacing=0.4)
        meta["a_curves"] = [main.name] + [r.name for r in rule_main.values() if r is not main] + a_runs
        meta["band_runs"] = [r.name for r in band_runs] if len(band_runs) > 1 else []

        # (b) the EEG-alone encoders frozen and fine-tuned, at the budget
        xb = np.arange(len(keys))
        wb = 0.36
        for j, mode in enumerate(("frozen", "finetune")):
            vals = [100 * (enc[k][mode].sens_at("complete", FA_BUDGET) or 0) if mode in enc[k]
                    else np.nan for k in keys]
            cols = [ENC_COLOR.get(k, MAIN["neutral"]) if mode == "finetune"
                    else lighten(ENC_COLOR.get(k, MAIN["neutral"])) for k in keys]
            b.bar(xb + (j - 0.5) * wb, vals, wb * 0.94, color=cols)
            data[f"encoders_{mode}"] = vals
        b.set_xticks(xb)                       # the encoder's size under its name
        b.set_xticklabels([ENC_NAME.get(k, k) + (f"\n{ENC_PARAMS_M[k]:.1f}M"
                                                 if k in ENC_PARAMS_M else "") for k in keys])
        b.set_xlim(-0.6, len(keys) - 0.4)
        b.set_ylim(30, 70)
        b.set_ylabel(f"Sensitivity at ≤ {FA_BUDGET:g} FA/h (%)")
        letter(b, "b", "Single-modality EEG encoders")
        from matplotlib.patches import Patch
        b.legend(handles=[Patch(color=lighten(MAIN["neutral_dark"]), label="frozen encoder"),
                          Patch(color=MAIN["neutral_dark"], label="fine-tuned")],
                 frameon=False, loc="upper right", handlelength=1.2, borderaxespad=0.2)
        data["encoders"] = np.array(keys)
        meta["encoders"] = keys

        # (c) RALF's reliability head: one line per modality across the clean set
        # and the corruptions; the corrupted modality's point ringed
        t = ralf.test()
        conds = [("complete", None, "clean")] + [(cid, m, lab) for cid, m, lab in CORRUPTIONS
                                                 if cid in t]
        xc = np.arange(len(conds))
        M = np.array([[t.get(cid, {}).get(f"reliability_{m}", np.nan) for cid, _, _ in conds]
                      for m in MODALITIES])
        off = {"eeg": -0.22, "ecg": 0.0, "emg": 0.22}       # the three dots side by side
        for i, m in enumerate(MODALITIES):
            c.axhline(M[i, 0], color=MAIN_MOD[m], lw=0.5, ls=":", alpha=0.7, zorder=1)
            c.plot(xc + off[m], M[i], "o", color=MAIN_MOD[m], ms=3.6, label=m.upper(),
                   zorder=3)
        for j, (cid, m, _) in enumerate(conds):
            if m is not None:                     # the corrupted modality: clean level -> now
                i = MODALITIES.index(m)
                xj = j + off[m]
                c.annotate("", (xj, M[i, j] + 0.035), (xj, M[i, 0] - 0.025),
                           arrowprops={"arrowstyle": "-|>", "color": MAIN_MOD[m],
                                       "lw": 0.8, "shrinkA": 0, "shrinkB": 0}, zorder=2)
                c.plot(xj, M[i, j], "o", ms=6.8, mfc="none", mec="black", mew=0.8, zorder=4)
        c.set_xticks(xc)
        c.set_xticklabels(["clean"] + [f"{m.upper()}\n{CORRUPTION_TICK.get(lab, lab)}"
                                      for _, m, lab in conds[1:]], fontsize=6)
        c.set_xlim(-0.5, len(conds) - 0.5)
        c.set_ylim(0, 1.05)
        c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        c.set_ylabel("Predicted reliability")
        c.legend(frameon=False, loc="lower left", ncol=3, handlelength=1.0, columnspacing=1.0,
                 borderaxespad=0.2, handletextpad=0.4)
        letter(c, "c", "Reliability under corrupted inputs")
        data["reliability"] = M
        data["reliability_conditions"] = np.array([cid for cid, _, _ in conds])
        meta["corruption_severity"] = (ralf.results or {}).get("corruption_severity")

        # (d) window-level AUPRC of every modality subset, one line per fusion rule
        order = ["eeg_only", "ecg_only", "emg_only", "eeg_ecg", "eeg_emg", "ecg_emg", "complete"]
        subsets = [cnd for cnd in order if any(cnd in r.test() for r in fusion)]
        x = np.arange(len(subsets))
        ends = []
        for r in fusion:
            data[f"auprc_subsets_{r.name}"] = [r.test().get(cnd, {}).get("auprc", np.nan)
                                               for cnd in subsets]
        for rule, r in rule_main.items():
            v = data[f"auprc_subsets_{r.name}"]
            rs = by_rule[rule]
            if len(rs) > 1:                        # the rule's spread over its runs
                V = np.array([data[f"auprc_subsets_{q.name}"] for q in rs], dtype=float)
                d.fill_between(x, np.nanmin(V, 0), np.nanmax(V, 0), color=fus_col[r.name],
                               alpha=0.14, lw=0, zorder=1)
            d.plot(x, v, marker="o", ms=3.8 if r is main else 2.8, color=fus_col[r.name],
                   lw=1.8 if r is main else 0.9, zorder=3 if r is main else 2)
            lab = r.short_label[0].upper() + r.short_label[1:]
            if len(rs) > 1:
                lab += f" ({len(rs)} runs)"
            ends.append((v[-1], lab, fus_col[r.name]))
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.016)
        for (y0, lab, col), y in zip(ends, ys):
            d.annotate(lab, (x[-1], y0), (x[-1] + 0.25, y), fontsize=6.5, color=col,
                       va="center", ha="left", annotation_clip=False)
        d.set_xticks(x)
        d.set_xticklabels([SUBSET_LABEL[cnd].replace("+", "\n+") for cnd in subsets])
        d.set_xlim(-0.4, len(subsets) - 0.6)
        lo = np.nanmin([v for k_, v in data.items() if k_.startswith("auprc_subsets_")])
        d.set_ylim(np.floor(lo * 20) / 20 - 0.02, 0.44)
        d.set_ylabel("Window-level AUPRC")
        letter(d, "d", "Modality combinations")
        meta["subsets"] = subsets

        sev = meta["corruption_severity"]
        fig.suptitle("SeizeIT2, official split, 29 test patients (2,912 h, 181 seizures). (a), (b): "
                     "the dataset paper's event scoring (any-overlap, recording means), the "
                     f"{FA_BUDGET:g} FA/h line is the SVM's operating point; each EEG-alone encoder in "
                     "(a) at its better of frozen / fine-tuned; (d): the window-level test set, "
                     "the metric the runs are selected on"
                     + (f"; the {main.short_label} band in (a), (d) is the min-max over its "
                        f"{len(band_runs)} frozen training variants (the schedule sweep), the "
                        "line the original configuration" if len(band_runs) > 1 else "")
                     + (f"; (c) corruption severity {sev:g}" if sev is not None else ""),
                     y=1.0, fontsize=6)
        _alignment_audit(fig, os.path.join(w.meta_dir, "fig_main.alignment.json"))
        w.save(fig, "fig_main", data=data, meta=meta)


# --------------------------------------------------------------------------- #

def write_readme(out_dir: str, w: FigureWriter, main: Run, args):
    lines = [f"# SeizeIT2 multimodal figures: {main.name}", "",
             f"Drawn by `scripts/visualize_seizeit2.py` at commit `{git_commit()}` from "
             f"`{main.path}` (and the runs it is compared with). Vector figures with editable "
             "text; the arrays behind each are in `figure_data/` (not committed), what produced "
             "each in `figure_metadata/`.", ""]
    for name in w.written:
        meta = json.load(open(os.path.join(w.meta_dir, f"{name}.json")))
        lines += [f"## {name}", ""]
        seen = set()
        for t in meta.get("figure_text", []):
            if t not in seen:
                lines.append(t)
                seen.add(t)
        if name == "fig_seizure_example":
            c = meta["chosen"]
            lines.append(f"\nRecording `{meta['recording']}`, chosen as rank {c['rank']} of "
                         f"{meta['n_candidates']} candidates by probability contrast "
                         f"({c['contrast']:.2f}); say so in the caption.")
        lines += ["", f"![{name}](figures/{name}.{w.formats[-1]})", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="SeizeIT2 BIDS root")
    ap.add_argument("--run-dir", required=True, help="the main run (RALF)")
    ap.add_argument("--compare", nargs="*", default=[], help="other trimodal runs")
    ap.add_argument("--eeg-runs", nargs="*", default=[], help="EEG-only runs (C1, EEGPT, LaBraM)")
    ap.add_argument("--out-dir", default=None, help="default <run-dir>/figures_seizeit2")
    ap.add_argument("--format", default="svg,pdf")
    ap.add_argument("--paper", action="store_true", help="no titles/notes/stamp on the figures")
    ap.add_argument("--example", default=None, help="recording for fig_seizure_example, "
                                                     "e.g. sub-101_run-03 (default: ranked)")
    ap.add_argument("--margin", type=float, default=60.0, help="seconds around the seizure")
    ap.add_argument("--window-start", type=float, default=None,
                    help="start of the 4 s input window drawn per modality, in seconds from the "
                         "seizure onset (default: centred on the seizure)")
    ap.add_argument("--only", nargs="*", default=None, help="figure names to draw")
    args = ap.parse_args(argv)
    apply_style()
    main_run = Run(args.run_dir)
    others = [Run(p) for p in args.compare]
    eeg_runs = [Run(p) for p in args.eeg_runs]
    for r in [main_run] + others + eeg_runs:
        print(f"  {r.name}: results {'yes' if r.results else 'NO'}, events "
              f"{'yes' if r.events else 'NO'}, predictions {len(r.predictions())}, "
              f"epochs {len(r.history)}")
    out_dir = args.out_dir or os.path.join(main_run.path, "figures_seizeit2")
    os.makedirs(out_dir, exist_ok=True)
    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    w = FigureWriter(out_dir, fmts, {"checkpoint": os.path.join(main_run.path, "best.pth"),
                                     "epoch": (main_run.results or {}).get("best_epoch"),
                                     "run": main_run.name, "git_commit": git_commit()},
                     paper=args.paper)
    want = set(args.only) if args.only else None

    def do(name):
        return want is None or name in want
    if (do("fig_seizure_example") or do("fig_signal")) and main_run.predictions():
        ex = load_example(main_run, args.raw_dir, args.example, args.margin)
        if do("fig_seizure_example"):
            fig_seizure_example(w, main_run, args.raw_dir, args.example, args.margin, ex)
        if do("fig_signal"):
            fig_signal_strips(w, main_run, ex, args.window_start)
    if do("fig_main") and main_run.events:
        fig_main(w, main_run, others, eeg_runs)
    if do("fig_sensitivity_fa") and main_run.events:
        fig_sensitivity_fa(w, main_run, others, eeg_runs)
    if do("fig_modality_subsets") and main_run.events:
        fig_modality_subsets(w, main_run, others)
    if do("fig_reliability") and main_run.results:
        fig_reliability(w, main_run, others)
    if do("fig_per_patient") and main_run.predictions():
        fig_per_patient(w, main_run)
    if do("fig_learning_curves") and main_run.history:
        fig_learning_curves(w, main_run, others, eeg_runs)
    write_readme(out_dir, w, main_run, args)
    print(f"wrote {len(w.written)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
