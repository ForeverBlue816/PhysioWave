"""Event-level seizure-detection scoring on continuous recordings.

Two scorers over the same per-second probabilities:

``official``  a port of the SeizeIT2 reference code (github.com/biomedepi/
              seizeit2, net/utils.py and net/main_func.py), so that a number
              here can sit in the same table as the dataset paper's SVM and
              ChronoNet. Per recording: predictions at 1 Hz; a second whose
              2 s EEG window has an RMS outside 13-150 uV is forced to 0
              (the reference gates on the cross-head channel alone -- its
              ``rmsa_f and rmsa_c`` evaluates to ``rmsa_c`` -- and so does
              ``rms_gate="cross"``); threshold; predicted events closer than
              2 s merged and events shorter than 8 s dropped (margin 10 s);
              any-overlap TP / FP / FN where TP counts (prediction, seizure)
              PAIRS that overlap; FA/h from those FP; the epoch-based
              (per-second) counts as well. The figures are averaged over
              recordings (nanmean: a recording without a seizure has no
              sensitivity), at thresholds 0, 0.02, ..., 1.
``events``    the usual event-based definition (SzCORE / Shah et al. any-overlap)
              pooled over all recordings: sensitivity = seizures hit by at least
              one predicted event / seizures, FA/24h = predicted events touching
              no seizure per 24 h of recording. Same post-processing.

Both report a sensitivity -- false-alarm curve over thresholds. Two kinds of
area under a curve, which are NOT the same quantity:

``auroc_paper`` / ``aupr_paper``  the dataset paper's (Bhagubai et al., Sci.
              Data 2025, Table 2): "derived from metrics calculated using the
              traditional epoch-based method for computing sensitivity/recall,
              specificity and precision", i.e. after the gate and the
              post-processing, per recording, averaged over recordings, at
              the 51 thresholds -- the area under mean (1 - specificity,
              sensitivity) and under mean (recall, precision). The released
              code does not compute them; this is the reading of that sentence
              that uses the code's own per-threshold metrics. Comparable with
              the paper's AUROC / AUPR, with that caveat.
``auroc_epoch`` / ``auprc_epoch``  pooled over every gated second of every
              recording, before post-processing. A model-quality number of our
              own; its AUPRC sits at the per-second seizure prevalence
              (~0.1%), far below the paper's AUPR, and is not comparable with it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THRESHOLDS = np.round(np.linspace(0, 1, 51), 2)
MARGIN = 10.0                   # post-processing: merge < 0.2*margin, keep >= 0.8*margin


def mask_to_events(mask: np.ndarray, fs: float = 1.0) -> List[List[float]]:
    """Runs of True -> ``[[start_s, end_s], ...]`` (end exclusive)."""
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [[s / fs, e / fs] for s, e in zip(starts, ends)]


def events_to_mask(events: Sequence[Sequence[float]], n: int, fs: float = 1.0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in events:
        m[max(0, int(a * fs)):min(n, int(b * fs))] = True
    return m


def merge_events(events: List[List[float]], distance: float) -> List[List[float]]:
    out = [list(e) for e in events]
    i = 1
    while i < len(out):
        if out[i][0] - out[i - 1][1] < distance:
            out[i - 1][1] = out[i][1]
            out.pop(i)
        else:
            i += 1
    return out


def post_process(prob: np.ndarray, th: float, fs: float = 1.0, margin: float = MARGIN
                 ) -> np.ndarray:
    """Threshold, merge events closer than 0.2*margin, drop those shorter than 0.8*margin."""
    ev = merge_events(mask_to_events(prob > th, fs), 0.2 * margin)
    ev = [e for e in ev if e[1] - e[0] >= 0.8 * margin]
    return events_to_mask(ev, len(prob), fs)


def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score_recording(prob: np.ndarray, label: np.ndarray, th: float, fs: float = 1.0
                    ) -> Dict[str, float]:
    """One recording at one threshold: the reference code's counts and rates,
    plus the pooled-event counts (hit seizures, false events)."""
    y_true = np.asarray(label, bool)
    y_pred = post_process(np.asarray(prob, float), th, fs)
    hours = len(y_pred) / fs / 3600.0
    t_ev, p_ev = mask_to_events(y_true, fs), mask_to_events(y_pred, fs)
    # reference any-overlap: TP counts overlapping (prediction, seizure) pairs
    tp_pairs = sum(1 for p in p_ev for t in t_ev if _overlap(p, t) > 0)
    fp = sum(1 for p in p_ev if not any(_overlap(p, t) > 0 for t in t_ev))
    fn = sum(1 for t in t_ev if not any(_overlap(t, p) > 0 for p in p_ev))
    hit = len(t_ev) - fn
    tp_e = int((y_true & y_pred).sum())
    fp_e = int((~y_true & y_pred).sum())
    fn_e = int((y_true & ~y_pred).sum())
    tn_e = int((~y_true & ~y_pred).sum())
    has_sz = bool(y_true.any())
    sens = tp_pairs / (tp_pairs + fn) if has_sz else np.nan
    prec = tp_pairs / (tp_pairs + fp) if has_sz and (tp_pairs + fp) else np.nan
    f1 = (2 * prec * sens / (prec + sens)) if has_sz and prec == prec and (prec + sens) else np.nan
    sens_e = tp_e / (tp_e + fn_e) if has_sz else np.nan
    prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) else np.nan
    return {"sens_ovlp": sens, "prec_ovlp": prec, "f1_ovlp": f1,
            "fa_ovlp_h": fp / hours if hours else np.nan,
            "sens_epoch": sens_e, "prec_epoch": prec_e,
            "spec_epoch": tn_e / (tn_e + fp_e) if (tn_e + fp_e) else np.nan,
            "fa_epoch_h": fp_e / hours if hours else np.nan,
            # pooled-event counts
            "n_seizures": len(t_ev), "hit": hit, "false_events": fp, "hours": hours}


def evaluate(recordings: Sequence[Tuple[np.ndarray, np.ndarray]],
             thresholds: Sequence[float] = THRESHOLDS, fs: float = 1.0,
             fa_grid: Sequence[float] = (0.5, 1.0, 2.0, 6.22, 11.0)) -> Dict[str, object]:
    """``recordings``: ``[(prob [n_seconds], label [n_seconds]), ...]``.

    Returns, per threshold, the reference figures (means over recordings) and
    the pooled event figures; the figures at threshold 0.5; the pooled
    epoch-based AUROC / AUPRC; and the best sensitivity reachable at each
    false-alarm budget in ``fa_grid`` (per hour), on both curves.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    import warnings
    per_th = []
    for th in thresholds:
        rows = [score_recording(p, y, th, fs) for p, y in recordings]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-nan columns stay nan
            mean = {k: float(np.nanmean([r[k] for r in rows]))
                    for k in ("sens_ovlp", "prec_ovlp", "f1_ovlp", "fa_ovlp_h", "sens_epoch",
                              "prec_epoch", "spec_epoch", "fa_epoch_h")}
        n_sz = sum(r["n_seizures"] for r in rows)
        hit = sum(r["hit"] for r in rows)
        fpe = sum(r["false_events"] for r in rows)
        hours = sum(r["hours"] for r in rows)
        pooled = {"sens": hit / n_sz if n_sz else float("nan"),
                  "fa_h": fpe / hours if hours else float("nan"),
                  "fa_24h": 24 * fpe / hours if hours else float("nan"),
                  "precision": hit / (hit + fpe) if (hit + fpe) else float("nan")}
        mean["score"] = mean["sens_ovlp"] * 100 - 0.4 * mean["fa_epoch_h"]
        per_th.append({"th": float(th), "official": mean, "events": pooled})
    probs = np.concatenate([np.asarray(p, float) for p, _ in recordings])
    labels = np.concatenate([np.asarray(y, bool) for _, y in recordings])
    auroc = float(roc_auc_score(labels, probs)) if labels.any() and (~labels).any() else float("nan")
    auprc = float(average_precision_score(labels, probs)) if labels.any() else float("nan")

    def best_at(curve_key, sens_key, fa_key, budget):
        ok = [r[curve_key][sens_key] for r in per_th
              if r[curve_key][fa_key] == r[curve_key][fa_key] and r[curve_key][fa_key] <= budget]
        return float(max(ok)) if ok else 0.0
    auroc_p, aupr_p = paper_areas(per_th)
    at05 = next(r for r in per_th if abs(r["th"] - 0.5) < 1e-9) if any(
        abs(r["th"] - 0.5) < 1e-9 for r in per_th) else per_th[len(per_th) // 2]
    return {
        "n_recordings": len(recordings),
        "n_seizures": int(sum(len(mask_to_events(np.asarray(y, bool), fs)) for _, y in recordings)),
        "hours": float(sum(len(p) for p, _ in recordings) / fs / 3600.0),
        "seizure_seconds_fraction": float(labels.mean()) if labels.size else float("nan"),
        "auroc_epoch": auroc, "auprc_epoch": auprc,
        "auroc_paper": auroc_p, "aupr_paper": aupr_p,
        "at_0.5": at05,
        "sens_at_fa_official": {str(b): best_at("official", "sens_ovlp", "fa_ovlp_h", b) for b in fa_grid},
        "sens_at_fa_events": {str(b): best_at("events", "sens", "fa_h", b) for b in fa_grid},
        "curve": per_th,
    }


def paper_areas(curve: Sequence[dict]) -> Tuple[float, float]:
    """The dataset paper's AUROC and AUPR from a per-threshold curve (``evaluate``'s
    ``curve``): areas under the recording-mean epoch-based (1 - spec, sens) and
    (recall, precision) points. ROC is closed at (0, 0) and (1, 1); PR runs
    over the thresholds where precision is defined and is extended flat from
    its lowest recall down to recall 0 (so a perfect detector scores 1).
    Older curves without ``prec_epoch`` give AUPR nan."""
    def trap(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    roc = [(1.0 - c["official"]["spec_epoch"], c["official"]["sens_epoch"]) for c in curve]
    roc = sorted([p for p in roc if p[0] == p[0] and p[1] == p[1]] + [(0.0, 0.0), (1.0, 1.0)])
    auroc = trap([p[0] for p in roc], [p[1] for p in roc])
    pr = [(c["official"]["sens_epoch"], c["official"].get("prec_epoch", float("nan")))
          for c in curve]
    pr = sorted(p for p in pr if p[0] == p[0] and p[1] == p[1])
    if pr and pr[0][0] > 0:
        pr = [(0.0, pr[0][1])] + pr
    aupr = trap([p[0] for p in pr], [p[1] for p in pr]) if len(pr) > 1 else float("nan")
    return auroc, aupr


def rms_gate(eeg_uv: Dict[str, np.ndarray], fs: float, n_seconds: int, mode: str = "cross"
             ) -> np.ndarray:
    """Per second k: True if the 2 s window from k passes 13 < RMS < 150 uV.

    The reference filters the channel first (0.5 Hz high-pass, 60 Hz low-pass,
    49.5-50.5 Hz band-stop, 4th-order Butterworth, zero phase) and gates on
    the cross-head channel (CROSStop, else BTEright); ``mode="both"`` requires
    both channels, ``"none"`` gates nothing.
    """
    if mode == "none":
        return np.ones(n_seconds, bool)
    from scipy.signal import butter, filtfilt
    names = list(eeg_uv)

    def pick(*cands):
        for c in cands:
            if c in eeg_uv:
                return eeg_uv[c]
        return None
    focal = pick("BTEleft SD", "BTEright SD")
    cross = pick("CROSStop SD", "BTEright SD")
    if cross is None and names:
        cross = eeg_uv[names[-1]]
    if focal is None and names:
        focal = eeg_uv[names[0]]

    def filt(x):
        y = np.asarray(x, float)
        for b, a in (butter(4, 0.5 / (fs / 2), "high"), butter(4, 60 / (fs / 2), "low"),
                     butter(4, [49.5 / (fs / 2), 50.5 / (fs / 2)], "bandstop")):
            y = filtfilt(b, a, y)
        return y

    def ok(x):
        y = filt(x)
        w, step = int(round(2 * fs)), int(round(fs))
        c = np.concatenate([[0.0], np.cumsum(y ** 2)])
        out = np.zeros(n_seconds, bool)
        starts = np.arange(n_seconds) * step
        full = starts + w <= len(y)
        r = np.sqrt((c[starts[full] + w] - c[starts[full]]) / w)
        out[np.where(full)[0]] = (r > 13.0) & (r < 150.0)
        return out
    if mode == "cross":
        return ok(cross)
    if mode == "both":
        return ok(cross) & ok(focal)
    raise ValueError(mode)


__all__ = ["THRESHOLDS", "post_process", "score_recording", "evaluate", "rms_gate", "paper_areas",
           "mask_to_events", "events_to_mask"]
