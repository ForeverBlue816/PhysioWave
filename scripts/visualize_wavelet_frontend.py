#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the learned wavelet front ends do: one figure for EEG, ECG and sEMG.

    python scripts/visualize_wavelet_frontend.py \
        --eeg $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth \
        --ecg $PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth \
        --emg $PW_CKPT_ROOT/pretrain_emg_c1_moe/epoch38.pth \
        --out-dir figures --report docs/runs/fig_wavelet.md

Each route owns a bank of M=5 learnable analysis filter pairs (16 taps, one
pair per electrode, initialised from named PyWavelets filters) and a selector
that predicts, from the window, softmax weights over the five. The filter a
window is actually analysed with is the bank mixed by those weights. This
script runs --windows pre-training windows of every corpus through the
selector, averages the weights, mixes the bank with the averages and draws:

  a  per route, the magnitude response of the learned low-pass and high-pass
     (mean over electrodes; the 10-90th percentiles are in figure_data)
     against the initialisation mixed with uniform weights, in Hz, with the
     modality's physiological bands shaded and the -3 dB cut-offs marked;
  b  per route, the mean selector weights at levels 1-3 as stacked bars, with
     the uniform initialisation as a reference bar carrying the wavelet names
     (the corpora of a route share one set of weights; the report lists them
     per corpus);
  c  per route, the low-pass -3 dB cut-off at initialisation and after
     training, over the modality's physiological band.

Redraw without the checkpoints, from a previous run's saved arrays:

    python scripts/visualize_wavelet_frontend.py --from-saved docs/runs/fig_wavelet \
        --out-dir figures

The selector is applied at every decomposition level to the current
approximation (one selector per route, three calls per window). The figure
shows LEVEL 1 -- the analysis of the raw window, where Hz on the axis means
what it says; levels 2 and 3 are measured on a subset and reported in the
markdown. Nothing is decimated between levels in this architecture, so a
level's filter always acts at the route's sampling rate.

Loading is checked tensor by tensor: for every route the count of front-end
tensors present in the checkpoint and equal to the loaded module's is
printed, and a route whose front end or selector is not fully loaded aborts.

Writes:
    <out-dir>/fig_wavelet.{pdf,png,svg}   the figure (FigureWriter layout:
                                           figures/, figure_data/, figure_metadata/)
    <report>                               docs/runs/fig_wavelet.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
import scripts.visualize_eeg_pretraining as viz                # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    PAPER, PAPER_FONT, PAPER_LETTER, PAPER_TITLE, PAPER_WIDTH, FigureWriter,
    _cap_threads, _paper_rc, apply_style, git_commit, sha256_file)
from physiowave.eeg_c1.data import CorpusIndex, EEGWindowDataset  # noqa: E402
from wavelet_modules import load_wavelet_kernel               # noqa: E402

MODALITIES = ("eeg", "ecg", "emg")
LABEL = {"eeg": "EEG", "ecg": "ECG", "emg": "sEMG"}
#: Modality colours from the journal palette: the learned low-pass in the
#: colour, the learned high-pass in its darker shade.
MOD_COLOR = {"eeg": PAPER["indigo"], "ecg": PAPER["mauve"], "emg": PAPER["gold"]}
MOD_DARK = {"eeg": "#23264F", "ecg": "#7A3466", "emg": "#9A6214"}
#: Five steps for the five basis slots of a bank, light to dark. The lightest
#: stays visibly blue on white: a bar that is nearly all of one basis (TUEG's
#: sym4) must still read as a bar. White hairlines separate the segments.
BASIS_COLORS = ["#C9D6EC", "#98AED6", "#7480BE", "#574F98", "#2E2F5F"]
#: Physiological bands, (label, lo, hi, emphasis) in Hz.
BANDS = {
    "eeg": [("δ", 0.5, 4, 0), ("θ", 4, 8, 0), ("α", 8, 13, 0), ("β", 13, 30, 0),
            ("γ", 30, 80, 0)],
    "ecg": [("P/T", 0.5, 10, 0), ("QRS", 10, 40, 0)],
    "emg": [("20–450 Hz", 20, 450, 0)],
}
#: Corpus display names (the EEG ones are visualize_eeg_pretraining's).
CORPUS_LABEL = {**viz.DATASET_LABEL,
                "mimic_iv_ecg": "MIMIC-IV-ECG", "code15": "CODE-15%", "sph": "SPH",
                "georgia": "Georgia", "medalcare_xl": "MedalCare-XL",
                "icentia11k": "Icentia11k", "pulsedb": "PulseDB",
                "emg2pose": "emg2pose", "emg2qwerty": "emg2qwerty", "hyser": "Hyser",
                "cemhsey_8x8": "CEMHSEY 8×8", "cemhsey_5x13": "CEMHSEY 5×13",
                "putemg": "putEMG"}
NFFT = 2048


# --------------------------------------------------------------------------- #
# Loading, with the tensor check the task asks for
# --------------------------------------------------------------------------- #

def load_checked(path: str, modality: str):
    """``(ck, cfg, mcfg, model, routes, datasets_spec, report)``.

    ``load_trained_model`` loads strictly and switches the figure module's
    registry to the modality. On top of that, every front-end tensor of every
    route is compared with the checkpoint's value after loading, so "loaded"
    means equal, not merely present.
    """
    ck, cfg, mcfg, objective, mod, model_cls, model = viz.load_trained_model(path, modality)
    if mod != modality:
        raise SystemExit(f"{path} is a {mod} checkpoint, given as --{modality}")
    sd_ck = ck["model"]
    sd_model = model.state_dict()
    report = {"checkpoint": path, "epoch": ck.get("epoch"),
              "tensors_in_checkpoint": len(sd_ck), "tensors_in_model": len(sd_model),
              "missing_in_checkpoint": sorted(set(sd_model) - set(sd_ck)),
              "unexpected_in_checkpoint": sorted(set(sd_ck) - set(sd_model)),
              "routes": {}}
    for rid in viz.ROUTES:
        prefix = f"wavelet_frontends.{rid}."
        keys = [k for k in sd_model if k.startswith(prefix)]
        matched = sum(1 for k in keys if k in sd_ck and torch.equal(sd_model[k], sd_ck[k]))
        sel_keys = [k for k in keys if ".decomp.selector." in k]
        sel_matched = sum(1 for k in sel_keys
                          if k in sd_ck and torch.equal(sd_model[k], sd_ck[k]))
        report["routes"][rid] = {"frontend_tensors": len(keys), "frontend_matched": matched,
                                 "selector_tensors": len(sel_keys),
                                 "selector_matched": sel_matched}
        print(f"  {LABEL[modality]} {rid}: front end {matched}/{len(keys)} tensors "
              f"matched, selector {sel_matched}/{len(sel_keys)}")
        if matched != len(keys) or sel_matched != len(sel_keys) or not keys:
            raise SystemExit(f"ABORT: {rid} front end did not load from {path}")
    return ck, cfg, mcfg, model, dict(viz.ROUTES), dict(viz.PRETRAIN_DATASETS), report


def open_datasets(cfg: Dict, split: str, routes) -> Dict[str, EEGWindowDataset]:
    manifest = cfg.get("data", {}).get(f"manifest_{split}")
    if not manifest or not os.path.isfile(manifest):
        raise SystemExit(f"manifest for split {split!r} not found: {manifest!r}")
    index = CorpusIndex.from_manifest(manifest)
    out = {}
    for d in sorted(index.by_dataset()):
        try:
            out[d] = EEGWindowDataset(index, d, routes=routes)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skipping {d}: {exc}")
    return out


def spread_indices(n_total: int, n: int) -> List[int]:
    if n_total <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, min(n, n_total))})


# --------------------------------------------------------------------------- #
# Selector weights and effective filters
# --------------------------------------------------------------------------- #

@torch.no_grad()
def selector_weights(model, route, ds, n_windows: int, n_deep: int, device,
                     chunk: int = 128):
    """Mean softmax weights per level: ``{1: [M], 2: [M], 3: [M]}`` and counts.

    Level 1 reads the selector's own MLP on the raw window (cheap, all
    ``n_windows``); levels 2 and 3 need the full decomposition and are
    measured on the first ``n_deep`` of the same windows.
    """
    front = model.wavelet_frontends[route.route_id].to(device)
    sel = front.decomp.selector
    idx = spread_indices(len(ds), n_windows)
    T = route.window_samples
    # The decomposition's cross-scale FFN attends over all T samples, so its
    # memory grows with B*T^2: a 10 s ECG window at 500 Hz took 48 GB at
    # B=128. The deep pass goes through in sub-batches sized to a budget.
    deep_bs = max(1, min(chunk, int(2.0e8 // (T * T))))
    sums = {1: None, 2: None, 3: None}
    sq1 = None                                   # for the across-window spread
    gap_abs = 0.0                                # mean |per-channel window mean|
    counts = {1: 0, 2: 0, 3: 0}
    captured: List[torch.Tensor] = []
    hook = sel.selector.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    try:
        for start in range(0, len(idx), chunk):
            x = ds.read_batch(idx[start:start + chunk]).to(device)
            w1 = sel.selector(x).float()                         # [B, M]
            sums[1] = w1.sum(0) if sums[1] is None else sums[1] + w1.sum(0)
            sq1 = (w1 ** 2).sum(0) if sq1 is None else sq1 + (w1 ** 2).sum(0)
            gap_abs += float(x.mean(dim=-1).abs().sum())
            counts[1] += w1.shape[0]
            need = n_deep - counts[2]
            for s0 in range(0, min(x.shape[0], max(need, 0)), deep_bs):
                captured.clear()
                front.decomp(x[s0:s0 + deep_bs])
                # One selector call per level, in order.
                for lvl, w in zip((1, 2, 3), captured[:3]):
                    if lvl == 1:
                        continue
                    w = w.float()
                    sums[lvl] = w.sum(0) if sums[lvl] is None else sums[lvl] + w.sum(0)
                    counts[lvl] += w.shape[0]
            del x
    finally:
        hook.remove()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    means = {lvl: (sums[lvl] / counts[lvl]).cpu().numpy() if counts[lvl] else None
             for lvl in sums}
    n1 = max(counts[1], 1)
    var1 = (sq1 / n1 - (sums[1] / n1) ** 2).clamp_min(0).cpu().numpy()
    stats = {"std_level1_across_windows": np.sqrt(var1).tolist(),
             "mean_abs_window_mean": gap_abs / (n1 * route.n_channels)}
    return means, counts, stats


def bank_kernels(model, route):
    """``(low [M, C, K], high [M, C, K])`` learned taps of the route's bank."""
    sel = model.wavelet_frontends[route.route_id].decomp.selector
    lo = torch.stack([f.low_filter.weight.detach().float().cpu()[:, 0, :]
                      for f in sel.wavelet_filters])
    hi = torch.stack([f.high_filter.weight.detach().float().cpu()[:, 0, :]
                      for f in sel.wavelet_filters])
    return lo.numpy(), hi.numpy()


def init_kernels(names: List[str], kernel_size: int, mode: str):
    lo, hi = zip(*[load_wavelet_kernel(n, kernel_size, mode) for n in names])
    return torch.stack(lo).numpy(), torch.stack(hi).numpy()         # [M, K]


def magnitude(kernels: np.ndarray, fs: float):
    """``kernels [..., K] -> (f [NFFT/2+1], |H| normalised to peak 1 per kernel)``."""
    H = np.abs(np.fft.rfft(kernels, n=NFFT, axis=-1))
    H = H / np.maximum(H.max(axis=-1, keepdims=True), 1e-12)
    f = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    return f, H


def cutoff_lowpass(f: np.ndarray, H: np.ndarray) -> float:
    """-3 dB point of a low-pass response: the first frequency above the peak
    where |H| crosses 1/sqrt(2), linearly interpolated."""
    k0 = int(np.argmax(H))
    thr = 1.0 / math.sqrt(2.0)
    for k in range(k0, len(H) - 1):
        if H[k] >= thr > H[k + 1]:
            t = (H[k] - thr) / max(H[k] - H[k + 1], 1e-12)
            return float(f[k] + t * (f[k + 1] - f[k]))
    return float("nan")


def cutoff_highpass(f: np.ndarray, H: np.ndarray) -> float:
    """-3 dB point of a high-pass response: the last frequency below the peak
    where |H| crosses 1/sqrt(2) rising."""
    k0 = int(np.argmax(H))
    thr = 1.0 / math.sqrt(2.0)
    for k in range(k0, 0, -1):
        if H[k] >= thr > H[k - 1]:
            t = (H[k] - thr) / max(H[k] - H[k - 1], 1e-12)
            return float(f[k] - t * (f[k] - f[k - 1]))
    return float("nan")


def analyse_route(model, route, mcfg, w_mean: Dict[int, Optional[np.ndarray]]):
    """Effective filters and responses for one route at every level measured."""
    names = list(mcfg.get("wavelet_names"))
    K = int(mcfg.get("wave_kernel_size", 16))
    mode = mcfg.get("wave_init_mode", "pad")
    lo, hi = bank_kernels(model, route)                       # [M, C, K]
    lo0, hi0 = init_kernels(names, K, mode)                   # [M, K]
    M = lo.shape[0]
    fs = float(route.sampling_rate)
    out = {"wavelet_names": names, "levels": {}}
    uni = np.full(M, 1.0 / M)
    eff0_lo = np.tensordot(uni, lo0, axes=1)                  # [K]
    eff0_hi = np.tensordot(uni, hi0, axes=1)
    f, H0_lo = magnitude(eff0_lo, fs)
    _, H0_hi = magnitude(eff0_hi, fs)
    out["f_hz"] = f
    out["init"] = {"H_lo": H0_lo, "H_hi": H0_hi,
                   "cutoff_lo_hz": cutoff_lowpass(f, H0_lo),
                   "cutoff_hi_hz": cutoff_highpass(f, H0_hi)}
    # How far the taps moved from their initialisation, per basis, relative.
    lo_dev = np.sqrt(((lo - lo0[:, None, :]) ** 2).mean(axis=(1, 2)))
    out["tap_rms_change_rel"] = (lo_dev / np.sqrt((lo0 ** 2).mean(axis=1))).tolist()
    # Each basis on its own: the half-band cut-off of its low-pass before and
    # after training (channel mean). Separates "the mix changed" from "the
    # filters changed" -- a uniform mix of five different wavelets is not
    # itself a half-band filter, since their phases do not line up.
    out["per_basis_cutoff_lo_hz_init"] = [cutoff_lowpass(f, magnitude(k, fs)[1])
                                          for k in lo0]
    _, Hb = magnitude(lo, fs)                                   # [M, C, F]
    out["per_basis_cutoff_lo_hz_learned"] = [
        float(np.nanmean([cutoff_lowpass(f, h) for h in Hb[i]])) for i in range(M)]
    for lvl, w in w_mean.items():
        if w is None:
            continue
        eff_lo = np.tensordot(w, lo, axes=1)                  # [C, K]
        eff_hi = np.tensordot(w, hi, axes=1)
        _, H_lo = magnitude(eff_lo, fs)                       # [C, F]
        _, H_hi = magnitude(eff_hi, fs)
        cut_lo = np.array([cutoff_lowpass(f, h) for h in H_lo])
        cut_hi = np.array([cutoff_highpass(f, h) for h in H_hi])
        out["levels"][lvl] = {
            "weights": w.tolist(),
            "H_lo_mean": H_lo.mean(0), "H_lo_p10": np.percentile(H_lo, 10, axis=0),
            "H_lo_p90": np.percentile(H_lo, 90, axis=0),
            "H_hi_mean": H_hi.mean(0), "H_hi_p10": np.percentile(H_hi, 10, axis=0),
            "H_hi_p90": np.percentile(H_hi, 90, axis=0),
            "cutoff_lo_hz_mean": float(np.nanmean(cut_lo)),
            "cutoff_lo_hz_p10": float(np.nanpercentile(cut_lo, 10)),
            "cutoff_lo_hz_p90": float(np.nanpercentile(cut_lo, 90)),
            "cutoff_lo_hz_of_mean_response": cutoff_lowpass(f, H_lo.mean(0)),
            "cutoff_hi_hz_mean": float(np.nanmean(cut_hi)),
            "cutoff_hi_hz_of_mean_response": cutoff_highpass(f, H_hi.mean(0)),
        }
    return out


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def _route_title(route) -> str:
    return f"C={route.n_channels}, {route.sampling_rate:g} Hz"


def _modality_tag(ax, m: str, y0: float, y1: float, x: float = 1.03):
    ax.text(x, 0.5 * (y0 + y1), LABEL[m], color=MOD_COLOR[m], fontweight="bold",
            ha="left", va="center", fontsize=PAPER_FONT, rotation=90,
            transform=ax.get_yaxis_transform())


def _titled_letter(fig, x: float, y: float, ch: str, title: str):
    """Panel letter with its title beside it, on one baseline."""
    fig.text(x, y, ch, fontsize=PAPER_LETTER, fontweight="bold", ha="left",
             va="baseline", gid="paper-keep")
    fig.text(x + 0.022, y, title, fontsize=PAPER_TITLE, fontweight="bold", ha="left",
             va="baseline", gid="paper-keep")


def draw(w: FigureWriter, R: Dict, out_name: str = "fig_wavelet"):
    """``R[modality] = {"routes": {rid: analysis}, "corpora": [...], "route_order": [...]}``.

    a  level-1 effective low-pass and high-pass per route (electrode mean)
       against the uniform initialisation, physiological bands shaded; the
       legend sits in the empty cells of the shortest row.
    b  route-mean selector weights at levels 1-3, one block per route. The
       corpora of a route get the same weights (the selector's input is ~0
       on normalised windows; see the report), so a row per corpus would
       repeat the route's bar.
    c  the low-pass -3 dB cut-off per route, initialisation to learned, over
       the modality's physiological band.
    """
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mods = [m for m in MODALITIES if m in R]
    n_cols = max(len(R[m]["route_order"]) for m in mods)
    inv2 = 1 / math.sqrt(2)
    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 5.6))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.36,
                                 left=0.062, right=0.968, top=0.915, bottom=0.075)
        ga = outer[0].subgridspec(len(mods), n_cols, wspace=0.12, hspace=0.62)
        gr = outer[1].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.30)

        # -- (a) frequency responses ---------------------------------------- #
        for r_, m in enumerate(mods):
            order = R[m]["route_order"]
            for c_, rid in enumerate(order):
                A = R[m]["routes"][rid]
                route = A["route"]
                ax = fig.add_subplot(ga[r_, c_])
                fs = float(route.sampling_rate)
                f = np.asarray(A["f_hz"])
                L = A["levels"][1]
                sel = (f >= 0.5) & (f <= fs / 2)
                for i, (lab, lo, hi, _emph) in enumerate(BANDS[m]):
                    hi_c = min(hi, fs / 2)
                    ax.axvspan(lo, hi_c, color=PAPER["pale_blue"],
                               alpha=0.50 if i % 2 == 0 else 0.25, lw=0, zorder=0)
                    if c_ == 0 and lab:
                        ax.text(math.sqrt(lo * hi_c), 1.15, lab, ha="center", va="bottom",
                                fontsize=PAPER_FONT - 1, color=PAPER["rule"], clip_on=False)
                if m == "ecg":
                    for mains in (50, 60):
                        ax.axvline(mains, color=PAPER["rule"], lw=0.6, ls=":", zorder=1)
                ax.plot(f[sel], np.asarray(A["init"]["H_lo"])[sel], color=PAPER["grey"],
                        lw=0.9, ls="--", zorder=3)
                ax.plot(f[sel], np.asarray(A["init"]["H_hi"])[sel], color=PAPER["grey"],
                        lw=0.9, ls="--", zorder=3)
                ax.plot(f[sel], np.asarray(L["H_hi_mean"])[sel], color=MOD_DARK[m], lw=1.0,
                        zorder=4)
                ax.plot(f[sel], np.asarray(L["H_lo_mean"])[sel], color=MOD_COLOR[m], lw=1.5,
                        zorder=5)
                fc0 = A["init"]["cutoff_lo_hz"]
                fc = L["cutoff_lo_hz_of_mean_response"]
                if np.isfinite(fc0):
                    ax.plot([fc0], [inv2], marker="o", ms=3.6, mfc="white",
                            mec=PAPER["rule"], mew=0.8, ls="", zorder=6)
                if np.isfinite(fc):
                    ax.plot([fc], [inv2], marker="o", ms=4.2, color=MOD_COLOR[m],
                            mec="white", mew=0.6, ls="", zorder=7)
                ax.set_xscale("log")
                ax.set_xlim(0.5, fs / 2)
                ax.set_ylim(0, 1.12)
                ax.set_title(_route_title(route), pad=11,
                             fontsize=PAPER_FONT)
                ax.set_yticks([0, 0.5, 1])
                if c_ == 0:
                    ax.set_ylabel(f"{LABEL[m]}  |H(f)|")
                else:
                    ax.tick_params(axis="y", labelleft=False)
                ticks = [t for t in (1, 10, 100, 1000) if 0.5 < t < fs / 2]
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)

        # The legend fills the empty cells of the row with the fewest routes.
        spare = max(range(len(mods)), key=lambda i: n_cols - len(R[mods[i]]["route_order"]))
        n_used = len(R[mods[spare]]["route_order"])
        tup = lambda cols, **kw: tuple(Line2D([], [], color=c, **kw) for c in cols)
        handles = [tup([MOD_COLOR[m] for m in mods], lw=1.5),
                   tup([MOD_DARK[m] for m in mods], lw=1.0),
                   Line2D([], [], color=PAPER["grey"], lw=0.9, ls="--"),
                   tup([MOD_COLOR[m] for m in mods], marker="o", ms=4.2, ls="",
                       mec="white", mew=0.6),
                   Line2D([], [], marker="o", ms=3.6, mfc="white", mec=PAPER["rule"],
                          mew=0.8, ls=""),
                   Patch(color=PAPER["pale_blue"], alpha=0.5, lw=0),
                   Line2D([], [], color=PAPER["rule"], lw=0.6, ls=":")]
        labels = ["Learned low-pass", "Learned high-pass", "Initialisation (uniform mix)",
                  "Learned −3 dB cut-off", "Initial −3 dB cut-off", "Physiological bands",
                  "Mains, 50 and 60 Hz"]
        if n_used < n_cols:
            axl = fig.add_subplot(ga[spare, n_used:])
            axl.set_axis_off()
            axl.legend(handles, labels, loc="center left", frameon=False,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
                       handlelength=2.4, handletextpad=0.6, labelspacing=0.45,
                       borderaxespad=0.0, bbox_to_anchor=(0.06, 0.5))
        box_a = outer[0].get_position(fig)
        fig.text(0.5 * (box_a.x0 + box_a.x1), 0.012, "Frequency (Hz)", ha="center",
                 va="bottom", gid="paper-keep")
        _titled_letter(fig, 0.008, 0.968, "a", "Effective analysis filters, level 1")

        # -- (b) selector weights, levels 1-3 per route ---------------------- #
        axb = fig.add_subplot(gr[0])
        bar_h, pitch, gap_route, gap_mod = 0.27, 0.34, 0.55, 0.35
        y = 0.0
        yt, yl, ycol = [], [], []
        for gi, m in enumerate(mods):
            names = R[m]["wavelet_names"]
            M = len(names)
            y0 = y
            left = 0.0
            for i, nm in enumerate(names):
                axb.barh(y, 1.0 / M, left=left, height=0.62, color=BASIS_COLORS[i], lw=0.4,
                         edgecolor="white", zorder=2)
                axb.text(left + 0.5 / M, y, nm, ha="center", va="center",
                         fontsize=PAPER_FONT - 1.5, zorder=5,
                         color=PAPER["ink"] if i < 2 else "white")
                left += 1.0 / M
            yt.append(y); yl.append("uniform init"); ycol.append(PAPER["rule"])
            y += 0.31 + gap_route
            for k, rid in enumerate(R[m]["route_order"]):
                A = R[m]["routes"][rid]
                ys = [y + j * pitch for j in range(3)]
                for j, lvl in enumerate((1, 2, 3)):
                    Lv = A["levels"].get(lvl)
                    if Lv is None:
                        continue
                    wv = np.asarray(Lv["weights"], dtype=float)
                    left = 0.0
                    for i in range(M):
                        axb.barh(ys[j], wv[i], left=left, height=bar_h, color=BASIS_COLORS[i],
                                 lw=0.3, edgecolor="white", zorder=2)
                        left += wv[i]
                yt.append(ys[1]); yl.append(_route_title(A["route"])); ycol.append(PAPER["ink"])
                y = ys[-1] + bar_h / 2 + gap_route
            _modality_tag(axb, m, y0, y - gap_route, x=1.03)
            y += gap_mod
        axb.set_yticks(yt)
        axb.set_yticklabels(yl)
        for lab, col in zip(axb.get_yticklabels(), ycol):
            lab.set_color(col)
            lab.set_fontsize(PAPER_FONT - 1)
        axb.set_ylim(y - gap_mod - gap_route + 0.2, -0.5)
        axb.set_xlim(0, 1)
        axb.set_xticks([0, 0.5, 1])
        axb.set_xticklabels(["0", "0.5", "1"])
        axb.set_xlabel("Mean selector weight", labelpad=2)
        axb.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            axb.spines[sp].set_visible(False)
        box_b = gr[0].get_position(fig)
        _titled_letter(fig, box_b.x0 - 0.135, 0.968, "b", "Selector weights, levels 1–3 top to bottom")

        # -- (c) low-pass cut-off: initialisation -> learned ----------------- #
        axc = fig.add_subplot(gr[1])
        yc = 0.0
        yt, yl = [], []
        all_x = []
        for m in mods:
            y0 = yc
            for rid in R[m]["route_order"]:
                A = R[m]["routes"][rid]
                fc0 = A["init"]["cutoff_lo_hz"]
                fc1 = A["levels"][1]["cutoff_lo_hz_of_mean_response"]
                if np.isfinite(fc0) and np.isfinite(fc1) and abs(math.log(fc1 / fc0)) > 0.12:
                    axc.annotate("", xy=(fc1, yc), xytext=(fc0, yc), zorder=3,
                                 arrowprops={"arrowstyle": "-|>", "color": MOD_COLOR[m],
                                             "lw": 0.9, "shrinkA": 3.0, "shrinkB": 3.4,
                                             "mutation_scale": 6})
                axc.plot([fc0], [yc], marker="o", ms=4.2, mfc="white", mec=PAPER["rule"],
                         mew=0.8, ls="", zorder=4)
                axc.plot([fc1], [yc], marker="o", ms=4.8, color=MOD_COLOR[m], mec="white",
                         mew=0.6, ls="", zorder=5)
                all_x += [fc0, fc1]
                yt.append(yc); yl.append(_route_title(A["route"]))
                yc += 1.0
            # the modality's physiological span behind its rows
            lo_b = min(b[1] for b in BANDS[m]); hi_b = max(b[2] for b in BANDS[m])
            axc.fill_betweenx([y0 - 0.45, yc - 0.55], lo_b, hi_b, color=PAPER["pale_blue"],
                              alpha=0.40, lw=0, zorder=0)
            _modality_tag(axc, m, y0, yc - 1.0, x=1.03)
            yc += 0.5
        axc.set_yticks(yt)
        axc.set_yticklabels(yl)
        for lab in axc.get_yticklabels():
            lab.set_fontsize(PAPER_FONT - 1)
        axc.set_ylim(yc - 0.5 - 0.4, -0.6)
        axc.tick_params(axis="y", length=0)
        axc.set_xscale("log")
        xs = [v for v in all_x if np.isfinite(v) and v > 0]
        axc.set_xlim(min(xs) / 1.5, max(xs) * 1.6)
        xt = [t for t in (10, 20, 50, 100, 200, 500) if min(xs) / 1.5 <= t <= max(xs) * 1.6]
        axc.set_xticks(xt)
        axc.set_xticklabels([str(t) for t in xt])
        axc.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axc.set_xlabel("Low-pass −3 dB cut-off (Hz)", labelpad=2)
        axc.grid(axis="x", color=PAPER["light"], lw=0.6, zorder=-1)
        for sp in ("top", "right", "left"):
            axc.spines[sp].set_visible(False)
        box_c = gr[1].get_position(fig)
        _titled_letter(fig, box_b.x0 - 0.135, box_c.y1 + 0.028, "c",
                       "Low-pass cut-off, initial → learned")

        data, meta = {}, {"modalities": {}}
        for m in mods:
            for rid, A in R[m]["routes"].items():
                data[f"{rid}_f_hz"] = A["f_hz"]
                data[f"{rid}_init_H_lo"] = A["init"]["H_lo"]
                data[f"{rid}_init_H_hi"] = A["init"]["H_hi"]
                for lvl, L in A["levels"].items():
                    for k in ("H_lo_mean", "H_lo_p10", "H_lo_p90", "H_hi_mean", "H_hi_p10",
                              "H_hi_p90"):
                        if k in L:
                            data[f"{rid}_L{lvl}_{k}"] = L[k]
            meta["modalities"][m] = {
                "wavelet_names": R[m]["wavelet_names"], "load": R[m]["load"],
                "corpora": R[m]["corpora"],
                "routes": {rid: {"init_cutoff_lo_hz": A["init"]["cutoff_lo_hz"],
                                 "init_cutoff_hi_hz": A["init"]["cutoff_hi_hz"],
                                 "tap_rms_change_rel": A["tap_rms_change_rel"],
                                 "per_basis_cutoff_lo_hz_init": A["per_basis_cutoff_lo_hz_init"],
                                 "per_basis_cutoff_lo_hz_learned":
                                     A["per_basis_cutoff_lo_hz_learned"],
                                 "levels": {str(l): {k: v for k, v in L.items()
                                                     if not isinstance(v, np.ndarray)}
                                            for l, L in A["levels"].items()}}
                           for rid, A in R[m]["routes"].items()}}
        w.save(fig, out_name, data, meta)
        return meta


def _route_stub(rid: str):
    """Channels and rate from a route id such as ``E19_256``."""
    import re
    from types import SimpleNamespace
    mt = re.fullmatch(r"[A-Z](\d+)_(\d+)", rid)
    if not mt:
        raise ValueError(f"cannot parse route id {rid!r}")
    return SimpleNamespace(route_id=rid, n_channels=int(mt.group(1)),
                           sampling_rate=int(mt.group(2)))


def load_saved(src: str, name: str = "fig_wavelet"):
    """``R`` as ``main`` builds it, from ``<src>/figure_data/<name>.npz`` and
    ``<src>/figure_metadata/<name>.json`` of an earlier run."""
    with open(os.path.join(src, "figure_metadata", f"{name}.json")) as fh:
        meta = json.load(fh)
    data = np.load(os.path.join(src, "figure_data", f"{name}.npz"))
    R = {}
    for m, Mm in meta["modalities"].items():
        routes = {}
        for rid, Rm in Mm["routes"].items():
            A = {"route": _route_stub(rid), "f_hz": data[f"{rid}_f_hz"],
                 "init": {"H_lo": data[f"{rid}_init_H_lo"], "H_hi": data[f"{rid}_init_H_hi"],
                          "cutoff_lo_hz": Rm["init_cutoff_lo_hz"],
                          "cutoff_hi_hz": Rm["init_cutoff_hi_hz"]},
                 "tap_rms_change_rel": Rm["tap_rms_change_rel"],
                 "per_basis_cutoff_lo_hz_init": Rm["per_basis_cutoff_lo_hz_init"],
                 "per_basis_cutoff_lo_hz_learned": Rm["per_basis_cutoff_lo_hz_learned"],
                 "levels": {}}
            for lvl, Lv in Rm["levels"].items():
                Lv = dict(Lv)
                for k in ("H_lo_mean", "H_lo_p10", "H_lo_p90", "H_hi_mean", "H_hi_p10",
                          "H_hi_p90"):
                    key = f"{rid}_L{lvl}_{k}"
                    if key in data:
                        Lv[k] = data[key]
                A["levels"][int(lvl)] = Lv
            routes[rid] = A
        R[m] = {"routes": routes, "route_order": list(Mm["routes"]),
                "corpora": Mm["corpora"], "wavelet_names": Mm["wavelet_names"],
                "load": Mm["load"]}
    return R, meta


# --------------------------------------------------------------------------- #
# The markdown report
# --------------------------------------------------------------------------- #

def write_report(path: str, R: Dict, n_windows: int, n_deep: int, split: str):
    mods = [m for m in MODALITIES if m in R]
    L = []
    L.append("# The learned wavelet front ends\n")
    L.append(f"Selector weights and effective analysis filters of the three C1 encoders, "
             f"from up to {n_windows} {split} windows per corpus (levels 2-3 on the first "
             f"{n_deep}). Figure: `figures/fig_wavelet.pdf` "
             f"(`scripts/visualize_wavelet_frontend.py`).\n")
    L.append("## Checkpoints and tensor matching\n")
    L.append("| modality | checkpoint | epoch | route | front-end tensors matched | selector |")
    L.append("|---|---|---|---|---|---|")
    for m in mods:
        ld = R[m]["load"]
        for rid, v in ld["routes"].items():
            L.append(f"| {LABEL[m]} | `{os.path.basename(os.path.dirname(ld['checkpoint']))}/"
                     f"{os.path.basename(ld['checkpoint'])}` | {ld['epoch']} | {rid} | "
                     f"{v['frontend_matched']}/{v['frontend_tensors']} | "
                     f"{v['selector_matched']}/{v['selector_tensors']} |")
    L.append("")
    L.append("## Windows used per corpus\n")
    L.append("| modality | corpus | route | windows (level 1) | windows (levels 2-3) |")
    L.append("|---|---|---|---|---|")
    for m in mods:
        for c in R[m]["corpora"]:
            L.append(f"| {LABEL[m]} | {CORPUS_LABEL.get(c['id'], c['id'])} | {c['route']} | "
                     f"{c['n_level1']} | {c['n_deep']} |")
    L.append("")
    L.append("## Mean selector weights, level 1 (corpus × basis)\n")
    flags = []
    for m in mods:
        names = R[m]["wavelet_names"]
        L.append(f"**{LABEL[m]}** (bank: {', '.join(names)})\n")
        L.append("| corpus | " + " | ".join(names) + " | max |w − 1/M| |")
        L.append("|---|" + "---|" * len(names) + "---|")
        for c in R[m]["corpora"]:
            wv = np.asarray(c["weights_level1"])
            dev = float(np.abs(wv - 1.0 / len(names)).max())
            L.append(f"| {CORPUS_LABEL.get(c['id'], c['id'])} | "
                     + " | ".join(f"{v:.3f}" for v in wv) + f" | {dev:.3f} |")
            if dev < 0.03:
                flags.append(f"{LABEL[m]} / {CORPUS_LABEL.get(c['id'], c['id'])}: level-1 "
                             f"selector weights within 0.03 of uniform (max deviation {dev:.3f}).")
        L.append("")
        L.append("Levels 2 and 3, per route (mean over the route's corpora):\n")
        L.append("| route | level | " + " | ".join(names) + " |")
        L.append("|---|---|" + "---|" * len(names))
        for rid, A in R[m]["routes"].items():
            for lvl in (2, 3):
                if lvl in A["levels"]:
                    L.append(f"| {rid} | {lvl} | "
                             + " | ".join(f"{v:.3f}" for v in A["levels"][lvl]["weights"]) + " |")
        L.append("")
    L.append("## Is the selector input-dependent?\n")
    L.append("The selector's input is the per-channel mean of the window (global "
             "average pooling over time). Windows are normalised per channel, so that "
             "mean is near zero and the softmax can only vary as far as the residual "
             "means do. Below, the across-window standard deviation of the level-1 "
             "weights and the mean |window mean| the selector actually saw.\n")
    L.append("| modality | corpus | std of weights across windows (per basis) | mean |window mean| |")
    L.append("|---|---|---|---|")
    for m in mods:
        for c in R[m]["corpora"]:
            sd = c.get("std_level1_across_windows")
            if sd is None:
                continue
            L.append(f"| {LABEL[m]} | {CORPUS_LABEL.get(c['id'], c['id'])} | "
                     + ", ".join(f"{v:.4f}" for v in sd)
                     + f" | {c.get('mean_abs_window_mean', float('nan')):.4f} |")
            if max(sd) < 1e-3:
                flags.append(f"{LABEL[m]} / {CORPUS_LABEL.get(c['id'], c['id'])}: the level-1 "
                             f"selector weights do not vary across windows (std < 1e-3): on "
                             f"normalised windows the selector's input is ~0 and its output "
                             f"is a learned per-route constant, not an input-adaptive choice.")
            wv = np.asarray(c["weights_level1"])
            if wv.max() > 0.8:
                flags.append(f"{LABEL[m]} / {CORPUS_LABEL.get(c['id'], c['id'])}: the selector "
                             f"has collapsed onto one basis "
                             f"({R[m]['wavelet_names'][int(wv.argmax())]}, {wv.max():.3f}).")
    L.append("")
    L.append("## −3 dB cut-offs (Hz)\n")
    L.append("Low-pass: the learned effective filter (bank mixed with the level's mean "
             "weights), cut-off of the channel-mean response, with the 10-90th percentile "
             "across channels; initialisation is the bank mixed uniformly. NOTE: each "
             "orthogonal wavelet's own low-pass is half-band (−3 dB at fs/4), but the "
             "uniform mix of five different wavelets is not -- their phases do not line "
             "up and the sum cancels at mid frequencies -- so the initialisation's "
             "cut-off sits well below fs/4. The per-basis table that follows separates "
             "a change of mixture from a change of the filters themselves. High-pass: "
             "the rising −3 dB point.\n")
    L.append("| route | fs | init LP | learned LP L1 (p10–p90) | L2 | L3 | init HP | learned HP L1 | "
             "tap RMS change / init RMS (per basis) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in mods:
        for rid, A in R[m]["routes"].items():
            fs = float(A["route"].sampling_rate)
            l1 = A["levels"][1]
            l2 = A["levels"].get(2, {}).get("cutoff_lo_hz_of_mean_response", float("nan"))
            l3 = A["levels"].get(3, {}).get("cutoff_lo_hz_of_mean_response", float("nan"))
            L.append(f"| {rid} | {fs:g} | {A['init']['cutoff_lo_hz']:.1f} | "
                     f"{l1['cutoff_lo_hz_of_mean_response']:.1f} "
                     f"({l1['cutoff_lo_hz_p10']:.1f}–{l1['cutoff_lo_hz_p90']:.1f}) | "
                     f"{l2:.1f} | {l3:.1f} | {A['init']['cutoff_hi_hz']:.1f} | "
                     f"{l1['cutoff_hi_hz_of_mean_response']:.1f} | "
                     + ", ".join(f"{v:.2f}" for v in A["tap_rms_change_rel"]) + " |")
            rel = abs(l1["cutoff_lo_hz_of_mean_response"] - A["init"]["cutoff_lo_hz"]) \
                / max(A["init"]["cutoff_lo_hz"], 1e-9)
            if rel < 0.05 and max(A["tap_rms_change_rel"]) < 0.05:
                flags.append(f"{rid}: learned low-pass cut-off within 5% of initialisation "
                             f"and taps moved < 5% RMS -- the filters barely moved.")
    L.append("")
    L.append("Per basis, the low-pass cut-off of each wavelet's own filter (channel "
             "mean), initial → learned:\n")
    for m in mods:
        names = R[m]["wavelet_names"]
        L.append(f"**{LABEL[m]}**\n")
        L.append("| route | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for rid, A in R[m]["routes"].items():
            L.append(f"| {rid} | " + " | ".join(
                f"{i0:.1f} → {l1:.1f}" for i0, l1 in zip(A["per_basis_cutoff_lo_hz_init"],
                                                          A["per_basis_cutoff_lo_hz_learned"]))
                     + " |")
        L.append("")
    # Do the cut-offs still scale with the sampling rate within a modality?
    for m in mods:
        pts = [(float(A["route"].sampling_rate), A["levels"][1]["cutoff_lo_hz_of_mean_response"])
               for A in R[m]["routes"].values()]
        rates = sorted({p[0] for p in pts})
        if len(rates) >= 2:
            by_rate = {r: np.mean([c for f, c in pts if f == r]) for r in rates}
            ratio_fs = rates[-1] / rates[0]
            ratio_fc = by_rate[rates[-1]] / by_rate[rates[0]]
            verdict = ("scale with the sampling rate (not a shared physical frequency)"
                       if ratio_fc > 0.5 * ratio_fs + 0.5 else
                       "are closer to a shared physical frequency than to fs/4")
            flags.append(f"{LABEL[m]}: fs ratio {ratio_fs:.2f}, learned cut-off ratio "
                         f"{ratio_fc:.2f} -- the cut-offs {verdict}.")
    L.append("## What does not show the expected pattern\n")
    L += [f"- {s}" for s in flags] or ["- (nothing flagged by the automatic checks)"]
    L.append("")
    L.append("The automatic checks above use fixed thresholds (uniform within 0.03; "
             "cut-off within 5% and taps within 5% RMS of initialisation; cut-off ratio "
             "against sampling-rate ratio). They flag, they do not interpret.\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eeg"); p.add_argument("--ecg"); p.add_argument("--emg")
    p.add_argument("--windows", type=int, default=2000)
    p.add_argument("--deep-windows", type=int, default=256,
                   help="windows per corpus run through the full decomposition for "
                        "the level-2/3 selector weights")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--report", default="docs/runs/fig_wavelet.md")
    p.add_argument("--format", default="pdf,png,svg")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-paper", action="store_true")
    p.add_argument("--from-saved", metavar="DIR",
                   help="redraw from DIR/figure_data/fig_wavelet.npz and "
                        "DIR/figure_metadata/fig_wavelet.json; no checkpoints are loaded "
                        "and the report is not rewritten")
    args = p.parse_args(argv)
    if args.from_saved:
        apply_style()
        R, old = load_saved(args.from_saved)
        base_meta = {"plotting_script_git_commit": git_commit(),
                     "checkpoints": old.get("checkpoints", {}),
                     "redrawn_from": args.from_saved}
        writer = FigureWriter(args.out_dir,
                              [s.strip() for s in args.format.split(",") if s.strip()],
                              base_meta, stamp=args.no_paper, paper=not args.no_paper)
        draw(writer, R)
        return 0
    if not (args.eeg or args.ecg or args.emg):
        p.error("give at least one of --eeg, --ecg, --emg")
    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    R: Dict[str, Dict] = {}
    base_meta = {"plotting_script_git_commit": git_commit(), "checkpoints": {}}
    # EEG first: load_trained_model switches the registry to ECG/EMG and never
    # back, so the EEG tables must be read before either of the others.
    for m in MODALITIES:
        path = getattr(args, m)
        if not path:
            continue
        print(f"== {LABEL[m]}: {path}")
        ck, cfg, mcfg, model, routes, specs, load_rep = load_checked(path, m)
        model.eval()
        base_meta["checkpoints"][m] = {"path": path, "sha256": sha256_file(path),
                                       "epoch": ck.get("epoch")}
        datasets = open_datasets(cfg, args.split, routes)
        corpora, per_route_w = [], {rid: {1: [], 2: [], 3: []} for rid in routes}
        for dsid in sorted(datasets):
            ds = datasets[dsid]
            if len(ds) == 0:
                continue
            route = routes[ds.route_id]
            means, counts, wstats = selector_weights(model, route, ds, args.windows,
                                                     args.deep_windows, device)
            print(f"  {dsid:14s} {route.route_id:9s} n={counts[1]:5d} level-1 weights "
                  + " ".join(f"{v:.3f}" for v in means[1]))
            corpora.append({"id": dsid, "route": route.route_id, "n_level1": counts[1],
                            "n_deep": counts[2], **wstats,
                            "weights_level1": means[1].tolist(),
                            "weights_level2": None if means[2] is None else means[2].tolist(),
                            "weights_level3": None if means[3] is None else means[3].tolist()})
            for lvl in (1, 2, 3):
                if means[lvl] is not None:
                    per_route_w[route.route_id][lvl].append((means[lvl], counts[lvl]))
            ds.close()
        analyses = {}
        for rid, route in routes.items():
            w_mean = {}
            for lvl in (1, 2, 3):
                items = per_route_w[rid][lvl]
                if items:
                    tot = sum(n for _, n in items)
                    w_mean[lvl] = sum(wv * n for wv, n in items) / tot
                else:
                    w_mean[lvl] = None
            if w_mean[1] is None:
                print(f"  {rid}: no windows in the {args.split} split; skipped")
                continue
            A = analyse_route(model, route, mcfg, w_mean)
            A["route"] = route
            analyses[rid] = A
            print(f"  {rid}: low-pass -3 dB {A['levels'][1]['cutoff_lo_hz_of_mean_response']:.1f} Hz "
                  f"(init {A['init']['cutoff_lo_hz']:.1f}), high-pass "
                  f"{A['levels'][1]['cutoff_hi_hz_of_mean_response']:.1f} Hz "
                  f"(init {A['init']['cutoff_hi_hz']:.1f})")
        R[m] = {"routes": analyses, "route_order": [r for r in routes if r in analyses],
                "corpora": corpora, "wavelet_names": list(mcfg.get("wavelet_names")),
                "load": load_rep}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer = FigureWriter(args.out_dir, [s.strip() for s in args.format.split(",") if s.strip()],
                          base_meta, stamp=args.no_paper, paper=not args.no_paper)
    draw(writer, R)
    write_report(args.report, R, args.windows, args.deep_windows, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
