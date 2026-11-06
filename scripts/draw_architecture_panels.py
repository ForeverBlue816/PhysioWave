#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-panel SVGs of one window's path through a trained C1 model, as
building blocks for an architecture diagram.

    python scripts/draw_architecture_panels.py --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe
    python scripts/draw_architecture_panels.py --run-dir ... --route E19_256 --channels 0,4,8
    python scripts/draw_architecture_panels.py --run-dir ... --axes      # with axes and labels

Every panel is drawn from the checkpoint run on one real validation window
under its real mask -- the same computation as the training step -- so the
pictures in the diagram are what the model does, not illustrations. By
default the window is the best-ranked one of the run's reconstruction survey
(figure_metadata/reconstruction_survey.json, written by
visualize_eeg_pretraining.py --recon-windows N) for that route.

Written to <run-dir>/architecture_panels/<route>/ (or --out-dir), one SVG
each, transparent background. The only text is the axes' tick labels (Times
New Roman, the paper figures' style); --bare drops the axes too.

  traces, a few channels stacked (the three closest to half masked):
    01_raw_input              the preprocessed window
    02_mask_grid              which patches are masked (those channels x patch)
    03_masked_input           masked patches zeroed: what the online view's
                              wavelet frontend receives; masked spans shaded
    04_wavelet_band_<k>_<b>   each scale of the learned decomposition
                              (d1..dJ detail bands, then the approximation)
    04_wavelet_bands_stacked  every scale of one channel, stacked
    05_folded_spec            the ScaleFold output: the scales folded back to
                              one row per channel
    08_raw_reconstruction     visible = target, masked = raw-head prediction
    09_raw_overlay            target vs raw-head prediction on masked spans
    10_spec_overlay           target vs spec-head prediction on masked spans

  heatmaps, every channel of the route (channel x time):
    H0_mask_grid_all          the whole mask, channel x patch
    H1_raw_target  H2_raw_masked  H3_raw_composite  H4_raw_error
    H5_spec_target H6_spec_masked H7_spec_composite H8_spec_error
                              target; target with masked patches blank;
                              target visible + prediction masked; |error| on
                              masked patches only (visible ones are not
                              supervised). A target and its composite share
                              one colour limit, taken from the target.

Colours: blue signal, orange prediction, purple mask; signal heatmaps on a
blue-white-red diverging map and errors on magma, as in the paper figures.
The window, mask seed and checkpoint are written to panels.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "ECG"), os.path.join(ROOT, "EMG")):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_eeg_pretraining as viz                      # noqa: E402
from visualize_eeg_pretraining import plt                    # noqa: E402

# --------------------------------------------------------------------------- #
# The window through the model
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_window(model, ds, route, mask_seed: int, window_index: int):
    """Everything a panel needs, on the [C, T] sample grid."""
    item = ds[window_index]
    batch = viz.collate_windows([item])
    meta = ds.montage()
    x = batch["x"]
    gen = viz._mask_generator(mask_seed, ds.dataset_id, window_index)
    out = model(x, route.route_id, channel_meta=meta,
                mask_ratio=model.mask_ratio, mask_generator=gen)
    C, P, pt = route.n_channels, route.patches_per_channel, route.patch_t
    clean_x = model._zero_padded_channels(x, meta)
    front = model.wavelet_frontends[route.route_id]
    bands = front.decomp(clean_x)[0].cpu().numpy()          # [(J+1)*C, T]
    J1 = bands.shape[0] // C
    mask_cp = out["mask"][0].cpu().numpy().reshape(C, P)

    def flat(key):
        return model.unpatchify(out[key], C, pt)[0].cpu().numpy()

    m = np.repeat(mask_cp, pt, axis=1)                       # [C, T]
    raw = clean_x[0].cpu().numpy()
    return {
        "raw": raw,
        "masked_raw": np.where(m, 0.0, raw),
        "mask_cp": mask_cp, "mask": m,
        "bands": bands.reshape(J1, C, -1),                   # [J+1, C, T]
        "folded": out["clean_spec"][0].cpu().numpy(),
        "target_spec": flat("target_spec"), "pred_spec": flat("pred_spec"),
        "target_raw": flat("target_raw"), "pred_raw": flat("pred_raw"),
        "subject_id": item["subject_id"], "recording_id": item["recording_id"],
        "fs": route.sampling_rate, "patch_t": pt,
    }


def pick_window(run_dir, route_id, datasets, dataset=None, window=None):
    """The survey's best window for the route, unless one is named."""
    if dataset and window is not None:
        return dataset, int(window)
    survey = os.path.join(run_dir, "figure_metadata", "reconstruction_survey.json")
    if os.path.isfile(survey):
        rows = [r for r in json.load(open(survey)).get("windows", [])
                if r.get("route_id") == route_id
                and (dataset is None or r.get("dataset_id") == dataset)
                and r.get("dataset_id") in datasets]
        if rows:
            best = min(rows, key=lambda r: (r.get("raw_nmse_rank", 99),
                                            r.get("spec_nmse_rank", 99)))
            return best["dataset_id"], int(best["window_index"])
    members = sorted(d for d in datasets
                     if datasets[d].route_id == route_id and len(datasets[d]))
    if dataset is None:
        if not members:
            raise SystemExit(f"no validation data for route {route_id}")
        dataset = members[0]
    return dataset, int(window or 0)


# --------------------------------------------------------------------------- #
# Style: purple / orange / blue, TPAMI-sized
# --------------------------------------------------------------------------- #

C_SIGNAL = "#2166AC"        # blue: the signal
C_PRED = "#E66101"          # orange: the model's prediction
C_MASK = "#5E3C99"          # purple: the mask
C_MASK_FILL = "#B2ABD2"     # light purple: masked spans behind traces
C_VISIBLE = "#EDEDED"       # a visible patch in the mask grid; blank cells
CMAP_SIGNAL = "RdBu_r"
CMAP_ERROR = "magma"
BAND_COLORS = ["#5E3C99", "#E66101", "#2166AC", "#4D4D4D"]   # d1 d2 d3 approx: the palette only


def tpami_style():
    viz.apply_style()                       # Times New Roman, bold, vector
    plt.rcParams.update({
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
    })


def _cmap(name):
    import copy
    cm = copy.copy(plt.get_cmap(name))
    cm.set_bad(C_VISIBLE)
    return cm


class Panels:
    def __init__(self, out_dir, size, bare: bool, lw: float):
        self.out_dir, self.size, self.bare, self.lw = out_dir, size, bare, lw
        os.makedirs(out_dir, exist_ok=True)
        self.written = []

    def new(self, h_scale=1.0):
        fig, ax = plt.subplots(figsize=(self.size[0], self.size[1] * h_scale))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        return fig, ax

    def finish(self, fig, ax, name, yticks=True):
        from matplotlib.ticker import MaxNLocator
        if self.bare:
            ax.set_axis_off()
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if yticks:
                ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            else:
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.margins(x=0)
        path = os.path.join(self.out_dir, f"{name}.svg")
        fig.savefig(path, format="svg", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        self.written.append(path)
        print(f"  {path}")


def _scale(r):
    r = np.asarray(r, float)
    v = np.nanpercentile(np.abs(r), 99.5) if np.isfinite(r).any() else 1.0
    return v if v > 0 else 1.0


def _stack(rows, ref=None, gap=1.2):
    """Rows scaled one by one (by ``ref``'s rows when given) and stacked.

    Per row, because these are diagram parts: the wavelet scales differ by an
    order of magnitude, and on one shared scale the fine bands draw flat. A
    compared pair -- target and prediction -- shares its row's scale via ref.
    """
    ref = rows if ref is None else ref
    out = [np.asarray(r, float) / _scale(q) for r, q in zip(rows, ref)]
    return out, [-i * 2 * gap for i in range(len(out))]


def _spans(mrow):
    d = np.diff(np.concatenate([[0], mrow.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _shade(ax, mrows, t, offs, gap=1.2):
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    for mrow, o in zip(mrows, offs):
        for a, b in _spans(mrow):
            ax.fill_between([t[a], t[b - 1] + dt], o - gap, o + gap,
                            color=C_MASK_FILL, alpha=0.35, lw=0)


def traces(pn, name, rows, t, color=C_SIGNAL, shade=None, ref=None,
           colors=None):
    fig, ax = pn.new()
    rows, offs = _stack(rows, ref)
    if shade is not None:
        _shade(ax, shade, t, offs)
    for i, (r, o) in enumerate(zip(rows, offs)):
        ax.plot(t, r + o, color=(colors[i % len(colors)] if colors else color),
                lw=pn.lw)
    pn.finish(fig, ax, name, yticks=False)


def overlay(pn, name, target, pred, mrows, t):
    """Target throughout, prediction only where it was masked."""
    fig, ax = pn.new()
    tg, offs = _stack(target)
    pr, _ = _stack(pred, ref=target)
    _shade(ax, mrows, t, offs)
    for r, q, o, mrow in zip(tg, pr, offs, mrows):
        ax.plot(t, r + o, color=C_SIGNAL, lw=pn.lw)
        ax.plot(t, np.where(mrow, q, np.nan) + o, color=C_PRED, lw=pn.lw)
    pn.finish(fig, ax, name, yticks=False)


def heatmap(pn, name, arr, seconds, lim=None, cmap=CMAP_SIGNAL, vmin=None,
            h_scale=1.0):
    fig, ax = pn.new(h_scale)
    C = arr.shape[0]
    if vmin is None:
        lim = lim or max(_scale(arr), 1e-9)
        lo, hi = -lim, lim
    else:
        lo, hi = vmin, (lim or max(_scale(arr), 1e-9))
    ax.imshow(arr, aspect="auto", cmap=_cmap(cmap), vmin=lo, vmax=hi,
              interpolation="nearest", extent=[0, seconds, C - 0.5, -0.5])
    pn.finish(fig, ax, name)


def mask_grid(pn, name, mask_cp, h_scale=None):
    from matplotlib.colors import ListedColormap
    C, P = mask_cp.shape
    fig, ax = pn.new(h_scale or max(0.25, min(1.0, C / 12)))
    ax.imshow(mask_cp.astype(float), aspect="auto", vmin=0, vmax=1,
              cmap=ListedColormap([C_VISIBLE, C_MASK]),
              interpolation="nearest")
    if C <= 32:                             # cell borders read as tokens
        for i in range(1, P):
            ax.axvline(i - 0.5, color="white", lw=0.5)
        for j in range(1, C):
            ax.axhline(j - 0.5, color="white", lw=0.5)
    pn.finish(fig, ax, name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--modality", default="auto", choices=["auto", "eeg", "ecg", "emg"])
    p.add_argument("--route", default=None, help="default: every route")
    p.add_argument("--dataset", default=None)
    p.add_argument("--window-index", type=int, default=None)
    p.add_argument("--channels", default=None,
                   help="comma-separated channel rows for the trace panels "
                        "(default: of the channels with typical amplitude, "
                        "the three closest to half masked)")
    p.add_argument("--mask-seed", type=int, default=None)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--size", default="2.4,1.5", help="panel size in inches, W,H")
    p.add_argument("--linewidth", type=float, default=0.8)
    p.add_argument("--bare", action="store_true",
                   help="no axes at all (default: axes with tick labels only)")
    args = p.parse_args(argv)

    tpami_style()
    run_dir = args.run_dir
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(run_dir, args.checkpoint))
    ck, cfg, mcfg, objective, modality, _cls, model = viz.load_trained_model(
        ckpt, args.modality)
    tpami_style()                            # the loader may reset rcParams
    mask_seed = (args.mask_seed if args.mask_seed is not None
                 else int(cfg.get("train", {}).get("val_mask_seed", 1234)))
    manifest = cfg.get("data", {}).get(f"manifest_{args.split}")
    if not manifest or not os.path.isfile(manifest):
        raise SystemExit(f"the run's {args.split} manifest is not here: {manifest!r}")
    index = viz.CorpusIndex.from_manifest(manifest)
    datasets = {}
    for d in sorted(index.by_dataset()):
        try:
            datasets[d] = viz.EEGWindowDataset(index, d, routes=viz.ROUTES)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skipping {d}: {exc}")

    size = tuple(float(v) for v in args.size.split(","))
    routes = [args.route] if args.route else list(viz.ROUTES)
    common = {"checkpoint": ckpt, "epoch": ck.get("epoch"),
              "mask_seed": mask_seed, "modality": modality}
    for rid in routes:
        route = viz.ROUTES[rid]
        if not any(ds.route_id == rid for ds in datasets.values()):
            print(f"  {rid}: no validation data, skipped")
            continue
        did, widx = pick_window(run_dir, rid, datasets, args.dataset,
                                args.window_index)
        ex = run_window(model, datasets[did], route, mask_seed, widx)
        C = route.n_channels
        if args.channels:
            chans = [int(c) for c in args.channels.split(",")]
        else:
            # Typical channels only: a row whose amplitude is far from the
            # route's median is a dead or drifting electrode (HBN window 92238
            # has one), and on a per-row scale the model's plausible EEG for
            # it is blown up to fill the panel. Among the rest, the three rows
            # closest to half masked, so each shows visible and masked patches.
            amp = np.array([_scale(ex["raw"][c]) for c in range(C)])
            med = float(np.median(amp[amp > 1e-6])) if (amp > 1e-6).any() else 1.0
            ok = [c for c in range(C) if med / 3 <= amp[c] <= med * 3] or list(range(C))
            frac = ex["mask_cp"].mean(1)
            chans = sorted(sorted(ok, key=lambda c: (abs(frac[c] - 0.5), c))
                           [:min(3, len(ok))])
        out_dir = os.path.join(args.out_dir or os.path.join(run_dir,
                                                            "architecture_panels"), rid)
        pn = Panels(out_dir, size, args.bare, args.linewidth)
        T = ex["raw"].shape[-1]
        secs = T / float(ex["fs"])
        t = np.arange(T) / float(ex["fs"])
        m = ex["mask"][chans]
        print(f"{rid}: {did} window {widx} ({ex['recording_id']}), channels {chans}")

        # -- traces ---------------------------------------------------------- #
        traces(pn, "01_raw_input", ex["raw"][chans], t)
        mask_grid(pn, "02_mask_grid", ex["mask_cp"][chans])
        traces(pn, "03_masked_input", ex["masked_raw"][chans], t, shade=m,
               ref=ex["raw"][chans])
        J1 = ex["bands"].shape[0]
        names = [f"d{j + 1}" for j in range(J1 - 1)] + ["approx"]
        for j, bname in enumerate(names):
            traces(pn, f"04_wavelet_band_{j + 1}_{bname}", ex["bands"][j][chans],
                   t, color=BAND_COLORS[j % len(BAND_COLORS)])
        traces(pn, "04_wavelet_bands_stacked", ex["bands"][:, chans[0]], t,
               colors=BAND_COLORS)
        traces(pn, "05_folded_spec", ex["folded"][chans], t, color=C_MASK)
        traces(pn, "08_raw_reconstruction",
               np.where(m, ex["pred_raw"][chans], ex["target_raw"][chans]), t,
               shade=m, ref=ex["target_raw"][chans])
        overlay(pn, "09_raw_overlay", ex["target_raw"][chans],
                ex["pred_raw"][chans], m, t)
        overlay(pn, "10_spec_overlay", ex["target_spec"][chans],
                ex["pred_spec"][chans], m, t)

        # -- heatmaps, every channel ------------------------------------------ #
        M = ex["mask"]
        hs = max(0.6, min(1.6, C / 24))
        mask_grid(pn, "H0_mask_grid_all", ex["mask_cp"], h_scale=hs)
        for tag, tk, pk in (("raw", "target_raw", "pred_raw"),
                            ("spec", "target_spec", "pred_spec")):
            tgt, prd = ex[tk], ex[pk]
            lim = max(_scale(tgt), 1e-9)
            n0 = 1 if tag == "raw" else 5
            heatmap(pn, f"H{n0}_{tag}_target", tgt, secs, lim, h_scale=hs)
            heatmap(pn, f"H{n0 + 1}_{tag}_masked", np.where(M, np.nan, tgt),
                    secs, lim, h_scale=hs)
            heatmap(pn, f"H{n0 + 2}_{tag}_composite", np.where(M, prd, tgt),
                    secs, lim, h_scale=hs)
            err = np.where(M, np.abs(prd - tgt), np.nan)
            heatmap(pn, f"H{n0 + 3}_{tag}_error", err, secs,
                    lim=max(_scale(err[np.isfinite(err)]) if np.isfinite(err).any()
                            else 1.0, 1e-9),
                    cmap=CMAP_ERROR, vmin=0.0, h_scale=hs)

        info = {"dataset": did, "window_index": widx,
                "recording_id": ex["recording_id"], "subject_id": ex["subject_id"],
                "channels": chans, "panels": [os.path.basename(x) for x in pn.written],
                **common}
        with open(os.path.join(out_dir, "panels.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)
    for ds in datasets.values():
        ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
