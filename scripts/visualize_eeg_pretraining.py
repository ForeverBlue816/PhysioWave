#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper figures for an EEG C1 pretraining run.

    python scripts/visualize_eeg_pretraining.py \
        --run-dir /path/to/run --checkpoint best.pth --split val --format svg

``--recon-windows N`` draws the two reconstruction figures on N different
validation windows per route instead of one, ranks them by masked NMSE and
writes the ranking to figure_metadata/reconstruction_survey.json. Choosing the
best of several is a normal way to pick a figure; the rank exists so the chosen
one can be captioned as chosen rather than as the window that came up.

Every figure is drawn from the run's own artefacts -- the checkpoint, the
metrics files and the corpus the run trained on. Nothing is illustrative: the
masks drawn are the masks the model was given, the reconstructions are that
checkpoint's outputs, and the wavelet responses are its learned filters.

WHAT THE MODEL RECONSTRUCTS. TWO things, at equal weight. One head predicts
FOLDED WAVELET PATCHES: Spec(x) is decomposed into J+1 scales and the dynamic
fold reduces the scale axis back to one row per electrode, and that folded
representation is what is patched, masked and predicted. The other head
predicts the PREPROCESSED EEG of the same masked patches -- a waveform, in the
z-scored units the preprocessing produced, and not an inverse wavelet
transform. Both appear in fig_mask_reconstruction; the raw head has
14_raw_waveform_reconstruction to itself.

THREE RULES THE RECONSTRUCTION FIGURES FOLLOW, because breaking any of them
flatters the model:

  * the spec target drawn is ``target_spec`` -- the per-patch NORMALISED
    tensor the loss is computed against -- not the unnormalised ``clean_spec``.
    Comparing a normalised prediction to an unnormalised target is a
    comparison of two different quantities;
  * a target and its composite share colour limits, taken from the target.
    Independent autoscaling makes a prediction with a third of the amplitude
    draw at the same contrast;
  * predictions are shown on MASKED patches only. No gradient ever reached a
    visible token's decoder output, so displaying one as reconstruction
    reports a number nothing trained.

Alongside every figure:
    figures/<name>.svg          the figure
    figure_data/<name>.npz      the arrays it was drawn from
    figure_metadata/<name>.json checkpoint hash, step, sample identity, seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from channel_embedding import CHANNEL_VOCAB                    # noqa: E402
from physiowave.eeg_c1.data import (CorpusIndex, EEGWindowDataset,  # noqa: E402
                                    collate_windows)
from physiowave.eeg_c1.model import (MultiRouteEEGPretrainer,  # noqa: E402
                                     masked_reconstruction_loss)
from physiowave.eeg_c1.objective import (objective_equation,  # noqa: E402
                                         resolve_eeg_c1_objective)
from physiowave.eeg_c1.routes import (PRETRAIN_DATASETS, ROUTES,  # noqa: E402
                                      Route)
from physiowave.eeg_c1.train import _mask_generator            # noqa: E402


# --------------------------------------------------------------------------- #
# House style
#
# Okabe-Ito: eight hues distinguishable under the common forms of colour vision
# deficiency, and still distinguishable printed in greyscale by lightness.
# Paper style: Times New Roman, bold, every piece of text.
# svg.fonttype="none" leaves text as text in the SVG, so the figure can be
# restyled by the journal's template instead of arriving as outlines;
# pdf.fonttype=42 embeds TrueType fonts in the PDF (editable in Illustrator,
# and what journals that reject Type 3 fonts ask for). savefig.dpi applies only
# to the raster parts of a vector figure -- the heatmaps.
# --------------------------------------------------------------------------- #
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
ROUTE_COLOR = {"E19_256": OKABE_ITO[0], "E32_512": OKABE_ITO[1],
               "E64_256": OKABE_ITO[2], "E128_512": OKABE_ITO[3]}
DATASET_COLOR = {d: OKABE_ITO[i % len(OKABE_ITO)]
                 for i, d in enumerate(PRETRAIN_DATASETS)}


#: Times New Roman first; the rest are Times-metric or Times-styled faces, in
#: the order a Linux node is likeliest to have them. STIXGeneral ships with
#: matplotlib itself, so the figures are never drawn in a sans-serif fallback.
SERIF_FONTS = ["Times New Roman", "Times", "Liberation Serif",
               "Nimbus Roman", "Nimbus Roman No9 L", "STIXGeneral"]


def _load_user_fonts() -> None:
    """Register any .ttf/.otf under $PW_FONT_DIR (default ~/.fonts).

    Times New Roman is not free to redistribute, so it is not in the
    repository; a cluster node rarely has it installed. Copying the four
    files (regular, bold, italic, bold italic) from your own machine into
    ~/.fonts is enough for every figure to use it.
    """
    from matplotlib import font_manager
    d = os.path.expanduser(os.environ.get("PW_FONT_DIR", "~/.fonts"))
    if not os.path.isdir(d):
        return
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(root, fn))
                except Exception:                              # noqa: BLE001
                    pass


def apply_style():
    """Paper style: Times New Roman (or the closest face present), bold."""
    _load_user_fonts()
    plt.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": SERIF_FONTS,
        "mathtext.fontset": "stix",
        "mathtext.default": "bf",          # tick labels like 10^-1, bold too
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })
    from matplotlib import font_manager
    used = font_manager.FontProperties(family="serif", weight="bold")
    name = font_manager.FontProperties(
        fname=font_manager.findfont(used)).get_name()
    print(f"  font: {name} (bold)"
          + ("" if name == "Times New Roman" else
             "  -- Times New Roman is not installed here; copy its .ttf "
             "files into ~/.fonts to use it"))


# --------------------------------------------------------------------------- #
# Output plumbing
# --------------------------------------------------------------------------- #

class FigureWriter:
    """Writes the figure, the arrays behind it, and what produced both."""

    def __init__(self, out_dir: str, fmt, base_meta: Dict, stamp: bool = True,
                 paper: bool = False):
        self.fig_dir = os.path.join(out_dir, "figures")
        self.data_dir = os.path.join(out_dir, "figure_data")
        self.meta_dir = os.path.join(out_dir, "figure_metadata")
        for d in (self.fig_dir, self.data_dir, self.meta_dir):
            os.makedirs(d, exist_ok=True)
        #: One or more of svg / pdf / png; every figure is written in each.
        self.formats = [fmt] if isinstance(fmt, str) else list(fmt)
        self.base_meta = base_meta
        self.stamp = stamp and not paper
        #: Paper mode: the figure-level title and notes -- sentences that
        #: belong in a caption -- are taken off the figure and kept in its
        #: metadata as ``figure_text``, from which the caption can be written.
        self.paper = paper
        self.written: List[str] = []

    #: Stamped onto every figure, not only into its metadata.
    #:
    #: The metadata already recorded the checkpoint, its epoch and the
    #: generation time, and that was enough for a file and useless for a
    #: picture: a PNG from epoch 32 and one from epoch 59 look identical, and
    #: the way you find out is by noticing an axis that stops too early and
    #: doubting the run instead of the file. Copies leave this directory --
    #: scp'd, pasted into slides, mailed -- and the metadata does not go with
    #: them. The stamp does.
    def _stamp(self, fig):
        bits = [os.path.basename(str(self.base_meta.get("checkpoint", "?")))]
        ep = self.base_meta.get("epoch")
        if ep is not None:
            bits.append(f"ckpt epoch {ep}")
        last = self.base_meta.get("metrics_last_epoch")
        if last is not None:
            bits.append(f"metrics through epoch {last}")
        bits.append(datetime.now(timezone.utc).strftime("drawn %Y-%m-%d %H:%M UTC"))
        fig.text(0.005, 0.005, "  ".join(bits), fontsize=5.5, color="0.45",
                 ha="left", va="bottom")

    def _alignment_audit(self, fig, name: str):
        """Measure the panels' final rectangles, when the auditor is present.

        ``PW_FIG_ALIGN_AUDIT`` names a directory holding the figure-QA
        ``audit_panel_alignment.py``; the report lands beside the figure's
        metadata as ``<name>.alignment.json`` and a misalignment is printed,
        not raised, so a working figure is never lost to a 2 pt gutter. Axes
        labelled ``free`` (hand-placed insets such as topomaps) and colourbars
        are left out of the comparison groups. Unset, nothing happens.
        """
        d = os.environ.get("PW_FIG_ALIGN_AUDIT")
        if not d or len(fig.axes) < 2:
            return
        try:
            if d not in sys.path:
                sys.path.insert(0, d)
            from audit_panel_alignment import (PanelAlignmentError,
                                               require_matplotlib_panel_alignment)
        except ImportError as exc:
            print(f"  alignment audit skipped: {exc}")
            return
        free = [a for a in fig.axes if str(a.get_label()).startswith("free")]
        out = os.path.join(self.meta_dir, f"{name}.alignment.json")
        try:
            require_matplotlib_panel_alignment(fig, json_out=out, exclude_axes=free,
                                               tolerance_pt=1.5, strict=False)
            print(f"  alignment: PASS ({out})")
        except PanelAlignmentError as exc:
            print(f"  alignment: {str(exc).splitlines()[0]} ({out})")

    def save(self, fig, name: str, data: Optional[Dict] = None,
             meta: Optional[Dict] = None):
        figure_text = []
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text():
            figure_text.append(sup.get_text())
        figure_text += [t.get_text() for t in fig.texts if t.get_text()]
        if self.paper:
            if sup is not None:
                sup.set_visible(False)
            # Figure-level notes go; panel letters and group titles (gid
            # "paper-keep") are part of the figure and stay.
            for t in fig.texts:
                if t.get_gid() != "paper-keep":
                    t.set_visible(False)
        if self.stamp:
            self._stamp(fig)
        self._alignment_audit(fig, name)
        paths = [os.path.join(self.fig_dir, f"{name}.{f}") for f in self.formats]
        for path, f in zip(paths, self.formats):
            fig.savefig(path, format=f)
        plt.close(fig)
        if data:
            np.savez_compressed(os.path.join(self.data_dir, f"{name}.npz"),
                                **{k: np.asarray(v) for k, v in data.items()})
        payload = dict(self.base_meta)
        payload.update({"figure": name,
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "figure_text": figure_text,
                        "paper_mode": self.paper})
        if meta:
            payload.update(meta)
        with open(os.path.join(self.meta_dir, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        self.written.append(name)
        print(f"  wrote {', '.join(paths)}")


def sha256_file(path: str, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while read < limit:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# 1. datasets and routes
# --------------------------------------------------------------------------- #

def fig_dataset_routes(w: FigureWriter, manifest: Dict):
    datasets = list(PRETRAIN_DATASETS)
    counts = [manifest.get("datasets", {}).get(d, {}).get("n_windows", 0)
              for d in datasets]
    mix = manifest.get("realised_mixture", {})
    by_step = [mix.get("by_step", {}).get(d, 0.0) * 100 for d in datasets]
    by_window = [mix.get("by_window", {}).get(d, 0.0) * 100 for d in datasets]
    target = [manifest.get("target_weights", {}).get(d, 0.0) * 100 for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})

    ax = axes[0]
    y = np.arange(len(datasets))
    ax.barh(y, counts, color=[ROUTE_COLOR[PRETRAIN_DATASETS[d].route_id]
                              for d in datasets])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n{PRETRAIN_DATASETS[d].route_id}" for d in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("windows in the training manifest")
    ax.set_title("corpus size by dataset")
    if max(counts or [0]) > 0:
        ax.set_xscale("log")

    ax = axes[1]
    width = 0.38
    ax.barh(y - width / 2, target, height=width, color=OKABE_ITO[7],
            label="target")
    ax.barh(y + width / 2, by_step, height=width, color=OKABE_ITO[0],
            label="realised")
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()
    ax.set_xlabel("share of optimizer steps (%)")
    ax.set_title("mixture, by step")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.barh(y, by_window, color=[ROUTE_COLOR[PRETRAIN_DATASETS[d].route_id]
                                 for d in datasets])
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()
    ax.set_xlabel("share of windows (%)")
    ax.set_title("mixture, by window\n(differs: micro-batch is route-specific)")

    handles = [plt.Line2D([0], [0], color=c, lw=4,
                          label=f"{r} · {ROUTES[r].n_channels}ch @ "
                                f"{ROUTES[r].sampling_rate}Hz · "
                                f"{ROUTES[r].n_tokens} tokens")
               for r, c in ROUTE_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Seven corpora, four hard routes, four wavelet frontends", y=1.02)
    w.save(fig, "fig_dataset_routes",
           {"datasets": datasets, "n_windows": counts, "target_pct": target,
            "by_step_pct": by_step, "by_window_pct": by_window})


# --------------------------------------------------------------------------- #
# 2 & 3. convergence
# --------------------------------------------------------------------------- #

def _series(rows: List[Dict], key: str):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            xs.append(r.get("global_step", r.get("step", len(xs))))
            ys.append(v)
    return np.asarray(xs, float), np.asarray(ys, float)


#: ``(logging key, fallback keys, label)``. The fallbacks are the pre-dual
#: names, so a run from before the raw head still renders. New code names the
#: head it means: `loss_masked_mse` does not say whether it is spec or raw.
CONVERGENCE_PANELS = (
    (("loss_total",), "total loss"),
    (("loss_masked_spec_mse", "loss_masked_mse"), "masked spec MSE"),
    (("loss_masked_raw_smoothl1",), "masked raw SmoothL1"),
    (("masked_spec_corr", "masked_corr"), "masked spec corr"),
)


def _first_key(rows: List[Dict], keys) -> str:
    """The first of ``keys`` this run actually logged, under any prefix.

    metrics_step.jsonl holds bare names and metrics_epoch.jsonl holds
    ``train/`` and ``val/`` ones, so a probe for the bare name alone would miss
    a run whose step file is absent and fall back to a key that does not exist.
    """
    for k in keys:
        if any(k in r or f"train/{k}" in r or f"val/{k}" in r for r in rows):
            return k
    return keys[0]


def fig_pretraining_convergence(w: FigureWriter, epoch_rows, step_rows):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    data = {}
    for ax, (keys, label) in zip(axes, CONVERGENCE_PANELS):
        key = _first_key(step_rows + epoch_rows, keys)
        sx, sy = _series(step_rows, key)
        if sx.size:
            # Raw step values, not a smoothed curve: a window long enough to
            # hide the step-to-step spread would also hide whether the run is
            # actually converging or bouncing.
            ax.plot(sx, sy, color=OKABE_ITO[5], lw=0.5, alpha=0.35,
                    label="step")
            data[f"step_x_{key}"], data[f"step_y_{key}"] = sx, sy
        ex, ey = _series(epoch_rows, f"train/{key}")
        if ex.size:
            ax.plot(ex, ey, color=OKABE_ITO[0], marker="o", ms=2.5,
                    label="train (epoch)")
            data[f"train_x_{key}"], data[f"train_y_{key}"] = ex, ey
        vx, vy = _series(epoch_rows, f"val/{key}")
        if vx.size:
            ax.plot(vx, vy, color=OKABE_ITO[1], marker="s", ms=2.5,
                    label="val (epoch)")
            data[f"val_x_{key}"], data[f"val_y_{key}"] = vx, vy
        ax.set_title(label)
        ax.set_xlabel("global step")
        if key.endswith("mse") or key.endswith("smoothl1"):
            ax.set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    fig.suptitle("Masked dual reconstruction: convergence. The spec and raw "
                 "panels are different units and are not comparable to each "
                 "other.", y=1.03)
    w.save(fig, "fig_pretraining_convergence", data)


def fig_route_convergence(w: FigureWriter, epoch_rows):
    """Per route and per dataset, on the TOTAL loss -- what is being selected on.

    These curves are globally aggregated across ranks: the validation sweep is
    partitioned, so before that reduction each of these lines was one rank's
    slice of the dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    data = {}
    metric = _first_key(epoch_rows, ("loss_total", "loss_masked_mse"))
    for rid, color in ROUTE_COLOR.items():
        x, y = _series(epoch_rows, f"val/route/{rid}/{metric}")
        if x.size:
            axes[0].plot(x, y, color=color, marker="o", ms=2.5, label=rid)
            data[f"route_{rid}_x"], data[f"route_{rid}_y"] = x, y
    mx, my = _series(epoch_rows, "val/macro_route_loss_total")
    if mx.size:
        # The unweighted mean of the four lines above, which is what
        # best_macro_total.pth is selected on. It is NOT the global validation
        # loss: that one is 97% TUEG and HBN by batch count.
        axes[0].plot(mx, my, color="0.2", ls="--", lw=1.4, label="macro (mean)")
        data["macro_x"], data["macro_y"] = mx, my
    axes[0].set_title(f"validation {metric}, by route")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel(metric)
    axes[0].legend(frameon=False)

    for d in PRETRAIN_DATASETS:
        x, y = _series(epoch_rows, f"val/dataset/{d}/{metric}")
        if x.size:
            axes[1].plot(x, y, color=DATASET_COLOR[d], marker=".", ms=3,
                         label=d)
            data[f"dataset_{d}_x"], data[f"dataset_{d}_y"] = x, y
    axes[1].set_title(f"validation {metric}, by dataset (supplementary)")
    axes[1].set_xlabel("global step")
    axes[1].legend(frameon=False, ncol=2)
    w.save(fig, "fig_route_convergence", data)


# --------------------------------------------------------------------------- #
# 4 & 5. real masks and real reconstructions
# --------------------------------------------------------------------------- #

def _expand(mask_cp: np.ndarray, patch_t: int) -> np.ndarray:
    """``[C, P]`` patch mask to ``[C, P*patch_t]``, the sample grid."""
    return np.repeat(mask_cp, patch_t, axis=1)


def _composite(target: np.ndarray, pred: np.ndarray,
               m: np.ndarray) -> np.ndarray:
    """Target where the model could SEE, prediction where it could not.

    Showing the decoder's output on visible patches as though it were
    reconstruction is the thing this function exists to prevent: nothing in the
    loss ever looks at a visible token, so those values are unsupervised, and a
    figure that displays them is reporting a number no gradient produced.
    """
    return np.where(m, pred, target)


def _hide(target: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The target with the masked patches removed -- NaN, drawn as blank."""
    out = target.astype(float).copy()
    out[m] = np.nan
    return out


def _corrupt(target: np.ndarray, m: np.ndarray) -> np.ndarray:
    """What the frontend was actually handed: masked patches zeroed.

    Zero rather than NaN: this is the SIGNAL the model saw, and the fill is
    zero because the window is z-scored, so zero is its mean.
    """
    out = target.astype(float).copy()
    out[m] = 0.0
    return out


def _masked_error(pred: np.ndarray, target: np.ndarray, m: np.ndarray,
                  valid_rows: np.ndarray) -> np.ndarray:
    """``|pred - target|`` on masked valid samples, NaN everywhere else."""
    e = np.abs(pred.astype(float) - target.astype(float))
    e[~m] = np.nan
    e[~valid_rows] = np.nan
    return e


def _sym_limits(target: np.ndarray, valid_rows: np.ndarray,
                q: float = 99.5) -> float:
    """A robust symmetric limit from the TARGET, shared by every panel of a pair.

    Autoscaling the target and the prediction independently is what makes a
    flat prediction look like a good one: matplotlib stretches whatever range
    it is given, so two panels with different limits are not comparable however
    similar they look. One limit, taken from the target, and a prediction that
    is too small stays visibly too small.
    """
    vals = target[valid_rows]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(vals), q))
    return v if v > 0 else 1.0


@torch.no_grad()
def _fixed_example(model, ds: EEGWindowDataset, route: Route, mask_seed: int,
                   window_index: int = 0, device="cpu",
                   objective: Optional[Dict] = None):
    """One fixed validation window under its fixed mask, through the model.

    Everything is returned on the ``[C, T]`` sample grid, unpatchified with the
    model's own inverse, so a panel can be indexed by channel and time without
    the caller reconstructing the channel-major token order.
    """
    item = ds[window_index]
    batch = collate_windows([item])
    meta = {k: v.to(device) for k, v in ds.montage().items()}
    gen = _mask_generator(mask_seed, ds.dataset_id, window_index)
    out = model(batch["x"].to(device), route.route_id, channel_meta=meta,
                mask_ratio=model.mask_ratio, mask_generator=gen)
    o = objective or {}
    _, metrics = masked_reconstruction_loss(
        out,
        spec_weight=o.get("spec_weight"), raw_weight=o.get("raw_weight"),
        fold_kl=o.get("fold_kl"), raw_beta=o.get("raw_beta"))
    C, P, pt = route.n_channels, route.patches_per_channel, route.patch_t

    def flat(key):
        return model.unpatchify(out[key], C, pt)[0].detach().cpu().numpy()

    mask = out["mask"][0].detach().cpu().numpy().reshape(C, P)
    valid = (out["valid_tokens"][0].detach().cpu().numpy().reshape(C, P)
             if out["valid_tokens"] is not None else np.ones((C, P), bool))
    return {
        "raw": batch["x"][0].cpu().numpy(),
        # The UNNORMALISED frontend output. Kept for the wavelet-analysis
        # figures; it is NOT the reconstruction target when
        # normalize_spec_target is on, and nothing below labels it as one.
        "clean_spec": out["clean_spec"][0].detach().cpu().numpy(),
        "spec": out["clean_spec"][0].detach().cpu().numpy(),
        # What the loss actually compares, on the sample grid.
        "target_spec": flat("target_spec"),
        "pred_spec": flat("pred_spec"),
        "target_raw": flat("target_raw"),
        "pred_raw": flat("pred_raw"),
        "normalize_spec_target": bool(out.get("normalize_spec_target", True)),
        # -- compatibility aliases: the older figures asked for these -------- #
        "target": flat("target_spec"),
        "pred": flat("pred_spec"),
        "mask": mask,
        "valid": valid,
        "metrics": metrics,
        "subject_id": item["subject_id"],
        "recording_id": item["recording_id"],
        "window_index": window_index,
    }


def _route_rows(datasets) -> List:
    """One deterministic ``(route, dataset)`` pair per route, in route order.

    ``sorted`` rather than "the first one the iteration happens to yield": a
    figure whose sample moves because a manifest was rebuilt in a different
    order is not the fixed sample it claims to be.
    """
    rows = []
    for rid in ROUTES:
        members = sorted(d for d in datasets
                         if datasets[d].route_id == rid and len(datasets[d]))
        if members:
            rows.append((rid, members[0]))
    return rows


def _survey_indices(ds, n: int) -> List[int]:
    """``n`` window indices spread across the dataset, deterministically.

    NOT ``range(n)``. The corpus is one shard per recording and windows sit
    consecutively inside it, so the first n indices are n consecutive windows
    of ONE recording -- often one minute of one subject. A survey drawn from
    those varies the mask and almost nothing else, and picking "the best of
    eight" out of it selects over noise. linspace spans the dataset, so the
    candidates come from different recordings and the spread between them is a
    spread over the corpus.
    """
    n_total = len(ds)
    if n_total <= 0:
        return []
    if n <= 1 or n_total == 1:
        return [0]
    return sorted({int(round(float(i)))
                   for i in np.linspace(0, n_total - 1, min(n, n_total))})


def survey_reconstruction_windows(model, datasets, mask_seed, device,
                                  objective, n_windows: int = 1):
    """Every window the two reconstruction figures draw, modelled once.

    Returns ``(rows, grid, ranks)``:

    * ``rows``      -- ``(route_id, dataset_id)``, one per route;
    * ``grid[k][i]``-- the example for ``rows[i]``'s k-th surveyed window;
    * ``ranks[(route_id, k)]`` -- ``{"raw": r, "spec": s, "n": K}``, where 1 is
      the LOWEST masked NMSE among the windows surveyed for that route.

    The rank is the point of surveying at all. Drawing eight windows and
    publishing the one that came out best is a legitimate way to choose a
    figure; publishing it as though it were simply the window that came up is
    not. The rank travels in the figure's caption and its metadata so the two
    stay distinguishable after the figure leaves this directory.

    One pass, shared by both figures. They show the same windows from two
    sides; computing them separately would run the model twice and let the two
    drift onto different samples the moment either index rule is edited.
    """
    rows = _route_rows(datasets)
    if not rows:
        return [], [], {}
    n_windows = max(1, int(n_windows))
    picks = {rid: _survey_indices(datasets[dsid], n_windows)
             for rid, dsid in rows}
    n_drawn = max((len(v) for v in picks.values()), default=0)

    grid: List[List[Optional[Dict]]] = []
    for k in range(n_drawn):
        col: List[Optional[Dict]] = []
        for rid, dsid in rows:
            if not picks[rid]:
                col.append(None)
                continue
            # A dataset with fewer windows than the survey repeats its last one
            # rather than dropping out of the later figures. A four-row figure
            # that silently becomes three rows reads as "this route failed".
            wi = picks[rid][min(k, len(picks[rid]) - 1)]
            col.append(_fixed_example(model, datasets[dsid], ROUTES[rid],
                                      mask_seed, window_index=wi,
                                      device=device, objective=objective))
        grid.append(col)

    return rows, grid, _rank_windows(rows, grid)


def _rank_windows(rows, grid) -> Dict:
    """``{(route_id, k): {"raw": r, "spec": s, "n": K}}``, 1 = lowest NMSE.

    Per route, because the routes are not comparable to each other: E19_256's
    NMSE and E128_512's are computed on different numbers of channels at
    different sampling rates, and a single pooled ranking would report "the
    best window" as "the window from the easiest route".

    The two heads rank independently. They are separately weighted halves of
    the objective and a window can be the best on one and the worst on the
    other; collapsing them into one score would hide exactly that.
    """
    n_drawn = len(grid)
    ranks: Dict = {}
    for i, (rid, _ds) in enumerate(rows):
        for tag, key in (("raw", "masked_raw_nmse"),
                         ("spec", "masked_spec_nmse")):
            scored = []
            for k in range(n_drawn):
                ex = grid[k][i]
                v = float(ex["metrics"][key]) if ex else float("nan")
                # NaN compares false against everything, so an unguarded sort
                # can leave a diverged window in first place -- and first place
                # is what someone reading the survey for a figure reaches for.
                scored.append((v if math.isfinite(v) else math.inf, k))
            scored.sort(key=lambda t: t[0])
            for pos, (_v, k) in enumerate(scored, start=1):
                ranks.setdefault((rid, k), {})[tag] = pos
    for key in ranks:
        ranks[key]["n"] = n_drawn
    return ranks


def write_reconstruction_survey(w: FigureWriter, rows, grid, ranks,
                                mask_seed: int) -> List[Dict]:
    """The table a figure gets picked from, printed and written next to them.

    Written, not only printed: "we showed window 431 of TUEG" is not a
    reproducible statement unless the windows it was chosen over are recorded
    too. The file is what a caption's "best of 8" can be checked against.
    """
    out: List[Dict] = []
    for k, col in enumerate(grid):
        for i, (rid, dsid) in enumerate(rows):
            ex = col[i]
            if ex is None:
                continue
            mt, rk = ex["metrics"], ranks.get((rid, k), {})
            out.append({
                "figure_index": k,
                "figure": _recon_name("fig_mask_reconstruction", k),
                "waveform_figure": _recon_name(
                    "14_raw_waveform_reconstruction", k),
                "route_id": rid, "dataset_id": dsid,
                "window_index": ex["window_index"],
                "subject_id": ex["subject_id"],
                "recording_id": ex["recording_id"],
                "mask_seed": mask_seed,
                "raw_corr": float(mt["masked_raw_corr"]),
                "raw_nmse": float(mt["masked_raw_nmse"]),
                "spec_corr": float(mt["masked_spec_corr"]),
                "spec_nmse": float(mt["masked_spec_nmse"]),
                "raw_nmse_rank": rk.get("raw"),
                "spec_nmse_rank": rk.get("spec"),
                "n_windows_surveyed": rk.get("n"),
            })
    if not out:
        return out

    n_drawn = len(grid)
    print(f"\n  reconstruction survey: {n_drawn} window(s) per route "
          f"(rank 1 = lowest masked NMSE for that route)")
    print(f"  {'fig':>3}  {'route':<10} {'dataset':<10} {'win':>7}  "
          f"{'raw r':>7} {'raw NMSE':>9} {'spec r':>7} {'spec NMSE':>10}  "
          f"{'rank raw/spec':>13}")
    for r in out:
        print(f"  {r['figure_index']:>3}  {r['route_id']:<10} "
              f"{r['dataset_id']:<10} {r['window_index']:>7}  "
              f"{r['raw_corr']:>7.3f} {r['raw_nmse']:>9.3f} "
              f"{r['spec_corr']:>7.3f} {r['spec_nmse']:>10.3f}  "
              f"{str(r['raw_nmse_rank']) + '/' + str(r['spec_nmse_rank']):>13}")
    path = os.path.join(w.meta_dir, "reconstruction_survey.json")
    with open(path, "w") as f:
        json.dump({"mask_seed": mask_seed, "n_windows": n_drawn,
                   "note": "one row per drawn window. rank 1 is the lowest "
                           "masked NMSE among the windows surveyed for that "
                           "route; a figure chosen by this table is a SELECTED "
                           "example and its caption should say so",
                   "windows": out}, f, indent=2)
    print(f"  wrote {path}")
    return out


def _recon_name(base: str, k: int) -> str:
    """``base`` for the first window, ``base_wNN`` after it.

    The first keeps the historical name so a paper script, a test or a figure
    reference written before the survey existed still resolves.
    """
    return base if k == 0 else f"{base}_w{k:02d}"


#: Panel titles, in order. The figure and the tests read the same list.
RECONSTRUCTION_PANEL_TITLES = (
    "raw EEG target", "raw EEG, mask applied", "raw composite reconstruction",
    "raw |error|, masked only", "spec target", "masked spec target",
    "spec composite reconstruction", "spec |error|, masked only",
)


def reconstruction_panels(ex: Dict, route: Route) -> List:
    """``[(array, title, cmap, vmin, vmax)] x 8`` for one route row.

    Pure, so the three properties that make the figure honest can be checked
    without rendering anything:

    * panels 0-2 share colour limits, and 4-6 share theirs. Taken from the
      TARGET, so a prediction with a third of the amplitude draws at a third of
      the contrast instead of being stretched to match.
    * the spec panels are ``target_spec``, the per-patch normalised tensor the
      loss is computed against -- never ``clean_spec``, which is on a different
      scale and is not what any gradient compared against.
    * the two error panels are NaN off the masked valid patches. Nothing
      supervises a visible token's decoder output.
    """
    pt = route.patch_t
    m = _expand(ex["mask"], pt)
    valid_rows = _expand(ex["valid"], pt)
    raw_t, raw_p = ex["target_raw"], ex["pred_raw"]
    spec_t, spec_p = ex["target_spec"], ex["pred_spec"]
    raw_lim = _sym_limits(raw_t, valid_rows)
    spec_lim = _sym_limits(spec_t, valid_rows)
    spec_label = ("normalised spec target"
                  if ex.get("normalize_spec_target", True)
                  else "spec target (raw scale)")
    titles = [_sig(t) for t in RECONSTRUCTION_PANEL_TITLES]
    titles[4] = spec_label
    titles[5] = f"masked {spec_label}"
    return [
        (raw_t, titles[0], "RdBu_r", -raw_lim, raw_lim),
        (_corrupt(raw_t, m), titles[1], "RdBu_r", -raw_lim, raw_lim),
        (_composite(raw_t, raw_p, m), titles[2], "RdBu_r", -raw_lim, raw_lim),
        (_masked_error(raw_p, raw_t, m, valid_rows), titles[3], "magma",
         0.0, None),
        (spec_t, titles[4], "RdBu_r", -spec_lim, spec_lim),
        (_hide(spec_t, m), titles[5], "RdBu_r", -spec_lim, spec_lim),
        (_composite(spec_t, spec_p, m), titles[6], "RdBu_r",
         -spec_lim, spec_lim),
        (_masked_error(spec_p, spec_t, m, valid_rows), titles[7], "magma",
         0.0, None),
    ]


def fig_mask_reconstruction(w: FigureWriter, rows, grid, ranks, mask_seed,
                            objective: Optional[Dict] = None):
    """Both heads, one row per route, on fixed windows under fixed masks.

    Eight panels, in two groups of four that read the same way: target, what
    the model was handed, the composite reconstruction, and the error on the
    masked patches alone.

    One figure per surveyed window. With ``--recon-windows 1`` that is the
    single window this figure always showed; above 1 the later figures are
    ``fig_mask_reconstruction_w01`` and so on, each carrying its rank among the
    survey so a chosen one can be captioned as chosen.

    Four things the previous version of this figure got wrong, each of which
    flattered the model:

    * it showed raw EEG only as context, so the second head -- half the
      objective -- never appeared;
    * it compared the UNNORMALISED clean_spec against a prediction trained
      against the normalised target, which is a comparison of two different
      quantities;
    * it autoscaled target and prediction independently, so a prediction with a
      third of the target's amplitude drew at the same contrast;
    * its error map covered visible patches too, and the visible patches are
      not supervised: their error is small because the encoder passes the token
      through, not because anything reconstructed them.
    """
    if not rows or not grid:
        return
    n_drawn = len(grid)
    for k, col in enumerate(grid):
        drawn = [(i, rid, dsid) for i, (rid, dsid) in enumerate(rows)
                 if col[i] is not None]
        if not drawn:
            continue
        fig, axes = plt.subplots(len(drawn), 8,
                                 figsize=(21, 2.6 * len(drawn)), squeeze=False)
        data, meta_rows = {}, []
        for r, (i, rid, dsid) in enumerate(drawn):
            ex = col[i]
            route = ROUTES[rid]
            pt = route.patch_t
            n_patches = route.patches_per_channel
            m = _expand(ex["mask"], pt)
            valid_rows = _expand(ex["valid"], pt)
            raw_t, raw_p = ex["target_raw"], ex["pred_raw"]
            spec_t, spec_p = ex["target_spec"], ex["pred_spec"]
            raw_err = _masked_error(raw_p, raw_t, m, valid_rows)
            spec_err = _masked_error(spec_p, spec_t, m, valid_rows)
            raw_lim = _sym_limits(raw_t, valid_rows)
            spec_lim = _sym_limits(spec_t, valid_rows)

            panels = reconstruction_panels(ex, route)
            for c, (arr, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(arr, aspect="auto", cmap=cmap,
                          interpolation="nearest", vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(title)
                if c == 0:
                    ax.set_ylabel(f"{rid}\n{dsid}\n{route.n_channels} ch")
                ax.set_xticks([])
                ax.set_yticks([])
                # The columns any channel has a masked patch in, marked on the
                # panels whose x axis is samples.
                for pi in range(n_patches):
                    if ex["mask"][:, pi].any():
                        ax.axvspan(pi * pt, (pi + 1) * pt, color="0.5",
                                   alpha=0.10, lw=0)
            mt = ex["metrics"]
            rk = ranks.get((rid, k), {})
            ratio = ex["mask"].sum() / max(1, ex["valid"].sum())
            axes[r][3].set_xlabel(
                f"SmoothL1 {mt['loss_masked_raw_smoothl1']:.4f}  "
                f"r {mt['masked_raw_corr']:.3f}  "
                f"NMSE {mt['masked_raw_nmse']:.3f}")
            axes[r][7].set_xlabel(
                f"MSE {mt['loss_masked_spec_mse']:.4f}  "
                f"r {mt['masked_spec_corr']:.3f}  "
                f"NMSE {mt['masked_spec_nmse']:.3f}")
            # The realised ratio, not the requested one: the budget is per
            # sample over the VALID tokens, so a route with padded slots masks
            # fewer. The rank rides along whenever there was a choice to make.
            label = f"window {ex['window_index']}   " \
                    f"masked {ratio * 100:.1f}% of valid patches"
            if n_drawn > 1 and rk:
                label += (f"\nNMSE rank raw {rk.get('raw')}/{rk.get('n')}, "
                          f"spec {rk.get('spec')}/{rk.get('n')}")
            axes[r][0].set_xlabel(label)
            data[f"{rid}_target_raw"] = raw_t
            data[f"{rid}_pred_raw"] = raw_p
            data[f"{rid}_target_spec"] = spec_t
            data[f"{rid}_pred_spec"] = spec_p
            data[f"{rid}_masked_error_raw"] = raw_err
            data[f"{rid}_masked_error_spec"] = spec_err
            data[f"{rid}_mask"] = ex["mask"]
            data[f"{rid}_valid"] = ex["valid"]
            # Compatibility with the pre-dual arrays, which meant the spec pair.
            data[f"{rid}_target"] = spec_t
            data[f"{rid}_pred"] = spec_p
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"],
                              "mask_seed": mask_seed,
                              "actual_mask_ratio": float(ratio),
                              "raw_color_limit": raw_lim,
                              "spec_color_limit": spec_lim,
                              "spec_target_is_normalised":
                                  ex["normalize_spec_target"],
                              "raw_nmse_rank": rk.get("raw"),
                              "spec_nmse_rank": rk.get("spec"),
                              "n_windows_surveyed": rk.get("n"),
                              **mt})
        title = ("Fixed validation windows under their fixed masks. Composite "
                 "= target on visible patches, prediction on masked ones; the "
                 "error maps are masked patches only, because visible patches "
                 "are not supervised. Target and composite share colour "
                 "limits.")
        if n_drawn > 1:
            title += (f"  Window {k + 1} of {n_drawn} surveyed per route -- a "
                      f"figure picked from this set is a SELECTED example.")
        fig.suptitle(title, y=1.005)
        w.save(fig, _recon_name("fig_mask_reconstruction", k), data,
               {"examples": meta_rows,
                "panels": [_sig(t) for t in RECONSTRUCTION_PANEL_TITLES],
                "survey_index": k, "n_windows_surveyed": n_drawn,
                "note": "spec panels use target_spec -- the per-patch "
                        "normalised target the loss is computed against -- not "
                        "clean_spec"})


def fig_mask_examples_by_dataset(w: FigureWriter, model, datasets, mask_seed,
                                 device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 3.0),
                             squeeze=False)
    data, meta_rows = {}, []
    for i, dsid in enumerate(present):
        route = ROUTES[datasets[dsid].route_id]
        ex = _fixed_example(model, datasets[dsid], route, mask_seed,
                            device=device)
        ax = axes[0][i]
        ax.imshow(ex["mask"].astype(float), aspect="auto", cmap="gray_r",
                  interpolation="nearest", vmin=0, vmax=1)
        ratio = ex["mask"].sum() / max(1, ex["valid"].sum())
        ax.set_title(f"{dsid}\n{route.route_id}  masked {ratio*100:.0f}%")
        ax.set_xlabel("time patch")
        if i == 0:
            ax.set_ylabel("channel slot")
        ax.set_xticks(range(route.patches_per_channel))
        data[f"{dsid}_mask"] = ex["mask"]
        meta_rows.append({"dataset_id": dsid, "route_id": route.route_id,
                          "subject_id": ex["subject_id"],
                          "recording_id": ex["recording_id"],
                          "window_index": ex["window_index"],
                          "mask_seed": mask_seed})
    fig.suptitle("The actual masks the model was given, one fixed window per "
                 "dataset", y=1.04)
    w.save(fig, "fig_mask_examples_by_dataset", data, {"examples": meta_rows})


def fig_mask_statistics(w: FigureWriter, model, datasets, mask_seed, device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    data = {}
    ratios, labels, colors = [], [], []
    for dsid in present:
        route = ROUTES[datasets[dsid].route_id]
        ex = _fixed_example(model, datasets[dsid], route, mask_seed,
                            device=device)
        ratios.append(float(ex["mask"].sum() / max(1, ex["valid"].sum())))
        labels.append(dsid)
        colors.append(ROUTE_COLOR[route.route_id])
        if dsid == present[0]:
            # Importance is the summed magnitude spectrum along the token
            # sequence. It is a rank over TOKENS and does not map to a physical
            # frequency, so the axis says token importance and not Hz.
            with torch.no_grad():
                item = datasets[dsid][0]
                batch = collate_windows([item])
                meta = {k: v.to(device) for k, v in datasets[dsid].montage().items()}
                spec = model.wavelet_frontends[route.route_id](
                    batch["x"].to(device))
                tok = model.patch_embed_by_rate[route.rate_key](spec.unsqueeze(1))
                imp = torch.abs(torch.fft.rfft(tok, dim=1)).sum(dim=2)
                imp = torch.nn.functional.interpolate(
                    imp.unsqueeze(1), size=tok.shape[1], mode="linear",
                    align_corners=True).squeeze(1)[0].cpu().numpy()
            sel = ex["mask"].reshape(-1)
            axes[1].scatter(np.arange(imp.size)[~sel], imp[~sel], s=3,
                            color="0.7", label="kept")
            axes[1].scatter(np.arange(imp.size)[sel], imp[sel], s=3,
                            color=OKABE_ITO[1], label="masked")
            axes[1].set_title(f"token importance vs selection ({dsid})")
            axes[1].set_xlabel("token index")
            axes[1].set_ylabel("token/patch importance (arb. units)")
            axes[1].legend(frameon=False, markerscale=3)
            data["importance"] = imp
            data["selected"] = sel

    axes[0].bar(range(len(ratios)), [r * 100 for r in ratios], color=colors)
    axes[0].axhline(model.mask_ratio * 100, color="0.3", ls="--", lw=1,
                    label=f"requested {model.mask_ratio*100:.0f}%")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("actual mask ratio over valid tokens (%)")
    axes[0].set_title("realised mask ratio")
    axes[0].legend(frameon=False)

    for rid, color in ROUTE_COLOR.items():
        members = [i for i, d in enumerate(labels)
                   if PRETRAIN_DATASETS[d].route_id == rid]
        if members:
            axes[2].bar([labels[i] for i in members],
                        [ratios[i] * 100 for i in members], color=color,
                        label=rid)
    axes[2].set_title("mask ratio by route")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(frameon=False)
    data["ratios"] = np.asarray(ratios)
    data["labels"] = np.asarray(labels)
    w.save(fig, "fig_mask_statistics", data, {"mask_seed": mask_seed})


# --------------------------------------------------------------------------- #
# 7, 8, 9. what the model learned
# --------------------------------------------------------------------------- #

def _filter_bank(frontend) -> Optional[np.ndarray]:
    """The learned analysis kernels of one frontend, as ``[n_filters, K]``."""
    kernels = []
    for name, p in frontend.named_parameters():
        if p.dim() >= 2 and p.shape[-1] >= 4 and ("filter" in name or "wave" in name
                                                  or "kernel" in name):
            kernels.append(p.detach().float().reshape(-1, p.shape[-1]).cpu().numpy())
    if not kernels:
        return None
    return np.concatenate(kernels, axis=0)


def fig_wavelet_frequency_response(w: FigureWriter, model,
                                   init_model: Optional[object]):
    # One panel per route of THIS modality: EEG has four, ECG two, sEMG three,
    # and a fixed four left an empty axes beside sEMG's.
    fig, axes = plt.subplots(1, len(ROUTES), figsize=(3.5 * len(ROUTES), 3.1))
    axes = np.atleast_1d(axes)
    data = {}
    for ax, (rid, route) in zip(axes, ROUTES.items()):
        for label, m, color, alpha in (
                ("at init", init_model, "0.6", 0.9),
                ("trained", model, ROUTE_COLOR[rid], 1.0)):
            if m is None:
                continue
            bank = _filter_bank(m.wavelet_frontends[rid])
            if bank is None:
                continue
            K = bank.shape[-1]
            H = np.abs(np.fft.rfft(bank, n=max(256, K * 8), axis=-1))
            H = H.mean(axis=0)
            H = H / max(H.max(), 1e-12)
            freqs = np.fft.rfftfreq(max(256, K * 8), d=1.0 / route.sampling_rate)
            ax.plot(freqs, 20 * np.log10(np.maximum(H, 1e-6)), color=color,
                    alpha=alpha, label=label)
            data[f"{rid}_{label.replace(' ', '_')}_freq"] = freqs
            data[f"{rid}_{label.replace(' ', '_')}_db"] = 20 * np.log10(
                np.maximum(H, 1e-6))
        ax.set_title(f"{rid}  ({route.sampling_rate} Hz)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(0, route.sampling_rate / 2)
        ax.set_ylim(-60, 2)
    axes[0].set_ylabel("mean magnitude response (dB)")
    axes[0].legend(frameon=False)
    fig.suptitle("Learned wavelet analysis filters, per route, on that route's "
                 "own frequency axis", y=1.03)
    w.save(fig, "fig_wavelet_frequency_response", data)


def fig_scale_fold_weights(w: FigureWriter, model, datasets, device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    data = {}
    alphas, spreads, labels, colors = [], [], [], []
    with torch.no_grad():
        for dsid in present:
            route = ROUTES[datasets[dsid].route_id]
            batch = collate_windows([datasets[dsid][0]])
            model.wavelet_frontends[route.route_id](batch["x"].to(device))
            fold = model.wavelet_frontends[route.route_id].fold
            a = fold.alpha_mean
            if a is None:
                continue
            alphas.append(a.detach().float().cpu().numpy())
            st, sc = fold.alpha_std_time, fold.alpha_std_chan
            spreads.append((float(st.mean()) if st is not None else 0.0,
                            float(sc.mean()) if sc is not None else 0.0))
            labels.append(dsid)
            colors.append(ROUTE_COLOR[route.route_id])
    if not alphas:
        return
    A = np.stack(alphas)
    S = A.shape[1]
    xs = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for s in range(S):
        axes[0].bar(xs, A[:, s], bottom=bottom, label=f"scale {s}",
                    color=OKABE_ITO[s % len(OKABE_ITO)])
        bottom += A[:, s]
    axes[0].axhline(1.0, color="0.3", lw=0.6)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("mean mixing weight")
    axes[0].set_title("ScaleFold alpha by dataset\n(a single full bar = collapse "
                      "onto one scale)")
    axes[0].legend(frameon=False, ncol=2)

    sp = np.asarray(spreads)
    axes[1].bar(xs - 0.2, sp[:, 0], width=0.4, color=OKABE_ITO[0],
                label="std over time blocks")
    axes[1].bar(xs + 0.2, sp[:, 1], width=0.4, color=OKABE_ITO[1],
                label="std over channels")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_title("fold spread (zero over time = a static fold)")
    axes[1].legend(frameon=False)
    data["alpha"] = A
    data["spread"] = sp
    data["labels"] = np.asarray(labels)
    w.save(fig, "fig_scale_fold_weights", data)


#: Lobe by 10-20 name prefix. A NAMING rule, applied to the channel's label --
#: C1 is a learned embedding of a name and contains no geometry, so the colours
#: are a reader's aid and not a property the model was given. Recorded in the
#: figure metadata so the grouping can be checked.
LOBE_RULES = (("frontal", ("FP", "AF", "F")), ("central", ("FC", "C", "CP")),
              ("parietal", ("P", "PO")), ("occipital", ("O", "I")),
              ("temporal", ("T",)))


#: The ECG equivalent: the frontal-plane leads, the chest leads, and the
#: single-lead recordings, which are neither.
LOBE_RULES_ECG = (("limb", ("I", "AV")), ("precordial", ("V",)),
                  ("single-lead", ("PATCH", "ML")))


#: Which signal the figures are of. Text is written for EEG and translated at
#: the moment it is drawn, so the EEG figures -- and the tests that read their
#: titles -- are untouched.
SIGNAL = "EEG"


def _sig(text: str) -> str:
    """A caption written for EEG, said of the signal actually drawn."""
    if SIGNAL == "EEG":
        return text
    for old, new in (("z-scored preprocessed EEG", "window-normalised "
                      f"preprocessed {SIGNAL}"),
                     ("z-scored EEG", f"window-normalised {SIGNAL}"),
                     ("raw EDF values", "raw mV values"),
                     ("z-scored", "window-normalised"),
                     ("EEG", SIGNAL)):
        text = text.replace(old, new)
    return text


def use_ecg_registry():
    """Point every figure at the ECG routes, corpora, model and lead names.

    The figures read these module-level tables when they run, so swapping
    them here, before any figure is drawn, is the whole of the ECG support:
    nothing below this line knows which modality it is drawing.
    """
    global ROUTES, PRETRAIN_DATASETS, ROUTE_COLOR, DATASET_COLOR, \
        CHANNEL_VOCAB, LOBE_RULES, SIGNAL
    from physiowave.ecg_c1.leads import ECG_LEAD_VOCAB
    from physiowave.ecg_c1.routes import PRETRAIN_DATASETS as _D
    from physiowave.ecg_c1.routes import ROUTES as _R
    ROUTES, PRETRAIN_DATASETS = _R, _D
    ROUTE_COLOR = {rid: OKABE_ITO[i] for i, rid in enumerate(_R)}
    DATASET_COLOR = {d: OKABE_ITO[i % len(OKABE_ITO)]
                     for i, d in enumerate(_D)}
    CHANNEL_VOCAB = ECG_LEAD_VOCAB
    LOBE_RULES = LOBE_RULES_ECG
    SIGNAL = "ECG"


LOBE_RULES_EMG = (("wristband", ("BAND",)), ("forearm rings", ("PUTEMG",)),
                  ("HD grid", ("HYSER", "CEMHSEY")))


def use_emg_registry():
    """The sEMG routes, corpora and electrode names; see use_ecg_registry."""
    global ROUTES, PRETRAIN_DATASETS, ROUTE_COLOR, DATASET_COLOR, \
        CHANNEL_VOCAB, LOBE_RULES, SIGNAL
    from physiowave.emg_c1.electrodes import EMG_ELECTRODE_VOCAB
    from physiowave.emg_c1.routes import PRETRAIN_DATASETS as _D
    from physiowave.emg_c1.routes import ROUTES as _R
    ROUTES, PRETRAIN_DATASETS = _R, _D
    ROUTE_COLOR = {rid: OKABE_ITO[i] for i, rid in enumerate(_R)}
    DATASET_COLOR = {d: OKABE_ITO[i % len(OKABE_ITO)]
                     for i, d in enumerate(_D)}
    CHANNEL_VOCAB = EMG_ELECTRODE_VOCAB
    LOBE_RULES = LOBE_RULES_EMG
    SIGNAL = "EMG"


def _slot_name(dsid: str, route, ch: int) -> str:
    """The dataset's own electrode name for slot ``ch``.

    A route's slots are named after one of its corpora (G64_2000's after
    Hyser), and a CEMHSEY window captioned ``hyser_r3c8`` names an electrode
    it was not recorded with. Registries whose datasets carry no names of
    their own (EEG) fall back to the route's.
    """
    spec = PRETRAIN_DATASETS.get(dsid)
    names = tuple(getattr(spec, "lead_slots", None) or ()) if spec else ()
    if len(names) == route.n_channels:
        return names[ch]
    return route.slots[ch]


def _lobe(name: str) -> str:
    u = name.upper()
    for lobe, prefixes in LOBE_RULES:
        for p in sorted(prefixes, key=len, reverse=True):
            if u.startswith(p):
                return lobe
    return "other"


def fig_channel_embedding(w: FigureWriter, model, datasets):
    if model.channel_encoder is None:
        return
    used = sorted({int(i) for ds in datasets.values()
                   for i in ds.montage()["channel_ids"].tolist() if i > 1})
    if len(used) < 3:
        used = list(range(2, min(len(CHANNEL_VOCAB), 40)))
    E = model.channel_encoder.id_embed.weight.detach().float().cpu().numpy()[used]
    names = [CHANNEL_VOCAB[i] for i in used]
    lobes = [_lobe(n) for n in names]
    palette = {l: OKABE_ITO[i] for i, l in enumerate(sorted(set(lobes)))}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    norms = np.linalg.norm(E, axis=1)
    axes[0].bar(range(len(names)), norms,
                color=[palette[l] for l in lobes])
    if len(names) <= 80:
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=90, fontsize=4)
    else:
        # The sEMG vocabulary puts 232 bars here, and 232 names -- or every
        # third of them -- merge into a black band. Label each run of one
        # electrode family (band, putemg, hyser, cemhsey, cemhsey13) once,
        # with its count, and separate the runs.
        fam = [re.sub(r"[_\d].*$", "", n) if "_" in n else re.sub(r"\d+$", "", n)
               for n in names]
        runs, start = [], 0
        for i in range(1, len(fam) + 1):
            if i == len(fam) or fam[i] != fam[start]:
                runs.append((start, i - 1, fam[start]))
                start = i
        axes[0].set_xticks([(a + b) / 2 for a, b, _ in runs])
        axes[0].set_xticklabels([f"{f}\n({b - a + 1})" for a, b, f in runs],
                                fontsize=7)
        for a, _, _ in runs[1:]:
            axes[0].axvline(a - 0.5, color="0.5", lw=0.5)
    axes[0].set_ylabel("embedding L2 norm")
    axes[0].set_title("C1 channel-name embedding norm")

    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    # This machine's BLAS raises spurious FP warnings from matmul on ordinary
    # input; the result is checked below rather than trusted silently.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Cm = En @ En.T
    if not np.all(np.isfinite(Cm)):
        raise SystemExit("channel embedding cosine matrix is not finite; the "
                         "checkpoint's embedding table contains NaN or Inf.")
    Cm = np.clip(Cm, -1.0, 1.0)
    im = axes[1].imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("cosine similarity")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    Ec = E - E.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T
    for l in sorted(set(lobes)):
        sel = [i for i, x in enumerate(lobes) if x == l]
        axes[2].scatter(P[sel, 0], P[sel, 1], s=12, color=palette[l], label=l)
    axes[2].set_title("PCA (2 components)")
    axes[2].set_xlabel(f"PC1 ({S[0]**2/np.sum(S**2)*100:.0f}% var)")
    axes[2].set_ylabel(f"PC2 ({S[1]**2/np.sum(S**2)*100:.0f}% var)")
    axes[2].legend(frameon=False)
    fig.suptitle("C1 is a learned embedding of the channel NAME. Group colours "
                 "come from a name-prefix rule, not from electrode geometry.",
                 y=1.04)
    w.save(fig, "fig_channel_embedding",
           {"embedding": E, "names": np.asarray(names),
            "lobes": np.asarray(lobes), "cosine": Cm, "pca": P},
           {"lobe_rules": {k: list(v) for k, v in LOBE_RULES},
            "channel_ids": used})


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 10. the two objectives, side by side
# --------------------------------------------------------------------------- #

def fig_dual_objective(w: FigureWriter, epoch_rows):
    """Both reconstruction terms over training, on their own scales.

    They are not comparable in magnitude -- one is an MSE on folded wavelet
    coefficients, the other a SmoothL1 on z-scored EEG -- so plotting them
    against a shared axis would say more about their units than about their
    progress. Two panels, and a third for the weighted contributions, which IS
    the comparison that matters: it shows which term the optimizer is actually
    following.
    """
    if not epoch_rows:
        return
    ep = [r["epoch"] for r in epoch_rows]

    def series(prefix, key):
        return [r.get(f"{prefix}/{key}", float("nan")) for r in epoch_rows]

    spec_tr, spec_va = series("train", "loss_masked_spec_mse"), \
        series("val", "loss_masked_spec_mse")
    raw_tr, raw_va = series("train", "loss_masked_raw_smoothl1"), \
        series("val", "loss_masked_raw_smoothl1")
    kl_tr = series("train", "loss_fold_kl")
    if all(math.isnan(v) for v in spec_tr):
        return

    # From the run's own resolved configuration, through the same helper the
    # trainer used. NOT a literal: a plotting default is exactly how a figure
    # comes to caption weights the run never trained under.
    meta = resolve_eeg_c1_objective({"objective": w.base_meta.get("objective")})
    ws = float(meta["spec_weight"])
    wr = float(meta["raw_weight"])
    wk = float(meta["fold_kl"])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    for ax, (tr, va, title, ylab) in zip(axes[:2], (
            (spec_tr, spec_va, "Spec reconstruction",
             "masked MSE, folded wavelet"),
            (raw_tr, raw_va, _sig("Raw EEG reconstruction"),
             _sig("masked SmoothL1, z-scored")))):
        ax.plot(ep, tr, color=OKABE_ITO[0], label="train", marker="o", ms=2.5)
        ax.plot(ep, va, color=OKABE_ITO[1], label="val", marker="s", ms=2.5,
                ls="--")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)

    ax = axes[2]
    contrib = {
        f"spec x{ws:g}": ([ws * v for v in spec_tr], OKABE_ITO[0]),
        f"raw x{wr:g}": ([wr * v for v in raw_tr], OKABE_ITO[1]),
        f"foldKL x{wk:g}": ([wk * v for v in kl_tr], OKABE_ITO[2]),
    }
    bottom = np.zeros(len(ep))
    for label, (vals, colour) in contrib.items():
        v = np.nan_to_num(np.asarray(vals, dtype=float))
        ax.fill_between(ep, bottom, bottom + v, label=label, color=colour,
                        alpha=0.85, lw=0)
        bottom = bottom + v
    ax.set_title("Weighted contribution to the total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight x loss")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(f"The two reconstruction targets:  spec x{ws:g}   "
                 f"raw x{wr:g}   fold KL x{wk:g}", y=1.04, fontsize=10)
    fig.tight_layout()
    w.save(fig, "10_dual_objective",
           data={"epoch": ep, "spec_train": spec_tr, "spec_val": spec_va,
                 "raw_train": raw_tr, "raw_val": raw_va, "fold_kl": kl_tr},
           meta={"weights": {"spec": ws, "raw": wr, "fold_kl": wk},
                 "objective_equation": objective_equation(meta),
                 "note": "panels 1-2 are unweighted losses on their own "
                         "scales and are NOT comparable to each other; panel 3 "
                         "is what enters the total"})


# --------------------------------------------------------------------------- #
# 11. masked against visible -- the control the loss cannot provide
# --------------------------------------------------------------------------- #

def fig_masked_vs_visible(w: FigureWriter, run_dir: str):
    """Where the reconstruction error sits, on hidden tokens and on seen ones.

    The loss only ever looks at masked tokens, so it cannot distinguish a model
    that has learned structure from one that has learned to copy its input: the
    second has a near-zero VISIBLE error and an unimproved masked one. The gap
    between the two distributions is the diagnostic, and it is why the visible
    half is collected during validation despite contributing nothing to
    training.
    """
    path = os.path.join(run_dir, "error_histogram.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        h = json.load(f)
    edges = np.asarray(h["edges"], dtype=float)
    centres = np.sqrt(np.maximum(edges[:-1], 1e-4) * edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, tag, title in ((axes[0], "spec", "Folded-wavelet patches"),
                           (axes[1], "raw", _sig("Preprocessed EEG patches"))):
        for half, colour, ls in (("masked", OKABE_ITO[1], "-"),
                                 ("visible", OKABE_ITO[0], "--")):
            key = f"{tag}_{half}"
            counts = np.asarray(h["counts"].get(key, []), dtype=float)
            if counts.sum() == 0:
                continue
            dens = counts / counts.sum()
            ax.step(centres, dens, where="mid", color=colour, ls=ls,
                    label=f"{half}  (mean {h['mean_abs_error'][key]:.3g})")
            ax.axvline(h["mean_abs_error"][key], color=colour, lw=0.6,
                       alpha=0.5)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("|prediction - target|")
        ax.set_ylabel("fraction of tokens")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)

    ratio = {}
    for tag in ("spec", "raw"):
        m = h["mean_abs_error"].get(f"{tag}_masked", 0.0)
        v = h["mean_abs_error"].get(f"{tag}_visible", 0.0)
        ratio[tag] = (m / v) if v else float("nan")
    fig.suptitle(
        f"Masked vs visible reconstruction error   "
        f"(masked/visible: spec {ratio['spec']:.2f}x, raw {ratio['raw']:.2f}x)",
        y=1.05, fontsize=10)
    fig.tight_layout()
    w.save(fig, "11_masked_vs_visible_error",
           data={"edges": edges,
                 **{k: np.asarray(v) for k, v in h["counts"].items()}},
           meta={"epoch": h.get("epoch"), "n_tokens": h.get("n"),
                 "mean_abs_error": h.get("mean_abs_error"),
                 "masked_over_visible": ratio,
                 "note": "a ratio near 1 means masking costs the model "
                         "nothing, which is what copying looks like"})


# --------------------------------------------------------------------------- #
# 12. where the gradient goes
# --------------------------------------------------------------------------- #

def fig_gradient_flow(w: FigureWriter, step_rows):
    """Gradient norm per branch over training.

    A single global norm cannot answer the question this model raises: with two
    decoders, a frontend run twice and a channel path gated to zero at
    initialisation, "the gradient is healthy" has to be a statement about
    branches. A branch pinned at exactly zero for the whole run is either
    disconnected or deliberately gated, and those look identical in the total.
    """
    keys = sorted({k for r in step_rows for k in r if k.startswith("gradnorm/")})
    if not keys:
        return
    steps = [r["step"] for r in step_rows if any(k in r for k in keys)]
    if len(steps) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, key in enumerate(keys):
        xs = [r["step"] for r in step_rows if key in r]
        ys = [r[key] for r in step_rows if key in r]
        if not ys or all(v == 0 for v in ys):
            ax.plot([], [], color=OKABE_ITO[i % 8], ls=":",
                    label=f"{key.split('/', 1)[1]}  (zero throughout)")
            continue
        ax.plot(xs, ys, color=OKABE_ITO[i % 8], lw=1.0,
                label=key.split("/", 1)[1])
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Gradient norm by branch")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper right")
    fig.tight_layout()
    w.save(fig, "12_gradient_flow",
           data={"step": steps,
                 **{k.replace("/", "_"): [r.get(k, float("nan"))
                                          for r in step_rows] for k in keys}},
           meta={"branches": keys,
                 "note": "the channel encoder is zero while tanh(gate) is "
                         "zero -- that is the initialisation, not a fault"})


# --------------------------------------------------------------------------- #
# 13. what each route cost
# --------------------------------------------------------------------------- #

def fig_route_cost(w: FigureWriter, epoch_rows):
    """Share of steps, share of windows and share of WALL CLOCK, per route.

    The three differ and the third is the one nobody configures. E128_512 draws
    12 windows to E19_256's 64 and each carries 6.7x the tokens, so a mixture
    set by share of steps buys a very different share of the compute -- and a
    run can be spending most of its hours on the route with the smallest share
    of the data.
    """
    if not epoch_rows:
        return
    last = epoch_rows[-1]
    routes = sorted({k.split("/")[-1] for k in last
                     if k.startswith("train/route_share_of_time/")})
    if not routes:
        return

    time_share = [last.get(f"train/route_share_of_time/{r}", 0.0) for r in routes]
    windows = [last.get(f"train/route_windows/{r}", 0.0) for r in routes]
    tot_w = sum(windows) or 1.0
    window_share = [v / tot_w for v in windows]

    x = np.arange(len(routes))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.2, window_share, 0.4, label="share of windows",
           color=[ROUTE_COLOR.get(r, OKABE_ITO[0]) for r in routes], alpha=0.55)
    ax.bar(x + 0.2, time_share, 0.4, label="share of wall clock",
           color=[ROUTE_COLOR.get(r, OKABE_ITO[0]) for r in routes])
    for xi, (ws_, ts) in enumerate(zip(window_share, time_share)):
        if ws_ > 0:
            ax.text(xi + 0.2, ts + 0.01, f"{ts / ws_:.1f}x", ha="center",
                    fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(routes)
    ax.set_ylabel("share of the epoch")
    ax.set_title("What each route cost, against what it contributed")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    w.save(fig, "13_route_cost",
           data={"route": routes, "window_share": window_share,
                 "time_share": time_share, "windows": windows},
           meta={"epoch": last.get("epoch"),
                 "note": "the multiplier above each bar is wall clock per "
                         "window relative to that route's share of windows"})


# --------------------------------------------------------------------------- #
# 14. the raw decoder's output, as a waveform
# --------------------------------------------------------------------------- #

def fig_raw_waveform_reconstruction(w: FigureWriter, rows, grid, ranks,
                                    mask_seed, mask_ratio: float):
    """The second head's prediction against the EEG it was asked to predict.

    One row per route, so E19_256's reconstruction can be read against
    E128_512's rather than against nothing. The previous version showed the
    first dataset the iteration happened to yield and one channel of it, which
    is a sample, not a comparison.

    One figure per surveyed window, on the SAME windows as
    fig_mask_reconstruction, so the two figures at index k describe one sample
    from two sides.

    This is the only figure in the set showing a WAVEFORM in the units the
    recording was preprocessed into. The spec figures show folded wavelet
    patches, which are not EEG and must not be captioned as though they were.

    The prediction is drawn ONLY inside masked intervals. Outside them the
    decoder is unsupervised -- no gradient ever reached those outputs -- and a
    continuous orange line across the whole window would be read as a
    reconstruction of the whole window.
    """
    if not rows or not grid:
        return
    n_drawn = len(grid)
    for k, col in enumerate(grid):
        present = [(i, rid, dsid) for i, (rid, dsid) in enumerate(rows)
                   if col[i] is not None]
        if not present:
            continue
        fig, axes = plt.subplots(
            2 * len(present), 1, figsize=(9.4, 2.3 * len(present)),
            gridspec_kw={"height_ratios": [3, 1] * len(present)})
        axes = np.atleast_1d(axes).reshape(-1)
        data, meta_rows = {}, []

        for i_row, (i, rid, dsid) in enumerate(present):
            route = ROUTES[rid]
            ex = col[i]
            pt, n_patches = route.patch_t, route.patches_per_channel
            mask, valid = ex["mask"], ex["valid"]
            n_masked = mask.sum(axis=1)
            # A VALID channel with SOME masked patches and some visible ones.
            # Not simply the most-masked channel: at mask_ratio 0.70 a good
            # many channels have all their patches hidden, and a row with no
            # visible stretch shows neither the context the model had nor the
            # boundary between the two -- the shading covers the whole window
            # and the composite is the prediction everywhere.
            ok = valid.all(axis=1)
            mixed = np.where(ok & (n_masked > 0)
                             & (n_masked < n_patches))[0]
            if mixed.size:
                want = max(1, min(n_patches - 1,
                                  int(round(mask_ratio * n_patches))))
                ch = int(mixed[np.argmin(np.abs(n_masked[mixed] - want))])
            else:
                fallback = np.where(n_masked > 0)[0]
                if fallback.size == 0:
                    continue
                ch = int(fallback[np.argmax(n_masked[fallback])])

            ch_name = _slot_name(dsid, route, ch)
            target = ex["target_raw"][ch]
            pred = ex["pred_raw"][ch]
            m = _expand(mask[ch:ch + 1], pt)[0]
            composite = np.where(m, pred, target)
            t = np.arange(target.size) / route.sampling_rate

            # Dimensionless, and computed on the masked samples of THIS
            # channel, so the numbers under the row are the numbers the row
            # shows. float64 throughout. In float32 the sums of squares behind
            # a correlation over a few thousand samples overflow to inf on a
            # route whose decoder has not converged, and numpy reports it as
            # "overflow encountered in matmul" rather than as a number.
            pm = pred[m].astype(np.float64)
            tm = target[m].astype(np.float64)
            if tm.size:
                resid = float(np.mean((pm - tm) ** 2))
                base = float(np.mean(tm ** 2))
                nmse = resid / max(base, 1e-12)
                pc, tc = pm - pm.mean(), tm - tm.mean()
                den = float(np.linalg.norm(pc) * np.linalg.norm(tc))
                corr = (float(np.dot(pc, tc) / den)
                        if den > 0 and np.isfinite(den) else float("nan"))
                mae = float(np.mean(np.abs(pm - tm)))
                rmse = math.sqrt(resid)
            else:
                nmse = corr = mae = rmse = float("nan")

            ax, ax_err = axes[2 * i_row], axes[2 * i_row + 1]
            ax.plot(t, target, color="0.25", lw=0.9,
                    label=_sig("preprocessed EEG (target)"))
            ax.plot(t, np.where(m, composite, np.nan), color=OKABE_ITO[1],
                    lw=1.1, label="raw head, masked patches only")
            ax.plot(t, composite, color=OKABE_ITO[0], lw=0.6, alpha=0.55,
                    label="composite (target visible + prediction masked)")
            for pi in range(n_patches):
                if mask[ch, pi]:
                    ax.axvspan(pi * pt / route.sampling_rate,
                               (pi + 1) * pt / route.sampling_rate,
                               color=OKABE_ITO[4], alpha=0.16, lw=0)
            ax.set_ylabel(_sig("amplitude\n(z-scored)"))
            ax.set_title(
                f"{rid}  {dsid}  channel {ch_name}  "
                f"window {ex['window_index']}   "
                f"r {corr:.3f}   NMSE {nmse:.3f}   MAE {mae:.3f}   "
                f"RMSE {rmse:.3f}   masked {mask[ch].sum()}/{n_patches} "
                f"patches", loc="left")
            # The same limits on both panels, explicitly. fill_between
            # autoscales the error panel to the MASKED stretches it filled,
            # so when the last patch was visible the error panel ended early
            # and the one time axis at the bottom of the figure (that panel's)
            # was wrong for every trace above it.
            ax.set_xlim(float(t[0]), float(t[-1]))
            ax.grid(alpha=0.2, lw=0.5)
            ax.set_xticklabels([])
            if i_row == 0:
                ax.legend(frameon=False, ncol=3, fontsize=6.5,
                          loc="upper right")

            err = np.abs(pred - target)
            err[~m] = np.nan
            ax_err.fill_between(t, 0, np.nan_to_num(err), where=m,
                                color=OKABE_ITO[3], lw=0)
            ax_err.set_xlim(float(t[0]), float(t[-1]))
            ax_err.set_ylabel("|error|\nmasked")
            ax_err.grid(alpha=0.2, lw=0.5)
            if i_row == len(present) - 1:
                ax_err.set_xlabel("time (s)")
            else:
                ax_err.set_xticklabels([])

            rk = ranks.get((rid, k), {})
            data[f"{rid}_time"] = t
            data[f"{rid}_target"] = target
            data[f"{rid}_pred"] = pred
            data[f"{rid}_composite"] = composite
            data[f"{rid}_mask"] = mask[ch]
            data[f"{rid}_masked_error"] = err
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "channel": ch_name, "channel_slot": ch,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"],
                              "mask_seed": mask_seed,
                              "raw_corr": corr, "raw_nmse": nmse,
                              "raw_mae": mae, "raw_rmse": rmse,
                              "window_raw_nmse_rank": rk.get("raw"),
                              "n_windows_surveyed": rk.get("n")})

        if not meta_rows:
            plt.close(fig)
            continue
        # Compatibility with the single-route version of this figure, whose npz
        # held bare `time`/`target`/`pred`/`mask` and whose metadata was flat.
        # Paper scripts written against those names keep working and get the
        # first route's row, which is what they used to get.
        first = meta_rows[0]["route_id"]
        for bare in ("time", "target", "pred", "composite", "mask",
                     "masked_error"):
            data[bare] = data[f"{first}_{bare}"]
        title = _sig("Raw-head reconstruction, one row per route. The trace "
                     "is PREPROCESSED, z-scored EEG -- not raw EDF values, and "
                     "not the folded wavelet representation. Shaded = masked "
                     "before the frontend; the prediction is drawn only there, "
                     "because visible patches are unsupervised.")
        if n_drawn > 1:
            title += (f"  Window {k + 1} of {n_drawn} surveyed per route.")
        fig.suptitle(title, y=1.005, fontsize=8.5)
        fig.tight_layout()
        w.save(fig, _recon_name("14_raw_waveform_reconstruction", k), data,
               {"rows": meta_rows,
                "survey_index": k, "n_windows_surveyed": n_drawn,
                # The flat keys the single-route version wrote, from row one.
                "dataset_id": meta_rows[0]["dataset_id"],
                "route_id": meta_rows[0]["route_id"],
                "channel": meta_rows[0]["channel"],
                "note": _sig("z-scored preprocessed EEG, not raw EDF values, "
                             "and not the folded wavelet representation; the "
                             "prediction is shown on masked patches only. One "
                             "row per route; the unprefixed arrays are the "
                             "first route's, for scripts written against the "
                             "single-route version.")})


# --------------------------------------------------------------------------- #
# Paper composites: (A) the multi-route corpus and its convergence, (B) masked
# reconstruction. Drawn from the same artefacts as the working figures above.
# --------------------------------------------------------------------------- #

#: Display names for the paper; the registry keys stay the dataset ids.
DATASET_LABEL = {"tueg": "TUEG", "faced": "FACED", "tdbrain": "TDBRAIN",
                 "physionet_mi": "PhysioNet-MI", "m3cv": "M3CV", "hbn": "HBN",
                 "hgd": "HGD"}
#: The figures that need no checkpoint: everything they draw is in the run's
#: dataset_manifest.json and metrics_epoch.jsonl.
MODEL_FREE_FIGURES = {"fig_dataset_routes", "fig_pretraining_convergence",
                      "fig_route_convergence", "fig_dual_objective",
                      "fig_route_cost", "fig_paper_pretraining"}


def _route_label(rid: str) -> str:
    r = ROUTES[rid]
    return f"{rid.split('_')[0]}: {r.n_channels} ch, {r.sampling_rate:g} Hz"


def _panel_letter(ax, ch: str, title: str):
    """The panel letter and its title on one baseline, above the axes."""
    ax.text(-0.015, 1.04, ch, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="baseline")
    ax.text(0.01, 1.04, title, transform=ax.transAxes, fontsize=8, ha="left",
            va="baseline")


def fig_paper_pretraining(w: FigureWriter, manifest: Dict, epoch_rows: List[Dict]):
    """(a) corpus size per dataset, (b) the sampling policy against what the
    run realised, (c) validation loss of every route on one shared Transformer."""
    datasets = [d for d in PRETRAIN_DATASETS
                if manifest.get("datasets", {}).get(d, {}).get("n_windows")]
    if not datasets:
        print("  fig_paper_pretraining: the run manifest has no dataset sizes")
        return
    route_of = {d: PRETRAIN_DATASETS[d].route_id for d in datasets}
    counts = [manifest["datasets"][d]["n_windows"] for d in datasets]
    mix = manifest.get("realised_mixture", {})
    target = [100 * manifest.get("target_weights", {}).get(d, 0.0) for d in datasets]
    realised = [100 * mix.get("by_step", {}).get(d, 0.0) for d in datasets]
    labels = [DATASET_LABEL.get(d, d) for d in datasets]
    colors = [ROUTE_COLOR[route_of[d]] for d in datasets]

    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.55,
                          left=0.11, right=0.99, top=0.86, bottom=0.27)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    data = {"datasets": np.asarray(datasets), "n_windows": counts,
            "target_pct": target, "realised_pct": realised}
    y = np.arange(len(datasets))

    a.barh(y, counts, color=colors, height=0.68)
    a.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    # Plain labels: the mathtext exponents of 10^k render below the 5 pt floor.
    a.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: next((f"{v / d:g}{u}" for d, u in ((1e6, "M"), (1e3, "k"))
                           if v >= d), f"{v:g}")))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_yticks(y)
    a.set_yticklabels(labels)
    a.invert_yaxis()
    a.set_xlabel("Training windows (4 s)")
    _panel_letter(a, "a", "Pretraining corpora")

    hb = 0.36
    b.barh(y - hb / 2, target, height=hb, color="white", edgecolor="0.35", lw=0.7,
           label="Target")
    b.barh(y + hb / 2, realised, height=hb, color=colors, label="Realised")
    b.set_yticks(y)
    b.set_yticklabels([])
    b.invert_yaxis()
    b.set_xlabel("Share of optimiser steps (%)")
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(facecolor="white", edgecolor="0.35", label="Target"),
                      Patch(facecolor="0.55", label="Realised")],
             frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.6),
             handlelength=1.2)
    _panel_letter(b, "b", "Temperature sampling")

    metric = _first_key(epoch_rows, ("loss_total", "loss_masked_mse"))

    def by_epoch(key):
        xs, ys = [], []
        for r in epoch_rows:
            v = r.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                xs.append(int(r.get("epoch", len(xs))))
                ys.append(float(v))
        return np.asarray(xs), np.asarray(ys)
    for rid, col in ROUTE_COLOR.items():
        x, yv = by_epoch(f"val/route/{rid}/{metric}")
        if x.size:
            c.plot(x, yv, color=col, lw=1.3, label=_route_label(rid))
            data[f"val_{rid}_epoch"], data[f"val_{rid}"] = x, yv
    mx, my = by_epoch("val/macro_route_loss_total")
    if mx.size:
        c.plot(mx, my, color="0.15", lw=1.3, ls="--", label="Macro average")
        data["val_macro_epoch"], data["val_macro"] = mx, my
    c.set_xlabel("Epoch")
    c.set_ylabel("Validation loss")
    c.set_xlim(left=0)
    _panel_letter(c, "c", "Convergence per route")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, lw=4, label=_route_label(rid))
               for rid, col in ROUTE_COLOR.items()]
    handles.append(Line2D([], [], color="0.15", lw=1.3, ls="--", label="Macro average (c)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.55, 0.0), handlelength=1.6, columnspacing=1.2)
    fig.suptitle("Multi-route EEG pretraining: seven corpora on four routes "
                 "(montage x sampling rate) sharing one Transformer; (c) masked "
                 "dual-reconstruction loss on held-out subjects", y=1.02)
    w.save(fig, "fig_paper_pretraining", data,
           {"metric": metric, "datasets": datasets,
            "routes": {d: route_of[d] for d in datasets}})


def _mixed_channel(ex: Dict, route, mask_ratio: float) -> Optional[int]:
    """A valid channel with both masked and visible patches, the masked share
    closest to the mask ratio (the rule of fig_raw_waveform_reconstruction)."""
    mask, valid = ex["mask"], ex["valid"]
    n_patches = route.patches_per_channel
    n_masked = mask.sum(axis=1)
    ok = valid.all(axis=1)
    mixed = np.where(ok & (n_masked > 0) & (n_masked < n_patches))[0]
    if mixed.size:
        want = max(1, min(n_patches - 1, int(round(mask_ratio * n_patches))))
        return int(mixed[np.argmin(np.abs(n_masked[mixed] - want))])
    fallback = np.where(n_masked > 0)[0]
    return int(fallback[np.argmax(n_masked[fallback])]) if fallback.size else None


def _median_window(rows, grid, ranks, i: int) -> Optional[int]:
    """The surveyed window of route ``rows[i]`` whose masked raw NMSE is the
    median of the survey: neither the best example nor the worst."""
    rid = rows[i][0]
    cand = [(ranks.get((rid, k), {}).get("raw"), k) for k in range(len(grid))
            if grid[k][i] is not None]
    cand = [(r, k) for r, k in cand if r is not None]
    if not cand:
        return None
    n = len(cand)
    target = (n + 1) // 2
    return min(cand, key=lambda t: abs(t[0] - target))[1]


def fig_paper_reconstruction(w: FigureWriter, rows, grid, ranks, model, datasets,
                             mask_seed: int, device):
    """(a) masked reconstruction of one channel per route, on the window whose
    masked raw NMSE is the median of the survey; (b) the masks of the same
    windows; (c) frequency-guided selection: patch importance of the first
    route's window against the patches that were masked."""
    if not rows or not grid:
        return
    picks = []
    for i, (rid, dsid) in enumerate(rows):
        k = _median_window(rows, grid, ranks, i)
        if k is None:
            continue
        ex = grid[k][i]
        ch = _mixed_channel(ex, ROUTES[rid], model.mask_ratio)
        if ch is not None:
            picks.append((i, rid, dsid, k, ex, ch))
    if not picks:
        return
    n = len(picks)
    fig = plt.figure(figsize=(7.2, 1.05 * n + 0.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.3,
                             left=0.035, right=0.99, top=0.9, bottom=0.15)
    left = outer[0].subgridspec(n, 1, hspace=0.18)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55)
    data, meta_rows = {}, []

    for j, (i, rid, dsid, k, ex, ch) in enumerate(picks):
        route = ROUTES[rid]
        pt, n_p = route.patch_t, route.patches_per_channel
        target, pred = ex["target_raw"][ch], ex["pred_raw"][ch]
        m = _expand(ex["mask"][ch:ch + 1], pt)[0]
        t = np.arange(target.size) / route.sampling_rate
        tm, pm = target[m].astype(np.float64), pred[m].astype(np.float64)
        pc, tc = pm - pm.mean(), tm - tm.mean()
        den = float(np.linalg.norm(pc) * np.linalg.norm(tc))
        corr = float(np.dot(pc, tc) / den) if den > 0 else float("nan")
        nmse = float(np.mean((pm - tm) ** 2) / max(float(np.mean(tm ** 2)), 1e-12))
        ax = fig.add_subplot(left[j])
        for pi in range(n_p):
            if ex["mask"][ch, pi]:
                ax.axvspan(pi * pt / route.sampling_rate, (pi + 1) * pt / route.sampling_rate,
                           color="0.88", lw=0, zorder=0)
        ax.plot(t, target, color="0.15", lw=0.6, zorder=2, label="Target")
        ax.plot(t, np.where(m, pred, np.nan), color=ROUTE_COLOR[rid], lw=0.8, zorder=3,
                label="Reconstruction (masked)")
        ax.set_xlim(float(t[0]), float(t[-1]))
        lo = float(np.nanmin(target)); hi = float(np.nanmax(target))
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.42 * (hi - lo))   # a band for the labels
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        box = {"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85}
        ax.text(0.006, 0.95, f"{DATASET_LABEL.get(dsid, dsid)} ({rid.split('_')[0]}), "
                f"{_slot_name(dsid, route, ch)}", transform=ax.transAxes, ha="left",
                va="top", fontsize=6.5, bbox=box, zorder=5)
        ax.text(0.994, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.5, bbox=box, zorder=5)
        if j == n - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
        if j == 0:
            _panel_letter(ax, "a", "Masked raw reconstruction, one channel per route")
            handles, labels_ = ax.get_legend_handles_labels()
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], color="0.15", lw=0.8),
                       Line2D([], [], color="0.45", lw=1.4),
                       matplotlib.patches.Patch(color="0.88")]
            fig.legend(handles, ["Target (z-scored EEG)", "Reconstruction (route colour)",
                                 "Masked patch"], loc="lower left", ncol=3, frameon=False,
                       bbox_to_anchor=(0.03, 0.0), handlelength=1.6, columnspacing=1.4)
        rk = ranks.get((rid, k), {})
        data[f"{rid}_time"], data[f"{rid}_target"] = t, target
        data[f"{rid}_pred"], data[f"{rid}_mask"] = pred, ex["mask"][ch]
        meta_rows.append({"route_id": rid, "dataset_id": dsid,
                          "channel": _slot_name(dsid, route, ch),
                          "window_index": ex["window_index"],
                          "subject_id": ex["subject_id"],
                          "recording_id": ex["recording_id"], "mask_seed": mask_seed,
                          "raw_corr": corr, "raw_nmse": nmse,
                          "window_raw_nmse_rank": rk.get("raw"),
                          "n_windows_surveyed": rk.get("n")})

    # (b) the masks of the same windows, one small map per route
    mb = right[0].subgridspec(1, n, wspace=0.25)
    for j, (i, rid, dsid, k, ex, ch) in enumerate(picks):
        ax = fig.add_subplot(mb[j])
        shown = np.where(ex["valid"], ex["mask"].astype(float), np.nan)
        cmap = matplotlib.colors.ListedColormap(["white", ROUTE_COLOR[rid]])
        cmap.set_bad("0.85")
        ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
        ax.set_title(f"{ROUTES[rid].n_channels} ch", fontsize=7, pad=2)
        if j == 0:
            ax.set_ylabel("Channel", fontsize=7)
            ax.text(-0.07, 1.16, "b", transform=ax.transAxes, fontsize=10,
                    fontweight="bold", ha="right", va="baseline")
            ax.text(0.0, 1.16, "Masks of the same windows", transform=ax.transAxes,
                    fontsize=8, ha="left", va="baseline")
        data[f"{rid}_mask_map"] = ex["mask"]
    fig.text(0.5 * (outer[1].get_position(fig).x0 + outer[1].get_position(fig).x1),
             right[0].get_position(fig).y0 - 0.03, "Time patch (0.5 s)", ha="center",
             va="top", fontsize=7)

    # (c) frequency-guided selection on the first route's window
    i0, rid0, dsid0, k0, ex0, _ = picks[0]
    route0 = ROUTES[rid0]
    ds0 = datasets[dsid0]
    with torch.no_grad():
        batch = collate_windows([ds0[ex0["window_index"]]])
        spec = model.wavelet_frontends[rid0](batch["x"].to(device))
        tok = model.patch_embed_by_rate[route0.rate_key](spec.unsqueeze(1))
        imp = torch.abs(torch.fft.rfft(tok, dim=1)).sum(dim=2)
        imp = torch.nn.functional.interpolate(imp.unsqueeze(1), size=tok.shape[1],
                                              mode="linear", align_corners=True)
        imp = imp.squeeze(1)[0].cpu().numpy()
    sel = ex0["mask"].reshape(-1).astype(bool)
    valid0 = ex0["valid"].reshape(-1).astype(bool)
    imp, sel = imp[valid0], sel[valid0]
    order = np.argsort(-imp)
    ranks_pct = np.empty_like(order, dtype=float)
    ranks_pct[order] = 100.0 * np.arange(order.size) / max(order.size - 1, 1)
    ax = fig.add_subplot(right[1])
    bins = np.linspace(0, 100, 11)
    masked_share = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        inb = (ranks_pct >= lo) & (ranks_pct < hi if hi < 100 else ranks_pct <= hi)
        masked_share.append(100.0 * sel[inb].mean() if inb.any() else np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers, masked_share, width=9.0, color=ROUTE_COLOR[rid0])
    ax.axhline(100 * model.mask_ratio, color="0.2", lw=0.8, ls="--")
    ax.text(99, 100 * model.mask_ratio + 2, f"uniform ({100 * model.mask_ratio:.0f}%)",
            ha="right", va="bottom", fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Patch importance rank (%)")
    ax.set_ylabel("Patches masked (%)")
    _panel_letter(ax, "c", "Frequency-guided masking")
    top = ranks_pct < 30
    data.update({"importance": imp, "selected": sel, "importance_rank_pct": ranks_pct,
                 "masked_share_by_decile": np.asarray(masked_share)})
    fig.suptitle("Masked dual reconstruction on held-out subjects. (a) One channel per "
                 "route, on the window whose masked raw NMSE is the median of "
                 f"{len(grid)} surveyed (not a selected best example); the "
                 "reconstruction is drawn on masked patches only, which alone are "
                 "supervised. (b) The masks of the same windows (coloured: masked "
                 "patch; white: visible; grey: no electrode in that slot). (c) Share of patches masked by importance decile, "
                 f"{DATASET_LABEL.get(dsid0, dsid0)}.", y=1.02, fontsize=7)
    w.save(fig, "fig_paper_reconstruction", data,
           {"rows": meta_rows, "n_windows_surveyed": len(grid),
            "selection_rule": "per route, the surveyed window with the median masked "
                              "raw NMSE rank; the channel with masked and visible "
                              "patches whose masked share is closest to the mask ratio",
            "masked_share_top30pct": float(100.0 * sel[top].mean()) if top.any() else None,
            "mask_ratio": float(model.mask_ratio)})


# --------------------------------------------------------------------------- #
# Journal composites, drawn at their final printed size.
#
# The working figures above are drawn on large canvases (21 in wide for the
# eight-panel grid) and then scaled to a column, which is how 8 pt text arrives
# at 3 pt. These are drawn at the width they will be printed -- a double column
# is 7.16 in (IEEE) -- so the point sizes below are the point sizes on the
# page. They carry no figure-level title, no per-row metric lines, no window
# indices and no survey ranks: those sit in figure_metadata/<name>.json, where
# the caption is written from. The honesty rules of reconstruction_panels
# (shared colour limits, normalised spec target, masked-only error) are
# unchanged, because the panels come from the same function.
# --------------------------------------------------------------------------- #

#: IEEE double column, in inches.
PAPER_WIDTH = 7.16
PAPER_FONT = 8.0          # body: tick labels, row labels, legends
PAPER_TITLE = 8.5         # column and panel titles
PAPER_LETTER = 10.0       # panel letters

#: The journal composites' palette: a muted blue-violet family with one warm
#: accent, after the pastel scientific palettes of recent Nature covers. Every
#: categorical set below was checked for colour-vision deficiency (Machado 2009
#: protan/deutan/tritan at severity 1, OKLab distance): all pairs of the stage
#: set clear 15 (normal) and 8 (CVD); the scale ramp is ordinal and also carries
#: its order in the legend and in marker shape.
PAPER = {
    "indigo": "#3F4280", "blue": "#7F97C2", "pale_blue": "#BBD0E8",
    "lavender": "#8A84BF", "mauve": "#C3609E", "gold": "#DCA94F",
    "grey": "#A9ADB8", "light": "#EEF1F6", "ink": "#2E3440", "rule": "#6B7080",
}
#: Light-to-dark sequential ramp, for ranks and magnitudes.
PAPER_SEQ = ["#EEF1F6", "#BBD0E8", "#8A84BF", "#3F4280"]


#: One palette hue per route, for the journal composites. Rows of the
#: reconstruction figures are never adjacent in colour, so the four only need
#: to differ from the ink and from each other across a row label.
PAPER_ROUTE_COLOR = {"E19_256": PAPER["indigo"], "E32_512": PAPER["mauve"],
                     "E64_256": PAPER["blue"], "E128_512": PAPER["gold"]}


def paper_cmap(name: str = "paper_seq"):
    return matplotlib.colors.LinearSegmentedColormap.from_list(name, PAPER_SEQ)


def paper_cmap_div():
    """Diverging: amber through white to deep indigo, for signed signals.

    The ends are darker than the palette's gold and indigo: a z-scored window
    sits mostly near zero, so the ends carry the contrast, and at the
    palette's own lightness the heatmaps read as washed out.
    """
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "paper_div", ["#9A6214", PAPER["gold"], "#FFFFFF", PAPER["blue"], "#23264F"])


def paper_cmap_err():
    """Sequential white-to-mauve-to-plum, for the error maps (never indigo,
    which the signed panels beside them use for positive values). Mauve is
    reached a third of the way up, so small errors still show."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "paper_err", [(0.0, "#FFFFFF"), (0.35, PAPER["mauve"]), (0.7, "#7A3466"),
                      (1.0, "#33132B")])


def _paper_rc():
    """rcParams for the journal composites; use as a context manager."""
    return plt.rc_context({
        "font.size": PAPER_FONT, "axes.titlesize": PAPER_TITLE,
        "axes.labelsize": PAPER_FONT, "legend.fontsize": PAPER_FONT,
        "xtick.labelsize": PAPER_FONT, "ytick.labelsize": PAPER_FONT,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    })


def _letter(fig, x, y, ch):
    """A panel letter at figure coordinates, bold, on the figure itself."""
    fig.text(x, y, ch, fontsize=PAPER_LETTER, fontweight="bold",
             ha="left", va="top", gid="paper-keep")


def _paper_picks(rows, grid, ranks, mask_ratio):
    """Per route, the median-NMSE surveyed window and its mixed channel.

    The same selection rule as fig_paper_reconstruction, so the three journal
    composites describe the same four windows: neither the best example of
    the survey nor the worst.
    """
    picks = []
    for i, (rid, dsid) in enumerate(rows):
        k = _median_window(rows, grid, ranks, i)
        if k is None:
            continue
        ex = grid[k][i]
        ch = _mixed_channel(ex, ROUTES[rid], mask_ratio)
        if ch is not None:
            picks.append((i, rid, dsid, k, ex, ch))
    return picks


def _route_row_label(rid: str, dsid: str) -> str:
    r = ROUTES[rid]
    return f"{DATASET_LABEL.get(dsid, dsid)}\n{r.n_channels} ch, {r.sampling_rate:g} Hz"


def fig_paper_mask_reconstruction(w: FigureWriter, rows, grid, ranks, model,
                                  mask_seed: int):
    """fig_mask_reconstruction at journal size: four routes, eight panels each.

    Two groups of four that read the same way -- target, what the frontend was
    handed, the composite, and the error on the masked patches alone -- for the
    raw head (a) and the folded-wavelet head (b). The only number on the
    figure is the masked correlation of each head, in its error panel; the
    losses, NMSE, window indices and survey ranks go to the metadata.
    """
    picks = _paper_picks(rows, grid, ranks, model.mask_ratio)
    if not picks:
        return
    n = len(picks)
    col_titles = ["Target", "Masked input", "Reconstruction", "|Error|"]
    with _paper_rc():
        row_h = 0.98
        fig = plt.figure(figsize=(PAPER_WIDTH, row_h * n + 0.72))
        outer = fig.add_gridspec(1, 2, wspace=0.08, left=0.085, right=0.995,
                                 top=1 - 0.42 / (row_h * n + 0.72),
                                 bottom=0.34 / (row_h * n + 0.72))
        grids = [outer[g].subgridspec(n, 4, wspace=0.06, hspace=0.12)
                 for g in range(2)]
        data, meta_rows = {}, []
        for r, (i, rid, dsid, k, ex, ch) in enumerate(picks):
            route = ROUTES[rid]
            pt, n_patches = route.patch_t, route.patches_per_channel
            panels = reconstruction_panels(ex, route)
            T = ex["target_raw"].shape[1]
            mt = ex["metrics"]
            for c, (arr, _title, cmap, vmin, vmax) in enumerate(panels):
                g, cc = divmod(c, 4)
                ax = fig.add_subplot(grids[g][r, cc])
                # The working figure's RdBu_r / magma, in the journal palette:
                # gold-white-indigo for the signed panels, white-mauve for the
                # error maps. Limits are unchanged.
                cmap = paper_cmap_err() if cmap == "magma" else paper_cmap_div()
                ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest",
                          vmin=vmin, vmax=vmax)
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.5)
                    sp.set_color(PAPER["rule"])
                if r == 0:
                    ax.set_title(col_titles[cc], pad=3)
                if r == n - 1:
                    secs = route.window_seconds
                    ax.set_xticks([0, T / 2, T - 1])
                    ax.set_xticklabels(["0", "", f"{secs:g}"])
                    labs = ax.get_xticklabels()
                    labs[0].set_ha("left")
                    labs[-1].set_ha("right")
                else:
                    ax.set_xticks([])
                if c == 0:
                    ax.set_ylabel(_route_row_label(rid, dsid), labelpad=4)
                if cc == 3:
                    corr = (mt["masked_raw_corr"] if g == 0
                            else mt["masked_spec_corr"])
                    import matplotlib.patheffects as pe
                    ax.text(0.96, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                            ha="right", va="top", color=PAPER["ink"], fontsize=PAPER_FONT,
                            path_effects=[pe.withStroke(linewidth=1.8,
                                                        foreground="white")])
            valid_rows = _expand(ex["valid"], pt)
            raw_lim = _sym_limits(ex["target_raw"], valid_rows)
            spec_lim = _sym_limits(ex["target_spec"], valid_rows)
            ratio = ex["mask"].sum() / max(1, ex["valid"].sum())
            rk = ranks.get((rid, k), {})
            for key in ("target_raw", "pred_raw", "target_spec", "pred_spec",
                        "mask", "valid"):
                data[f"{rid}_{key}"] = ex[key]
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"],
                              "survey_index": k, "mask_seed": mask_seed,
                              "actual_mask_ratio": float(ratio),
                              "raw_color_limit": raw_lim,
                              "spec_color_limit": spec_lim,
                              "spec_target_is_normalised": ex["normalize_spec_target"],
                              "raw_nmse_rank": rk.get("raw"),
                              "spec_nmse_rank": rk.get("spec"),
                              "n_windows_surveyed": rk.get("n"), **mt})
        # Group titles and letters, on the figure so they share one baseline.
        for g, (ch_, title) in enumerate((("a", "Raw-EEG head"),
                                          ("b", "Folded-wavelet head"))):
            box = outer[g].get_position(fig)
            _letter(fig, box.x0 - 0.012, 0.995, ch_)
            fig.text(0.5 * (box.x0 + box.x1), 0.995, title, ha="center", va="top",
                     fontsize=PAPER_TITLE + 0.5, fontweight="bold", gid="paper-keep")
        fig.text(0.5 * (outer[0].get_position(fig).x0 + outer[1].get_position(fig).x1),
                 0.012, "Time (s)", ha="center", va="bottom", gid="paper-keep")
        w.save(fig, "fig_paper_mask_reconstruction", data,
               {"examples": meta_rows, "n_windows_surveyed": len(grid),
                "panels": [_sig(t) for t in RECONSTRUCTION_PANEL_TITLES],
                "selection_rule": "per route, the surveyed window with the median "
                                  "masked raw NMSE rank (fig_paper_reconstruction's)",
                "note": "spec panels use target_spec, the per-patch normalised "
                        "target the loss is computed against; target and "
                        "composite share colour limits taken from the target "
                        "(+-99.5th percentile over valid rows); the error panels "
                        "are masked valid patches only; the number in each error "
                        "panel is that head's masked correlation on the window"})


def fig_paper_raw_waveform(w: FigureWriter, rows, grid, ranks, model, mask_seed: int):
    """14_raw_waveform_reconstruction at journal size.

    One channel per route, the window and channel of the other two journal
    composites. The prediction is drawn on masked patches only; the composite
    line of the working figure is left out, since it repeats the target and
    the prediction where each is drawn. Under each trace, the absolute error
    on the masked stretches.
    """
    picks = _paper_picks(rows, grid, ranks, model.mask_ratio)
    if not picks:
        return
    n = len(picks)
    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 1.28 * n + 0.45))
        gs = fig.add_gridspec(n, 1, left=0.075, right=0.995,
                              top=1 - 0.3 / (1.28 * n + 0.45),
                              bottom=0.36 / (1.28 * n + 0.45), hspace=0.22)
        data, meta_rows = {}, []
        for j, (i, rid, dsid, k, ex, ch) in enumerate(picks):
            route = ROUTES[rid]
            pt, n_p = route.patch_t, route.patches_per_channel
            target, pred = ex["target_raw"][ch], ex["pred_raw"][ch]
            m = _expand(ex["mask"][ch:ch + 1], pt)[0]
            t = np.arange(target.size) / route.sampling_rate
            tm, pm = target[m].astype(np.float64), pred[m].astype(np.float64)
            resid = float(np.mean((pm - tm) ** 2)) if tm.size else float("nan")
            nmse = resid / max(float(np.mean(tm ** 2)), 1e-12) if tm.size else float("nan")
            pc, tc = pm - pm.mean(), tm - tm.mean()
            den = float(np.linalg.norm(pc) * np.linalg.norm(tc))
            corr = (float(np.dot(pc, tc) / den)
                    if den > 0 and np.isfinite(den) else float("nan"))
            err = np.abs(pred.astype(np.float64) - target.astype(np.float64))
            err[~m] = np.nan

            sub = gs[j].subgridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
            ax, ax_e = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            for pi in range(n_p):
                if ex["mask"][ch, pi]:
                    for a in (ax, ax_e):
                        a.axvspan(pi * pt / route.sampling_rate,
                                  (pi + 1) * pt / route.sampling_rate,
                                  color="#E9ECF3", lw=0, zorder=0)
            # Target in mid-grey, so the coloured prediction is the salient line
            # (indigo on ink was two dark lines).
            ax.plot(t, target, color=PAPER["rule"], lw=0.8, zorder=2,
                    label="Target (z-scored EEG)")
            ax.plot(t, np.where(m, pred, np.nan), color=PAPER_ROUTE_COLOR[rid], lw=1.1,
                    zorder=3, label="Prediction, masked patches only")
            ax.set_xlim(float(t[0]), float(t[-1]))
            lo, hi = float(np.nanmin(target)), float(np.nanmax(target))
            ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.45 * (hi - lo))
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
            ax.set_ylabel("z-score", labelpad=2)
            box = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.9}
            ax.text(0.006, 0.96,
                    f"{DATASET_LABEL.get(dsid, dsid)} ({route.n_channels} ch, "
                    f"{route.sampling_rate:g} Hz), {_slot_name(dsid, route, ch)}",
                    transform=ax.transAxes, ha="left", va="top", bbox=box, zorder=5)
            ax.text(0.994, 0.96, f"r = {corr:.2f}   NMSE = {nmse:.2f}",
                    transform=ax.transAxes, ha="right", va="top", bbox=box, zorder=5)
            ax_e.fill_between(t, 0, np.nan_to_num(err), where=m, color=PAPER["lavender"],
                              lw=0, zorder=2)
            ax_e.set_xlim(float(t[0]), float(t[-1]))
            top = float(np.nanmax(err)) if np.isfinite(np.nanmax(err)) else 1.0
            ax_e.set_ylim(0, top * 1.15)
            ax_e.set_yticks([0, round(top, 1)])
            ax_e.set_ylabel("|err|", labelpad=2)
            if j == n - 1:
                ax_e.set_xlabel("Time (s)")
            else:
                ax_e.tick_params(axis="x", labelbottom=False)
            if j == 0:
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], color=PAPER["rule"], lw=0.8),
                           Line2D([], [], color=PAPER["indigo"], lw=1.1),
                           matplotlib.patches.Patch(color="#E9ECF3")]
                fig.legend(handles, ["Target (z-scored EEG)",
                                     "Prediction, masked patches only (route colour)",
                                     "Masked patch"],
                           loc="upper left", ncol=3, frameon=False,
                           bbox_to_anchor=(0.06, 1.0), handlelength=1.6,
                           columnspacing=1.6, borderaxespad=0.2)
            rk = ranks.get((rid, k), {})
            data[f"{rid}_time"], data[f"{rid}_target"] = t, target
            data[f"{rid}_pred"], data[f"{rid}_mask"] = pred, ex["mask"][ch]
            data[f"{rid}_masked_error"] = err
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "channel": _slot_name(dsid, route, ch), "channel_slot": ch,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"], "survey_index": k,
                              "mask_seed": mask_seed, "raw_corr": corr, "raw_nmse": nmse,
                              "masked_patches": int(ex["mask"][ch].sum()),
                              "patches_per_channel": n_p,
                              "window_raw_nmse_rank": rk.get("raw"),
                              "n_windows_surveyed": rk.get("n")})
        w.save(fig, "fig_paper_raw_waveform", data,
               {"rows": meta_rows, "n_windows_surveyed": len(grid),
                "selection_rule": "per route, the surveyed window with the median "
                                  "masked raw NMSE rank; the channel with masked and "
                                  "visible patches whose masked share is closest to "
                                  "the mask ratio",
                "note": "z-scored preprocessed EEG, not raw EDF values and not the "
                        "folded wavelet representation; the prediction is shown on "
                        "masked patches only; r and NMSE are over the masked "
                        "samples of the channel drawn"})


# --------------------------------------------------------------------------- #
# Does the channel-NAME embedding know where the electrodes are?
#
# C1 is a lookup table indexed by the name's id. No coordinate ever enters the
# model, and the shared table is read by four routes whose slot orders differ,
# so the 2-D position embedding cannot tell the table where a name sits
# either. If the learned rows are nevertheless more alike for electrodes that
# are closer on the scalp, the model recovered the geometry from the signals
# alone -- neighbouring electrodes see the same sources through volume
# conduction, and a reconstruction objective that masks patches of one
# channel is served by knowing which other channels carry the same thing.
#
# The template coordinates come from MNE's standard_1005 montage (10-5 names,
# the TUEG/FACED/TDBRAIN/PhysioNet-MI/M3CV/HGD electrodes) and from its
# GSN-HydroCel-128 montage (HBN's E-numbers). The two are kept as separate
# families: the EGI net is a different set of positions and the model was
# given no bridge between the two naming systems, so a cross-family pair is
# not a test of anything. Distances are great-circle arcs on the sphere fitted
# to each family's positions, in degrees; an untrained table (the run's own
# initialisation, rebuilt from its seed) is the control.
# --------------------------------------------------------------------------- #

def _fit_sphere(P: np.ndarray):
    """Least-squares sphere through the points: ``(centre, radius)``."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(max(float(sol[3] + c @ c), 1e-12))
    return c, r


def _electrode_families(names):
    """``{family: {name: unit xyz}}`` for the names a template montage places.

    Case-insensitive against standard_1005 (the vocabulary spells halfway
    positions ``FFC5h``, the montage the same; TUEG's ``FP1`` has already been
    normalised to ``Fp1`` upstream) and exact against the EGI net.
    """
    import mne
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    egi = mne.channels.make_standard_montage("GSN-HydroCel-128").get_positions()["ch_pos"]
    std_up = {k.upper(): k for k in std}
    fam = {"10-5": {}, "EGI-128": {}}
    for n in names:
        if n.upper() in std_up:
            fam["10-5"][n] = np.asarray(std[std_up[n.upper()]], float)
        elif n in egi:
            fam["EGI-128"][n] = np.asarray(egi[n], float)
    out = {}
    for f, d in fam.items():
        if len(d) < 4:
            continue
        P = np.stack(list(d.values()))
        c, r = _fit_sphere(np.stack(list(
            (std if f == "10-5" else egi).values())))
        U = (P - c) / np.linalg.norm(P - c, axis=1, keepdims=True)
        out[f] = {"names": list(d), "unit": U, "xyz": P, "centre": c, "radius": r}
    return out


def _pairwise_arc_deg(U: np.ndarray) -> np.ndarray:
    G = np.clip(U @ U.T, -1.0, 1.0)
    return np.degrees(np.arccos(G))


def _cosine(E: np.ndarray) -> np.ndarray:
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        C = En @ En.T
    if not np.all(np.isfinite(C)):
        raise SystemExit("channel embedding cosine matrix is not finite")
    return np.clip(C, -1.0, 1.0)


def _trained_channel_names(model, datasets):
    """The electrode names whose rows pretraining actually updated.

    From the corpora's montages when they are open (the exact set: a padded
    slot's row is never touched), otherwise the union of the route slot lists,
    which is the same set whenever every slot is filled by some corpus
    (tests/test_eeg_c1_pretrain.py::test_12i).
    """
    ids = set()
    for ds in datasets.values():
        mo = ds.montage()
        cid = mo["channel_ids"][mo["valid_channel_mask"]].tolist()
        ids.update(int(i) for i in cid if i > 1)
    if not ids:
        from physiowave.eeg_c1.routes import SLOTS_128_HGD
        names = set(SLOTS_128_HGD)
        for r in ROUTES.values():
            names.update(r.slots)
        from channel_embedding import CHANNEL_TO_ID
        ids = {CHANNEL_TO_ID[n] for n in names if n in CHANNEL_TO_ID}
    vocab = model.channel_encoder.vocab_size
    ids = sorted(i for i in ids if i < vocab)
    return ids, [CHANNEL_VOCAB[i] for i in ids]


def _binned(x, y, edges):
    """Mean of ``y`` and its 95% CI per bin of ``x``; NaN where a bin is empty."""
    mids, means, half = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        mids.append(0.5 * (lo + hi))
        if sel.sum() < 2:
            means.append(np.nan)
            half.append(np.nan)
            continue
        v = y[sel]
        means.append(float(v.mean()))
        half.append(float(1.96 * v.std(ddof=1) / math.sqrt(v.size)))
    return np.asarray(mids), np.asarray(means), np.asarray(half)


def _classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    """Torgerson's metric MDS of a distance matrix: ``[n, k]``.

    Closed form (double-centred squared distances, top eigenvectors), so the
    layout is deterministic and needs nothing beyond numpy.
    """
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:k]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def fig_paper_channel_topology(w: FigureWriter, model, init_model, datasets,
                               n_perm: int = 2000, seed: int = 0):
    """(a) cosine similarity of two electrodes' name embeddings against their
    great-circle distance on the scalp, every pair of trained electrodes, with
    the untrained table as control; (b) for six seed electrodes, every trained
    electrode coloured by its embedding's similarity rank to the seed, with the
    seed joined to its five most similar. The nearest-neighbour statistic
    (mean scalp distance to the 5 most similar embeddings, against a shuffled
    null and the untrained table) and a Procrustes r of an MDS layout are
    computed for the text and go to the metadata only."""
    if SIGNAL != "EEG" or model.channel_encoder is None:
        return
    try:
        import mne  # noqa: F401
        from scipy.stats import spearmanr
        from scipy.spatial import procrustes
    except ImportError as exc:
        print(f"  skipping fig_paper_channel_topology: {exc}")
        return
    ids, names = _trained_channel_names(model, datasets)
    fams = _electrode_families(names)
    if "10-5" not in fams:
        print("  skipping fig_paper_channel_topology: no 10-5 electrode has "
              "a template coordinate")
        return
    E_all = model.channel_encoder.id_embed.weight.detach().float().cpu().numpy()
    E0_all = init_model.channel_encoder.id_embed.weight.detach().float().cpu().numpy()
    id_of = {CHANNEL_VOCAB[i]: i for i in ids}
    rng = np.random.default_rng(seed)

    stats, data = {}, {}
    curves = {}
    for f, d in fams.items():
        rows = [id_of[n] for n in d["names"]]
        E, E0 = E_all[rows], E0_all[rows]
        D = _pairwise_arc_deg(d["unit"])
        iu = np.triu_indices(len(rows), 1)
        x = D[iu]
        for tag, Emb in (("trained", E), ("init", E0)):
            C = _cosine(Emb)
            y = C[iu]
            rho = float(spearmanr(x, y).correlation)
            # Null: rows shuffled among electrodes, distances kept.
            null = np.empty(n_perm)
            for p in range(n_perm):
                perm = rng.permutation(len(rows))
                null[p] = spearmanr(x, C[np.ix_(perm, perm)][iu]).correlation
            pval = float((np.sum(null <= rho) + 1) / (n_perm + 1))   # one-sided, rho < 0
            stats[f"{f}/{tag}"] = {"n_electrodes": len(rows), "n_pairs": int(x.size),
                                   "spearman_rho": rho, "perm_p_one_sided": pval,
                                   "null_rho_mean": float(null.mean()),
                                   "null_rho_sd": float(null.std(ddof=1))}
            curves[(f, tag)] = (x, y)
            data[f"{f}_{tag}_cosine"] = C
        data[f"{f}_arc_deg"] = D
        data[f"{f}_names"] = np.asarray(d["names"])
        data[f"{f}_xyz"] = d["xyz"]

    main_f = "10-5"
    d = fams[main_f]
    rows = [id_of[n] for n in d["names"]]
    C = _cosine(E_all[rows])
    # (b) seeds: the midline from front to back and the two temporal sides --
    # chosen a priori, and replaced by the nearest trained electrode if absent.
    want = ["Fz", "Cz", "Pz", "Oz", "T7", "T8"]
    seeds = []
    for s in want:
        if s in d["names"]:
            seeds.append(s)
        else:
            import mne
            std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
            if s in std:
                v = (np.asarray(std[s]) - d["centre"])
                v /= np.linalg.norm(v)
                seeds.append(d["names"][int(np.argmax(d["unit"] @ v))])
    # (c) MDS of the cosine distance, aligned to the true 2-D layout.
    import mne
    info = mne.create_info(d["names"], 100.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     match_case=False)
    from mne.channels.layout import _find_topomap_coords
    pos2 = _find_topomap_coords(info, picks=None, sphere=None)
    Z = _classical_mds(1.0 - C)
    _, Zp, disparity = procrustes(pos2, Z)
    proc_r = math.sqrt(max(0.0, 1.0 - disparity))
    # The same for the untrained table: the number the trained one is read against.
    Z0 = _classical_mds(1.0 - _cosine(E0_all[rows]))
    proc_r0 = math.sqrt(max(0.0, 1.0 - procrustes(pos2, Z0)[2]))
    stats["procrustes_r"] = {"trained": proc_r, "init": proc_r0}
    data.update({"mds": Z, "mds_aligned": Zp, "layout_2d": pos2, "seeds": np.asarray(seeds)})

    # (c) Local structure: for each electrode, the mean scalp distance to the
    # k electrodes whose NAME embeddings are most like its own. Panel (a) tests
    # a global trend over all pairs; this asks whether the embedding's nearest
    # neighbours are scalp neighbours. Null: rows shuffled among electrodes, as
    # in (a). The floor is each electrode's k true scalp neighbours.
    k_nn = 5

    def _knn_dist(Cm, D):
        Cm = Cm.copy()
        np.fill_diagonal(Cm, -np.inf)
        nn = np.argsort(-Cm, axis=1)[:, :k_nn]
        return float(np.take_along_axis(D, nn, axis=1).mean())

    knn = {}
    for f, d_ in fams.items():
        rows_f = [id_of[n] for n in d_["names"]]
        Dm = _pairwise_arc_deg(d_["unit"])
        Ct, Ci = _cosine(E_all[rows_f]), _cosine(E0_all[rows_f])
        obs, ini = _knn_dist(Ct, Dm), _knn_dist(Ci, Dm)
        null = np.empty(n_perm)
        for p_ in range(n_perm):
            perm = rng.permutation(len(rows_f))
            null[p_] = _knn_dist(Ct[np.ix_(perm, perm)], Dm)
        Df = Dm.copy()
        np.fill_diagonal(Df, np.inf)
        floor = float(np.sort(Df, axis=1)[:, :k_nn].mean())
        knn[f] = {"k": k_nn, "trained_deg": obs, "init_deg": ini, "scalp_floor_deg": floor,
                  "null_mean_deg": float(null.mean()),
                  "null_p2_5_deg": float(np.percentile(null, 2.5)),
                  "null_p97_5_deg": float(np.percentile(null, 97.5)),
                  "perm_p_one_sided": float((np.sum(null <= obs) + 1) / (n_perm + 1))}
        data[f"{f}_knn_null_deg"] = null
    stats["knn_scalp_distance"] = knn

    with _paper_rc():
        # (a) is the global test over all pairs; (b) shows the local structure
        # the kNN statistic in the metadata measures. An MDS layout and a dot
        # plot of that statistic were drawn in earlier drafts and dropped.
        fig = plt.figure(figsize=(PAPER_WIDTH, 3.25))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.2,
                                 left=0.075, right=0.99, top=0.9, bottom=0.14)
        # (a)
        ax = fig.add_subplot(outer[0])
        edges = np.linspace(0, 180, 13)
        style = {("10-5", "trained"): (PAPER["indigo"], "-", "10-5 names, trained"),
                 ("EGI-128", "trained"): (PAPER["mauve"], "-", "EGI names, trained"),
                 ("10-5", "init"): (PAPER["grey"], "--", "Untrained table")}
        x, y = curves[(main_f, "trained")]
        ax.scatter(x, y, s=2.5, color=PAPER["pale_blue"], alpha=0.35, lw=0,
                   rasterized=True, zorder=1)
        for key, (col, ls, lab) in style.items():
            if key not in curves:
                continue
            xm, ym, hm = _binned(*curves[key], edges)
            ok = np.isfinite(ym)
            ax.errorbar(xm[ok], ym[ok], yerr=hm[ok], color=col, ls=ls, lw=1.3,
                        marker="o", ms=3, capsize=1.5, elinewidth=0.7, label=lab, zorder=3)
        ax.axhline(0, color=PAPER["rule"], lw=0.5, zorder=0)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("Scalp distance between electrodes (°)")
        ax.set_ylabel("Cosine similarity of name embeddings")

        def _rho_line(f, lab):
            s_ = stats.get(f"{f}/trained")
            if s_ is None:
                return None
            p_ = s_["perm_p_one_sided"]
            # The smallest p a permutation test can report is 1/(n_perm + 1).
            ptxt = f"p < {1e-3:g}" if p_ < 1e-3 else f"p = {p_:.3f}"
            return f"{lab}: ρ = {s_['spearman_rho']:.2f}, {ptxt}"
        lines = [t for t in (_rho_line("10-5", "10-5"), _rho_line("EGI-128", "EGI"))
                 if t]
        lines.append(f"untrained: ρ = {stats[f'{main_f}/init']['spearman_rho']:.2f}")
        ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right",
                va="top", fontsize=PAPER_FONT - 0.5, linespacing=1.25, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
        ax.legend(loc="lower left", frameon=False, handlelength=1.8,
                  fontsize=PAPER_FONT - 0.5, borderaxespad=0.3)
        _letter(fig, 0.005, 0.99, "a")
        # (b)
        # Six seeds in two rows of three. Each head shows every trained 10-5
        # electrode as a dot coloured by the percentile rank of its embedding's
        # similarity to the seed (dark = most alike), and joins the seed to its
        # five most similar electrodes. No interpolation: 130 rows interpolated
        # over the scalp gave a mottled field on any colour scale, whereas the
        # dots and links show the statistic (the mean length of those five
        # links, stats["knn_scalp_distance"]) as it is.
        from scipy.stats import rankdata
        box1 = outer[1].get_position(fig)
        n_col, n_row = 3, int(math.ceil(len(seeds) / 3))
        gap_x, gap_y = 0.012, 0.05
        side_w = (box1.x1 - box1.x0 - gap_x * (n_col - 1)) / n_col
        side_h = side_w * PAPER_WIDTH / fig.get_figheight()
        avail_h = (box1.y1 - box1.y0 - gap_y * (n_row - 1) - 0.09) / n_row  # 0.09: colourbar
        if side_h > avail_h:
            side_h = avail_h
            side_w = side_h * fig.get_figheight() / PAPER_WIDTH
        x_start = 0.5 * (box1.x0 + box1.x1) - 0.5 * (n_col * side_w + gap_x * (n_col - 1))
        cmap = paper_cmap()
        white = matplotlib.colors.ListedColormap(["white"])
        Dmain = _pairwise_arc_deg(d["unit"])
        sc = None
        link_deg = {}
        for j, s in enumerate(seeds):
            r_, c_ = divmod(j, n_col)
            axb = fig.add_axes([x_start + c_ * (side_w + gap_x),
                                box1.y1 - side_h - r_ * (side_h + gap_y), side_w, side_h],
                               label=f"free-head-{j}")
            mne.viz.plot_topomap(np.zeros(len(rows)), info, axes=axb, show=False,
                                 cmap=white, vlim=(-1, 1), contours=0, sensors=False,
                                 outlines="head", sphere=None)
            for line in axb.lines:                       # the head outline, in ink
                line.set_color(PAPER["rule"])
                line.set_linewidth(0.7)
            si = d["names"].index(s)
            others = np.arange(len(rows)) != si
            rank = np.zeros(len(rows))
            rank[others] = 100.0 * (rankdata(C[si][others]) - 1) / max(others.sum() - 1, 1)
            top = np.argsort(-np.where(others, C[si], -np.inf))[:k_nn]
            for t in top:
                axb.plot([pos2[si, 0], pos2[t, 0]], [pos2[si, 1], pos2[t, 1]],
                         color=PAPER["mauve"], lw=1.1, alpha=0.9, zorder=3,
                         solid_capstyle="round")
            sc = axb.scatter(pos2[others, 0], pos2[others, 1], c=rank[others], cmap=cmap,
                             vmin=0, vmax=100, s=11, lw=0.25, edgecolor="white", zorder=4)
            axb.scatter(pos2[si, 0], pos2[si, 1], s=55, marker="*", color=PAPER["gold"],
                        edgecolor=PAPER["ink"], linewidth=0.5, zorder=6)
            axb.set_title(s, pad=1.5)
            link_deg[s] = float(Dmain[si, top].mean())
            data[f"rank_{s}"] = rank
            data[f"top{k_nn}_{s}"] = np.asarray([d["names"][t] for t in top])
        stats["seed_link_mean_deg"] = link_deg
        cb_y = box1.y1 - n_row * side_h - (n_row - 1) * gap_y - 0.075
        cax = fig.add_axes([box1.x0 + 0.18 * (box1.x1 - box1.x0), cb_y,
                            0.64 * (box1.x1 - box1.x0), 0.03])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 50, 100])
        cb.set_label("Embedding similarity to the starred electrode (rank, %)",
                     labelpad=1.5)
        cb.outline.set_linewidth(0.5)
        # The link legend at the top right of the panel, beside the letter and
        # clear of the colourbar's tick labels, which it sat on when placed below.
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=PAPER["mauve"], lw=1.1),
                    Line2D([], [], color=PAPER["gold"], marker="*", ls="", ms=7,
                           markeredgecolor=PAPER["ink"], markeredgewidth=0.5)],
                   [f"Seed to its {k_nn} most similar embeddings", "Seed electrode"],
                   loc="upper right", bbox_to_anchor=(box1.x1, 1.0), ncol=2, frameon=False,
                   fontsize=PAPER_FONT - 1, handlelength=1.6, columnspacing=1.2,
                   handletextpad=0.5, borderaxespad=0.0)
        _letter(fig, box1.x0 - 0.03, 0.99, "b")
        w.save(fig, "fig_paper_channel_topology", data,
               {"stats": stats, "seeds": seeds,
                "n_trained_electrodes": {f: len(v_["names"]) for f, v_ in fams.items()},
                "trained_names_source": "corpus montages" if datasets else "route slot union",
                "coordinates": "MNE standard_1005 (10-5 names) and GSN-HydroCel-128 "
                               "(EGI names) template montages; great-circle distance "
                               "on the sphere fitted to each family; the model was "
                               "given no coordinate",
                "permutation_null": "embedding rows shuffled among electrodes, "
                                    f"{n_perm} draws, one-sided (rho < 0)",
                "procrustes": "2-D metric MDS of 1 - cosine, aligned to the "
                              "montage's 2-D topomap layout; r = sqrt(1 - disparity); "
                              "reported, not drawn",
                "panel_b": "six seeds; every trained 10-5 electrode coloured by "
                           "the percentile rank of its cosine similarity to the "
                           "seed; lines join the seed to its 5 most similar "
                           "embeddings (mean length per seed in "
                           "stats.seed_link_mean_deg)",
                "knn": "panel (c): mean over electrodes of the mean great-circle "
                       "distance to the k=5 electrodes with the most similar name "
                       "embedding; grey bar the 2.5-97.5th percentiles of the "
                       "row-shuffled null, open circle the untrained table, black "
                       "tick each electrode's 5 true scalp neighbours"})


#: Threads for the figures that run the model. A login node has many cores and
#: a small per-user thread quota, and torch defaults to one thread per core: the
#: figure that first touches the model then dies with "libgomp: Thread creation
#: failed: Resource temporarily unavailable", followed by "free(): corrupted
#: unsorted chunks" and a segfault, because libgomp failing part-way through a
#: parallel region leaves the allocator inconsistent. None of that names the
#: cause. PW_VIZ_THREADS overrides; the work here is small enough that four is
#: not the constraint.
DEFAULT_THREADS = 4


def _cap_threads() -> int:
    n = int(os.environ.get("PW_VIZ_THREADS") or DEFAULT_THREADS)
    n = max(1, min(n, os.cpu_count() or n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
    return n


def load_trained_model(ckpt_path: str, modality_arg: str = "auto"):
    """The checkpoint's model, built from its own config and weights.

    Also switches this module's registry (routes, datasets, channel names) to
    the run's modality. Returns ``(ck, cfg, mcfg, objective, modality,
    model_cls, model)``; scripts/draw_architecture_panels.py uses it too.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mcfg = cfg.get("model", {})
    device = torch.device("cpu")
    objective = resolve_eeg_c1_objective(cfg)

    modality = modality_arg
    if modality == "auto":
        modality = {"ecg_c1_moe": "ecg", "emg_c1_moe": "emg"}.get(
            cfg.get("trainer"), "eeg")
    model_cls, extra = MultiRouteEEGPretrainer, {}
    if modality == "ecg":
        use_ecg_registry()
        from physiowave.ecg_c1.model import MultiRouteECGPretrainer
        model_cls = MultiRouteECGPretrainer
        extra = {"lead_group_masking": bool(mcfg.get("lead_group_masking", True))}
    elif modality == "emg":
        use_emg_registry()
        from physiowave.emg_c1.model import MultiRouteEMGPretrainer
        model_cls = MultiRouteEMGPretrainer
    print(f"  modality: {modality}")

    model = model_cls(**extra,
        embed_dim=int(mcfg.get("embed_dim", 384)),
        depth=int(mcfg.get("depth", 6)), num_heads=int(mcfg.get("num_heads", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
        qk_norm=bool(mcfg.get("qk_norm", True)),
        max_level=int(mcfg.get("max_level", 3)),
        wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
        wavelet_names=mcfg.get("wavelet_names"),
        wave_init_mode=mcfg.get("wave_init_mode", "pad"),
        fold_synthesis=int(mcfg.get("fold_synthesis", 3)),
        fold_gamma=float(mcfg.get("fold_gamma", 0.1)),
        mask_ratio=float(mcfg.get("mask_ratio", 0.5)),
        # Not cosmetic. normalize_spec_target decides what target_spec IS, and
        # mask_before_frontend decides whether the online view was corrupted at
        # all; defaulting both would render an ablation run as though it had
        # been trained the standard way.
        mask_before_frontend=bool(objective["mask_before_frontend"]),
        normalize_spec_target=bool(objective["normalize_spec_target"]),
        mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
        use_separate_channel=bool(mcfg.get("use_separate_channel", True)),
        masking_strategy=mcfg.get("masking_strategy", "frequency_guided"),
        importance_ratio=float(mcfg.get("importance_ratio", 0.6)),
        channel_encoding=mcfg.get("channel_encoding", "id"),
        channel_injection=mcfg.get("channel_injection", "token"),
        channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
        channel_vocab_size=ck.get("channel_vocab_size"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    return ck, cfg, mcfg, objective, modality, model_cls, model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--paper", action="store_true",
                   help="for a manuscript: no figure-level title or notes "
                        "(kept in figure_metadata/<name>.json as figure_text, "
                        "for the caption) and no corner stamp; panel titles, "
                        "axes and legends stay")
    p.add_argument("--no-stamp", action="store_true",
                   help="leave out the grey checkpoint/epoch line in each "
                        "figure's corner (for a paper); figure_metadata/ "
                        "still records it")
    p.add_argument("--format", default="svg,pdf",
                   help="svg, pdf, png, or several comma-separated "
                        "(default svg,pdf: vector, fonts editable)")
    p.add_argument("--manifest", default=None,
                   help="override the manifest the run recorded")
    p.add_argument("--mask-seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=None,
                   help=f"threads for the figures that run the model "
                        f"(default {DEFAULT_THREADS}; a login node's per-user "
                        f"quota is what one-thread-per-core runs into)")
    p.add_argument("--recon-windows", type=int, default=1, metavar="N",
                   help="how many DIFFERENT validation windows per route the "
                        "two reconstruction figures draw (default 1). Above 1 "
                        "they are numbered _w01, _w02 ... and each carries its "
                        "rank among the survey, so a figure chosen for looking "
                        "good can be captioned as chosen. The ranking table is "
                        "printed and written to "
                        "figure_metadata/reconstruction_survey.json")
    p.add_argument("--only", nargs="*", default=None,
                   help="figure names to regenerate")
    p.add_argument("--modality", default="auto",
                   choices=["auto", "eeg", "ecg", "emg"],
                   help="auto reads the checkpoint's trainer (ecg_c1_moe -> "
                        "ecg, emg_c1_moe -> emg)")
    args = p.parse_args(argv)

    # Rank 0 only. Under torchrun every rank would otherwise write the same
    # files at the same time and the SVGs would interleave.
    if int(os.environ.get("RANK", "0")) != 0:
        return 0

    apply_style()
    run_dir = args.run_dir
    ckpt_path = (args.checkpoint if os.path.isabs(args.checkpoint)
                 else os.path.join(run_dir, args.checkpoint))
    if not os.path.isfile(ckpt_path):
        if args.only and set(args.only) <= MODEL_FREE_FIGURES:
            # These read only the run's manifest and metrics, so a copy of the
            # run directory without its checkpoint (the synced results) draws them.
            return _main_model_free(args, run_dir)
        print(f"ERROR: no checkpoint at {ckpt_path}", file=sys.stderr)
        return 1

    ck, cfg, mcfg, objective, modality, model_cls, model = load_trained_model(
        ckpt_path, args.modality)
    device = torch.device("cpu")

    # A second model at initialisation, for the "before training" trace. Built
    # rather than stored: it is a deterministic function of the seed and the
    # config, and carrying a copy in every checkpoint would double their size.
    torch.manual_seed(int(cfg.get("seed", 42)))
    init_model = model_cls(
        embed_dim=int(mcfg.get("embed_dim", 384)),
        depth=int(mcfg.get("depth", 6)), num_heads=int(mcfg.get("num_heads", 6)),
        max_level=int(mcfg.get("max_level", 3)),
        wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
        wavelet_names=mcfg.get("wavelet_names"),
        wave_init_mode=mcfg.get("wave_init_mode", "pad"),
        channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
        channel_vocab_size=ck.get("channel_vocab_size"))
    init_model.eval()

    manifest_path = args.manifest or cfg.get("data", {}).get(
        f"manifest_{args.split}")
    datasets: Dict[str, EEGWindowDataset] = {}
    if manifest_path and os.path.isfile(manifest_path):
        index = CorpusIndex.from_manifest(manifest_path)
        for d in sorted(index.by_dataset()):
            try:
                datasets[d] = EEGWindowDataset(index, d, routes=ROUTES)
            except Exception as exc:                          # noqa: BLE001
                print(f"  skipping {d}: {exc}")
    else:
        print(f"  note: manifest for split {args.split!r} is not available "
              f"({manifest_path!r}); figures that need real windows are skipped.")

    run_manifest = {}
    mf = os.path.join(run_dir, "dataset_manifest.json")
    if os.path.isfile(mf):
        with open(mf) as f:
            run_manifest = json.load(f)

    mask_seed = (args.mask_seed if args.mask_seed is not None
                 else int(cfg.get("train", {}).get("val_mask_seed", 1234)))
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    bad = [f for f in formats if f not in ("svg", "pdf", "png")]
    if bad or not formats:
        raise SystemExit(f"--format: {args.format!r}; use svg, pdf, png or a "
                         f"comma-separated list of them")
    writer = FigureWriter(run_dir, formats, {
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
        "channel_vocab_sha256": ck.get("channel_vocab_sha256"),
        "modality": modality,
        "plotting_script_git_commit": git_commit(),
        "split": args.split, "mask_seed": mask_seed,
        "manifest": manifest_path,
        # The weights the run was trained under, resolved by the SAME helper
        # the trainer uses. This used to read train.spec_recon_weight and
        # train.raw_recon_weight -- keys the config stopped writing -- and fell
        # back to hard-coded 1.0/0.25, which happened to be right. At 0.5/0.5
        # it would have captioned every figure with weights the run never
        # trained under, and nothing in the pipeline would have disagreed.
        "objective": objective,
        "objective_equation": objective_equation(objective),
    }, stamp=not args.no_stamp, paper=args.paper)

    epoch_rows = read_jsonl(os.path.join(run_dir, "metrics_epoch.jsonl"))
    step_rows = read_jsonl(os.path.join(run_dir, "metrics_step.jsonl"))
    # How far the curves in these figures actually go, stamped on each one.
    # A checkpoint's epoch and the metrics' last epoch are different numbers --
    # best.pth is whichever epoch won, the metrics run to the end -- and seeing
    # both is what tells a stale figure from a current one.
    if epoch_rows:
        writer.base_meta["metrics_last_epoch"] = epoch_rows[-1].get("epoch")
        print(f"  metrics: {len(epoch_rows)} epochs "
              f"(through epoch {epoch_rows[-1].get('epoch')}), "
              f"{len(step_rows)} step rows")

    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()

    todo = set(args.only) if args.only else None

    def want(name):
        return todo is None or name in todo

    print(f"figures for {run_dir} (checkpoint {os.path.basename(ckpt_path)})")
    if want("fig_paper_pretraining"):
        fig_paper_pretraining(writer, run_manifest, epoch_rows)
    if want("fig_dataset_routes"):
        fig_dataset_routes(writer, run_manifest)
    if want("fig_pretraining_convergence"):
        fig_pretraining_convergence(writer, epoch_rows, step_rows)
    if want("fig_route_convergence"):
        fig_route_convergence(writer, epoch_rows)
    if want("fig_dual_objective"):
        fig_dual_objective(writer, epoch_rows)
    if want("fig_masked_vs_visible"):
        fig_masked_vs_visible(writer, run_dir)
    if want("fig_gradient_flow"):
        fig_gradient_flow(writer, step_rows)
    if want("fig_route_cost"):
        fig_route_cost(writer, epoch_rows)
    if datasets:
        # One survey, both reconstruction figures. Built only if one of them is
        # wanted: it is the only part of this script that runs the model over
        # more than a single window, and `--only fig_channel_embedding` should
        # not pay for it.
        if want("fig_mask_reconstruction") or \
                want("fig_raw_waveform_reconstruction") or \
                want("fig_paper_reconstruction") or \
                want("fig_paper_mask_reconstruction") or \
                want("fig_paper_raw_waveform"):
            rows, grid, ranks = survey_reconstruction_windows(
                model, datasets, mask_seed, device, objective,
                n_windows=args.recon_windows)
            if want("fig_paper_reconstruction"):
                fig_paper_reconstruction(writer, rows, grid, ranks, model, datasets,
                                         mask_seed, device)
            if want("fig_paper_mask_reconstruction"):
                fig_paper_mask_reconstruction(writer, rows, grid, ranks, model,
                                              mask_seed)
            if want("fig_paper_raw_waveform"):
                fig_paper_raw_waveform(writer, rows, grid, ranks, model, mask_seed)
            if want("fig_mask_reconstruction"):
                fig_mask_reconstruction(writer, rows, grid, ranks, mask_seed,
                                        objective=objective)
            if want("fig_raw_waveform_reconstruction"):
                fig_raw_waveform_reconstruction(writer, rows, grid, ranks,
                                                mask_seed, model.mask_ratio)
            if len(grid) > 1:
                write_reconstruction_survey(writer, rows, grid, ranks,
                                            mask_seed)
        if want("fig_mask_examples_by_dataset"):
            fig_mask_examples_by_dataset(writer, model, datasets, mask_seed,
                                         device)
        if want("fig_mask_statistics"):
            fig_mask_statistics(writer, model, datasets, mask_seed, device)
        if want("fig_scale_fold_weights"):
            fig_scale_fold_weights(writer, model, datasets, device)
    if want("fig_wavelet_frequency_response"):
        fig_wavelet_frequency_response(writer, model, init_model)
    if want("fig_channel_embedding"):
        fig_channel_embedding(writer, model, datasets)
    if want("fig_paper_channel_topology"):
        fig_paper_channel_topology(writer, model, init_model, datasets)

    for ds in datasets.values():
        ds.close()
    print(f"\n{len(writer.written)} figure(s) in {writer.fig_dir}")
    return 0


def _main_model_free(args, run_dir: str) -> int:
    """The figures that read only dataset_manifest.json and metrics_epoch.jsonl."""
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    writer = FigureWriter(run_dir, formats, {"checkpoint": None,
                                             "plotting_script_git_commit": git_commit(),
                                             "note": "drawn without the checkpoint"},
                          stamp=not args.no_stamp, paper=args.paper)
    epoch_rows = read_jsonl(os.path.join(run_dir, "metrics_epoch.jsonl"))
    step_rows = read_jsonl(os.path.join(run_dir, "metrics_step.jsonl"))
    if epoch_rows:
        writer.base_meta["metrics_last_epoch"] = epoch_rows[-1].get("epoch")
    manifest = {}
    mf = os.path.join(run_dir, "dataset_manifest.json")
    if os.path.isfile(mf):
        with open(mf) as f:
            manifest = json.load(f)
    todo = set(args.only)
    if "fig_paper_pretraining" in todo:
        fig_paper_pretraining(writer, manifest, epoch_rows)
    if "fig_dataset_routes" in todo:
        fig_dataset_routes(writer, manifest)
    if "fig_pretraining_convergence" in todo:
        fig_pretraining_convergence(writer, epoch_rows, step_rows)
    if "fig_route_convergence" in todo:
        fig_route_convergence(writer, epoch_rows)
    if "fig_dual_objective" in todo:
        fig_dual_objective(writer, epoch_rows)
    if "fig_route_cost" in todo:
        fig_route_cost(writer, epoch_rows)
    print(f"\n{len(writer.written)} figure(s) in {writer.fig_dir} (no checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
