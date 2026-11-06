#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Does the dynamic scale fold actually vary alpha per time block?

    python scripts/alpha_probe.py best_model.pth --data test.h5 --out-dir alpha/
    python scripts/alpha_probe.py kl0.pth kl1e3.pth --labels "kl=0" "kl=1e-3" \
        --data test.h5 --out-dir alpha/

The training log prints ``alpha=[0.252 0.252 0.250 0.246]``, which is the mean
over batch, channel and time block. That number cannot answer the question it
looks like it answers: a fold whose weights swing hard from block to block and
one frozen at 1/S both report a flat vector. The claim that makes ``dynamic``
different from a fixed inverse transform is that the weight is *decided per
block*, and the marginal is silent about it.

This reads the full ``[B, C, N, S]`` field off a trained checkpoint and writes:

    alpha_trace.png     alpha along time for one (sample, channel), per scale
    alpha_spread.png    distribution of alpha across every block, per scale
    alpha_compare.png   the across-block spread under each checkpoint given
    alpha_blocks.npz    the raw field, so the figures can be redrawn

and prints the numbers the figures are made of. The one that matters is the
standard deviation across time blocks: at zero the fold is static whatever the
mean looks like, and the argument for this mode collapses.

Two things to know before reading a flat result as a negative:

* The MLP's output layer is zero-initialised, so an *untrained* fold gives
  exactly uniform alpha and exactly zero spread. That is design, not failure.
  A checkpoint from step 0 will always look static.

  This only became true when model.py stopped letting its generic
  ``apply(_init_weights)`` overwrite that layer. Every checkpoint trained
  before that fix started from a randomly initialised MLP and carries roughly
  5e-4 of across-block spread that training never chose -- including the
  Sleep-EDF and PhysioP300 runs. Read anything at that magnitude in those as
  initialisation, not as a learned weighting.
* ``--fold_kl`` pulls alpha towards uniform on purpose, because a fold collapsed
  onto one band fits the training set better and the task loss will not object.
  Some flattening is the regulariser working. Comparing two checkpoints that
  differ only in fold_kl is what separates "the regulariser is holding it near
  uniform" from "nothing is driving it away from uniform".
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BERTWaveletTransformer                      # noqa: E402
from scripts.inspect_checkpoint import model_kwargs_from_args  # noqa: E402

# Categorical slots 1-4 of the reference palette, in its fixed order. Assigned
# to scales in order and never cycled; four scales, four slots. Line style and
# marker repeat the identity, so a reader who cannot separate the hues still
# can -- the colour is not carrying identity on its own.
SCALE_STYLE = [
    ("#2a78d6", "-",  "o"),
    ("#eb6834", "--", "s"),
    ("#1baf7a", "-.", "^"),
    ("#eda100", ":",  "D"),
]
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#d8d7d0"


def load_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved = ck.get("args")
    if saved is None:
        raise SystemExit(f"{path} has no stored args; it predates them.")
    if saved.get("scale_fold") != "dynamic":
        raise SystemExit(
            f"{path} was trained with scale_fold={saved.get('scale_fold')!r}.\n"
            f"  Only 'dynamic' predicts alpha per block -- the other modes have "
            f"no per-block weight\n  for this script to read."
        )
    model = BERTWaveletTransformer(**model_kwargs_from_args(saved),
                                   task_type="classification")
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  note: {len(missing)} tensor(s) not in the checkpoint, e.g. "
              f"{missing[:3]}", file=sys.stderr)
    model.eval()
    model.fold.keep_alpha = True
    return model, saved


@torch.no_grad()
def collect(model, x: torch.Tensor, batch: int = 16) -> np.ndarray:
    """Run the windows through and stack the per-block alpha: ``[B, C, N, S]``."""
    out = []
    for i in range(0, len(x), batch):
        model(x[i:i + batch], task="classify")
        a = model.scale_fold_blocks()
        if a is None:
            raise SystemExit("the fold returned no alpha; is scale_fold dynamic?")
        out.append(a.cpu().numpy())
    return np.concatenate(out, axis=0)


def summarise(name: str, a: np.ndarray) -> dict:
    """Print, and return, the numbers the figures are drawn from."""
    B, C, N, S = a.shape
    sd_time = a.std(axis=2).mean(axis=(0, 1))        # across blocks  -> [S]
    sd_chan = a.std(axis=1).mean(axis=(0, 1))        # across channels-> [S]
    sd_samp = a.std(axis=0).mean(axis=(0, 1))        # across samples -> [S]
    kl = float((a * (np.log(np.clip(a, 1e-12, None)) + np.log(S))).sum(-1).mean())
    print(f"\n  {name}   [B,C,N,S] = {a.shape}")
    def row(label, vals, fmt="{:.4f}", note=""):
        print(f"    {label:<20}" + "  ".join(fmt.format(v) for v in vals) + note)

    row("mean over B,C,N", a.mean(axis=(0, 1, 2)))
    row("sd across blocks", sd_time, note="   <- the claim rests on this")
    row("sd across channels", sd_chan)
    row("sd across samples", sd_samp)
    # Against 1/S, because 0.01 means one thing at S=4 and another at S=16.
    row("sd / (1/S)", sd_time * S, "{:.3f}")
    print(f"    KL(alpha||uniform) {kl:.5f}   (0 = exactly uniform)")
    return {"sd_time": sd_time, "sd_chan": sd_chan, "sd_samp": sd_samp, "kl": kl}


def _axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def figures(runs: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(runs))
    a = runs[first]["alpha"]
    S = a.shape[-1]

    # 1. alpha along time, one (sample, channel). A trace, because the question
    #    is change-over-time; the y axis is the weight itself, so no second axis
    #    exists to be tempted by.
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=160)
    for s in range(S):
        c, ls, mk = SCALE_STYLE[s % len(SCALE_STYLE)]
        ax.plot(a[0, 0, :, s], color=c, linestyle=ls, marker=mk, markersize=3.5,
                linewidth=1.6, label=f"scale {s}")
    ax.axhline(1.0 / S, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax.annotate(f"uniform = 1/{S}", (0.995, 1.0 / S), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("time block (one per backbone token)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("alpha", color=INK_MUTED, fontsize=9)
    ax.set_title(f"Mixing weight along time -- {first}, sample 0, channel 0",
                 color=INK, fontsize=10.5, loc="left")
    _axes(ax)
    # Below the axes, not inside them: with S traces crossing the middle of the
    # panel there is no in-plot position that does not sit on the data.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, ncol=S,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_trace.png"),
                                    bbox_inches="tight")
    plt.close(fig)

    # 2. Every block's alpha, per scale. A flat fold is a spike at 1/S; a
    #    working one is spread. Violin rather than bars: the shape is the point.
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    data = [a[..., s].ravel() for s in range(S)]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for s, body in enumerate(parts["bodies"]):
        body.set_facecolor(SCALE_STYLE[s % len(SCALE_STYLE)][0])
        body.set_edgecolor(SCALE_STYLE[s % len(SCALE_STYLE)][0])
        body.set_alpha(0.55)
    for s in range(S):
        ax.plot([s + 1], [data[s].mean()], marker="_", markersize=16,
                color=INK, markeredgewidth=1.8)
    ax.axhline(1.0 / S, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax.set_xticks(range(1, S + 1)); ax.set_xticklabels([f"scale {s}" for s in range(S)])
    ax.set_ylabel("alpha over all blocks", color=INK_MUTED, fontsize=9)
    ax.set_title(f"Spread across every (sample, channel, block) -- {first}",
                 color=INK, fontsize=10.5, loc="left")
    _axes(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_spread.png"))
    plt.close(fig)

    # 3. The across-block sd under each checkpoint. One grouped bar chart, one
    #    axis; the whole comparison is a single measure at the same scale.
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    names = list(runs)
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        sd = runs[name]["stats"]["sd_time"]
        xs = np.arange(S) + (i - (len(names) - 1) / 2) * width
        ax.bar(xs, sd, width=width * 0.92,
               color=SCALE_STYLE[i % len(SCALE_STYLE)][0], label=name)
    ax.set_xticks(range(S)); ax.set_xticklabels([f"scale {s}" for s in range(S)])
    ax.set_ylabel("sd of alpha across time blocks", color=INK_MUTED, fontsize=9)
    ax.set_title("Across-block spread by checkpoint  (0 = static fold)",
                 color=INK, fontsize=10.5, loc="left")
    _axes(ax)
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "alpha_compare.png"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--labels", nargs="*", default=None,
                   help="one per checkpoint; defaults to the file names")
    p.add_argument("--data", required=True, help="HDF5 with a 'data' dataset")
    p.add_argument("-n", type=int, default=64, help="windows to push through")
    p.add_argument("--out-dir", default="alpha_probe")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(c)) or
                             os.path.basename(c) for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit(f"{len(labels)} labels for {len(args.checkpoints)} checkpoints")
    os.makedirs(args.out_dir, exist_ok=True)

    import h5py
    with h5py.File(args.data, "r") as f:
        x = torch.from_numpy(f["data"][:args.n]).float()
    print(f"\n{args.data}: {tuple(x.shape)}")

    runs, raw = {}, {}
    for ckpt, label in zip(args.checkpoints, labels):
        model, saved = load_model(ckpt)
        print(f"\n  {label}: fold_kl={saved.get('fold_kl')} "
              f"gamma={saved.get('fold_gamma')} "
              f"synthesis={saved.get('fold_synthesis')} "
              f"patch={saved.get('patch_size')}")
        a = collect(model, x)
        runs[label] = {"alpha": a, "stats": summarise(label, a)}
        raw[label] = a

    np.savez_compressed(os.path.join(args.out_dir, "alpha_blocks.npz"), **raw)

    if not args.no_figures:
        try:
            figures(runs, args.out_dir)
            print(f"\n  Wrote alpha_trace.png, alpha_spread.png, alpha_compare.png "
                  f"and alpha_blocks.npz\n  to {args.out_dir}")
        except ImportError:
            print("\n  matplotlib is not installed; alpha_blocks.npz was still "
                  "written.\n  Draw from it elsewhere, or pip install matplotlib.",
                  file=sys.stderr)

    worst = max(r["stats"]["sd_time"].max() for r in runs.values())
    print("\n  A freshly initialised fold gives alpha exactly uniform and a spread of\n"
          "  exactly zero, so any number above is something training put there --\n"
          "  but only since the init fix in model.py. A checkpoint trained before it\n"
          "  started from a non-uniform MLP and carries about 5e-4 of spread that\n"
          "  training never chose; read anything at that magnitude as noise.")
    print()
    if worst < 1e-4:
        print("  *** alpha does not move across time blocks (max sd "
              f"{worst:.2e}).\n"
              "  *** The fold is static. Either the checkpoint is untrained --\n"
              "  *** the MLP output layer is zero-initialised, so step 0 is\n"
              "  *** uniform by construction -- or nothing drove it away from\n"
              "  *** uniform during training. This is the result that would\n"
              "  *** remove the reason for preferring 'dynamic' over 'mean'.")
        sys.exit(2)
    print(f"  alpha varies across time blocks (max sd {worst:.4f}). The per-block\n"
          f"  claim is measurable in this checkpoint; the figures show how much.")


if __name__ == "__main__":
    main()
