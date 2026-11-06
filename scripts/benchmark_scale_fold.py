#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What ScaleFold buys: the encoder with the fold against the same encoder fed
the unfolded bands, on every route, measured on one GPU.

    python scripts/benchmark_scale_fold.py --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe

The wavelet stage emits (J+1) bands per electrode at the full sampling rate.
Without the fold those (J+1)·C rows are what gets patched, so the Transformer
sees (J+1) times the tokens and attention pays (J+1)^2. The fold collapses the
scale axis to one row per electrode before patching, with an MLP shared across
channels deciding the mixture per 0.5 s block. This measures, per route and at
the training batch size:

    tokens per window        C·P with the fold, (J+1)·C·P without
    forward FLOPs            of the encoder (decomposition, patcher, Transformer)
    latency per window       forward, bf16 autocast as in training, CUDA-timed
    peak memory per batch    forward only (training adds activations and
                             optimiser state in the same proportion)
    time breakdown           decomposition / fold / patcher + Transformer, with
                             the fold's own cost beside a plain mean fold

The unfolded path is the model's own modules with the fold bypassed: the
decomposition's [B, (J+1)·C, T] output goes straight to the rate's patcher,
the 2-D position code is built for (J+1)·C rows, and no channel code is added
(a name code per band would be a design decision this benchmark does not
make). The Transformer, patcher and decomposition weights are the checkpoint's.

This is a COMPUTE comparison. Whether the fold loses reconstruction quality
against the unfolded bands is a training ablation, not a benchmark, and is
not measured here.

Writes, under --out-dir (default: the run directory):
    fold_efficiency.json                        every number, per route
    figures/fig_paper_fold_efficiency.{pdf,svg}  the journal figure
    figure_data/, figure_metadata/               as for the other figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from channel_embedding import channel_ids_for                 # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                   # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    PAPER, PAPER_FONT, PAPER_LETTER, PAPER_TITLE, PAPER_WIDTH, FigureWriter,
    _cap_threads,
    _paper_rc, apply_style, git_commit, load_trained_model, sha256_file)
from wavelet_modules import ScaleFold                         # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device, warmup: int, iters: int) -> float:
    """Median wall time of ``fn()`` in seconds, after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def _peak_mb(fn, device) -> float:
    if device.type != "cuda":
        fn()
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _sync(device)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def _flops(fn) -> float:
    """Forward FLOPs of ``fn()`` via torch's flop counter (NaN if unavailable)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return float("nan")
    try:
        counter = FlopCounterMode(display=False)
        with torch.enable_grad(), counter:
            fn()
        return float(counter.get_total_flops())
    except Exception as exc:                                  # noqa: BLE001
        print(f"  flop counter failed: {exc}")
        return float("nan")


def _meta_for(route, device):
    ids, _ = channel_ids_for(route.slots)
    return {"channel_ids": torch.as_tensor(ids, dtype=torch.long, device=device),
            "valid_channel_mask": torch.ones(route.n_channels, dtype=torch.bool,
                                             device=device)}


class _Paths:
    """The two encoder paths, built from one model's modules."""

    def __init__(self, model, route, device):
        self.model, self.route, self.device = model, route, device
        self.front = model.wavelet_frontends[route.route_id]
        self.patch = model.patch_embed_by_rate[route.rate_key]
        self.meta = _meta_for(route, device)
        self.n_scales = model.max_level + 1
        # A plain mean fold, for the fold's own cost: same bands in, one row
        # per channel out, no statistics and no MLP.
        self.mean_fold = ScaleFold(mode="mean", num_scales=self.n_scales,
                                   in_channels=route.n_channels,
                                   patch_len=route.patch_t).to(device).eval()

    # -- pieces ------------------------------------------------------------ #
    def decomp(self, x):
        return self.front.decomp(x)

    def fold(self, bands):
        return self.front.fold(bands)

    def tokens_folded(self, spec):
        return self.model._build_token_view(
            spec, self.route.rate_key, self.model._channel_code(self.meta),
            self.route.n_channels, self.route.patches_per_channel)

    def tokens_unfolded(self, bands):
        tok = self.patch(bands.unsqueeze(1))                   # [B, (J+1)C*P, D]
        return self.model.pos_embed(tok, freq_size=self.n_scales * self.route.n_channels,
                                    time_size=self.route.patches_per_channel)

    def transformer(self, tokens):
        return self.model.shared_transformer(tokens)

    # -- whole paths ------------------------------------------------------- #
    def encode_folded(self, x):
        return self.transformer(self.tokens_folded(self.fold(self.decomp(x))))

    def encode_unfolded(self, x):
        return self.transformer(self.tokens_unfolded(self.decomp(x)))


@torch.no_grad()
def benchmark_route(model, route, batch_size: int, device, warmup: int, iters: int,
                    amp: bool) -> Dict:
    P = _Paths(model, route, device)
    B, C, T = batch_size, route.n_channels, route.window_samples
    x = torch.randn(B, C, T, device=device)
    n_tok = C * route.patches_per_channel
    row = {"route": route.route_id, "channels": C, "sampling_rate": route.sampling_rate,
           "window_samples": T, "patch_samples": route.patch_t, "batch_size": B,
           "tokens_folded": n_tok, "tokens_unfolded": P.n_scales * n_tok,
           "scales": P.n_scales}
    ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
           if amp and device.type == "cuda" else torch.autocast("cpu", enabled=False))

    def run(fn):
        def f():
            with ctx:
                return fn(x)
        return f

    for tag, fn in (("folded", P.encode_folded), ("unfolded", P.encode_unfolded)):
        try:
            lat = _timed(run(fn), device, warmup, iters)
            mem = _peak_mb(run(fn), device)
            row[f"latency_s_per_batch_{tag}"] = lat
            row[f"latency_ms_per_window_{tag}"] = 1e3 * lat / B
            row[f"peak_mem_mb_{tag}"] = mem
            row[f"windows_per_s_{tag}"] = B / lat
            row[f"status_{tag}"] = "ok"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row[f"status_{tag}"] = "oom"
            print(f"  {route.route_id} {tag}: out of memory at batch {B}")
    # FLOPs on ONE window in fp32 (the counter sees the same ops whatever the
    # dtype; a single window keeps the trace short).
    x1 = x[:1]
    with torch.no_grad():
        bands1 = P.decomp(x1)
        spec1 = P.fold(bands1)
        tok_f = P.tokens_folded(spec1)
        tok_u = P.tokens_unfolded(bands1)
    f_dec = _flops(lambda: P.decomp(x1))
    f_fold = _flops(lambda: P.fold(bands1))
    f_tf_f = _flops(lambda: P.transformer(tok_f))
    f_tf_u = _flops(lambda: P.transformer(tok_u))
    f_pat_f = _flops(lambda: P.tokens_folded(spec1))
    f_pat_u = _flops(lambda: P.tokens_unfolded(bands1))
    row.update({"gflops_per_window_decomp": f_dec / 1e9, "gflops_per_window_fold": f_fold / 1e9,
                "gflops_per_window_patch_folded": f_pat_f / 1e9,
                "gflops_per_window_patch_unfolded": f_pat_u / 1e9,
                "gflops_per_window_transformer_folded": f_tf_f / 1e9,
                "gflops_per_window_transformer_unfolded": f_tf_u / 1e9,
                "gflops_per_window_folded": (f_dec + f_fold + f_pat_f + f_tf_f) / 1e9,
                "gflops_per_window_unfolded": (f_dec + f_pat_u + f_tf_u) / 1e9})
    # Time breakdown of the folded encoder, at the training batch.
    with torch.no_grad(), ctx:
        bands = P.decomp(x)
        spec = P.fold(bands)
        tok = P.tokens_folded(spec)
    t_dec = _timed(run(P.decomp), device, warmup, iters)
    t_fold = _timed(lambda: P.fold(bands), device, warmup, iters)
    t_mean = _timed(lambda: P.mean_fold(bands), device, warmup, iters)
    t_pat = _timed(lambda: P.tokens_folded(spec), device, warmup, iters)
    t_tf = _timed(lambda: P.transformer(tok), device, warmup, iters)
    row.update({"t_ms_decomp": 1e3 * t_dec, "t_ms_fold_dynamic": 1e3 * t_fold,
                "t_ms_fold_mean": 1e3 * t_mean, "t_ms_patch": 1e3 * t_pat,
                "t_ms_transformer_folded": 1e3 * t_tf})
    del x, bands, spec, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def _ch_label(rid: str) -> str:
    return f"{ROUTES[rid].n_channels} ch"


def _plain_log_ticks(ax, ticks):
    """Log axis with plain-number labels (no 10^x mathtext, whose exponents
    render below the 5 pt floor)."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())


def fig_fold_efficiency(w: FigureWriter, rows: List[Dict], device_name: str):
    """(a) forward latency and (b) peak memory, with and without the fold, per
    route at its training batch; (c) the saving factor of each cost against the
    sequence length, with the 4x token ratio for reference."""
    rids = [r["route"] for r in rows]
    x = np.arange(len(rids))
    wd = 0.38
    col_f, col_u = PAPER["indigo"], PAPER["mauve"]

    def ratio(key):
        return np.array([r.get(f"{key}_unfolded", np.nan) / r.get(f"{key}_folded", np.nan)
                         for r in rows], float)

    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 2.3))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.36,
                              left=0.06, right=0.985, top=0.88, bottom=0.18)
        ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

        def bars(ax, vf, vu, ylabel, log, ticks=None):
            ax.bar(x - wd / 2, vf, wd, color=col_f, label="With ScaleFold", lw=0)
            ax.bar(x + wd / 2, vu, wd, color=col_u, label="Without ScaleFold", lw=0)
            top = float(np.nanmax(vu))
            for i, (a, b) in enumerate(zip(vf, vu)):
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    y = b * 1.18 if log else b + 0.03 * top
                    ax.text(x[i] + wd / 2, y, f"{b / a:.1f}×", ha="center", va="bottom",
                            fontsize=PAPER_FONT - 1, color=PAPER["ink"])
            if log:
                ax.set_yscale("log")
                _plain_log_ticks(ax, ticks)
                ax.set_ylim(ticks[0] * 0.8, top * 3.2)
            else:
                ax.set_ylim(0, top * 1.18)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([_ch_label(r) for r in rids])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, len(rids) - 0.4)

        lat_f = np.array([r["latency_ms_per_window_folded"] for r in rows])
        lat_u = np.array([r.get("latency_ms_per_window_unfolded", np.nan) for r in rows])
        bars(ax_a, lat_f, lat_u, "ms per window", log=True, ticks=[0.3, 1, 3, 10, 30])
        mem_f = np.array([r["peak_mem_mb_folded"] for r in rows]) / 1024
        mem_u = np.array([r.get("peak_mem_mb_unfolded", np.nan) for r in rows]) / 1024
        bars(ax_b, mem_f, mem_u, "GB per batch", log=False)
        # One legend, in (a), whose short routes leave its upper left empty; (b)
        # shares the colours.
        ax_a.legend(loc="upper left", frameon=False, fontsize=PAPER_FONT - 1,
                    handlelength=1.0, handletextpad=0.4, borderaxespad=0.1,
                    labelspacing=0.3)

        # (c) saving factor against the sequence length
        tok = np.array([r["tokens_folded"] for r in rows])
        series = [("peak_mem_mb", "Memory", PAPER["gold"], "D"),
                  ("latency_ms_per_window", "Time", PAPER["blue"], "o"),
                  ("gflops_per_window", "FLOPs", PAPER["rule"], "s")]
        for key, lab, col, mk in series:
            v = ratio(key)
            ax_c.plot(tok, v, color=col, marker=mk, ms=4, lw=1.4, zorder=3)
            ax_c.text(tok[-1] * 1.12, v[-1], lab, color=col, ha="left", va="center",
                      fontsize=PAPER_FONT - 0.5, fontweight="bold")
        # Drawn over the data's x range only, so its label at the right end
        # sits past the line's end rather than on it.
        ax_c.plot([tok[0] / 1.35, tok[-1] * 1.05], [4.0, 4.0], color=PAPER["grey"],
                  lw=0.8, ls="--", zorder=1)
        # Labelled at the right end like the series, where it crosses no line.
        ax_c.text(tok[-1] * 1.12, 4.0, "Tokens", color=PAPER["grey"], ha="left",
                  va="center", fontsize=PAPER_FONT - 0.5, fontweight="bold")
        ax_c.set_xscale("log", base=2)
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        ax_c.xaxis.set_major_locator(FixedLocator(list(tok)))
        ax_c.xaxis.set_major_formatter(FixedFormatter([str(int(t)) for t in tok]))
        ax_c.xaxis.set_minor_locator(NullLocator())
        ax_c.set_xlim(tok[0] / 1.35, tok[-1] * 2.3)
        ax_c.set_ylim(0, max(np.nanmax(ratio(k)) for k, *_ in series) * 1.15)
        ax_c.set_xlabel("Tokens per window (with ScaleFold)")
        ax_c.set_ylabel("Without / with")
        ax_a.set_xlabel("Route")
        ax_b.set_xlabel("Route")
        # Letter and title on one baseline just above each axes, so no band of
        # empty figure sits between the label and its panel.
        titles = ("Forward time", "Peak memory", "Saving against length")
        for ax, ch, title in zip((ax_a, ax_b, ax_c), "abc", titles):
            box = ax.get_position()
            y = box.y1 + 0.035
            fig.text(box.x0 - 0.045, y, ch, fontsize=PAPER_LETTER, fontweight="bold",
                     ha="left", va="baseline", gid="paper-keep")
            fig.text(box.x0, y, title, fontsize=PAPER_TITLE, fontweight="bold",
                     ha="left", va="baseline", gid="paper-keep")

        data = {k: np.array([r.get(k, np.nan) for r in rows], float)
                for k in rows[0] if isinstance(rows[0][k], (int, float))}
        data["route"] = np.asarray(rids)
        fold_share = {r["route"]: 100 * r["t_ms_fold_dynamic"] /
                      (r["t_ms_decomp"] + r["t_ms_fold_dynamic"] + r["t_ms_patch"]
                       + r["t_ms_transformer_folded"]) for r in rows}
        w.save(fig, "fig_paper_fold_efficiency", data,
               {"rows": rows, "device": device_name,
                "fold_share_of_forward_time_pct": fold_share,
                "unfolded_path": "decomposition output [(J+1)·C rows] patched "
                                 "directly, 2-D position code for (J+1)·C rows, no "
                                 "channel code; same weights",
                "latency": "median of forward passes at the training batch size, "
                           "bf16 autocast, CUDA-synchronised, per window",
                "memory": "peak allocated, forward only, at the training batch",
                "flops": "torch.utils.flop_counter on one window, fp32",
                "not_measured": "reconstruction quality of the unfolded path"})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--format", default="svg,pdf")
    p.add_argument("--routes", nargs="*", default=None)
    p.add_argument("--batch-sizes", default=None,
                   help="route=batch,... ; default the run's train.batch_size_by_route")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-amp", action="store_true", help="fp32 instead of bf16 autocast")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-paper", action="store_true")
    p.add_argument("--from-json", default=None,
                   help="redraw the figure from a fold_efficiency.json; nothing runs")
    args = p.parse_args(argv)

    if args.from_json:
        apply_style()
        with open(args.from_json) as f:
            res = json.load(f)
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.from_json))
        writer = FigureWriter(out_dir, [s.strip() for s in args.format.split(",") if s.strip()],
                              {"checkpoint": res.get("checkpoint"), "device": res.get("device"),
                               "redrawn_from": os.path.abspath(args.from_json),
                               "plotting_script_git_commit": git_commit()},
                              stamp=args.no_paper, paper=not args.no_paper)
        fig_fold_efficiency(writer, res["rows"], res.get("device", "?"))
        return 0

    if not args.run_dir:
        p.error("--run-dir is required unless --from-json is given")
    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(args.run_dir, args.checkpoint))
    ck, cfg, mcfg, objective, modality, model_cls, model = load_trained_model(ckpt)
    if modality != "eeg":
        raise SystemExit("this benchmark is written for the EEG routes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    print(f"  device: {dev_name}")
    if device.type != "cuda":
        print("  WARNING: no GPU; latencies and memory will not be the training node's")

    bs = dict(cfg.get("train", {}).get("batch_size_by_route", {}) or {})
    if args.batch_sizes:
        for kv in args.batch_sizes.split(","):
            k, v = kv.split("=")
            bs[k.strip()] = int(v)
    rids = args.routes or list(ROUTES)
    rows = []
    for rid in rids:
        route = ROUTES[rid]
        B = int(bs.get(rid, 16))
        print(f"  {rid}: batch {B}, {route.n_channels} ch x {route.window_samples} samples")
        row = benchmark_route(model, route, B, device, args.warmup, args.iters,
                              amp=not args.no_amp)
        rows.append(row)
        print(f"    tokens {row['tokens_folded']} vs {row['tokens_unfolded']};  "
              f"GFLOPs/window {row['gflops_per_window_folded']:.2f} vs "
              f"{row['gflops_per_window_unfolded']:.2f};  "
              f"ms/window {row.get('latency_ms_per_window_folded', float('nan')):.2f} vs "
              f"{row.get('latency_ms_per_window_unfolded', float('nan')):.2f};  "
              f"peak MB {row.get('peak_mem_mb_folded', float('nan')):.0f} vs "
              f"{row.get('peak_mem_mb_unfolded', float('nan')):.0f};  "
              f"fold {row['t_ms_fold_dynamic']:.2f} ms (mean fold "
              f"{row['t_ms_fold_mean']:.2f} ms) of "
              f"{row['t_ms_decomp'] + row['t_ms_fold_dynamic'] + row['t_ms_patch'] + row['t_ms_transformer_folded']:.1f} ms")

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fold_efficiency.json"), "w") as f:
        json.dump({"device": dev_name, "checkpoint": ckpt, "amp": not args.no_amp,
                   "warmup": args.warmup, "iters": args.iters, "rows": rows}, f, indent=2)
    writer = FigureWriter(out_dir, [s.strip() for s in args.format.split(",") if s.strip()],
                          {"checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
                           "epoch": ck.get("epoch"), "device": dev_name,
                           "plotting_script_git_commit": git_commit()},
                          stamp=args.no_paper, paper=not args.no_paper)
    fig_fold_efficiency(writer, rows, dev_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
