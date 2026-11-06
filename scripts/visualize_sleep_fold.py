#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale-fold weights alpha by sleep stage, from a fine-tuned Sleep-EDF model.

    python scripts/visualize_sleep_fold.py \
        --run-dir $PW_CKPT_ROOT/eeg_downstream/sleep_edf_ft_lr1e-4 \
        --data-dir $PW_DATA_EEG/sleep_edf --format svg,pdf

The dynamic ScaleFold decides, per channel and per 0.5 s block, how much of
each wavelet scale goes into the one row per electrode the Transformer sees
(wavelet_modules.ScaleFold). The methods section calls this a differentiable
best-basis selection. This figure is the check that the selection is
physiological rather than arbitrary: if the fold leans on the approximation
band (0-6.25 Hz at 100 Hz) in N3, where the EEG is slow-wave, and on the
detail bands in wake and REM, the weights follow the signal's spectral
content, which is what a best basis is supposed to do.

Only the frontend runs: alpha is decided before the Transformer, a minute on
one GPU for the test split (most of an hour on CPU). The stage labels are never
shown to the fold -- it sees 30 s of two-channel EEG and nothing else.

On Sleep-EDF the frontend is trained from scratch during fine-tuning (the
montage is on no pretraining route), so the weights drawn are the ones the
fold learned for sleep staging, not pretrained ones.

The alpha drawn is the fold's UNDROPPED softmax over scales (``alpha_blocks``,
the field scripts/alpha_probe.py reads), in eval mode. Nominal band edges are
the dyadic ones for the sampling rate; the learned filters are not brick-wall,
so the labels say what the band is centred on, not where it ends.

Writes, under --out-dir (default: the run directory):
    figures/fig_paper_fold_by_stage.{pdf,svg}
    figure_data/fig_paper_fold_by_stage.npz      per-window alpha, labels, subjects,
                                                 band-power fractions
    figure_metadata/fig_paper_fold_by_stage.json the numbers in the figure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from physiowave.config import load_config                     # noqa: E402
from physiowave.models.build import build_model               # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    PAPER, PAPER_FONT, PAPER_WIDTH, FigureWriter, _cap_threads, _letter,
    _paper_rc, apply_style, git_commit, sha256_file)

STAGES = ["W", "N1", "N2", "N3", "REM"]
#: The journal palette (visualize_eeg_pretraining.PAPER), fixed order, never
#: cycled: one hue per stage. Wake is the warm accent; sleep deepens from pale
#: blue to indigo; REM is the mauve. Checked for colour-vision deficiency there.
STAGE_COLOR = {"W": PAPER["gold"], "N1": PAPER["pale_blue"], "N2": PAPER["blue"],
               "N3": PAPER["indigo"], "REM": PAPER["mauve"]}
#: The scales are ordinal (high to low frequency), so they take one ramp, pale
#: blue to indigo, and a marker shape each, so the order survives greyscale.
SCALE_COLORS = ["#C3D2E8", "#8EA5D0", "#6760A9", "#2F3263"]
SCALE_MARKERS = ["o", "s", "^", "D"]


# --------------------------------------------------------------------------- #
# The model, rebuilt the way finetune_main built it
# --------------------------------------------------------------------------- #

def read_split(path: str):
    """``(data, label, subject, channel_names, sampling_rate)`` of one HDF5."""
    import h5py
    with h5py.File(path, "r") as f:
        X = np.asarray(f["data"][:], dtype=np.float32)
        y = np.asarray(f["label"][:]).astype(np.int64)
        subj = (np.asarray(f["subject"][:]).astype(np.int64) if "subject" in f
                else np.zeros(len(y), np.int64))
        names = ([c.decode() if isinstance(c, bytes) else str(c)
                  for c in f["channel_names"][:]] if "channel_names" in f else None)
        fs = float(f.attrs.get("sampling_rate", 0.0)) or None
    return X, y, subj, names, fs


def load_downstream(run_dir: str, checkpoint: str, X, names, fs):
    """The fine-tuned model from its checkpoint and the config it recorded.

    finetune_main stores ``{"args": vars(args), "config": <config name>}`` and
    fills the montage facts (channels, window, names, rate) from the data file
    at build time; the same is done here, so the model has the shapes the
    weights were saved with.
    """
    ckpt = checkpoint if os.path.isabs(checkpoint) else os.path.join(run_dir, checkpoint)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    a = saved.get("args", {})
    cfg_name = a.get("config") or saved.get("config")
    if not cfg_name:
        raise SystemExit(f"{ckpt} does not record the config it was trained with")
    cfg = load_config(cfg_name, a.get("set") or [])
    mcfg = dict(cfg.get("model", {}) or {})
    name = mcfg.get("name")
    if name != "eeg_c1":
        raise SystemExit(f"{ckpt} is a {name!r} model; this reads the EEG C1 "
                         f"downstream's dynamic fold")
    mcfg["num_classes"] = int(a.get("num_classes", 5))
    c1 = dict(mcfg.get("eeg_c1", {}) or {})
    c1.pop("pretrained", None)
    if a.get("freeze_encoder"):
        c1["freeze_encoder"] = True
    c1.setdefault("in_channels", int(X.shape[1]))
    c1.setdefault("window_samples", int(X.shape[2]))
    if names and "channel_names" not in c1:
        c1["channel_names"] = list(names)
    if fs and "sampling_rate" not in c1:
        c1["sampling_rate"] = float(fs)
    mcfg["eeg_c1"] = c1
    cfg["model"] = mcfg
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith("spatial_filter")]
    if missing or unexpected:
        raise SystemExit(f"{ckpt} does not fit the rebuilt model: missing "
                         f"{missing[:5]}, unexpected {unexpected[:5]}")
    model.eval()
    fold = model.wavelet_frontend.fold
    if fold.mode != "dynamic":
        raise SystemExit(f"scale_fold is {fold.mode!r}; only 'dynamic' decides "
                         f"alpha per block")
    return ck, ckpt, cfg, model


def band_labels(fs: float, n_scales: int):
    """Nominal dyadic band of each scale, scale-major as the frontend emits
    them: detail 1 (highest) ... detail L, then the approximation."""
    L = n_scales - 1
    out = []
    for s in range(L):
        out.append((f"d{s + 1}", fs / 2 ** (s + 2), fs / 2 ** (s + 1)))
    out.append((f"a{L}", 0.0, fs / 2 ** (L + 1)))
    return out


def band_power_fractions(X: np.ndarray, fs: float, edges=(0.5, 4.0, 8.0, 13.0, 30.0)):
    """Fraction of 0.5-30 Hz power in delta / theta / alpha / beta: ``[N, C, 4]``."""
    F = np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)
    P = (F.real ** 2 + F.imag ** 2)
    f = np.fft.rfftfreq(X.shape[-1], 1.0 / fs)
    tot = P[..., (f >= edges[0]) & (f < edges[-1])].sum(-1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(P[..., (f >= lo) & (f < hi)].sum(-1) / np.maximum(tot, 1e-12))
    return np.stack(out, -1)


@torch.no_grad()
def fold_alpha(model, X: np.ndarray, batch_size: int, keep_blocks_for=None,
               device="cpu"):
    """Per-window block-mean alpha ``[N, C, S]`` and, for the windows in
    ``keep_blocks_for`` (a boolean mask), the full ``[n, C, N_blocks, S]``.

    fp32 on whatever device is given. On CPU the frontend costs about 90 ms a
    30 s window (the cross-scale FFNs run at the full 3000-sample length), so
    a whole test split is most of an hour; on a GPU it is seconds.
    """
    front = model.wavelet_frontend.to(device)
    fold = front.fold
    fold.keep_alpha = True
    means, blocks = [], []
    n_batches = (len(X) + batch_size - 1) // batch_size
    report = max(1, n_batches // 10)
    for b, i in enumerate(range(0, len(X), batch_size)):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        spec = front.decomp(x)
        fold(spec)
        a = fold.alpha_blocks                                   # [B, C, N, S]
        means.append(a.mean(dim=2).cpu().numpy())
        if keep_blocks_for is not None:
            sel = keep_blocks_for[i:i + batch_size]
            if sel.any():
                blocks.append(a[torch.from_numpy(sel).to(a.device)].cpu().numpy())
        if (b + 1) % report == 0 or b + 1 == n_batches:
            print(f"  fold alpha: {min(i + batch_size, len(X))}/{len(X)} windows")
    fold.keep_alpha = False
    A = np.concatenate(means)
    B = np.concatenate(blocks) if blocks else None
    return A, B


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def _per_subject_stage_mean(v, y, subj, subjects):
    """``[n_subjects, n_stages]`` mean of ``v`` per subject and stage; NaN where
    a subject has no window of that stage."""
    out = np.full((len(subjects), len(STAGES)), np.nan)
    for i, s in enumerate(subjects):
        for k in range(len(STAGES)):
            sel = (subj == s) & (y == k)
            if sel.any():
                out[i, k] = float(v[sel].mean())
    return out


def _t_ci(x):
    """Mean and 95% t half-width over the finite entries of ``x``."""
    from scipy.stats import t as student_t
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(x.mean()) if x.size else float("nan")), float("nan"), int(x.size)
    return (float(x.mean()),
            float(student_t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)),
            int(x.size))


def fig_fold_by_stage(w: FigureWriter, A, y, subj, names, fs, bands, frac,
                      model_note: str):
    """One row per derivation, three columns:

    (a) alpha of every scale by stage, mean over test subjects of each
        subject's mean, with a 95% t-interval across subjects;
    (b) alpha of the approximation scale relative to the same subject's
        wake -- thin lines are subjects, the thick one their mean -- which is
        the paired comparison the claim is about, free of between-subject
        offsets;
    (c) every window's approximation alpha against the share of its 0.5-30 Hz
        power in delta, coloured by stage, with Spearman rho. Drawn for both
        derivations, so neither is chosen for the figure.
    """
    from scipy.stats import spearmanr, wilcoxon
    C, S = A.shape[1], A.shape[2]
    scale_colors = (SCALE_COLORS if S == len(SCALE_COLORS)
                    else [plt.get_cmap("Blues")(v) for v in np.linspace(0.45, 1.0, S)])
    scale_markers = [SCALE_MARKERS[s % len(SCALE_MARKERS)] for s in range(S)]
    approx = S - 1
    present = [k for k in range(len(STAGES)) if (y == k).any()]
    subjects = np.unique(subj)
    W = STAGES.index("W")
    N3 = STAGES.index("N3")
    ch_names = list(names) if names else [f"ch{c}" for c in range(C)]
    stats = {"n_windows": int(len(y)), "n_subjects": int(len(subjects)),
             "per_stage": {}, "paired_vs_wake": {}, "delta_vs_alpha_approx": {},
             "delta_vs_alpha_approx_within_stage": {},
             "bands": [{"scale": b[0], "lo_hz": b[1], "hi_hz": b[2]} for b in bands]}
    data = {"alpha_window_mean": A, "label": y, "subject": subj, "band_fractions": frac,
            "channel_names": np.asarray(ch_names),
            "band_edges_hz": np.asarray([[b[1], b[2]] for b in bands])}

    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 1.62 * C + 0.75))
        H = fig.get_figheight()
        gs = fig.add_gridspec(C, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.42,
                              hspace=0.4, left=0.105, right=0.985,
                              top=1 - 0.42 / H, bottom=0.42 / H)
        xs = np.arange(len(present))
        stage_ticks = [STAGES[k] for k in present]
        axes_a, axes_b = [], []
        for c in range(C):
            ch = ch_names[c]
            # (a)
            ax = fig.add_subplot(gs[c, 0])
            axes_a.append(ax)
            for s in range(S):
                per = _per_subject_stage_mean(A[:, c, s], y, subj, subjects)
                m, h = [], []
                for k in present:
                    mm, hh, n = _t_ci(per[:, k])
                    m.append(mm)
                    h.append(hh)
                    stats["per_stage"].setdefault(STAGES[k], {}).setdefault(ch, {})[
                        bands[s][0]] = {"mean_over_subjects": mm, "ci95_half": hh,
                                        "n_subjects": n}
                lab = (f"{bands[s][0]} {bands[s][1]:g}–{bands[s][2]:g} Hz" if s < approx
                       else f"{bands[s][0]} <{bands[s][2]:g} Hz")
                ax.errorbar(xs, m, yerr=h, color=scale_colors[s], marker=scale_markers[s],
                            ms=3, lw=1.3, capsize=1.5, elinewidth=0.7, label=lab)
            ax.axhline(1.0 / S, color=PAPER["rule"], lw=0.5, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels(stage_ticks if c == C - 1 else [])
            ax.set_xlim(-0.4, len(present) - 0.6)
            ax.set_ylabel(f"{ch}\nFold weight α", linespacing=1.3)

            # (b)
            axb = fig.add_subplot(gs[c, 1])
            axes_b.append(axb)
            per = _per_subject_stage_mean(A[:, c, approx], y, subj, subjects)
            rel = per - per[:, [W]]
            for row in rel:
                axb.plot(xs, row[present], color=PAPER["pale_blue"], lw=0.7, zorder=1)
            m = [_t_ci(rel[:, k]) for k in present]
            axb.errorbar(xs, [t[0] for t in m], yerr=[t[1] for t in m],
                         color=scale_colors[approx], marker="o", ms=3.2, lw=1.6,
                         capsize=1.5, elinewidth=0.8, zorder=3)
            axb.axhline(0, color=PAPER["rule"], lw=0.6, zorder=0)
            axb.set_xticks(xs)
            axb.set_xticklabels(stage_ticks if c == C - 1 else [])
            axb.set_xlim(-0.4, len(present) - 0.6)
            axb.set_ylabel(f"Δα of {bands[approx][0]} vs. W")
            d = rel[:, N3]
            ok = np.isfinite(d)
            n_pos, n_ok = int((d[ok] > 0).sum()), int(ok.sum())
            try:
                p_w = float(wilcoxon(d[ok]).pvalue) if n_ok >= 6 else float("nan")
            except ValueError:
                p_w = float("nan")
            ptxt = ("p < 0.001" if p_w < 1e-3 else f"p = {p_w:.3f}") if np.isfinite(p_w) else ""
            # As the panel's title, above the axes: inside them it sat on the
            # subjects' lines.
            axb.set_title(f"N3 > W in {n_pos}/{n_ok} subjects" + (f", {ptxt}" if ptxt else ""),
                          fontsize=PAPER_FONT - 0.5, fontweight="bold", pad=3)
            mm, hh, _ = _t_ci(d)
            stats["paired_vs_wake"][ch] = {
                "scale": bands[approx][0], "n3_minus_w_mean": mm, "ci95_half": hh,
                "n_subjects": n_ok, "n_positive": n_pos, "wilcoxon_p": p_w,
                "per_stage_mean": {STAGES[k]: _t_ci(rel[:, k])[0] for k in present}}
            data[f"{ch}_approx_by_subject_stage"] = per

            # (c)
            axc = fig.add_subplot(gs[c, 2])
            xd, ya = frac[:, c, 0], A[:, c, approx]
            # Every window, in a shuffled draw order so no stage is painted on
            # top of the rest; nothing is subsampled.
            order = np.random.default_rng(0).permutation(len(xd))
            for k in present:
                sel = y[order] == k
                axc.scatter(xd[order][sel], ya[order][sel], s=1.2,
                            color=STAGE_COLOR[STAGES[k]], alpha=0.25, lw=0,
                            rasterized=True, label=STAGES[k])
            rho = float(spearmanr(xd, ya).correlation)
            stats["delta_vs_alpha_approx"][ch] = {"spearman_rho": rho, "n": int(len(xd))}
            stats["delta_vs_alpha_approx_within_stage"][ch] = {
                STAGES[k]: float(spearmanr(xd[y == k], ya[y == k]).correlation)
                for k in present if (y == k).sum() > 2}
            axc.text(0.03, 0.96, f"ρ = {rho:.2f}", transform=axc.transAxes, ha="left",
                     va="top", fontsize=PAPER_FONT - 0.5,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8,
                           "alpha": 0.85})
            axc.set_xlim(0, 1)
            axc.set_ylabel(f"α of {bands[approx][0]}")
            if c == C - 1:
                axc.set_xlabel("Delta share of 0.5–30 Hz power")
            else:
                axc.tick_params(axis="x", labelbottom=False)
            if c == 0:
                hc, lc = axc.get_legend_handles_labels()
        # Shared y within columns (a) and (b), so the two derivations read on
        # one scale.
        for col in (axes_a, axes_b):
            lo = min(a_.get_ylim()[0] for a_ in col)
            hi = max(a_.get_ylim()[1] for a_ in col)
            for a_ in col:
                a_.set_ylim(lo, hi)
        axes_a[-1].set_xlabel("Sleep stage")
        axes_b[-1].set_xlabel("Sleep stage")
        # Legends above their columns, off the data.
        ha, la = axes_a[0].get_legend_handles_labels()
        box_a = gs[0, 0].get_position(fig)
        fig.legend(ha, la, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5 * (box_a.x0 + box_a.x1), box_a.y1 + 0.005),
                   fontsize=PAPER_FONT - 1, handlelength=1.4, columnspacing=1.0,
                   labelspacing=0.2, borderaxespad=0.0)
        box_c = gs[0, 2].get_position(fig)
        leg = fig.legend(hc, lc, loc="lower center", ncol=len(present), frameon=False,
                         bbox_to_anchor=(0.5 * (box_c.x0 + box_c.x1), box_c.y1 + 0.005),
                         fontsize=PAPER_FONT - 1, markerscale=4, handlelength=0.6,
                         handletextpad=0.25, columnspacing=0.6, borderaxespad=0.0)
        for h_ in leg.legend_handles:
            h_.set_alpha(1.0)
        for col, ch_ in enumerate("abc"):
            x0 = gs[0, col].get_position(fig).x0
            _letter(fig, max(0.002, x0 - (0.08 if col == 0 else 0.055)), 0.995, ch_)

        w.save(fig, "fig_paper_fold_by_stage", data,
               {"stats": stats, "model": model_note, "stages": STAGES,
                "uncertainty": "mean over test subjects of each subject's mean "
                               "window alpha; 95% t-interval across subjects",
                "paired": "panel (b): per subject, the approximation scale's mean "
                          "alpha in each stage minus that subject's own W mean; "
                          "Wilcoxon signed-rank on N3 - W across subjects",
                "alpha": "fold.alpha_blocks: the undropped softmax over scales, "
                         "per channel and 0.5 s block, averaged over the 60 blocks "
                         "of a 30 s window; eval mode. A mixing weight, not a "
                         "share of energy: a scale with little power can carry a "
                         "large weight",
                "bands": "nominal dyadic bands at the data's sampling rate; the "
                         "learned filters are not brick-wall",
                "delta_share": "power in 0.5-4 Hz over power in 0.5-30 Hz, per "
                               "window, from a plain periodogram of the z-scored EEG",
                "frontend": "trained from scratch on Sleep-EDF during fine-tuning "
                            "(the montage is on no pretraining route), so these "
                            "weights were learned for sleep staging"})
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="a Sleep-EDF downstream run")
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--data-dir", required=True, help="holds <split>.h5")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default=None, help="default: the run directory")
    p.add_argument("--format", default="svg,pdf")
    p.add_argument("--batch-size", type=int, default=32,
                   help="the training batch; the cross-scale FFNs attend over all 3000 "
                        "samples, so memory grows with T^2 and 128 does not fit")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="auto: cuda when present, else cpu")
    p.add_argument("--no-paper", action="store_true", help="keep the corner stamp")
    args = p.parse_args(argv)

    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()

    path = os.path.join(args.data_dir, f"{args.split}.h5")
    X, y, subj, names, fs = read_split(path)
    if fs is None:
        raise SystemExit(f"{path} records no sampling_rate; rebuild it with "
                         f"EEG/sleep_edf_finetune.py --stage split")
    print(f"  {path}: {len(y)} windows, {X.shape[1]} ch x {X.shape[2]} samples at "
          f"{fs:g} Hz, {len(np.unique(subj))} subjects, channels {names}")
    ck, ckpt, cfg, model = load_downstream(args.run_dir, args.checkpoint, X, names, fs)
    S = model.wavelet_frontend.fold.num_scales
    bands = band_labels(fs, S)
    print(f"  fold: dynamic, {S} scales, blocks of "
          f"{model.wavelet_frontend.fold.patch_len} samples; bands "
          + ", ".join(f"{b[0]} {b[1]:g}-{b[2]:g} Hz" for b in bands))

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"  device: {device}")
    A, _ = fold_alpha(model, X, args.batch_size, device=device)
    frac = band_power_fractions(X, fs)
    for k in range(len(STAGES)):
        if (y == k).any():
            print(f"  {STAGES[k]:>3}: alpha mean over windows "
                  + " ".join(f"{bands[s][0]}={A[y == k, :, s].mean():.3f}" for s in range(S)))

    out_dir = args.out_dir or args.run_dir
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    writer = FigureWriter(out_dir, formats, {
        "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
        "epoch": ck.get("epoch"), "data": path, "split": args.split,
        "plotting_script_git_commit": git_commit()},
        stamp=args.no_paper, paper=not args.no_paper)
    note = (f"{cfg.get('model', {}).get('name')} downstream, "
            f"{os.path.basename(args.run_dir)}, epoch {ck.get('epoch')}")
    stats = fig_fold_by_stage(writer, A, y, subj, names, fs, bands, frac, note)
    for ch, v in stats["paired_vs_wake"].items():
        print(f"  {ch}: alpha_{v['scale']} N3 - W = {v['n3_minus_w_mean']:+.4f} "
              f"(+-{v['ci95_half']:.4f}), positive in {v['n_positive']}/{v['n_subjects']} "
              f"subjects, Wilcoxon p {v['wilcoxon_p']:.2g}; delta-share rho "
              f"{stats['delta_vs_alpha_approx'][ch]['spearman_rho']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
