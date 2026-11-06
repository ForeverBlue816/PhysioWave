#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""What a pretraining run has actually done so far, per route and per dataset.

    python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_eeg_c1_moe_n1
    python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_ecg_c1_moe      # ECG too

THE LOSS IS NOT THE SUMMARY. `val total` is an average over validation BATCHES,
and the validation sweep is proportional to corpus size while the training
mixture is not: 96.8% of validation batches are TUEG or HBN, against 37% of
training steps. A model specialising toward the five small corpora -- which
balanced sampling shows it 20 to 50 times each -- therefore looks like a model
getting worse, and the per-route columns and `macro_total` are what tell the
two apart.

THE TWO RECONSTRUCTION LOSSES ARE NOT COMPARABLE TO EACH OTHER. `val_spec` is
an MSE on per-patch normalised wavelet coefficients; `val_raw` is a SmoothL1 on
z-scored volts with a knee at beta=0.5. The raw column is the smaller number
for reasons of units, not of quality. The columns that compare the two heads
are `spec_corr` against `raw_corr` (dimensionless) and `spec_nmse` against
`raw_nmse` (1.0 is what predicting zero scores on that head's own target).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROUTES = ("E19_256", "E32_512", "E64_256", "E128_512")
DATASETS = ("tueg", "hbn", "faced", "tdbrain", "m3cv", "physionet_mi", "hgd")

#: The default table. ``(header, width, decimals, [row keys, first present
#: wins])`` -- the fallbacks are what let a run from before the dual objective
#: still render instead of raising.
COLUMNS = (
    ("train_total", 12, 5, ("train/loss_total",)),
    ("val_total", 10, 5, ("val/loss_total",)),
    ("val_spec", 10, 5, ("val/loss_masked_spec_mse", "val/loss_masked_mse")),
    ("val_raw", 10, 5, ("val/loss_masked_raw_smoothl1",)),
    ("spec_corr", 10, 4, ("val/masked_spec_corr", "val/masked_corr")),
    ("raw_corr", 9, 4, ("val/masked_raw_corr",)),
    ("spec_nmse", 10, 4, ("val/masked_spec_nmse",)),
    ("raw_nmse", 9, 4, ("val/masked_raw_nmse",)),
    ("macro_total", 12, 5, ("val/macro_route_loss_total",)),
)

#: What selects each checkpoint. Mirrors
#: physiowave.eeg_c1.train.CHECKPOINT_SELECTION; a checkpoint written by this
#: code also carries the mapping in its own state under the same name.
SELECTION = (
    ("total", "val/loss_total", "best_total.pth (and best.pth)"),
    ("spec", "val/loss_masked_spec_mse", "best_spec.pth"),
    ("raw", "val/loss_masked_raw_smoothl1", "best_raw.pth"),
    ("macro-route total", "val/macro_route_loss_total", "best_macro_total.pth"),
)

#: The route-macro summary, printed for the latest epoch. Unweighted across the
#: routes present in validation, so a route with 2% of the validation batches
#: counts as much as TUEG -- which is the point, and the reason the global
#: numbers in the table above cannot answer the same question.
MACRO = (
    ("loss_total", "macro_route_loss_total"),
    ("loss_spec", "macro_route_loss_spec"),
    ("loss_raw", "macro_route_loss_raw"),
    ("spec_corr", "macro_route_spec_corr"),
    ("raw_corr", "macro_route_raw_corr"),
    ("spec_nmse", "macro_route_spec_nmse"),
    ("raw_nmse", "macro_route_raw_nmse"),
)


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(v, width=8, places=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return " " * (width - 1) + "-"
    return f"{v:>{width}.{places}f}"


def pick(row, keys):
    """The first key this row has. A run from before the dual objective has
    `loss_masked_mse` and no `loss_masked_spec_mse`; both mean the spec term."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the --output-dir of the run")
    p.add_argument("--by", choices=("route", "dataset"), default="route")
    p.add_argument("--metric", default="masked_spec_corr",
                   help="the per-route/per-dataset metric in the breakdown "
                        "columns, e.g. masked_spec_corr, masked_raw_corr, "
                        "loss_total, masked_raw_nmse, "
                        "loss_masked_raw_smoothl1")
    args = p.parse_args(argv)

    path = os.path.join(args.run_dir, "metrics_epoch.jsonl")
    if not os.path.isfile(path):
        print(f"ERROR: no {path}\n"
              f"  If the path looks like it is missing its root, PW_CKPT_ROOT\n"
              f"  was empty in your shell: source scripts/cineca_env.sh first.",
              file=sys.stderr)
        return 1

    data = rows(path)
    if not data:
        print(f"ERROR: {path} is empty -- no epoch has finished yet.",
              file=sys.stderr)
        return 1

    keys = ROUTES if args.by == "route" else DATASETS
    # Whatever the run actually logged, in the known order first. An ECG run
    # (physiowave.ecg_c1) has routes L12_500 / L1_250 and its own datasets,
    # and a fixed EEG list would print a table of dashes for it.
    seen = []
    for r in data:
        for k in r:
            parts = k.split("/")
            if len(parts) >= 4 and parts[0] == "val" and parts[1] == args.by \
                    and parts[2] not in seen:
                seen.append(parts[2])
    keys = [k for k in keys if k in seen] + [k for k in seen if k not in keys]
    breakdown = f"val/{args.by}/{{k}}/{args.metric}"
    present = [k for k in keys
               if any(breakdown.format(k=k) in r for r in data)]
    if not present and args.metric != "masked_corr":
        # An older run has masked_corr and not masked_spec_corr. Say so rather
        # than printing a table of dashes.
        legacy = f"val/{args.by}/{{k}}/masked_corr"
        if any(legacy.format(k=k) in r for k in keys for r in data):
            print(f"note: no per-{args.by} {args.metric} in this run; falling "
                  f"back to masked_corr, its pre-dual-objective name.")
            breakdown = legacy
            present = [k for k in keys
                       if any(breakdown.format(k=k) in r for r in data)]

    head = f"{'ep':>3}" + "".join(f"{h:>{w}}" for h, w, _, _ in COLUMNS)
    head += "".join(f"{k[:9]:>10}" for k in present)
    head += f"{'gate':>8}{'mins':>7}"
    print(head)
    print("-" * len(head))

    best = {name: (None, float("inf")) for name, _, _ in SELECTION}
    for r in data:
        for name, key, _file in SELECTION:
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v) \
                    and v < best[name][1]:
                best[name] = (r["epoch"], float(v))
        line = f"{r['epoch']:>3}"
        for _h, w, places, ks in COLUMNS:
            line += fmt(pick(r, ks), w, places)
        for k in present:
            line += fmt(r.get(breakdown.format(k=k)), 10)
        secs = (r.get("train/epoch_seconds") or 0) + (r.get("val/val_seconds") or 0)
        line += fmt(r.get("train/channel_token_gate_tanh"), 8)
        line += f"{secs / 60:>7.1f}" if secs else "      -"
        print(line)

    last = data[-1]
    macro = [(label, last[f"val/{key}"]) for label, key in MACRO
             if isinstance(last.get(f"val/{key}"), (int, float))]
    if macro:
        print(f"\nroute-macro at epoch {last['epoch']} "
              f"(unweighted mean over the routes in validation):")
        print("  " + "  ".join(f"{label} {value:.4f}"
                               for label, value in macro))

    print("\ncheckpoint selection (each file is the minimum of its own metric):")
    for name, key, filename in SELECTION:
        epoch, value = best[name]
        if epoch is None:
            print(f"  best {name:<18s} -        "
                  f"({key} is not in this run's metrics)")
        else:
            print(f"  best {name:<18s} epoch {epoch:<4d} {value:.5f}"
                  f"   -> {filename}")
    if best["total"][0] is not None:
        print("  best.pth is a copy of best_total.pth: the model with the "
              "lowest validation\n  TOTAL loss, which is the loss that was "
              "trained. It is NOT the best spec loss;\n  that is "
              "best_spec.pth.")
    else:
        # Do NOT describe this run's best.pth by the current policy. A run
        # with no val/loss_total predates the change, and its best.pth was
        # selected on the spec term under the name loss_masked_mse -- saying
        # otherwise would mislabel a checkpoint that is already on disk.
        print("  This run predates dual-objective selection: it wrote only "
              "best.pth, and\n  that file holds its best SPEC loss "
              "(logged as loss_masked_mse). The\n  best_total/spec/raw/"
              "macro files do not exist for it.")

    if len(data) >= 3:
        for label, keys_ in (("spec corr", ("val/masked_spec_corr",
                                            "val/masked_corr")),
                             ("raw corr", ("val/masked_raw_corr",))):
            recent = [pick(r, keys_) for r in data[-3:]]
            if all(isinstance(x, (int, float)) for x in recent):
                drift = recent[-1] - recent[0]
                verdict = ("still improving" if drift > 0.005 else
                           "flat" if drift > -0.005 else "going backwards")
                print(f"val {label} over the last 3 epochs: "
                      f"{recent[0]:.4f} -> {recent[-1]:.4f}  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
