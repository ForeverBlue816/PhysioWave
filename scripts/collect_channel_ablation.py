#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarise a channel-embedding ablation.

    python scripts/collect_channel_ablation.py <sweep_root> [--out-dir DIR]
    python scripts/collect_channel_ablation.py <sweep_root> --classes nontarget,target

Reads every ``<sweep_root>/fold*/C*/seed*/test_results.json`` and writes a CSV
and a JSON beside them.

The number that matters is the **paired** delta against C0. Fold-to-fold and
seed-to-seed variation on Sleep-EDF is far larger than the effect being looked
for, so an unpaired difference of means is mostly a difference of folds. Every
delta here is computed within a (fold, seed) cell, where the data, the batch
order and the legacy initialisation are all identical and the channel flags are
the only thing that moved.

Only the standard library is used, so this runs in whichever environment is
active on the login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

BASELINE = "C0"
VARIANT_DESC = {
    "C0": "none/none      baseline",
    "C1": "id/token       EEGPT-style name embedding",
    "C2": "signed/token   derivation geometry",
    "C3": "signed/fold    scale choice only",
    "C4": "signed/dual    both sites",
    "C5": "hybrid/dual    name + geometry",
}
METRICS = [
    ("test_balanced_acc", "BalAcc"),
    ("test_weighted_f1", "WF1"),
    ("test_kappa", "Kappa"),
    ("test_auroc", "AUROC"),
    ("test_acc", "Acc"),
]
# Overridden by --classes. Only affects labelling; nothing is computed from it.
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def load(root: str) -> dict:
    """``{(fold, variant, seed): result}`` for every finished run."""
    runs = {}
    for fold in sorted(os.listdir(root)):
        if not fold.startswith("fold"):
            continue
        fdir = os.path.join(root, fold)
        if not os.path.isdir(fdir):
            continue
        for variant in sorted(os.listdir(fdir)):
            vdir = os.path.join(fdir, variant)
            if not os.path.isdir(vdir):
                continue
            for seed in sorted(os.listdir(vdir)):
                path = os.path.join(vdir, seed, "test_results.json")
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    runs[(fold, variant, seed)] = json.load(f)
    return runs


def mean_sd(xs):
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    global CLASS_NAMES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_root")
    p.add_argument("--out-dir", default=None, help="default: <sweep_root>")
    p.add_argument("--classes", default=None,
                   help="comma-separated class names for the confusion matrix "
                        "and per-class F1, in label order "
                        "(default: %s)" % ",".join(CLASS_NAMES))
    args = p.parse_args()
    out_dir = args.out_dir or args.sweep_root

    if args.classes:
        CLASS_NAMES = [c.strip() for c in args.classes.split(",") if c.strip()]

    runs = load(args.sweep_root)
    if not runs:
        raise SystemExit(
            f"no fold*/C*/seed*/test_results.json under {args.sweep_root}\n"
            f"  Nothing has finished yet, or every run failed -- each run's "
            f"run.log says which.")

    variants = sorted({v for _, v, _ in runs})
    cells = sorted({(f, s) for f, _, s in runs})
    print(f"\n{args.sweep_root}")
    print(f"  {len(runs)} run(s), {len(variants)} variant(s), "
          f"{len(cells)} (fold, seed) cell(s)\n")

    # -- per-run table ---------------------------------------------------- #
    rows = []
    for (fold, variant, seed), r in sorted(runs.items()):
        row = {"fold": fold, "variant": variant, "seed": seed,
               **{k: r.get(k) for k, _ in METRICS}}
        prov = r.get("provenance", {})
        row["encoding"] = prov.get("channel_encoding")
        row["injection"] = prov.get("channel_injection")
        row["metadata_hash"] = prov.get("metadata_hash")
        row["git_commit"] = (prov.get("git_commit") or "")[:8]
        row["best_epoch"] = r.get("best_epoch")
        for i, name in enumerate(r.get("per_class_f1", []) or []):
            label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"c{i}"
            row[f"f1_{label}"] = name
        rows.append(row)

    hdr = ["fold", "variant", "seed"] + [k for k, _ in METRICS]
    print("  " + "  ".join(f"{h:>10}" for h in hdr))
    print("  " + "-" * (12 * len(hdr)))
    for row in rows:
        print("  " + "  ".join(
            f"{row[h]:>10.4f}" if isinstance(row.get(h), float) else f"{str(row.get(h)):>10}"
            for h in hdr))

    # -- mean +- sd per variant ------------------------------------------- #
    print(f"\n  mean +- sd over {len(cells)} cell(s)\n")
    print(f"  {'variant':<34}" + "  ".join(f"{lbl:>16}" for _, lbl in METRICS))
    summary = {}
    for v in variants:
        summary[v] = {}
        line = f"  {v + '  ' + VARIANT_DESC.get(v, ''):<34}"
        for key, lbl in METRICS:
            vals = [r[key] for (f, vv, s), r in runs.items()
                    if vv == v and r.get(key) is not None]
            m, sd = mean_sd(vals)
            summary[v][key] = {"mean": m, "sd": sd, "n": len(vals)}
            line += f"  {m:>8.4f}+-{sd:<6.4f}"
        print(line)

    # -- paired delta against C0 ------------------------------------------ #
    deltas = {}
    if BASELINE in variants:
        print(f"\n  paired delta vs {BASELINE}, within each (fold, seed)\n")
        print(f"  {'variant':<34}" + "  ".join(f"{lbl:>16}" for _, lbl in METRICS))
        for v in variants:
            if v == BASELINE:
                continue
            deltas[v] = {}
            line = f"  {v:<34}"
            for key, lbl in METRICS:
                d = [runs[(f, v, s)][key] - runs[(f, BASELINE, s)][key]
                     for (f, s) in cells
                     if (f, v, s) in runs and (f, BASELINE, s) in runs
                     and runs[(f, v, s)].get(key) is not None
                     and runs[(f, BASELINE, s)].get(key) is not None]
                m, sd = mean_sd(d)
                deltas[v][key] = {"mean": m, "sd": sd, "n": len(d)}
                line += f"  {m:>+8.4f}+-{sd:<6.4f}"
            print(line)
    else:
        print(f"\n  {BASELINE} is absent, so no paired delta can be computed. "
              f"An unpaired\n  comparison across folds would mostly measure the "
              f"folds.", file=sys.stderr)

    # -- confusion matrices ----------------------------------------------- #
    print("\n  confusion matrix, summed over cells (rows = true)\n")
    for v in variants:
        mats = [r["confusion_matrix"] for (f, vv, s), r in runs.items()
                if vv == v and r.get("confusion_matrix")]
        if not mats:
            continue
        n = len(mats[0])
        tot = [[sum(m[i][j] for m in mats) for j in range(n)] for i in range(n)]
        print(f"    {v}   " + " ".join(f"{CLASS_NAMES[j] if j < len(CLASS_NAMES) else j:>7}"
                                       for j in range(n)))
        for i, r_ in enumerate(tot):
            lab = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            print(f"      {lab:<4}" + " ".join(f"{x:>7}" for x in r_))
        print()

    # -- files ------------------------------------------------------------ #
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "channel_ablation.csv")
    keys = sorted({k for row in rows for k in row})
    ordered = [k for k in hdr if k in keys] + [k for k in keys if k not in hdr]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(out_dir, "channel_ablation.json")
    with open(json_path, "w") as f:
        json.dump({"runs": [{"key": list(k), **v} for k, v in
                            sorted(runs.items(), key=lambda kv: kv[0])],
                   "summary": summary, "paired_delta_vs_" + BASELINE: deltas,
                   "cells": [list(c) for c in cells]}, f, indent=2)
    print(f"  Wrote {csv_path}\n        {json_path}")

    # Last, because it decides whether the table above may be quoted.
    incomplete = [(f, v, s) for f in {c[0] for c in cells} for v in variants
                  for s in {c[1] for c in cells} if (f, v, s) not in runs]
    if incomplete:
        print(f"\n  *** {len(incomplete)} cell(s) of the matrix are missing, so the "
              f"means above\n  *** are over different subsets per variant and the "
              f"paired deltas skip those cells:\n  *** "
              + ", ".join(f"{f}/{v}/{s}" for f, v, s in incomplete[:8])
              + (" ..." if len(incomplete) > 8 else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
