#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mean and spread over the PhysioP300 LOSO folds, against the published rows.

    python scripts/collect_p300_folds.py <sweep_root>

Reads every ``<sweep_root>/fold*/test_results.json``. Only the standard library
is needed, so it runs in whichever environment happens to be active.

Two things it refuses to do quietly:

* average over fewer than the nine folds without saying so. A LOSO mean over
  six subjects is not comparable to one over nine, and the difference does not
  show in the number.
* print a standard deviation next to EEGPT's without noting that they are not
  the same quantity. Theirs is the spread of three repeats of a nine-fold mean;
  ours is the spread across the folds themselves, which is far larger. The
  published +-0.0139 on kappa is not an error bar our +- can be compared to.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# EEGPT Table 4, PhysioP300 row -- one copy, in physiowave/train/published.py,
# shared with finetune_main's test block and scripts/report_finetune.py. The
# module is stdlib-only, so importing it does not cost this script the property
# that it runs in whichever environment happens to be active.
from physiowave.train.published import (EEGPT, PUBLISHED as ALL_PUBLISHED,  # noqa: E402
                                        TASK_METRICS)

PUBLISHED = ALL_PUBLISHED["p300"]
METRICS = list(TASK_METRICS["p300"])
N_FOLDS = 9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_root")
    p.add_argument("--folds", type=int, default=N_FOLDS,
                   help=f"how many folds a complete sweep has (default {N_FOLDS})")
    args = p.parse_args()

    rows = {}
    for name in sorted(os.listdir(args.sweep_root)):
        if not name.startswith("fold"):
            continue
        path = os.path.join(args.sweep_root, name, "test_results.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            rows[name] = json.load(f)

    if not rows:
        raise SystemExit(
            f"no fold*/test_results.json under {args.sweep_root}\n"
            f"  Either the sweep has not run, or every fold failed -- the job's "
            f".err file says which."
        )

    print(f"\n{args.sweep_root}\n")
    print(f"  {'fold':<8} " + "  ".join(f"{lbl:>8}" for _, lbl in METRICS))
    print("  " + "-" * (8 + 1 + 10 * len(METRICS)))
    for name in sorted(rows):
        r = rows[name]
        print(f"  {name:<8} " + "  ".join(
            f"{r.get('test_' + k, float('nan')):>8.4f}" for k, _ in METRICS))

    print()
    stats = {}
    for k, lbl in METRICS:
        vals = [r["test_" + k] for r in rows.values() if "test_" + k in r]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        stats[k] = (mean, sd)
        print(f"  {lbl:<8} mean {mean:.4f}   sd {sd:.4f}   (n={len(vals)})")

    print(f"\n  {'':<22}" + "  ".join(f"{lbl:>8}" for _, lbl in METRICS))
    ours = "ours (" + str(len(rows)) + " folds)"
    print(f"  {ours:<22}" + "  ".join(f"{stats[k][0]:>8.4f}" for k, _ in METRICS))
    for name, vals in PUBLISHED.items():
        print(f"  {name:<22}" + "  ".join(f"{vals[k]:>8.4f}" for k, _ in METRICS))
    print(f"\n  {'delta vs EEGPT':<22}" + "  ".join(
        f"{stats[k][0] - PUBLISHED[EEGPT][k]:>+8.4f}" for k, _ in METRICS))

    print("\n  The sd above is across folds. EEGPT's published +- is the spread of\n"
          "  three repeats of a nine-fold mean, which is a much smaller quantity --\n"
          "  do not read the two as the same kind of error bar.")

    # Last, and on stdout. A caveat that decides whether the table above may be
    # quoted at all does not belong on stderr, where the shell interleaves it
    # somewhere in the middle of the output and it scrolls past unread.
    if len(rows) != args.folds:
        missing = [f"fold{i}" for i in range(args.folds) if f"fold{i}" not in rows]
        print(f"\n  *** INCOMPLETE: {len(rows)} of {args.folds} folds "
              f"({', '.join(missing)} missing).\n"
              f"  *** A LOSO mean over a subset of the subjects is not the "
              f"published protocol\n"
              f"  *** and is not comparable to it. Do not quote the table above.\n")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
