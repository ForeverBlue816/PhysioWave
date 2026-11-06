#!/usr/bin/env python
"""
Table of the multimodal fusion runs: fusion x encoder mode x condition.

    python scripts/collect_fusion.py --root $PW_CKPT_ROOT/fusion [--markdown out.md] [--json out.json]

Reads <root>/*/results.json as physiowave/fusion/train.py writes them. One
block per task; rows are runs (fusion core, encoder mode, tag), columns are
the test conditions, cells the task's headline metric (MPDB: balanced
accuracy; SeizeIT2: AUPRC). Every number is the test set scored once at the
checkpoint validation chose.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HEADLINE = {"mpdb": "balanced_acc", "seizeit2": "auprc"}
SHORT = {"complete": "all", "eeg_only": "eeg", "ecg_only": "ecg", "emg_only": "emg",
         "eeg_ecg": "eeg+ecg", "eeg_emg": "eeg+emg", "ecg_emg": "ecg+emg",
         "eeg_noise": "eeg~noise", "eeg_bad_channel": "eeg~badch", "eeg_drift": "eeg~drift",
         "ecg_saturation": "ecg~sat", "ecg_noise": "ecg~noise", "emg_dropout": "emg~drop",
         "emg_noise": "emg~noise"}


def collect(root):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        r["_name"] = os.path.basename(os.path.dirname(path))
        runs.append(r)
    return runs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--markdown", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    runs = collect(args.root)
    if not runs:
        print(f"no results.json under {args.root}")
        return 1
    tasks = sorted({t for r in runs for t in r.get("tasks", {})})
    md = ["# Multimodal fusion", ""]
    for t in tasks:
        metric = HEADLINE.get(t, "balanced_acc")
        conds = []
        for r in runs:
            for c in (r["tasks"].get(t) or {}).get("test", {}):
                if c not in conds:
                    conds.append(c)
        print("=" * (34 + 9 * len(conds)))
        print(f"{t}: test {metric} under each condition (val-selected checkpoint)")
        print("-" * (34 + 9 * len(conds)))
        head = f"  {'run':32s}" + "".join(f"{SHORT.get(c, c):>9s}" for c in conds)
        print(head)
        md += [f"## {t} ({metric})", "", "| run | " + " | ".join(SHORT.get(c, c) for c in conds)
               + " |", "|---|" + "---|" * len(conds)]
        for r in sorted(runs, key=lambda r: (r.get("fusion", ""), r.get("encoder_mode", ""),
                                             r["_name"])):
            entry = r["tasks"].get(t)
            if not entry:
                continue
            label = f"{r.get('fusion', '?')}/{r.get('encoder_mode', '?')} [{r['_name']}]"
            vals = []
            for c in conds:
                v = (entry["test"].get(c) or {}).get(metric)
                vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "   -  ")
            print(f"  {label[:32]:32s}" + "".join(f"{v:>9s}" for v in vals))
            md.append(f"| {label} | " + " | ".join(vals) + " |")
        # seeds: mean +- std per (fusion, encoder mode) when a run was repeated
        groups = {}
        for r in runs:
            if t not in r.get("tasks", {}):
                continue
            key = (r.get("fusion"), r.get("encoder_mode"), re.sub(r"_s\d+$", "", r["_name"]))
            groups.setdefault(key, []).append(r)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        if multi:
            print(f"  {'mean +- std over seeds':32s}")
            for (fus, mode, base), rs in sorted(multi.items()):
                cells = []
                for c in conds:
                    vals = [((r["tasks"][t]["test"].get(c) or {}).get(metric)) for r in rs]
                    vals = [v for v in vals if isinstance(v, (int, float))]
                    cells.append(f"{np.mean(vals):.3f}+-{np.std(vals):.3f}" if vals else "   -  ")
                label = f"{fus}/{mode} x{len(rs)}"
                print(f"  {label[:32]:32s}" + "".join(f"{v:>12s}" for v in cells))
                md.append(f"| {label} (mean +- std) | " + " | ".join(cells) + " |")
        md.append("")
    print("=" * 60)
    print("  conditions: a modality name alone = only that modality kept; ~ = corrupted at "
          "the run's severity on top of the complete set.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=2, default=str)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
