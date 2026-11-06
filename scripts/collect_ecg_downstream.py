#!/usr/bin/env python
"""
Table of the ECG C1 downstream runs, with the paper's PhysioWave (v1) row.

    python scripts/collect_ecg_downstream.py --root $PW_CKPT_ROOT/ecg_downstream \\
        [--json summary.json] [--markdown summary.md]

Reads <root>/<task>_<mode>[_<tag>]/results.json, as ECG/finetune_ecg_c1.sh
writes them, and prints per task every run -- fine-tuned, linear probe and
from scratch, at each learning rate tried -- with the test metrics the paper
reports for that task.

SELECTION IS BY VALIDATION. Within each mode the run with the best validation
score (the metric the config selects on) is marked with a check and is the one
"pretrained - scratch" compares; the others are shown, not chosen. Picking the
learning rate by its test score would report a number tuned on the test set.

The v1 row is printed for reference and is NOT a like-for-like comparison:
v1 scored 4.1 s windows (not records), with min-max / z-score normalisation,
a fixed 0.3 multi-label threshold and its own random splits. Here the test
set is scored once, at the checkpoint validation chose, with per-class F1
thresholds also chosen on validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

TASKS = ("ptbxl", "ptbxl_super", "cpsc2018", "chapman")
MODES = ("ft", "probe", "scratch")
MODE_NAME = {"ft": "C1 fine-tuned", "probe": "C1 linear probe",
             "scratch": "C1 from scratch"}

#: (key in results.json "test", column label). The first is the paper's metric.
COLUMNS = {
    "ptbxl": [("acc", "Acc"), ("balanced_acc", "BalAcc"), ("macro_f1", "F1-mac"),
              ("auroc", "AUROC")],
    "ptbxl_super": [("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic"),
                    ("macro_auprc", "AUPRC"), ("macro_f1_tuned", "F1-mac")],
    "cpsc2018": [("micro_f1_tuned", "F1-mic"), ("macro_f1_tuned", "F1-mac"),
                 ("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic")],
    "chapman": [("micro_f1_tuned", "F1-mic"), ("macro_f1_tuned", "F1-mac"),
                ("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic")],
}
#: PhysioWave (v1), README "Benchmark Results" -- window-level, see above.
PAPER_V1 = {
    "ptbxl": {"acc": 0.731},
    "cpsc2018": {"micro_f1_tuned": 0.7709, "macro_f1_tuned": 0.6500,
                 "auroc": 0.9280, "micro_auroc": 0.9584},
    "chapman": {"micro_f1_tuned": 0.9462, "macro_f1_tuned": 0.9413,
                "auroc": 0.9930, "micro_auroc": 0.9949},
}
TITLE = {"ptbxl": "PTB-XL, 5 superclasses, single label (paper: accuracy)",
         "ptbxl_super": "PTB-XL superdiagnostic, multi-label (literature: macro AUROC)",
         "cpsc2018": "CPSC 2018, 9 classes, multi-label, per record (paper: F1-micro)",
         "chapman": "Chapman-Shaoxing, 4 rhythm classes, multi-label (paper: F1-micro)"}

def dir_re(tasks):
    return re.compile(rf"^({'|'.join(tasks)})_({'|'.join(MODES)})(?:_([\w.\-]+))?$")


#: What a modality's table is made of. scripts/collect_emg_downstream.py
#: passes its own.
SPEC = {"heading": "ECG C1 downstream", "tasks": TASKS, "columns": COLUMNS,
        "titles": TITLE, "paper": PAPER_V1,
        "paper_note": ["v1 scored 4.1 s windows with a fixed threshold and its own",
                       "splits; this table scores records once, at the checkpoint",
                       "validation chose, with F1 thresholds chosen on validation.",
                       "Indicative, not like-for-like."]}


def fmt(v):
    return "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def collect(root, tasks=TASKS):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(path))
        m = dir_re(tasks).match(name)
        if not m:
            continue
        task, mode, tag = m.group(1), m.group(2), m.group(3) or ""
        with open(path) as f:
            r = json.load(f)
        if "test" not in r:
            continue
        runs.setdefault(task, []).append({
            "mode": mode, "tag": tag, "test": r["test"],
            "lr": (r.get("hparams") or {}).get("lr"),
            "best_val": r.get("best_val"), "best_epoch": r.get("best_epoch"),
            "select_by": r.get("select_by"), "dir": os.path.dirname(path)})
    return runs


SEED_RE = re.compile(r"(?:^|_)s(\d+)$")


def by_seed(rows):
    """Runs that differ only in their seed (a trailing ``_s<seed>`` in the tag;
    none is seed 42) as one row: test metrics, val and epoch averaged, with the
    per-metric standard deviation under ``std`` and the count under ``n``."""
    groups = {}
    for r in rows:
        tag = SEED_RE.sub("", r["tag"])
        groups.setdefault((r["mode"], tag), []).append(r)
    out = []
    for (mode, tag), rs in groups.items():
        if len(rs) == 1:
            out.append({**rs[0], "tag": tag, "n": 1, "std": {}})
            continue
        keys = set().union(*(r["test"] for r in rs))
        test, std = {}, {}
        for k in keys:
            v = [r["test"][k] for r in rs if isinstance(r["test"].get(k), (int, float))]
            if v:
                m = sum(v) / len(v)
                test[k] = m
                std[k] = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        vals = [r["best_val"] for r in rs if r["best_val"] is not None]
        out.append({**rs[0], "tag": tag, "test": test, "std": std, "n": len(rs),
                    "best_val": sum(vals) / len(vals) if vals else None,
                    "best_epoch": "/".join(str(r["best_epoch"]) for r in rs),
                    "dirs": [r["dir"] for r in rs]})
    return out


def selected(rows):
    """Per mode, the run with the best validation score."""
    best = {}
    for r in rows:
        v = r["best_val"] if r["best_val"] is not None else float("-inf")
        if r["mode"] not in best or v > best[r["mode"]][0]:
            best[r["mode"]] = (v, r)
    return {m: r for m, (_, r) in best.items()}


def main(argv=None, spec=None) -> int:
    spec = spec or SPEC
    TASKS, COLUMNS, TITLE, PAPER_V1 = (spec["tasks"], spec["columns"],
                                       spec["titles"], spec["paper"])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--markdown", default=None)
    args = p.parse_args(argv)
    runs = collect(args.root, TASKS)
    if not runs:
        print(f"no finished runs under {args.root}")
        return 1
    md = [f"# {spec['heading']}", ""]
    for task in TASKS:
        if task not in runs:
            continue
        cols = COLUMNS[task]
        rows = sorted(by_seed(runs[task]), key=lambda r: (MODES.index(r["mode"]),
                                                 -(r["lr"] or 0), r["tag"]))
        pick = selected(rows)
        print("=" * 100)
        print(TITLE[task])
        print("-" * 100)
        print(f"  {'':36s}{'lr':>8s}" + "".join(f"{lab:>11s}" for _, lab in cols)
              + f"{'val':>9s}  ep")
        md += [f"## {TITLE[task]}", "",
               "| | lr | " + " | ".join(lab for _, lab in cols) + " | val | |",
               "|---|---|" + "---|" * len(cols) + "---|---|"]
        for r in rows:
            mark = "\u2713" if pick.get(r["mode"]) is r else " "
            # The lr is its own column; what is left of the tag (an encoder
            # checkpoint, a seed) labels the row: "ft_lr1e-3_e38" -> [e38].
            label = re.sub(r"^lr[^_]*_?", "", r["tag"])
            name = f"{mark} {MODE_NAME[r['mode']]}" + (f" [{label}]" if label else "")
            vals = [r["test"].get(k) for k, _ in cols]
            lr = f"{r['lr']:.0e}" if r["lr"] else "-"
            seeds = f"  n={r['n']}" if r["n"] > 1 else ""
            print(f"  {name:36s}{lr:>8s}" + "".join(f"{fmt(v):>11s}" for v in vals)
                  + f"{fmt(r['best_val']):>9s}  {r['best_epoch']}{seeds}")
            if r["n"] > 1:
                print(f"  {'':36s}{'± std':>8s}" + "".join(
                    f"{fmt(r['std'].get(k)):>11s}" for k, _ in cols))
            sd = [f" ± {r['std'][k]:.4f}" if k in r["std"] else "" for k, _ in cols]
            md.append(f"| {name} | {lr} | " + " | ".join(fmt(v) + e for v, e in zip(vals, sd))
                      + f" | {fmt(r['best_val'])} | ep {r['best_epoch']}{seeds} |")
        # Reference rows: the default single row (PhysioWave v1, ``paper``), or
        # spec["references"] = {task: [(label, {metric: value}), ...]}.
        refs = list((spec.get("references") or {}).get(task, []))
        if task in PAPER_V1:
            refs.insert(0, (spec.get("paper_label", "PhysioWave v1 (paper)"), PAPER_V1[task]))
        for label, ref in refs:
            print(f"  {'  ' + label + '*':36s}{'':>8s}" + "".join(
                f"{fmt(ref.get(k)):>11s}" for k, _ in cols))
            md.append(f"| {label}* | | " + " | ".join(
                fmt(ref.get(k)) for k, _ in cols) + " | | |")
        if "ft" in pick and "scratch" in pick:
            k, lab = cols[0]
            d = pick["ft"]["test"].get(k, float("nan")) - pick["scratch"]["test"].get(k, float("nan"))
            line = (f"pretrained - scratch ({lab}, each chosen on validation): {d:+.4f}")
            print("  " + line)
            md.append("")
            md.append(line)
        md.append("")
    print("=" * 100)
    print("  \u2713 the run of each mode with the best validation score; only it is compared.")
    if any(r["n"] > 1 for t in runs for r in by_seed(runs[t])):
        print("  n=K: mean over K seeds (std below), selected on the mean validation score.")
    if spec.get("paper_note") and any(t in runs for t in
                                      set(PAPER_V1) | set(spec.get("references") or {})):
        print("  * " + "\n    ".join(spec["paper_note"]))
        md += ["\\* " + " ".join(spec["paper_note"])]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=2, default=str)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
