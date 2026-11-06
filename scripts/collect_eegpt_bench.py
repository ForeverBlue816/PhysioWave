#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Every EEGPT-benchmark run, in one table, next to EEGPT's published rows.

    python scripts/collect_eegpt_bench.py --root $PW_CKPT_ROOT/eegpt_bench

Reads <root>/<task>_f<fold>_<mode>/results.json -- the files
EEG/finetune_eegpt_bench.sh writes -- and prints the TEST metrics.

The reference rows are EEGPT's Table 4 (NeurIPS 2024), mean ± std. They are
NOT like-for-like with ours, and the table says so rather than leaving it to a
footnote. The preprocessing differs -- ours is the pretraining pipeline, theirs
a per-task one -- and so does the protocol: theirs are a frozen-encoder linear probe averaged over folds (nine
LOSO folds on BCIC, four on KaggleERN) and scored on the subjects that were
also their validation set. Ours is one fold, with a test set nothing selected
on, and every parameter fine-tuned. `ft` starts from the pretrained encoder;
`scratch` is the same model from random initialisation -- the control that
says whether pretraining did anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

#: EEGPT, Table 4 ("Ours"): (balanced acc, kappa, weighted F1 or AUROC).
EEGPT = {
    "bcic2a":    {"balanced_acc": 0.5846, "kappa": 0.4462, "weighted_f1": 0.5715},
    "bcic2b":    {"balanced_acc": 0.7212, "kappa": 0.4426, "auroc": 0.8059},
    "kaggleern": {"balanced_acc": 0.5837, "kappa": 0.1882, "auroc": 0.6621},
}
#: The third column EEGPT reports: weighted F1 for multi-class, AUROC for binary.
THIRD = {"bcic2a": "weighted_f1", "bcic2b": "auroc", "kaggleern": "auroc"}
MODE_ORDER = {"ft": 0, "probe": 1, "scratch": 2}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--json", help="also write the table here")
    args = p.parse_args(argv)

    rows = []
    for path in glob.glob(os.path.join(args.root, "*", "results.json")):
        m = re.fullmatch(r"(bcic2a|bcic2b|kaggleern)_f(\d+)_(ft|probe|scratch)(?:_([\w.\-]+))?",
                         os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        with open(path) as f:
            res = json.load(f)
        te = res.get("test")
        if not te:
            continue
        rows.append({"task": m.group(1), "fold": int(m.group(2)), "mode": m.group(3),
                     "tag": m.group(4) or "",
                     "best_epoch": res.get("best_epoch"),
                     "best_val": res.get("best_val"),
                     "select_by": res.get("select_by"),
                     "trainable": res.get("trainable_params"),
                     **{k: te.get(k) for k in ("balanced_acc", "kappa",
                                               "weighted_f1", "auroc")}})
    if not rows:
        print(f"no finished runs under {args.root}")
        return 1

    rows.sort(key=lambda r: (r["task"], r["fold"], r["tag"], MODE_ORDER[r["mode"]]))
    # Sweep runs ("sw_<config>" tags) get their own section below: a dozen
    # configurations per task would bury the single-run rows, and the number
    # that matters from a sweep is the one validation selected.
    sweep = [r for r in rows if r["tag"].startswith("sw_")]
    rows = [r for r in rows if not r["tag"].startswith("sw_")]
    print(f"{'task':10} {'fold':>4} {'mode[:tag]':18} {'BAC':>7} {'kappa':>7} "
          f"{'3rd':>7}  {'3rd is':12} {'trainable':>11}  best@ep")
    last = None
    for r in rows:
        if r["task"] != last:
            ref = EEGPT[r["task"]]
            third = THIRD[r["task"]]
            print(f"{r['task']:10} {'':>4} {'EEGPT':18} {ref['balanced_acc']:7.4f} "
                  f"{ref['kappa']:7.4f} {ref[third]:7.4f}  {third:12} "
                  f"{'probe':>11}  (Table 4, fold mean)")
            last = r["task"]
        third = THIRD[r["task"]]
        f = lambda v: f"{v:7.4f}" if isinstance(v, (int, float)) else f"{'n/a':>7}"  # noqa: E731
        label = r["mode"] + (f":{r['tag']}" if r["tag"] else "")
        print(f"{'':10} {r['fold']:>4} {label:18} {f(r['balanced_acc'])} "
              f"{f(r['kappa'])} {f(r[third])}  {third:12} "
              f"{r['trainable'] or 0:>11,}  {r['best_epoch']} ({r['select_by']})")

    # The comparison the benchmark exists for: did pretraining help?
    print()
    for task in sorted({r["task"] for r in rows}):
        key = "kappa" if task == "bcic2a" else "auroc"
        for fold in sorted({r["fold"] for r in rows if r["task"] == task}):
            here = [r for r in rows if r["task"] == task and r["fold"] == fold]
            scratch = {r["tag"]: r for r in here if r["mode"] == "scratch"}
            for r in here:
                if r["mode"] != "ft":
                    continue
                # The control trained under the SAME preprocessing: a
                # PREP=eegpt run has its own scratch, tagged like its ft. An
                # encoder-only tag (TAG=final, run with MODES=ft) has none and
                # falls back to the untagged control, which is the same model
                # under the same preprocessing.
                sc = scratch.get(r["tag"]) or (None if "prep" in r["tag"]
                                               else scratch.get(""))
                if sc is None:
                    continue
                tag = f" [{r['tag']}]" if r["tag"] else ""
                print(f"  {task} f{fold}{tag}: pretrained - scratch {key} "
                      f"= {(r[key] or 0) - (sc[key] or 0):+.4f}")

    if any(r["task"] == "kaggleern" for r in rows):
        print("\n  KaggleERN: without true_labels.csv (the Kaggle download has none) the")
        print("  test set is 4 of the 16 labelled subjects, not EEGPT's 10 Kaggle test")
        print("  subjects -- their row is a reference, not the same test data. The")
        print("  split's split.json says which ('test_split').")
    print("\n  Preprocessing: our pretraining pipeline (0.5 Hz high-pass, 50 Hz notch,")
    print("  per-window z-score). Rows tagged 'eegptprep' use EEGPT's instead on BCIC")
    print("  (0-38 Hz low-pass, Euclidean alignment, common average).")
    print("\n  EEGPT's rows: frozen encoder + linear probe, mean over folds, scored on")
    print("  their validation subjects. Ours: one fold, test subjects select nothing.")
    selected = report_sweep(sweep) if sweep else []
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"runs": rows, "sweep_runs": sweep, "sweep_selected": selected,
                       "eegpt_table4": EEGPT}, fh, indent=2)
    return 0


def report_sweep(sweep):
    """Per task and mode, the configuration VALIDATION picked, and its test score.

    ft and scratch are selected separately, each on its own validation score,
    over the same grid: pretraining gets no tuning the control did not. The
    test subjects never choose a configuration -- reporting the best TEST score
    across a grid would be selecting on the test set.
    """
    def f(v):
        return f"{v:7.4f}" if isinstance(v, (int, float)) else f"{'n/a':>7}"

    groups = {}
    for r in sweep:
        cfg, _, prep = r["tag"][3:].partition("_")      # sw_<cfg>[_<prep>]
        groups.setdefault((r["task"], r["fold"], prep, r["mode"]), []).append(
            dict(r, config=cfg))
    print("\n" + "=" * 92)
    print("  SWEEP  (configuration chosen on VALIDATION; test is that configuration's)")
    print("=" * 92)
    print(f"{'task':10} {'fold':>4} {'mode':8} {'chosen':8} {'val':>7}  "
          f"{'BAC':>7} {'kappa':>7} {'3rd':>7}  {'3rd is':12} best@ep")
    chosen, last = [], None
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], MODE_ORDER[k[3]])):
        task, fold, prep, mode = key
        cand = [r for r in groups[key] if isinstance(r["best_val"], (int, float))]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["best_val"])
        third = THIRD[task]
        if task != last:
            ref = EEGPT[task]
            print(f"{task:10} {'':>4} {'EEGPT':8} {'':8} {'':>7}  "
                  f"{ref['balanced_acc']:7.4f} {ref['kappa']:7.4f} {ref[third]:7.4f}  "
                  f"{third:12} (Table 4)")
            last = task
        label = mode + (f":{prep}" if prep else "")
        print(f"{'':10} {fold:>4} {label:8} {best['config']:8} {best['best_val']:7.4f}  "
              f"{f(best['balanced_acc'])} {f(best['kappa'])} {f(best[third])}  "
              f"{third:12} {best['best_epoch']}")
        others = "  ".join(f"{r['config']} {r['best_val']:.3f}"
                           for r in sorted(cand, key=lambda r: -r["best_val"]))
        print(f"{'':10} {'':>4} {'':8} val by config: {others}")
        chosen.append(dict(best, prep=prep))

    print()
    for task in sorted({c["task"] for c in chosen}):
        key = "kappa" if task == "bcic2a" else "auroc"
        for fold in sorted({c["fold"] for c in chosen if c["task"] == task}):
            for prep in sorted({c["prep"] for c in chosen if c["task"] == task}):
                by = {c["mode"]: c for c in chosen
                      if (c["task"], c["fold"], c["prep"]) == (task, fold, prep)}
                if "ft" in by and "scratch" in by:
                    d = (by["ft"][key] or 0) - (by["scratch"][key] or 0)
                    tag = f" [{prep}]" if prep else ""
                    print(f"  {task} f{fold}{tag}: tuned pretrained - tuned scratch "
                          f"{key} = {d:+.4f}   ({by['ft']['config']} vs "
                          f"{by['scratch']['config']})")
    return chosen


if __name__ == "__main__":
    sys.exit(main())
