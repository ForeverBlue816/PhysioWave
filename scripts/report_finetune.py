#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The six downstream runs as one table, instead of six greps for TEST.

    python scripts/report_finetune.py $PW_CKPT_ROOT
    python scripts/report_finetune.py $PW_CKPT_ROOT/ft_p300_pre $PW_CKPT_ROOT/probe_p300_pre

Given a directory it finds every ``*/results.json`` under it; given run
directories it reads those. Runs are grouped by task (P300 / Sleep-EDF) and by
regime (linear probe / full fine-tune), because those answer different
questions and putting them in one ranking invites reading them as one number.

WHAT THE TABLE IS FOR. The pretrained-minus-scratch delta on the LINEAR PROBE
row is the evidence about the representation: the head is a few thousand
parameters, the encoder does not move, so the score is the encoder's. The
full-fine-tune delta is smaller by nature and answers "is starting from this
checkpoint worth it", which is a different and easier question.

EEGPT's published row is printed underneath with what separates it from ours,
every time. Only the standard library is used, so this runs in whatever
environment is active.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiowave.train.published import (CHANCE, EEGPT, EEGPT_PROTOCOL,  # noqa: E402
                                        PUBLISHED, TASK_METRICS, task_of)

BLOCKS = " ▏▎▍▌▋▊▉█"
SPARK = "▁▂▃▄▅▆▇█"
TASK_TITLE = {"p300": "P300  ·  PhysioNet ERP-BCI  ·  2 classes",
              "sleep": "Sleep-EDFx  ·  5-class staging"}


def bar(value, width=14, lo=0.0, hi=1.0):
    """A metric drawn from its own floor, in eighths of a character."""
    if value is None or value != value:
        return "?" * width
    frac = min(max((value - lo) / (hi - lo or 1.0), 0.0), 1.0)
    full, rest = divmod(int(round(frac * width * 8)), 8)
    return ("█" * full + (BLOCKS[rest] if rest else "")).ljust(width)


def sparkline(values):
    """The validation curve in one line, so a flat run is visible as flat."""
    vals = [v for v in values if v is not None and v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(int((v - lo) / (hi - lo) * 7.999), 7)] for v in vals)


def load_run(path):
    """One run directory -> the row this script prints, or None."""
    results = os.path.join(path, "results.json")
    if not os.path.isfile(results):
        return None
    with open(results) as fh:
        data = json.load(fh)
    test = data.get("test")
    if not test:
        return None
    name = os.path.basename(os.path.normpath(path))
    hp = data.get("hparams", {})
    # Older runs predate these keys; the directory name is the fallback and is
    # said to be a fallback rather than presented as if it had been recorded.
    frozen = data.get("frozen_encoder")
    pretrained = data.get("pretrained")
    guessed = frozen is None or pretrained is None
    if frozen is None:
        frozen = name.startswith("probe")
    if pretrained is None:
        pretrained = name.endswith("_pre") or "_pre_" in name
    history = []
    hpath = os.path.join(path, "history.json")
    if os.path.isfile(hpath):
        with open(hpath) as fh:
            history = json.load(fh)
    return {"name": name, "path": path, "test": test, "hparams": hp,
            "trainable": data.get("trainable_params"),
            "frozen": bool(frozen), "pretrained": bool(pretrained),
            "guessed": guessed, "best_epoch": data.get("best_epoch"),
            "select_by": data.get("select_by", "?"), "history": history,
            "task": task_of(name) or task_of(str(hp))}


def discover(targets):
    runs = []
    for target in targets:
        if os.path.isfile(os.path.join(target, "results.json")):
            runs.append(target)
            continue
        for entry in sorted(os.listdir(target)):
            sub = os.path.join(target, entry)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "results.json")):
                runs.append(sub)
    return runs


def print_group(task, regime, rows, metrics, width):
    label = "linear probe" if regime else "full fine-tune"
    print(f"  {label}")
    # Each metric cell is "  0.7431 <bar>": two spaces, seven for the number,
    # one, then the bar. The header field has to be the same width or the
    # labels drift left of their columns as --width changes.
    head = f"    {'run':<22}"
    for _, name in metrics:
        head += f"  {name:^{8 + width}}"
    head += f"  {'best':>5}  {'trainable':>10}"
    print(head)

    by_init = {}
    for row in sorted(rows, key=lambda r: not r["pretrained"]):
        line = f"    {row['name']:<22}"
        for key, _ in metrics:
            value = row["test"].get(key, float("nan"))
            floor = CHANCE.get(key)
            line += f"  {value:>7.4f} {bar(value, width, lo=floor or 0.0)}"
        best = row["best_epoch"]
        line += f"  {('ep' + str(best)) if best is not None and best >= 0 else '-':>5}"
        # The head's size, every row. A probe's number is evidence about the
        # encoder only in proportion to how little of it the head could have
        # supplied, and that ratio is not visible in the score.
        line += f"  {row['trainable']:>10,}" if row["trainable"] is not None else " " * 12
        if row["guessed"]:
            line += "  (init read off the name)"
        print(line)
        by_init[row["pretrained"]] = row

        curve = sparkline([h.get(f"val_{row['select_by']}") for h in row["history"]])
        if curve:
            print(f"    {'':<22}  val {row['select_by']}: {curve}")

    if True in by_init and False in by_init:
        line = f"    {'Δ  pre - scratch':<22}"
        for key, _ in metrics:
            delta = (by_init[True]["test"].get(key, float("nan"))
                     - by_init[False]["test"].get(key, float("nan")))
            line += f"  {delta:>+7.4f} {'':<{width}}"
        print(line)
    elif rows:
        missing = "scratch" if True in by_init else "pretrained"
        print(f"    (no {missing} run here; the delta is the evidence, "
              f"a single row is not)")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+",
                   help="a checkpoint root, or the run directories themselves")
    p.add_argument("--width", type=int, default=14, help="bar width")
    p.add_argument("--no-protocol", action="store_true",
                   help="drop the paragraph on what separates us from EEGPT")
    args = p.parse_args()

    paths = discover(args.targets)
    runs = [r for r in (load_run(path) for path in paths) if r]
    if not runs:
        raise SystemExit(
            f"no run with a results.json and a test block under {args.targets}\n"
            f"  A run that crashed or has no test.h5 writes neither.")

    for task in ("p300", "sleep", None):
        group = [r for r in runs if r["task"] == task]
        if not group:
            continue
        metrics = TASK_METRICS.get(task, (("balanced_acc", "BalAcc"),
                                          ("kappa", "Kappa"), ("auroc", "AUROC")))
        print()
        print("═" * 96)
        print(f"  {TASK_TITLE.get(task, 'unrecognised task')}")
        print("═" * 96)
        for regime in (True, False):
            rows = [r for r in group if r["frozen"] is regime]
            if rows:
                print_group(task, regime, rows, metrics, args.width)

        ref = PUBLISHED.get(task, {})
        if ref:
            print("  published, linear-probing (EEGPT Table 4)")
            for name, values in ref.items():
                line = f"    {name:<22}"
                for key, _ in metrics:
                    value = values.get(key, float("nan"))
                    floor = CHANCE.get(key)
                    line += f"  {value:>7.4f} {bar(value, args.width, lo=floor or 0.0)}"
                print(line)
            print()

        proto = EEGPT_PROTOCOL.get(task)
        if proto and not args.no_protocol:
            print("  their protocol, and where ours differs")
            for key in ("source", "split", "epochs", "batch_size", "optimizer",
                        "schedule", "loss", "monitor", "trainable", "pooling"):
                if key in proto:
                    print(f"    {key:<12} {proto[key]}")
            print()
            print("    Their score is on the split they also monitor -- no test set, and")
            print("    the reported figure is averaged over folds. Ours is one")
            print("    subject-disjoint split with a test set nothing selected on, which")
            print("    is the harder of the two. Say which is which wherever both appear.")
            print()


if __name__ == "__main__":
    main()
