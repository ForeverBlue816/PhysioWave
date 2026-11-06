#!/usr/bin/env python3
"""
Is each ECG (or sEMG: --modality emg) corpus fully preprocessed? One table, read from the part files.

    python scripts/ecg_corpus_status.py
    python scripts/ecg_corpus_status.py --corpus-root /path/to/ecg_c1_corpus

A unit of records is done exactly when its part file exists (see
ECG/preprocess_ecg_corpus.py), so this counts part files against the units
the record listing implies and sums the statistics the parts carry. It does
not trust a task's final dataset_statistics file, which is only written when
a task reaches the end -- a task cut off by the walltime leaves finished
units behind and no statistics, and those units are counted here.

    DONE      every unit has its part file
    PARTIAL   some do: resubmit the same --array, it continues
    NOT RUN   a listing exists, no unit has finished
    -         nothing for this corpus yet
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ECG"))

from physiowave.ecg_c1.routes import PRETRAIN_DATASETS, ROUTES  # noqa: E402

DEFAULT_ROOT = "$SCRATCH/bio/ecg/ecg_c1_corpus"
DEFAULT_ROOTS = {
    "ecg": DEFAULT_ROOT,
    "emg": "$SCRATCH/bio/emg/emg_c1_corpus",
}


def use_modality(modality: str) -> None:
    """Point the table at another modality's registry and readers."""
    global PRETRAIN_DATASETS, ROUTES
    if modality == "emg":
        sys.path.insert(0, os.path.join(ROOT, "EMG"))
        from physiowave.emg_c1.routes import PRETRAIN_DATASETS as D
        from physiowave.emg_c1.routes import ROUTES as R
        PRETRAIN_DATASETS, ROUTES = D, R


def records_per_unit(dataset_id: str) -> int:
    """The unit size preprocessing used by default, read from its adapters."""
    override = os.environ.get("RECORDS_PER_UNIT")
    if override:
        return int(override)
    try:
        if dataset_id in getattr(sys.modules.get("preprocess_emg_corpus"),
                                 "ADAPTERS", {}):
            from preprocess_emg_corpus import ADAPTERS
        else:
            from preprocess_ecg_corpus import ADAPTERS
        n = ADAPTERS[dataset_id].records_per_unit
        if n:
            return int(n)
    except Exception:                                          # noqa: BLE001
        pass
    route = ROUTES[PRETRAIN_DATASETS[dataset_id].route_id]
    return 64 if route.n_channels == 1 else 1000


def status(root: str, dataset_id: str) -> dict:
    d = os.path.join(root, dataset_id)
    out = {"dataset": dataset_id, "state": "-", "keys": 0, "units": 0,
           "parts": 0, "read": 0, "failed": 0, "short": 0, "kept": 0,
           "train": 0, "val": 0, "qc": 0, "subj_train": 0, "subj_val": 0,
           "keys_done": 0, "cand": 0, "qc_by": {}}
    listing = os.path.join(d, "record_list.txt")
    if os.path.isfile(listing):
        with open(listing) as f:
            out["keys"] = sum(1 for ln in f if ln.strip())
        out["units"] = math.ceil(out["keys"] / records_per_unit(dataset_id))
    parts = glob.glob(os.path.join(d, "parts", "unit_*.json"))
    out["parts"] = len(parts)
    subj = {"train": set(), "val": set()}
    for p in parts:
        try:
            with open(p) as f:
                part = json.load(f)
        except Exception:                                      # noqa: BLE001
            continue
        s = part["stats"]
        out["keys_done"] += int(part.get("n_keys", 0))
        out["cand"] += s.get("windows_candidate", 0)
        for r, c in s.get("qc_dropped", {}).items():
            out["qc_by"][r] = out["qc_by"].get(r, 0) + int(c)
        out["read"] += s.get("records_read", 0)
        out["failed"] += s.get("records_failed", 0)
        out["short"] += s.get("records_shorter_than_window", 0)
        out["kept"] += s.get("windows_kept", 0)
        out["qc"] += sum(s.get("qc_dropped", {}).values())
        for side in ("train", "val"):
            out[side] += sum(r["n_windows"] for r in part["rows"][side])
            subj[side].update(part["subjects"][side])
    out["subj_train"], out["subj_val"] = len(subj["train"]), len(subj["val"])
    out["leak"] = len(subj["train"] & subj["val"])
    # Done is every LISTED RECORD covered by a part file, not a unit count:
    # the count depends on the unit size, and a corpus processed before the
    # default changed has fewer, larger parts that still cover everything.
    if out["parts"] and out["keys"]:
        per = out["keys_done"] / out["parts"] if out["keys_done"] else 0
        left = max(0, out["keys"] - out["keys_done"])
        out["units"] = out["parts"] + (math.ceil(left / per) if per else 0)
    if out["keys"] and out["keys_done"] >= out["keys"]:
        out["state"] = "DONE"
    elif out["parts"]:
        out["state"] = "PARTIAL"
    elif out["keys"]:
        out["state"] = "NOT RUN"
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["ecg", "emg"], default="ecg")
    p.add_argument("--corpus-root", default=None)
    args = p.parse_args(argv)
    if args.modality == "emg":
        use_modality("emg")
        import preprocess_emg_corpus  # noqa: F401  (registers its readers)
    args.corpus_root = (args.corpus_root
                        or os.environ.get(f"{args.modality.upper()}_CORPUS")
                        or os.path.expandvars(DEFAULT_ROOTS[args.modality]))

    rows = [status(args.corpus_root, d) for d in PRETRAIN_DATASETS]
    head = (f"{'dataset':14s} {'route':8s} {'state':8s} {'units':>11s} "
            f"{'read':>10s} {'failed':>7s} {'short':>8s} {'qc drop':>8s} "
            f"{'train win':>10s} {'val win':>8s} {'subjects tr/val':>16s}")
    print(head)
    print("-" * len(head))
    tot = {"train": 0, "val": 0}
    problems = []
    notes = []
    for r in rows:
        route = PRETRAIN_DATASETS[r["dataset"]].route_id
        units = f"{r['parts']}/{r['units']}" if r["units"] else "-"
        print(f"{r['dataset']:14s} {route:8s} {r['state']:8s} {units:>11s} "
              f"{r['read']:>10,} {r['failed']:>7,} {r['short']:>8,} "
              f"{r['qc']:>8,} {r['train']:>10,} {r['val']:>8,} "
              f"{r['subj_train']:>8,}/{r['subj_val']:<7,}")
        tot["train"] += r["train"]
        tot["val"] += r["val"]
        if r["state"] != "DONE":
            problems.append(f"{r['dataset']}: {r['state']}")
        if r.get("leak"):
            problems.append(f"{r['dataset']}: {r['leak']} subject(s) on BOTH "
                            f"sides of the split")
        if r["cand"] and r["qc"] > 0.05 * r["cand"]:
            notes.append(f"{r['dataset']}: QC dropped {r['qc'] / r['cand']:.1%} "
                         f"of candidate windows -- " + ", ".join(
                             f"{k} {v:,}" for k, v in sorted(
                                 r["qc_by"].items(), key=lambda kv: -kv[1]) if v))
        if r["read"] and r["failed"] > 0.01 * (r["read"] + r["failed"]):
            problems.append(f"{r['dataset']}: {r['failed']:,} failed records "
                            f"(>1%) -- see preprocessing_failures*.jsonl")
    print("-" * len(head))
    print(f"{'total':14s} {'':8s} {'':8s} {'':>11s} {'':>10s} {'':>7s} "
          f"{'':>8s} {'':>8s} {tot['train']:>10,} {tot['val']:>8,}")
    print()
    print("  units   part files written / units in the record listing")
    if args.modality == "ecg":
        print("  short   records holding no whole 10 s window (CODE-15%'s "
              "7.3 s exams, Georgia's 5 s ones)")
    else:
        print("  short   records holding no whole 1 s window")
    print("  qc drop windows dropped for non-finite samples, a flat channel "
          "or > 25 mV")
    if notes:
        print("\nNOTE (not blocking):")
        for x in notes:
            print(f"  {x}")
    if problems:
        print("\nNOT READY:")
        for x in problems:
            print(f"  {x}")
        print("\n  PARTIAL / NOT RUN: resubmit that dataset's array with the "
              "same --array;\n  finished units are skipped.")
        return 1
    print("\nall corpora DONE -- merge with:\n"
          f"  python scripts/build_eeg_c1_manifest.py --modality {args.modality} "
          f"--corpus-root {args.corpus_root} --check-shards --check-level ends "
          f"--jobs 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
