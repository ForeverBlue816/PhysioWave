#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The ECG downstream benchmarks -> train/val/test HDF5 for the ECG C1 encoder.

    python ECG/ecg_downstream_prep.py --task ptbxl    --raw-dir $R/ptbxl/raw    --out-dir $D/ptbxl
    python ECG/ecg_downstream_prep.py --task cpsc2018 --raw-dir $R/cpsc2018/raw --out-dir $D/cpsc2018
    python ECG/ecg_downstream_prep.py --task chapman  --raw-dir $R/chapman/raw  --out-dir $D/chapman

TASKS -- the paper's three, with its label definitions and splits, plus the
standard PTB-XL multi-label one the ECG foundation-model papers report:

  ptbxl        PTB-XL 1.0.3, 5 superclasses, ONE label per record (the paper's
               "ECG Arrhythmia, accuracy"): diagnostic statements with
               likelihood >= 80, mapped to NORM/MI/STTC/CD/HYP; several ->
               the one latest in that order. Records with none are left out.
               Split: strat_fold 1-8 train, 9 val, 10 test (patient-disjoint
               by construction, the dataset's recommended split).
  ptbxl_super  PTB-XL superdiagnostic, MULTI-LABEL (Strodthoff et al. 2021):
               every diagnostic statement's superclass, any likelihood; the
               same folds. Scored by macro AUROC.
  cpsc2018     CPSC 2018, the 9 official classes (SNR AF IAVB LBBB RBBB PAC
               PVC STD STE) by SNOMED code, MULTI-LABEL; the cpsc_2018 and
               cpsc_2018_extra parts of PhysioNet Challenge 2021, records
               with none of the 9 left out. Record-level 70/20/10 split,
               seed 42.
  chapman      Chapman-Shaoxing / Ningbo (PhysioNet ecg-arrhythmia 1.0.0),
               4 merged rhythm classes SB, AFIB (AF, AFL), GSVT (SVT, AT,
               AVNRT, AVRT, SAAWR), SR (SR, SA, ST), MULTI-LABEL by SNOMED
               code, records with none left out. Record-level 70/20/10, seed 42.

PREPROCESSING IS THE PRETRAINING CORPUS'S, so the encoder sees downstream ECG
exactly as it saw pretraining ECG (physiowave.ecg_c1.preprocess): mV -> DC
removal -> 50 Hz notch -> 0.5 Hz high-pass -> 500 Hz -> leads placed by NAME
in the L12_500 slots -> 10 s windows -> one scale per window across the 12
leads (window_shared), clip at 20. A 12-lead, 500 Hz, 10 s window is the
L12_500 route, so the pretrained frontend and patcher transfer too.

Unlike the pretraining corpus, NO window is dropped for quality: a benchmark's
test set is the test set, and removing the records a filter dislikes would
change it. Non-finite samples are the one exception, and are counted.

CPSC records run 6-60 s: they are cut into 10 s windows with a 5 s stride, a
record shorter than 10 s is zero-padded to 10 s, and every window carries its
record id -- the fine-tuning evaluation averages a record's windows and scores
records, as the challenge did. PTB-XL and Chapman records are 10 s: one window
each.

Output, per split: data (N, 12, 5000) float32, label (N,) int64 or (N, K)
float32 0/1, record and subject (N,) bytes, channel_names (12,) bytes; attrs
sampling_rate, window_samples, class_names, task, multilabel, prep_version,
provenance. split.json beside them says what went where.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from physiowave.ecg_c1.leads import LEADS_12                   # noqa: E402
from physiowave.ecg_c1.preprocess import (ECGPreprocessConfig,  # noqa: E402
                                          ECGPreprocessError,
                                          process_ecg_record)
from physiowave.ecg_c1.routes import ROUTES                     # noqa: E402

PREP_VERSION = "ecg-c1-downstream-v1"
ROUTE = ROUTES["L12_500"]
SEED = 42

# --------------------------------------------------------------------------- #
# Label definitions (the paper's, from ECG/ptbxl_finetune.py,
# ECG/cpsc_multilabel.py and ECG/shaoxing_multilabel.py)
# --------------------------------------------------------------------------- #

PTBXL_SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

CPSC_CLASSES = [
    ("SNR", {"426783006"}),
    ("AF", {"164889003"}),
    ("IAVB", {"270492004"}),
    ("LBBB", {"733534002", "164909002"}),
    ("RBBB", {"713427006", "59118001"}),
    ("PAC", {"284470004"}),
    ("PVC", {"427172004", "164884008", "17338001"}),
    ("STD", {"429622005"}),
    ("STE", {"164931005"}),
]

CHAPMAN_CLASSES = [
    ("SB", {"426177001"}),
    ("AFIB", {"164889003", "164890007"}),
    ("GSVT", {"426761007", "713422000", "233896004", "233897008", "195101003"}),
    ("SR", {"427084000", "426783006", "427393009"}),
]

TASKS = {
    "ptbxl": {"classes": PTBXL_SUPER, "multilabel": False, "stride": None},
    "ptbxl_super": {"classes": PTBXL_SUPER, "multilabel": True, "stride": None},
    "cpsc2018": {"classes": [c for c, _ in CPSC_CLASSES], "multilabel": True,
                 "stride": 5.0},
    "chapman": {"classes": [c for c, _ in CHAPMAN_CLASSES], "multilabel": True,
                "stride": None},
}


def config_for(task: str) -> ECGPreprocessConfig:
    return ECGPreprocessConfig(
        highpass_hz=0.5, notch_harmonics=2, normalization="window_shared",
        window_seconds=ROUTE.window_seconds, stride_seconds=TASKS[task]["stride"],
        # No quality filter on a benchmark: every record stays in its split.
        flat_lead_std_mv=0.0, max_abs_mv=float("inf"),
        derive_limb_leads=True)


# --------------------------------------------------------------------------- #
# Record lists: (record id, subject, path stem relative to raw, label, split)
# --------------------------------------------------------------------------- #

def _dx_codes(hea_path: str) -> List[str]:
    with open(hea_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") and "Dx" in line.split(":", 1)[0]:
                return [t for t in re.split(r"[,\s;]+", line.split(":", 1)[1]) if t]
    return []


#: Abbreviations some releases write on the Dx line instead of SNOMED codes
#: (the CPSC 2018 original and the early Challenge 2020 headers: "AF",
#: "I-AVB", "Normal"), matched case-insensitively alongside the codes.
DX_ALIASES = {
    "SNR": {"normal", "snr", "nsr", "sr"}, "AF": {"af", "afib"},
    "IAVB": {"i-avb", "iavb", "1avb"}, "LBBB": {"lbbb", "clbbb"},
    "RBBB": {"rbbb", "crbbb"}, "PAC": {"pac", "apc"},
    "PVC": {"pvc", "vpc", "ves"}, "STD": {"std"}, "STE": {"ste"},
}


def _multihot(codes, classes) -> np.ndarray:
    y = np.zeros(len(classes), np.float32)
    low = {str(c).strip().lower() for c in codes}
    for k, (name, cset) in enumerate(classes):
        if any(c in cset for c in codes) or low & DX_ALIASES.get(name, set()):
            y[k] = 1.0
    return y


def _random_split(ids: List[str]) -> Dict[str, str]:
    """Record-level 70/20/10, deterministic in the sorted ids and the seed."""
    ids = sorted(ids)
    order = np.random.default_rng(SEED).permutation(len(ids))
    n_tr, n_va = int(0.7 * len(ids)), int(0.2 * len(ids))
    side = {}
    for rank, i in enumerate(order):
        side[ids[i]] = ("train" if rank < n_tr else
                        "val" if rank < n_tr + n_va else "test")
    return side


def list_ptbxl(raw: str, multilabel: bool) -> List[Dict]:
    import pandas as pd
    db_path = glob.glob(os.path.join(raw, "**", "ptbxl_database.csv"), recursive=True)
    if not db_path:
        raise SystemExit(f"no ptbxl_database.csv under {raw}")
    base = os.path.dirname(db_path[0])
    db = pd.read_csv(db_path[0], index_col="ecg_id")
    scp = pd.read_csv(os.path.join(base, "scp_statements.csv"), index_col=0)
    diag = {c: r["diagnostic_class"] for c, r in scp[scp["diagnostic"] == 1].iterrows()}
    out = []
    for ecg_id, row in db.iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        if multilabel:
            found = {diag[c] for c in codes if c in diag}
            if not found:
                continue
            y = np.array([1.0 if c in found else 0.0 for c in PTBXL_SUPER], np.float32)
        else:
            found = [diag[c] for c, lk in codes.items() if lk >= 80.0 and c in diag]
            found = [c for c in found if c in PTBXL_SUPER]
            if not found:
                continue
            y = max(PTBXL_SUPER.index(c) for c in found)      # NORM<MI<STTC<CD<HYP
        fold = int(row["strat_fold"])
        out.append({"record": f"ptbxl_{ecg_id:05d}",
                    "subject": f"ptbxl_p{int(row['patient_id'])}",
                    "stem": os.path.relpath(os.path.join(base, row["filename_hr"]), raw),
                    "label": y,
                    "split": "train" if fold <= 8 else "val" if fold == 9 else "test"})
    return out


def _list_by_dx(raw: str, classes, prefix: str, dirs=None) -> List[Dict]:
    heas = []
    for d in (dirs or [raw]):
        heas += glob.glob(os.path.join(d, "**", "*.hea"), recursive=True)
    recs = []
    for h in sorted(heas):
        y = _multihot(_dx_codes(h), classes)
        if y.sum() == 0:
            continue
        rid = os.path.splitext(os.path.basename(h))[0]
        recs.append({"record": f"{prefix}_{rid}", "subject": f"{prefix}_{rid}",
                     "stem": os.path.relpath(h[:-4], raw), "label": y})
    side = _random_split([r["record"] for r in recs])
    for r in recs:
        r["split"] = side[r["record"]]
    return recs


#: The CPSC 2018 training set and CPSC-Extra under the names they ship as:
#: Challenge 2021 (cpsc_2018, cpsc_2018_extra) and PhysioNet's Kaggle copies
#: of Challenge 2020 (Training_WFDB, Training_2). Records are A* and Q*.
CPSC_DIRS = ("cpsc_2018", "cpsc_2018_extra", "Training_WFDB", "Training_2")


def list_cpsc(raw: str) -> List[Dict]:
    dirs = sorted({d for name in CPSC_DIRS
                   for d in glob.glob(os.path.join(raw, "**", name), recursive=True)
                   if os.path.isdir(d)})
    if not dirs:
        raise SystemExit(f"none of {CPSC_DIRS} under {raw}")
    recs = _list_by_dx(raw, CPSC_CLASSES, "cpsc", dirs)
    seen, out = set(), []
    for r in recs:                     # one copy per record if two releases overlap
        if r["record"] not in seen:
            seen.add(r["record"])
            out.append(r)
    return out


def list_chapman(raw: str) -> List[Dict]:
    return _list_by_dx(raw, CHAPMAN_CLASSES, "chapman")


LISTERS = {
    "ptbxl": lambda raw: list_ptbxl(raw, multilabel=False),
    "ptbxl_super": lambda raw: list_ptbxl(raw, multilabel=True),
    "cpsc2018": list_cpsc,
    "chapman": list_chapman,
}


# --------------------------------------------------------------------------- #
# One record through the pretraining pipeline
# --------------------------------------------------------------------------- #

def process_one(args) -> Tuple[Optional[np.ndarray], str]:
    raw, rec, task = args
    from preprocess_ecg_corpus import read_wfdb
    try:
        sig, names, fs, units = read_wfdb(raw, rec["stem"], {})
        from physiowave.ecg_c1.preprocess import to_millivolts
        x = np.stack([to_millivolts(sig[i], units[i] or "mV") for i in range(sig.shape[0])])
        win = int(round(ROUTE.window_seconds * fs))
        if x.shape[1] < win:
            # Shorter than one window (CPSC has 6 s records): pad to 10 s
            # around each lead's median, so the record is scored rather than
            # dropped.
            med = np.nanmedian(x, axis=1, keepdims=True)
            x = np.concatenate([x, np.repeat(med, win - x.shape[1], axis=1)], axis=1)
        out = process_ecg_record(x, names, fs, "mV", ROUTE, 50.0, config_for(task),
                                 record_key=rec["record"], slots=LEADS_12)
        if out.windows.shape[0] == 0:
            return None, "no window survived (non-finite samples)"
        return out.windows.astype(np.float32), ""
    except (ECGPreprocessError, Exception) as exc:              # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

class SplitWriter:
    def __init__(self, path: str, task: str, n_classes: int, multilabel: bool, cfg):
        import h5py
        self.f = h5py.File(path + ".part", "w")
        self.path = path
        W = ROUTE.window_samples
        self.data = self.f.create_dataset("data", (0, 12, W), maxshape=(None, 12, W),
                                          dtype="float32", chunks=(16, 12, W))
        lab_shape = (0, n_classes) if multilabel else (0,)
        self.label = self.f.create_dataset(
            "label", lab_shape, maxshape=(None,) + lab_shape[1:],
            dtype="float32" if multilabel else "int64")
        dt = h5py.string_dtype()
        self.record = self.f.create_dataset("record", (0,), maxshape=(None,), dtype=dt)
        self.subject = self.f.create_dataset("subject", (0,), maxshape=(None,), dtype=dt)
        self.f.create_dataset("channel_names", data=np.array(LEADS_12, dtype="S8"))
        self.f.attrs.update({
            "sampling_rate": float(ROUTE.sampling_rate),
            "window_samples": int(W),
            "class_names": json.dumps(TASKS[task]["classes"]),
            "task": task, "multilabel": bool(multilabel),
            "prep_version": PREP_VERSION,
            "provenance": json.dumps(cfg.provenance({"route": ROUTE.route_id,
                                                     "task": task}), default=str),
        })
        self.n = 0

    def add(self, windows: np.ndarray, label, record: str, subject: str):
        k = windows.shape[0]
        for ds in (self.data, self.label, self.record, self.subject):
            ds.resize(self.n + k, axis=0)
        self.data[self.n:self.n + k] = windows
        self.label[self.n:self.n + k] = np.repeat(np.asarray(label)[None], k, axis=0) \
            if np.ndim(label) else np.full(k, int(label))
        self.record[self.n:self.n + k] = [record] * k
        self.subject[self.n:self.n + k] = [subject] * k
        self.n += k

    def close(self):
        self.f.close()
        os.replace(self.path + ".part", self.path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-records", type=int, default=None,
                   help="a quick look: this many records, spread over the list")
    args = p.parse_args(argv)

    task = TASKS[args.task]
    t0 = time.time()
    recs = LISTERS[args.task](args.raw_dir)
    if args.max_records and args.max_records < len(recs):
        idx = np.linspace(0, len(recs) - 1, args.max_records).astype(int)
        recs = [recs[i] for i in sorted(set(idx.tolist()))]
    if not recs:
        raise SystemExit(f"no labelled records found under {args.raw_dir}")
    print(f"{args.task}: {len(recs):,} labelled record(s) under {args.raw_dir} "
          f"({time.time() - t0:.0f}s to list)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = config_for(args.task)
    K = len(task["classes"])
    writers = {s: SplitWriter(os.path.join(args.out_dir, f"{s}.h5"), args.task, K,
                              task["multilabel"], cfg) for s in ("train", "val", "test")}
    stats = {s: {"records": 0, "windows": 0, "subjects": set(),
                 "label_count": np.zeros(K, np.int64)} for s in writers}
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        it = ex.map(process_one, [(args.raw_dir, r, args.task) for r in recs],
                    chunksize=16)
        for i, (rec, (windows, err)) in enumerate(zip(recs, it)):
            if windows is None:
                failures.append({"record": rec["record"], "split": rec["split"],
                                 "reason": err})
                continue
            writers[rec["split"]].add(windows, rec["label"], rec["record"], rec["subject"])
            st = stats[rec["split"]]
            st["records"] += 1
            st["windows"] += windows.shape[0]
            st["subjects"].add(rec["subject"])
            if task["multilabel"]:
                st["label_count"] += np.asarray(rec["label"]).astype(np.int64)
            else:
                st["label_count"][int(rec["label"])] += 1
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(recs):,} records, {len(failures)} failed",
                      flush=True)
    for w in writers.values():
        w.close()

    # Splits must not share a subject; PTB-XL's folds and the record-level
    # splits both guarantee it, and this says so rather than assuming.
    sides = list(stats)
    leak = {f"{a}/{b}": len(stats[a]["subjects"] & stats[b]["subjects"])
            for i, a in enumerate(sides) for b in sides[i + 1:]}
    summary = {
        "task": args.task, "classes": task["classes"], "multilabel": task["multilabel"],
        "prep_version": PREP_VERSION, "route": ROUTE.route_id,
        "window_seconds": ROUTE.window_seconds, "stride_seconds": task["stride"],
        "splits": {s: {"records": v["records"], "windows": v["windows"],
                       "subjects": len(v["subjects"]),
                       "per_class_records": dict(zip(task["classes"],
                                                     v["label_count"].tolist()))}
                   for s, v in stats.items()},
        "subject_overlap": leak, "failed_records": len(failures),
        "failures": failures[:200], "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['records']:>7,} records  {v['windows']:>8,} windows  "
              f"{v['subjects']:>7,} subjects  {v['per_class_records']}")
    print(f"  {len(failures)} record(s) failed; subject overlap {leak}")
    if any(leak.values()):
        print("ERROR: a subject is on two sides of the split", file=sys.stderr)
        return 1
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
