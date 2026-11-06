#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPDB (Tao et al., Sci. Data 2024) -> aligned EEG + ECG + sEMG windows.

    python Fusion/mpdb_prep.py --raw-dir $R/mpdb --out-dir $D/mpdb

THE DATA. 35 participants in a driving simulator; 59-channel EEG and one
chest ECG electrode (Neuracle, 1 kHz), four limb sEMG channels (NeuSen WM,
1 kHz: right tibialis anterior, right gastrocnemius, both brachioradialis).
Ten simulator events fall into five behaviours, by the marker codes of the
paper's Table 2:

    smooth driving  133              lane change  129 131
    acceleration    135 137 143      turning      125 127
    deceleration    139 141 145

THE SOURCE is the authors' PREPROCESSED release (figshare 22192831): per
subject EEG_{n}.set, ECG_{n}.set, EMG_{n}.set, EEGLAB epochs of 2 s (0.5 s
before the marker to 1.5 s after), already band-passed. The raw release
(22193119, 23 GB of continuous .set files) is not needed. The release carries
no timestamps, and the three files of a subject hold different epoch counts
(the EEG is pruned of rejected epochs, sometimes the ECG too), so they are
ALIGNED BY MARKER ORDER: the shortest file anchors and the others are paired
to it by the longest common subsequence of marker codes. split.json records
per subject how many pairs were ambiguous up to a same-behaviour neighbour;
a subject whose files disagree beyond pruning is dropped and named.

PREPROCESSING. The authors' filters stand (re-filtering a 2 s epoch at 0.5 Hz
would put the transient inside the window). Each epoch is resampled to the
encoder rates (EEG 256 Hz, ECG 250 Hz, sEMG 2000 Hz) and normalised the way
each pretraining corpus was (physiowave/fusion/prep_common.py).

SPLIT by subject, 25 / 5 / 5 (seed 42). sEMG from the pedal leg can carry
the label directly (stepping on the pedal IS the behaviour), which is why the
evaluation reports every modality subset and not only the complete one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.fusion.prep_common import (MODALITIES, MultimodalWriter,  # noqa: E402
                                           prepare_window, split_summary,
                                           subject_split)

CLASSES = ["smooth", "acceleration", "deceleration", "lane_change", "turning"]
#: Table 2 says overtaking is 135; the release carries 137 in its place
#: (alternating with 143 through the throttle experiment), so both map.
MARKER_CLASS = {133: 0, 135: 1, 137: 1, 143: 1, 139: 2, 141: 2, 145: 2,
                129: 3, 131: 3, 125: 4, 127: 4}
EMG_CHANNELS = ["r_tibialis_anterior", "r_gastrocnemius",
                "brachioradialis_a", "brachioradialis_b"]
ECG_CHANNELS = ["patch1"]           # one chest patch: the L1_250 route's slot
WINDOW_SECONDS = 2.0                # the release's epoch: -0.5 s .. +1.5 s
SEED = 42
FILE_RE = {"eeg": r"EEG_(\d+)\.set$", "ecg": r"ECG_(\d+)\.set$", "emg": r"EMG_(\d+)\.set$"}


def load_eeglab_epochs(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """``(data [n, C, T], marker codes [n], channel names, sfreq)`` from a .set.

    mne reads both the in-file and the .fdt-backed layouts. The marker code
    is the integer in the event name ('133', 'S133', 'condition 133' all
    give 133).
    """
    import mne
    ep = mne.io.read_epochs_eeglab(path, verbose="error")
    by_id = {v: k for k, v in ep.event_id.items()}
    codes = []
    for ev in ep.events[:, 2]:
        name = by_id.get(int(ev), str(ev))
        m = re.search(r"(\d+)", str(name))
        codes.append(int(m.group(1)) if m else -1)
    return ep.get_data(), np.asarray(codes, np.int64), list(ep.ch_names), float(ep.info["sfreq"])


def find_subjects(raw: str) -> Dict[int, Dict[str, str]]:
    found: Dict[int, Dict[str, str]] = {}
    for p in glob.glob(os.path.join(raw, "**", "*.set"), recursive=True):
        for m, rx in FILE_RE.items():
            mm = re.search(rx, os.path.basename(p), re.I)
            if mm:
                found.setdefault(int(mm.group(1)), {})[m] = p
    return found


def _lcs_pairs(a: np.ndarray, b: np.ndarray) -> List[Tuple[int, int]]:
    """Order-preserving pairing of ``a`` into ``b`` (longest common subsequence)."""
    n, m = len(a), len(b)
    L = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i, j] = L[i + 1, j + 1] + 1 if a[i] == b[j] else max(L[i + 1, j], L[i, j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif L[i + 1, j] >= L[i, j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def align(codes: Dict[str, np.ndarray]) -> Tuple[Optional[List[Tuple[int, ...]]], Dict]:
    """Epoch indices per modality that describe the same event, and how sure.

    The release carries no timestamps (no urevent, latencies are the epoch
    grid), only each file's ordered marker sequence, and the files differ:
    the EEG is pruned of rejected epochs, sometimes the ECG too, and one
    EMG file holds twice the lane-change epochs. So the shortest sequence
    anchors, and each other file is paired to it by the longest common
    subsequence of marker codes -- order preserved, a pruned epoch skipped.
    Every anchor epoch has to find a partner in every file, or the subject is
    refused. ``ambiguous`` counts pairs where the skipped stretch before the
    partner held the same code: the pairing there is right up to a neighbour
    of the same behaviour, which the label does not see but the fusion does.
    """
    mods = list(codes)
    anchor = min(mods, key=lambda m: len(codes[m]))
    a = codes[anchor]
    idx = {anchor: list(range(len(a)))}
    info = {"anchor": anchor, "epochs": {m: int(len(codes[m])) for m in mods},
            "ambiguous": 0, "unused": {}}
    for m in mods:
        if m == anchor:
            continue
        pairs = _lcs_pairs(a, codes[m])
        if len(pairs) != len(a):
            info["unmatched"] = {m: int(len(a) - len(pairs))}
            return None, info
        idx[m] = [j for _, j in pairs]
        info["unused"][m] = int(len(codes[m]) - len(pairs))
    # Ambiguity: an anchor epoch inside a run of one code whose partner run in
    # some file is longer. Which epoch of that run was pruned there is not
    # recoverable, so every pair in the run may be off by a same-behaviour
    # neighbour. Counted once per anchor epoch over all files.
    ambiguous = set()
    i = 0
    while i < len(a):
        k = i
        while k + 1 < len(a) and a[k + 1] == a[i]:
            k += 1
        for m in mods:
            if m == anchor:
                continue
            seq = codes[m]
            j0, j1 = idx[m][i], idx[m][k]
            while j0 > 0 and seq[j0 - 1] == a[i]:
                j0 -= 1
            while j1 + 1 < len(seq) and seq[j1 + 1] == a[i]:
                j1 += 1
            if j1 - j0 > k - i:
                ambiguous.update(range(i, k + 1))
        i = k + 1
    info["ambiguous"] = len(ambiguous)
    return [tuple(idx[m][k] for m in mods) for k in range(len(a))], info


def subject_windows(subj: int, files: Dict[str, str]) -> Tuple[List[dict], Dict]:
    loaded = {m: load_eeglab_epochs(files[m]) for m in MODALITIES if m in files}
    codes = {m: v[1] for m, v in loaded.items()}
    note = {"subject": subj, "files": {m: os.path.basename(p) for m, p in files.items()},
            "epochs": {m: int(len(c)) for m, c in codes.items()},
            "channels": {m: v[2] for m, v in loaded.items()},
            "sfreq": {m: v[3] for m, v in loaded.items()}}
    if set(loaded) != set(MODALITIES):
        note["dropped"] = f"missing {sorted(set(MODALITIES) - set(loaded))}"
        return [], note
    pairs, how = align(codes)
    note["alignment"] = how
    if pairs is None:
        note["dropped"] = (f"{how['unmatched']} anchor epoch(s) have no partner: the "
                           f"marker sequences disagree beyond pruning")
        return [], note
    items = []
    unknown = 0
    for k, idx in enumerate(pairs):
        code = int(codes["eeg"][idx[0]])
        if code not in MARKER_CLASS:
            unknown += 1
            continue
        item = {"label": MARKER_CLASS[code], "subject": f"mpdb_s{subj:02d}",
                "segment": f"mpdb_s{subj:02d}/m{code}/{k}"}
        for m, i in zip(MODALITIES, idx):
            data, _, names, fs = loaded[m]
            x = data[i]
            if m == "ecg":
                x = x[:1]                       # the release's ECG file: one channel
            if m == "emg":
                x = x[:4]
            item[m] = prepare_window(x, fs, m, WINDOW_SECONDS)
        items.append(item)
    note["unknown_markers"] = unknown
    note["windows"] = len(items)
    return items, note


def inspect_files(raw: str, n: int) -> int:
    """What is in the .set files: epochs, events, channels, the raw event fields.

    Run this on a login node before the first preprocessing: it shows how the
    three files of a subject can be aligned (identical sequences, the same
    multiset, or only through urevent numbers / latencies).
    """
    found = find_subjects(raw)
    for subj in sorted(found)[:n]:
        print(f"subject {subj}")
        for m in MODALITIES:
            path = found[subj].get(m)
            if not path:
                print(f"  {m}: MISSING")
                continue
            try:
                data, codes, names, fs = load_eeglab_epochs(path)
                hist = {int(c): int((codes == c).sum()) for c in np.unique(codes)}
                print(f"  {m}: {os.path.basename(path)}  epochs {data.shape[0]}  "
                      f"channels {data.shape[1]} x {data.shape[2]} samples @ {fs:g} Hz")
                print(f"      codes {hist}")
                print(f"      first 24: {codes[:24].tolist()}")
                print(f"      channels: {names[:70]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {m}: mne could not read it: {exc!r}")
            try:
                import scipy.io
                mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
                # EEGLAB writes the EEG struct's fields at the top level.
                get = (lambda k: mat.get(k)) if "event" in mat else \
                    (lambda k: getattr(mat.get("EEG"), k, None))

                def rows(arr, n=6):
                    arr = np.atleast_1d(arr)
                    f = list(getattr(arr[0], "_fieldnames", []))
                    out = [f"{arr.size} entries, fields {f}"]
                    for e in arr[:n]:
                        out.append("        " + ", ".join(
                            f"{k}={getattr(e, k)!r}"[:60] for k in f[:10]))
                    return "\n".join(out)
                print(f"      trials {get('trials')}  pnts {get('pnts')}  xmin {get('xmin')}  "
                      f"xmax {get('xmax')}  setname {get('setname')!r}")
                for key in ("event", "urevent", "epoch"):
                    v = get(key)
                    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
                        print(f"      {key}: none")
                    else:
                        print(f"      {key}: " + rows(v))
                ur = get("urevent")
                if ur is not None and np.atleast_1d(ur).size:
                    ur = np.atleast_1d(ur)
                    if hasattr(ur[0], "latency"):
                        lat = np.array([float(getattr(e, "latency", np.nan)) for e in ur])
                        typ = [str(getattr(e, "type", "")) for e in ur]
                        print(f"      urevent latencies (s): {np.round(lat[:12] / 1000.0, 3).tolist()}")
                        print(f"      urevent types: {typ[:12]}  intervals (s): "
                              f"{np.round(np.diff(lat[:12]) / 1000.0, 3).tolist()}")
            except NotImplementedError:
                print("      (MATLAB v7.3 file: fields not listed here)")
            except Exception as exc:  # noqa: BLE001
                print(f"      mat peek failed: {exc!r}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-subjects", type=int, default=5)
    p.add_argument("--test-subjects", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--max-subjects", type=int, default=None, help="a quick look")
    p.add_argument("--inspect", type=int, default=0, metavar="N",
                   help="print what the first N subjects' files contain and exit")
    args = p.parse_args(argv)
    if args.inspect:
        return inspect_files(args.raw_dir, args.inspect)
    t0 = time.time()

    found = find_subjects(args.raw_dir)
    if not found:
        raise SystemExit(f"no EEG_n.set / ECG_n.set / EMG_n.set under {args.raw_dir}")
    subjects = sorted(found)[: args.max_subjects] if args.max_subjects else sorted(found)
    print(f"mpdb: {len(subjects)} subject(s) under {args.raw_dir}", flush=True)

    # Load everything first: the split is drawn over the subjects that
    # actually yield windows, so a dropped subject cannot leave a split empty.
    loaded, notes, channels = {}, [], None
    for subj in subjects:
        items, note = subject_windows(subj, found[subj])
        notes.append(note)
        if not items:
            print(f"  subject {subj}: DROPPED -- {note.get('dropped')}", flush=True)
            continue
        eeg_names = [str(n) for n in note["channels"]["eeg"]]
        if channels is None:
            channels = {"eeg": eeg_names, "ecg": ECG_CHANNELS, "emg": EMG_CHANNELS}
        elif eeg_names != channels["eeg"]:
            note["dropped"] = "EEG channel names differ from the first subject's"
            print(f"  subject {subj}: DROPPED -- {note['dropped']}", flush=True)
            continue
        loaded[subj] = items
        how = note["alignment"]
        print(f"  subject {subj}: {len(items)} windows  (anchor {how['anchor']}, epochs "
              f"{how['epochs']}, ambiguous pairs {how['ambiguous']}, unused {how['unused']})",
              flush=True)
    if not loaded:
        raise SystemExit("every subject was dropped; see the notes above")
    side = subject_split([f"mpdb_s{s:02d}" for s in loaded],
                         args.val_subjects, args.test_subjects, args.split_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writers = {s: MultimodalWriter(os.path.join(args.out_dir, f"{s}.h5"), "mpdb",
                                   WINDOW_SECONDS, CLASSES, channels,
                                   {"marker_class": MARKER_CLASS, "source": "figshare 22192831"})
               for s in ("train", "val", "test")}
    stats = {s: {"windows": 0, "subjects": set(), "per_class": np.zeros(len(CLASSES), np.int64),
                 "present": {m: 0 for m in MODALITIES}} for s in writers}
    for subj, items in loaded.items():
        s = side[f"mpdb_s{subj:02d}"]
        for it in items:
            writers[s].add(it)
            st = stats[s]
            st["windows"] += 1
            st["subjects"].add(it["subject"])
            st["per_class"][it["label"]] += 1
            for m in MODALITIES:
                st["present"][m] += int(it.get(m) is not None)
    for w in writers.values():
        w.close()
    summary = {"task": "mpdb", "classes": CLASSES, "channels": channels,
               "window_seconds": WINDOW_SECONDS, "marker_class": MARKER_CLASS,
               "splits": split_summary(stats),
               "subject_split": side, "subjects": notes,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:6,} windows  {v['subjects']} subjects  "
              f"per class {v['per_class']}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json in "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
