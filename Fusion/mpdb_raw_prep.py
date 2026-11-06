#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPDB from the RAW release -> aligned EEG + ECG + sEMG windows, on our own terms.

    python Fusion/mpdb_raw_prep.py --raw-dir $R/mpdb_raw --out-dir $D/mpdb
    python Fusion/mpdb_raw_prep.py --raw-dir $R/mpdb_raw --out-dir /tmp/x --inspect 2

THE RAW RELEASE (figshare 22193119, 23 GB): per subject four experiments --
brake, throttle, turn, change -- each a CONTINUOUS EEGLAB .set file per
device: EEG&ECG (59 electrodes + the chest ECG, one Neuracle amplifier, 1 kHz)
and EMG (four limb channels, NeuSen WM, 1 kHz), with the simulator's event
markers in each file's event list. Unlike the preprocessed release this has
timestamps, so an EMG window is paired with the EEG/ECG window of THE SAME
EVENT: within an experiment the two files' marker sequences are matched in
order (identical, or the longest common subsequence when one device missed
a trigger) and the inter-event intervals are compared -- a pairing whose
intervals disagree by more than --tolerance seconds is refused, not kept.

WINDOWS of --window seconds (4) starting --pre seconds (0.5) before the
marker, so the event's onset sits where the preprocessed release put it and
the window is the EEG encoder's pretraining length. Each record is filtered
whole with its modality's pretraining pipeline (physiowave/fusion/prep_common.py:
EEG detrend + 50 Hz notch + 0.5 Hz; ECG notch + 0.5 Hz; EMG notch + 20 Hz),
then cut, resampled to the encoder rate and normalised as that corpus was.
No ICA, no epoch rejection: what the authors pruned stays in, and quality is
the reliability estimator's problem, which is the point of the fusion.

LABELS from the marker codes (smooth 133; acceleration 135/137/143;
deceleration 139/141/145; lane change 129/131; turning 125/127). SPLIT by
subject, 25 / 5 / 5 (seed 42). ECG is the EEG file's ECG channel, named
``patch1`` so the L1_250 route's frontend transfers.
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
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from mpdb_prep import (CLASSES, ECG_CHANNELS, EMG_CHANNELS, MARKER_CLASS,  # noqa: E402
                       SEED, _lcs_pairs)
from physiowave.fusion.prep_common import (MODALITIES, MultimodalWriter,  # noqa: E402
                                           cut, filter_continuous, prepare_window,
                                           split_summary, subject_split)

EXPERIMENTS = ("brake", "throttle", "turn", "change")
MAINS_HZ = 50.0


def load_eeglab_raw(path: str) -> Tuple[np.ndarray, List[str], float, List[Tuple[float, int]]]:
    """``(data [C, T], channel names, sfreq, [(onset s, marker code)])`` from a continuous .set."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    events = []
    for on, desc in zip(raw.annotations.onset, raw.annotations.description):
        m = re.search(r"(\d+)", str(desc))
        if m:
            events.append((float(on), int(m.group(1))))
    return raw.get_data(), list(raw.ch_names), float(raw.info["sfreq"]), events


#: The "Rawdata (Additional)" archive adds five participants in another layout:
#: additiondrive/sub<k>/<experiment>/<EEG|EMG|GSR>/, each folder a Neuracle
#: recording -- data.bdf (the signal), evt.bdf (the markers), sometimes a
#: second, dated .bdf export -- or, where someone already converted it, a
#: merged.set. They follow the main archive's thirty: sub<k> is subject 30 + k.
ADDITIONAL_OFFSET = 30
#: The four EMG channels, as the two archives name them: the .set files S01
#: S05 S09 S13 (sensor slots), the Neuracle folders EMG1..EMG4 among each
#: sensor's gyroscope, accelerometer and magnetometer rows.
EMG_NAME_SETS = (("S01", "S05", "S09", "S13"), ("EMG1", "EMG2", "EMG3", "EMG4"))


def emg_rows(names: List[str]) -> List[int]:
    for want in EMG_NAME_SETS:
        if all(n in names for n in want):
            return [names.index(n) for n in want]
    return [i for i, n in enumerate(names)
            if not re.search(r"trig|stim|marker|GYR|ACC|MAG", n, re.I)][:4]


def find_files(raw: str) -> Dict[int, Dict[str, Dict[str, str]]]:
    """``{subject: {experiment: {"eegecg": source, "emg": source}}}``.

    A source is a .set file (the main archive) or a Neuracle folder (the
    additional one); ``load_record`` reads either.
    """
    found: Dict[int, Dict[str, Dict[str, str]]] = {}
    for p in glob.glob(os.path.join(raw, "**", "*.set"), recursive=True):
        m = re.match(r"(\d+)_(brake|throttle|turn|change)\.set$", os.path.basename(p), re.I)
        if not m:
            continue
        parent = os.path.basename(os.path.dirname(p)).upper()
        kind = "emg" if "EMG" in parent else "eegecg" if "EEG" in parent else None
        if kind:
            found.setdefault(int(m.group(1)), {}).setdefault(m.group(2).lower(), {})[kind] = p
    for d in glob.glob(os.path.join(raw, "**", "sub*", "*", "*"), recursive=True):
        if not os.path.isdir(d):
            continue
        m = re.search(r"/sub(\d+)/(brake|throttle|turn|change)/(EEG|EMG)$", d.replace("\\", "/"), re.I)
        if not m:
            continue
        kind = "eegecg" if m.group(3).upper() == "EEG" else "emg"
        subj, exp = ADDITIONAL_OFFSET + int(m.group(1)), m.group(2).lower()
        sets = sorted(glob.glob(os.path.join(d, "*.set")))
        bdfs = [q for q in glob.glob(os.path.join(d, "*.bdf"))
                if os.path.basename(q).lower() != "evt.bdf"]
        if bdfs:
            found.setdefault(subj, {}).setdefault(exp, {})[kind] = d
        elif sets:
            merged = [q for q in sets if os.path.basename(q).lower() == "merged.set"]
            found.setdefault(subj, {}).setdefault(exp, {})[kind] = (merged or sets)[0]
    return found


def neuracle_files(folder: str) -> List[dict]:
    """Every signal .bdf in a Neuracle folder with its channels, rate, length."""
    import mne
    out = []
    for q in sorted(glob.glob(os.path.join(folder, "*.bdf"))):
        if os.path.basename(q).lower() == "evt.bdf":
            continue
        raw = mne.io.read_raw_bdf(q, preload=False, verbose="error")
        out.append({"path": q, "channels": list(raw.ch_names), "sfreq": float(raw.info["sfreq"]),
                    "seconds": raw.n_times / float(raw.info["sfreq"]),
                    "annotations": int(len(raw.annotations))})
    return out


def choose_neuracle(folder: str, kind: str) -> dict:
    """Which .bdf carries the signal: for EMG a file naming the four EMG
    channels, preferring the one that also carries the markers; for EEG
    data.bdf, else the one with the most channels."""
    files = neuracle_files(folder)
    if not files:
        raise FileNotFoundError(f"no signal .bdf in {folder}")
    if kind == "emg":
        named = [f for f in files if any(all(n in f["channels"] for n in want)
                                         for want in EMG_NAME_SETS)]
        pool = named or files
        # the export that carries the markers, else the one with the fewest channels
        pick = max(pool, key=lambda f: (f["annotations"] > 0, -len(f["channels"])))
    else:
        data = [f for f in files if os.path.basename(f["path"]).lower() == "data.bdf"]
        pick = data[0] if data else max(files, key=lambda f: len(f["channels"]))
    return pick


def load_neuracle(folder: str, kind: str) -> Tuple[np.ndarray, List[str], float, List[Tuple[float, int]]]:
    """A Neuracle recording folder -> ``(data [C, T], names, sfreq, [(onset s, code)])``.

    The markers come from evt.bdf (a BDF+ file of annotations) when it has
    them, else from the signal file's own annotations. An evt.bdf whose
    recorded start differs from the signal file's is shifted accordingly.
    """
    import mne
    pick = choose_neuracle(folder, kind)
    raw = mne.io.read_raw_bdf(pick["path"], preload=True, verbose="error")

    def numeric(ann):
        out = []
        for on, desc in zip(ann.onset, ann.description):
            m = re.search(r"(\d+)", str(desc))
            if m:
                out.append((float(on), int(m.group(1))))
        return out
    events = numeric(raw.annotations)
    evt = os.path.join(folder, "evt.bdf")
    if os.path.isfile(evt):
        ann = mne.read_annotations(evt)
        shift = 0.0
        if ann.orig_time is not None and raw.info.get("meas_date") is not None:
            shift = (ann.orig_time - raw.info["meas_date"]).total_seconds()
        ev2 = [(t + shift, c) for t, c in numeric(ann)]
        if len(ev2) > len(events):
            events = ev2
    return raw.get_data(), list(raw.ch_names), float(raw.info["sfreq"]), events


def load_record(source: str, kind: str) -> Tuple[np.ndarray, List[str], float, List[Tuple[float, int]]]:
    """A .set file or a Neuracle folder, whichever the archive holds."""
    if os.path.isdir(source):
        return load_neuracle(source, kind)
    return load_eeglab_raw(source)


def split_eeg_ecg(data: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """The EEG&ECG file's rows -> (eeg [59, T], eeg names, ecg [1, T])."""
    ecg_rows = [i for i, n in enumerate(names) if "ECG" in n.upper() or "EKG" in n.upper()]
    # The release's auxiliaries: ECG, and the four EOG channels named HEOR,
    # HEOL, VEOU, VEOL (horizontal right/left, vertical up/low).
    drop = set(ecg_rows) | {i for i, n in enumerate(names)
                            if re.search(r"trig|stim|marker|status|EOG|GSR|^[HV]EO[RLUD]$", n, re.I)}
    eeg_rows = [i for i in range(len(names)) if i not in drop]
    ecg = data[ecg_rows[:1]] if ecg_rows else None
    return data[eeg_rows], [names[i] for i in eeg_rows], ecg


def _offset_mode(ta, ca, tb, cb, seed, avail_b):
    """The clock offset for the EEG events ``seed``: the mode of their same-code
    time differences to the available EMG events. Right pairs pile into one
    20 ms bin; wrong pairs scatter over the 300 s range."""
    diffs = []
    for i in seed:
        ib = np.where((cb == ca[i]) & avail_b)[0]
        if len(ib):
            d = tb[ib] - ta[i]
            diffs.append(d[np.abs(d) < 300.0])
    diffs = np.concatenate(diffs) if diffs else np.zeros(0)
    if diffs.size < 2:
        return None
    edges = np.arange(diffs.min() - 0.02, diffs.max() + 0.04, 0.02)
    hist, _ = np.histogram(diffs, bins=edges)
    k = int(np.argmax(hist))
    lo, hi = edges[max(0, k - 2)], edges[min(len(edges) - 1, k + 3)]
    return float(np.median(diffs[(diffs >= lo) & (diffs <= hi)]))


def pair_events(a: List[Tuple[float, int]], b: List[Tuple[float, int]], tol: float,
                seed_events: int = 60, break_after: int = 8
                ) -> Tuple[List[Tuple[int, int]], Dict]:
    """Indices into ``a`` (EEG&ECG file) and ``b`` (EMG file) of the same events.

    The two devices share the simulator's trigger stream but not a clock: the
    EMG file is offset by a constant plus a drift of ~20 ppm, may hold a few
    events the other file does not, and -- when a device was stopped and
    restarted inside an experiment (the extra 69/71 start/stop markers in a
    throttle file) -- its time axis jumps, so one offset explains only part
    of the recording. Codes repeat hundreds of times, so order is ambiguous;
    time is not.

    The pairing marches through the EEG events in time. A clock segment
    starts with the offset estimated from the next ``seed_events`` events
    (the mode of their same-code time differences) and a drift fitted to
    what that offset matches; each following event takes the nearest
    same-code EMG event within ``tol`` of its predicted time, with the fit
    refreshed as pairs accumulate. ``break_after`` consecutive events without
    a partner end the segment, and the next one is estimated afresh from
    there -- that is what a restart looks like. A lone missed trigger does
    not break a segment. Unpaired events leave their window without EMG.
    """
    if not a or not b:
        return [], {"events_eeg": len(a), "events_emg": len(b), "paired": 0}
    ta, ca = np.array([t for t, _ in a]), np.array([c for _, c in a])
    tb, cb = np.array([t for t, _ in b]), np.array([c for _, c in b])
    order = np.argsort(ta)
    avail_b = np.ones(len(b), bool)
    pairs: List[Tuple[int, int]] = []
    segments = []
    resid: List[float] = []

    def nearest(i, off, scale, tol_):
        pred = scale * ta[i] + off
        cand = np.where((cb == ca[i]) & avail_b & (np.abs(tb - pred) <= tol_))[0]
        return int(cand[np.argmin(np.abs(tb[cand] - pred))]) if cand.size else None

    def refit(seg, off, scale):
        if len(seg) < 3:
            return off, scale
        xa = ta[[i for i, _ in seg[-200:]]]
        xb = tb[[j for _, j in seg[-200:]]]
        sc, of = np.polyfit(xa, xb, 1)
        return float(of), float(sc)

    pos = 0
    while pos < len(order):
        seed = order[pos:pos + seed_events]
        off = _offset_mode(ta, ca, tb, cb, seed, avail_b)
        if off is None:
            break
        scale = 1.0
        # the drift, from what the offset alone explains inside the seed
        trial = [(int(i), j) for i in seed for j in [nearest(i, off, scale, 0.1)] if j is not None]
        off, scale = refit(trial, off, scale)
        seg, fails, k, end_k = [], 0, pos, pos
        while k < len(order):
            i = int(order[k])
            j = nearest(i, off, scale, tol)
            if j is None:
                fails += 1
                if fails >= break_after and len(seg) >= 3:
                    break
            else:
                fails = 0
                seg.append((i, j))
                resid.append(abs(scale * ta[i] + off - tb[j]))
                avail_b[j] = False
                end_k = k
                if len(seg) % 50 == 0:
                    off, scale = refit(seg, off, scale)
            k += 1
        if len(seg) >= 3:
            pairs += seg
            segments.append({"paired": len(seg), "clock_offset_s": round(off, 3),
                             "clock_drift_ppm": round((scale - 1.0) * 1e6, 1),
                             "from_s": round(float(ta[seg[0][0]]), 1),
                             "to_s": round(float(ta[seg[-1][0]]), 1)})
            pos = end_k + 1
        else:
            for i, j in seg:
                avail_b[j] = True
            del resid[len(resid) - len(seg):]
            pos += max(1, break_after)
    pairs.sort()
    info = {"events_eeg": len(a), "events_emg": len(b), "paired": len(pairs),
            "unpaired_eeg": len(a) - len(pairs), "unused_emg": len(b) - len(pairs),
            "max_residual_ms": round(1000 * float(max(resid)), 1) if resid else None,
            "segments": segments}
    if segments:
        info["clock_offset_s"] = segments[0]["clock_offset_s"]
        info["clock_drift_ppm"] = segments[0]["clock_drift_ppm"]
    return pairs, info


def subject_windows(subj: int, exps: Dict[str, Dict[str, str]], window: float, pre: float,
                    tol: float) -> Tuple[List[dict], Dict]:
    items: List[dict] = []
    note: Dict = {"subject": subj, "experiments": {}}
    eeg_names_ref: Optional[List[str]] = None
    for exp in EXPERIMENTS:
        files = exps.get(exp, {})
        en: Dict = {"files": {k: os.path.basename(v) for k, v in files.items()}}
        note["experiments"][exp] = en
        if "eegecg" not in files:
            en["skipped"] = "no EEG&ECG file"
            continue
        data, names, fs, ev_a = load_record(files["eegecg"], "eegecg")
        eeg, eeg_names, ecg = split_eeg_ecg(data, names)
        if eeg_names_ref is None:
            eeg_names_ref = eeg_names
        elif eeg_names != eeg_names_ref:
            en["skipped"] = f"EEG channels differ from {EXPERIMENTS[0]}'s: {eeg_names[:5]}..."
            continue
        en.update({"eeg_channels": len(eeg_names), "sfreq": fs, "seconds": eeg.shape[1] / fs,
                   "ecg": ecg is not None, "codes": {int(c): int(sum(1 for _, cc in ev_a if cc == c))
                                                     for c in sorted({cc for _, cc in ev_a})}})
        streams = {"eeg": (filter_continuous(eeg, fs, "eeg", MAINS_HZ), fs)}
        if ecg is not None:
            streams["ecg"] = (filter_continuous(ecg, fs, "ecg", MAINS_HZ), fs)
        emg_events: Dict[int, float] = {}
        if "emg" in files:
            d2, n2, fs2, ev_b = load_record(files["emg"], "emg")
            rows = emg_rows(n2)
            emg = d2[rows]
            kept, how = pair_events(ev_a, ev_b, tol)
            en["emg_pairing"] = how
            en["emg_channels"] = [n2[i] for i in rows]
            if emg.shape[0] == 4:
                streams["emg"] = (filter_continuous(emg, fs2, "emg", MAINS_HZ), fs2)
                emg_events = {i: ev_b[j][0] for i, j in kept}
            else:
                en["emg_skipped"] = f"{emg.shape[0]} EMG channels, not 4"
        else:
            en["emg_skipped"] = "no EMG file"
        n_win = 0
        for k, (t, code) in enumerate(ev_a):
            if code not in MARKER_CLASS:
                continue
            item = {"label": MARKER_CLASS[code], "subject": f"mpdb_s{subj:02d}",
                    "segment": f"mpdb_s{subj:02d}/{exp}/m{code}/{k}"}
            x, r = streams["eeg"]
            seg = cut(x, r, t - pre, window)
            if seg is None:
                continue
            item["eeg"] = prepare_window(seg, r, "eeg", window)
            if "ecg" in streams:
                x, r = streams["ecg"]
                seg = cut(x, r, t - pre, window)
                item["ecg"] = prepare_window(seg, r, "ecg", window) if seg is not None else None
            else:
                item["ecg"] = None
            item["emg"] = None
            if "emg" in streams and k in emg_events:
                x, r = streams["emg"]
                seg = cut(x, r, emg_events[k] - pre, window)
                if seg is not None:
                    item["emg"] = prepare_window(seg, r, "emg", window)
            items.append(item)
            n_win += 1
        en["windows"] = n_win
    note["eeg_channels"] = eeg_names_ref
    note["windows"] = len(items)
    return items, note


def inspect_files(raw: str, n: int, tol: float) -> int:
    found = find_files(raw)
    print(f"{len(found)} subject(s) with raw files under {raw}: {sorted(found)}")
    for subj in sorted(found)[:n]:
        print(f"subject {subj}")
        for exp in EXPERIMENTS:
            files = found[subj].get(exp, {})
            print(f"  {exp}: {', '.join(f'{k}={os.path.basename(v)}' for k, v in files.items()) or 'no files'}")
            evs = {}
            for kind, path in files.items():
                if os.path.isdir(path):
                    for f in neuracle_files(path):
                        print(f"    {kind} folder: {os.path.basename(f['path'])}  {len(f['channels'])} ch x "
                              f"{f['seconds']:.0f} s @ {f['sfreq']:g} Hz, {f['annotations']} annotations; "
                              f"channels {f['channels'][:8]}{' ...' if len(f['channels']) > 8 else ''}")
                    print(f"    {kind} folder: using {os.path.basename(choose_neuracle(path, kind)['path'])}")
                try:
                    data, names, fs, ev = load_record(path, kind)
                except Exception as exc:  # noqa: BLE001
                    print(f"    {kind}: mne could not read it: {exc!r}")
                    continue
                evs[kind] = ev
                hist = {}
                for _, c in ev:
                    hist[c] = hist.get(c, 0) + 1
                print(f"    {kind}: {data.shape[0]} ch x {data.shape[1] / fs:.0f} s @ {fs:g} Hz; "
                      f"events {len(ev)} {dict(sorted(hist.items()))}")
                print(f"      channels ({len(names)}): {names}")
                beh = {c: n for c, n in sorted(hist.items()) if c in MARKER_CLASS}
                print(f"      behaviour markers: {beh}  (others are simulator state codes)")
                print(f"      first events (s, code): {[(round(t, 2), c) for t, c in ev[:8]]}")
            if "eegecg" in evs and "emg" in evs:
                kept, how = pair_events(evs["eegecg"], evs["emg"], tol)
                print(f"    pairing EEG<->EMG: {how}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--pre", type=float, default=0.5, help="seconds before the marker")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="max disagreement (s) between the devices' inter-event intervals")
    p.add_argument("--val-subjects", type=int, default=5)
    p.add_argument("--test-subjects", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--inspect", type=int, default=0, metavar="N")
    args = p.parse_args(argv)
    if args.inspect:
        return inspect_files(args.raw_dir, args.inspect, args.tolerance)
    t0 = time.time()
    found = find_files(args.raw_dir)
    if not found:
        raise SystemExit(f"no <n>_{{brake,throttle,turn,change}}.set under {args.raw_dir}")
    subjects = sorted(found)[: args.max_subjects] if args.max_subjects else sorted(found)
    print(f"mpdb raw: {len(subjects)} subject(s) under {args.raw_dir}", flush=True)

    loaded, notes, channels = {}, [], None
    for subj in subjects:
        items, note = subject_windows(subj, found[subj], args.window, args.pre, args.tolerance)
        notes.append(note)
        if not items:
            print(f"  subject {subj}: DROPPED -- no windows", flush=True)
            continue
        if channels is None:
            channels = {"eeg": note["eeg_channels"], "ecg": ECG_CHANNELS, "emg": EMG_CHANNELS}
        elif note["eeg_channels"] != channels["eeg"]:
            note["dropped"] = "EEG channel names differ from the first subject's"
            print(f"  subject {subj}: DROPPED -- {note['dropped']}", flush=True)
            continue
        loaded[subj] = items
        with_emg = sum(1 for it in items if it["emg"] is not None)
        with_ecg = sum(1 for it in items if it["ecg"] is not None)
        pairing = " ".join(
            f"{e}:{v.get('windows', 0)}w/emg {v['emg_pairing']['paired']}/{v['emg_pairing']['events_eeg']}"
            if "emg_pairing" in v else f"{e}:{v.get('windows', 0)}w/{v.get('skipped', v.get('emg_skipped', 'no emg'))}"
            for e, v in note["experiments"].items())
        print(f"  subject {subj}: {len(items)} windows ({with_ecg} with ECG, {with_emg} with EMG)  "
              f"{pairing}", flush=True)
    if not loaded:
        raise SystemExit("every subject was dropped; see the notes above")
    side = subject_split([f"mpdb_s{s:02d}" for s in loaded], args.val_subjects,
                         args.test_subjects, args.split_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writers = {s: MultimodalWriter(os.path.join(args.out_dir, f"{s}.h5"), "mpdb", args.window,
                                   CLASSES, channels,
                                   {"marker_class": MARKER_CLASS, "source": "figshare 22193119 (raw)",
                                    "pre_seconds": args.pre})
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
    summary = {"task": "mpdb", "source": "raw", "classes": CLASSES, "channels": channels,
               "window_seconds": args.window, "pre_seconds": args.pre,
               "tolerance_seconds": args.tolerance, "marker_class": MARKER_CLASS,
               "splits": split_summary(stats), "subject_split": side, "subjects": notes,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:6,} windows  {v['subjects']} subjects  "
              f"per class {v['per_class']}  present {v['present']}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
