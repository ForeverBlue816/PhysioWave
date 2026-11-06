#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess one EEG corpus into the C1 pretraining HDF5 schema.

    python EEG/preprocess_pretrain_corpus.py \
        --dataset faced --root /path/to/raw --out-dir /path/to/processed/faced

One dataset per invocation, because they are acquired at different rates on
different montages and the failures are dataset-shaped. Every run writes
manifest_train.jsonl, manifest_val.jsonl, dataset_statistics.json and
preprocessing_failures.jsonl into --out-dir; a recording that cannot be read
lands in the failures file with its reason and does not stop the run.

WHAT THIS DOES NOT DO. It does not download anything. TUEG, HBN, M3CV and
TDBRAIN are all behind data use agreements, and a script that fetches them on
the user's behalf is a script that agrees to a licence on the user's behalf.
Point --root at data you already have.

SYNTHETIC DATA. --smoke-test generates noise so the pipeline itself can be
exercised without any corpus present. It is the only path that produces
synthetic windows, it stamps ``"synthetic": true`` into every shard's
provenance, and it writes to a directory it names ``smoke_*``. Nothing falls
back to it: a missing --root is an error, not a reason to invent data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_embedding import (CHANNEL_TO_ID,             # noqa: E402
                               channel_ids_for, normalize_channel_name)
from physiowave.eeg_c1.preprocess import (                # noqa: E402
    PreprocessConfig, PreprocessError, detrend, highpass, map_to_slots, notch,
    place_on_slots, resample_to, split_subjects, subject_split_side,
    to_microvolts, window_qc, window_signal, write_shard, zscore_windows)
from physiowave.eeg_c1.routes import EEG_DATASETS, PRETRAIN_DATASETS, ROUTES  # noqa: E402


# --------------------------------------------------------------------------- #
# What an adapter returns
# --------------------------------------------------------------------------- #

@dataclass
class Recording:
    """One readable recording, before any processing."""

    recording_id: str
    subject_id: str
    data: np.ndarray            # [C, T]
    channel_names: List[str]
    sampling_rate: float
    unit: str = "uV"
    mains_hz: Optional[float] = None
    notes: Dict = None


# --------------------------------------------------------------------------- #
# Adapters
#
# Each yields Recording objects. They differ only in how a file is opened and
# which channels are kept; everything downstream is shared.
#
# NOT ONE OF THESE HAS BEEN RUN AGAINST ITS REAL CORPUS in this environment --
# no raw data is present here. They are written against each dataset's published
# format and the reader they need; the file layout assumptions each one makes
# are stated in its docstring so they can be checked against a real tree before
# a long run. --dry-run lists what an adapter would read without processing it.
# --------------------------------------------------------------------------- #

def _require_mne():
    try:
        import mne
    except ImportError as exc:                       # pragma: no cover
        raise PreprocessError(
            "mne is needed to read this corpus and is not installed in this "
            "interpreter. On CINECA: source $HOME/pwprep/bin/activate") from exc
    mne.set_log_level("ERROR")
    return mne


#: Every container these corpora actually ship in. ONE list, because six
#: adapters with six hand-maintained extension tuples is how M3CV -- 2,469
#: BrainVision recordings -- came to report "no readable recording": its adapter
#: walked .set/.edf/.fif/.cnt and the release is .vhdr.
#:
#: Only the HEADER of a multi-file format is listed. BrainVision is a .vhdr
#: pointing at a .eeg and a .vmrk, and walking .eeg as well would read every
#: recording twice.
READABLE_EXTS: Tuple[str, ...] = (
    ".edf", ".bdf", ".set", ".fif", ".cnt", ".vhdr", ".mff",
)


def read_raw_any(path: str, preload: bool = True):
    """Open a recording by what it IS, not by which adapter reached it.

    Each adapter used to carry its own if/elif chain over extensions, so a
    corpus published in a format its adapter's chain did not mention fell
    through to read_raw_edf and failed on a file that was never an EDF.
    """
    mne = _require_mne()
    low = path.lower()
    if low.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    if low.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    if low.endswith(".fif"):
        return mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    if low.endswith(".cnt"):
        return mne.io.read_raw_cnt(path, preload=preload, verbose="ERROR")
    if low.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=preload, verbose="ERROR")
    if low.endswith(".mff"):
        return mne.io.read_raw_egi(path, preload=preload, verbose="ERROR")
    return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")


def _walk(root: str, exts: Tuple[str, ...], cache: Optional[str] = None,
          quiet: bool = False) -> List[str]:
    """Every recording under root, with progress, and optionally cached.

    Silence here is indistinguishable from a hang. Walking a BIDS tree of tens
    of thousands of entries on a parallel filesystem takes minutes, and this
    printed nothing at all until it finished -- which is how TUEG's tree walk
    came to look like two frozen array tasks. It now says what it is doing
    within a couple of seconds and keeps counting.

    `cache` reads the listing if the file exists and writes it if it does not,
    so an array does not walk the same tree once per task.
    """
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            hits = [ln.strip() for ln in f if ln.strip()]
        if not quiet:
            print(f"  {len(hits):,} file(s) from the cached listing {cache}",
                  flush=True)
        return hits

    t0 = time.time()
    hits: List[str] = []
    n_dirs = 0
    announced = False
    for dirpath, _, files in os.walk(root):
        n_dirs += 1
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                hits.append(os.path.join(dirpath, fn))
        if not quiet and not announced and time.time() - t0 > 2.0:
            print(f"  scanning {root} for {'/'.join(exts)} ...", flush=True)
            announced = True
        if announced and n_dirs % 2000 == 0:
            print(f"    {n_dirs:,} directories, {len(hits):,} recording(s), "
                  f"{time.time()-t0:.0f}s", flush=True)
    hits = sorted(hits)
    if announced:
        print(f"  scan finished: {len(hits):,} recording(s) in "
              f"{n_dirs:,} directories, {time.time()-t0:.0f}s", flush=True)
    if cache and hits:
        tmp = f"{cache}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            f.write("\n".join(hits) + "\n")
        os.replace(tmp, cache)
        if not quiet:
            print(f"  cached the listing at {cache}", flush=True)
    return hits


def shard_files(files: List[str], args, root: str) -> List[str]:
    """Keep only the files this array task owns, sharded BY SUBJECT.

    By subject and not by file: a subject's recordings must all be processed by
    one task, because the train/val side is decided per subject and a subject
    split across two tasks that disagree would be the leak the subject-level
    split exists to prevent. Hashing the subject id also balances the tasks
    without anyone counting files first.
    """
    idx, total = args.shard
    return [p for p in files
            if subject_shard(tueg_identity(p, root)["subject"], total) == idx]


def subject_shard(subject_id: str, total: int) -> int:
    """Which array task owns this subject. The one definition of it."""
    h = int.from_bytes(hashlib.sha256(str(subject_id).encode()).digest()[:4], "big")
    return h % total


#: TUH filenames carry the identity: ``aaaaaaaa_s001_t000.edf`` is subject
#: aaaaaaaa, session s001, token t000. Parsing the NAME rather than counting
#: directory levels is what makes this survive the layout differences between
#: TUEG releases (v1.x had edf/<subject>/..., v2.x inserts a bucket directory),
#: and between TUEG and TUAB (which groups by train/eval and normal/abnormal
#: instead of by subject at all).
TUH_FILENAME = re.compile(r"^(?P<subject>[a-z0-9]{4,12})_s(?P<session>\d{2,4})"
                          r"_t(?P<token>\d{2,4})\.edf$", re.IGNORECASE)

#: The montage a TUH session was recorded under, as its directory is named.
#: Recorded in provenance: 01_tcp_ar and 02_tcp_le are different references, and
#: a corpus that turns out to be a mixture is something to know about rather
#: than average over.
TUH_MONTAGE_DIRS = ("01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a",
                    "05_tcp_ar_a")


def tueg_identity(path: str, root: str) -> Dict[str, str]:
    """``subject``/``session``/``montage`` for one TUH EDF, from its name.

    Falls back to the parent directory only when the filename does not follow
    the convention, and says which rule fired, so an --inspect run shows whether
    the fallback is carrying the corpus.
    """
    base = os.path.basename(path)
    m = TUH_FILENAME.match(base)
    parts = os.path.normpath(os.path.relpath(path, root)).split(os.sep)
    montage = next((p for p in parts if p in TUH_MONTAGE_DIRS), "")
    if m:
        return {"subject": m.group("subject"),
                "session": f"s{m.group('session')}",
                "montage": montage, "rule": "filename"}
    # Fallback: the directory above the montage directory is the session, and
    # the one above that the subject, in every TUH layout that has one.
    subject = session = ""
    if montage and montage in parts:
        i = parts.index(montage)
        session = parts[i - 1] if i >= 1 else ""
        subject = parts[i - 2] if i >= 2 else ""
    if not subject:
        subject = parts[-2] if len(parts) > 1 else "unknown"
    return {"subject": subject, "session": session, "montage": montage,
            "rule": "path"}


def iter_tueg_files(root: str, cache: Optional[str] = None) -> List[str]:
    """Every .edf under root. TUEG puts them all under ``edf/``; not assumed.

    Walking this tree is not cheap: seventy thousand files under a hundred
    thousand directories, on a parallel filesystem whose metadata server is the
    bottleneck. Sixteen array tasks each doing it independently made two of them
    spend forty-four minutes before processing a single file, while the tasks
    whose walk happened to be served first finished the whole shard in eleven.

    So the listing is cached to a text file and shared. Written atomically, so
    a task reading it while another writes gets the old file or the new one and
    never half of one. Build it once, before submitting the array:

        python EEG/preprocess_pretrain_corpus.py --dataset tueg --root <root> \
            --out-dir <out> --write-file-list <out>/tueg_files.txt
    """
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        if paths:
            return paths
    edf_root = os.path.join(root, "edf")
    paths = _walk(edf_root if os.path.isdir(edf_root) else root, (".edf",))
    if cache and paths:
        tmp = f"{cache}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            f.write("\n".join(paths) + "\n")
        os.replace(tmp, cache)
    return paths


#: Per-corpus acquisition facts (mains frequency, expected native rates,
#: whether the publisher already notched). Loaded from configs so the same fact
#: is not restated in six shell scripts, and so a corpus whose mains frequency
#: is a guess is marked as one.
DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "pretrain", "eeg_c1_datasets.yaml")

_REGISTRY_CACHE: Dict[str, Dict] = {}


def load_registry(path: Optional[str] = None) -> Dict[str, Dict]:
    """Read the dataset registry. Missing file is fatal, not defaulted.

    Defaulting a mains frequency is exactly the failure this table exists to
    prevent, so an unreadable registry stops the run rather than falling back
    to 50 Hz for everything.
    """
    path = path or DEFAULT_REGISTRY
    if path in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[path]
    try:
        import yaml
    except ImportError as exc:                                  # pragma: no cover
        raise PreprocessError(f"pyyaml is needed to read {path}") from exc
    if not os.path.isfile(path):
        raise PreprocessError(
            f"dataset registry {path} not found. It carries the mains "
            f"frequency per corpus; guessing one silently removes real signal.")
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    _REGISTRY_CACHE[path] = reg
    return reg


def registry_for(dataset_id: str, args=None) -> Dict:
    """The registry entry for one corpus, or {} when it has none."""
    try:
        reg = load_registry(getattr(args, "registry", None))
    except PreprocessError:
        if getattr(args, "mains_hz", None) is not None:
            return {}          # explicit --mains-hz stands on its own
        raise
    return reg.get(dataset_id, {}) or {}


def resolved_mains(dataset_id: str, args) -> Tuple[Optional[float], str]:
    """(frequency, why). --mains-hz beats the registry; --no-notch beats both."""
    if getattr(args, "no_notch", False):
        return None, "disabled by --no-notch"
    if getattr(args, "mains_hz", None) is not None:
        return float(args.mains_hz), "--mains-hz"
    entry = registry_for(dataset_id, args)
    if entry.get("notch_already_applied"):
        return None, (f"registry: publisher already notched at "
                      f"{entry.get('powerline_hz')} Hz")
    if entry.get("skip_notch"):
        # Distinct from notch_already_applied: nobody filtered this, there is
        # simply nothing at the line to remove, and the filter would cost more
        # than it saves. The reason travels into every shard's provenance.
        return None, (f"registry: notch skipped -- "
                      f"{entry.get('skip_notch_reason', 'no reason recorded')}")
    if entry.get("powerline_hz") is not None:
        return float(entry["powerline_hz"]), "registry"
    return None, "unset"


def upsample_allowed(dataset_id: str, fs: float, args) -> bool:
    """Whether this corpus may legitimately arrive below its route's rate.

    FACED ships 31 of 123 subjects at 250 Hz and PhysioNetMI is 160 Hz
    throughout; both are the real acquisition, not a preprocessed derivative,
    and refusing them would drop a quarter of one corpus and all of another.
    A rate the registry does not list is still refused, because that is how the
    250 Hz FACED *derivative* is told apart from the 250 Hz FACED *subjects*.
    """
    if getattr(args, "allow_upsample_faced", False) and dataset_id == "faced":
        return True
    allowed = registry_for(dataset_id, args).get("native_upsample_ok") or []
    return any(abs(float(a) - fs) < 1e-6 for a in allowed)


def owns(subject_id: str, args) -> bool:
    """Whether this array task is the one that processes this subject.

    Called by every adapter BEFORE the file is read. Filtering after the read
    would be correct and useless: each of 64 tasks would decode and resample all
    1.9 TB of HBN and throw away 63/64 of it. Every adapter derives its subject
    id from the path, so this costs nothing.
    """
    if not getattr(args, "shard", None):
        return True
    return subject_shard(subject_id, args.shard[1]) == args.shard[0]


BIDS_SUBJECT = re.compile(r"(?:^|[/\\])(sub-[A-Za-z0-9]+)(?:[/\\_]|$)")


def _bids_subject(path: str) -> Optional[str]:
    """The BIDS sub-<label> anywhere in the path, or None.

    Subject identity decides the train/val split, so a wrong answer here is a
    subject leak across the split that nothing downstream can detect. The
    NEMAR conversions of FACED, M3CV, HBN and HGD are all BIDS, where the label
    is in both the directory and the filename; taking the parent directory
    instead would label every recording "eeg".
    """
    m = BIDS_SUBJECT.search(path.replace(os.sep, "/"))
    return m.group(1) if m else None


def mains_for(dataset_id: str, args) -> Optional[float]:
    """Mains frequency for a Recording, from --mains-hz or the registry."""
    return resolved_mains(dataset_id, args)[0]


def adapt_tueg(root: str, args) -> Iterator[Recording]:
    """TUH EEG Corpus: .edf, one recording per file, montage-mixed.

    Identity comes from the FILENAME (see tueg_identity), so the same code reads
    v1.x and v2.x layouts. Sampling rate varies per recording -- TUEG holds at
    least 250, 256, 400, 512 and 1000 Hz -- and is read from each file, never
    assumed; every one of them is resampled to E19_256's 256 Hz.

    Channel labels arrive as "EEG FP1-REF" / "EEG T3-LE", which
    normalize_channel_name strips and maps, T3/T4/T5/T6 included. Channels that
    are not scalp EEG (EKG, PHOTIC, IBI, BURSTS, SUPPR) simply match no slot and
    are dropped by map_to_slots, which names them in provenance.

    Mains is 60 Hz: TUH is recorded in Philadelphia. Passing --mains-hz
    overrides it; there is no guessing either way.
    """
    mne = _require_mne()
    files = iter_tueg_files(root, getattr(args, "file_list", None))
    if getattr(args, "shard", None):
        files = shard_files(files, args, root)
    for path in files:
        ident = tueg_identity(path, root)
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        except Exception as exc:                              # noqa: BLE001
            raise PreprocessError(f"unreadable EDF: {exc}") from exc
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=ident["subject"],
            data=raw.get_data(),                    # mne returns volts
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            unit="V",
            mains_hz=mains_for("physionet_mi", args),
            notes={"tuh_session": ident["session"],
                   "tuh_montage": ident["montage"],
                   "tuh_identity_rule": ident["rule"]},
        )


def adapt_faced(root: str, args) -> Iterator[Recording]:
    """FACED: the raw 1000 Hz BDF release only.

    The widely mirrored FACED release is already preprocessed to 250 Hz. This
    route needs 512 Hz, and 250 -> 512 is upsampling: it invents no information
    and leaves everything above 125 Hz as interpolation artefact, in a corpus
    whose point is that it is 32-channel affective EEG. Refused by default
    rather than silently upsampled, because the resulting shard is
    indistinguishable from a real one downstream.
    """
    mne = _require_mne()
    # The BIDS conversions (NEMAR nm000112) carry the same signal as the raw
    # BDF in .set or .edf, so the extension says nothing about which release
    # this is. The RATE does, per file, which is also the only way to separate
    # FACED's 31 genuinely-250 Hz subjects from a wholesale 250 Hz derivative.
    files = _walk(root, READABLE_EXTS,
                  cache=getattr(args, "file_list", None))
    if not files:
        raise PreprocessError(f"no .bdf/.set/.fif/.edf under {root}")
    for path in files:
        subject = _bids_subject(path) or os.path.splitext(os.path.basename(path))[0]
        if not owns(subject, args):
            continue
        raw = read_raw_any(path)
        fs = float(raw.info["sfreq"])
        target = ROUTES["E32_512"].sampling_rate
        if fs < target and not upsample_allowed("faced", fs, args):
            raise PreprocessError(
                f"{path} is {fs} Hz and the route needs {target} Hz. "
                f"{fs} is not in FACED's native_upsample_ok list, so this is a "
                f"preprocessed derivative rather than one of the 31 subjects "
                f"recorded at 250 Hz. Refusing to upsample; "
                f"--allow-upsample-faced overrides.")
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=fs, unit="V", mains_hz=mains_for("faced", args),
        )


def adapt_tdbrain(root: str, args) -> Iterator[Recording]:
    """TDBRAIN: 26 electrodes at 500 Hz, in per-subject .npy/.csv or BIDS .edf.

    The 26 are placed into E32_512's 32 slots BY NAME. The six slots TDBRAIN
    does not record stay zero with valid_channel_mask False -- no interpolation
    and no CSD, because a spatially interpolated channel is a smooth function of
    its neighbours and the model would learn to reproduce the interpolation.
    """
    mne = _require_mne()
    files = _walk(root, READABLE_EXTS,
                  cache=getattr(args, "file_list", None))
    for path in files:
        base = os.path.basename(path)
        subject = _bids_subject(path) or (
            base.split("_")[0] if "_" in base else os.path.splitext(base)[0])
        if not owns(subject, args):
            continue
        raw = read_raw_any(path)
        yield Recording(
            recording_id=os.path.relpath(path, root), subject_id=subject,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=mains_for("tdbrain", args),
        )


def adapt_physionet_mi(root: str, args) -> Iterator[Recording]:
    """PhysioNet EEG Motor Movement/Imagery: 64 channels at 160 Hz, .edf.

    The 64 are kept. Cropping to 32 would throw away half the montage of the
    only 64-channel corpus with a large subject count, and E64_256 exists so it
    does not have to be.
    """
    mne = _require_mne()
    # EDF+ only; this corpus is published in exactly one format.
    for path in _walk(root, (".edf",),
                      cache=getattr(args, "file_list", None)):
        base = os.path.basename(path)
        subject = _bids_subject(path) or (
            base[:4] if base.lower().startswith("s") else base)
        if not owns(subject, args):
            continue
        raw = read_raw_any(path)
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz if args.mains_hz else 60.0,
        )


def adapt_m3cv(root: str, args) -> Iterator[Recording]:
    """M3CV: 64 channels at 250 Hz. Pretraining only, never a downstream split."""
    mne = _require_mne()
    files = _walk(root, READABLE_EXTS,
                  cache=getattr(args, "file_list", None))
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        subject = _bids_subject(path) or base.split("_")[0]
        if not owns(subject, args):
            continue
        raw = read_raw_any(path)
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=raw.get_data(),
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=mains_for("m3cv", args),
        )


#: EGI HydroCel GSN-129 non-scalp / reference labels. Removing the RECORDED
#: REFERENCE is what takes 129 to 128, and it is E129 (vertex reference, "Cz")
#: on this net -- a named electrode, identified from the montage, not "the last
#: row". Dropping the last row would remove whichever channel the file happened
#: to end with.
HBN_NON_SCALP = ("E129", "Cz", "VREF", "REF")


def adapt_hbn(root: str, args) -> Iterator[Recording]:
    """HBN: EGI HydroCel 129 at 500 Hz -> 128 scalp channels.

    The channel removed is identified by NAME from the montage (the net's vertex
    reference), and the names actually dropped are written into every shard's
    provenance under ``removed_channels``. If the reference cannot be identified
    the recording is failed rather than trimmed to length.
    """
    mne = _require_mne()
    files = _walk(root, READABLE_EXTS,
                  cache=getattr(args, "file_list", None))
    seen_dirs = set()
    for path in files:
        base = os.path.basename(os.path.normpath(path))
        subject = _bids_subject(path) or base.split("_")[0].split(".")[0]
        if not owns(subject, args):
            continue
        if path.lower().endswith(".mff"):
            if path in seen_dirs:
                continue
            seen_dirs.add(path)
            raw = read_raw_any(path)
        else:
            raw = read_raw_any(path)
        names = list(raw.ch_names)
        drop = [n for n in names if n.strip().upper() in
                {s.upper() for s in HBN_NON_SCALP}]
        if len(names) == 129 and not drop:
            raise PreprocessError(
                f"{path}: 129 channels but none of {HBN_NON_SCALP} is among "
                f"them, so the reference cannot be identified by name. "
                f"Refusing to drop a row by position.")
        keep = [n for n in names if n not in drop]
        data = raw.get_data(picks=[names.index(n) for n in keep])
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=data, channel_names=keep,
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=mains_for("hbn", args),
            notes={"removed_channels": drop, "n_channels_before": len(names)},
        )


def adapt_hgd(root: str, args) -> Iterator[Recording]:
    """High-Gamma Dataset: 128 channels at 500 Hz. Pretraining only.

    HGD's electrodes are 10-5 scalp names and HBN's are EGI net positions, so
    the two do not share slot identities even though they share the route's
    shape. HGD channels that match no E-numbered slot land as padded, which is
    the honest outcome; --hgd-own-slots places them on HGD's own 128-name list
    instead, which is right if HGD is the only corpus on the route.
    """
    mne = _require_mne()
    for path in _walk(root, READABLE_EXTS + (".mat",),
                      cache=getattr(args, "file_list", None)):
        base = os.path.splitext(os.path.basename(path))[0]
        subject = _bids_subject(path) or base.split("_")[0]
        if not owns(subject, args) and not path.lower().endswith(".mat"):
            continue
        if path.lower().endswith(".mat"):
            raise PreprocessError(
                f"{path}: HGD's .mat release needs the braindecode reader; "
                f"export to FIF or EDF first, or point --root at a converted "
                f"tree. Not guessing a MATLAB layout.")
        raw = read_raw_any(path)
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=raw.get_data(),
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=mains_for("hgd", args),
        )


def adapt_smoke(root: str, args) -> Iterator[Recording]:
    """Synthetic noise, for exercising the pipeline with no corpus present.

    Reached only via --smoke-test. Every shard it writes carries
    ``"synthetic": true`` in its provenance so a smoke run can never be mistaken
    for a real one in a manifest or a figure.
    """
    spec = EEG_DATASETS[args.smoke_as]
    route = spec.route
    rng = np.random.default_rng(args.seed)
    n_src = len(spec.slots)
    for s in range(args.smoke_subjects):
        for r in range(args.smoke_recordings):
            secs = args.window_seconds * (args.smoke_windows + 1)
            n = int(secs * route.sampling_rate)
            yield Recording(
                recording_id=f"smoke_s{s:02d}_r{r:02d}",
                subject_id=f"smoke_s{s:02d}",
                data=rng.normal(0, 20e-6, size=(n_src, n)),
                channel_names=list(spec.slots),
                sampling_rate=float(route.sampling_rate),
                unit="V", mains_hz=50.0, notes={"synthetic": True},
            )


SIENA_SUBJECT = re.compile(r"(PN\d+)", re.IGNORECASE)


def adapt_siena(root: str, args) -> Iterator[Recording]:
    """Siena Scalp EEG Database (PhysioNet siena-scalp-eeg 1.0.0, ODC-By 1.0).

    Layout as published: ``<root>/PNxx/PNxx-k.edf``, one directory per patient
    (14 adults with focal epilepsy), EDF at 512 Hz. Labels arrive as ``EEG Fp1``
    / ``EEG T3``; normalize_channel_name maps them, T3/T4/T5/T6 included, onto
    E19_256's 19 slots, all of which every recording fills. The extra
    electrodes (Fc1, Cp5, F9, ...), EKG, SPO2, HR and the marker channels match
    no slot and are dropped by map_to_slots, which names them in provenance.

    Subject identity is the patient directory, so the train/val split is by
    patient. The seizure annotations (Seizures-list-PNxx.txt) are not read:
    this is self-supervised pretraining, and every window, ictal or not, is a
    window. Recorded in Siena, Italy: 50 Hz mains (registry).
    """
    for path in _walk(root, (".edf",), cache=getattr(args, "file_list", None)):
        m = SIENA_SUBJECT.search(os.path.basename(path)) or SIENA_SUBJECT.search(path)
        subject = m.group(1).upper() if m else os.path.basename(os.path.dirname(path))
        if not owns(subject, args):
            continue
        raw = read_raw_any(path)
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=subject,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=mains_for("siena", args),
        )


ADAPTERS: Dict[str, Callable] = {
    "tueg": adapt_tueg, "faced": adapt_faced, "tdbrain": adapt_tdbrain,
    "physionet_mi": adapt_physionet_mi, "m3cv": adapt_m3cv,
    "hbn": adapt_hbn, "hgd": adapt_hgd, "siena": adapt_siena,
}


# --------------------------------------------------------------------------- #
# The shared pipeline
# --------------------------------------------------------------------------- #

def process_recording(rec: Recording, dataset_id: str, cfg: PreprocessConfig,
                      slots: Sequence[str], route, out_dir: str,
                      extra_prov: Dict, args_ref=None) -> Optional[Dict]:
    """One recording -> one shard, or None when it yields no whole window."""
    if len(slots) != route.n_channels:
        # Caught here rather than at training time. A slot list one short writes
        # a perfectly valid-looking shard whose channel axis does not match the
        # route, and the failure then surfaces thousands of windows later as a
        # shape error inside the model.
        raise PreprocessError(
            f"{dataset_id}: slot list has {len(slots)} entries but route "
            f"{route.route_id} has {route.n_channels} channels")

    x = to_microvolts(np.asarray(rec.data, dtype=np.float64), rec.unit)
    x = detrend(x, cfg.detrend)
    mains = rec.mains_hz if rec.mains_hz is not None else cfg.notch_hz
    x = notch(x, rec.sampling_rate, mains, cfg.notch_harmonics, cfg.notch_quality)
    x = highpass(x, rec.sampling_rate, cfg.highpass_hz)
    x = resample_to(x, rec.sampling_rate, route.sampling_rate)

    mapping = map_to_slots(rec.channel_names, slots)
    if not mapping.matrix_rows:
        raise PreprocessError(
            f"none of {len(rec.channel_names)} channels matched a slot of "
            f"{route.route_id}. First few recorded: {rec.channel_names[:6]}")
    coverage = float(mapping.valid.sum()) / len(slots)
    if coverage < getattr(args_ref, "min_slot_coverage", 0.0):
        raise PreprocessError(
            f"fills only {int(mapping.valid.sum())} of {len(slots)} slots "
            f"({coverage:.0%}); empty: {mapping.empty_slots[:12]}")

    placed = place_on_slots(x, mapping, len(slots))

    win = int(cfg.window_seconds * route.sampling_rate)
    stride = int(cfg.stride() * route.sampling_rate)
    windows, starts = window_signal(placed, win, stride)
    if windows.shape[0] == 0:
        return None
    windows = zscore_windows(windows, mapping.valid, cfg.zscore_eps,
                             cfg.clip_sigma)

    ids, _ = channel_ids_for(slots)
    ids = [i if mapping.valid[k] else 0 for k, i in enumerate(ids)]   # PAD_ID
    # Recorded for EVERY corpus, not just FACED: PhysioNetMI is 160 Hz against
    # a 256 Hz route and M3CV is 250 against 256, so a shard whose top spectrum
    # is interpolation rather than measurement must say so in its own
    # provenance. Otherwise it is indistinguishable downstream from a shard
    # that was genuinely sampled that high.
    prov_rate = {}
    if rec.sampling_rate < route.sampling_rate:
        prov_rate = {
            "upsampled_from_hz": float(rec.sampling_rate),
            "true_nyquist_hz": float(rec.sampling_rate) / 2.0,
            "spectrum_above_hz_is_interpolated": float(rec.sampling_rate) / 2.0,
        }
    prov = cfg.provenance({
        "dataset_id": dataset_id,
        "mains_hz": mains,
        "notch_decision": (getattr(args_ref, "_mains_why", None) or "unrecorded"),
        "source_sampling_rate": rec.sampling_rate,
        "target_sampling_rate": route.sampling_rate,
        **prov_rate,
        "unmatched_source_channels": mapping.unmatched_sources,
        "empty_slots": mapping.empty_slots,
        "unknown_channel_names": mapping.unknown_names,
        **(rec.notes or {}), **extra_prov,
    })
    safe = rec.recording_id.replace(os.sep, "__").replace(" ", "_")
    path = os.path.join(out_dir, "shards", f"{safe}.h5")
    entry = write_shard(
        path, windows, route, dataset_id, list(slots), ids, mapping.valid,
        [rec.subject_id] * windows.shape[0],
        [rec.recording_id] * windows.shape[0],
        (starts / route.sampling_rate).tolist(), rec.sampling_rate, prov)
    entry["qc"] = window_qc(windows, mapping.valid, route.sampling_rate)
    entry["source_sampling_rate"] = float(rec.sampling_rate)
    entry["upsampled"] = bool(rec.sampling_rate < route.sampling_rate)
    entry["empty_slots"] = list(mapping.empty_slots)
    entry["unmatched_source_channels"] = list(mapping.unmatched_sources)
    entry["placed_total"] = len(mapping.matrix_rows)
    entry["placed_unk"] = sum(1 for k, i in enumerate(ids)
                              if mapping.valid[k] and i == 1)     # UNK_ID
    entry["unknown_channel_names"] = mapping.unknown_names
    entry["n_channels_recorded"] = len(rec.channel_names)
    entry["duration_seconds"] = float(windows.shape[0] * cfg.window_seconds)
    return entry


#: A recording longer than this is split into chunks before processing rather
#: than held whole. TUEG holds multi-hour files, and get_data() returns float64:
#: four hours at 1000 Hz on 41 channels is 4.7 GB before a single filter runs,
#: and every step after it wants another copy. With eight workers against a
#: 64 GB task that is how one file takes the node down.
MAX_MINUTES_IN_MEMORY = 30.0


class _RedoShard(Exception):
    """Raised inside the resume scan to mean 'this one is to be redone'."""


def _shard_data_is_readable(f, n_win: int) -> bool:
    """Whether a shard's chunks inflate -- not whether its header parses.

    ``f["data"].shape`` is answered from the object header, which HDF5 lays
    down before any chunk, so a resume that reads only the shape accepts a file
    that a killed task left half-written. It then accepts it on every later
    resume too, and the corpus carries the shard until a training epoch reads
    the window it cannot inflate. Forcing the first and last chunk turns that
    into "not done yet", which is what it always was: truncation takes the
    tail, so the last window is the one that tells you.
    """
    if n_win <= 0:
        return True
    try:
        f["data"][0]
        f["data"][n_win - 1]
    except Exception:                                          # noqa: BLE001
        return False
    return True


def _process_one_path(payload):
    """One file, start to finish, in a worker process.

    A module-level function taking only picklable arguments, because a closure
    over the CLI namespace cannot cross a process boundary and the failure mode
    for that is a pool that hangs rather than one that raises.
    """
    (path, root, dataset_id, cfg, slots, route_id, out_dir, mains_hz,
     min_cov, synthetic, max_minutes) = payload
    route = ROUTES[route_id]
    ident = tueg_identity(path, root)
    rec_id = os.path.relpath(path, root)
    try:
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        cap = max_minutes if max_minutes else MAX_MINUTES_IN_MEMORY
        n_keep = min(raw.n_times, int(cap * 60 * fs))
        truncated = n_keep < raw.n_times
        data = raw.get_data(start=0, stop=n_keep)
        names = list(raw.ch_names)
        del raw
        rec = Recording(
            recording_id=rec_id, subject_id=ident["subject"],
            data=data, channel_names=names,
            sampling_rate=fs, unit="V",
            mains_hz=mains_hz,
            notes={"tuh_session": ident["session"],
                   "tuh_montage": ident["montage"],
                   "tuh_identity_rule": ident["rule"],
                   "truncated_to_minutes": (cap if truncated
                                            else None)})

        class _A:
            min_slot_coverage = min_cov

        entry = process_recording(rec, dataset_id, cfg, slots, route, out_dir,
                                  {"synthetic": synthetic}, args_ref=_A())
        if entry is None:
            return None, {"recording_id": rec_id,
                          "reason": "no whole window in recording"}
        return entry, None
    except Exception as exc:                                   # noqa: BLE001
        return None, {"recording_id": rec_id, "reason": str(exc),
                      "traceback": traceback.format_exc(limit=3)}


def _run_parallel(paths, args, dataset_id, cfg, slots, route):
    """Map _process_one_path over a pool. Returns (entries, failures).

    Used for the path-addressable corpora -- TUEG above all, where a task holds
    32 cores and would otherwise decode EDF on one of them. Progress is printed
    as results arrive, not as work is submitted, so the rate shown is the rate
    achieved.
    """
    import concurrent.futures as cf

    payloads = [(p, args.root, dataset_id, cfg, tuple(slots), route.route_id,
                 args.out_dir, args.mains_hz if args.mains_hz else 60.0,
                 args.min_slot_coverage, bool(args.smoke_test),
                 args.max_recording_minutes) for p in paths]
    entries, failures = [], []
    started = time.time()
    # as_completed, not pool.map. map yields in INPUT order, so one slow
    # recording -- TUEG holds multi-hour files -- blocks the reporting of every
    # finished result behind it. The workers keep going and the counter stops,
    # which is indistinguishable from a hang and was read as one.
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_process_one_path, pl): pl[0] for pl in payloads}
        done = 0
        for fut in cf.as_completed(futures):
            path = futures[fut]
            try:
                entry, failure = fut.result()
            except Exception as exc:                           # noqa: BLE001
                # A worker that died -- almost always the OOM killer on a long
                # recording -- must be recorded, not lost with the pool.
                entry, failure = None, {
                    "recording_id": os.path.relpath(path, args.root),
                    "reason": f"worker died: {exc}"}
            if entry is not None:
                entries.append(entry)
            if failure is not None:
                failures.append(failure)
            done += 1
            if done % 50 == 0 or done == len(payloads):
                rate = done / max(1e-9, time.time() - started)
                left = (len(payloads) - done) / max(1e-9, rate)
                print(f"  {done}/{len(payloads)}  {rate:.1f} files/s  "
                      f"{len(failures)} failed  ETA {left/60:.0f} min",
                      flush=True)
    return entries, failures


def _shard_spec(text: str):
    try:
        i, n = text.split("/")
        i, n = int(i), int(n)
    except Exception:                                          # noqa: BLE001
        raise argparse.ArgumentTypeError(f"--shard wants I/N, got {text!r}")
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError(
            f"--shard {text}: need 0 <= I < N")
    return (i, n)


def mains_peak_ratios(raw, seconds: float = 60.0) -> Optional[Dict[str, float]]:
    """Band power at 50 and 60 Hz relative to the spectrum either side of it.

    Two of these corpora publish PowerLineFrequency as n/a, and a notch at the
    wrong frequency is silent in both directions: it leaves the interference in
    and takes real signal out. This measures rather than assumes. A ratio near
    1.0 means no peak -- which is itself an answer, and the right one for a
    corpus the publisher already notched.
    """
    fs = float(raw.info["sfreq"])
    if fs < 130.0:
        # Below a ~130 Hz sampling rate there is no 60 Hz band to compare
        # against a 65 Hz baseline. PhysioNetMI at 160 Hz is the marginal case
        # and it does have room; anything slower cannot answer the question.
        return None
    n = int(min(seconds, raw.n_times / fs) * fs)
    if n < int(4 * fs):
        return None
    x = np.asarray(raw.get_data(start=0, stop=n), dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x, axis=-1)) ** 2).mean(axis=0)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(psd[m].mean()) if m.any() else float("nan")

    out = {}
    for name, centre in (("50", 50.0), ("60", 60.0)):
        if centre + 5.0 >= fs / 2.0:
            continue
        peak = band(centre - 1.0, centre + 1.0)
        base = np.nanmean([band(centre - 6.0, centre - 2.0),
                           band(centre + 2.0, centre + 6.0)])
        if base and np.isfinite(base) and base > 0:
            out[name] = float(peak / base)
    return out or None


def dump_channels(dataset_id: str, root: str, args, slots, route) -> int:
    """Print each sampled recording's channel list exactly as it is stored."""
    from collections import Counter

    mne = _require_mne()
    if dataset_id == "tueg":
        files = iter_tueg_files(root, getattr(args, "file_list", None))
    else:
        files = _walk(root, READABLE_EXTS,
                      cache=getattr(args, "file_list", None))
    if not files:
        print(f"ERROR: no readable recording under {root}", file=sys.stderr)
        return 1

    n = min(args.dump_channels, len(files))
    step = max(1, len(files) // n)
    print(f"{len(files)} file(s); dumping {n}\n")

    layouts = Counter()
    for path in files[::step][:n]:
        try:
            raw = read_raw_any(path, preload=False)
        except Exception as exc:                               # noqa: BLE001
            print(f"UNREADABLE {os.path.relpath(path, root)}: {exc}")
            continue
        names = list(raw.ch_names)
        layouts[tuple(names)] += 1
        mapping = map_to_slots(names, slots)
        print(f"--- {os.path.relpath(path, root)}")
        print(f"    {len(names)} channels @ {raw.info['sfreq']:g} Hz, "
              f"{int(mapping.valid.sum())}/{len(slots)} slots filled")
        print("    " + " ".join(names))
        print()

    print(f"{len(layouts)} distinct montage(s) among {n} file(s):")
    for names, c in layouts.most_common():
        print(f"  {len(names)} channels x{c} file(s)")

    # The union across layouts is what a corrected slot list has to cover.
    union: List[str] = []
    for names in layouts:
        for nm in names:
            norm = normalize_channel_name(nm)
            if norm and norm not in union:
                union.append(norm)
    print(f"\nUNION of normalised names across those montages ({len(union)}):")
    print("  " + " ".join(union))
    known = [u for u in union if u in CHANNEL_TO_ID]
    unknown = [u for u in union if u not in CHANNEL_TO_ID]
    print(f"\n  in the vocabulary : {len(known)}")
    print(f"  NOT in it         : {len(unknown)}")
    if unknown:
        print("  " + " ".join(unknown))
        print("\n  Names absent from the vocabulary become <unk> and their "
              "slots stay padded. If these are real electrodes, they belong in "
              "channel_embedding.py (append only) and in the route's slot list.")
    return 0


def derive_slots(dataset_id: str, root: str, args, slots, route) -> int:
    """Build a route slot list FROM THE CORPUS, and say what it needs.

    A slot list is a permanent decision: the channel ids it produces are written
    into every shard and indexed by the checkpoint's embedding table. So it must
    not be a transcription of a montage from a paper. This reads the recordings,
    takes the names present in EVERY sampled montage -- the reliable core, which
    excludes the aux channels that only some files carry -- and writes that out
    alongside the vocabulary entries it would need.

    Nothing is applied automatically. The two files it writes are for review and
    for checking in, because a slot list that changed between two preprocessing
    runs would silently relabel every channel of whichever shards came first.
    """
    from collections import Counter

    mne = _require_mne()
    files = _walk(root, READABLE_EXTS,
                  cache=getattr(args, "file_list", None))
    if not files:
        print(f"ERROR: no readable recording under {root}", file=sys.stderr)
        return 1

    n = min(args.derive_slots_from, len(files))
    step = max(1, len(files) // n)
    montages: List[List[str]] = []
    seen_counts = Counter()
    for path in files[::step][:n]:
        try:
            raw = read_raw_any(path, preload=False)
        except Exception as exc:                               # noqa: BLE001
            print(f"  UNREADABLE {os.path.relpath(path, root)}: {exc}")
            continue
        names = [normalize_channel_name(x) for x in raw.ch_names]
        montages.append([x for x in names if x])
        seen_counts[len(names)] += 1

    if not montages:
        print("ERROR: nothing readable", file=sys.stderr)
        return 1

    print(f"read {len(montages)} montage(s); channel counts {dict(seen_counts)}")

    # Order comes from the montage that has the most channels, so the slot list
    # follows the cap's own layout rather than alphabetical noise.
    core = set(montages[0])
    union: List[str] = []
    for m in montages:
        core &= set(m)
    for m in sorted(montages, key=len, reverse=True):
        for nm in m:
            if nm not in union:
                union.append(nm)
    ordered_core = [nm for nm in union if nm in core]

    print(f"  union over montages     {len(union)}")
    print(f"  present in EVERY one    {len(ordered_core)}")
    print(f"  route {route.route_id} needs   {route.n_channels}")

    known = [x for x in ordered_core if x in CHANNEL_TO_ID]
    unknown = [x for x in ordered_core if x not in CHANNEL_TO_ID]
    print(f"  of the common core, in the vocabulary: {len(known)}, "
          f"absent: {len(unknown)}")

    if len(ordered_core) != route.n_channels:
        print(f"\n  NOTE: the common core is {len(ordered_core)}, not "
              f"{route.n_channels}. Names carried by only some recordings are "
              f"excluded on purpose -- they are usually EOG/EMG/mastoid. "
              f"Decide which to add or drop before committing the list; this "
              f"tool will not pad or truncate it for you, because both choices "
              f"are silent.")

    out = args.derive_slots
    with open(out, "w") as f:
        json.dump({
            "dataset_id": dataset_id,
            "route_id": route.route_id,
            "n_required": route.n_channels,
            "derived_from_files": len(montages),
            "slots_common_core": ordered_core,
            "union": union,
            "not_in_vocabulary": unknown,
        }, f, indent=2)
    add = f"{out}.vocab_additions.txt"
    with open(add, "w") as f:
        f.write("\n".join(unknown) + ("\n" if unknown else ""))
    print(f"\n  wrote {out}")
    print(f"  wrote {add}  ({len(unknown)} name(s))")
    print("\n  Neither file is read by anything. Review them, then append the "
          "names to CHANNEL_VOCAB (append only) and set the route's slot list.")
    return 0


def inspect_corpus(dataset_id: str, root: str, args, slots, route) -> int:
    """Report what the corpus actually is, without processing any of it.

    The number that decides whether a run is worth launching is the PER-FILE
    slot coverage, not the union over the sample. A union of 19 of 19 only says
    that every slot was filled by some file somewhere; a corpus whose files each
    carry twelve of the nineteen would report exactly the same line while
    producing windows that are more mask than measurement.
    """
    from collections import Counter

    mne = _require_mne()
    if dataset_id == "tueg":
        files = iter_tueg_files(root, getattr(args, "file_list", None))
    else:
        files = _walk(root, READABLE_EXTS,
                      cache=getattr(args, "file_list", None))
    if not files:
        # "no readable files" on a directory that plainly has things in it is
        # the least useful thing this can say. Name what is actually there.
        from collections import Counter as _C
        present, n_any = _C(), 0
        for dirpath, _dirs, names in os.walk(root):
            for nm in names:
                n_any += 1
                present[os.path.splitext(nm)[1].lower() or "(no extension)"] += 1
            if n_any > 20000:
                break
        print(f"ERROR: no readable recording under {root}", file=sys.stderr)
        if not n_any:
            print("  The directory is empty. The download has not run, or it "
                  "landed elsewhere.", file=sys.stderr)
        else:
            print(f"  {n_any:,} file(s) are there, but none with an extension "
                  f"this adapter reads. What is present:", file=sys.stderr)
            for ext, c in present.most_common(12):
                print(f"    {ext:<16} x{c:,}", file=sys.stderr)
            if ".zip" in present:
                print("\n  There is still a .zip here -- it has not been "
                      "unpacked. unzip refuses these archives; use:",
                      file=sys.stderr)
                print("    python scripts/unpack_archive.py <file>.zip "
                      "--extract-to .", file=sys.stderr)
        return 1

    n = min(args.inspect, len(files))
    print(f"{len(files)} file(s) under {root}")
    print(f"reading the headers of {n} of them\n")

    rates, montages, rules, counts = Counter(), Counter(), Counter(), Counter()
    union_hits, unmatched = Counter(), Counter()
    coverage: List[int] = []
    mains_ratios: Dict[str, List[float]] = {"50": [], "60": []}
    seconds = 0.0
    subjects = set()
    failed = 0
    step = max(1, len(files) // n)
    for path in files[::step][:n]:
        try:
            raw = read_raw_any(path, preload=False)
        except Exception as exc:                               # noqa: BLE001
            failed += 1
            print(f"  UNREADABLE {os.path.relpath(path, root)}: {exc}")
            continue
        fs = float(raw.info["sfreq"])
        rates[fs] += 1
        counts[len(raw.ch_names)] += 1
        seconds += raw.n_times / fs
        if dataset_id == "tueg":
            ident = tueg_identity(path, root)
            montages[ident["montage"] or "(none)"] += 1
            rules[ident["rule"]] += 1
            subjects.add(ident["subject"])
        if getattr(args, "verify_powerline", False):
            try:
                ratios = mains_peak_ratios(raw)
            except Exception:                                   # noqa: BLE001
                ratios = None
            for k, v in (ratios or {}).items():
                if np.isfinite(v):
                    mains_ratios[k].append(v)
        mapping = map_to_slots(raw.ch_names, slots)
        coverage.append(int(mapping.valid.sum()))
        for j, filled in enumerate(mapping.valid):
            if filled:
                union_hits[slots[j]] += 1
        for name in mapping.unmatched_sources:
            unmatched[normalize_channel_name(name)] += 1

    read = len(coverage)
    print(f"sampling rates   : {dict(sorted(rates.items()))}")
    print(f"channel counts   : {dict(sorted(counts.items()))}")
    if dataset_id == "tueg":
        print(f"montage dirs     : {dict(montages)}")
        print(f"identity rule    : {dict(rules)}   "
              f"({len(subjects)} distinct subjects in this sample)")
        if rules.get("path", 0):
            print("  WARNING: the filename rule did not fire on every file; "
                  "the subject ids, and so the train/val split, are coming "
                  "from a fallback. Check the layout before a long run.")
    print(f"unreadable       : {failed} of {n}")

    # -- per-file coverage, which is the thing that matters ----------------- #
    if coverage:
        cov = np.asarray(coverage)
        full = int((cov == len(slots)).sum())
        print(f"\nPER-FILE slot coverage of {route.route_id} "
              f"({len(slots)} slots), over {read} file(s):")
        print(f"  min {cov.min()}   median {int(np.median(cov))}   "
              f"mean {cov.mean():.1f}   max {cov.max()}")
        print(f"  files filling every slot : {full} of {read} "
              f"({full/read*100:.0f}%)")
        hist = Counter(cov.tolist())
        print("  distribution: " + "  ".join(
            f"{k}ch x{v}" for k, v in sorted(hist.items())))
        thresh = args.min_slot_coverage
        would_skip = int((cov < thresh * len(slots)).sum())
        print(f"  --min-slot-coverage {thresh:.2f} would skip {would_skip} of "
              f"{read} ({would_skip/read*100:.0f}%) as too sparse")
        # Which slots go unfilled most often: this names the electrode the
        # adapter cannot read, if there is one.
        misses = [(s, read - union_hits.get(s, 0)) for s in slots]
        misses = [(s, m) for s, m in misses if m]
        if misses:
            print("  slots not always filled: " + ", ".join(
                f"{s} (missing in {m})"
                for s, m in sorted(misses, key=lambda x: -x[1])[:12]))
        else:
            print("  every slot filled in every file read")

    # -- is the registry's mains frequency actually the one in the data? ---- #
    if getattr(args, "verify_powerline", False):
        entry = registry_for(dataset_id, args)
        claimed = entry.get("powerline_hz")
        print(f"\nPOWERLINE, band power at the line relative to 4 Hz either "
              f"side, over the files read:")
        medians = {}
        for k in ("50", "60"):
            vals = mains_ratios[k]
            if not vals:
                print(f"  {k} Hz : not measurable at this sampling rate")
                continue
            med = float(np.median(vals))
            medians[k] = med
            share = sum(1 for v in vals if v > 2.0) / len(vals)
            print(f"  {k} Hz : median x{med:.2f}   "
                  f"peak in {share*100:.0f}% of {len(vals)} file(s)")
        if len(medians) == 2:
            winner = max(medians, key=medians.get)
            if max(medians.values()) < 1.5:
                print("  VERDICT: no line peak at either frequency. Consistent "
                      "with a corpus the publisher already notched -- check "
                      "notch_already_applied before adding a second filter.")
            elif claimed is not None and abs(float(claimed) - float(winner)) > 1:
                print(f"  VERDICT: the data peaks at {winner} Hz but the "
                      f"registry says {claimed:g} Hz. DO NOT run the array "
                      f"until this is resolved -- notching {claimed:g} Hz here "
                      f"would remove signal and leave the interference.")
            else:
                print(f"  VERDICT: peak at {winner} Hz, matching the registry. "
                      f"Re-run the real job with --psd-verified.")

    # -- what was dropped, and whether any of it is a real electrode -------- #
    print(f"\nDROPPED channel names ({len(unmatched)} distinct), most common:")
    known, unknown = [], []
    for name, c in unmatched.most_common():
        (known if name in CHANNEL_TO_ID else unknown).append((name, c))
    for name, c in unknown[:30]:
        print(f"  {name:<20} x{c}")
    if len(unknown) > 30:
        print(f"  ... and {len(unknown) - 30} more")
    if known:
        print(f"\n  These ARE electrodes in the vocabulary, dropped only "
              f"because {route.route_id} has no slot for them:")
        for name, c in known:
            print(f"    {name:<18} x{c}")
        print(f"  {route.route_id} has {len(slots)} slots and the recording "
              f"has more electrodes than fit; that is not a naming bug.")

    # -- what it will produce, and what it does NOT tell you ---------------- #
    if read and seconds > 0:
        per_file = seconds / read
        est_files = len(files) - int(failed / max(1, read) * len(files))
        win = int(est_files * per_file / args.window_seconds)
        bytes_per = len(slots) * route.window_samples * 4
        total_h = est_files * per_file / 3600
        print(f"\nPROJECTED OUTPUT, from {read} file(s) averaging "
              f"{per_file/60:.1f} min of RECORDING each:")
        print(f"  {est_files:,} files  ~{total_h:,.0f} hours of signal")
        print(f"  ~{win:,} windows of {len(slots)}x{route.window_samples}")
        print(f"  ~{win*bytes_per/1e12:.2f} TB uncompressed, "
              f"~{win*bytes_per*0.5/1e12:.2f} TB after gzip")
        print("  MAX_RECORDINGS or STRIDE_SECONDS cap this if it is too large "
              "for the filesystem.")
        # These are DURATIONS, not runtimes. Header reads cost milliseconds;
        # what the array actually spends its time on is decoding, filtering and
        # resampling, which this mode never does. Sizing an array off the
        # "hours" line above reads a number about the corpus as a number about
        # the job, and they differ by more than an order of magnitude.
        print(f"\n  THE {total_h:,.0f} HOURS ABOVE IS SIGNAL DURATION, NOT "
              f"COMPUTE TIME. --inspect opens headers only.")
        print("  To size the array, measure one task on a few files:")
        print(f"    time JOBS=8 MAX_RECORDINGS=40 OUT_DIR=/tmp/{dataset_id}_probe \\")
        print(f"        DATASET={dataset_id} RAW_ROOT={root} \\")
        print("        bash EEG/preprocess_eeg_corpus.sh")
        print(f"  then: {est_files:,} files / (40 / elapsed) / N_TASKS, against "
              f"the sbatch --time.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=sorted(ADAPTERS) + ["smoke"])
    p.add_argument("--root", default=None, help="raw corpus root (not downloaded)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mains-hz", type=float, default=None,
                   help="50 or 60. Required when the corpus does not say and "
                        "the adapter has no default: a notch at the wrong "
                        "frequency leaves the interference and removes signal.")
    p.add_argument("--no-notch", action="store_true",
                   help="skip mains notch entirely")
    p.add_argument("--registry", default=None, metavar="PATH",
                   help=f"dataset registry with the per-corpus mains frequency "
                        f"and native rates (default: {DEFAULT_REGISTRY})")
    p.add_argument("--dump-channels", type=int, default=None, metavar="N",
                   help="print the VERBATIM channel list of N recordings and "
                        "exit. When --inspect reports poor slot coverage, the "
                        "next question is always which names the corpus "
                        "actually uses, and no summary answers it -- the "
                        "10-5 'h' positions (FFC5h, CPP3h) and vendor spellings "
                        "are only visible in full.")
    p.add_argument("--derive-slots", default=None, metavar="PATH",
                   help="read the corpus and write a candidate slot list for "
                        "its route to PATH, plus the vocabulary entries it "
                        "would need. Nothing is applied: a slot list is "
                        "permanent once shards exist, so it is reviewed and "
                        "checked in by hand.")
    p.add_argument("--derive-slots-from", type=int, default=8, metavar="N",
                   help="how many recordings --derive-slots reads (default 8)")
    p.add_argument("--verify-powerline", action="store_true",
                   help="with --inspect: measure the 49-51 and 59-61 Hz bands "
                        "against their neighbours and report which one carries "
                        "a peak. FACED and PhysioNetMI publish no trustworthy "
                        "PowerLineFrequency field; this is how the registry's "
                        "value gets confirmed instead of assumed.")
    p.add_argument("--psd-verified", action="store_true",
                   help="acknowledge that --verify-powerline has been run for "
                        "this corpus; silences the reminder.")
    p.add_argument("--highpass-hz", type=float, default=0.5)
    p.add_argument("--clip-sigma", type=float, default=20.0)
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--stride-seconds", type=float, default=None,
                   help="default: no overlap")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-recordings", type=int, default=None)
    p.add_argument("--allow-upsample-faced", action="store_true",
                   help="permit FACED's 250 Hz release. The official "
                        "configuration never sets this.")
    p.add_argument("--unk-rate-max", type=float, default=0.01,
                   help="fail if more than this fraction of the channels placed "
                        "into slots carry an <unk> id")
    p.add_argument("--min-slot-coverage", type=float, default=0.75,
                   help="skip a RECORDING that fills less than this fraction of "
                        "the route's slots, logging it as a failure. A file "
                        "carrying twelve of nineteen electrodes yields windows "
                        "that are more mask than measurement; the aggregate "
                        "gate below averages those away.")
    p.add_argument("--max-empty-slot-rate", type=float, default=0.25,
                   help="fail if more than this fraction of the route's slots "
                        "are left unfilled on average. This is the gate that "
                        "catches an adapter that cannot name the montage; "
                        "TDBRAIN legitimately sits at 18.8%%.")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be read, process nothing")
    p.add_argument("--inspect", type=int, default=None, metavar="N",
                   help="read the headers of N recordings and report what the "
                        "corpus actually looks like -- layout, identity rule, "
                        "sampling rates, montages, channel names, how many "
                        "resolve. Run this before a long job; it opens headers "
                        "only and processes nothing.")
    p.add_argument("--shard", type=_shard_spec, default=None, metavar="I/N",
                   help="process only shard I of N, sharded by subject so a "
                        "subject is never split across tasks. For SLURM arrays.")
    p.add_argument("--max-recording-minutes", type=float, default=30.0,
                   help="process at most this much of any one recording. "
                        "get_data() returns float64 and every filter wants "
                        "another copy, so a four-hour 1000 Hz file is ~20 GB in "
                        "one worker and eight of those take the node down. "
                        "Truncation is recorded per shard as "
                        "truncated_to_minutes, never silent. 0 disables it.")
    p.add_argument("--file-list", default=None, metavar="PATH",
                   help="cache the corpus file listing here. Read if it exists, "
                        "written if it does not. Walking TUEG's tree costs "
                        "tens of minutes on a parallel filesystem and every "
                        "array task would otherwise pay it separately.")
    p.add_argument("--write-file-list", default=None, metavar="PATH",
                   help="walk the corpus, write the listing to PATH, and exit. "
                        "Run once before submitting an array.")
    p.add_argument("--jobs", type=int, default=1,
                   help="worker processes within this task. The filtering and "
                        "resampling are single-threaded, so a task holding 32 "
                        "cores decodes on one of them unless this is set. "
                        "Only the path-addressable corpora (TUEG) use it.")
    p.add_argument("--redo-truncated", action="store_true",
                   help="with --resume, reprocess the recordings a previous run "
                        "cut short, and only those. Raising "
                        "--max-recording-minutes helps nothing otherwise: every "
                        "shard already exists, so resume skips the whole "
                        "corpus. Each shard records the cap it was written "
                        "under, so this reaches exactly the ones that lost "
                        "data.")
    p.add_argument("--resume", action="store_true",
                   help="skip recordings whose shard HDF5 already exists")
    p.add_argument("--split-mode", choices=["exact", "hash"], default=None,
                   help="'exact' partitions the subject list (needs all of it); "
                        "'hash' decides per subject independently, which is "
                        "required when sharding. Defaults to hash under "
                        "--shard, exact otherwise.")
    p.add_argument("--smoke-test", action="store_true",
                   help="synthetic data; the ONLY path that fabricates signal")
    p.add_argument("--smoke-as", default="tdbrain",
                   choices=sorted(EEG_DATASETS))
    p.add_argument("--smoke-subjects", type=int, default=6)
    p.add_argument("--smoke-recordings", type=int, default=2)
    p.add_argument("--smoke-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.smoke_test:
        dataset_id = args.smoke_as
        adapter = adapt_smoke
    else:
        dataset_id = args.dataset
        if dataset_id == "smoke":
            return _fail("--dataset smoke needs --smoke-test.")
        adapter = ADAPTERS[dataset_id]
        if not args.root:
            return _fail("--root is required. This script never downloads a "
                         "corpus and never falls back to synthetic data; "
                         "--smoke-test is the explicit way to run without one.")
        if not os.path.isdir(args.root):
            return _fail(f"--root {args.root} does not exist.")

    spec = EEG_DATASETS[dataset_id]
    route = ROUTES[spec.route_id]
    slots = spec.slots

    mains_hz, mains_why = resolved_mains(dataset_id, args)
    args._mains_why = mains_why
    print(f"mains: {mains_hz if mains_hz is not None else 'no notch'} "
          f"({mains_why})", file=sys.stderr)
    entry = registry_for(dataset_id, args)
    if entry.get("verify_powerline_by_psd") and mains_hz is not None \
            and not getattr(args, "psd_verified", False):
        print(f"  NOTE: {dataset_id}'s mains frequency is inferred, not "
              f"published. Run --inspect N --verify-powerline once and confirm "
              f"the peak is at {mains_hz:g} Hz before the full array.",
              file=sys.stderr)

    cfg = PreprocessConfig(
        highpass_hz=args.highpass_hz,
        notch_hz=mains_hz,
        clip_sigma=args.clip_sigma, window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds, val_fraction=args.val_fraction,
        split_seed=args.split_seed)

    if args.derive_slots:
        return derive_slots(dataset_id, args.root, args, slots, route)

    if args.dump_channels:
        return dump_channels(dataset_id, args.root, args, slots, route)

    if args.write_file_list:
        if not args.root:
            return _fail("--write-file-list needs --root.")
        t0 = time.time()
        paths = iter_tueg_files(args.root) if dataset_id == "tueg" else \
            _walk(args.root, READABLE_EXTS)
        tmp = f"{args.write_file_list}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(args.write_file_list)) or ".",
                    exist_ok=True)
        with open(tmp, "w") as f:
            f.write("\n".join(paths) + "\n")
        os.replace(tmp, args.write_file_list)
        print(f"{len(paths)} file(s) listed in {time.time()-t0:.0f}s -> "
              f"{args.write_file_list}")
        print(f"  pass --file-list {args.write_file_list} to every array task")
        return 0

    if args.inspect:
        if not args.root:
            return _fail("--inspect needs --root.")
        return inspect_corpus(dataset_id, args.root, args, slots, route)

    # An array task must not write over its siblings' manifests, so every output
    # this task owns carries its shard index. build_eeg_c1_manifest.py merges
    # them; without the suffix, task 7 finishing last would leave a manifest
    # describing only task 7's shards.
    suffix = f".{args.shard[0]:04d}" if args.shard else ""
    if args.split_mode is None:
        args.split_mode = "hash" if args.shard else "exact"
    if args.shard and args.split_mode != "hash":
        return _fail("--shard requires --split-mode hash: no task sees the "
                     "whole subject list, so an exact partition would put the "
                     "same subject on different sides in different tasks.")

    os.makedirs(args.out_dir, exist_ok=True)
    fail_path = os.path.join(args.out_dir,
                             f"preprocessing_failures{suffix}.jsonl")
    entries: List[Dict] = []
    failures: List[Dict] = []

    print(f"dataset={dataset_id}  route={route.route_id}  "
          f"{route.n_channels}x{route.window_samples} @ {route.sampling_rate}Hz")
    print(f"slots={len(slots)}  out={args.out_dir}"
          + ("  [SMOKE TEST -- synthetic]" if args.smoke_test else ""))

    n_seen = 0
    if dataset_id == "tueg" and args.jobs > 1 and not args.dry_run:
        paths = iter_tueg_files(args.root, args.file_list)
        if args.shard:
            paths = shard_files(paths, args, args.root)
        if args.resume:
            keep = []
            for path in paths:
                safe = os.path.relpath(path, args.root).replace(
                    os.sep, "__").replace(" ", "_")
                done = os.path.join(args.out_dir, "shards", f"{safe}.h5")
                if os.path.isfile(done) and os.path.getsize(done) > 0:
                    try:
                        import h5py
                        with h5py.File(done, "r") as f:
                            prov = json.loads(
                                f.attrs.get("preprocessing_provenance", "{}"))
                            # Raising the cap only helps the recordings the old
                            # cap actually cut. Reprocessing the whole corpus to
                            # reach them costs a full round and another 0.5 TB;
                            # the shard says whether it was one of them.
                            was_cut = prov.get("truncated_to_minutes")
                            if (args.redo_truncated and was_cut
                                    and (not args.max_recording_minutes
                                         or was_cut < args.max_recording_minutes)):
                                raise _RedoShard()
                            valid = np.asarray(f["valid_channel_mask"][...], bool)
                            n_win = int(f["data"].shape[0])
                            if not _shard_data_is_readable(f, n_win):
                                raise _RedoShard()
                            ident = tueg_identity(path, args.root)
                            entries.append({
                                "path": done, "dataset_id": dataset_id,
                                "route_id": route.route_id, "n_windows": n_win,
                                "subjects": [ident["subject"]],
                                "qc": {"channel_missing_rate":
                                       float(1.0 - valid.mean())},
                                "unknown_channel_names":
                                    prov.get("unknown_channel_names", []),
                                "empty_slots": prov.get("empty_slots", []),
                                "unmatched_source_channels":
                                    prov.get("unmatched_source_channels", []),
                                "placed_total": int(valid.sum()),
                                "placed_unk": int((np.asarray(
                                    f["channel_ids"][...])[valid] == 1).sum()),
                                "n_channels_recorded": int(valid.size),
                                "source_sampling_rate": float(
                                    f.attrs.get("source_sampling_rate", 0.0)),
                                "duration_seconds": float(
                                    n_win * cfg.window_seconds)})
                        continue
                    except _RedoShard:
                        pass          # deliberately reprocess this one
                    except Exception:                          # noqa: BLE001
                        pass
                keep.append(path)
            print(f"  {len(entries)} already done, {len(keep)} to do")
            paths = keep
        if args.max_recordings:
            paths = paths[: args.max_recordings]
        print(f"  {len(paths)} file(s) on {args.jobs} worker(s)")
        got, failed_rows = _run_parallel(paths, args, dataset_id, cfg, slots,
                                         route)
        entries.extend(got)
        failures.extend(failed_rows)
        n_seen = len(paths)
    else:
      try:
          for rec in adapter(args.root or "", args):
              # Sharding for the streaming adapters. TUEG shards its file list
              # up front because it has one; these adapters yield Recordings, so
              # the filter goes here -- but on the same subject hash, so a
              # subject lands on exactly one task either way. Without this every
              # task of an --array=0-63 read the whole corpus and raced its
              # siblings for the same output paths.
              # Backstop. Every adapter calls owns() before reading, which is
              # where the saving is; this catches one that forgets to.
              if not owns(rec.subject_id, args):
                  continue
              n_seen += 1
              if args.max_recordings and n_seen > args.max_recordings:
                  break
              if args.dry_run:
                  print(f"  would read {rec.recording_id}: "
                        f"{len(rec.channel_names)}ch @ {rec.sampling_rate}Hz")
                  continue
              safe = rec.recording_id.replace(os.sep, "__").replace(" ", "_")
              done = os.path.join(args.out_dir, "shards", f"{safe}.h5")
              if args.resume and os.path.isfile(done) and os.path.getsize(done) > 0:
                  # Re-read the shard rather than re-deriving it, so a resumed run
                  # writes the same manifest a complete one would.
                  try:
                      import h5py
                      with h5py.File(done, "r") as f:
                          # Every field the summary needs is already in the shard,
                          # so a resumed run reports what a complete one would.
                          # Rebuilding the entry from the Recording alone left the
                          # coverage gates reading keys that were not there.
                          prov = json.loads(
                              f.attrs.get("preprocessing_provenance", "{}"))
                          valid = np.asarray(f["valid_channel_mask"][...], bool)
                          n_win = int(f["data"].shape[0])
                          if not _shard_data_is_readable(f, n_win):
                              raise _RedoShard()
                          entries.append({
                              "path": done, "dataset_id": dataset_id,
                              "route_id": route.route_id,
                              "n_windows": n_win,
                              "subjects": [rec.subject_id],
                              "qc": {"channel_missing_rate":
                                     float(1.0 - valid.mean())},
                              "unknown_channel_names":
                                  prov.get("unknown_channel_names", []),
                              "empty_slots": prov.get("empty_slots", []),
                              "unmatched_source_channels":
                                  prov.get("unmatched_source_channels", []),
                              "placed_total": int(valid.sum()),
                              "placed_unk": int(
                                  (np.asarray(f["channel_ids"][...])[valid] == 1).sum()),
                              "n_channels_recorded": len(rec.channel_names),
                              "source_sampling_rate": float(rec.sampling_rate),
                              "duration_seconds": float(
                                  n_win * cfg.window_seconds)})
                      continue
                  except Exception:                              # noqa: BLE001
                      pass          # unreadable: fall through and rebuild it
              try:
                  entry = process_recording(
                      rec, dataset_id, cfg, slots, route, args.out_dir,
                      {"synthetic": bool(args.smoke_test)}, args_ref=args)
                  if entry is None:
                      failures.append({"recording_id": rec.recording_id,
                                       "reason": "no whole window in recording"})
                  else:
                      entries.append(entry)
                      print(f"  {rec.recording_id}: {entry['n_windows']} windows")
              except Exception as exc:                       # noqa: BLE001
                  failures.append({"recording_id": rec.recording_id,
                                   "reason": str(exc),
                                   "traceback": traceback.format_exc(limit=3)})
      except PreprocessError as exc:
          return _fail(str(exc))

    if args.dry_run:
        print(f"\ndry run: {n_seen} recording(s) would be read")
        return 0
    if not entries:
        with open(fail_path, "w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")
        return _fail(f"no recording produced a window. See {fail_path}")

    # -- coverage gates ------------------------------------------------------ #
    #
    # Two different failures, measured separately, because the obvious single
    # ratio conflates them and stops a corpus that is perfectly fine.
    #
    # TUEG carries five non-EEG rows in every recording -- EKG, PHOTIC, IBI,
    # BURSTS, SUPPR -- which is 21% of a 24-row file. They match no slot and are
    # dropped, which is correct: they are not electrodes and never reach the
    # model. Counting them as "outside the vocabulary" and refusing the corpus
    # measures how many auxiliary channels the recording system writes, not
    # whether the montage was understood.
    #
    # What actually matters is whether the route's SLOTS got filled. A scalp
    # electrode whose name the adapter fails to normalise does not become <unk>
    # -- it fails to match a slot, the slot stays empty and masked out, and the
    # montage silently shrinks. That is the one to stop for.
    total_slots = len(slots) * len(entries)
    empty_slots = sum(len(e["empty_slots"]) for e in entries)
    empty_rate = empty_slots / max(1, total_slots)

    # And of the channels that DID reach a slot, how many carry an <unk> id.
    # Zero by construction -- a slot name is in the vocabulary -- so a non-zero
    # value means the slot list and the vocabulary have drifted apart.
    placed_unk = sum(e["placed_unk"] for e in entries)
    placed_total = sum(e["placed_total"] for e in entries)
    unk_rate = placed_unk / max(1, placed_total)

    dropped = sorted({n for e in entries for n in e["unmatched_source_channels"]})
    # A dropped name that IS a known electrode is worth saying out loud: it is a
    # real channel this route has no slot for, not an auxiliary row.
    dropped_electrodes = [n for n in dropped
                          if normalize_channel_name(n) in CHANNEL_TO_ID]
    if dropped:
        print(f"\n  dropped {len(dropped)} non-slot channel name(s): "
              f"{dropped[:12]}" + (" ..." if len(dropped) > 12 else ""))
    if dropped_electrodes:
        print(f"  NOTE: {len(dropped_electrodes)} of them are known electrodes "
              f"that {route.route_id} has no slot for: {dropped_electrodes[:12]}")

    if unk_rate > args.unk_rate_max:
        return _fail(
            f"{unk_rate:.1%} of the channels placed into slots carry <unk> "
            f"(limit {args.unk_rate_max:.1%}). The slot list and CHANNEL_VOCAB "
            f"have drifted apart; the channel embedding would be learning one "
            f"row for several electrodes.")

    if empty_rate > args.max_empty_slot_rate:
        return _fail(
            f"{empty_rate:.1%} of {route.route_id}'s slots were left empty "
            f"(limit {args.max_empty_slot_rate:.1%}), averaged over "
            f"{len(entries)} recording(s).\n\n"
            f"  Slots nothing filled: "
            f"{sorted({n for e in entries for n in e['empty_slots']})[:24]}\n"
            f"  Channel names that matched no slot: {dropped[:24]}\n\n"
            f"  A montage this incomplete is usually an adapter naming problem "
            f"rather than a corpus that genuinely lacks the electrodes. Run "
            f"--inspect to see what the files actually carry. Raise "
            f"--max-empty-slot-rate if the corpus really is this sparse -- "
            f"TDBRAIN legitimately leaves 6 of 32 empty, which is 18.8%.")

    # -- subject split ------------------------------------------------------- #
    subjects = sorted({s for e in entries for s in e["subjects"]})
    if args.split_mode == "hash":
        vset = {s for s in subjects
                if subject_split_side(s, cfg.val_fraction, cfg.split_seed) == "val"}
        tset = set(subjects) - vset
        train_subj, val_subj = sorted(tset), sorted(vset)
    else:
        train_subj, val_subj = split_subjects(subjects, cfg.val_fraction,
                                              cfg.split_seed)
        tset, vset = set(train_subj), set(val_subj)
    assert not (tset & vset), "a subject landed in both splits"

    def side(entry):
        subs = set(entry["subjects"])
        if subs <= vset:
            return "val"
        if subs <= tset:
            return "train"
        raise SystemExit(f"shard {entry['path']} spans train and val subjects")

    manifests = {"train": [], "val": []}
    for e in entries:
        manifests[side(e)].append(e)

    for split, rows in manifests.items():
        with open(os.path.join(args.out_dir,
                                f"manifest_{split}{suffix}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if k != "qc"}) + "\n")

    stats = {
        "dataset_id": dataset_id, "route_id": route.route_id,
        "n_subjects": len(subjects),
        "n_subjects_train": len(train_subj), "n_subjects_val": len(val_subj),
        "n_recordings": len(entries),
        "n_windows": sum(e["n_windows"] for e in entries),
        "n_windows_train": sum(e["n_windows"] for e in manifests["train"]),
        "n_windows_val": sum(e["n_windows"] for e in manifests["val"]),
        "duration_seconds": sum(e["duration_seconds"] for e in entries),
        "n_failed": len(failures),
        "unk_channel_rate": unk_rate,
        "empty_slot_rate": empty_rate,
        "dropped_channel_names": dropped,
        "target_sampling_rate": route.sampling_rate,
        # The rates actually seen, not the rate the corpus is documented at.
        # TUEG's varies per recording, and a corpus that turns out to hold two
        # rates is something to find here rather than in a training curve.
        "source_sampling_rates": sorted({e["source_sampling_rate"]
                                         for e in entries}),
        "channel_missing_rate": float(np.mean(
            [e["qc"].get("channel_missing_rate", 0.0) for e in entries])),
        "qc": {k: float(np.mean([e["qc"][k] for e in entries if k in e["qc"]]))
               for k in ("amplitude_mean", "amplitude_std", "amplitude_min",
                         "amplitude_max", "psd_delta", "psd_theta", "psd_alpha",
                         "psd_beta", "psd_gamma", "psd_high_gamma")
               if any(k in e["qc"] for e in entries)},
        "preprocess_config": cfg.provenance({"dataset_id": dataset_id}),
        "synthetic": bool(args.smoke_test),
        "shard": list(args.shard) if args.shard else None,
        "split_mode": args.split_mode,
    }
    with open(os.path.join(args.out_dir,
                           f"dataset_statistics{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(fail_path, "w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")

    print(f"\n{stats['n_recordings']} recordings, {stats['n_windows']} windows, "
          f"{stats['n_subjects']} subjects "
          f"({len(train_subj)} train / {len(val_subj)} val)")
    print(f"  {len(failures)} failure(s) -> {fail_path}")
    print(f"  manifests + dataset_statistics.json in {args.out_dir}")
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
