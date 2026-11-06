#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One ECG corpus -> HDF5 shards and manifests for ECG C1 pretraining.

    # look before processing: headers, leads, rates, units, amplitudes, QC
    python ECG/preprocess_ecg_corpus.py --dataset mimic_iv_ecg \\
        --root $ECG_ROOT/MIMIC-IV-ECG/raw --out-dir $CORPUS/mimic_iv_ecg --inspect 40

    # process (one process, 16 workers)
    python ECG/preprocess_ecg_corpus.py --dataset mimic_iv_ecg \\
        --root $ECG_ROOT/MIMIC-IV-ECG/raw --out-dir $CORPUS/mimic_iv_ecg --jobs 16

    # or as one task of a SLURM array
    ... --task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT

    # then merge every dataset's manifests into the two the trainer reads
    python scripts/build_eeg_c1_manifest.py --modality ecg --corpus-root $CORPUS

``--root`` is the raw download: a directory, or for the WFDB corpora the
PhysioNet zip itself -- records are read out of the archive one at a time, so
Icentia11k's 1.1 TB never has to be unpacked. PulseDB's root is the directory
of its downloaded pieces (``PulseDB_MIMIC.zip.001`` ...), read as the archive
they were cut from, one subject inflated at a time into ``--tmp-dir``.

HOW THE WORK IS CUT. The records are listed once (cached in
``<out-dir>/record_list.txt``), sorted, and cut into UNITS of
``--records-per-unit``. A worker takes a whole unit, writes its shards -- one
for the training side, one for the validation side -- and then a part file
``parts/unit_NNNNNN.json`` that records them. The part file is written last and
atomically, so a unit either has one or does not exist yet; a task that is
killed and resubmitted skips every unit with a part file and redoes the rest.
``--task I/N`` takes units I, I+N, I+2N, ... so an array covers each once.

THE SPLIT IS BY SUBJECT, by hash: ``subject_split_side`` decides a subject's
side from its id alone, so every task, in any order, puts a subject on the
same side and no shard ever spans both. Where a corpus carries no patient id
(Georgia) the record is its own subject, and the log says so.

Everything a window goes through after it is read is
physiowave.ecg_c1.preprocess -- the same for every corpus.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.ecg_c1.preprocess import (QC_REASONS,  # noqa: E402
                                          ECGPreprocessConfig,
                                          ECGPreprocessError,
                                          process_ecg_record,
                                          strip_zero_padding, to_millivolts)
from physiowave.ecg_c1.routes import (DOWNSTREAM_ONLY,  # noqa: E402
                                      PRETRAIN_DATASETS, ROUTES)
from physiowave.eeg_c1.preprocess import (subject_split_side,  # noqa: E402
                                          write_shard)


@dataclass
class ECGRecord:
    record_id: str
    subject_id: str
    data: np.ndarray                    # [leads, samples]
    lead_names: List[str]
    fs: float
    unit: str
    notes: Dict = field(default_factory=dict)


@dataclass
class Modality:
    """What the driver below needs to know about one modality's corpus.

    The driver -- units, part files, tasks, resume, assemble, inspect -- is
    the same for every modality. ECG registers itself at the bottom of this
    file; EMG/preprocess_emg_corpus.py imports this module, registers its own
    and calls ``main(modality="emg")``. A worker is handed the modality's
    NAME and looks it up here, so the registration has to happen at import
    time of the script that runs -- which a spawned worker re-imports.
    """

    name: str
    datasets: Dict
    routes: Dict
    downstream_only: Tuple[str, ...]
    adapters: Dict
    smoke_records: object          # (dataset_id, n, seed) -> [ECGRecord]
    process_record: object         # process_ecg_record's signature
    root_var: str                  # the env var of the download root
    download_script: str
    #: ECGPreprocessConfig fields this modality sets differently
    config_defaults: Dict = field(default_factory=dict)
    amplitude_hint: str = ""


MODALITIES: Dict[str, Modality] = {}


# =========================================================================== #
# Reading: WFDB, from a directory or straight out of a PhysioNet zip
# =========================================================================== #

def _is_zip(root: str) -> bool:
    return root.lower().endswith(".zip") and os.path.isfile(root)


# A record key for a zipped corpus carries where its members live:
#
#     <stem>|<name>:<offset>:<csize>:<usize>:<method>;<name>:...
#
# so a worker reads exactly those bytes and never parses the archive's central
# directory. MIMIC-IV-ECG's zip holds 1.6 M members and Icentia11k's about as
# many; zipfile.ZipFile builds a ZipInfo for every one of them, which is on the
# order of a gigabyte per process -- sixteen workers would each hold one. The
# listing opens the archive once and pays that once.
_WFDB_PARTS = (".hea", ".dat", ".mat")


def _zip_members_by_stem(root: str) -> Dict[str, List[zipfile.ZipInfo]]:
    by_stem: Dict[str, List[zipfile.ZipInfo]] = {}
    with zipfile.ZipFile(root) as zf:
        for info in zf.infolist():
            stem, ext = os.path.splitext(info.filename)
            if ext in _WFDB_PARTS:
                by_stem.setdefault(stem, []).append(info)
    return by_stem


def _zip_key(stem: str, infos: List[zipfile.ZipInfo]) -> str:
    spec = ";".join(f"{os.path.basename(i.filename)}:{i.header_offset}:"
                    f"{i.compress_size}:{i.file_size}:{i.compress_type}"
                    for i in sorted(infos, key=lambda i: i.filename))
    return f"{stem}|{spec}"


def record_stem(key: str) -> str:
    """The record's path inside the corpus, whatever the key carries."""
    return key.split("|", 1)[0]


def _read_zip_member(root: str, offset: int, csize: int, usize: int,
                     method: int, ctx: Dict) -> bytes:
    """One member's bytes, from its local header on, without the directory."""
    import struct
    import zlib
    f = ctx.get(("zipfh", root))
    if f is None:
        f = open(root, "rb")
        ctx[("zipfh", root)] = f
    f.seek(offset)
    head = f.read(30)
    if head[:4] != b"PK\x03\x04":
        raise ECGPreprocessError(
            f"no local file header at offset {offset} of {root}: the archive "
            f"changed since record_list.txt was written (--refresh-list)")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    f.seek(offset + 30 + name_len + extra_len)
    data = f.read(csize)
    if method == zipfile.ZIP_STORED:
        out = data
    elif method == zipfile.ZIP_DEFLATED:
        out = zlib.decompressobj(-15).decompress(data)
    else:
        raise ECGPreprocessError(f"zip compression method {method} unsupported")
    if len(out) != usize:
        raise ECGPreprocessError(
            f"member at {offset} inflated to {len(out)} bytes, expected {usize}"
            f" -- a truncated archive")
    return out


def _header_files(header_text: str) -> List[str]:
    """The signal file names a WFDB header refers to, in order, deduplicated."""
    lines = [ln for ln in header_text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    out: List[str] = []
    for ln in lines[1:]:
        name = ln.split()[0]
        if name not in out:
            out.append(name)
    return out


def _read_mat_record(path_stem: str):
    """A Challenge-style record: WFDB header + MATLAB v4 ``val`` array.

    Used only when wfdb-python cannot read the .mat itself. The header's gain
    and baseline turn the stored integers into physical units, exactly as
    wfdb would.
    """
    import scipy.io
    import wfdb
    hdr = wfdb.rdheader(path_stem)
    mat = scipy.io.loadmat(os.path.join(os.path.dirname(path_stem),
                                        hdr.file_name[0]))
    val = np.asarray(mat["val"], dtype=np.float64)
    gain = np.asarray([g if g else 200.0 for g in hdr.adc_gain], dtype=np.float64)
    base = np.asarray(hdr.baseline, dtype=np.float64)
    sig = (val - base[:, None]) / gain[:, None]
    return sig, list(hdr.sig_name), float(hdr.fs), list(hdr.units)


def read_wfdb(root: str, stem: str, ctx: Dict):
    """``(signal [C, T] physical units, names, fs, units per lead)``."""
    import wfdb

    def _read(path_stem):
        try:
            rec = wfdb.rdrecord(path_stem)
            return (np.asarray(rec.p_signal, dtype=np.float64).T,
                    list(rec.sig_name), float(rec.fs), list(rec.units))
        except Exception:                                      # noqa: BLE001
            if any(f.endswith(".mat") for f in
                   _header_files(open(path_stem + ".hea").read())):
                return _read_mat_record(path_stem)
            raise

    if not _is_zip(root):
        return _read(os.path.join(root, record_stem(stem)))

    key = stem
    stem, _, spec = key.partition("|")
    if not spec:
        raise ECGPreprocessError(
            f"{stem}: a zipped corpus needs keys with member offsets; "
            f"record_list.txt was written for a directory (--refresh-list)")
    tmp = ctx.get("tmp")
    if tmp is None:
        tmp = tempfile.mkdtemp(prefix="pw_ecg_")
        ctx["tmp"] = tmp
    written = []
    try:
        for part in spec.split(";"):
            name, off, csz, usz, meth = part.rsplit(":", 4)
            dst = os.path.join(tmp, name)
            with open(dst, "wb") as out:
                out.write(_read_zip_member(root, int(off), int(csz), int(usz),
                                           int(meth), ctx))
            written.append(dst)
        return _read(os.path.join(tmp, os.path.basename(stem)))
    finally:
        for p in written:
            try:
                os.unlink(p)
            except OSError:
                pass


def list_wfdb_records(root: str) -> List[str]:
    """Every record stem under ``root`` (a directory or a zip), sorted."""
    if _is_zip(root):
        by_stem = _zip_members_by_stem(root)
        return sorted(_zip_key(stem, infos) for stem, infos in by_stem.items()
                      if any(i.filename.endswith(".hea") for i in infos))
    else:
        stems = []
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for fn in files:
                if fn.endswith(".hea"):
                    stems.append(os.path.relpath(os.path.join(dirpath, fn[:-4]),
                                                 root))
    return sorted(stems)


def _wfdb_record(root, stem, ctx, subject, dataset_note=None) -> ECGRecord:
    sig, names, fs, units = read_wfdb(root, stem, ctx)
    # Per lead, so a file whose leads disagree on units is still converted
    # correctly -- and a unit nobody recognises fails the record by name.
    rows = [to_millivolts(sig[i], units[i] or "mV") for i in range(sig.shape[0])]
    return ECGRecord(record_stem(stem), subject, np.stack(rows), names, fs,
                     "mV", dataset_note or {})


# =========================================================================== #
# Adapters. Each: list_keys(root) -> sorted record keys; read(root, key, ctx)
# =========================================================================== #

class Adapter:
    #: Whether the corpus carries a patient id. False means record == subject.
    has_subject_ids = True
    #: Keys per unit of work (one shard pair). None: 1000 on 12-lead routes,
    #: 64 on single-lead ones.
    records_per_unit: Optional[int] = None

    def list_keys(self, root: str) -> List[str]:
        raise NotImplementedError

    def read(self, root: str, key: str, ctx: Dict) -> ECGRecord:
        raise NotImplementedError

    def read_records(self, root: str, key: str, ctx: Dict) -> List[ECGRecord]:
        """Every record behind one key. One, except where a file holds many."""
        return [self.read(root, key, ctx)]


class MimicIVECG(Adapter):
    """files/p1000/p10000032/s40689238/40689238 -- 10 s, 12 leads, 500 Hz."""

    _SUBJ = re.compile(r"(?:^|/)p(\d{8})(?:/|$)")

    def list_keys(self, root):
        # record_list.csv names every record, which is far cheaper on Lustre
        # than walking 800k directories. Fall back to the walk without it.
        csv_path = None
        if not _is_zip(root):
            for cand in glob.glob(os.path.join(root, "**", "record_list.csv"),
                                  recursive=True):
                csv_path = cand
                break
        if csv_path:
            prefix = os.path.relpath(os.path.dirname(csv_path), root)
            with open(csv_path) as f:
                keys = [os.path.normpath(os.path.join(prefix, r["path"]))
                        for r in csv.DictReader(f)]
            return sorted(keys)
        return [k for k in list_wfdb_records(root)
                if self._SUBJ.search(record_stem(k))]

    def read(self, root, key, ctx):
        m = self._SUBJ.search(record_stem(key))
        subject = f"p{m.group(1)}" if m else record_stem(key)
        return _wfdb_record(root, key, ctx, subject)


class GenericWFDB(Adapter):
    """One record per subject: Georgia, which carries no patient id."""

    has_subject_ids = False

    def list_keys(self, root):
        return list_wfdb_records(root)

    def read(self, root, key, ctx):
        return _wfdb_record(root, key, ctx,
                            os.path.basename(record_stem(key)))


class Icentia11k(Adapter):
    """p00/p00000/p00000_s00 -- one lead, 250 Hz, ~70 min per segment."""

    _SUBJ = re.compile(r"(p\d{5})_s\d+$")

    def list_keys(self, root):
        return [k for k in list_wfdb_records(root)
                if self._SUBJ.search(os.path.basename(record_stem(k)))]

    def read(self, root, key, ctx):
        m = self._SUBJ.search(os.path.basename(record_stem(key)))
        rec = _wfdb_record(root, key, ctx,
                           m.group(1) if m else record_stem(key))
        # One lead, whatever the header calls it: the route's only slot. Named
        # here and not by alias, so a two-lead file fails instead of being
        # half-read.
        if rec.data.shape[0] != 1:
            raise ECGPreprocessError(
                f"Icentia11k segment with {rec.data.shape[0]} leads "
                f"({rec.lead_names}); expected one")
        rec.notes["source_lead_name"] = rec.lead_names[0]
        rec.lead_names = ["patch1"]
        return rec


class CODE(Adapter):
    """CODE-15%: exams_partN.hdf5, ``tracings`` [N, 4096, 12] float32 @ 400 Hz.

    Checked against the published files (2026-09), which differ from their own
    README in three ways that matter here:

    * Every part carries ONE EXTRA ROW, all zeros, with ``exam_id == 0`` -- 20001
      rows for 20000 exams in exams.csv. It is not an exam and is not listed.
    * The documented padding (2800 or 4000 samples, zero-filled to 4096) is not
      what is there: 55-59% of exams are 2934 samples (7.3 s) with 581 zeros
      each side, 37-41% fill all 4096, a few percent are other lengths. The
      zeros are stripped wherever they are, so the layout does not matter.
    * The README calls the unit "1e-4 V" and in the same sentence says to
      multiply a signal in V by 1000 -- which is mV. The amplitudes settle it:
      the median lead spans ~2-3.4 units between its 0.5th and 99.5th
      percentiles, an ordinary ECG in mV and a tenth of one in 1e-4 V. Read as
      mV; --inspect's amplitude column is the check on your copy.

    Lead order DI, DII, DIII, AVR, AVL, AVF, V1-V6, mapped by name. The patient
    comes from exams.csv (exam_id -> patient_id).
    """

    LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    FS = 400.0
    UNIT = "mV"

    def _files(self, root):
        return sorted(glob.glob(os.path.join(root, "**", "*.hdf5"),
                                recursive=True))

    def list_keys(self, root):
        import h5py
        keys = []
        for path in self._files(root):
            with h5py.File(path, "r") as f:
                if "tracings" not in f or "exam_id" not in f:
                    continue
                ids = np.asarray(f["exam_id"][...])
            rel = os.path.relpath(path, root)
            keys.extend(f"{rel}#{i}" for i in np.nonzero(ids != 0)[0].tolist())
        return keys

    def _patients(self, root, ctx) -> Dict[int, str]:
        pat = ctx.get("code_patients")
        if pat is None:
            pat = {}
            for p in glob.glob(os.path.join(root, "**", "exams.csv"),
                               recursive=True):
                with open(p) as f:
                    for r in csv.DictReader(f):
                        try:
                            pat[int(float(r["exam_id"]))] = str(
                                int(float(r["patient_id"])))
                        except (KeyError, ValueError, TypeError):
                            continue
            ctx["code_patients"] = pat
        return pat

    def read(self, root, key, ctx):
        import h5py
        rel, row = key.rsplit("#", 1)
        row = int(row)
        path = os.path.join(root, rel)
        f = ctx.get(("h5", path))
        if f is None:
            f = h5py.File(path, "r")
            ctx[("h5", path)] = f
        exam_id = int(f["exam_id"][row])
        x = np.asarray(f["tracings"][row], dtype=np.float64).T      # [12, 4096]
        x = strip_zero_padding(x)
        if x.shape[1] == 0:
            raise ECGPreprocessError(f"exam {exam_id}: all-zero tracing")
        subject = self._patients(root, ctx).get(exam_id)
        if subject is None:
            raise ECGPreprocessError(
                f"exam {exam_id} is not in exams.csv, so its patient -- and "
                f"therefore its split side -- is unknown")
        return ECGRecord(f"{rel}#{exam_id}", f"code15_{subject}", x,
                         list(self.LEADS), self.FS, self.UNIT,
                         {"exam_id": exam_id,
                          "samples_after_unpadding": int(x.shape[1])})


class MedalCareXL(Adapter):
    """Simulated 12-lead ECGs: ``.../<class>/<split>/run_SXX/NNNNNN_<variant>.csv``.

    Each CSV is 12 rows (I, II, III, aVR, aVL, aVF, V1-V6) by 5000 samples at
    500 Hz, in mV, no header. Every signal comes in three variants: ``raw``
    (the clean simulation), ``noise`` (with measured ECG noise added at 15-20
    dB SNR) and ``filtered`` (the noisy one band-passed). Taking all three
    would put each simulated heart in the corpus three times, so ONE is taken:
    ``noise`` by default, the one that looks like a recording.

    THE SUBJECT IS THE TORSO MODEL. There are no patients; ``run_S62`` ...
    ``run_S74`` are 13 anatomical models, each simulated under many
    parameters. Splitting by file would put the same torso on both sides.
    """

    has_subject_ids = True
    FS = 500.0
    UNIT = "mV"
    VARIANT = os.environ.get("PW_MEDALCARE_VARIANT", "noise")
    _RUN = re.compile(r"(?:^|/)(run_S\d+)(?:/|$)")

    def list_keys(self, root):
        suffix = f"_{self.VARIANT}.csv"
        keys = []
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            if "__MACOSX" in dirpath or "/examples" in dirpath.replace(os.sep, "/"):
                continue
            for fn in files:
                if fn.endswith(suffix):
                    keys.append(os.path.relpath(os.path.join(dirpath, fn), root))
        return sorted(keys)

    def read(self, root, key, ctx):
        from physiowave.ecg_c1.leads import LEADS_12
        x = np.loadtxt(os.path.join(root, key), delimiter=",", ndmin=2)
        if x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T
        if x.shape[0] != 12:
            raise ECGPreprocessError(f"{key}: array {x.shape}, expected 12 leads")
        m = self._RUN.search(key.replace(os.sep, "/"))
        if m is None:
            raise ECGPreprocessError(
                f"{key}: no run_SXX directory, so its torso model -- the split "
                f"unit -- is unknown")
        rid = os.path.splitext(key)[0]
        return ECGRecord(rid, f"medalcare_{m.group(1)}", x, list(LEADS_12),
                         self.FS, self.UNIT, {"variant": self.VARIANT})


class SPH(Adapter):
    """Shandong Provincial Hospital 12-lead database (figshare collection 5779802).

    ``records/<ECG_ID>.h5``, one dataset ``ecg`` of shape [12, N], float16, in
    mV at 500 Hz; leads I, II, III, aVR, aVL, aVF, V1-V6. 10 to 56 s, so a
    record gives one to five windows. ``metadata.csv`` maps ECG_ID to
    Patient_ID -- 1,066 patients have more than one record, so the split is by
    patient, not by file. The machine already removed mains, baseline wander
    and muscle noise; the shared pipeline's notch and high-pass run anyway, and
    change little.
    """

    FS = 500.0
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

    def list_keys(self, root):
        return sorted(os.path.relpath(p, root) for p in
                      glob.glob(os.path.join(root, "**", "A*.h5"),
                                recursive=True))

    def _patients(self, root, ctx) -> Dict[str, str]:
        pat = ctx.get(("sph_patients", root))
        if pat is None:
            pat = {}
            for p in glob.glob(os.path.join(root, "**", "metadata.csv"),
                               recursive=True):
                with open(p, newline="") as f:
                    for r in csv.DictReader(f):
                        if r.get("ECG_ID") and r.get("Patient_ID"):
                            pat[r["ECG_ID"]] = r["Patient_ID"]
            ctx[("sph_patients", root)] = pat
        return pat

    def read(self, root, key, ctx):
        import h5py
        ecg_id = os.path.splitext(os.path.basename(key))[0]
        patient = self._patients(root, ctx).get(ecg_id)
        if patient is None:
            raise ECGPreprocessError(
                f"{ecg_id} is not in metadata.csv, so its patient -- and "
                f"therefore its split side -- is unknown")
        with h5py.File(os.path.join(root, key), "r") as f:
            x = np.asarray(f["ecg"][...], dtype=np.float64)
        if x.shape[0] != 12 and x.shape[-1] == 12:
            x = x.T
        return ECGRecord(ecg_id, f"sph_{patient}", x, list(self.LEADS),
                         self.FS, "mV")


class _PartsFile(io.RawIOBase):
    """``NAME.001``, ``NAME.002``, ... read as the one file they were cut from.

    PulseDB ships each half as a ZIP64 archive split into 15 GB pieces with no
    multi-volume structure: the pieces concatenate to an ordinary zip. This
    reads across them without writing the 388 GB concatenation, and zipfile
    and the member reader below take it like any seekable file.
    """

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = np.cumsum([0] + self.sizes)
        self.pos = 0
        self._fh: Dict[int, object] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = int(self.starts[-1]) + offset
        return self.pos

    def _part(self, i):
        fh = self._fh.get(i)
        if fh is None:
            fh = open(self.paths[i], "rb")
            self._fh[i] = fh
        return fh

    def read(self, n=-1):
        total = int(self.starts[-1])
        if n is None or n < 0:
            n = total - self.pos
        n = max(0, min(n, total - self.pos))
        out = bytearray()
        while n > 0:
            i = int(np.searchsorted(self.starts, self.pos, side="right") - 1)
            local = self.pos - int(self.starts[i])
            take = min(n, self.sizes[i] - local)
            fh = self._part(i)
            fh.seek(local)
            chunk = fh.read(take)
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
        super().close()


def _open_archive(path: str):
    """``path`` if it is a whole file, else its ``.001 ... .NNN`` pieces."""
    if os.path.isfile(path):
        return open(path, "rb")
    parts = sorted(glob.glob(glob.escape(path) + ".[0-9][0-9][0-9]"))
    if not parts:
        raise ECGPreprocessError(f"neither {path} nor {path}.001... exists")
    return _PartsFile(parts)


class PulseDB(Adapter):
    """PulseDB (pulselabteam/PulseDB): 10 s ICU/OR segments from MIMIC-III and VitalDB.

    One MATLAB 7.3 (HDF5) file per subject, ``PulseDB_{MIMIC,Vital}/pNNNNNN.mat``,
    holding a 1xN struct array ``Subj_Wins``: one element per 10 s segment,
    with the ECG, PPG and ABP in several renderings. The ECG taken here is
    ``ECG_Record`` -- lead II at 125 Hz in mV, as recorded. NOT ``ECG_Raw`` or
    ``ECG_F``: those are min-max scaled to [0, 1] per segment, which destroys
    the amplitude the normalisation and the QC work from.

    THE SUBJECT KEY INCLUDES THE SOURCE. The two halves number their subjects
    independently and 28 ids occur in both; ``p000188`` in MIMIC and in VitalDB
    are different people.

    ``--root`` is either the unpacked tree, or the directory holding the
    downloaded ``PulseDB_MIMIC.zip.001 ...`` and ``PulseDB_Vital.zip.001 ...``
    pieces, which are read in place: listing takes each archive's directory
    from its last piece, and a subject's .mat is inflated into ``--tmp-dir`` for
    as long as it is being read and then deleted. The largest is ~3 GB.
    """

    FS = 125.0
    records_per_unit = 4             # subjects; ~1,000 segments each on average
    ARCHIVES = ("PulseDB_MIMIC.zip", "PulseDB_Vital.zip")
    _SOURCE = re.compile(r"PulseDB_(MIMIC|Vital)", re.I)

    def list_keys(self, root):
        keys = []
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for fn in files:
                if fn.endswith(".mat") and self._SOURCE.search(dirpath):
                    keys.append(os.path.relpath(os.path.join(dirpath, fn), root))
        if keys:
            return sorted(keys)
        for name in self.ARCHIVES:
            path = os.path.join(root, name)
            if not (os.path.isfile(path) or os.path.isfile(path + ".001")):
                continue
            with _open_archive(path) as fh, zipfile.ZipFile(fh) as zf:
                for i in zf.infolist():
                    if i.filename.endswith(".mat"):
                        keys.append(f"{i.filename}|{name}:{i.header_offset}:"
                                    f"{i.compress_size}:{i.file_size}:"
                                    f"{i.compress_type}")
        return sorted(keys)

    def _materialise(self, root, key, ctx) -> Tuple[str, bool]:
        """A path h5py can open, and whether it is a temporary to delete."""
        member, _, spec = key.partition("|")
        if not spec:
            return os.path.join(root, member), False
        import struct
        import zlib
        name, off, csize, usize, method = spec.rsplit(":", 4)
        off, csize, usize, method = int(off), int(csize), int(usize), int(method)
        tmp_dir = ctx.get("tmp_dir") or os.environ.get("PW_ECG_TMP") \
            or tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        dst = os.path.join(tmp_dir, f"pulsedb_{os.getpid()}_"
                                    f"{os.path.basename(member)}")
        with _open_archive(os.path.join(root, name)) as fh:
            fh.seek(off)
            head = fh.read(30)
            if head[:4] != b"PK\x03\x04":
                raise ECGPreprocessError(
                    f"{member}: no local header at {off} -- the pieces changed "
                    f"since the listing (--refresh-list)")
            n_name, n_extra = struct.unpack("<HH", head[26:30])
            fh.seek(off + 30 + n_name + n_extra)
            inflate = (zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED
                       else None)
            if method not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
                raise ECGPreprocessError(f"{member}: zip method {method}")
            left, written = csize, 0
            with open(dst, "wb") as out:
                while left > 0:
                    chunk = fh.read(min(left, 1 << 24))
                    if not chunk:
                        break
                    left -= len(chunk)
                    data = inflate.decompress(chunk) if inflate else chunk
                    out.write(data)
                    written += len(data)
                if inflate:
                    tail = inflate.flush()
                    out.write(tail)
                    written += len(tail)
        if written != usize:
            os.unlink(dst)
            raise ECGPreprocessError(
                f"{member}: inflated to {written} bytes, expected {usize} -- a "
                f"missing or truncated piece")
        return dst, True

    def read_records(self, root, key, ctx):
        import h5py
        member = record_stem(key)
        m = self._SOURCE.search(member)
        source = m.group(1).lower() if m else "unknown"
        path, temporary = self._materialise(root, key, ctx)
        out: List[ECGRecord] = []
        try:
            with h5py.File(path, "r") as f:
                S = f["Subj_Wins"]

                def text(field, i):
                    return "".join(map(chr, np.asarray(
                        f[S[field][0, i]][()]).ravel()))

                refs = S["ECG_Record"][0]
                include = S["IncludeFlag"][0] if "IncludeFlag" in S else None
                subject = text("SubjectID", 0) if "SubjectID" in S else \
                    os.path.splitext(os.path.basename(member))[0]
                for i in range(len(refs)):
                    if include is not None and not bool(
                            np.asarray(f[include[i]][()]).ravel()[0]):
                        continue
                    x = np.asarray(f[refs[i]][()], dtype=np.float64).ravel()
                    seg = int(np.asarray(f[S["SegmentID"][0, i]][()]).ravel()[0]) \
                        if "SegmentID" in S else i
                    out.append(ECGRecord(
                        f"{source}/{subject}/seg{seg:06d}",
                        f"pulsedb_{source}_{subject}", x[None, :], ["II"],
                        self.FS, "mV", {"source": source}))
        finally:
            if temporary:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if not out:
            raise ECGPreprocessError(f"{member}: no included segment")
        return out

    def read(self, root, key, ctx):
        return self.read_records(root, key, ctx)[0]


ADAPTERS: Dict[str, Adapter] = {
    "mimic_iv_ecg": MimicIVECG(),
    "code15": CODE(),
    "medalcare_xl": MedalCareXL(),
    "georgia": GenericWFDB(),
    "icentia11k": Icentia11k(),
    "sph": SPH(),
    "pulsedb": PulseDB(),
}


# =========================================================================== #
# Synthetic ECG, for --smoke-test
# =========================================================================== #

#: How each corpus spells its leads and at what rate it records, so the smoke
#: corpus exercises the alias table and the resampler the real one will.
_SMOKE_NAMES = {
    "code15": CODE.LEADS,
    # MIMIC-IV-ECG's files store aVF before aVL.
    "mimic_iv_ecg": ["I", "II", "III", "aVR", "aVF", "aVL",
                     "V1", "V2", "V3", "V4", "V5", "V6"],
    "icentia11k": ["patch1"],
    "pulsedb": ["II"],
}

#: Record length per corpus, so each smoke corpus has the shape of the real
#: one: Icentia's long segments exercise the per-record cap, SPH's 10-60 s
#: records give several windows each.
_SMOKE_SECONDS = {"icentia11k": 120.0, "sph": 30.0}


def synthetic_ecg(n_leads: int, fs: float, seconds: float,
                  rng: np.random.Generator) -> np.ndarray:
    """``[12 or 1, T]`` in mV: beats from a rotating dipole, projected to leads.

    Three orthogonal components (a vectorcardiogram), each a sum of Gaussian
    P, QRS and T waves per beat; the limb leads follow Einthoven exactly and
    the precordials are fixed projections. Not physiology -- a signal with an
    ECG's shape and its inter-lead constraints, so the limb-group masking and
    the shared normalisation are exercised on something that has them.
    """
    T = int(round(seconds * fs))
    t = np.arange(T) / fs
    hr = rng.uniform(55, 110)
    rr = 60.0 / hr
    beats = np.arange(rng.uniform(0, rr), seconds, rr)
    waves = ((-0.20, 0.025, np.array([0.10, 0.15, 0.05])),     # P
             (0.00, 0.012, np.array([0.90, 1.20, -0.60])),     # QRS
             (0.25, 0.050, np.array([0.25, 0.35, 0.10])))      # T
    vcg = np.zeros((3, T))
    for b in beats:
        for off, width, amp in waves:
            vcg += amp[:, None] * np.exp(-0.5 * ((t - b - off) / width) ** 2)
    x, y, z = vcg
    i, ii = x, 0.5 * x + 0.87 * y
    leads = {"I": i, "II": ii, "III": ii - i, "aVR": -(i + ii) / 2,
             "aVL": i - ii / 2, "aVF": ii - i / 2}
    for k, ang in enumerate(np.linspace(-0.3, 1.4, 6), start=1):
        leads[f"V{k}"] = np.cos(ang) * z + np.sin(ang) * x
    order = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    sig = np.stack([leads[n] for n in order]) if n_leads == 12 else ii[None]
    wander = 0.1 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6.28))
    return sig + wander + rng.normal(0, 0.01, size=sig.shape)


def smoke_records(dataset_id: str, n: int, seed: int) -> List[ECGRecord]:
    spec = PRETRAIN_DATASETS[dataset_id]
    rng = np.random.default_rng(seed)
    leads = spec.native_leads
    names = _SMOKE_NAMES.get(dataset_id,
                             ["I", "II", "III", "aVR", "aVL", "aVF",
                              "V1", "V2", "V3", "V4", "V5", "V6"])
    fs = float(spec.native_rate or spec.route.sampling_rate)
    seconds = _SMOKE_SECONDS.get(dataset_id, 10.0)
    out = []
    for r in range(n):
        x = synthetic_ecg(leads, fs, seconds, rng)
        if dataset_id == "mimic_iv_ecg":
            x = x[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]   # the file order
        unit = "mV"
        if dataset_id == "code15":
            # Exercise the real path: zero-padded into a 4096-sample array.
            pad = np.zeros((x.shape[0], 4096))
            lo = (4096 - x.shape[1]) // 2
            pad[:, lo:lo + x.shape[1]] = x
            x = strip_zero_padding(pad)
        subject = f"smoke_{dataset_id}_s{r // 2:03d}"       # two records each
        out.append(ECGRecord(f"smoke_{r:05d}", subject, x, list(names), fs,
                             unit, {"synthetic": True}))
    return out


# =========================================================================== #
# One unit of work
# =========================================================================== #

_CTX: Dict = {}


def _cleanup_ctx():
    tmp = _CTX.get("tmp")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


import atexit  # noqa: E402
atexit.register(_cleanup_ctx)


def _unit_id(i: int) -> str:
    return f"{i:06d}"


def _keys_sha(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def _process_one(rec: ECGRecord, dataset_id, route, spec, mains_hz, cfg,
                 stats: Dict, failures: List[Dict], buf: Dict,
                 mod: "Modality") -> None:
    """One record through the pipeline, into the unit's train or val buffer."""
    stats["records_read"] += 1
    rate_key = f"{rec.fs:g}"
    stats["source_rates"][rate_key] = stats["source_rates"].get(rate_key, 0) + 1
    # As the FILE spelled them, for the statistics: the first few
    # distinct spellings are what tell you the alias table is right.
    lead_key = ",".join(map(str, [rec.notes["source_lead_name"]]
                            if "source_lead_name" in rec.notes
                            else rec.lead_names))
    seen = stats["source_lead_names"]
    if lead_key in seen or len(seen) < 8:
        seen[lead_key] = seen.get(lead_key, 0) + 1
    try:
        out = mod.process_record(rec.data, rec.lead_names, rec.fs,
                                 rec.unit, route, mains_hz, cfg,
                                 record_key=f"{dataset_id}:{rec.record_id}",
                                 slots=spec.lead_slots)
    except ECGPreprocessError as exc:
        stats["records_failed"] += 1
        failures.append({"record": rec.record_id, "reason": str(exc)})
        return
    except Exception as exc:                               # noqa: BLE001
        stats["records_failed"] += 1
        failures.append({"record": rec.record_id, "reason": str(exc),
                         "traceback": traceback.format_exc(limit=2)})
        return
    stats["windows_candidate"] += out.n_candidate_windows
    if out.n_candidate_windows == 0:
        # Not a failure and not QC: the record holds no whole window. A
        # corpus of 10 s ECGs never does this; CODE's shorter exams do,
        # and the count is how you find out how many.
        stats["records_shorter_than_window"] += 1
    for r, c in out.qc.items():
        stats["qc_dropped"][r] += int(c)
    if out.derived:
        stats["derived_limb_leads"] += 1
    n = out.windows.shape[0]
    if n == 0:
        return
    stats["records_with_windows"] += 1
    stats["windows_kept"] += n
    buf["_ids"] = out.channel_ids
    side = subject_split_side(rec.subject_id, cfg.val_fraction,
                              cfg.split_seed)
    # The windows are all a shard needs of the record; its raw signal -- up
    # to 70 minutes for an Icentia11k segment -- is let go now rather than
    # held until the unit's shards are written.
    rec.data = None
    buf[side].append((out, rec))


def process_unit(payload) -> Dict:
    """Read, preprocess and write one unit. Returns its part-file content.

    Module-level and fed only picklable arguments, so it crosses a process
    boundary. Shards first, part file last: the part file's existence is what
    says the unit is done.
    """
    (dataset_id, root, uid, keys, cfg, mains_hz, out_dir, smoke,
     smoke_seed, modality) = payload
    mod = MODALITIES[modality]
    spec = mod.datasets[dataset_id]
    route = mod.routes[spec.route_id]
    adapter = mod.adapters[dataset_id]
    t0 = time.time()

    buf = {"train": [], "val": [], "_ids": None}
    stats = {"records_read": 0, "records_with_windows": 0,
             "records_shorter_than_window": 0,
             "windows_candidate": 0, "windows_kept": 0,
             "qc_dropped": {r: 0 for r in QC_REASONS},
             "records_failed": 0, "derived_limb_leads": 0,
             "source_rates": {}, "source_lead_names": {}}
    failures: List[Dict] = []
    ids_ref = None

    synthetic = (mod.smoke_records(dataset_id, len(keys), smoke_seed)
                 if smoke else None)

    for j, k in enumerate(keys):
        # One read per key, not a generator over them: an exception raised
        # inside a generator finishes it, so the first unreadable record
        # would silently end the unit.
        try:
            recs = [synthetic[j]] if smoke else adapter.read_records(root, k, _CTX)
        except ECGPreprocessError as exc:
            stats["records_failed"] += 1
            failures.append({"record": record_stem(k), "reason": str(exc)})
            continue
        except Exception as exc:                               # noqa: BLE001
            stats["records_failed"] += 1
            failures.append({"record": record_stem(k), "reason": f"unreadable: {exc}",
                             "traceback": traceback.format_exc(limit=2)})
            continue
        for rec in recs:
            _process_one(rec, dataset_id, route, spec, mains_hz, cfg, stats,
                         failures, buf, mod)
        if buf["_ids"]:
            ids_ref = buf["_ids"]

    rows = {"train": [], "val": []}
    subjects = {"train": [], "val": []}
    for side in ("train", "val"):
        items = buf[side]
        if not items:
            continue
        windows = np.concatenate([o.windows for o, _ in items])
        subj = [r.subject_id for o, r in items for _ in range(o.windows.shape[0])]
        rid = [r.record_id for o, r in items for _ in range(o.windows.shape[0])]
        starts = np.concatenate([o.starts_seconds for o, _ in items])
        rates = sorted({r.fs for _, r in items})
        prov = cfg.provenance({
            "dataset_id": dataset_id, "route_id": route.route_id,
            "mains_hz": mains_hz,
            "source_sampling_rates": rates,
            "target_sampling_rate": route.sampling_rate,
            "upsampled_from_hz": [f for f in rates if f < route.sampling_rate],
            "records": len(items),
            "records_with_derived_limb_leads": sum(1 for o, _ in items
                                                   if o.derived),
            "synthetic": bool(smoke),
            "unit": uid,
        })
        path = os.path.abspath(os.path.join(out_dir, "shards",
                                            f"{uid}_{side}.h5"))
        entry = write_shard(path, windows, route, dataset_id,
                            list(spec.lead_slots), ids_ref, items[0][0].valid,
                            subj, rid, starts.tolist(), rates[0], prov)
        subjects[side] = entry.pop("subjects")
        entry["n_subjects"] = len(subjects[side])
        entry["n_records"] = len(items)
        entry["side"] = side
        rows[side].append(entry)

    part = {"unit": uid, "dataset_id": dataset_id, "n_keys": len(keys),
            "keys_sha": _keys_sha(keys), "rows": rows, "subjects": subjects,
            "stats": stats, "failures": failures,
            "seconds": time.time() - t0}
    part_path = os.path.join(out_dir, "parts", f"unit_{uid}.json")
    tmp = f"{part_path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(part, f)
    os.replace(tmp, part_path)
    return {"unit": uid, "stats": stats, "n_failures": len(failures)}


# =========================================================================== #
# Inspect
# =========================================================================== #

def inspect(dataset_id: str, root: str, keys: List[str], n: int,
            cfg: ECGPreprocessConfig, mains_hz, modality: str = "ecg") -> int:
    """Read ``n`` records spread over the corpus and say what they are.

    Nothing is written. The questions this answers are the ones that decide
    whether a full run is worth launching: are the leads named what the slot
    map expects, is the rate what the registry says, is the unit right (the
    amplitude column says -- a QRS is ~1 mV, not 1000), and how many windows
    would QC keep.
    """
    mod = MODALITIES[modality]
    spec = mod.datasets[dataset_id]
    route = mod.routes[spec.route_id]
    adapter = mod.adapters[dataset_id]
    pick = (np.linspace(0, len(keys) - 1, num=min(n, len(keys))).astype(int)
            if keys else [])
    print(f"{dataset_id}: {len(keys):,} record(s) under {root}")
    print(f"  route {route.describe()}")
    print(f"  mains notch {mains_hz or 'none'}  normalisation {cfg.normalization}")
    print(f"  {'record':<44s} {'fs':>6s} {'leads':>5s} {'sec':>7s} "
          f"{'p2p mV (median lead)':>21s}  windows kept/cand  verdict")
    agg = {"ok": 0, "failed": 0, "kept": 0, "cand": 0}
    names_seen: Dict[str, int] = {}
    for i in pick:
        k = keys[int(i)]
        shown = record_stem(k)[-44:]
        try:
            recs = adapter.read_records(root, k, _CTX)
            if len(recs) > 1:
                # A file of many segments (PulseDB): report its first, and say
                # how many there are.
                print(f"  {shown:<44s} {len(recs):,} segment(s) in this file")
                shown = recs[0].record_id[-44:]
            rec = recs[0]
            mv = to_millivolts(rec.data, rec.unit)
            p2p = float(np.median(np.nanmax(mv, axis=1) - np.nanmin(mv, axis=1)))
            spelled = ",".join([rec.notes["source_lead_name"]]
                               if "source_lead_name" in rec.notes
                               else rec.lead_names)
            names_seen[spelled] = names_seen.get(spelled, 0) + 1
            out = mod.process_record(rec.data, rec.lead_names, rec.fs,
                                     rec.unit, route, mains_hz, cfg,
                                     record_key=f"{dataset_id}:{rec.record_id}",
                                     slots=spec.lead_slots)
            drop = {r: c for r, c in out.qc.items() if c}
            verdict = ("shorter than one window" if not out.n_candidate_windows
                       else "ok" if not drop else f"dropped {drop}")
            agg["ok"] += 1
            agg["kept"] += out.windows.shape[0]
            agg["cand"] += out.n_candidate_windows
            print(f"  {shown:<44s} {rec.fs:6g} {rec.data.shape[0]:5d} "
                  f"{rec.data.shape[1] / rec.fs:7.1f} {p2p:21.3f}  "
                  f"{out.windows.shape[0]:>6d}/{out.n_candidate_windows:<6d}   "
                  f"{verdict}  subj={rec.subject_id}")
        except Exception as exc:                               # noqa: BLE001
            agg["failed"] += 1
            print(f"  {shown:<44s} FAILED: {exc}")
    print(f"\n  {agg['ok']} read, {agg['failed']} failed; "
          f"{agg['kept']}/{agg['cand']} candidate windows pass QC")
    print("  lead names as the files spell them:")
    for names, c in sorted(names_seen.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {c:4d} x  {names}")
    if mod.amplitude_hint:
        print(f"  {mod.amplitude_hint}")
    if not adapter.has_subject_ids:
        print("  NOTE: this corpus carries no patient id; each record is its "
              "own subject for the split.")
    return 0 if agg["ok"] else 1


# =========================================================================== #
# Driver
# =========================================================================== #

def _task_spec(text: str) -> Tuple[int, int]:
    try:
        i, n = (int(v) for v in text.split("/"))
    except Exception:                                          # noqa: BLE001
        raise argparse.ArgumentTypeError(f"--task wants I/N, got {text!r}")
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--task {text}: need 0 <= I < N")
    return i, n


#: How long a non-zero array task waits for task 0's listing. Listing a
#: million-file corpus on Lustre is the slow case; this is a bound, not an
#: estimate, and reaching it means task 0 died.
LIST_WAIT_S = 4 * 3600


def _load_or_list_keys(args, adapter) -> List[str]:
    path = args.record_list or os.path.join(args.out_dir, "record_list.txt")
    if (args.task and args.task[0] != 0 and not os.path.isfile(path)
            and not args.refresh_list):
        # One task lists; the rest wait for it. Every task walking the same
        # million-file tree at once is N times the metadata load for one
        # answer. Task 0 of the array writes it -- or run --list-only first.
        print(f"  waiting for task 0 to write {path} ...", flush=True)
        deadline = time.time() + LIST_WAIT_S
        while not os.path.isfile(path):
            if time.time() > deadline:
                raise SystemExit(
                    f"{path} did not appear in {LIST_WAIT_S // 3600} h. Task 0 "
                    f"lists the corpus; its log is the one to read. Or run "
                    f"with --list-only once, then submit the array.")
            time.sleep(20)
    if os.path.isfile(path) and not args.refresh_list:
        with open(path) as f:
            keys = [ln.rstrip("\n") for ln in f if ln.strip()]
        print(f"  {len(keys):,} record(s) from {path} "
              f"(--refresh-list to re-scan)", flush=True)
        return keys
    t0 = time.time()
    print(f"  listing records under {args.root} ...", flush=True)
    keys = adapter.list_keys(args.root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Every task of an array may race to write this; the content is the same
    # sorted listing, so whichever rename lands last is correct.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(keys) + ("\n" if keys else ""))
    os.replace(tmp, path)
    print(f"  {len(keys):,} record(s) in {time.time() - t0:.0f}s -> {path}",
          flush=True)
    return keys


def _spread(keys: List[str], n: Optional[int]) -> List[str]:
    """``n`` keys evenly over the list, not the first ``n``."""
    if not n or n >= len(keys):
        return keys
    idx = np.linspace(0, len(keys) - 1, num=n).astype(int)
    return [keys[i] for i in sorted(set(idx.tolist()))]


def assemble(args, dataset_id: str, unit_ids: List[str], cfg) -> int:
    """This task's part files -> manifests, subject lists, statistics."""
    suffix = f".{args.task[0]:04d}" if args.task else ""
    rows = {"train": [], "val": []}
    subjects = {"train": set(), "val": set()}
    failures: List[Dict] = []
    agg = {"records_read": 0, "records_with_windows": 0, "records_failed": 0,
           "records_shorter_than_window": 0, "windows_candidate": 0, "windows_kept": 0, "derived_limb_leads": 0,
           "qc_dropped": {r: 0 for r in QC_REASONS}, "source_rates": {}}
    missing = []
    for uid in unit_ids:
        p = os.path.join(args.out_dir, "parts", f"unit_{uid}.json")
        if not os.path.isfile(p):
            missing.append(uid)
            continue
        with open(p) as f:
            part = json.load(f)
        for side in ("train", "val"):
            rows[side].extend(part["rows"][side])
            subjects[side].update(part["subjects"][side])
        failures.extend(part["failures"])
        s = part["stats"]
        for k in ("records_read", "records_with_windows", "records_failed",
                  "records_shorter_than_window", "windows_candidate",
                  "windows_kept", "derived_limb_leads"):
            agg[k] += int(s.get(k, 0))
        for r in QC_REASONS:
            agg["qc_dropped"][r] += int(s["qc_dropped"].get(r, 0))
        for rate, c in s.get("source_rates", {}).items():
            agg["source_rates"][rate] = agg["source_rates"].get(rate, 0) + c
    leak = subjects["train"] & subjects["val"]
    if leak:
        print(f"ERROR: {len(leak)} subject(s) on both sides, e.g. "
              f"{sorted(leak)[:5]}. The split is by subject hash and this "
              f"cannot happen unless units were written under different "
              f"--val-fraction/--split-seed.", file=sys.stderr)
        return 1
    for side in ("train", "val"):
        with open(os.path.join(args.out_dir,
                               f"manifest_{side}{suffix}.jsonl"), "w") as f:
            for r in rows[side]:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir,
                               f"subjects_{side}{suffix}.txt"), "w") as f:
            f.write("\n".join(sorted(subjects[side])) +
                    ("\n" if subjects[side] else ""))
    with open(os.path.join(args.out_dir,
                           f"preprocessing_failures{suffix}.jsonl"), "w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")
    mod = MODALITIES[args.modality]
    spec = mod.datasets[dataset_id]
    stats = {
        "dataset_id": dataset_id, "route_id": spec.route_id, **agg,
        "n_units": len(unit_ids), "n_units_missing": len(missing),
        "n_subjects_train": len(subjects["train"]),
        "n_subjects_val": len(subjects["val"]),
        "n_windows_train": sum(r["n_windows"] for r in rows["train"]),
        "n_windows_val": sum(r["n_windows"] for r in rows["val"]),
        "record_is_subject": not mod.adapters[dataset_id].has_subject_ids,
        "preprocess_config": cfg.provenance({"dataset_id": dataset_id}),
        "synthetic": bool(args.smoke_test),
        "task": list(args.task) if args.task else None,
    }
    with open(os.path.join(args.out_dir,
                           f"dataset_statistics{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n{dataset_id}: {agg['records_read']:,} records read, "
          f"{agg['records_failed']:,} failed, {agg['windows_kept']:,} windows "
          f"kept of {agg['windows_candidate']:,} "
          f"(QC dropped {agg['qc_dropped']})")
    print(f"  train {stats['n_windows_train']:,} windows / "
          f"{stats['n_subjects_train']:,} subjects;  val "
          f"{stats['n_windows_val']:,} / {stats['n_subjects_val']:,}")
    if agg["records_shorter_than_window"]:
        print(f"  {agg['records_shorter_than_window']:,} record(s) were shorter "
              f"than one {cfg.window_seconds:g} s window and gave none")
    if agg["derived_limb_leads"]:
        print(f"  {agg['derived_limb_leads']:,} record(s) had III/aVR/aVL/aVF "
              f"derived from I and II")
    if missing:
        print(f"  {len(missing)} unit(s) of this task have no part file yet -- "
              f"re-run to finish them; their manifests are not included",
              flush=True)
        return 2
    print(f"  manifests, subject lists and statistics in {args.out_dir}")
    return 0


def main(argv=None, modality: str = "ecg") -> int:
    mod = MODALITIES[modality]
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default=None,
                   help="the raw download: a directory, or a PhysioNet .zip "
                        "for the WFDB corpora")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", type=_task_spec, default=None, metavar="I/N")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tmp-dir", default=None,
                   help="where archive members are inflated while read "
                        "(default <out-dir>/tmp)")
    p.add_argument("--records-per-unit", type=int, default=None,
                   help="keys per unit of work (and per shard pair). "
                        "Default 1000 for 12-lead corpora, 64 for Icentia11k "
                        "segments, 4 for PulseDB subject files. Changing it "
                        "re-cuts every unit, so a resumed run must keep it.")
    p.add_argument("--windows-per-shard", type=int, default=None,
                   help=argparse.SUPPRESS)        # smoke only: small units
    p.add_argument("--max-records", type=int, default=None,
                   help="use this many records, spread evenly over the corpus")
    p.add_argument("--max-windows-per-record", type=int, default=None,
                   help="cap per record (default: the registry's; 16 for "
                        "Icentia11k, none otherwise). 0 disables the cap.")
    p.add_argument("--record-list", default=None,
                   help="cached listing (default <out-dir>/record_list.txt)")
    p.add_argument("--refresh-list", action="store_true")
    p.add_argument("--list-only", action="store_true",
                   help="write <out-dir>/record_list.txt and exit")
    p.add_argument("--mains-hz", type=float, default=None,
                   help="override the registry's mains frequency")
    p.add_argument("--no-notch", action="store_true")
    p.add_argument("--highpass-hz", type=float, default=None,
                   help="default: the modality's (0.5 Hz ECG, 20 Hz EMG)")
    p.add_argument("--normalization", default="window_shared",
                   choices=["window_shared", "window_per_lead", "none"])
    p.add_argument("--window-seconds", type=float, default=None,
                   help="default: the route's (physiowave.ecg_c1.routes."
                        "WINDOW_SECONDS); anything else is refused")
    p.add_argument("--stride-seconds", type=float, default=None)
    p.add_argument("--medalcare-variant", default="noise",
                   choices=["noise", "raw", "filtered"],
                   help="which of MedalCare-XL's three renderings of each "
                        "signal to take (one, so no heart is counted thrice)")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--inspect", type=int, default=None, metavar="N")
    p.add_argument("--force", action="store_true",
                   help="redo units that already have a part file")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-records", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    args.modality = modality

    dataset_id = args.dataset
    if dataset_id in mod.downstream_only:
        print(f"ERROR: {dataset_id} is a downstream evaluation set and is never "
              f"pretrained on; a pretrained encoder that has seen it makes every "
              f"number reported on it a report on training data.",
              file=sys.stderr)
        return 1
    if dataset_id not in mod.datasets:
        print(f"ERROR: unknown dataset {dataset_id!r}; one of "
              f"{', '.join(mod.datasets)}", file=sys.stderr)
        return 1
    spec = mod.datasets[dataset_id]
    route = mod.routes[spec.route_id]
    if args.window_seconds is None:
        args.window_seconds = route.window_seconds
    # Both, so it reaches the workers however they start: a forked worker
    # inherits the class attribute, a spawned one re-imports and reads the
    # environment.
    os.environ["PW_MEDALCARE_VARIANT"] = args.medalcare_variant
    MedalCareXL.VARIANT = args.medalcare_variant
    if abs(args.window_seconds * route.sampling_rate
           - route.window_samples) > 1e-6:
        print(f"ERROR: --window-seconds {args.window_seconds} does not match "
              f"route {route.route_id}'s {route.window_seconds} s window; the "
              f"trainer would refuse every shard.", file=sys.stderr)
        return 1

    cap = (args.max_windows_per_record if args.max_windows_per_record is not None
           else spec.default_max_windows_per_record)
    settings = dict(mod.config_defaults)
    if args.highpass_hz is not None:
        settings["highpass_hz"] = args.highpass_hz
    cfg = ECGPreprocessConfig(**{
        **settings, "normalization": args.normalization,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "val_fraction": args.val_fraction, "split_seed": args.split_seed,
        "max_windows_per_record": cap or None})
    mains = None if args.no_notch else (args.mains_hz if args.mains_hz
                                        else spec.mains_hz)

    os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)

    per_unit = (args.records_per_unit or mod.adapters[dataset_id].records_per_unit
                or (64 if route.n_channels == 1 else 1000))
    # Scratch for archive members inflated while they are read (PulseDB: up
    # to ~3 GB per subject per worker). On the corpus's filesystem, not the
    # node's /tmp, which on a compute node may not hold sixteen of them. The
    # environment carries it to spawned workers.
    os.environ["PW_ECG_TMP"] = os.path.abspath(
        args.tmp_dir or os.path.join(args.out_dir, "tmp"))
    if args.smoke_test:
        keys = [f"smoke_{i:05d}" for i in range(args.smoke_records)]
        per_unit = args.windows_per_shard or per_unit
        args.root = "<synthetic>"
    else:
        if not args.root or not os.path.exists(args.root):
            print(f"ERROR: --root {args.root!r} does not exist. It is the raw "
                  f"download; {mod.download_script} puts {dataset_id} under "
                  f"${mod.root_var}/{spec.raw_dir}/raw.", file=sys.stderr)
            return 1
        keys = _load_or_list_keys(args, mod.adapters[dataset_id])
        if not keys:
            print(f"ERROR: no {dataset_id} records under {args.root}.",
                  file=sys.stderr)
            return 1
        if args.list_only:
            return 0
        keys = _spread(keys, args.max_records)

    if args.inspect is not None:
        if args.smoke_test:
            print("--inspect reads real files; there are none in a smoke run.")
            return 1
        return inspect(dataset_id, args.root, keys, args.inspect, cfg, mains,
                       modality)

    units = [keys[i:i + per_unit] for i in range(0, len(keys), per_unit)]
    unit_ids = [_unit_id(i) for i in range(len(units))]
    mine = [i for i in range(len(units))
            if not args.task or i % args.task[1] == args.task[0]]

    todo = []
    for i in mine:
        pp = os.path.join(args.out_dir, "parts", f"unit_{unit_ids[i]}.json")
        if os.path.isfile(pp) and not args.force:
            with open(pp) as f:
                if json.load(f).get("keys_sha") == _keys_sha(units[i]):
                    continue
            print(f"  unit {unit_ids[i]} was written for different records "
                  f"(the listing or --records-per-unit changed); redoing it",
                  flush=True)
        todo.append(i)

    print(f"{dataset_id} -> {route.route_id}: {len(keys):,} records in "
          f"{len(units)} unit(s) of {per_unit}; this task owns {len(mine)}, "
          f"{len(todo)} to do; mains {mains or 'none'}; window cap "
          f"{cfg.max_windows_per_record or 'none'}; {args.jobs} worker(s)",
          flush=True)

    payloads = [(dataset_id, args.root, unit_ids[i], units[i], cfg, mains,
                 os.path.abspath(args.out_dir), bool(args.smoke_test),
                 args.seed + i, modality) for i in todo]
    started = time.time()
    done = 0

    def report(res):
        nonlocal done
        done += 1
        if done % max(1, len(payloads) // 50) == 0 or done == len(payloads):
            rate = done / max(1e-9, time.time() - started)
            print(f"  {done}/{len(payloads)} units  {rate * 60:.1f}/min  "
                  f"ETA {(len(payloads) - done) / max(1e-9, rate) / 60:.0f} min"
                  f"  (last: {res['stats']['windows_kept']} windows, "
                  f"{res['n_failures']} failed)", flush=True)

    if args.jobs > 1 and len(payloads) > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_unit, pl): pl[2] for pl in payloads}
            for fut in cf.as_completed(futures):
                try:
                    report(fut.result())
                except Exception as exc:                       # noqa: BLE001
                    # A worker that died loses its unit, not the run: no part
                    # file was written, so a re-run redoes exactly that unit.
                    print(f"  unit {futures[fut]} died: {exc}", flush=True)
    else:
        for pl in payloads:
            report(process_unit(pl))

    return assemble(args, dataset_id, [unit_ids[i] for i in mine], cfg)


MODALITIES["ecg"] = Modality(
    name="ecg", datasets=PRETRAIN_DATASETS, routes=ROUTES,
    downstream_only=DOWNSTREAM_ONLY, adapters=ADAPTERS,
    smoke_records=smoke_records, process_record=process_ecg_record,
    root_var="ECG_ROOT",
    download_script="scripts/download_ecg_pretrain_corpora.sh",
    amplitude_hint="A median lead peak-to-peak far from ~0.5-3 mV means the "
                   "unit is wrong, and every QC threshold with it.")


if __name__ == "__main__":
    sys.exit(main())
