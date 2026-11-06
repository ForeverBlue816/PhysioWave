#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One sEMG corpus -> HDF5 shards and manifests for sEMG C1 pretraining.

    python EMG/preprocess_emg_corpus.py --dataset emg2pose \\
        --root $EMG_ROOT/emg2pose/raw --out-dir $CORPUS/emg2pose --inspect 20
    python EMG/preprocess_emg_corpus.py --dataset emg2pose \\
        --root $EMG_ROOT/emg2pose/raw --out-dir $CORPUS/emg2pose --jobs 16
    python scripts/build_eeg_c1_manifest.py --modality emg --corpus-root $CORPUS

THE DRIVER IS ECG/preprocess_ecg_corpus.py's: units of records, a part file per
unit written last and atomically, ``--task I/N`` arrays, resume by part file,
subject-hash splits, ``--inspect``. This file supplies only what is sEMG:

* the readers, one per corpus (below);
* the signal settings -- a 20 Hz high-pass, which is where sEMG work cuts
  movement artefact and electrode drift, the mains notch with three
  harmonics, and QC thresholds in the units of sEMG;
* a synthetic sEMG generator for ``--smoke-test``.

The pipeline after reading is physiowave.ecg_c1.preprocess, unchanged:
to mV -> DC removal -> notch -> high-pass -> polyphase resample to 2000 Hz ->
channels onto the route's slots by name -> 1 s windows -> QC -> one scale per
window across the channels, so a window keeps which electrodes were active
relative to which.
"""

from __future__ import annotations

import io
import os
import re
import sys
import zipfile
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "ECG")):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess_ecg_corpus as base                                # noqa: E402
from preprocess_ecg_corpus import (Adapter, ECGRecord, Modality,    # noqa: E402
                                   record_stem)
from physiowave.ecg_c1.preprocess import (ECGPreprocessError,       # noqa: E402
                                          process_ecg_record)
from physiowave.emg_c1.electrodes import (BAND_16,                  # noqa: E402
                                          CEMHSEY_5X13_BY_CHANNEL,
                                          CEMHSEY_8X8_BY_CHANNEL,
                                          electrode_ids_for,
                                          normalize_electrode)
from physiowave.emg_c1.routes import (DOWNSTREAM_ONLY,              # noqa: E402
                                      PRETRAIN_DATASETS, ROUTES,
                                      WINDOW_SECONDS)

#: The settings sEMG needs that ECG's defaults do not have. Thresholds in mV:
#: a resting surface channel is a few uV RMS, a maximal contraction a few mV.
EMG_SETTINGS = {
    "highpass_hz": 20.0,
    "notch_harmonics": 3,
    "flat_lead_std_mv": 0.0005,      # 0.5 uV: an open electrode, not rest
    "max_abs_mv": 25.0,              # no surface EMG reaches this
    "derive_limb_leads": False,
    "val_fraction": 0.05,
    # The config class's own default is ECG's 10 s; the driver sets the
    # route's window anyway, and this keeps the settings true on their own.
    "window_seconds": WINDOW_SECONDS,
}


def process_emg_record(data, lead_names, fs, unit, route, mains_hz, cfg,
                       record_key="", slots=None):
    """physiowave.ecg_c1.preprocess.process_ecg_record with sEMG electrodes."""
    return process_ecg_record(data, lead_names, fs, unit, route, mains_hz, cfg,
                              record_key=record_key, slots=slots,
                              normalize=normalize_electrode,
                              ids_for=electrode_ids_for)


# =========================================================================== #
# Synthetic sEMG, for --smoke-test
# =========================================================================== #

def synthetic_semg(n_channels: int, fs: float, seconds: float,
                   rng: np.random.Generator) -> np.ndarray:
    """``[C, T]`` in mV: band-limited bursts under slow activation envelopes.

    Each channel is a weighted mix of a few latent "muscles", each a 20-450 Hz
    noise process gated by a smooth on/off envelope -- enough to give the
    windows sEMG's spectrum, its bursts and its cross-channel correlation.
    """
    from scipy.signal import butter, sosfiltfilt
    T = int(round(seconds * fs))
    n_src = 4
    sos = butter(4, [20.0 / (fs / 2), min(450.0, 0.45 * fs) / (fs / 2)],
                 btype="band", output="sos")
    src = sosfiltfilt(sos, rng.normal(size=(n_src, T)), axis=-1)
    t = np.arange(T) / fs
    env = np.stack([0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t
                                       + rng.uniform(0, 6.28)) for _ in range(n_src)])
    mix = rng.uniform(0, 1, size=(n_channels, n_src))
    x = mix @ (src * env)
    x = 0.3 * x / (x.std() + 1e-9)                                  # ~0.3 mV
    return x + rng.normal(0, 0.002, size=x.shape)


def emg_smoke_records(dataset_id: str, n: int, seed: int) -> List[ECGRecord]:
    spec = PRETRAIN_DATASETS[dataset_id]
    rng = np.random.default_rng(seed)
    fs = float(spec.native_rate or spec.route.sampling_rate)
    slots = list(spec.lead_slots)
    out = []
    for r in range(n):
        x = synthetic_semg(len(slots), fs, 6.0, rng)
        # The unit's seed in the subject id, so units do not reuse subjects
        # and a small smoke corpus still lands on both sides of the split.
        out.append(ECGRecord(f"smoke_{seed}_{r:05d}",
                             f"smoke_{dataset_id}_{seed}_s{r // 2:03d}",
                             x, slots, fs, "mV", {"synthetic": True}))
    return out


# =========================================================================== #
# Readers. Each: list_keys(root) -> sorted keys; read_records(root, key, ctx)
# =========================================================================== #

def _member_bytes(path: str, offset: int, size: int, ctx: Dict) -> bytes:
    """``size`` bytes at ``offset`` of an archive, through one open handle."""
    f = ctx.get(("fh", path))
    if f is None:
        f = open(path, "rb")
        ctx[("fh", path)] = f
    f.seek(offset)
    data = f.read(size)
    if len(data) != size:
        raise ECGPreprocessError(
            f"{path}: {len(data)} of {size} bytes at {offset} -- the archive "
            f"is shorter than when it was listed (still downloading?)")
    return data


def _walk(root: str, suffix: str) -> List[str]:
    out = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for fn in files:
            if fn.endswith(suffix) and not fn.startswith("."):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(out)


def _attr(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


class EMG2Pose(Adapter):
    """One HDF5 file per wrist per stage: group ``emg2pose``, a compound
    ``timeseries`` whose ``emg`` field is ``[N, 16]`` float32 in uV at 2000 Hz.

    Read straight out of ``emg2pose_dataset.tar``: the tar is uncompressed,
    so a file is a contiguous run of bytes and the listing records where each
    one starts. Unpacking 431 GiB to read it once would double the footprint.
    A directory of extracted files works too.
    """

    records_per_unit = 200           # ~1 min each

    def list_keys(self, root):
        if os.path.isfile(root):
            import tarfile
            keys = []
            with tarfile.open(root, "r:") as tf:
                for m in tf:
                    # The tar was written on a Mac: skip any ._ resource
                    # forks beside the files, which are not HDF5.
                    if (m.isfile() and m.name.endswith(".hdf5")
                            and not os.path.basename(m.name).startswith(".")):
                        keys.append(f"{m.name}|{m.offset_data}:{m.size}")
            return sorted(keys)
        return _walk(root, ".hdf5")

    def read(self, root, key, ctx):
        import h5py
        name, _, spec = key.partition("|")
        if spec:
            off, size = (int(x) for x in spec.split(":"))
            src = io.BytesIO(_member_bytes(root, off, size, ctx))
        else:
            src = os.path.join(root, name)
        with h5py.File(src, "r") as h:
            g = h["emg2pose"]
            emg = np.asarray(g["timeseries"].fields("emg")[:], dtype=np.float64)
            fs = float(g.attrs.get("sample_rate", 2000.0))
            user = _attr(g.attrs["user"])
            side = _attr(g.attrs.get("side", ""))
        return ECGRecord(os.path.splitext(os.path.basename(name))[0],
                         f"user_{user}", emg.T, list(BAND_16), fs, "uV",
                         {"side": side})


class EMG2Qwerty(Adapter):
    """One HDF5 file per session: ``emg2qwerty/timeseries`` with ``emg_left``
    and ``emg_right``, each ``[N, 16]`` float32 in uV at 2000 Hz. Each wrist
    is its own record on W16_2000 -- the same band, the same electrode names.
    """

    # Sessions; two records each. Four, not eight: a session runs 10-25 min,
    # so a unit holds up to ~200 MB of windows per wrist on top of the raw
    # float64 signal being filtered, and fifteen workers share one node's RAM.
    records_per_unit = 4

    def list_keys(self, root):
        return _walk(root, ".hdf5")

    def read_records(self, root, key, ctx):
        import h5py
        with h5py.File(os.path.join(root, key), "r") as h:
            g = h["emg2qwerty"]
            ts = g["timeseries"]
            fs = float(_attr(g.attrs.get("daq_sample_rate", "2000.0")))
            user = _attr(g.attrs["user"])
            stem = os.path.splitext(os.path.basename(key))[0]
            out = []
            for side in ("left", "right"):
                emg = np.asarray(ts.fields(f"emg_{side}")[:], dtype=np.float64)
                out.append(ECGRecord(f"{stem}#{side}", f"user_{user}", emg.T,
                                     list(BAND_16), fs, "uV", {"side": side}))
        return out


class Hyser(Adapter):
    """PhysioNet hd-semg 2.0.0, the ``*_raw_*`` records only -- the
    ``preprocess`` ones are the same signals band-passed by the authors.

    256 channels at 2048 Hz named ``XX-i-j``: grid XX (ED, EP, FD, FP) row i
    column j. Each grid is its own 64-channel record. The headers say V and
    the values are mV -- a contracting forearm at 0.4 V peak-to-peak is not a
    surface signal, at 0.4 mV it is -- so the unit is set, not read.
    """

    records_per_unit = 64            # files; four records each
    _NAME = re.compile(r"^(ED|EP|FD|FP)-(\d)-(\d)$")
    _SUBJ = re.compile(r"(subject\d+)_session\d+")

    def list_keys(self, root):
        return [k[:-4] for k in _walk(root, ".hea")
                if "_raw_" in os.path.basename(k)]

    def read_records(self, root, key, ctx):
        import wfdb
        rec = wfdb.rdrecord(os.path.join(root, key))
        sig = np.asarray(rec.p_signal, dtype=np.float64).T
        m = self._SUBJ.search(key)
        subject = m.group(1) if m else key.split("/")[0]
        rows: Dict[str, List[int]] = {}
        names: Dict[str, List[str]] = {}
        for i, n in enumerate(rec.sig_name):
            mm = self._NAME.match(n)
            if not mm:
                continue
            g, r, c = mm.groups()
            rows.setdefault(g, []).append(i)
            names.setdefault(g, []).append(f"hyser_r{r}c{c}")
        if not rows:
            raise ECGPreprocessError(f"no XX-i-j channels in {rec.sig_name[:4]}")
        return [ECGRecord(f"{key}#{g}", subject, sig[rows[g]], names[g],
                          float(rec.fs), "mV", {"grid": g,
                                                "source_lead_name": g})
                for g in sorted(rows)]


class CEMHSEY(Adapter):
    """Zenodo 15077957 + 15070187: ``data_sEMG``, ``[320, N]`` mV at 2048 Hz.

    Rows 1-192 are three 8x8 grids, rows 193-320 two 5x13 grids (the paper,
    Fig. 2). ``grids`` picks which this corpus reads; the electrode names come
    from the dataset's own channel maps (physiowave.emg_c1.electrodes).

    GRASP's zips are deflate and are read in place, member by member. The
    GESTURE zips are deflate64, which Python's zipfile cannot inflate; the
    download script unpacks them into a directory named after the zip, and a
    zip with such a directory beside it is skipped in favour of the directory.

    GRASP's S4 and GESTURE's S4 are different people (the parts recruited
    separately), so the subject is ``grasp_S4`` / ``gesture_S4``. Four trials
    the authors list as failed recordings are left out.
    """

    records_per_unit = 16
    FAILED = frozenset({"S4_Day1_Session1_Task1_Trial1.mat",       # GRASP
                        "S2_Day3_Trial3.mat", "S3_Day6_Trial2.mat",
                        "S3_Day10_Trial5.mat"})                    # GESTURE
    _GRASP = re.compile(r"^S(\d+)_Day\d+_Session\d+_Task\d+_Trial\d+\.mat$")
    _GESTURE = re.compile(r"^S(\d+)_Day\d+_Trial\d+\.mat$")

    def __init__(self, grids: Sequence[int], names: Sequence[str]):
        self.grids = tuple(grids)
        self.names = tuple(names)

    def list_keys(self, root):
        keys = []
        zips = sorted(f for f in os.listdir(root) if f.lower().endswith(".zip")
                      and not f.startswith("MyFunction"))
        for z in zips:
            if os.path.isdir(os.path.join(root, z[:-4])):
                continue                          # unpacked; walked below
            with zipfile.ZipFile(os.path.join(root, z)) as zf:
                infos = [i for i in zf.infolist()
                         if i.filename.endswith(".mat")]
            bad = [i for i in infos if i.compress_type not in
                   (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)]
            if bad:
                raise SystemExit(
                    f"{z} is compressed with method {bad[0].compress_type} "
                    f"(deflate64), which Python cannot read. Unpack it next "
                    f"to itself:\n  cd {root} && unzip -q {z} -d {z[:-4]}\n"
                    f"(bash scripts/download_emg_pretrain_corpora.sh cemhsey "
                    f"does this).")
            keys += [f"{z}/{i.filename}|{i.header_offset}:{i.compress_size}:"
                     f"{i.file_size}:{i.compress_type}" for i in infos]
        keys += [k for k in _walk(root, ".mat")
                 if not os.path.basename(k).startswith(".")]
        return sorted(k for k in keys if self._part(k)
                      and os.path.basename(record_stem(k)) not in self.FAILED)

    def _part(self, key: str):
        base = os.path.basename(record_stem(key))
        m = self._GRASP.match(base)
        if m:
            return f"grasp_S{int(m.group(1))}"
        m = self._GESTURE.match(base)
        if m:
            return f"gesture_S{int(m.group(1))}"
        return None

    @staticmethod
    def _load(src) -> np.ndarray:
        import scipy.io
        try:
            return np.asarray(scipy.io.loadmat(src, variable_names=["data_sEMG"])
                              ["data_sEMG"], dtype=np.float64)
        except NotImplementedError:              # MAT v7.3 is HDF5
            import h5py
            if hasattr(src, "seek"):
                src.seek(0)
            with h5py.File(src, "r") as h:
                return np.asarray(h["data_sEMG"], dtype=np.float64).T

    def read_records(self, root, key, ctx):
        stem, _, spec = key.partition("|")
        if spec:
            zname, member = stem.split("/", 1)
            off, csz, usz, meth = (int(x) for x in spec.split(":"))
            src = io.BytesIO(base._read_zip_member(os.path.join(root, zname),
                                                   off, csz, usz, meth, ctx))
        else:
            src = os.path.join(root, stem)
        x = self._load(src)
        if x.shape[0] != 320:
            if x.shape[1] == 320:
                x = x.T
            else:
                raise ECGPreprocessError(f"data_sEMG is {x.shape}, not 320 rows")
        subject = self._part(key)
        name = os.path.splitext(os.path.basename(stem))[0]
        return [ECGRecord(f"{subject}/{name}#g{g + 1}", subject,
                          x[g * 64:(g + 1) * 64], list(self.names), 2048.0,
                          "mV", {"grid": g + 1,
                                 "source_lead_name": f"grid{g + 1}"})
                for g in self.grids]


class PutEMG(Adapter):
    """putEMG's HDF5 files: a pandas ``frame_table`` whose ``values_block_3``
    holds EMG_1..24 (with trajectory, force or subject columns, which differ
    between the gesture and force files -- hence by name) as raw 12-bit ADC
    counts at 5120 Hz: 5 V over 4096 counts behind a gain of 200, so
    5/4096*1000/200 mV a count. EMG_1-8 is the ring by the elbow, 9-16 the
    middle one, 17-24 the one by the wrist.
    """

    records_per_unit = 8
    MV_PER_COUNT = 5.0 / 4096 * 1000.0 / 200.0
    FS = 5120.0
    _SUBJ = re.compile(r"-(\d{2})-")

    def list_keys(self, root):
        return [k for k in _walk(root, ".hdf5")
                if os.path.basename(k).startswith("emg_")]

    def read(self, root, key, ctx):
        import h5py
        import pickle
        with h5py.File(os.path.join(root, key), "r") as h:
            t = h["data/table"]
            cols = pickle.loads(bytes(t.attrs["values_block_3_kind"]),
                                encoding="latin1")
            pick = [cols.index(f"EMG_{k}") for k in range(1, 25)]
            v = t.fields("values_block_3")[:]
        x = np.asarray(v[:, pick], dtype=np.float64).T * self.MV_PER_COUNT
        m = self._SUBJ.search(os.path.basename(key))
        subject = f"putemg_{m.group(1)}" if m else os.path.basename(key)
        names = [f"putemg_r{(k - 1) // 8 + 1}e{(k - 1) % 8 + 1}"
                 for k in range(1, 25)]
        return ECGRecord(os.path.splitext(os.path.basename(key))[0], subject,
                         x, names, self.FS, "mV", {})


ADAPTERS: Dict[str, Adapter] = {
    "emg2pose": EMG2Pose(),
    "emg2qwerty": EMG2Qwerty(),
    "hyser": Hyser(),
    "cemhsey_8x8": CEMHSEY((0, 1, 2), CEMHSEY_8X8_BY_CHANNEL),
    "cemhsey_5x13": CEMHSEY((3, 4), CEMHSEY_5X13_BY_CHANNEL),
    "putemg": PutEMG(),
}
assert set(ADAPTERS) == set(PRETRAIN_DATASETS)


base.MODALITIES["emg"] = Modality(
    name="emg", datasets=PRETRAIN_DATASETS, routes=ROUTES,
    downstream_only=DOWNSTREAM_ONLY, adapters=ADAPTERS,
    smoke_records=emg_smoke_records, process_record=process_emg_record,
    root_var="EMG_ROOT",
    download_script="scripts/download_emg_pretrain_corpora.sh",
    config_defaults=EMG_SETTINGS,
    amplitude_hint="A median channel peak-to-peak far from ~0.05-5 mV means "
                   "the unit is wrong, and every QC threshold with it.")


if __name__ == "__main__":
    sys.exit(base.main(modality="emg"))
