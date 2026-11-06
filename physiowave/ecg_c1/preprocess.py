"""
The preprocessing every ECG corpus goes through.

Adapters (ECG/preprocess_ecg_corpus.py) differ only in how a record is READ.
Everything after that is here and identical for every corpus, so a difference
between two datasets in training is a difference between the datasets.

    to mV -> DC removal -> notch (mains, if any) -> 0.5 Hz high-pass
          -> polyphase resample to the route's rate -> leads onto slots
          -> 10 s windows -> window QC -> normalise

What is ECG-specific, and why:

* **Units are mV,** the unit every ECG convention is stated in, so the QC
  thresholds below mean the same thing on every corpus. The normalisation is
  scale-free, so the unit only matters to the QC -- which is exactly where a
  unit error would otherwise pass silently.

* **The high-pass is second-order sections with a long mirrored pad.** 0.5 Hz
  at 500 Hz is a normalised cutoff of 0.002, where a 4th-order Butterworth in
  (b, a) form is badly conditioned, and filtfilt's default pad (15 samples) is
  one thirtieth of a second against a filter that rings for seconds. On a
  10 s record the edge transient would be a fifth of the window.

  The pad is EVEN (mirrored), not odd. Odd extension reflects through the
  last sample, so a record that ends on an R wave gets a pad whose level is
  the R-wave amplitude -- a step the filter then rings on, back into the
  record. Measured against the same filter run on 70 s of context, over 40
  random 10 s crops of synthetic 12-lead ECG with baseline wander: odd padding
  left a worst-case edge error of 1.06 mV (mean relative error 12.6%), even
  padding 0.12 mV (6.0%). Linear detrending first changed neither.

* **No low-pass or band-pass.** The polyphase resampler's anti-alias filter is
  the only one. A 40 Hz low-pass is a visualisation convention; it removes the
  QRS energy that notching and fragmentation live in.

* **The normalisation keeps the leads' relative amplitudes.** The default
  (``window_shared``) removes each lead's mean and divides the whole window by
  ONE standard deviation, taken over all its leads together. Per-lead z-scoring
  -- what the EEG corpus does -- would erase the ratios between leads, and in
  ECG those ratios ARE the diagnosis: the frontal axis is the ratio of I to aVF,
  low voltage and LVH are amplitude criteria, a lead's inversion relative to
  another is the finding. ``window_per_lead`` is available for comparison, and
  ``none`` keeps mV.

* **Missing limb leads are derived, not padded.** III, aVR, aVL and aVF are
  exact combinations of I and II; a file that stores only I, II and V1-V6 (the
  eight independent leads) has all twelve.

* **A window is dropped, not repaired.** A window containing a non-finite
  sample, a flat lead (disconnected electrode) or an implausible amplitude
  (a unit error, or saturation) is left out and counted by reason. A 10 s
  record is exactly one window, so that is a dropped record; the counts say
  how many, per reason, per corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..eeg_c1.preprocess import resample_to, window_signal
from .leads import LEADS_12, lead_ids_for, normalize_lead


@dataclass
class ECGPreprocessConfig:
    """Everything that changes the numbers, recorded into every shard."""

    highpass_hz: float = 0.5
    notch_harmonics: int = 2
    notch_quality: float = 30.0
    normalization: str = "window_shared"   # | window_per_lead | none
    clip_sigma: float = 20.0
    eps: float = 1e-6
    window_seconds: float = 10.0
    stride_seconds: Optional[float] = None  # None -> no overlap
    val_fraction: float = 0.05
    split_seed: int = 42
    #: A lead whose std over a window is below this (mV) is not recording.
    #: 5 uV is far below any real ECG lead after a 0.5 Hz high-pass.
    flat_lead_std_mv: float = 0.005
    #: No surface ECG reaches this; a window that does is a unit error or a
    #: saturated amplifier.
    max_abs_mv: float = 25.0
    #: For the continuous corpora. None keeps every window.
    max_windows_per_record: Optional[int] = None
    derive_limb_leads: bool = True

    def stride(self) -> float:
        return (self.window_seconds if self.stride_seconds is None
                else self.stride_seconds)

    def provenance(self, extra: Optional[Dict] = None) -> Dict:
        d = asdict(self)
        d["pipeline"] = ("units_mV -> dc_remove -> notch -> highpass(sos, "
                         "even pad) -> resample_poly -> lead_slots(+derived "
                         "limb leads) -> window -> qc -> normalise")
        d["resampler"] = "scipy.signal.resample_poly (polyphase, anti-aliased)"
        d["band_pass"] = "none (0.5 Hz high-pass + resampling anti-alias only)"
        if extra:
            d.update(extra)
        d["config_sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in sorted(d.items())},
                       default=str).encode()).hexdigest()
        return d


class ECGPreprocessError(RuntimeError):
    """A record that cannot be processed. Recorded, never silently skipped."""


# --------------------------------------------------------------------------- #
# Signal steps
# --------------------------------------------------------------------------- #

_UNIT_TO_MV = {"mv": 1.0, "millivolt": 1.0, "millivolts": 1.0,
               "uv": 1e-3, "µv": 1e-3, "microvolt": 1e-3, "microvolts": 1e-3,
               "v": 1e3, "volt": 1e3, "volts": 1e3,
               # CODE's documented unit: tracings are in units of 1e-4 V.
               "1e-4v": 0.1, "100uv": 0.1}


def to_millivolts(x: np.ndarray, unit: str) -> np.ndarray:
    scale = _UNIT_TO_MV.get(str(unit).strip().lower().replace(" ", ""))
    if scale is None:
        raise ECGPreprocessError(
            f"unknown unit {unit!r}; say mV, uV, V or 1e-4V rather than let a "
            f"factor of a thousand pass as an amplitude difference")
    return np.asarray(x, dtype=np.float64) * scale


def notch(x: np.ndarray, fs: float, freq: Optional[float], harmonics: int,
          q: float) -> np.ndarray:
    """IIR notch at the mains frequency and its harmonics below Nyquist."""
    if not freq:
        return x
    from scipy.signal import filtfilt, iirnotch
    y = x
    for k in range(1, max(1, harmonics) + 1):
        f0 = freq * k
        if f0 >= fs / 2 - 1:
            break
        b, a = iirnotch(f0, q, fs)
        y = filtfilt(b, a, y, axis=-1, padlen=min(y.shape[-1] - 1, int(fs)))
    return y


def highpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth high-pass in second-order sections.

    Padded by mirroring over ``fs / cutoff`` samples (one period of the
    cutoff, 2 s at 0.5 Hz) or as much of the record as there is, so the edge
    transient lands in the pad and not in the window. Mirrored rather than
    odd: see the module docstring for the measurement.
    """
    if not cutoff:
        return x
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff / (fs / 2.0), btype="highpass", output="sos")
    padlen = int(min(x.shape[-1] - 1, round(fs / cutoff)))
    return sosfiltfilt(sos, x, axis=-1, padtype="even", padlen=max(0, padlen))


# --------------------------------------------------------------------------- #
# Leads -> slots
# --------------------------------------------------------------------------- #

@dataclass
class LeadMapping:
    placed: np.ndarray                 # [n_slots, T]
    valid: np.ndarray                  # [n_slots] bool
    derived: List[str]                 # slots computed from I and II
    unmatched_sources: List[str]       # recorded leads with no slot
    empty_slots: List[str]
    source_rows: List[int]             # which input rows were placed


def derive_limb_leads(leads: Dict[str, np.ndarray]) -> List[str]:
    """Fill III, aVR, aVL, aVF from I and II where they are missing. In place.

    Einthoven and Goldberger, exactly: III = II - I, aVR = -(I + II)/2,
    aVL = I - II/2, aVF = II - I/2. Nothing is estimated -- these are the
    definitions of the leads, and a recorder that stores eight leads is
    relying on them.
    """
    if "I" not in leads or "II" not in leads:
        return []
    i, ii = leads["I"], leads["II"]
    formulas = {"III": lambda: ii - i, "aVR": lambda: -(i + ii) / 2.0,
                "aVL": lambda: i - ii / 2.0, "aVF": lambda: ii - i / 2.0}
    made = []
    for name, f in formulas.items():
        if name not in leads:
            leads[name] = f()
            made.append(name)
    return made


def map_leads(x: np.ndarray, names: Sequence[str], slots: Sequence[str],
              derive: bool = True, normalize=None) -> LeadMapping:
    """``[C_src, T]`` -> ``[n_slots, T]``, placed by lead NAME, never position.

    ``normalize`` maps a file's spelling onto the slot names -- ECG's lead
    aliases by default; physiowave.emg_c1 passes its electrode names.
    """
    normalize = normalize or normalize_lead
    by_name: Dict[str, np.ndarray] = {}
    used: List[int] = []
    unmatched: List[str] = []
    slot_set = set(slots)
    for row, raw in enumerate(names):
        name = normalize(raw)
        if name not in slot_set:
            unmatched.append(str(raw))
            continue
        if name in by_name:
            unmatched.append(f"{raw} (duplicate of {name})")
            continue
        by_name[name] = np.asarray(x[row])
        used.append(row)
    derived = (derive_limb_leads(by_name)
               if derive and set(LEADS_12) <= slot_set else [])
    T = x.shape[-1]
    placed = np.zeros((len(slots), T), dtype=np.float64)
    valid = np.zeros(len(slots), dtype=bool)
    for k, s in enumerate(slots):
        if s in by_name:
            placed[k] = by_name[s]
            valid[k] = True
    empty = [s for k, s in enumerate(slots) if not valid[k]]
    return LeadMapping(placed, valid, derived, unmatched, empty, used)


# --------------------------------------------------------------------------- #
# Windows: QC and normalisation
# --------------------------------------------------------------------------- #

#: Why a window was dropped. Counted per corpus.
QC_REASONS = ("non_finite", "flat_lead", "amplitude")


def window_qc_mask(w: np.ndarray, valid: np.ndarray, bad: np.ndarray,
                   flat: np.ndarray, cfg: ECGPreprocessConfig
                   ) -> Tuple[np.ndarray, Dict[str, int]]:
    """``(keep [N] bool, {reason: count})``. ``w`` is filtered, in mV.

    ``bad`` and ``flat`` are ``[N]`` and come from the SOURCE signal over each
    window's span: whether it held a non-finite sample, and whether a placed
    lead was flat there. Flatness is judged before filtering because the
    filter itself un-flattens a dead stretch -- the high-pass and the
    resampler ring into it from the live signal either side. The reasons are
    checked in order and a window is counted once, under the first it fails.
    """
    v = np.asarray(valid, dtype=bool)
    sig = w[:, v, :]
    counts = {r: 0 for r in QC_REASONS}
    keep = ~bad
    counts["non_finite"] = int(bad.sum())
    flat = flat & keep
    counts["flat_lead"] = int(flat.sum())
    keep &= ~flat
    loud = (np.abs(sig).max(axis=(-1, -2)) > cfg.max_abs_mv) & keep
    counts["amplitude"] = int(loud.sum())
    keep &= ~loud
    return keep, counts


def normalise_windows(w: np.ndarray, valid: np.ndarray,
                      cfg: ECGPreprocessConfig) -> np.ndarray:
    """``[N, C, T]`` mV -> the stored representation. Invalid slots stay zero."""
    out = w.astype(np.float32, copy=True)
    v = np.asarray(valid, dtype=bool)
    mode = cfg.normalization
    sig = out[:, v, :]
    if mode == "none":
        pass
    elif mode == "window_shared":
        sig = sig - sig.mean(axis=-1, keepdims=True)
        # One scale for the whole window: the RMS of the de-meaned leads.
        sd = np.sqrt((sig ** 2).mean(axis=(-1, -2), keepdims=True))
        sig = sig / np.maximum(sd, cfg.eps)
    elif mode == "window_per_lead":
        mu = sig.mean(axis=-1, keepdims=True)
        sd = sig.std(axis=-1, keepdims=True)
        sig = (sig - mu) / np.maximum(sd, cfg.eps)
    else:
        raise ValueError(f"normalization must be window_shared, "
                         f"window_per_lead or none; got {mode!r}")
    if mode != "none" and cfg.clip_sigma and cfg.clip_sigma > 0:
        sig = np.clip(sig, -cfg.clip_sigma, cfg.clip_sigma)
    out[:, v, :] = sig
    out[:, ~v, :] = 0.0
    return out


def _choose_windows(n: int, cap: Optional[int], key: str) -> np.ndarray:
    """Which of ``n`` windows to keep: all, or ``cap`` of them, fixed by ``key``.

    Seeded by the record's identity, so a re-run -- or a resumed task --
    picks the same windows, and the choice is spread over the whole record
    rather than being its first minutes.
    """
    if cap is None or n <= cap:
        return np.arange(n)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


@dataclass
class ProcessedRecord:
    windows: np.ndarray                # [N, C, T] float32, normalised
    starts_seconds: np.ndarray         # [N]
    valid: np.ndarray                  # [C] bool
    channel_ids: List[int]
    derived: List[str]
    qc: Dict[str, int]
    n_candidate_windows: int


def process_ecg_record(data: np.ndarray, lead_names: Sequence[str],
                       fs: float, unit: str, route, mains_hz: Optional[float],
                       cfg: ECGPreprocessConfig,
                       record_key: str = "",
                       slots: Optional[Sequence[str]] = None,
                       normalize=None, ids_for=None) -> ProcessedRecord:
    """One record, start to finish. Raises ECGPreprocessError to be counted.

    ``slots`` are the corpus's leads on the route, in row order -- the route's
    own when not given. Two corpora can share a route's shape and record
    different leads (Icentia11k's chest patch and PulseDB's lead II are both
    one lead at 250 Hz), and the shard's channel ids are what tell them apart.
    """
    slots = tuple(slots) if slots else tuple(route.slots)
    if len(slots) != route.n_channels:
        raise ECGPreprocessError(
            f"{len(slots)} slots for {route.route_id}'s {route.n_channels} rows")
    x = to_millivolts(data, unit)
    if x.ndim != 2:
        raise ECGPreprocessError(f"expected [leads, samples], got {x.shape}")
    if x.shape[0] != len(lead_names):
        raise ECGPreprocessError(
            f"{x.shape[0]} signal rows but {len(lead_names)} lead names")

    # Non-finite samples are zeroed for filtering and REMEMBERED: any window
    # that overlaps one is dropped afterwards. Filtering through a NaN would
    # spread it over the whole lead, and interpolating over it would be
    # inventing signal.
    finite = np.isfinite(x)
    if not finite.any(axis=-1).all():
        dead = [str(lead_names[i]) for i in np.where(~finite.any(axis=-1))[0]]
        raise ECGPreprocessError(f"lead(s) with no finite sample: {dead}")
    bad_t = ~finite.all(axis=0)                            # [T_src]
    x = np.where(finite, x, 0.0)

    x = x - x.mean(axis=-1, keepdims=True)
    # Running sums of the SOURCE signal, for the per-window flatness check.
    src_sum = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, -1)], -1)
    src_sq = np.concatenate([np.zeros((x.shape[0], 1)),
                             np.cumsum(x * x, -1)], -1)
    x = notch(x, fs, mains_hz, cfg.notch_harmonics, cfg.notch_quality)
    x = highpass(x, fs, cfg.highpass_hz)
    x = resample_to(x, fs, route.sampling_rate)

    mapping = map_leads(x, lead_names, slots, cfg.derive_limb_leads,
                        normalize=normalize)
    if mapping.empty_slots:
        raise ECGPreprocessError(
            f"{route.route_id} needs {list(slots)}; missing "
            f"{mapping.empty_slots} (recorded: {list(lead_names)[:14]})")

    win = int(round(cfg.window_seconds * route.sampling_rate))
    stride = int(round(cfg.stride() * route.sampling_rate))
    windows, starts = window_signal(mapping.placed, win, stride)
    n_cand = int(windows.shape[0])
    if n_cand == 0:
        return ProcessedRecord(np.zeros((0, len(slots), win), np.float32),
                               np.zeros(0), mapping.valid, [], mapping.derived,
                               {r: 0 for r in QC_REASONS}, 0)

    pick = _choose_windows(n_cand, cfg.max_windows_per_record, record_key)
    windows, starts = windows[pick], starts[pick]

    # A window is bad if the source span it came from held a non-finite sample.
    ratio = fs / float(route.sampling_rate)
    csum = np.concatenate([[0], np.cumsum(bad_t.astype(np.int64))])
    lo = np.clip(np.floor(starts * ratio).astype(np.int64), 0, bad_t.size)
    hi = np.clip(np.ceil((starts + win) * ratio).astype(np.int64), 0, bad_t.size)
    bad = (csum[hi] - csum[lo]) > 0
    rows = np.asarray(mapping.source_rows, dtype=np.int64)
    n_span = np.maximum(hi - lo, 1)[None, :]
    mean = (src_sum[rows][:, hi] - src_sum[rows][:, lo]) / n_span
    var = (src_sq[rows][:, hi] - src_sq[rows][:, lo]) / n_span - mean ** 2
    flat = (np.sqrt(np.maximum(var, 0.0)) < cfg.flat_lead_std_mv).any(axis=0)

    keep, qc = window_qc_mask(windows, mapping.valid, bad, flat, cfg)
    windows, starts = windows[keep], starts[keep]
    windows = normalise_windows(windows, mapping.valid, cfg)

    ids, _ = (ids_for or lead_ids_for)(slots)
    ids = [i if mapping.valid[k] else 0 for k, i in enumerate(ids)]
    return ProcessedRecord(windows, starts / float(route.sampling_rate),
                           mapping.valid, ids, mapping.derived, qc, n_cand)


def strip_zero_padding(x: np.ndarray) -> np.ndarray:
    """Drop leading and trailing samples at which EVERY lead is exactly zero.

    CODE stores each exam in a fixed 4096-sample array and zero-fills whatever
    the recording did not cover. Those zeros are not a flat ECG, and leaving
    them in would put a step into the high-pass and a stretch of nothing into
    the window.
    """
    nz = np.where(np.any(x != 0, axis=0))[0]
    if nz.size == 0:
        return x[:, :0]
    return x[:, nz[0]:nz[-1] + 1]
