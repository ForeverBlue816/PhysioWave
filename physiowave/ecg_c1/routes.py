"""
The routes and the corpus of the ECG C1 pretraining run.

Same idea as physiowave.eeg_c1.routes: a route is the (lead count, sampling
rate) shape a batch arrives in, chosen by ``route_id`` carried in the data. It
is not learned and it is not a mixture of experts -- whether a recording is a
12-lead resting ECG or a single-lead monitor trace is known before the model
sees it.

Two routes, because the corpus has exactly two shapes in it:

    route      leads  rate   window         patch         tokens
    L12_500    12     500    10 s = 5000    0.5 s = 250   12 x 20 = 240
    L1_250      1     250    10 s = 2500    0.5 s = 125    1 x 20 =  20

L12_500 is every 12-lead corpus: MIMIC-IV-ECG, CODE-15%, SPH, Georgia and
MedalCare-XL. 500 Hz is the native rate of all but CODE, and it is what the
repository's ECG fine-tuning already feeds the model; CODE-15% is recorded at
400 Hz and resampled up, which each shard records in its provenance.

L1_250 is the single-lead corpora: Icentia11k's chest patch at its native
250 Hz, and PulseDB's lead II, recorded at 125 Hz and resampled up. 125 Hz
cannot have a route of its own: a 0.5 s patch there is 62.5 samples. Putting
either on the 12-lead route would mean eleven padded rows in every window,
eleven twelfths of the compute spent on zeros. The two share the route's
frontend and are told apart by their lead names, as HBN and HGD share an EEG
route with different electrode sets -- a dataset's ``slots`` are its own.

TEN SECONDS, because that is what a resting ECG is, and PulseDB's segments are
10 s too: for most corpora a record is one window and nothing is cut or
padded. SPH's 10-60 s records give one window per 10 s. A 0.5 s patch is the
EEG route's patch length too, which keeps the token count per second identical
across the two modalities; it is a judgement, not a measurement.

WHAT TEN SECONDS COSTS. CODE-15% is not a 10 s corpus: measured on its part
17, 58% of exams are 2934 samples at 400 Hz (7.3 s). A 7.3 s exam holds no
10 s window, so under this setting roughly three fifths of CODE-15%
contributes nothing -- preprocessing counts them as
``records_shorter_than_window``. Georgia's 52 five-second records are dropped
the same way. ``WINDOW_SECONDS = 5.0`` keeps every one of them (a 10 s record
then gives two windows) at the price of half the rhythm context per window.
It is one number, below, and it requires re-preprocessing every corpus: the
shards store the window, and the model refuses a window of the wrong length.

As on the EEG side, the rates share nothing but the Transformer: each route has
its own wavelet frontend (the filters are per-lead), each RATE its own patch
embedding and decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..eeg_c1.routes import Route
from .leads import LEADS_12, LIMB_LEADS

#: Icentia11k's one lead. Its own vocabulary entry -- see leads.py for why it
#: is not called lead I.
SLOTS_PATCH: Tuple[str, ...] = ("patch1",)

#: A bedside monitor's lead II: the same frontal-plane projection as the
#: 12-lead ECG's lead II, so it shares that embedding row.
SLOTS_LEAD_II: Tuple[str, ...] = ("II",)

#: The window every route uses, and preprocessing's default. See above.
WINDOW_SECONDS = 10.0
PATCH_SECONDS = 0.5

ROUTES: Dict[str, Route] = {
    r.route_id: r
    for r in (
        Route("L12_500", 12, 500, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=LEADS_12),
        Route("L1_250", 1, 250, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=SLOTS_PATCH),
    )
}
ROUTE_IDS: Tuple[str, ...] = tuple(ROUTES)
RATE_KEYS: Tuple[str, ...] = tuple(
    sorted({r.rate_key for r in ROUTES.values()}, key=int))

#: Per route, the slot indices that are masked TOGETHER at a time patch. The
#: limb leads are six views of one two-dimensional frontal-plane vector, so any
#: two determine the rest exactly; masking them independently would leave the
#: model half its masked limb patches to compute rather than infer.
LEAD_GROUPS: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "L12_500": (tuple(LEADS_12.index(n) for n in LIMB_LEADS),),
    "L1_250": (),
}


@dataclass(frozen=True)
class ECGDatasetSpec:
    """One pretraining corpus: where it goes, and what it takes to get it."""

    dataset_id: str
    route_id: str
    native_rate: Optional[float]
    native_leads: int
    #: Mains frequency where the recording was made, for the notch. None for
    #: simulated data, which has no mains to remove.
    mains_hz: Optional[float]
    #: Directory under the ECG download root.
    raw_dir: str
    #: This corpus's leads on the route, in row order. Empty means the route's
    #: own; a corpus that records a different lead than its route-mates names it
    #: here, and its shards carry that lead's id.
    slots: Tuple[str, ...] = ()
    #: A cap on windows per record, for the continuous corpora. None keeps all.
    default_max_windows_per_record: Optional[int] = None
    notes: str = ""

    @property
    def route(self) -> Route:
        return ROUTES[self.route_id]

    @property
    def lead_slots(self) -> Tuple[str, ...]:
        return tuple(self.slots) or tuple(self.route.slots)


PRETRAIN_DATASETS: Dict[str, ECGDatasetSpec] = {
    d.dataset_id: d
    for d in (
        # -- 12 leads ------------------------------------------------------
        ECGDatasetSpec("mimic_iv_ecg", "L12_500", 500.0, 12, 60.0,
                       "MIMIC-IV-ECG",
                       notes="~800k 10 s 12-lead ECGs, ~160k subjects, Beth "
                             "Israel (US); files store aVF before aVL"),
        ECGDatasetSpec("code15", "L12_500", 400.0, 12, 60.0, "CODE-15",
                       notes="345,779 exams, 233,770 patients; 400 -> 500 Hz, "
                             "zero padding stripped; ~58% are 7.3 s; Brazil"),
        ECGDatasetSpec("sph", "L12_500", 500.0, 12, 50.0, "SPH",
                       notes="25,770 ECGs, 24,666 patients, 10-60 s; Shandong "
                             "Provincial Hospital (China), CC0"),
        ECGDatasetSpec("georgia", "L12_500", 500.0, 12, 60.0, "Georgia",
                       notes="10,344 ECGs, PhysioNet/CinC Challenge 2021; no "
                             "patient id; 52 are 5 s"),
        ECGDatasetSpec("medalcare_xl", "L12_500", 500.0, 12, None,
                       "MedalCare-XL",
                       notes="16,842 SIMULATED 12-lead ECGs from 13 torso "
                             "models; the 'noise' variant; no mains"),
        # -- one lead ------------------------------------------------------
        ECGDatasetSpec("icentia11k", "L1_250", 250.0, 1, 60.0, "Icentia11k",
                       slots=SLOTS_PATCH, default_max_windows_per_record=16,
                       notes="11,000 patients, up to 2 weeks each; 16 windows "
                             "per 70-minute segment"),
        ECGDatasetSpec("pulsedb", "L1_250", 125.0, 1, 60.0, "PulseDB",
                       slots=SLOTS_LEAD_II,
                       notes="~5.2M 10 s ICU/OR segments of lead II from "
                             "MIMIC-III and VitalDB; 125 -> 250 Hz"),
    )
}
DATASET_IDS: Tuple[str, ...] = tuple(PRETRAIN_DATASETS)

#: Never pretrained on: the repository's ECG fine-tuning benchmarks
#: (ECG/ptbxl_finetune.py, ECG/cpsc_multilabel.py, ECG/shaoxing_multilabel.py).
#: Named rather than merely absent, so preprocessing and the merge refuse them
#: by name and a test pins it.
DOWNSTREAM_ONLY: Tuple[str, ...] = ("ptbxl", "cpsc2018", "chapman_shaoxing")

#: Per-rank micro-batch. A 12-lead window is 240 tokens and a single-lead one
#: is 20, so one batch size would either starve L1_250 or exhaust memory on
#: L12_500. These keep tokens per step within ~15% of each other; the trainer's
#: banner prints the window share each dataset actually gets.
DEFAULT_BATCH_BY_ROUTE: Dict[str, int] = {
    "L12_500": 96,
    "L1_250": 1024,
}
