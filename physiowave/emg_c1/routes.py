"""
The routes and the corpus of the sEMG C1 pretraining run.

A route is the shape a batch arrives in, as on the EEG and ECG side, and it is
chosen by the data, not learned. Three, all at 2000 Hz:

    route      channels   montage                       tokens (8 per channel)
    W16_2000   16         a wristband (emg2pose, emg2qwerty)        128
    A24_2000   24         three forearm rings of 8 (putEMG)         192
    G64_2000   64         ONE high-density grid                     512
                          (Hyser's 8x8; CEMHSEY's 8x8 and 5x13)

ONE GRID PER SAMPLE. Hyser's 256 channels are four 8x8 grids and CEMHSEY's 320
three 8x8 and two 5x13 grids of 64 electrodes, and each grid is its own sample
on G64_2000 -- the way emg2qwerty's two wrists are two samples on W16_2000.
CEMHSEY's two grid types are two corpora, ``cemhsey_8x8`` and
``cemhsey_5x13``, read from the same files: a corpus's shards carry one
electrode map, and the two grids place their 64 electrodes differently. Two measurements decided it, both at the
standard settings:

    whole recording as one route     256 ch      320 ch
      wavelet frontend               9.8 M       15.3 M parameters
      tokens per window              2,048       2,560
    one grid as one sample           64 ch
      wavelet frontend               ~0.6 M
      tokens per window              512

The frontend's cross-channel FFN grows with the square of the channel count,
so at 320 channels it alone would be larger than the 10.6 M shared
Transformer, and the attention would pay for 2,560 tokens a window. A grid
keeps the spatial structure HD-sEMG exists for -- 64 neighbouring electrodes
over one muscle group -- and gives up only the co-activation of grids over
different muscles within a single sample.

2000 Hz, because emg2pose and emg2qwerty are recorded at it and they are most
of the corpus: nothing is upsampled. Hyser and CEMHSEY (2048 Hz) and putEMG
(5120 Hz) are resampled down; sEMG power lies below ~500 Hz, well inside the
1000 Hz Nyquist. A 1 s window of eight 0.125 s patches: 250 samples a patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..eeg_c1.routes import Route
from .electrodes import (BAND_16, CEMHSEY_GRID, CEMHSEY_GRID_5X13,
                         HYSER_GRID, PUTEMG_24)

SAMPLING_RATE = 2000
WINDOW_SECONDS = 1.0
PATCH_SECONDS = 0.125

ROUTES: Dict[str, Route] = {
    r.route_id: r
    for r in (
        Route("W16_2000", 16, SAMPLING_RATE, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=BAND_16),
        Route("A24_2000", 24, SAMPLING_RATE, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=PUTEMG_24),
        Route("G64_2000", 64, SAMPLING_RATE, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=HYSER_GRID),
    )
}
ROUTE_IDS: Tuple[str, ...] = tuple(ROUTES)
RATE_KEYS: Tuple[str, ...] = tuple(sorted({r.rate_key for r in ROUTES.values()}))


@dataclass(frozen=True)
class EMGDatasetSpec:
    dataset_id: str
    route_id: str
    native_rate: Optional[float]
    native_channels: int
    mains_hz: Optional[float]
    raw_dir: str
    slots: Tuple[str, ...] = ()
    default_max_windows_per_record: Optional[int] = None
    notes: str = ""

    @property
    def route(self) -> Route:
        return ROUTES[self.route_id]

    @property
    def lead_slots(self) -> Tuple[str, ...]:
        return tuple(self.slots) or tuple(self.route.slots)


PRETRAIN_DATASETS: Dict[str, EMGDatasetSpec] = {
    d.dataset_id: d
    for d in (
        # -- large-scale wearable sEMG: most of the corpus -----------------
        EMGDatasetSpec("emg2pose", "W16_2000", 2000.0, 16, 60.0, "emg2pose",
                       slots=BAND_16,
                       notes="193 users, ~370 h; Meta sEMG-RD wristband"),
        EMGDatasetSpec("emg2qwerty", "W16_2000", 2000.0, 32, 60.0,
                       "emg2qwerty", slots=BAND_16,
                       notes="108 users, ~346 h; two bands, each wrist its "
                             "own 16-channel sample"),
        # -- high-density sEMG: acquisition diversity ----------------------
        EMGDatasetSpec("hyser", "G64_2000", 2048.0, 256, 50.0, "Hyser",
                       slots=HYSER_GRID,
                       notes="20 subjects; 256 ch = four 8x8 grids, each its "
                             "own sample; 2048 -> 2000 Hz"),
        EMGDatasetSpec("cemhsey_8x8", "G64_2000", 2048.0, 192, 50.0,
                       "CEMHSEY", slots=CEMHSEY_GRID,
                       notes="channels 1-192: three 8x8 grids (10 mm), "
                             "proximal forearm; 19 subjects over 11 days"),
        EMGDatasetSpec("cemhsey_5x13", "G64_2000", 2048.0, 128, 50.0,
                       "CEMHSEY", slots=CEMHSEY_GRID_5X13,
                       notes="channels 193-320: two 5x13 grids (8 mm), "
                             "distal forearm; the same files"),
        # -- acquisition diversity, optional --------------------------------
        EMGDatasetSpec("putemg", "A24_2000", 5120.0, 24, 50.0, "putEMG",
                       slots=PUTEMG_24,
                       notes="44 subjects; 24 ch in three rings; "
                             "5120 -> 2000 Hz"),
    )
}
DATASET_IDS: Tuple[str, ...] = tuple(PRETRAIN_DATASETS)

#: Downstream evaluation only, never pretrained on.
DOWNSTREAM_ONLY: Tuple[str, ...] = ("db5", "ninapro_db5", "epn612", "epn_612",
                                    "grabmyo", "db2", "ninapro_db2", "uci_emg",
                                    "uci_emg_xsess")

#: Per rank. 128 / 192 / 512 tokens a window, so ~24 k tokens a step on each
#: route; BATCH_SIZE_BY_ROUTE overrides at submission.
DEFAULT_BATCH_BY_ROUTE: Dict[str, int] = {
    "W16_2000": 192,
    "A24_2000": 128,
    "G64_2000": 48,
}
