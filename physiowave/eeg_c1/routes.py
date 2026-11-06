"""
The four hard routes of the EEG C1 pretraining corpus.

A route is the (electrode count, sampling rate) shape a batch arrives in. It is
fixed by the table below and chosen by ``route_id`` carried in the data -- there
is no learned router here, and this is deliberately not a mixture of experts:
which frontend runs is a property of the recording, known before the model sees
it, so making it a learned decision would be inventing a choice that does not
exist.

Every route window is 4 seconds and every patch is 0.5 seconds, so every route has
exactly 8 time patches per channel and the token count is just the electrode
count times eight. That is what keeps one Transformer usable across all four:
the sequence length changes, the meaning of a position does not.

    route       C     rate    samples   patch_t   tokens
    E19_256     19    256     1024      128       152
    E32_512     32    512     2048      256       256
    E64_256     64    256     1024      128       512
    E128_512    128   512     2048      256       1024
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Canonical electrode slots
#
# A route's slot list IS its channel axis: row i of every window on that route
# is the electrode named at index i, on every dataset. A dataset that does not
# record one of them leaves the row zero-filled with valid_channel_mask False.
# Mapping is by name, never by position -- two datasets that both have "26
# channels" do not have the same 26.
# --------------------------------------------------------------------------- #

#: The 10-20 nineteen. TUEG records these under the old T3/T4/T5/T6 spelling,
#: which channel_embedding.normalize_channel_name resolves to T7/T8/P7/P8.
SLOTS_19: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
)

#: Thirty-two slots chosen so that TDBRAIN's twenty-six land exactly, in order,
#: and the six it does not record are the padded ones. The alternative -- a
#: generic 32-channel layout -- would have scattered TDBRAIN across the axis and
#: left its FC3/FCz/FC4/CP3/CPz/CP4 with nowhere to go but UNK.
SLOTS_32: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    # The six TDBRAIN leaves empty; FACED and anything else may fill them.
    "FC5", "FC6", "CP5", "CP6", "PO3", "PO4",
)

#: The 10-10 sixty-four, the montage PhysioNetMI and M3CV both record.
SLOTS_64: Tuple[str, ...] = (
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
    # The last two slots were P9/P10, which --inspect showed NEITHER corpus on
    # this route records: PhysioNetMI fills 62 of 64 and M3CV 60, both missing
    # exactly those. Two guaranteed-dead slots in every window of the route.
    #
    # PhysioNetMI wants T9/T10 for them and M3CV wants FT9/FT10/TP9/TP10, and
    # only two are free, so the total filled is 124 either way -- each free slot
    # serves one corpus. The tiebreak is how much data is behind each: M3CV has
    # 92,340 windows against PhysioNetMI's 43,519, so TP9/TP10 buys 184,680
    # electrode-windows of real signal against T9/T10's 87,038.
    #
    # T9/T10 and FT9/FT10 stay in the vocabulary and are recorded in every
    # affected shard's unmatched_source_channels; they are dropped from this
    # route, not from the corpus.
    "TP9", "TP10",
)

#: 128 slots. HBN records on an EGI HydroCel net whose labels are E1..E128 after
#: the reference is removed, and HGD records 10-5 scalp names; the two do not
#: share an electrode naming scheme, so the slot list is the EGI one and HGD's
#: names are mapped onto it by the adapter's own table rather than by pretending
#: E-numbers and 10-5 labels are interchangeable. A dataset that fills none of a
#: slot leaves it padded, which is the honest outcome for two montages that
#: genuinely do not align.
SLOTS_128: Tuple[str, ...] = tuple(f"E{i}" for i in range(1, 129))

#: HGD's 128 electrodes, MEASURED from the corpus rather than transcribed.
#:
#: Derived with --derive-slots over eight recordings: the names present in every
#: montage, in the order the widest one lists them. That is what excludes the
#: five aux channels the 133-channel files carry (EOGh, EOGv, EMG_RH, EMG_LH,
#: EMG_RF) without a hand-maintained blocklist, and it is why this is exactly
#: 128 rather than a list padded to fit.
#:
#: It is 10-05, not 10-10: fifty of these are halfway positions (FFC5h lies
#: between FFC5 and FFC3). An earlier version of this list was assembled from
#: 10-10 names and a guess, and it resolved only 74 of the 128 -- below the
#: coverage gate, so every HGD recording would have been skipped.
#:
#: Kept separate from SLOTS_128 because HGD occupies the E128_512 route by
#: shape, not by sharing HBN's electrode identities: HBN records EGI net
#: positions (E1..E128) and these are scalp labels. A shard from one leaves the
#: other's slots padded, which is the honest outcome for two montages that
#: genuinely do not align.
#:
#: Cz is at index 15 and is the RECORDING REFERENCE -- the authors note residual
#: signal remains on it, so it is a real but attenuated channel, not a dead one.
#: M1/M2 at 12 and 18 are the mastoids.
SLOTS_128_HGD: Tuple[str, ...] = (
    "Fp1", "Fp2", "Fpz", "F7", "F3", "Fz",
    "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "Cz", "C4", "T8",
    "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1",
    "Oz", "O2", "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6", "FC3", "FCz",
    "FC4", "C5", "C1", "C2", "C6", "CP3",
    "CPz", "CP4", "P5", "P1", "P2", "P6",
    "PO5", "PO3", "PO4", "PO6", "FT7", "FT8",
    "TP7", "TP8", "PO7", "PO8", "FT9", "FT10",
    "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10",
    "AFF1", "AFz", "AFF2", "FFC5h", "FFC3h", "FFC4h",
    "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h",
    "CCP3h", "CCP4h", "CCP6h", "CPP5h", "CPP3h", "CPP4h",
    "CPP6h", "PPO1", "PPO2", "I1", "Iz", "I2",
    "AFP3h", "AFP4h", "AFF5h", "AFF6h", "FFT7h", "FFC1h",
    "FFC2h", "FFT8h", "FTT9h", "FTT7h", "FCC1h", "FCC2h",
    "FTT8h", "FTT10h", "TTP7h", "CCP1h", "CCP2h", "TTP8h",
    "TPP7h", "CPP1h", "CPP2h", "TPP8h", "PPO9h", "PPO5h",
    "PPO6h", "PPO10h", "POO9h", "POO3h", "POO4h", "POO10h",
    "OI1h", "OI2h",
)
assert len(SLOTS_128_HGD) == 128, (
    f"HGD needs exactly 128 slots for E128_512, has {len(SLOTS_128_HGD)}")


@dataclass(frozen=True)
class Route:
    """One hard route. Everything the model and the loader need to agree on."""

    route_id: str
    n_channels: int
    sampling_rate: int
    window_seconds: float = 4.0
    patch_seconds: float = 0.5
    slots: Tuple[str, ...] = ()

    @property
    def window_samples(self) -> int:
        return int(self.window_seconds * self.sampling_rate)

    @property
    def patch_t(self) -> int:
        return int(self.patch_seconds * self.sampling_rate)

    @property
    def patch_size(self) -> Tuple[int, int]:
        """``(freq rows per patch, samples per patch)``.

        One row per patch: the fold has already reduced the scale axis, so a
        row is a channel and a patch must not span two of them.
        """
        return (1, self.patch_t)

    @property
    def patches_per_channel(self) -> int:
        return self.window_samples // self.patch_t

    @property
    def n_tokens(self) -> int:
        return self.n_channels * self.patches_per_channel

    @property
    def rate_key(self) -> str:
        """Which PatchEmbed and reconstruction head this route shares."""
        return str(self.sampling_rate)

    def describe(self) -> str:
        return (f"{self.route_id}: {self.n_channels}x{self.window_samples} @ "
                f"{self.sampling_rate}Hz -> {self.n_channels}x"
                f"{self.patches_per_channel} = {self.n_tokens} tokens "
                f"(patch {self.patch_size})")


ROUTES: Dict[str, Route] = {
    r.route_id: r
    for r in (
        Route("E19_256", 19, 256, slots=SLOTS_19),
        Route("E32_512", 32, 512, slots=SLOTS_32),
        Route("E64_256", 64, 256, slots=SLOTS_64),
        Route("E128_512", 128, 512, slots=SLOTS_128),
    )
}

ROUTE_IDS: Tuple[str, ...] = tuple(ROUTES)

#: Which sampling rates need their own PatchEmbed and decoder. Two, not four:
#: the patch is 0.5 s everywhere, so the kernel is a function of the rate alone.
RATE_KEYS: Tuple[str, ...] = tuple(
    sorted({r.rate_key for r in ROUTES.values()}, key=int)
)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    route_id: str
    native_rate: float | None      # None where it varies per recording (TUEG)
    native_channels: int
    slots: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def route(self) -> Route:
        return ROUTES[self.route_id]


#: The seven pretraining corpora. DEAP is deliberately absent and a test pins
#: that: it is a downstream evaluation set, and pretraining on it would make
#: every DEAP number a report on data the encoder had already seen.
PRETRAIN_DATASETS: Dict[str, DatasetSpec] = {
    d.dataset_id: d
    for d in (
        DatasetSpec("tueg", "E19_256", None, 19, SLOTS_19,
                    "rate varies per recording; resampled to 256"),
        DatasetSpec("faced", "E32_512", 1000.0, 32, SLOTS_32,
                    "raw 1000 Hz BDF only; the 250 Hz release is refused"),
        DatasetSpec("tdbrain", "E32_512", 500.0, 26, SLOTS_32,
                    "26 electrodes mapped by name into 32 slots; 6 padded"),
        DatasetSpec("physionet_mi", "E64_256", 160.0, 64, SLOTS_64,
                    "native 64 kept; 160 -> 256"),
        DatasetSpec("m3cv", "E64_256", 250.0, 64, SLOTS_64,
                    "native 64 kept; 250 -> 256; pretraining only"),
        DatasetSpec("hbn", "E128_512", 500.0, 129, SLOTS_128,
                    "129 -> 128 by removing the recorded reference; 500 -> 512"),
        DatasetSpec("hgd", "E128_512", 500.0, 128, SLOTS_128_HGD,
                    "native 128; 500 -> 512; pretraining only"),
    )
}

DATASET_IDS: Tuple[str, ...] = tuple(PRETRAIN_DATASETS)

#: Corpora for CONTINUED pretraining only: not part of the seven the released
#: encoder was pretrained on, so a run draws from one only when its config names
#: it (``data.continuation_datasets``) and ``PRETRAIN_DATASETS`` keeps meaning
#: what the original run read.
CONTINUATION_DATASETS: Dict[str, DatasetSpec] = {
    d.dataset_id: d
    for d in (
        DatasetSpec("siena", "E19_256", 512.0, 29, SLOTS_19,
                    "Siena Scalp EEG (PhysioNet, ODC-By): 14 adults with focal "
                    "epilepsy, 10-20 + extras at 512 Hz; the 19 10-20 electrodes "
                    "kept; 512 -> 256; continued pretraining only"),
    )
}

#: Every EEG corpus a run may draw from.
EEG_DATASETS: Dict[str, DatasetSpec] = {**PRETRAIN_DATASETS, **CONTINUATION_DATASETS}


def datasets_with(continuation: Optional[Sequence[str]] = None) -> Dict[str, DatasetSpec]:
    """The seven pretraining corpora plus the named continuation corpora."""
    out = dict(PRETRAIN_DATASETS)
    for name in continuation or ():
        if name not in CONTINUATION_DATASETS:
            raise SystemExit(f"unknown continuation corpus {name!r}; known: "
                             f"{sorted(CONTINUATION_DATASETS)}")
        out[name] = CONTINUATION_DATASETS[name]
    return out

#: Never pretrained on. Named rather than merely omitted so the refusal is
#: testable and survives someone adding datasets later.
DOWNSTREAM_ONLY: Tuple[str, ...] = ("deap", "sleep_edf", "erpbci", "physio_p300")


def datasets_for_route(route_id: str,
                       datasets: Optional[Dict[str, DatasetSpec]] = None
                       ) -> List[str]:
    datasets = PRETRAIN_DATASETS if datasets is None else datasets
    return [d for d, s in datasets.items() if s.route_id == route_id]


def balanced_sampling_weights(
        routes: Optional[Dict[str, Route]] = None,
        datasets: Optional[Dict[str, DatasetSpec]] = None) -> Dict[str, float]:
    """P(route) = 1/4, then uniform over that route's datasets.

    Every route gets equal attention regardless of how much data happened to be
    collected for it. The cost is repetition: PhysioNetMI holds ~39k training
    windows against TUEG's 13.7M, so an eighth of the mixture means each of its
    windows is revisited on the order of a hundred times per run while a TUEG
    window is seen less than once.

    Kept, and selectable with ``weights: balanced``, because for a run whose
    point is the four frontends rather than the corpus this is the right
    trade -- but it is no longer the default.
    """
    routes = ROUTES if routes is None else routes
    weights: Dict[str, float] = {}
    per_route = 1.0 / len(routes)
    for route_id in routes:
        members = datasets_for_route(route_id, datasets)
        if not members:
            continue
        for dataset_id in members:
            weights[dataset_id] = per_route / len(members)
    return weights


def proportional_sampling_weights(
        window_counts: Dict[str, int],
        batch_by_route: Optional[Dict[str, int]] = None,
        datasets: Optional[Dict[str, DatasetSpec]] = None) -> Dict[str, float]:
    """Each dataset contributes windows in proportion to how many it has.

    An epoch is then one pass over the corpus: every window is seen once, and a
    small dataset is not revisited a hundred times to fill a quota.

    THE DIVISION BY BATCH SIZE IS THE POINT. These weights are probabilities
    over STEPS, and a step draws ``batch_by_route[route]`` windows -- 64 on
    E19_256, 12 on E128_512, because a 128-channel window is 6.7x the tokens of
    a 19-channel one. Weighting steps by window count directly would therefore
    give E19_256 five times the windows its share of the corpus warrants.
    Dividing by the route's batch size cancels it exactly:

        P(d) ∝ n_d / b_d   =>   windows(d) ∝ P(d)·b_d = n_d

    which is the identity that makes ``by_window`` in the epoch log come out
    equal to each dataset's share of the corpus.
    """
    batch_by_route = batch_by_route or {}
    datasets = PRETRAIN_DATASETS if datasets is None else datasets
    raw: Dict[str, float] = {}
    for dataset_id, n in window_counts.items():
        if n <= 0 or dataset_id not in datasets:
            continue
        route_id = datasets[dataset_id].route_id
        b = float(batch_by_route.get(route_id, 1) or 1)
        raw[dataset_id] = float(n) / b
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def temperature_sampling_weights(
        window_counts: Dict[str, int], alpha: float,
        batch_by_route: Optional[Dict[str, int]] = None,
        datasets: Optional[Dict[str, DatasetSpec]] = None) -> Dict[str, float]:
    """``P(d) ∝ n_d**alpha / b_d`` -- the dial between the two extremes.

    ``alpha=1`` is proportional sampling and ``alpha=0`` makes every dataset
    equal regardless of size. In between, a corpus a hundred times larger gets
    ``100**alpha`` times the windows rather than a hundred: at 0.5 that is ten.
    This is the usual way multilingual pretraining stops its largest language
    from being the whole run without pretending a tiny one is the same size.

    The division by batch size is the same correction as in
    ``proportional_sampling_weights`` and for the same reason: these are
    probabilities over steps, and a step draws a route-specific number of
    windows.
    """
    batch_by_route = batch_by_route or {}
    datasets = PRETRAIN_DATASETS if datasets is None else datasets
    raw: Dict[str, float] = {}
    for dataset_id, n in window_counts.items():
        if n <= 0 or dataset_id not in datasets:
            continue
        b = float(batch_by_route.get(datasets[dataset_id].route_id, 1) or 1)
        raw[dataset_id] = (float(n) ** float(alpha)) / b
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def default_sampling_weights() -> Dict[str, float]:
    """The configured mixture when no corpus is in hand to measure."""
    return balanced_sampling_weights()
