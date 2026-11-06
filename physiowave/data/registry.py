"""Dataset registry: one declarative entry per corpus.

An entry records everything the pipeline needs to decide *how* a dataset may be
used -- not just where it lives.  In particular:

* ``has_coordinates=False`` automatically disables the SSL branch for that
  dataset and logs the reason, because a spherical-spline Laplacian without
  electrode positions is undefined;
* ``emg_region`` separates limb/skeletal sEMG from facial EMG.  Facial EMG has
  different generators, bandwidth and artefact structure and must not enter the
  limb sEMG pretraining corpus; :func:`assert_limb_semg` enforces it;
* ``requires_agreement=True`` datasets are never downloaded automatically.  The
  pipeline points at a local path the user has already obtained under the data
  use agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Declarative description of one corpus."""

    dataset_id: str
    modality: str
    task: str = "pretrain"                     # 'pretrain' | 'classification' | 'multilabel'
    root: Optional[str] = None                 # local path; never auto-downloaded
    sampling_rate: float = 256.0
    num_channels: Optional[int] = None
    montage: Optional[str] = None              # template montage name
    has_coordinates: bool = True
    reference_type: str = "unknown"
    derivation_type: str = "monopolar"
    emg_region: str = "unknown"                # 'limb' | 'facial' | 'trunk' | 'unknown'
    num_classes: Optional[int] = None
    requires_agreement: bool = False
    url: Optional[str] = None
    notes: str = ""
    file_glob: str = "*.h5"
    subject_from_path: Optional[str] = None    # regex with one capture group

    def uses_ssl(self) -> bool:
        """Whether the SSL branch is admissible for this dataset."""
        if self.modality != "eeg":
            return False
        if not self.has_coordinates:
            logger.warning(
                "Dataset %s has no electrode coordinates; the SSL branch is disabled "
                "for it (spherical-spline CSD needs positions).", self.dataset_id
            )
            return False
        if self.derivation_type.lower().startswith("bipolar"):
            logger.warning(
                "Dataset %s is a bipolar montage; the SSL branch is disabled for it "
                "(the surface Laplacian is defined on monopolar potentials).",
                self.dataset_id,
            )
            return False
        return True


REGISTRY: Dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    """Add (or replace) a dataset entry."""
    REGISTRY[spec.dataset_id] = spec
    return spec


def get(dataset_id: str) -> DatasetSpec:
    if dataset_id not in REGISTRY:
        raise KeyError(f"Unknown dataset {dataset_id!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]


def list_datasets(modality: Optional[str] = None) -> List[DatasetSpec]:
    specs = list(REGISTRY.values())
    return [s for s in specs if modality is None or s.modality == modality]


def assert_limb_semg(specs: Sequence[DatasetSpec]) -> None:
    """Raise if any sEMG corpus in ``specs`` is not limb/skeletal.

    Guards the sEMG pretraining entry point.  ``unknown`` is rejected too: an
    unlabelled region is not evidence of a limb recording.
    """
    bad = [s.dataset_id for s in specs
           if s.modality == "semg" and s.emg_region != "limb"]
    if bad:
        raise ValueError(
            "limb sEMG pretraining received non-limb EMG datasets: "
            f"{bad}. Facial EMG has different generators, bandwidth and artefact "
            "structure and must not be mixed into the limb sEMG corpus. Set "
            "emg_region='limb' in the registry only for limb/skeletal recordings."
        )


# --------------------------------------------------------------------------- #
# Built-in entries.  `root` is left None on purpose: paths come from config, and
# nothing here is downloaded automatically.
# --------------------------------------------------------------------------- #
_EEG = dict(modality="eeg", task="pretrain", montage="standard_1020_19")

register(DatasetSpec("tueg", **_EEG, sampling_rate=256.0, requires_agreement=True,
                     url="https://isip.piconepress.com/projects/tuh_eeg/",
                     notes="TUH EEG Corpus. Requires a signed data use agreement; "
                           "provide a local path, never auto-downloaded."))
register(DatasetSpec("siena", **_EEG, sampling_rate=512.0, requires_agreement=False,
                     url="https://physionet.org/content/siena-scalp-eeg/1.0.0/",
                     notes="Siena Scalp EEG Database."))
register(DatasetSpec("tuab", modality="eeg", task="classification", num_classes=2,
                     montage="standard_1020_19", sampling_rate=256.0,
                     requires_agreement=True, notes="TUH Abnormal EEG; normal vs abnormal."))
register(DatasetSpec("tuar", modality="eeg", task="multilabel", num_classes=5,
                     montage="standard_1020_19", sampling_rate=256.0,
                     requires_agreement=True, notes="TUH Artifact Corpus."))
register(DatasetSpec("tusl", modality="eeg", task="classification", num_classes=4,
                     montage="standard_1020_19", sampling_rate=256.0,
                     requires_agreement=True, notes="TUH Slowing Corpus."))
register(DatasetSpec("bci_iv_2a", modality="eeg", task="classification", num_classes=4,
                     montage="standard_1010_61", num_channels=22, sampling_rate=250.0,
                     url="https://www.bbci.de/competition/iv/",
                     notes="BCI Competition IV-2a motor imagery, 4 classes."))
register(DatasetSpec("seizeit2", modality="eeg", task="classification", num_classes=2,
                     montage="standard_1020_19", sampling_rate=256.0,
                     requires_agreement=True,
                     notes="SeizeIT2 multimodal (EEG + ECG/EMG wearable) seizure data."))
register(DatasetSpec("mpdb", modality="eeg", task="classification", num_classes=5,
                     montage="standard_1020_19", sampling_rate=256.0,
                     url="https://doi.org/10.6084/m9.figshare.22192831",
                     notes="MPDB (Tao et al., Sci. Data 2024): 59-ch EEG, chest ECG, four "
                           "limb sEMG channels; FIVE driving behaviours (smooth, "
                           "acceleration, deceleration, lane change, turning)."))

register(DatasetSpec("mimic_iv_ecg", modality="ecg", task="pretrain", sampling_rate=500.0,
                     num_channels=12, has_coordinates=False, derivation_type="ecg_12lead",
                     url="https://physionet.org/content/mimic-iv-ecg/1.0/"))
register(DatasetSpec("ptbxl", modality="ecg", task="classification", num_classes=5,
                     sampling_rate=500.0, num_channels=12, has_coordinates=False,
                     derivation_type="ecg_12lead",
                     url="https://physionet.org/content/ptb-xl/1.0.3/"))
register(DatasetSpec("cpsc2018", modality="ecg", task="multilabel", num_classes=9,
                     sampling_rate=500.0, num_channels=12, has_coordinates=False,
                     derivation_type="ecg_12lead"))
register(DatasetSpec("shaoxing", modality="ecg", task="multilabel", num_classes=4,
                     sampling_rate=500.0, num_channels=12, has_coordinates=False,
                     derivation_type="ecg_12lead"))

register(DatasetSpec("ninapro_db6", modality="semg", task="pretrain", sampling_rate=2000.0,
                     num_channels=14, has_coordinates=False, emg_region="limb",
                     url="https://ninapro.hevs.ch/instructions/DB6.html",
                     notes="Forearm sEMG electrode array -- limb/skeletal muscle."))
register(DatasetSpec("epn612", modality="semg", task="classification", num_classes=6,
                     sampling_rate=200.0, num_channels=8, has_coordinates=False,
                     emg_region="limb", url="https://zenodo.org/records/4421500",
                     notes="Forearm ring array, hand gestures -- limb/skeletal muscle."))

register(DatasetSpec("synthetic_eeg", modality="eeg", task="pretrain", sampling_rate=256.0,
                     montage="standard_1020_19", notes="Synthetic smoke-test corpus."))
register(DatasetSpec("synthetic_ecg", modality="ecg", task="pretrain", sampling_rate=500.0,
                     num_channels=12, has_coordinates=False, derivation_type="ecg_12lead",
                     notes="Synthetic smoke-test corpus."))
register(DatasetSpec("synthetic_semg", modality="semg", task="pretrain", sampling_rate=1000.0,
                     num_channels=8, has_coordinates=False, emg_region="limb",
                     notes="Synthetic smoke-test corpus."))
