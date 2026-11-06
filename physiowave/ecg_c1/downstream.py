"""
The ECG C1 encoder with a classification head, for downstream evaluation.

physiowave.eeg_c1.downstream.EEGC1Downstream with the modality swapped: the
ECG routes (L12_500, L1_250), the ECG lead vocabulary and its hash, and the
ECG wavelet families. Everything else -- slot placement, the channel code at
the token site, pooling, freezing, and the accounting of exactly which weights
a checkpoint supplied -- is the EEG implementation, unchanged.

A 12-lead, 500 Hz, 10 s downstream recording IS the L12_500 route, so with
``route_id: L12_500`` the pretrained wavelet frontend and patch embedding are
transferred along with the transformer and the channel encoder. That is the
whole pretrained encoder; on EEG it is usually only the transformer.
"""

from __future__ import annotations

from .leads import ECG_LEAD_VOCAB, ecg_vocab_payload, lead_ids_for, normalize_lead
from .model import ECG_WAVELETS
from .routes import ROUTES
from ..eeg_c1.downstream import EEGC1Downstream


class ECGC1Downstream(EEGC1Downstream):
    MODALITY = "ecg"
    DEFAULT_WAVELETS = tuple(ECG_WAVELETS)
    ROUTES_TABLE = ROUTES
    VOCAB_SIZE = len(ECG_LEAD_VOCAB)

    @staticmethod
    def ids_for(names):
        return lead_ids_for(names)

    @staticmethod
    def normalize_name(name):
        return normalize_lead(name)

    @staticmethod
    def vocab_payload():
        return ecg_vocab_payload()
