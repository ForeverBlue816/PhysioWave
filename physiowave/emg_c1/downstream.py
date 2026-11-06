"""
The sEMG C1 encoder with a classification head, for downstream evaluation.

physiowave.eeg_c1.downstream.EEGC1Downstream with the modality swapped: the
sEMG routes, the sEMG electrode vocabulary and its hash, and the sEMG wavelet
families. The benchmark montages (EPN-612's Myo armband, GRABMyo's forearm
and wrist rings, NinaPro DB2's Delsys electrodes) are none of the pretraining
routes, so each gets its own wavelet frontend and patch embedding and takes
the pretrained channel encoder and Transformer -- route_id stays unset.
Their electrode names are not in the vocabulary either, so every channel
resolves to <unk>: the channel code is then one shared vector, and channels
are told apart by the 2-D position encoding, as on any montage the
vocabulary does not know.
"""

from __future__ import annotations

from .electrodes import EMG_ELECTRODE_VOCAB, electrode_ids_for, emg_vocab_payload, normalize_electrode
from .model import EMG_WAVELETS
from .routes import ROUTES
from ..eeg_c1.downstream import EEGC1Downstream


class EMGC1Downstream(EEGC1Downstream):
    MODALITY = "emg"
    DEFAULT_WAVELETS = tuple(EMG_WAVELETS)
    ROUTES_TABLE = ROUTES
    VOCAB_SIZE = len(EMG_ELECTRODE_VOCAB)

    @staticmethod
    def ids_for(names):
        return electrode_ids_for(names)

    @staticmethod
    def normalize_name(name):
        return normalize_electrode(name)

    @staticmethod
    def vocab_payload():
        return emg_vocab_payload()
