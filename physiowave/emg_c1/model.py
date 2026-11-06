"""
Multi-route sEMG pretrainer: the EEG C1 model on sEMG routes.

Everything but the routes, the electrode vocabulary and the wavelets the
frontend starts from is the EEG model, inherited: per-route
SoftGateWaveletDecomp + dynamic ScaleFold, a patch embedding and two decoders
per rate, one shared RoPE Transformer (RMSNorm, QK-norm, SwiGLU), C1 electrode
embedding at the token site, signal masking before the frontend, the detached
per-patch-normalised spectral target and the raw target. Unlike ECG there is
no group masking: no two sEMG channels are exact linear functions of others.

THE WAVELETS. db2, db4, db7, sym4 and sym5 -- the families sEMG denoising and
feature extraction most often settle on: short Daubechies filters for the
sharp motor-unit action potentials, sym4/sym5 for their near-symmetry. All are
4-14 taps, so the ``pad`` initialisation places each in the 16-tap kernel
exactly. As on the other modalities the bank is shared across levels and runs
at full rate, so the family sets each filter's starting shape; the filters are
learnt and the selector weights them per window.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..eeg_c1.model import MultiRouteEEGPretrainer
from ..eeg_c1.routes import Route
from .electrodes import EMG_ELECTRODE_VOCAB, emg_vocab_payload
from .routes import ROUTES

EMG_WAVELETS = ("db2", "db4", "db7", "sym4", "sym5")


class MultiRouteEMGPretrainer(MultiRouteEEGPretrainer):
    def __init__(self, *, wavelet_names=None,
                 routes: Optional[Dict[str, Route]] = None,
                 channel_vocab_size: Optional[int] = None, **kwargs):
        super().__init__(
            wavelet_names=list(wavelet_names or EMG_WAVELETS),
            routes=dict(ROUTES if routes is None else routes),
            channel_vocab_size=channel_vocab_size or len(EMG_ELECTRODE_VOCAB),
            **kwargs)

    def vocab_fingerprint(self) -> Dict[str, object]:
        payload = emg_vocab_payload()
        if self.channel_encoder is None:
            payload["channel_vocab_size"] = 0
        return payload
