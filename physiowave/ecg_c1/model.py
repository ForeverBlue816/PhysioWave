"""
Multi-route ECG pretrainer: the EEG C1 model, on ECG routes, with limb-lead masking.

Everything that is not ECG-specific is the EEG model, inherited rather than
copied: per-route SoftGateWaveletDecomp + dynamic ScaleFold, per-rate patch
embedding and the two decoders, the shared RoPE Transformer, C1 lead-name
embedding at the token site, signal masking before the frontend, the detached
per-patch-normalised spectral target and the raw target. The objective is
``physiowave.eeg_c1.model.masked_reconstruction_loss``, unchanged.

Three things differ, and each is here because of a fact about ECG:

**The routes.** 12 leads at 500 Hz and one patch lead at 250 Hz, 10 s windows
(physiowave.ecg_c1.routes).

**The wavelets the frontend starts from.** db4, db6, sym4, sym8, coif2 -- the
families the ECG literature decomposes QRS complexes and P/T waves with, all
short enough (8-16 taps) that the ``pad`` initialisation places them in the
16-tap kernel exactly, as a perfect-reconstruction filter bank. coif3, the
original PhysioWave default, has 18 taps and would not fit. The filters are
learnable and the selector weights them per window; this is where they START.

**Limb leads are masked together.** I, II, III, aVR, aVL and aVF are six
projections of one frontal-plane vector: III = II - I and the augmented leads
are fixed combinations of I and II, so any two of the six determine the other
four exactly. Under per-token masking at ratio 0.7, a masked limb-lead patch
has at least two of its five siblings visible at the same instant about 47% of
the time (1 - 0.7^5 - 5*0.3*0.7^4), and its "reconstruction" is then
arithmetic. So a time patch of the limb group is masked as ONE unit -- all six
leads or none -- and the precordial leads V1-V6, which are genuinely separate
measurements, are masked per token as before. Each part is masked at the
configured ratio, so the overall ratio is unchanged. ``lead_group_masking:
false`` restores the EEG behaviour exactly.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..eeg_c1.model import MultiRouteEEGPretrainer
from ..eeg_c1.routes import Route
from .leads import ECG_LEAD_VOCAB, ecg_vocab_payload
from .routes import LEAD_GROUPS, ROUTES

#: Where the frontend's filters start. See the module docstring.
ECG_WAVELETS = ("db4", "db6", "sym4", "sym8", "coif2")


class MultiRouteECGPretrainer(MultiRouteEEGPretrainer):
    """The EEG C1 pretrainer on the ECG routes. ``forward`` is one route's batch."""

    def __init__(self, *, wavelet_names=None,
                 routes: Optional[Dict[str, Route]] = None,
                 channel_vocab_size: Optional[int] = None,
                 lead_group_masking: bool = True,
                 **kwargs):
        super().__init__(
            wavelet_names=list(wavelet_names or ECG_WAVELETS),
            routes=dict(ROUTES if routes is None else routes),
            channel_vocab_size=channel_vocab_size or len(ECG_LEAD_VOCAB),
            **kwargs)
        self.lead_group_masking = bool(lead_group_masking)
        self.lead_groups = {rid: tuple(LEAD_GROUPS.get(rid, ()))
                            for rid in self.routes}
        # Which route the forward in progress belongs to. _select_mask is
        # called from the inherited forward with tokens alone, and the lead
        # groups are a property of the route.
        self._mask_route: Optional[str] = None

    def vocab_fingerprint(self) -> Dict[str, object]:
        payload = ecg_vocab_payload()
        if self.channel_encoder is None:
            payload["channel_vocab_size"] = 0
        return payload

    def forward(self, x, route_id, channel_meta=None, mask_ratio=None,
                mask_generator=None, mask_override=None):
        self._mask_route = route_id
        try:
            return super().forward(x, route_id, channel_meta=channel_meta,
                                   mask_ratio=mask_ratio,
                                   mask_generator=mask_generator,
                                   mask_override=mask_override)
        finally:
            self._mask_route = None

    # -- masking ------------------------------------------------------------ #
    def _select_mask(self, tokens, mask_ratio, valid_tokens, generator):
        groups = (self.lead_groups.get(self._mask_route, ())
                  if self.lead_group_masking and self._mask_route else ())
        if not groups:
            return super()._select_mask(tokens, mask_ratio, valid_tokens,
                                        generator)

        route = self.routes[self._mask_route]
        B, L, _ = tokens.shape
        C = route.n_channels
        P, rem = divmod(L, C)
        if rem:
            raise RuntimeError(f"{L} tokens is not C*P for C={C}")
        # The score is the inherited one -- frequency-guided importance mixed
        # with noise, the same single random draw -- so only WHICH tokens a
        # score selects changes here, not how it is made.
        scores = self._mask_scores(tokens, generator).reshape(B, C, P)
        valid = (None if valid_tokens is None
                 else valid_tokens.reshape(B, C, P).to(torch.bool))
        mask = torch.zeros(B, C, P, device=tokens.device, dtype=torch.bool)

        grouped = set()
        for members in groups:
            members = list(members)
            grouped.update(members)
            s = scores[:, members, :]                              # [B, g, P]
            if valid is None:
                unit_score = s.mean(dim=1)                          # [B, P]
                k = int(P * mask_ratio)
            else:
                vg = valid[:, members, :]
                unit_score = (s.masked_fill(~vg, 0.0).sum(dim=1)
                              / vg.sum(dim=1).clamp_min(1))
                unit_valid = vg.any(dim=1)                          # [B, P]
                unit_score = unit_score.masked_fill(~unit_valid, float("-inf"))
                k = min(int(P * mask_ratio), int(unit_valid.sum(dim=1).min()))
            if k > 0:
                idx = torch.topk(unit_score, k, dim=1).indices
                cols = torch.zeros(B, P, device=tokens.device, dtype=torch.bool)
                cols.scatter_(1, idx, True)
                mask[:, members, :] |= cols.unsqueeze(1)

        rest = [c for c in range(C) if c not in grouped]
        if rest:
            s = scores[:, rest, :].reshape(B, -1)
            if valid is None:
                k = int(s.shape[1] * mask_ratio)
            else:
                v = valid[:, rest, :].reshape(B, -1)
                s = s.masked_fill(~v, float("-inf"))
                k = int(int(v.sum(dim=1).min()) * mask_ratio)
            if k > 0:
                idx = torch.topk(s, k, dim=1).indices
                sub = torch.zeros_like(s, dtype=torch.bool)
                sub.scatter_(1, idx, True)
                mask[:, rest, :] = sub.reshape(B, len(rest), P)

        if valid is not None:
            # A group unit that is valid on some members and padded on others
            # masks only the members that hold a measurement.
            mask &= valid
        return mask.reshape(B, L)
