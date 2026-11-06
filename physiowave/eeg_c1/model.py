"""
Multi-route EEG pretrainer: four wavelet frontends, one Transformer.

The seven corpora do not share an electrode count or a sampling rate, and the
parts of the model that depend on those are exactly the parts that are
duplicated. Everything that does not is shared, because sharing it is the whole
point -- a separate model per corpus would be seven pretrainings, not one.

    per route (4)     SoftGateWaveletDecomp + dynamic ScaleFold
                      the wavelet filters are per-channel, so a frontend is
                      tied to its electrode count and cannot be shared
    per rate  (2)     PatchEmbed, reconstruction head
                      a 0.5 s patch is 128 samples at 256 Hz and 256 at 512,
                      so the conv kernel and the decoder width follow the rate
                      and nothing else -- two of each, not four
    shared    (1)     ChannelEncoder (C1, id), its projection and gate,
                      the 2-D position encoding, the mask token,
                      the RoPE Transformer encoder

The 2-D position encoding is sinusoidal and parameter-free -- it is generated
from (freq_size, time_size) at every call -- so one instance serves all four
token counts with nothing to resize and nothing to learn per route.

Routing is by ``route_id``, which the data carries. There is no learned gate
and this is not a mixture of experts: the recording's montage is known before
the model sees it.

The objective
-------------

::

                             clean EEG x
                             /         \
                    target branch     online branch
                          |                 |
                Wavelet + ScaleFold     choose mask
                          |                 |
                     clean spec        mask raw EEG
                          |                 |
                       DETACH        Wavelet + ScaleFold
                          |                 |
                    target_spec         PatchEmbed
                                            |
                                       mask token
                                            |
                                   shared Transformer
                                      /          \
                              spec decoder    raw decoder
                                   |                |
                               pred_spec         pred_raw

    L = w_spec * L_spec + w_raw * L_raw + lambda_fold * L_foldKL

    L_spec = masked MSE      ( pred_spec, stopgrad(clean folded wavelet) )
    L_raw  = masked SmoothL1 ( pred_raw,  stopgrad(clean preprocessed EEG) )

The canonical weighting is w_spec = w_raw = 0.5: two reconstruction terms of
equal standing rather than an objective and a hint. The numbers themselves live
in configs/pretrain/eeg_c1_moe.yaml and reach every consumer through
``physiowave.eeg_c1.objective.resolve_eeg_c1_objective``; the defaults here are
that module's, not a second copy.

Three properties this shape exists to get, each of which the previous version
lacked:

**The target is stop-gradient.** It is produced by the very frontend being
trained -- learnable wavelet filters and a learnable ScaleFold -- so without
``detach`` the reconstruction term could be reduced by moving the target
instead of by predicting it. Nothing about that is a representation getting
better.

**The corruption happens to the SIGNAL, before the frontend.** The frontend
contains temporal convolution and cross-scale attention, so a patch that is
masked at the token site has already spread into its neighbours' features by
the time the mask token replaces it. Masking the raw samples first closes that
path. The mask token is kept as well: with the samples zeroed, the frontend's
response to a zero-filled stretch would otherwise identify the missing patch
just as reliably.

**There is a second target.** Reconstructing only the folded wavelet
representation lets the frontend choose what is easy to reconstruct. Predicting
the preprocessed EEG as well ties the tokens to the measured signal.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

from channel_embedding import CHANNEL_VOCAB, ChannelEncoder, vocab_sha256
from head_modules import ReconstructionHead
from transformer_modules import PatchEmbed, PositionEmbedding, TransformerEncoder
from wavelet_modules import ScaleFold, SoftGateWaveletDecomp

from .objective import DEFAULT_OBJECTIVE
from .routes import ROUTES, Route


def apply_patch_mask_to_signal(x: torch.Tensor, mask: torch.Tensor,
                               patch_t: int,
                               fill_value: float = 0.0) -> torch.Tensor:
    """Zero exactly the ``(channel, time-patch)`` regions the token mask names.

    ``x`` is ``[B, C, T]`` and ``mask`` is ``[B, C*P]`` in the CHANNEL-MAJOR
    order the tokens use: token ``c*P + p`` is channel ``c``, patch ``p``. So a
    mask is not a time mask shared across electrodes -- Fp1's patch 4 can be
    masked while Fp2's patch 4 is visible, and only Fp1's samples are removed.
    Reshaping to ``[B, C, P, patch_t]`` is that same order read back, not a
    reinterpretation of it.

    Zero is the fill because the preprocessed signal is z-scored per window, so
    zero is its mean rather than an arbitrary value the frontend could learn to
    recognise as "masked". No noise is injected: that would put a second,
    uncontrolled corruption into an objective whose point is a controlled one.
    """
    B, C, T = x.shape
    n_patches = T // patch_t
    if n_patches * patch_t != T:
        raise ValueError(f"T={T} is not a whole number of {patch_t}-wide patches")
    if mask.shape != (B, C * n_patches):
        raise ValueError(
            f"mask {tuple(mask.shape)} does not match [B, C*P] = "
            f"[{B}, {C * n_patches}]")

    patches = x.reshape(B, C, n_patches, patch_t)          # [B, C, P, pt]
    keep = ~mask.reshape(B, C, n_patches, 1)               # [B, C, P, 1]
    if fill_value == 0.0:
        out = patches * keep.to(patches.dtype)
    else:
        out = torch.where(keep, patches,
                          torch.as_tensor(fill_value, dtype=patches.dtype,
                                          device=patches.device))
    return out.reshape(B, C, T)


class WaveletFrontend(nn.Module):
    """One route's ``[B, C, T] -> [B, C, T]``: decompose, then fold the scales.

    The fold reduces the scale axis only. Its output still has one row per
    electrode, which is what lets a shared patcher treat a row as a channel on
    every route.
    """

    def __init__(self, route: Route, max_level: int = 3,
                 wave_kernel_size: int = 16, wavelet_names=None,
                 use_separate_channel: bool = True, wave_init_mode: str = "pad",
                 fold_synthesis: int = 3, fold_gamma: float = 0.1,
                 ffn_drop: float = 0.1):
        super().__init__()
        self.route_id = route.route_id
        self.decomp = SoftGateWaveletDecomp(
            in_channels=route.n_channels,
            max_level=max_level,
            kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            init_mode=wave_init_mode,
            ffn_ratio=4.0, ffn_kernel_size=3, ffn_drop=ffn_drop,
        )
        self.fold = ScaleFold(
            mode="dynamic",
            num_scales=max_level + 1,
            in_channels=route.n_channels,
            patch_len=route.patch_t,
            synthesis_kernel=fold_synthesis,
            gamma_init=fold_gamma,
        )

    def forward(self, x):
        return self.fold(self.decomp(x))

    @property
    def reg_loss(self):
        return self.fold.reg_loss

    @property
    def alpha_mean(self):
        return self.fold.alpha_mean


class MultiRouteEEGPretrainer(nn.Module):
    """Frequency-guided masked wavelet-patch reconstruction across four routes.

    ``forward`` takes one route's batch. A batch never mixes routes: the shapes
    differ, and mixing them would mean padding one route's tokens to another's
    length and then explaining the padding to the attention.
    """

    def __init__(self,
                 # -- the standard model, matching EEG/finetune_sleep.sh -------
                 max_level: int = 3,
                 wave_kernel_size: int = 16,
                 wavelet_names=None,
                 wave_init_mode: str = "pad",
                 use_separate_channel: bool = True,
                 fold_synthesis: int = 3,
                 fold_gamma: float = 0.1,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 # Corrupt the SIGNAL before the frontend rather than only the
                 # tokens after it. False reproduces the older ordering, which
                 # exists for the ablation and not as a recommendation.
                 mask_before_frontend: bool = True,
                 normalize_spec_target: bool = True,
                 dropout: float = 0.1,
                 norm: str = "rmsnorm",
                 ffn: str = "swiglu",
                 qk_norm: bool = True,
                 rope_dim=None,
                 masking_strategy: str = "frequency_guided",
                 importance_ratio: float = 0.6,
                 mask_ratio: float = 0.5,
                 # -- C1, exactly as the ablation defines it -------------------
                 channel_encoding: str = "id",
                 channel_injection: str = "token",
                 channel_embed_dim: int = 64,
                 channel_token_gate_init: float = 0.0,
                 channel_vocab_size: Optional[int] = None,
                 routes: Optional[Dict[str, Route]] = None):
        super().__init__()
        if wavelet_names is None:
            wavelet_names = ["sym4", "sym5", "db6", "sym8", "db8"]
        if channel_encoding not in ("none", "id"):
            raise ValueError(
                f"this pretrainer implements C0 and C1 only; "
                f"channel_encoding={channel_encoding!r} belongs to the ablation's "
                f"geometric rows, which are not part of the settled model.")
        if channel_injection not in ("none", "token"):
            raise ValueError(
                f"C1 injects at the token site only; channel_injection="
                f"{channel_injection!r} would bias the fold's scale logits, "
                f"which is C3/C4 and not what was settled on.")
        if (channel_encoding == "none") != (channel_injection == "none"):
            raise ValueError("channel_encoding and channel_injection must both "
                             "be 'none', or neither.")

        self.routes = dict(ROUTES if routes is None else routes)
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.importance_ratio = importance_ratio
        self.channel_encoding = channel_encoding
        self.channel_injection = channel_injection
        self.max_level = max_level

        # -- per route: the wavelet expert -------------------------------- #
        self.wavelet_frontends = nn.ModuleDict({
            rid: WaveletFrontend(
                route, max_level=max_level, wave_kernel_size=wave_kernel_size,
                wavelet_names=wavelet_names,
                use_separate_channel=use_separate_channel,
                wave_init_mode=wave_init_mode, fold_synthesis=fold_synthesis,
                fold_gamma=fold_gamma,
                # The frontend's cross-scale FFN used to drop 10% whatever the
                # config said, so `dropout: 0` did not mean no dropout.
                ffn_drop=dropout)
            for rid, route in self.routes.items()
        })

        # -- per rate: patcher and decoder -------------------------------- #
        rate_to_patch_t = {r.rate_key: r.patch_t for r in self.routes.values()}
        self.patch_embed_by_rate = nn.ModuleDict({
            rate: PatchEmbed(input_channels=1, patch_size=(1, patch_t),
                             embed_dim=embed_dim)
            for rate, patch_t in rate_to_patch_t.items()
        })
        self.reconstruction_heads = nn.ModuleDict({
            rate: ReconstructionHead(embed_dim=embed_dim, patch_dim=patch_t,
                                     dropout=dropout)
            for rate, patch_t in rate_to_patch_t.items()
        })
        # The second decoder, predicting the preprocessed EEG rather than the
        # folded wavelet representation. Per RATE for the same reason as the
        # first: a 0.5 s patch is 128 samples at 256 Hz and 256 at 512, and
        # nothing else about the head depends on the route. PRETRAINING ONLY --
        # it is not part of what export_eeg_pretrained_encoder.py ships, and
        # the downstream encoder does not grow by it.
        self.raw_reconstruction_heads = nn.ModuleDict({
            rate: ReconstructionHead(embed_dim=embed_dim, patch_dim=patch_t,
                                     dropout=dropout)
            for rate, patch_t in rate_to_patch_t.items()
        })
        self._rate_patch_t = dict(rate_to_patch_t)
        self.mask_before_frontend = bool(mask_before_frontend)
        self.normalize_spec_target = bool(normalize_spec_target)

        # -- shared -------------------------------------------------------- #
        self.channel_encoder = None
        if channel_encoding != "none":
            self.channel_encoder = ChannelEncoder(
                channel_encoding, channel_embed_dim,
                vocab_size=channel_vocab_size or len(CHANNEL_VOCAB))
            self.channel_to_token = nn.Linear(channel_embed_dim, embed_dim)
            # Zero gate, non-zero projection: the backbone is untouched at step
            # 0 and the gate still has gradient, so the branch can start
            # learning. Zeroing both would put a zero on each side of the
            # product and neither would ever move.
            self.channel_token_gate = nn.Parameter(
                torch.tensor(float(channel_token_gate_init)))

        self.pos_embed = PositionEmbedding(embed_dim=embed_dim, pos_type="2d")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.shared_transformer = TransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, rope_dim=rope_dim,
            norm=norm, ffn=ffn, qk_norm=qk_norm)

        self.apply(self._init_weights)
        self.reset_channel_parameters()
        nn.init.normal_(self.mask_token, std=0.02)

    # -- initialisation ---------------------------------------------------- #
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_channel_parameters(self):
        """Re-initialise the channel modules after the generic apply().

        ``apply(_init_weights)`` walks every nn.Linear, including this module's
        projection and the encoder's, and would overwrite what ChannelEncoder
        set for itself. The gate is not touched here: it is meant to start at
        its configured value, which is zero.
        """
        if self.channel_encoder is None:
            return
        self.channel_encoder.reset_channel_parameters()
        with torch.no_grad():
            nn.init.normal_(self.channel_to_token.weight, std=0.02)
            nn.init.zeros_(self.channel_to_token.bias)

    # -- introspection ------------------------------------------------------ #
    def route(self, route_id: str) -> Route:
        if route_id not in self.routes:
            raise KeyError(f"unknown route_id {route_id!r}; "
                           f"have {sorted(self.routes)}")
        return self.routes[route_id]

    def channel_gate_value(self) -> Optional[float]:
        """``tanh(gate)`` -- how much of the channel code reaches the tokens."""
        if self.channel_encoder is None:
            return None
        return float(torch.tanh(self.channel_token_gate).detach())

    def parameter_report(self) -> Dict[str, int]:
        """Parameter counts by part, for the startup banner and the report."""
        def count(mod):
            return sum(p.numel() for p in mod.parameters())

        report = {
            "total": sum(p.numel() for p in self.parameters()),
            "shared_transformer": count(self.shared_transformer),
        }
        for rid, front in self.wavelet_frontends.items():
            report[f"wavelet_frontend.{rid}"] = count(front)
        for rate, pe in self.patch_embed_by_rate.items():
            report[f"patch_embed.{rate}"] = count(pe)
        for rate, head in self.reconstruction_heads.items():
            report[f"reconstruction_head.{rate}"] = count(head)
        for rate, head in self.raw_reconstruction_heads.items():
            report[f"raw_reconstruction_head.{rate}"] = count(head)
        if self.channel_encoder is not None:
            report["channel_encoder"] = count(self.channel_encoder)
            report["channel_to_token"] = count(self.channel_to_token)
            report["channel_token_gate"] = 1
        report["mask_token"] = self.mask_token.numel()
        # What a fine-tuning checkpoint actually carries. Both decoders are
        # pretraining-only, so quoting `total` as the size of the downstream
        # encoder would overstate it.
        report["pretraining_only"] = sum(
            count(h) for h in list(self.reconstruction_heads.values())
            + list(self.raw_reconstruction_heads.values())) + self.mask_token.numel()
        report["downstream_encoder"] = report["total"] - report["pretraining_only"]
        return report

    def vocab_fingerprint(self) -> Dict[str, object]:
        return {"channel_vocab_size": (
                    0 if self.channel_encoder is None
                    else self.channel_encoder.vocab_size),
                "channel_vocab_sha256": vocab_sha256()}

    # -- pieces of the forward pass ----------------------------------------- #
    def _channel_code(self, channel_meta):
        if self.channel_encoder is None:
            return None
        if channel_meta is None:
            raise ValueError(
                "channel_encoding='id' needs channel metadata and got None. "
                "The HDF5 carries channel_ids; the loader must pass them.")
        return self.channel_encoder(channel_meta)

    def _inject_channel_tokens(self, tokens, code, n_channels, n_patches):
        """``[B, C*P, D]`` + one vector per channel, over that channel's P only.

        PatchEmbed emits the sequence channel-major: Conv2d over ``[B, 1, C, T]``
        gives ``[B, D, C, T/p]`` and ``flatten(2)`` walks C then P, so token
        ``c*P + p`` belongs to channel ``c``. Reshaping to ``[B, C, P, D]`` is
        therefore the semantic view, not a reinterpretation.
        """
        B, L, D = tokens.shape
        if n_channels * n_patches != L:
            raise RuntimeError(
                f"token count {L} is not C*P = {n_channels}*{n_patches}")
        if code.dim() == 2:
            code = code.unsqueeze(0)
        if code.shape[-2] != n_channels:
            raise ValueError(
                f"channel code has {code.shape[-2]} channels, the route has "
                f"{n_channels}")
        delta = torch.tanh(self.channel_token_gate) * self.channel_to_token(code)
        tokens = tokens.reshape(B, n_channels, n_patches, D) + delta.unsqueeze(-2)
        return tokens.reshape(B, L, D)

    @staticmethod
    def _zero_padded_channels(x, channel_meta):
        """Force padded channel rows to zero before anything reads them.

        The wavelet frontend is not channel-separable end to end -- its
        ChannelAggregationFFN mixes across the channel axis -- so whatever
        happens to sit in a padded row leaks into every real channel's
        representation. Preprocessing already writes zeros there, which makes
        this a no-op on real data; it is here so the guarantee holds for any
        input, and so that "a padded slot carries no measurement" is enforced by
        the model rather than assumed of the loader.
        """
        if channel_meta is None:
            return x
        valid = channel_meta.get("valid_channel_mask")
        if valid is None:
            return x
        v = valid.to(device=x.device, dtype=torch.bool).reshape(-1)
        if v.numel() != x.shape[-2] or bool(v.all()):
            return x
        return x * v.to(x.dtype).view(1, -1, 1)

    @staticmethod
    def _valid_token_mask(channel_meta, n_channels, n_patches, batch_size,
                          device):
        """``[B, C*P]`` or ``None``. A padded slot is padded for its whole window."""
        if channel_meta is None:
            return None
        valid = channel_meta.get("valid_channel_mask")
        if valid is None:
            return None
        valid = valid.to(device=device, dtype=torch.bool).reshape(-1)
        if valid.numel() != n_channels:
            raise ValueError(
                f"valid_channel_mask has {valid.numel()} entries, the route has "
                f"{n_channels} channels")
        return valid.repeat_interleave(n_patches).unsqueeze(0).expand(
            batch_size, -1)

    @staticmethod
    def patchify(spec, patch_t):
        """``[B, C, T] -> [B, C*P, patch_t]``, channel-major to match the tokens."""
        B, C, T = spec.shape
        P = T // patch_t
        return spec.reshape(B, C, P, patch_t).reshape(B, C * P, patch_t)

    @staticmethod
    def normalize_patches(patches, eps: float = 1e-6):
        """Zero mean, unit variance WITHIN each patch.

        THE TARGET MOVES. It is the folded wavelet output of the frontend being
        trained -- detached, which stops the gradient and does not stop the
        drift. Over seven epochs the target's standard deviation grew 67% while
        the prediction's correlation with it did not move (0.689 -> 0.692), so
        the MSE nearly tripled while the model was tracking exactly as well as
        before. A loss that reports a model getting worse when it is not cannot
        select a checkpoint, cannot stop early, and cannot be compared between
        runs.

        Normalising each patch makes the loss invariant to the frontend's scale
        and to any per-patch offset, so what it measures is the SHAPE the model
        predicted. This is what masked autoencoding does to pixels for the same
        reason. Only the spectrogram target needs it -- the raw target is the
        preprocessed signal, which never passes through the frontend and cannot
        drift.
        """
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True, unbiased=False)
        return (patches - mean) / (var + eps).sqrt()

    @staticmethod
    def unpatchify(patches, n_channels, patch_t):
        """``[B, C*P, patch_t] -> [B, C, T]``. The exact inverse of patchify."""
        B, L, D = patches.shape
        if D != patch_t:
            raise ValueError(f"patch width {D} != {patch_t}")
        P = L // n_channels
        if n_channels * P != L:
            raise ValueError(f"{L} patches is not a whole number per channel")
        return patches.reshape(B, n_channels, P, patch_t).reshape(
            B, n_channels, P * patch_t)

    def _select_mask(self, tokens, mask_ratio, valid_tokens, generator):
        B, L, D = tokens.shape
        if valid_tokens is None:
            budget = int(L * mask_ratio)
        else:
            eligible = int(valid_tokens.sum(dim=1).min().item())
            budget = max(1, int(eligible * mask_ratio)) if eligible else 0
        if budget == 0:
            return torch.zeros(B, L, device=tokens.device, dtype=torch.bool)

        scores = self._mask_scores(tokens, generator)
        if valid_tokens is not None:
            scores = scores.masked_fill(~valid_tokens, float("-inf"))
        idx = torch.topk(scores, budget, dim=1).indices
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _mask_scores(self, tokens, generator):
        """``[B, L]`` -- higher is masked first. Padding is the caller's job.

        Split out of ``_select_mask`` so a subclass can change WHICH tokens a
        score selects (physiowave.ecg_c1 masks the six limb leads together)
        without a second copy of how the score is made. The random draw
        happens here and only here, in the same order as before, so a seeded
        validation mask is the mask it always was.
        """
        B, L, D = tokens.shape
        if generator is None:
            noise = torch.rand(B, L, device=tokens.device)
        else:
            noise = torch.rand(B, L, device=generator.device,
                               generator=generator).to(tokens.device)

        if self.masking_strategy == "frequency_guided":
            # Importance is the summed magnitude spectrum along the sequence --
            # the same statistic the legacy model masks by, kept so the
            # objective is the one the ablation was run under.
            # Along the sequence axis directly, so the input stays contiguous.
            #
            # The filter is for one specific warning from torch.fft's output
            # allocation, which some builds emit on every call -- that is once
            # per step per route for an entire run, and it drowns the log. It is
            # cosmetic: the transform's result is unaffected, which is why this
            # narrows to that one message rather than silencing the call.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="An output with one or more elements was "
                                      "resized", category=UserWarning)
                fft = torch.abs(torch.fft.rfft(tokens, dim=1))  # [B, L//2+1, D]
            importance = fft.sum(dim=2)
            importance = torch.nn.functional.interpolate(
                importance.unsqueeze(1), size=L, mode="linear",
                align_corners=True).squeeze(1)
            scores = (self.importance_ratio * importance
                      + (1 - self.importance_ratio) * noise)
        else:
            scores = noise
        return scores

    # -- forward ------------------------------------------------------------ #
    def encode(self, x, route_id, channel_meta=None):
        """``[B, C, T] -> [B, C*P, D]`` with no masking. What downstream uses.

        THE WINDOW NEED NOT BE THE PRETRAINING WINDOW. P300 epochs are 2 s at
        256 Hz against E64_256's 4 s, and nothing on this path has a length
        baked in: the frontend is convolutional, the patcher is a Conv2d, the
        position encoding is generated from (freq_size, time_size) at every
        call, and the attention is RoPE. So the patch count comes from the
        tokens rather than from the route, which is what it always described.
        The CHANNEL count does not follow the same rule -- the wavelet filters
        are per-electrode -- and a montage with fewer channels goes in the
        route's slots by name, with the rest padded and masked.
        """
        route = self.route(route_id)
        # BEFORE the frontend: ScaleFold blocks the signal into patch_len
        # columns and a width that is not a whole number of them dies inside a
        # synthesis convolution with "Padding size 2 is not supported for 4D
        # input tensor", which says nothing about the window. The patcher would
        # truncate rather than refuse, dropping the tail of every window
        # silently. Both are the same requirement and it belongs here.
        if x.shape[-1] % route.patch_t:
            raise ValueError(
                f"{route_id}: a {x.shape[-1]}-sample window is not a whole "
                f"number of {route.patch_t}-sample patches "
                f"({x.shape[-1] / route.patch_t:.2f} of them). Crop or pad the "
                f"window to a multiple of {route.patch_t}.")
        x = self._zero_padded_channels(x, channel_meta)
        spec = self.wavelet_frontends[route_id](x)
        tokens = self.patch_embed_by_rate[route.rate_key](spec.unsqueeze(1))
        n_patches, rem = divmod(tokens.shape[1], route.n_channels)
        if rem or n_patches < 1:
            raise ValueError(
                f"{tokens.shape[1]} tokens is not a whole number of patches per "
                f"channel on {route_id} ({route.n_channels} channels)")
        code = self._channel_code(channel_meta)
        if code is not None:
            tokens = self._inject_channel_tokens(
                tokens, code, route.n_channels, n_patches)
        tokens = self.pos_embed(tokens, freq_size=route.n_channels,
                                time_size=n_patches)
        return self.shared_transformer(tokens)

    def _build_token_view(self, spec, rate, code, n_channels, n_patches):
        """``[B, C, T] spec -> [B, C*P, D]`` tokens, ready for the encoder.

        Patcher, channel code, position encoding -- the three steps that are
        identical for the clean view and the corrupted one, so that the two
        cannot drift apart by being written twice.
        """
        tokens = self.patch_embed_by_rate[rate](spec.unsqueeze(1))  # [B, C*P, D]
        if code is not None:
            tokens = self._inject_channel_tokens(tokens, code, n_channels,
                                                 n_patches)
        return self.pos_embed(tokens, freq_size=n_channels, time_size=n_patches)

    @staticmethod
    def _check_mask_override(mask_override, batch, n_tokens, valid_tokens,
                             device):
        """An explicit mask, validated. For tests and figures, not training."""
        m = mask_override
        if not torch.is_tensor(m):
            raise TypeError(f"mask_override must be a tensor, got {type(m)}")
        if m.dtype != torch.bool:
            raise TypeError(f"mask_override must be bool, got {m.dtype}")
        if tuple(m.shape) != (batch, n_tokens):
            raise ValueError(
                f"mask_override {tuple(m.shape)} != [B, C*P] = "
                f"[{batch}, {n_tokens}]")
        m = m.to(device)
        if valid_tokens is not None and bool((m & ~valid_tokens).any()):
            # Refused rather than silently cleared: a caller that asked to mask
            # a padded slot has the token order wrong, and quietly dropping
            # those entries would hide it behind a mask ratio that is a little
            # lower than requested.
            bad = int((m & ~valid_tokens).sum())
            raise ValueError(
                f"mask_override selects {bad} padded channel token(s). Padded "
                f"slots hold no measurement and are never reconstruction "
                f"targets.")
        return m

    def forward(self, x, route_id, channel_meta=None, mask_ratio=None,
                mask_generator=None, mask_override=None):
        """One route's masked-reconstruction step, as two views of one window.

        The clean view supplies both detached targets and the mask. The online
        view is the same signal with the masked patches zeroed BEFORE the
        wavelet frontend, so nothing inside a masked patch can reach a visible
        token through the frontend's temporal convolution or its cross-scale
        attention.

        Returns a dict rather than a tuple: there are now two predictions, two
        targets, the validity mask and the fold's KL, and an eight-tuple whose
        order has to be remembered is how those get silently swapped.
        """
        route = self.route(route_id)
        if x.shape[-2] != route.n_channels or x.shape[-1] != route.window_samples:
            raise ValueError(
                f"{route_id} expects [B, {route.n_channels}, "
                f"{route.window_samples}], got {tuple(x.shape)}")
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        n_channels = route.n_channels
        n_patches = route.patches_per_channel
        patch_t = route.patch_t
        rate = route.rate_key
        batch = x.shape[0]

        # -- A. the clean signal -------------------------------------------- #
        # A padded slot holds no measurement; make that true of the tensor
        # before anything -- including the target -- reads it.
        clean_x = self._zero_padded_channels(x, channel_meta)   # [B, C, T]

        # -- B. the target view, and the mask ------------------------------- #
        clean_spec = self.wavelet_frontends[route_id](clean_x)  # [B, C, T]

        # DETACHED. Both targets come from the frontend that is being trained,
        # so a target left attached could be moved toward the prediction rather
        # than the other way round.
        target_spec = self.patchify(clean_spec, patch_t).detach()   # [B, C*P, pt]
        if self.normalize_spec_target:
            target_spec = self.normalize_patches(target_spec)
        target_raw = self.patchify(clean_x, patch_t).detach()       # [B, C*P, pt]

        code = self._channel_code(channel_meta)
        clean_tokens = self._build_token_view(clean_spec, rate, code,
                                              n_channels, n_patches)

        valid_tokens = self._valid_token_mask(
            channel_meta, n_channels, n_patches, batch, clean_tokens.device)

        if mask_override is not None:
            mask = self._check_mask_override(
                mask_override, batch, n_channels * n_patches, valid_tokens,
                clean_tokens.device)
        else:
            # From the clean view, and detached: mask selection is a decision
            # about which tokens to hide, not a differentiable operation.
            mask = self._select_mask(clean_tokens.detach(), mask_ratio,
                                     valid_tokens, mask_generator)

        # -- C. the online view --------------------------------------------- #
        if self.mask_before_frontend:
            masked_x = apply_patch_mask_to_signal(clean_x, mask, patch_t)
            online_spec = self.wavelet_frontends[route_id](masked_x)
            online_tokens = self._build_token_view(online_spec, rate, code,
                                                   n_channels, n_patches)
        else:
            # The ablation ordering: the frontend sees the whole window and only
            # the tokens are replaced. One frontend pass, so no second one is
            # paid for a view that would be identical.
            online_spec = clean_spec
            online_tokens = clean_tokens

        # The fold's KL and alpha are module state overwritten by each call, so
        # they are read HERE -- after the last frontend call in this forward --
        # and belong to the online pass. Reading them earlier would report the
        # clean pass's statistics for a step trained on the corrupted one.
        fold_reg = self.wavelet_frontends[route_id].reg_loss
        fold_alpha = self.wavelet_frontends[route_id].alpha_mean

        # -- D. mask token, encoder, two decoders --------------------------- #
        # Kept even though the samples are already gone: the frontend's response
        # to a zero-filled stretch is itself a signature, and without the mask
        # token the encoder could find the missing patch by that alone.
        masked_tokens = online_tokens.clone()
        masked_tokens[mask] = self.mask_token.expand_as(online_tokens)[mask]
        encoded = self.shared_transformer(masked_tokens)

        pred_spec = self.reconstruction_heads[rate](encoded)      # [B, C*P, pt]
        pred_raw = self.raw_reconstruction_heads[rate](encoded)   # [B, C*P, pt]

        with torch.no_grad():
            # Diagnostic, never a loss: if this is ~0 under
            # mask_before_frontend, the corruption is not reaching the frontend.
            delta = (clean_spec - online_spec).abs().mean() \
                if self.mask_before_frontend else torch.zeros(
                    (), device=clean_spec.device)

        return {
            "pred_spec": pred_spec,
            "target_spec": target_spec,
            "pred_raw": pred_raw,
            "target_raw": target_raw,
            "mask": mask,
            "valid_tokens": valid_tokens,
            "clean_spec": clean_spec.detach(),
            "online_spec": online_spec,
            "clean_online_spec_delta": delta,
            "fold_reg": fold_reg,
            "fold_alpha": fold_alpha,
            "mask_before_frontend": self.mask_before_frontend,
            "normalize_spec_target": self.normalize_spec_target,
            "route_id": route_id,
            # -- compatibility aliases -------------------------------------- #
            # scripts/visualize_eeg_pretraining.py and the older tests read
            # these three names. They point at the spec objective, which is what
            # they meant before there was a second one.
            "pred": pred_spec,
            "target": target_spec,
            "spec": clean_spec.detach(),
            "tokens": online_tokens,
        }


#: Denominator floor for NMSE. Small enough that a real target's energy is
#: never what it clamps; present so that an all-zero target gives a finite
#: number instead of an inf that poisons every mean downstream.
NMSE_EPS = 1e-12


def _masked_nmse(pred, target, eps: float = NMSE_EPS):
    """``MSE(pred, target) / MSE(0, target)`` over the elements handed in.

    Dimensionless, and anchored to a baseline anyone can check: 1.0 is what
    predicting zero scores, below 1.0 is better than that, above 1.0 is worse.
    The spec target is normalised per patch, so its denominator sits near 1 --
    but it is computed rather than assumed, because a patch of masked zeros is
    normalised to zeros and its energy is not 1.
    """
    num = torch.nn.functional.mse_loss(pred, target)
    den = torch.nn.functional.mse_loss(torch.zeros_like(target), target)
    return num / den.clamp_min(eps)


def _masked_corr(pred, target):
    """Pearson r over the elements handed in; 0 when either side is constant.

    Dimensionless, so it survives any change to the target's scale -- which is
    the whole reason the spec target is normalised, and the reason this is the
    number to compare the two heads on rather than their losses.
    """
    pc = pred - pred.mean()
    tc = target - target.mean()
    denom = pc.norm() * tc.norm()
    if float(denom) <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (pc @ tc) / denom


def masked_reconstruction_loss(out, spec_weight: Optional[float] = None,
                               raw_weight: Optional[float] = None,
                               fold_kl: Optional[float] = None,
                               raw_beta: Optional[float] = None):
    """``L = w_spec*MSE(spec) + w_raw*SmoothL1(raw) + fold_kl*KL``, and metrics.

    The weights default to the canonical objective in
    ``physiowave.eeg_c1.objective.DEFAULT_OBJECTIVE`` (0.5 / 0.5 / 1e-3), which
    is the same dict the config resolver falls back to. There is no second copy
    of these numbers here to drift from the config.

    Both terms are computed over masked VALID tokens only. Padded channel slots
    are never masked -- ``_select_mask`` excludes them and
    ``_check_mask_override`` refuses them -- and the intersection is taken here
    as well, so the guarantee is local to this function rather than inherited
    from two callers.

    SmoothL1 for the raw term rather than MSE. The preprocessed EEG is clipped
    but still holds clinical events an order of magnitude above the background,
    and under a squared penalty a handful of those would set the gradient for
    the whole term.

    THE TWO LOSSES ARE NOT COMPARABLE IN MAGNITUDE. One is an MSE on per-patch
    normalised wavelet coefficients, the other a SmoothL1 on z-scored volts;
    the raw number is smaller for reasons of units and of the knee at beta, not
    because that head is doing better. ``masked_{spec,raw}_corr`` and
    ``masked_{spec,raw}_nmse`` are dimensionless and are the comparison.

    The old single-argument call ``masked_reconstruction_loss(out, 1e-3)`` is
    NOT compatible: the second positional is now spec_weight. Callers pass
    fold_kl by name.
    """
    spec_weight = (float(DEFAULT_OBJECTIVE["spec_weight"])
                   if spec_weight is None else float(spec_weight))
    raw_weight = (float(DEFAULT_OBJECTIVE["raw_weight"])
                  if raw_weight is None else float(raw_weight))
    fold_kl = (float(DEFAULT_OBJECTIVE["fold_kl"])
               if fold_kl is None else float(fold_kl))
    raw_beta = (float(DEFAULT_OBJECTIVE["raw_beta"])
                if raw_beta is None else float(raw_beta))

    pred_spec, target_spec = out["pred_spec"], out["target_spec"]
    pred_raw, target_raw = out["pred_raw"], out["target_raw"]
    mask = out["mask"]
    valid = out.get("valid_tokens")
    if valid is not None:
        mask = mask & valid

    n = int(mask.sum())
    reg = out.get("fold_reg")

    if n == 0:
        # Differentiable zero, not a Python 0.0: the caller calls .backward()
        # on this and an epoch whose first step happened to mask nothing must
        # not be the one that raises.
        zero = pred_spec.sum() * 0.0 + pred_raw.sum() * 0.0
        return zero, {
            "loss_total": 0.0, "loss_masked_spec_mse": 0.0,
            "loss_masked_raw_smoothl1": 0.0, "loss_fold_kl": 0.0,
            "masked_spec_mae": 0.0, "masked_spec_rmse": 0.0,
            "masked_raw_mae": 0.0, "masked_raw_rmse": 0.0,
            "masked_spec_corr": 0.0, "masked_raw_corr": 0.0,
            "masked_spec_nmse": 0.0, "masked_raw_nmse": 0.0,
            "masked_corr": 0.0, "actual_mask_ratio": 0.0,
            "clean_online_spec_delta": 0.0,
            # compatibility aliases, as below
            "loss_masked_mse": 0.0, "masked_mae": 0.0, "masked_rmse": 0.0,
        }

    sel = mask.unsqueeze(-1).expand_as(pred_spec)
    p_spec, t_spec = pred_spec[sel], target_spec[sel]
    p_raw, t_raw = pred_raw[sel], target_raw[sel]

    loss_spec = torch.nn.functional.mse_loss(p_spec, t_spec)
    loss_raw = torch.nn.functional.smooth_l1_loss(p_raw, t_raw, beta=raw_beta)

    kl = reg if reg is not None else pred_spec.sum() * 0.0
    # Explicit and unaveraged. Predicting one head from a blend of the two, or
    # folding them into a single term before weighting, would make neither
    # loss the thing its logged value names.
    total = spec_weight * loss_spec + raw_weight * loss_raw + fold_kl * kl

    with torch.no_grad():
        spec_mae = (p_spec - t_spec).abs().mean()
        spec_rmse = loss_spec.sqrt()
        raw_mae = (p_raw - t_raw).abs().mean()
        raw_mse = torch.nn.functional.mse_loss(p_raw, t_raw)
        raw_rmse = raw_mse.sqrt()
        spec_corr = _masked_corr(p_spec.float(), t_spec.float())
        raw_corr = _masked_corr(p_raw.float(), t_raw.float())
        spec_nmse = _masked_nmse(p_spec.float(), t_spec.float())
        raw_nmse = _masked_nmse(p_raw.float(), t_raw.float())
        denom_tokens = int(valid.sum()) if valid is not None else mask.numel()
        ratio = n / max(1, denom_tokens)
        delta = out.get("clean_online_spec_delta")

    # float(), not a "make it finite" wrapper. Every value below is finite by
    # construction for finite inputs -- NMSE clamps its denominator and corr
    # guards a zero one -- so a NaN here means the MODEL produced one, and
    # coercing it to 0.0 would report a diverged run as a perfect loss and let
    # it win checkpoint selection. The accumulator drops non-finite values and
    # the best-score bars refuse them; both need to see the NaN to do that.
    return total, {
        "loss_total": float(total.detach()),
        "loss_masked_spec_mse": float(loss_spec.detach()),
        "loss_masked_raw_smoothl1": float(loss_raw.detach()),
        "loss_fold_kl": float(kl.detach() if torch.is_tensor(kl) else kl),
        "masked_spec_mae": float(spec_mae),
        "masked_spec_rmse": float(spec_rmse),
        "masked_raw_mae": float(raw_mae),
        "masked_raw_rmse": float(raw_rmse),
        # Dimensionless, and the only honest way to say which head is doing
        # better: r is scale-free, and NMSE is measured against the
        # zero-prediction baseline on that head's own target.
        "masked_spec_corr": float(spec_corr),
        "masked_raw_corr": float(raw_corr),
        "masked_spec_nmse": float(spec_nmse),
        "masked_raw_nmse": float(raw_nmse),
        "actual_mask_ratio": float(ratio),
        "clean_online_spec_delta": float(delta) if delta is not None else 0.0,
        # Compatibility aliases. The history JSONL and the figure captions
        # were written against these names when the spec term was the whole
        # objective, so they still resolve to it. New code uses the explicit
        # spec/raw names -- `loss_masked_mse` does not say which head it is.
        "loss_masked_mse": float(loss_spec.detach()),
        "masked_corr": float(spec_corr),
        "masked_mae": float(spec_mae),
        "masked_rmse": float(spec_rmse),
    }
