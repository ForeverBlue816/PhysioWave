"""
Fine-tuning a C1 pretrained encoder on a task with its own montage.

WHAT TRANSFERS IS DECIDED BY THE ARCHITECTURE, NOT CHOSEN
---------------------------------------------------------

The pretrainer's parts are bound to different things, and that binding is what
says which of them a downstream task can reuse:

    wavelet frontend    per ELECTRODE COUNT -- the filters are per-channel, so
                        a frontend built for 64 electrodes is not a frontend
                        for 2. NOT transferable.
    patch embedding     per PATCH LENGTH -- 128 samples at 256 Hz, 50 at 100.
                        NOT transferable across sampling rates.
    channel encoder     per CHANNEL NAME, against a vocabulary shared by every
                        corpus. Transferable, and the point of C1.
    channel projection  }
    channel gate        } the same, and the gate is what scales the whole
                        contribution -- an encoder loaded without it has the
                        C1 mechanism switched off.
    position encoding   generated per call from (channels, patches). No
                        parameters, so nothing to transfer.
    transformer         bound to nothing. THE thing pretraining produced.

So a task whose montage and rate match a pretraining route (P300 is 62
channels at 256 Hz, all of them E64_256 slots, at that route's rate) reuses
everything,
and a task that does not (Sleep-EDF is two bipolar derivations at 100 Hz)
builds a fresh frontend and patcher and reuses the rest. Both are the same
class; the difference is which keys the checkpoint supplies.

THE COMPARISON THAT MEANS SOMETHING is this model with pretrained weights
against this model without them -- same architecture, same recipe, same split.
``pretrained=None`` is that control and is one flag away, because a downstream
number with nothing to compare it to says only that the architecture works.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from channel_embedding import (CHANNEL_VOCAB, UNK_ID, ChannelEncoder, PAD_ID,
                               channel_ids_for, vocab_payload)
from transformer_modules import PatchEmbed, PositionEmbedding, TransformerEncoder

from .heads import AdaptiveSpatialFilter, AttentiveStatsPool, TemporalHead
from .model import WaveletFrontend
from .routes import ROUTES, Route


#: The pretraining config's list, and not the module default. coif3 is 18 taps
#: and does not fit kernel_size 16; bior4.4 is biorthogonal and is not
#: power-complementary, so neither initialises a filter bank. A fresh
#: downstream frontend that started from a different family than the encoder it
#: is bolted to would also be one more difference than the experiment needs.
DEFAULT_WAVELETS = ("sym4", "sym5", "db6", "sym8", "db8")

#: Keys an exported encoder may supply, and the submodule each belongs to.
#: `channel_token_gate` is a bare parameter rather than a submodule, which is
#: how an allowlist of module prefixes came to drop it.
TRANSFERABLE = ("channel_encoder.", "channel_to_token.", "channel_token_gate",
                "shared_transformer.")
#: Only when the montage and rate match the route the checkpoint was exported
#: for. Refused otherwise, loudly, rather than reshaped.
ROUTE_BOUND = ("wavelet_frontend.", "patch_embed.")

#: What ``--freeze-encoder`` leaves trainable. The spatial filter is in here
#: because it is in EEGPT's probe too: their PhysioP300 optimiser is given
#: ``[self.chan_scale] + linear_probe1 + linear_probe2``, and the ablation in
#: their Table 8 is between having that filter and not. It is an ADAPTER
#: BETWEEN TWO MONTAGES, not part of the representation -- freezing it would
#: measure the encoder's response to the wrong electrode gains rather than the
#: encoder.
TRAINABLE_WHEN_FROZEN = ("head", "spatial_filter", "new_channel_embed")

#: What ``freeze_encoder`` freezes.
#:
#:   encoder     everything except TRAINABLE_WHEN_FROZEN. The original meaning,
#:               and the default, so existing probes run as they always have.
#:   pretrained  exactly what came from the checkpoint: the TRANSFERABLE set,
#:               plus ROUTE_BOUND when the montage IS the route and those were
#:               loaded too. A frontend and patcher built fresh for this
#:               montage stay trainable.
#:
#: The difference matters as soon as the frontend is not pretrained, which with
#: no route_id is always. Under `encoder` the fresh wavelet frontend and patcher
#: are frozen at their initialisation, so the pretrained transformer is handed
#: tokens from a random projection it never saw, and the probe measures that
#: projection as much as the representation. EEGPT's probe freezes an encoder
#: that is pretrained end to end, patch embedding included; `pretrained` is the
#: analogue of that when the input layers cannot come from pretraining.
FREEZE_SCOPES = ("encoder", "pretrained")


def _slots_for(route: Route, channel_names: Sequence[str],
               aliases: Optional[Dict[str, str]] = None,
               normalize=None) -> torch.Tensor:
    """``[route.n_channels]`` index into the incoming montage, -1 where absent.

    Matched by NAME through the same normalisation the vocabulary uses, so
    "FP1", "Fp1" and "EEG Fp1-REF" are one electrode. A downstream channel that
    is not one of the route's slots is refused rather than dropped: silently
    discarding a measured electrode is how a montage mismatch becomes a quiet
    accuracy loss.

    ``aliases`` reads one electrode as another, ``{"P9": "TP9"}``. It exists
    because erpbci records P9/P10 where E64_256 has TP9/TP10 -- one position
    apart in the same inferior lateral chain, close but not the same site. That
    substitution is a modelling assumption, so it is something a config states
    out loud and never a default: an alias that were applied automatically
    would put a P9 signal through a filter bank trained on TP9 with nothing
    anywhere recording that it happened.
    """
    if normalize is None:
        from channel_embedding import normalize_channel_name as normalize
    normalize_channel_name = normalize

    want = {normalize_channel_name(s): i for i, s in enumerate(route.slots)}
    alias = {normalize_channel_name(k): normalize_channel_name(v)
             for k, v in (aliases or {}).items()}
    bad = {k: v for k, v in alias.items() if v not in want}
    if bad:
        raise ValueError(
            f"slot_aliases maps onto {sorted(bad.values())}, which are not "
            f"slots of {route.route_id}. An alias has to name a slot that "
            f"exists; its whole purpose is to reach one.")
    idx = torch.full((route.n_channels,), -1, dtype=torch.long)
    unplaced = []
    for j, name in enumerate(channel_names):
        key = normalize_channel_name(name)
        slot = want.get(alias.get(key, key))
        if slot is None:
            unplaced.append(name)
            continue
        idx[slot] = j
    if unplaced:
        raise ValueError(
            f"{len(unplaced)} channel(s) are not slots of {route.route_id}: "
            f"{unplaced[:8]}. Either alias them onto slots that exist "
            f"(model.eeg_c1.slot_aliases) or build the model without route_id "
            f"to give this montage its own frontend instead.")
    return idx


def downstream_route(in_channels: int, sampling_rate: float,
                     window_samples: int, patch_samples: int,
                     channel_names: Sequence[str] = ()) -> Route:
    """A Route for a montage that is not one of the four pretraining routes.

    Route is a description, not a registry entry: the model reads n_channels,
    patch_t and rate_key off it and nothing looks the id up. Sleep-EDF is
    2 x 3000 at 100 Hz with a 50-sample patch; P300 is 62 x 512 at 256 Hz with
    128. Neither is a route and both are describable.
    """
    if window_samples % patch_samples:
        raise ValueError(
            f"a {window_samples}-sample window is not a whole number of "
            f"{patch_samples}-sample patches "
            f"({window_samples / patch_samples:.2f} of them)")
    return Route(route_id="downstream", n_channels=int(in_channels),
                 sampling_rate=int(sampling_rate),
                 window_seconds=window_samples / float(sampling_rate),
                 patch_seconds=patch_samples / float(sampling_rate),
                 slots=tuple(channel_names))


class EEGC1Downstream(nn.Module):
    """``[B, C, T] -> {"logits": [B, K]}`` on a pretrained shared transformer.

    The modality lives in five class attributes -- routes, channel-name
    resolution, the vocabulary and its hash, the default wavelets -- so the ECG
    model (physiowave.ecg_c1.downstream) is this class with those swapped and
    nothing else, and an EEG checkpoint cannot be loaded into it or vice versa.
    """

    MODALITY = "eeg"
    DEFAULT_WAVELETS = DEFAULT_WAVELETS
    ROUTES_TABLE = ROUTES
    VOCAB_SIZE = len(CHANNEL_VOCAB)

    @staticmethod
    def ids_for(names):
        return channel_ids_for(names)

    @staticmethod
    def normalize_name(name):
        from channel_embedding import normalize_channel_name
        return normalize_channel_name(name)

    @staticmethod
    def vocab_payload():
        return vocab_payload()

    def __init__(self, in_channels: int, window_samples: int,
                 sampling_rate: float, patch_samples: int, num_classes: int,
                 channel_names: Optional[Sequence[str]] = None,
                 route_id: Optional[str] = None,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 norm: str = "rmsnorm", ffn: str = "swiglu",
                 qk_norm: bool = True, rope_dim: int = 2,
                 max_level: int = 3, wave_kernel_size: int = 16,
                 wavelet_names=None, wave_init_mode: str = "pad",
                 use_separate_channel: bool = True, fold_synthesis: int = 3,
                 fold_gamma: float = 0.1,
                 channel_encoding: str = "id", channel_embed_dim: int = 64,
                 channel_token_gate_init: float = 0.0,
                 channel_vocab_size: Optional[int] = None,
                 pool: str = "mean", head_dropout: float = 0.0,
                 probe_dim: int = 16, head_depth: int = 4, head_heads: int = 4,
                 pool_heads: int = 4,
                 head_max_norm: float = 0.0, channel_pool: str = "mean",
                 spatial_filter: str = "none",
                 spatial_channels: Optional[Sequence[str]] = None,
                 spatial_max_norm: float = 1.0,
                 slot_aliases: Optional[Dict[str, str]] = None,
                 freeze_encoder: bool = False,
                 freeze_scope: str = "encoder",
                 new_channel_rows: bool = True):
        super().__init__()
        if wavelet_names is None:
            wavelet_names = self.DEFAULT_WAVELETS
        # THE SPATIAL FILTER COMES FIRST, because with `mix` it decides what
        # montage the rest of the model is built for: the frontend's electrode
        # count, the route slots, and -- the part that matters -- which channel
        # vocabulary rows this dataset gets to use.
        self.raw_channels = int(in_channels)
        self.spatial_filter = None
        if spatial_filter != "none":
            self.spatial_filter = AdaptiveSpatialFilter(
                spatial_filter, self.raw_channels, spatial_channels,
                max_norm=spatial_max_norm)
            if spatial_filter == "mix":
                channel_names = list(self.spatial_filter.out_names)
                in_channels = self.spatial_filter.out_channels
        self.model_channel_names = list(channel_names) if channel_names else None
        # ON A ROUTE, IF IT FITS. P300 is 62 electrodes at 256 Hz and every one
        # of them is one of E64_256's 64 slots, so putting them in those slots
        # and masking the two that are absent (TP9, TP10 -- the EDF records
        # P9/P10 instead) makes the pretrained FRONTEND and PATCHER
        # transferable too, not just the transformer. That is a much stronger
        # transfer than a fresh 62-channel frontend, and it is what
        # valid_channel_mask exists for -- pretraining itself runs with padded
        # slots on every corpus that does not fill its route.
        if route_id is not None:
            self.route = self.ROUTES_TABLE[route_id]
            if self.route.patch_t != patch_samples:
                raise ValueError(
                    f"{route_id} patches {self.route.patch_t} samples and this "
                    f"asks for {patch_samples}; a route is its patch length")
            if int(self.route.sampling_rate) != int(sampling_rate):
                raise ValueError(
                    f"{route_id} is {self.route.sampling_rate} Hz and this data "
                    f"is {sampling_rate} Hz")
            if not channel_names:
                raise ValueError(f"route_id={route_id} needs channel_names to "
                                 f"place the montage in its slots")
            self.slot_aliases = dict(slot_aliases or {})
            self.register_buffer("slot_index",
                                 _slots_for(self.route, channel_names,
                                            self.slot_aliases,
                                            normalize=self.normalize_name),
                                 persistent=False)
            self.in_channels = int(in_channels)
        else:
            self.route = downstream_route(in_channels, sampling_rate,
                                          window_samples, patch_samples,
                                          channel_names or ())
            self.in_channels = int(in_channels)
            # Registered either way, so `self.slot_index is None` is the test
            # for "this montage is not on a route" and not for "the attribute
            # happens not to exist".
            self.slot_aliases = {}
            if slot_aliases:
                raise ValueError(
                    "slot_aliases only means something on a route; without one "
                    "this montage gets its own frontend and every channel it "
                    "has is already its own slot.")
            self.register_buffer("slot_index", None, persistent=False)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.pool = pool
        if pool not in ("mean", "max", "cls_free_mean", "stat", "time", "attn"):
            raise ValueError(
                f"pool must be mean, max, stat, time or attn, got {pool!r}")
        # ``time`` keeps the time axis and only averages over channels, which is
        # what an ERP head needs and what EEGPT's linear probe does: theirs is a
        # per-position Linear(2048, 16), a flatten over the 15 positions, then a
        # Linear(240, 2). A mean over channels AND time hands the head one
        # vector per window, and a P300 is a deflection 250-500 ms wide --
        # averaging over four 500 ms patches averages it away before the head
        # ever sees it.
        self.n_patches = int(window_samples) // int(patch_samples)

        self.wavelet_frontend = WaveletFrontend(
            self.route, max_level=max_level, wave_kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            wave_init_mode=wave_init_mode, fold_synthesis=fold_synthesis,
            fold_gamma=fold_gamma, ffn_drop=dropout)
        self.patch_embed = PatchEmbed(input_channels=1,
                                      patch_size=(1, self.route.patch_t),
                                      embed_dim=embed_dim)

        self.channel_encoder = None
        if channel_encoding != "none":
            self.channel_encoder = ChannelEncoder(
                channel_encoding, channel_embed_dim,
                vocab_size=channel_vocab_size or self.VOCAB_SIZE)
            self.channel_to_token = nn.Linear(channel_embed_dim, embed_dim)
            self.channel_token_gate = nn.Parameter(
                torch.tensor(float(channel_token_gate_init)))
        # ROWS FOR NAMES THE VOCABULARY DOES NOT HAVE. A Myo electrode, a
        # GRABMyo forearm position, a behind-the-ear EEG channel all resolve
        # to <unk>, and a montage whose every channel is <unk> has switched
        # the channel-identity path off: one shared vector, channels told
        # apart by position alone. Each unknown name gets a row of its own
        # here, initialised to the pretrained <unk> row so the first forward
        # is exactly what it was, and trained as a montage adapter -- under a
        # frozen encoder too, like the fresh frontend and the spatial filter.
        # Only off a route: a route's slots are vocabulary names.
        self.new_channel_embed = None
        self.new_channel_names: List[str] = []
        self.register_buffer("new_channel_index", None, persistent=False)
        if (new_channel_rows and self.channel_encoder is not None
                and self.slot_index is None and self.model_channel_names):
            ids, _ = self.ids_for(self.model_channel_names)
            pos = [i for i, k in enumerate(ids) if k == UNK_ID]
            if pos:
                self.new_channel_embed = nn.Embedding(len(pos), channel_embed_dim)
                index = torch.full((len(ids),), -1, dtype=torch.long)
                index[pos] = torch.arange(len(pos))
                self.new_channel_index = index
                self.new_channel_names = [self.model_channel_names[i] for i in pos]
                self._init_new_rows()
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim, pos_type="2d")
        self.shared_transformer = TransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, rope_dim=rope_dim,
            norm=norm, ffn=ffn, qk_norm=qk_norm)

        if pool == "stat":
            # Attention-weighted mean and standard deviation over the whole
            # token grid. The queries start at zero, so the first forward is
            # the mean head plus a standard deviation -- a strict superset of
            # what this replaces.
            self.head_norm = None
            self.head_drop = nn.Dropout(head_dropout)
            # `head_pool`, not `stat_pool`: TRAINABLE_WHEN_FROZEN is a prefix
            # tuple, and the pooling IS part of the head -- its queries are
            # most of what a probe has to learn. Named outside the convention
            # it was frozen along with the encoder, and the probe trained a
            # bare Linear on a fixed mean.
            self.head_pool = AttentiveStatsPool(embed_dim, heads=pool_heads)
            self.head = nn.Linear(self.head_pool.out_features, self.num_classes)
        elif pool in ("time", "attn"):
            # `time` flattens the projected positions (EEGPT's PhysioP300 head);
            # `attn` reads them with a cls token and a small transformer (their
            # Sleep-EDFx head). Both keep the time axis, which is the property
            # that matters and the one a mean does not have.
            self.head_norm = self.head_drop = None
            self.head = TemporalHead(
                "flatten" if pool == "time" else "attn",
                embed_dim=embed_dim, n_positions=self.n_patches,
                num_classes=self.num_classes, probe_dim=probe_dim,
                depth=head_depth, num_heads=head_heads, dropout=head_dropout,
                max_norm=head_max_norm, channel_pool=channel_pool,
                norm=norm, ffn=ffn, qk_norm=qk_norm)
        else:
            self.head_norm = nn.LayerNorm(embed_dim)
            self.head_drop = nn.Dropout(head_dropout)
            self.head = nn.Linear(embed_dim, self.num_classes)

        if freeze_scope not in FREEZE_SCOPES:
            raise ValueError(f"freeze_scope must be one of {FREEZE_SCOPES}, "
                             f"got {freeze_scope!r}")
        self._frozen = bool(freeze_encoder)
        self.freeze_scope = freeze_scope
        # Under `pretrained`, what CAN come from a checkpoint whatever the
        # route. load_pretrained adds ROUTE_BOUND if it actually loads those.
        self._frozen_prefixes = TRANSFERABLE
        # What a checkpoint actually supplied, frozen or not: fine-tuning's
        # encoder_lr_scale and layer decay apply to exactly these.
        self.pretrained_prefixes = TRANSFERABLE
        self._apply_freeze()

    def _is_frozen_name(self, name: str) -> bool:
        if not self._frozen:
            return False
        if self.freeze_scope == "encoder":
            return not name.startswith(TRAINABLE_WHEN_FROZEN)
        return name.startswith(self._frozen_prefixes)

    def _apply_freeze(self) -> None:
        for name, p in self.named_parameters():
            p.requires_grad_(not self._is_frozen_name(name))

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode even inside ``model.train()``.

        A linear probe measures the representation, so the features it is handed
        have to be the ones the encoder actually produces. Left in train mode a
        frozen encoder still applies its dropout, so every epoch shows the head a
        different corrupted view of a fixed representation -- noise the head
        cannot fit and the encoder cannot adapt to, which reads as a probe that
        will not converge. ``--set model.dropout=0.0`` worked around it and had
        to be remembered on every probe command.
        """
        super().train(mode)
        if mode and self._frozen:
            for name, module in self.named_children():
                # name + "." so a child is matched as a module prefix -- the
                # TRANSFERABLE entries end in "." and a bare child name would
                # match none of them.
                if self._is_frozen_name(name + "."):
                    module.eval()
        return self

    # -- channel metadata ---------------------------------------------------- #
    def _init_new_rows(self) -> None:
        """Every new row starts as the <unk> row the encoder currently holds."""
        if self.new_channel_embed is None:
            return
        with torch.no_grad():
            unk = self.channel_encoder.id_embed.weight[UNK_ID]
            self.new_channel_embed.weight.copy_(
                unk.unsqueeze(0).expand_as(self.new_channel_embed.weight))

    def _with_new_rows(self, cm):
        """Attach the new rows to the channel metadata the encoder reads."""
        if cm is None or self.new_channel_embed is None:
            return cm
        idx = self.new_channel_index.to(cm["channel_ids"].device)
        rows = self.new_channel_embed.weight[idx.clamp_min(0)]
        return {**cm, "code_override": rows, "code_override_mask": idx >= 0}

    def _meta_tensors(self, meta, device) -> Optional[Dict[str, torch.Tensor]]:
        """``ChannelMeta`` or the C1 dict, resolved to ids against THIS vocab.

        Names, not ids, are what a downstream file and a pretraining corpus can
        agree on: an id means whichever electrode held that row when it was
        learned, so resolving names here -- under the vocabulary the weights
        were checked against on load -- is what keeps Fp1 meaning Fp1.
        """
        if self.channel_encoder is None:
            return None
        if meta is None:
            raise ValueError(
                "channel_encoding='id' needs channel metadata and got None. "
                "The downstream HDF5 carries channel_names; pass them.")
        if isinstance(meta, dict) and "channel_ids" in meta:
            ids = torch.as_tensor(meta["channel_ids"]).long()
            valid = meta.get("valid_channel_mask")
        else:
            names = list(getattr(meta, "channel_names", []) or [])
            if not names:
                raise ValueError("channel metadata carries no channel_names")
            ids_list, _ = self.ids_for(names)
            ids = torch.as_tensor(ids_list, dtype=torch.long)
            valid = getattr(meta, "channel_mask", None)
        if valid is None:
            valid = ids != PAD_ID
        valid = torch.as_tensor(valid).bool().reshape(-1)
        if ids.numel() != self.in_channels:
            raise ValueError(
                f"{ids.numel()} channel ids for {self.in_channels} channels")
        return {"channel_ids": ids.to(device),
                "valid_channel_mask": valid.to(device)}

    def _model_name_meta(self, device) -> Optional[Dict[str, torch.Tensor]]:
        """Channel metadata for the montage the MODEL sees, not the recorded one.

        With a ``mix`` spatial filter the two are different objects: the file
        carries two bipolar derivations and the model is looking at thirteen
        named 10-20 electrodes that the mix produced. Resolving the incoming
        names here would attach Fpz-Cz's vocabulary row to a virtual Fz, so the
        model's own output names are what is looked up, and every one of them
        is valid by construction.
        """
        if self.channel_encoder is None:
            return None
        if not self.model_channel_names:
            raise ValueError("a mix spatial filter needs output electrode names")
        ids_list, _ = self.ids_for(self.model_channel_names)
        ids = torch.as_tensor(ids_list, dtype=torch.long, device=device)
        return {"channel_ids": ids,
                "valid_channel_mask": torch.ones_like(ids, dtype=torch.bool)}

    def _to_slots(self, x: torch.Tensor) -> torch.Tensor:
        """Place a montage in the route's slots; absent slots stay zero.

        Zero, not an interpolation: pretraining ran with padded slots zeroed
        and the frontend's channel-mixing FFN learned against that, so a
        different filler is a different input distribution.
        """
        if self.slot_index is None:
            return x
        B, _, T = x.shape
        out = x.new_zeros(B, self.route.n_channels, T)
        have = self.slot_index >= 0
        out[:, have] = x[:, self.slot_index[have]]
        return out

    def _slot_meta(self, cm):
        """The channel ids and mask AS THE ROUTE SEES THEM, after placement."""
        if self.slot_index is None or cm is None:
            return cm
        ids, _ = self.ids_for(self.route.slots)
        ids = torch.as_tensor(ids, dtype=torch.long, device=cm["channel_ids"].device)
        have = self.slot_index >= 0
        valid = have.to(cm["valid_channel_mask"].device).clone()
        valid[have] &= cm["valid_channel_mask"][self.slot_index[have]]
        ids = torch.where(valid, ids, torch.full_like(ids, PAD_ID))
        return {"channel_ids": ids, "valid_channel_mask": valid}

    # -- forward ------------------------------------------------------------- #
    def encode(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        r = self.route
        if x.shape[-1] % r.patch_t:
            raise ValueError(
                f"a {x.shape[-1]}-sample window is not a whole number of "
                f"{r.patch_t}-sample patches")
        if x.shape[-2] != self.raw_channels:
            raise ValueError(
                f"{x.shape[-2]} channels, the model was built for "
                f"{self.raw_channels}"
                + (f" (before its {self.spatial_filter.kind} spatial filter)"
                   if self.spatial_filter is not None else ""))
        if self.spatial_filter is not None and self.spatial_filter.kind == "mix":
            # A mix replaces the montage, so the recorded channel metadata
            # describes the filter's INPUT and has nothing to say about its
            # output. The mask would be meaningless here too: every output
            # electrode is a combination of all the inputs.
            x = self.spatial_filter(x)
            cm = self._model_name_meta(x.device)
        else:
            cm = self._meta_tensors(meta, x.device)
            if cm is not None and not bool(cm["valid_channel_mask"].all()):
                x = x * cm["valid_channel_mask"].to(x.dtype).view(1, -1, 1)
            if self.spatial_filter is not None:
                x = self.spatial_filter(x)
        x = self._to_slots(x)
        cm = self._slot_meta(cm) if self.slot_index is not None else self._with_new_rows(cm)
        spec = self.wavelet_frontend(x)
        tokens = self.patch_embed(spec.unsqueeze(1))
        n_patches = tokens.shape[1] // r.n_channels
        if cm is not None:
            code = self.channel_encoder(cm)
            if code.dim() == 2:
                code = code.unsqueeze(0)
            delta = torch.tanh(self.channel_token_gate) * self.channel_to_token(code)
            B, L, D = tokens.shape
            tokens = (tokens.reshape(B, r.n_channels, n_patches, D)
                      + delta.unsqueeze(-2)).reshape(B, L, D)
        tokens = self.pos_embed(tokens, freq_size=r.n_channels,
                                time_size=n_patches)
        return self.shared_transformer(tokens)

    def forward(self, x: torch.Tensor, meta=None) -> Dict[str, torch.Tensor]:
        tokens = self.encode(x, meta)                       # [B, C*P, D]
        if self.pool == "stat":
            pooled = self.head_pool(tokens)
            return {"logits": self.head(self.head_drop(pooled)),
                    "features": pooled}
        if isinstance(self.head, TemporalHead):
            # The patcher emits tokens channel-major, [C, P] flattened, which is
            # the order `encode` adds the channel code and the 2-D position in,
            # so this reshape is electrodes on axis 1 and time on axis 2. The
            # head decides what to do with each; nothing here pools.
            B, L, D = tokens.shape
            logits, pooled = self.head(
                tokens.reshape(B, L // self.n_patches, self.n_patches, D))
            return {"logits": logits, "features": pooled}
        pooled = tokens.max(dim=1).values if self.pool == "max" else tokens.mean(dim=1)
        return {"logits": self.head(self.head_drop(self.head_norm(pooled))),
                "features": pooled}

    # -- loading ------------------------------------------------------------- #
    def load_pretrained(self, path: str, route_id: Optional[str] = None,
                        strict_shapes: bool = True,
                        allow_missing_gate: bool = False) -> Dict[str, List[str]]:
        """Load what an exported encoder can supply, and report exactly what.

        Silence here is the failure mode. A checkpoint whose shapes disagree
        loads nothing under a permissive loader and trains from scratch while
        the log says "pretrained", so every key is accounted for: taken,
        skipped because this montage does not share it, or refused.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        recorded = ck.get("channel_vocab_sha256")
        current = self.vocab_payload()["channel_vocab_sha256"]
        if recorded and recorded != current:
            raise SystemExit(
                f"{path} was trained under channel vocabulary {recorded[:16]} "
                f"and this one is {current[:16]}. Every embedding row would "
                f"mean a different electrode.")

        ck_route = ck.get("route_id") or route_id
        own = self.state_dict()
        # The frontend and patcher come across only when the montage IS the
        # route: same electrode count, same patch length. Otherwise they are a
        # different shape and a different meaning.
        same_route = False
        if ck_route in self.ROUTES_TABLE:
            r = self.ROUTES_TABLE[ck_route]
            same_route = (r.n_channels == self.route.n_channels
                          and r.patch_t == self.route.patch_t)

        taken, shape_mismatch, skipped = [], [], []
        for k, v in sd.items():
            if k not in own:
                skipped.append(k)
                continue
            transferable = k.startswith(TRANSFERABLE)
            if k.startswith(ROUTE_BOUND) and not same_route:
                skipped.append(k)
                continue
            if not transferable and not k.startswith(ROUTE_BOUND):
                skipped.append(k)
                continue
            if own[k].shape != v.shape:
                shape_mismatch.append(
                    f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}")
                continue
            own[k] = v
            taken.append(k)

        if shape_mismatch and strict_shapes:
            raise SystemExit(
                f"{path}: {len(shape_mismatch)} tensor(s) do not fit this "
                f"model, so loading it would train from scratch and report "
                f"otherwise:\n  " + "\n  ".join(shape_mismatch[:8]))
        if not any(k.startswith("shared_transformer.") for k in taken):
            raise SystemExit(
                f"{path} supplied no transformer weights. That is the part "
                f"pretraining produced, and without it this is a from-scratch "
                f"run with extra steps.\n"
                f"  Keys in the checkpoint: {sorted(sd)[:6]} ...")
        if (self.channel_encoder is not None and not allow_missing_gate
                and "channel_token_gate" not in taken):
            # Silent zero, and the worst kind: the channel encoder's weights
            # load, the log says they loaded, and every one of them is then
            # multiplied by tanh(0) = 0. Under --freeze-encoder the gate never
            # moves, so the C1 mechanism is off for the whole run -- in the
            # experiment that exists to measure it. Exports before the fix in
            # 3ba3b8f dropped this key, because it is a bare scalar and the
            # allowlist was module prefixes.
            raise SystemExit(
                f"{path} has no channel_token_gate.\n"
                f"  It scales the ENTIRE channel-identity path "
                f"(delta = tanh(gate) * proj(code)) and initialises to zero, so "
                f"loading this encoder gives a model whose C1 mechanism "
                f"contributes exactly nothing -- and a frozen probe can never "
                f"move it off zero.\n"
                f"  The export dropped it; the pretraining checkpoint still has "
                f"it. Re-export:\n"
                f"      python scripts/export_eeg_pretrained_encoder.py \\\n"
                f"          --checkpoint <pretrain>/best.pth --route "
                f"{ck_route or '<route>'} --output {path}\n"
                f"  --allow-missing-gate runs anyway, which is only right if "
                f"you are deliberately ablating the mechanism.")
        self.load_state_dict(own)
        self._init_new_rows()
        if same_route and any(k.startswith(ROUTE_BOUND) for k in taken):
            self.pretrained_prefixes = TRANSFERABLE + ROUTE_BOUND
        if self._frozen and self.freeze_scope == "pretrained" and same_route \
                and any(k.startswith(ROUTE_BOUND) for k in taken):
            # The frontend and patcher came from pretraining too, so under
            # `pretrained` they are frozen with the rest of what did.
            self._frozen_prefixes = TRANSFERABLE + ROUTE_BOUND
            self._apply_freeze()
        return {"taken": taken, "skipped": skipped,
                "shape_mismatch": shape_mismatch,
                "route_reused": [ck_route] if same_route else [],
                "new_channel_rows": list(self.new_channel_names)}

    def describe_transfer(self, report: Dict[str, List[str]]) -> str:
        def n(prefix):
            return sum(1 for k in report["taken"] if k.startswith(prefix))
        parts = [f"transformer {n('shared_transformer.')}",
                 f"channel encoder {n('channel_encoder.') + n('channel_to_token.')}",
                 f"gate {'yes' if 'channel_token_gate' in report['taken'] else 'NO'}",
                 f"frontend {n('wavelet_frontend.')}",
                 f"patcher {n('patch_embed.')}"]
        rows = report.get("new_channel_rows") or []
        if rows:
            parts.append(f"{len(rows)} new channel row(s) for names outside the "
                         f"vocabulary (trained as adapters)")
        return "loaded: " + ", ".join(parts)
