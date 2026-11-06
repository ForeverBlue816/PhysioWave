#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export one route's encoder from a pretraining checkpoint, for fine-tuning.

EEG, ECG and sEMG C1 checkpoints alike: the modality is read from the
checkpoint (--modality overrides), and with it the routes, the channel
vocabulary that is attached, and the preprocessing note.

    python scripts/export_eeg_pretrained_encoder.py \
        --checkpoint pretrain_ecg_c1_moe/best.pth --route L12_500 --output ecg_L12_500.pth

    python scripts/export_eeg_pretrained_encoder.py \
        --checkpoint best.pth --route E32_512 --output exported_E32_512.pth

    python scripts/export_eeg_pretrained_encoder.py \
        --checkpoint best.pth --output encoder.pth        # no --route

A pretraining checkpoint holds four wavelet frontends, two patchers and two
reconstruction decoders. A downstream task uses at most one route and no
decoder, so carrying the rest means shipping several times the weights and
inviting someone to load a frontend built for a different electrode count.

--route IS OPTIONAL, and omitting it is now the common case. downstream.py
splits what a checkpoint may supply into two sets:

    TRANSFERABLE  channel_encoder, channel_to_token, channel_token_gate,
                  shared_transformer   -- always loaded
    ROUTE_BOUND   wavelet_frontend, patch_embed
                  -- loaded ONLY when the downstream montage IS the route

Both downstream tasks build their own frontend and declare no route_id, so for
them ROUTE_BOUND is always skipped. Passing --route to feed those tasks means
choosing a route for weights that are then discarded on load; without it the
file holds the part that actually transfers, and is a good deal smaller.

Pass --route only to fine-tune ON one of the pretraining routes, at its exact
electrode count and patch length.

What comes out:

    channel_encoder.*     the C1 channel-name embedding, whole
    channel_to_token.*    its projection, and the gate
    shared_transformer.*  the encoder
    + the channel vocabulary, its hash, and the preprocessing spec
    + wavelet_frontend.* and patch_embed.*, ONLY with --route

What does not:

    raw_reconstruction_heads.*  the second pretraining decoder, which predicts
                             the preprocessed EEG. Pretraining-only, like the
                             one below.
    reconstruction_heads.*   the pretraining decoder. It predicts folded
                             wavelet patches, which no downstream head wants,
                             and keeping it invites fine-tuning against the
                             pretext objective by accident.

The channel vocabulary travels with the weights because an embedding row means
whatever channel held that id when the row was learned. A fine-tuning run that
resolves names against a different vocabulary silently trains on relabelled
electrodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_embedding import vocab_payload                   # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                    # noqa: E402

PIPELINES = {
    "eeg": ("units_uV -> detrend -> notch -> 0.5 Hz high-pass -> "
            "resample_poly -> slot_map -> window -> zscore_clip"),
    "ecg": ("units_mV -> dc_remove -> notch -> 0.5 Hz high-pass (sos, even pad) "
            "-> resample_poly -> lead_slots (+derived limb leads) -> 10 s window "
            "-> window_shared normalisation (one scale across leads) -> clip"),
    "emg": ("units_mV -> dc_remove -> notch x3 -> 20 Hz high-pass -> "
            "resample_poly -> electrode slots -> 1 s window -> window_shared "
            "normalisation -> clip"),
}


def modality_registry(modality: str):
    """``(routes, vocab_payload)`` of one modality."""
    if modality == "ecg":
        from physiowave.ecg_c1.leads import ecg_vocab_payload
        from physiowave.ecg_c1.routes import ROUTES as R
        return R, ecg_vocab_payload
    if modality == "emg":
        from physiowave.emg_c1.electrodes import emg_vocab_payload
        from physiowave.emg_c1.routes import ROUTES as R
        return R, emg_vocab_payload
    return ROUTES, vocab_payload

#: `channel_token_gate` is a bare scalar, not a submodule, so it does not start
#: with "channel_to_token." and an allowlist of module prefixes dropped it. It
#: is the gate on the whole channel-identity contribution -- delta =
#: tanh(gate) * proj(code) -- and it initialises to zero, so an export without
#: it hands fine-tuning an encoder whose C1 mechanism contributes exactly
#: nothing, silently, in the experiment that exists to measure that mechanism.
#: test_export_carries_every_parameter_the_encoder_uses now derives this list
#: from what the forward pass actually reaches rather than from a reading of it.
KEEP_PREFIXES = ("channel_encoder.", "channel_to_token.", "channel_token_gate",
                 "shared_transformer.", "pos_embed.", "mask_token")
#: Documentation of what the allowlist above already excludes. Both decoders
#: are pretraining-only: the spec head predicts folded wavelet patches and the
#: raw head predicts preprocessed EEG, and fine-tuning needs neither.
DROP_PREFIXES = ("reconstruction_heads.", "raw_reconstruction_heads.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--modality", default="auto", choices=["auto", "eeg", "ecg", "emg"],
                   help="default: read from the checkpoint's trainer")
    p.add_argument("--route", default=None,
                   help="include this route's wavelet frontend and patcher. "
                        "OMIT for the encoder alone -- see below")
    p.add_argument("--output", required=True)
    p.add_argument("--keep-mask-token", action="store_true",
                   help="keep the pretraining mask token (downstream ignores it)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model")
    if sd is None:
        print("ERROR: checkpoint has no 'model' state dict", file=sys.stderr)
        return 1

    modality = args.modality
    if modality == "auto":
        trainer = (ck.get("config") or {}).get("trainer", "")
        modality = {"ecg_c1_moe": "ecg", "emg_c1_moe": "emg"}.get(
            trainer, ck.get("channel_vocab_modality") or "eeg")
    routes, vocab = modality_registry(modality)
    if args.route and args.route not in routes:
        print(f"ERROR: {args.route} is not a {modality} route; those are "
              f"{sorted(routes)}", file=sys.stderr)
        return 1
    route = routes[args.route] if args.route else None
    # Prefixes that match no key when there is no route, so the loop below
    # needs no second branch: an f-string on None would match "None." and
    # quietly export nothing under a name that looks exported.
    front = f"wavelet_frontends.{args.route}." if route else None
    patch = f"patch_embed_by_rate.{route.rate_key}." if route else None

    out = {}
    for k, v in sd.items():
        if any(k.startswith(d) for d in DROP_PREFIXES):
            continue
        if k.startswith("wavelet_frontends."):
            if front and k.startswith(front):
                out["wavelet_frontend." + k[len(front):]] = v
            continue
        if k.startswith("patch_embed_by_rate."):
            if patch and k.startswith(patch):
                out["patch_embed." + k[len(patch):]] = v
            continue
        if k.startswith("mask_token") and not args.keep_mask_token:
            continue
        if any(k.startswith(pref) for pref in KEEP_PREFIXES):
            out[k] = v

    if route and not any(k.startswith("wavelet_frontend.") for k in out):
        print(f"ERROR: the checkpoint has no frontend for route {args.route}",
              file=sys.stderr)
        return 1
    if not any(k.startswith("shared_transformer.") for k in out):
        # The encoder is the whole point with or without a route, and a file
        # without it loads downstream as "0 keys taken" rather than as an error.
        print("ERROR: the checkpoint has no shared_transformer", file=sys.stderr)
        return 1

    cfg = ck.get("config", {})
    payload = {
        "model": out,
        "route_id": args.route,
        "route": ({"n_channels": route.n_channels,
                   "sampling_rate": route.sampling_rate,
                   "window_samples": route.window_samples,
                   "patch_size": list(route.patch_size),
                   "n_tokens": route.n_tokens,
                   "slots": list(route.slots)} if route else None),
        "model_config": cfg.get("model", {}),
        # Without a route there is no single sampling rate or window length to
        # state -- the encoder saw four. The pipeline line still applies: it is
        # how every window reaching this encoder was prepared, at whatever rate.
        "preprocessing_spec": {
            "window_seconds": route.window_seconds if route else None,
            "patch_seconds": route.patch_seconds if route else None,
            "target_sampling_rate": route.sampling_rate if route else None,
            "pipeline": PIPELINES[modality],
            "note": ("no 0.5-45 Hz band-pass; the encoder has seen the full "
                     "band up to Nyquist and expects input prepared the same "
                     "way"),
        },
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        # WHICH checkpoint this is, and what picked it. A file called best.pth
        # says nothing about the criterion, and the criterion changed: it used
        # to be the spec loss alone and is now the total. Both are carried, so
        # an exported encoder can be traced to the bar it cleared.
        "objective": ck.get("objective"),
        "best_scores": ck.get("best_scores"),
        "best_epochs": ck.get("best_epochs"),
        "checkpoint_selection": ck.get("checkpoint_selection"),
        # Alias for the spec bar, kept for readers written against it.
        "best_val_loss_masked_mse": ck.get("best_val_loss_masked_mse"),
        "modality": modality,
        **vocab(),
    }
    recorded = ck.get("channel_vocab_sha256")
    if recorded and recorded != payload["channel_vocab_sha256"]:
        print(f"ERROR: the checkpoint was trained under channel vocabulary "
              f"{recorded[:16]} and this working tree has "
              f"{payload['channel_vocab_sha256'][:16]}. Exporting would attach "
              f"the wrong vocabulary to these weights.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    torch.save(payload, args.output)

    n_par = sum(v.numel() for v in out.values() if hasattr(v, "numel"))
    print(f"exported {args.route or 'encoder (no route)'} "
          f"from {args.checkpoint}")
    print(f"  {len(out)} tensors, {n_par:,} parameters -> {args.output}")
    if route:
        print(f"  input shape  [B, {route.n_channels}, "
              f"{route.window_samples}] @ {route.sampling_rate} Hz")
        print(f"  tokens       {route.n_tokens}")
        print(f"  frontend     included (loads only onto this exact route)")
    else:
        print(f"  frontend     excluded -- downstream builds its own; "
              f"pass --route to include one")
    print(f"  vocab sha    {payload['channel_vocab_sha256'][:16]}")
    print(f"  decoder      excluded (pretraining head)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
