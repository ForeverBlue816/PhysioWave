#!/usr/bin/env python
"""Rebuild a finetune.py checkpoint from the args it stored and report on it.

Reconstructing the model by hand means restating every architecture flag the
run used, and getting one wrong produces a wall of state_dict key errors rather
than a wrong number -- which is the good case. The checkpoint records
``vars(args)``, so the model that matches it can be built from the file itself.

Usage:
    python scripts/inspect_checkpoint.py <best_model.pth> [--data <test.h5>] [-n 256]

Without --data it reports the architecture and the fold's learned parameters.
With it, it also pushes real windows through the wavelet module and reports how
the energy of Spec(X) is distributed over the decomposition levels -- which is
what says whether the scale fold's plain 1/(J+1) average is dominated by the
approximation band or not.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_wavelet_classifier  # noqa: E402


# Constructor arguments that finetune.py takes from the CLI under a different
# name, or shapes differently, than BERTWaveletTransformer expects.
def model_kwargs_from_args(a: dict) -> dict:
    kw = dict(
        in_channels=a["in_channels"], max_level=a["max_level"],
        wave_kernel_size=a["wave_kernel_size"], wavelet_names=a["wavelet_names"],
        use_separate_channel=a["use_separate_channel"],
        patch_size=(1, a["patch_size"]), embed_dim=a["embed_dim"],
        depth=a["depth"], num_heads=a["num_heads"], mlp_ratio=a["mlp_ratio"],
        dropout=a["dropout"], num_classes=a["num_classes"],
        use_pos_embed=a["use_pos_embed"], pos_embed_type=a["pos_embed_type"],
        pooling=a["pooling"],
        head_config={
            "hidden_dims": [a["head_hidden_dim"]] if a.get("head_hidden_dim") else None,
            "dropout": a["head_dropout"], "pooling": a["pooling"],
        },
    )
    # Everything below post-dates the original checkpoints, so each is optional
    # and falls back to the value that reproduces the original block.
    for key, default in (("norm", "layernorm"), ("ffn", "mlp"), ("qk_norm", False),
                         ("scale_fold", "none")):
        kw[key] = a.get(key, default)
    for cli, ctor, default in (("fold_patch_len", "fold_patch_len", None),
                               ("fold_synthesis", "fold_synthesis", 0),
                               ("fold_synthesis_norm", "fold_synthesis_norm", False),
                               ("fold_share_channels", "fold_share_channels", False),
                               ("fold_shrinkage", "fold_shrinkage", False),
                               ("fold_scale_dropout", "fold_scale_dropout", 0.0),
                               ("fold_gamma", "fold_gamma", 0.1)):
        kw[ctor] = a.get(cli, default)
    return kw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--data", help="an HDF5 file with a 'data' dataset, for band energies")
    p.add_argument("-n", type=int, default=256, help="windows to read from --data")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ck.get("args")
    if saved is None:
        raise SystemExit("This checkpoint has no stored args; it predates them.")

    print(f"epoch {ck.get('epoch')}  val_acc {ck.get('val_acc')}  "
          f"val_balanced_acc {ck.get('val_balanced_acc')}")
    print(f"  block     norm={saved.get('norm')} ffn={saved.get('ffn')} "
          f"qk_norm={saved.get('qk_norm')}")
    print(f"  fold      scale_fold={saved.get('scale_fold')} "
          f"synthesis={saved.get('fold_synthesis')} "
          f"norm={saved.get('fold_synthesis_norm')} "
          f"share_channels={saved.get('fold_share_channels')}")

    model = create_wavelet_classifier(**model_kwargs_from_args(saved))
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        # Loud, because a silently partial load reports numbers from a model
        # that is part trained and part freshly initialised.
        raise SystemExit(f"state_dict mismatch\n  missing: {list(missing)}\n"
                         f"  unexpected: {list(unexpected)}")
    model.eval()
    print("  loaded    all keys matched")

    fold = [(n, p) for n, p in model.named_parameters() if n.startswith("fold.")]
    if fold:
        print("\nfold parameters")
        for n, t in fold:
            flat = t.detach().flatten()
            head = ", ".join(f"{v:.4f}" for v in flat[:8].tolist())
            print(f"  {n:24s} {list(t.shape)}  [{head}{', ...' if flat.numel() > 8 else ''}]")

    if not args.data:
        return

    import h5py

    with h5py.File(args.data, "r") as f:
        x = torch.tensor(f["data"][: args.n]).float()
    with torch.no_grad():
        spec = model.wavelet_decomp(x)
    B, FC, T = spec.shape
    S = saved["max_level"] + 1
    bands = spec.view(B, S, FC // S, T)

    print(f"\nSpec(X) energy by level, over {B} windows of {x.shape[1]}x{T}")
    e = bands.pow(2).mean(dim=(0, 2, 3))
    for i, share in enumerate((e / e.sum()).tolist()):
        tag = "approximation" if i == S - 1 else f"detail level {i}"
        print(f"  band {i} ({tag:15s}) {share * 100:5.1f}%")
    print(f"\n  a plain 1/{S} average weights all of these equally, so it is "
          f"dominated by\n  whichever band carries the energy.")


if __name__ == "__main__":
    main()
