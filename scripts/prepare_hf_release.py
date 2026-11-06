#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assemble a Hugging Face release from a finished C1 pretraining run: EEG, ECG
or sEMG, read from the checkpoint (--modality overrides).

    python scripts/prepare_hf_release.py \
        --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe \
        --out-dir ~/hf_release --repo-id ForeverBlue/EEG

    python scripts/prepare_hf_release.py \
        --run-dir $PW_CKPT_ROOT/pretrain_ecg_c1_moe \
        --out-dir ~/hf_release_ecg --repo-id ForeverBlue/ECG

    python scripts/prepare_hf_release.py \
        --run-dir $PW_CKPT_ROOT/pretrain_emg_c1_moe --best epoch38.pth \
        --main-why "..." --out-dir ~/hf_release_emg --repo-id ForeverBlue/EMG

--best names the checkpoint the release is built around and the encoder is
exported from. It is best.pth unless the paper reports another file, as for
sEMG (epoch38.pth); best.pth is then published beside it under its own name,
and --main-why is the sentence the card gives for the choice.

Uploading best.pth as it sits on disk is the obvious move and is wrong three
times over.

  * It carries the AdamW state, which is two moments per parameter -- roughly
    three times the weights for a file nobody downstream will ever resume from.
  * Its embedded config holds the absolute manifest paths the run trained
    against, which name the scratch filesystem, the project allocation and the
    username. That is not a secret, but publishing it is a decision rather than
    an accident, and this makes it one.
  * It holds four route frontends and both pretraining decoders. Someone
    fine-tuning on a 19-channel montage wants one frontend and no decoder, and
    handing them the rest invites loading a frontend built for a different
    electrode count.

So the release carries two shapes: weights-only checkpoints for anyone who
wants the whole multi-route model, and ONE route-agnostic encoder for anyone
who wants to fine-tune. Not one encoder per route -- downstream.py loads a
checkpoint's wavelet frontend and patcher only when the target montage IS the
route, and both downstream tasks here build their own frontend, so four
per-route files would differ only in bytes nobody loads. The full checkpoints
still hold all four frontends for anyone fine-tuning on a route exactly.

The model card is written from the run's own
metrics_epoch.jsonl rather than typed, because a card whose numbers were typed
is a card that disagrees with the run the moment either changes.

NOTHING IS UPLOADED. This writes a directory and prints the commands.
Publishing is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physiowave.eeg_c1.objective import (objective_equation,   # noqa: E402
                                         resolve_eeg_c1_objective)

#: What differs between the three modalities' releases. Everything else -- the
#: stripping, the scrub, the leak check, the card's structure -- is shared,
#: because the three pretrainers are one implementation with the registry
#: swapped (physiowave/{ecg,emg}_c1/model.py).
SPECS = {
    "eeg": {
        "label": "EEG", "tags": ["eeg"],
        "intro": ("Masked dual-reconstruction pretraining on multi-montage EEG. One "
                  "shared transformer serves {n} electrode layouts through "
                  "route-specific wavelet frontends, and channel identity enters as a "
                  "learned embedding of the channel NAME rather than of its index."),
        "downstream": ("physiowave.eeg_c1.downstream", "EEGC1Downstream"),
        "wavelets": ("sym4", "sym5", "db6", "sym8", "db8"),
        "example": {"ch": '["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]',
                    "what": "your electrodes", "window": 1024, "rate": "256.0",
                    "patch": 64},
        "route_example": "E64_256",
        "transfer_note": ("So a fresh frontend is the normal case, not a fallback. "
                          "Both downstream tasks in the source repository build "
                          "their own."),
        "raw_target": "z-scored volts", "names": "channel names",
    },
    "ecg": {
        "label": "ECG", "tags": ["ecg", "electrocardiogram"],
        "intro": ("Masked dual-reconstruction pretraining on 12-lead and single-lead "
                  "ECG. One shared transformer serves {n} lead configurations "
                  "through route-specific wavelet frontends, and lead identity enters "
                  "as a learned embedding of the lead NAME rather than of its index."),
        "downstream": ("physiowave.ecg_c1.downstream", "ECGC1Downstream"),
        "wavelets": ("db4", "db6", "sym4", "sym8", "coif2"),
        "example": {"ch": '["I", "II", "III", "aVR", "aVL", "aVF",\n      '
                          '"V1", "V2", "V3", "V4", "V5", "V6"]',
                    "what": "your leads", "window": 5000, "rate": "500.0",
                    "patch": 250},
        "route_example": "L12_500",
        "transfer_note": ("A 12-lead recording at 500 Hz in 10 s windows IS the "
                          "`L12_500` route, so on the most common ECG input the "
                          "pretrained frontend and patcher transfer as well: cut that "
                          "route out (below) and pass `route_id=\"L12_500\"`. Any "
                          "other lead set, rate or window length builds its own."),
        "raw_target": "the window-normalised waveform", "names": "lead names",
    },
    "emg": {
        "label": "sEMG", "tags": ["emg", "semg", "electromyography"],
        "intro": ("Masked dual-reconstruction pretraining on surface EMG from "
                  "wristbands, forearm arrays and high-density grids. One shared "
                  "transformer serves {n} electrode layouts through route-specific "
                  "wavelet frontends, and electrode identity enters as a learned "
                  "embedding of the electrode NAME rather than of its index."),
        "downstream": ("physiowave.emg_c1.downstream", "EMGC1Downstream"),
        "wavelets": ("db2", "db4", "db7", "sym4", "sym5"),
        "example": {"ch": '[f"ch{i}" for i in range(1, 9)]',
                    "what": "an 8-electrode armband", "window": 2000,
                    "rate": "2000.0", "patch": 250},
        "route_example": "G64_2000",
        "transfer_note": ("The usual sEMG benchmark montages -- armbands, forearm "
                          "rings, NinaPro's electrodes -- are none of these routes, so "
                          "a fresh frontend is the normal case. Their electrode names "
                          "are outside the vocabulary too. Each unknown name gets an "
                          "embedding row of its own, initialised from the pretrained "
                          "`<unk>` row and trained as a montage adapter; "
                          "`new_channel_rows=False` keeps the one shared `<unk>` "
                          "vector instead, and channels are then told apart by the "
                          "position encoding alone."),
        "raw_target": "the window-normalised waveform", "names": "electrode names",
    },
}

#: The pretraining trainer each modality's checkpoint records.
TRAINER_MODALITY = {"eeg_c1_moe": "eeg", "ecg_c1_moe": "ecg", "emg_c1_moe": "emg"}

NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def modality_routes(modality: str):
    if modality == "ecg":
        from physiowave.ecg_c1.routes import ROUTES as R
    elif modality == "emg":
        from physiowave.emg_c1.routes import ROUTES as R
    else:
        from physiowave.eeg_c1.routes import ROUTES as R
    return R

#: Dropped from a published checkpoint. Each is either resume-only state or a
#: path into the machine the run happened on.
TRAINING_ONLY = ("optimizer", "scheduler", "scaler", "rng", "sampler",
                 "history")

#: Fields outside the config that hold a path on the training filesystem.
#: An exported encoder records the checkpoint it came from as an absolute path,
#: which on a cluster names the scratch mount, the project allocation and the
#: username.
PATH_FIELDS = ("source_checkpoint", "run_dir", "out_dir", "output_dir")

REDACTED = "<path redacted for release>"


def _scrub(value):
    """Recursively replace absolute-path strings.

    A walk rather than a list of known keys. The list was
    data.manifest_train/manifest_val/root/corpus_root, and it missed
    `source_checkpoint` in the exported encoder -- which is the failure mode
    of every allowlist: it protects what was thought of. A config value that
    begins with "/" or "~" is a path, and a release has no use for the
    training machine's directory layout.
    """
    if isinstance(value, dict):
        return {k: (REDACTED if k in PATH_FIELDS and value[k] else _scrub(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and (value.startswith("/")
                                   or value.startswith("~/")):
        return REDACTED
    return value


def scrub_config(cfg: dict) -> dict:
    """A copy of the config with training-filesystem paths replaced.

    Replaced rather than deleted: a reader needs to know a manifest WAS
    configured, and an absent key reads as "this run had no validation split".
    """
    return _scrub(json.loads(json.dumps(cfg, default=str)))


def find_paths(path: str):
    """Absolute paths surviving anywhere in a released checkpoint's metadata.

    The last line of defence, run over what is about to be published rather
    than over what was intended to be published. Tensors are skipped; only the
    metadata can carry a string.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def walk(v, where):
        if torch.is_tensor(v):
            return
        if isinstance(v, dict):
            for k, sub in v.items():
                walk(sub, f"{where}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, sub in enumerate(v):
                walk(sub, f"{where}[{i}]")
        elif isinstance(v, str) and (v.startswith("/") or v.startswith("~/")) \
                and len(v) > 8 and "/" in v[1:]:
            found.append((where, v))

    walk(ck, os.path.basename(path))
    return found


#: An absolute path in a text file: a "/" run with at least two segments. The
#: README legitimately contains URLs and code fences, so http(s) and the
#: repository's own relative paths must not trip it.
_PATH_RE = re.compile(r"(?<![\w:/])(/[A-Za-z0-9_.-]+){2,}/?")


def text_paths(path: str):
    """``(line number, match)`` for absolute paths in a released text file."""
    out = []
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if "://" in line:
                    continue
                for m in _PATH_RE.finditer(line):
                    if m.group(0) != REDACTED:
                        out.append((i, m.group(0)))
    except (OSError, UnicodeDecodeError):
        pass
    return out


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def strip_checkpoint(src: str, dst: str) -> dict:
    """Weights, config and provenance. No optimizer, no paths."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: v for k, v in ck.items() if k not in TRAINING_ONLY}
    keep["config"] = scrub_config(ck.get("config", {}))
    for field in PATH_FIELDS:
        if keep.get(field):
            keep[field] = REDACTED
    keep["released_from"] = os.path.basename(src)
    keep["released_utc"] = datetime.now(timezone.utc).isoformat()
    torch.save(keep, dst)
    return {"epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "best_scores": ck.get("best_scores", {}),
            "best_epochs": ck.get("best_epochs", {}),
            "objective": resolve_eeg_c1_objective(ck.get("config", {})),
            "config": ck.get("config", {}),
            "n_params": sum(v.numel() for v in ck["model"].values()
                            if hasattr(v, "numel")),
            "src_bytes": os.path.getsize(src),
            "dst_bytes": os.path.getsize(dst)}


#: Rendered in the card so a reader can see what each file costs and what it
#: buys, rather than being told. Order is fixed; sizes are measured.
COMPONENTS = (
    ("shared_transformer.", "shared transformer", "encoder, always transfers"),
    ("channel_encoder.", "channel-name embedding", "always transfers"),
    ("channel_to_token.", "channel projection", "always transfers"),
    ("wavelet_frontends.", "wavelet frontends (all {routes})",
     "route-bound; downstream usually builds its own"),
    ("patch_embed_by_rate.", "patchers ({rates})", "route-bound"),
    ("reconstruction_heads.", "spec decoder", "pretraining only"),
    ("raw_reconstruction_heads.", "raw decoder", "pretraining only"),
)


def _count(n: int, one: str, many: str) -> str:
    if n == 2 and many == "rates":
        return "both rates"
    return f"{NUMBER.get(n, n)} {one if n == 1 else many}"


def component_table(sd: dict, routes=None):
    """``(markdown rows, {prefix: share})`` for the full model, all measured.

    The shares come back with the table because the sentence under it used to
    be prose -- "the shared transformer is most of the model" -- written from
    the 512/8/8 run it happened to be true for. Beside a measured table it is
    a claim that can contradict the rows directly above it, and at 32/1 it did:
    the transformer was 0.4% and the card said it was most of the model.
    """
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) or 1
    routes = routes or {}
    n_routes = len(routes) or 4
    n_rates = len({r.rate_key for r in routes.values()}) or 2
    names = {"routes": _count(n_routes, "route", "routes").replace("one route", "the route"),
             "rates": _count(n_rates, "rate", "rates")}
    lines, share = [], {}
    for prefix, label, note in COMPONENTS:
        label = label.format(**names)
        n = sum(v.numel() for k, v in sd.items()
                if k.startswith(prefix) and hasattr(v, "numel"))
        share[prefix] = n / total
        if not n:
            continue
        lines.append(f"| {label} | {n / 1e6:.2f} M | {n / total * 100:.1f}% "
                     f"| {note} |")
    lines.append(f"| **total** | **{total / 1e6:.2f} M** | 100% | |")
    return "\n".join(lines), share


def why_no_per_route_files(share: dict, n_routes: int = 4) -> str:
    """The reason there is one encoder and not one per route, from the actual shares."""
    enc = share.get("shared_transformer.", 0.0)
    fronts = share.get("wavelet_frontends.", 0.0)
    n = NUMBER.get(n_routes, str(n_routes))
    if enc > fronts:
        return (
            f"The shared transformer is {enc * 100:.0f}% of the model and the "
            f"{n} frontends together are {fronts * 100:.0f}%, which is why "
            f"there is no per-route encoder file here: {n} of them would "
            f"duplicate that transformer {n} times to deliver frontends that "
            f"are a small fraction of it. Cut the one you want out of the "
            f"complete checkpoint instead.")
    return (
        f"At this width the {n} frontends ({fronts * 100:.0f}%) outweigh the "
        f"shared transformer ({enc * 100:.0f}%). There is still one encoder "
        f"file rather than {n}, because what a per-route file adds is a "
        f"frontend that only loads onto that exact montage; cut the one you "
        f"want out of the complete checkpoint.")


def model_digest(path: str) -> str:
    """A hash of the WEIGHTS, so two checkpoints can be compared by content.

    Not of the file: two checkpoints written at the same epoch under different
    names differ in their metadata -- released_utc, released_from -- and are
    the same model. What matters for a release is whether the tensors differ.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes() if hasattr(v, "detach")
                 else repr(v).encode())
    return h.hexdigest()


def read_metrics(run_dir: str):
    rows = []
    path = os.path.join(run_dir, "metrics_epoch.jsonl")
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def _row_for(rows, epoch):
    """The metrics row of the epoch a checkpoint was saved at.

    A checkpoint records the number of epochs completed; metrics_epoch.jsonl
    numbers them from 0. So best.pth at epoch 58 is the row with epoch 57.
    """
    if isinstance(epoch, int):
        for r in rows:
            if r.get("epoch") == epoch - 1:
                return r
    return rows[-1] if rows else {}


def model_card(repo_id: str, info: dict, rows, run_dir: str,
               components: str = "", shares: Optional[dict] = None,
               published=(), aliases: Optional[dict] = None,
               modality: str = "eeg", routes=None, main_name: str = "",
               enc_name: str = "", main_label: str = "",
               main_why: str = "", final_epoch: Optional[int] = None) -> str:
    spec = SPECS[modality]
    routes = routes or {}
    n_routes = len(routes)
    nw = NUMBER.get(n_routes, str(n_routes))
    cfg = info["config"]
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
    obj = info["objective"]
    row = _row_for(rows, info.get("epoch"))
    last = rows[-1] if rows else {}
    bep = info.get("best_epochs", {})
    aliases = aliases or {}
    ex = spec["example"]
    ds_mod, ds_cls = spec["downstream"]
    wavelets = ", ".join(mcfg.get("wavelet_names") or spec["wavelets"])
    # Each row its own line, and none when there is nothing extra: an empty
    # join left a blank line that ended the markdown table halfway.
    checkpoint_rows = "".join(
        f"| `{name}` | complete model, {what} (epoch {ep}) |\n"
        for name, what, ep, dup in published if not dup)
    alias_note = ""
    if aliases:
        alias_note = ("\n\n" + "\n".join(
            f"`{a}` would have been byte-identical to `{t}` -- the same epoch "
            f"won both bars -- so it is not published separately."
            for a, t in aliases.items()))
    why_note = f"\n\n**Why `{main_name}` is the main file.** {main_why}" if main_why else ""

    def g(key, fmt="{:.4f}", r=None):
        v = (row if r is None else r).get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

    # The last epoch is the last CHECKPOINT, not the last metrics row: a run
    # can be continued past its last logged validation (the sEMG run's
    # latest.pth is epoch 45 with 39 rows), and a card that took the rows as
    # the whole run would call an earlier file "the last epoch".
    main_ep = info.get("epoch")
    final_ep = final_epoch if isinstance(final_epoch, int) else (
        (last.get("epoch") + 1) if isinstance(last.get("epoch"), int) else None)
    final_row = _row_for(rows, final_ep) if isinstance(final_ep, int) else {}
    if isinstance(final_ep, int) and final_row.get("epoch") != final_ep - 1:
        final_row = {}                               # no logged row for it

    # Measured, not asserted: how far the released checkpoint and the last
    # epoch are apart on the loss that was trained.
    gap_line = ""
    a, b = row.get("val/loss_total"), final_row.get("val/loss_total")
    if main_ep == final_ep:
        gap_line = ("The released checkpoint IS the last epoch, so there is no "
                    "separate final file.")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        gap_line = (f"The released checkpoint (epoch {main_ep}) and the last one "
                    f"(epoch {final_ep}) differ by "
                    f"{abs(a - b) / max(abs(b), 1e-12) * 100:.1f}% in validation "
                    f"total loss ({a:.4f} against {b:.4f}). Which transfers better is "
                    f"a downstream question, not a reconstruction-loss question -- "
                    f"measure both.")
    elif isinstance(final_ep, int):
        gap_line = (f"The last checkpoint (epoch {final_ep}) has no logged validation "
                    f"row, so it cannot be compared with the released one here. "
                    f"Which transfers better is a downstream question -- measure both.")

    n_epochs = final_ep if isinstance(final_ep, int) else (len(rows) or tcfg.get("epochs", "?"))
    planned = tcfg.get("epochs")
    epochs_note = (f" of {planned} configured" if isinstance(planned, int)
                   and isinstance(n_epochs, int) and planned != n_epochs else "")
    if rows and isinstance(n_epochs, int) and len(rows) < n_epochs:
        epochs_note += (f"; validation was logged for the first {len(rows)}, "
                        f"which is what `metrics_epoch.jsonl` holds")

    route_rows = "\n".join(
        f"| `{rid}` | {routes[rid].n_channels} | {routes[rid].sampling_rate:g} Hz |"
        for rid in sorted(routes))
    tags = "\n".join(f"  - {t}" for t in spec["tags"] + [
        "biosignal", "self-supervised", "masked-autoencoder", "pretrained"])

    return f"""---
license: apache-2.0
tags:
{tags}
library_name: pytorch
---

# {spec['label']} C1 multi-route pretrained encoder

{spec['intro'].format(n=nw)}

Trained for {n_epochs} epochs{epochs_note}
(the last logged validation at step {last.get('global_step', '?')}) on a mixture of public
{spec['label']} corpora, {info['n_params'] / 1e6:.1f} M parameters. The files
here are built around `{main_name}`, epoch {info.get('epoch', '?')}.

## Objective

```
{objective_equation(obj)}
```

Two heads at equal weight. One predicts folded wavelet patches, the other
predicts the preprocessed {spec['label']} waveform of the same masked patches.
Masking is applied BEFORE the wavelet frontend ({obj['mask_before_frontend']}),
at ratio {mcfg.get('mask_ratio', '?')}, and the spec target is per-patch
normalised ({obj['normalize_spec_target']}).

## Validation, at the released checkpoint (epoch {info.get('epoch', '?')})

| metric | value | what 1.0 would mean |
|---|---|---|
| total loss | {g('val/loss_total')} | — |
| spec MSE | {g('val/loss_masked_spec_mse')} | — |
| spec NMSE | {g('val/masked_spec_nmse', '{:.3f}')} | predicting zero |
| spec Pearson r | {g('val/masked_spec_corr', '{:.3f}')} | perfect |
| raw SmoothL1 | {g('val/loss_masked_raw_smoothl1')} | — |
| raw NMSE | {g('val/masked_raw_nmse', '{:.3f}')} | predicting zero |
| raw Pearson r | {g('val/masked_raw_corr', '{:.3f}')} | perfect |
| route-macro total | {g('val/macro_route_loss_total')} | — |

NMSE is dimensionless: MSE(prediction, target) / MSE(0, target), so 1.0 is
what predicting zero scores on that head's own target. The two losses are NOT
comparable to each other -- one is an MSE on normalised wavelet coefficients,
the other a SmoothL1 on {spec['raw_target']}. The columns that compare the
heads are the correlations and the NMSEs.

The route-macro row weights the {nw} routes equally. The plain rows weight by
sample count, and validation is dominated by the largest corpora, so a model
specialising toward the smaller ones moves the two rows in opposite
directions.

## Files

### Which file do you want

| your situation | download |
|---|---|
| fine-tune on any montage | `{enc_name}` |
| your montage IS one of the {nw} routes below | `{main_name}`, then cut the route out (one command, below) |
| continue pretraining, or take the model apart yourself | `{main_name}` |

**`{main_name}` is the complete model** -- all {nw} wavelet frontends, the
patchers, both reconstruction decoders. Nothing is held back; only the
optimiser state was stripped, which no downstream use needs. Take it if you
want to keep or strip pieces yourself.

`{enc_name}` is the same weights minus everything route-bound and
pretraining-only. It exists because it cannot be loaded wrong.

| file | what it is |
|---|---|
| `{enc_name}` | encoder alone: channel embedding, gate, transformer |
| `{main_name}` | complete model, {main_label} (epoch {info.get('epoch', '?')}) |
{checkpoint_rows}| `metrics_epoch.jsonl` | the full training curve |
| `config_resolved.yaml` | every resolved hyperparameter |

### Why several checkpoints

Each is the minimum of a different validation metric, and they are published
together so a downstream comparison can ask which criterion transfers instead
of assuming the one that names `best.pth`. Reconstruction loss cannot answer
that question; a downstream task can.

`best.pth` is selected on the TOTAL loss -- the loss that was trained (epoch
{bep.get('total', '?')} counted from 0). It is not the best spec loss, and not
the best raw loss.{why_note}{alias_note}

### What is in the complete model

| component | parameters | share | |
|---|---|---|---|
{components}

{why_no_per_route_files(shares or {}, n_routes)}

### What transfers, and what does not

`downstream.py` splits a checkpoint into two sets. The channel embedding, its
gate and the shared transformer **always** load. The wavelet frontend and the
patcher load **only when your montage is the route they were trained for** --
same channel count, same patch length -- and are skipped otherwise rather
than reshaped.

{spec['transfer_note']}

The {nw} routes the shared transformer was pretrained across:

| route | channels | rate |
|---|---|---|
{route_rows}

To fine-tune ON one of them exactly, take that route's frontend and patcher out
of the complete checkpoint:

```bash
python scripts/export_eeg_pretrained_encoder.py \\
    --checkpoint {main_name} --route {spec['route_example']} \\
    --output encoder_{spec['route_example']}.pth
```

The exporter reads the modality from the checkpoint; its name is historical.
Omit `--route` and you get `{enc_name}` again. Both pretraining decoders are
dropped either way: fine-tuning against the pretext objective by accident is
the failure that prevents.

{gap_line}

## The channel vocabulary travels with the weights

An embedding row means whatever channel held that id when the row was learned.
Every checkpoint here carries the vocabulary and its hash. Resolving
{spec['names']} against a different vocabulary silently trains on relabelled
channels, which is why the hash is checked rather than assumed.

## Architecture

```
embed_dim   {mcfg.get('embed_dim', '?')}      depth {mcfg.get('depth', '?')}      heads {mcfg.get('num_heads', '?')}
norm        {mcfg.get('norm', '?')}     ffn {mcfg.get('ffn', '?')}     qk_norm {mcfg.get('qk_norm', '?')}
wavelet     max_level {mcfg.get('max_level', '?')}, kernel {mcfg.get('wave_kernel_size', '?')}, bank {wavelets}
channel     {mcfg.get('channel_encoding', '?')} embedding, injected at the {mcfg.get('channel_injection', '?')} site
mask        {mcfg.get('masking_strategy', '?')}, ratio {mcfg.get('mask_ratio', '?')}
```

## Loading

`{ds_cls}` builds the frontend and patcher for your montage and loads this
file into everything else. Nothing here is route-bound, so the montage, the
window length and the patch length are yours to choose.

```python
import torch
from {ds_mod} import {ds_cls}

ENC = "{enc_name}"
ck = torch.load(ENC, map_location="cpu", weights_only=False)
mc = ck["model_config"]                  # the width the weights were trained at
CH = {ex['ch']}   # {ex['what']}

model = {ds_cls}(
    in_channels=len(CH), window_samples={ex['window']}, sampling_rate={ex['rate']},
    patch_samples={ex['patch']}, num_classes=2, channel_names=CH,
    embed_dim=mc["embed_dim"], depth=mc["depth"], num_heads=mc["num_heads"],
    channel_embed_dim=mc.get("channel_embed_dim", 64),
    channel_vocab_size=ck["channel_vocab_size"])

report = model.load_pretrained(ENC)
print(model.describe_transfer(report))
# loaded: transformer N, channel encoder N, gate yes, frontend 0, patcher 0

class Meta:                               # or your dataset's ChannelMeta
    channel_names = CH
    channel_mask = None

out = model(torch.randn(2, len(CH), {ex['window']}), Meta())   # {{"logits": [2, 2]}}
```

`frontend 0, patcher 0` is the expected line, not a warning: those are the two
things your montage supplies.

**The frontend you build is not randomly initialised.** Its filters are seeded
from real wavelet families -- `{wavelets}` -- and learned from there during
fine-tuning. What a fresh frontend costs you is the adaptation those filters
underwent during pretraining, not the wavelet prior itself.

Names, not indices, are what this encoder and your data agree on. Pass real
{spec['names']} and check `channel_vocab_sha256` -- `load_pretrained` refuses a
mismatch rather than silently training on relabelled channels.

Code: <https://github.com/ForeverBlue816/PhysioWave>

---
Prepared by `scripts/prepare_hf_release.py` from `{os.path.basename(run_dir)}`
on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo-id", default=None,
                   help="default ForeverBlue/<EEG|ECG|EMG>")
    p.add_argument("--modality", default="auto", choices=["auto", "eeg", "ecg", "emg"],
                   help="default: read from the checkpoint's trainer")
    p.add_argument("--best", default="best.pth",
                   help="the checkpoint the release is built around and the encoder "
                        "is exported from")
    p.add_argument("--main-label", default=None,
                   help="its row in the card's file table (default: what best.pth "
                        "is, or 'the reported checkpoint')")
    p.add_argument("--main-why", default="",
                   help="one or two sentences for the card on why --best is not "
                        "best.pth")
    p.add_argument("--final", default="latest.pth")
    p.add_argument("--only-best", action="store_true",
                   help="skip the per-criterion checkpoints "
                        "(best_spec/best_raw/best_macro_total)")
    args = p.parse_args(argv)

    run_dir = os.path.expanduser(args.run_dir)
    out_dir = os.path.expanduser(args.out_dir)
    best_src = os.path.join(run_dir, args.best)
    if not os.path.isfile(best_src):
        print(f"ERROR: no checkpoint at {best_src}", file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    modality = args.modality
    if modality == "auto":
        head = torch.load(best_src, map_location="cpu", weights_only=False)
        modality = TRAINER_MODALITY.get((head.get("config") or {}).get("trainer", ""),
                                        head.get("channel_vocab_modality") or "eeg")
        del head
    routes = modality_routes(modality)
    repo_id = args.repo_id or f"ForeverBlue/{ {'eeg': 'EEG', 'ecg': 'ECG', 'emg': 'EMG'}[modality] }"
    args.repo_id = repo_id
    prefix = f"pretrain_{modality}_c1_moe"
    stem = os.path.splitext(args.best)[0]
    main_name = f"{prefix}_{stem}.pth"
    enc_name = f"{modality}_c1_encoder.pth"
    main_label = args.main_label or ("lowest validation **total** loss"
                                     if args.best == "best.pth"
                                     else "the reported checkpoint")
    if args.best != "best.pth" and not args.main_why:
        print(f"  NOTE: --best {args.best} without --main-why; the card will not "
              f"say why this file and not best.pth", file=sys.stderr)

    print(f"release for {args.repo_id} ({SPECS[modality]['label']})\n"
          f"  from {run_dir}\n  into {out_dir}\n")

    best_dst = os.path.join(out_dir, main_name)
    info = strip_checkpoint(best_src, best_dst)
    components, shares = component_table(
        torch.load(best_dst, map_location="cpu", weights_only=False)["model"], routes)
    print(f"  {os.path.basename(best_dst)}   "
          f"{human_bytes(info['src_bytes'])} -> "
          f"{human_bytes(info['dst_bytes'])}   "
          f"(epoch {info['epoch']}, {info['n_params'] / 1e6:.1f} M params)")

    # Every bar, so a downstream comparison can ask which criterion transfers
    # rather than assuming the one that selects best.pth. They are not four
    # different models: several bars are routinely won by the same epoch, and
    # the checkpoints then hold identical weights. Publishing 116 MB twice
    # under two names is not a second data point, so identical ones are
    # detected and recorded as aliases.
    seen = {model_digest(best_dst): main_name}
    aliases, published = {}, []
    final_epoch = None
    extra = [(args.final, f"{prefix}_final.pth", "last epoch")]
    if args.best != "best.pth":
        # The main file is not best.pth, so best.pth is a different bar and
        # is published under its own name, not silently dropped.
        extra.append(("best.pth", f"{prefix}_best.pth", "lowest validation total loss"))
    if not args.only_best:
        extra += [("best_spec.pth", f"{prefix}_best_spec.pth",
                   "lowest spec MSE"),
                  ("best_raw.pth", f"{prefix}_best_raw.pth",
                   "lowest raw SmoothL1"),
                  ("best_macro_total.pth",
                   f"{prefix}_best_macro.pth",
                   "lowest route-macro total")]
    for src_name, dst_name, what in extra:
        src = os.path.join(run_dir, src_name)
        if not os.path.isfile(src):
            print(f"  (no {src_name})")
            continue
        dst = os.path.join(out_dir, dst_name)
        fi = strip_checkpoint(src, dst)
        if src_name == args.final:
            final_epoch = fi.get("epoch")
        digest = model_digest(dst)
        if digest in seen:
            os.remove(dst)
            aliases[dst_name] = seen[digest]
            print(f"  {dst_name}   IDENTICAL to {seen[digest]} "
                  f"(both epoch {fi['epoch']}) -- not published twice")
            published.append((seen[digest], what, fi["epoch"], True))
            continue
        seen[digest] = dst_name
        published.append((dst_name, what, fi["epoch"], False))
        print(f"  {dst_name}   {human_bytes(fi['src_bytes'])} -> "
              f"{human_bytes(fi['dst_bytes'])}   (epoch {fi['epoch']})")

    # One encoder, no route. What transfers is the channel embedding, its gate
    # and the shared transformer; the frontend and patcher load only onto the
    # exact route they were trained for, and both downstream tasks here build
    # their own. Four per-route files would differ only in weights nobody loads.
    enc_dst = os.path.join(out_dir, enc_name)
    r = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "scripts", "export_eeg_pretrained_encoder.py"),
         "--checkpoint", best_src, "--modality", modality, "--output", enc_dst],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR exporting the encoder: {r.stderr[-800:]}",
              file=sys.stderr)
        return 1
    # The export records the checkpoint it came from, absolutely. Harmless in
    # $PW_CKPT_ROOT and not harmless on a public model page.
    enc = torch.load(enc_dst, map_location="cpu", weights_only=False)
    for field in PATH_FIELDS:
        if enc.get(field):
            enc[field] = REDACTED
    if isinstance(enc.get("model_config"), dict):
        enc["model_config"] = scrub_config(enc["model_config"])
    torch.save(enc, enc_dst)
    print(f"  {enc_name}   {human_bytes(os.path.getsize(enc_dst))}   "
          f"(no frontend, no patcher, no decoder)")

    # config_resolved.yaml is TEXT, and copying it verbatim published the
    # manifest paths the .pth files had just been scrubbed of. It is rewritten
    # through the same scrub, not copied.
    src = os.path.join(run_dir, "config_resolved.yaml")
    if os.path.isfile(src):
        try:
            import yaml
            with open(src) as f:
                cfg_yaml = yaml.safe_load(f)
            with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
                yaml.safe_dump(scrub_config(cfg_yaml), f, sort_keys=False)
            print("  config_resolved.yaml   (paths scrubbed)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  SKIPPING config_resolved.yaml: {exc}", file=sys.stderr)

    # metrics_epoch.jsonl is numbers only, but it is checked below like
    # everything else rather than trusted for being numbers.
    src = os.path.join(run_dir, "metrics_epoch.jsonl")
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(out_dir, "metrics_epoch.jsonl"))
        print("  metrics_epoch.jsonl")

    rows = read_metrics(run_dir)
    card = os.path.join(out_dir, "README.md")
    with open(card, "w") as f:
        f.write(model_card(args.repo_id, info, rows, run_dir,
                           components=components, shares=shares,
                           published=published, aliases=aliases,
                           modality=modality, routes=routes, main_name=main_name,
                           enc_name=enc_name, main_label=main_label,
                           main_why=args.main_why, final_epoch=final_epoch))
    print(f"  README.md   (model card, {len(rows)} epochs of metrics read)")

    # Every file, not only the checkpoints. The first version of this check
    # scanned .pth alone and passed while config_resolved.yaml sat beside them
    # carrying the very paths the checkpoints had been scrubbed of.
    leaks = []
    for dp, _dn, fn in os.walk(out_dir):
        for f in fn:
            full = os.path.join(dp, f)
            if f.endswith(".pth"):
                leaks += [(f, w, v) for w, v in find_paths(full)]
            else:
                leaks += [(f, f"line {i}", m)
                          for i, m in text_paths(full)]
    if leaks:
        print("\n  REFUSING: absolute paths survive in what would be "
              "published:", file=sys.stderr)
        for f, where, v in leaks[:10]:
            print(f"    {f}  {where} = {v}", file=sys.stderr)
        print("  Nothing was uploaded. Extend PATH_FIELDS or scrub_config.",
              file=sys.stderr)
        return 1
    print("  no absolute paths survive in any published checkpoint")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(out_dir) for f in fn)
    print(f"\n  total {human_bytes(total)}")
    print(f"""
  NOTHING HAS BEEN UPLOADED. To publish (PRIVATE=1 for a private repo):

      PRIVATE=1 bash scripts/upload_hf_release.sh {out_dir} {args.repo_id}

  Read {card} first -- it is generated, and a generated card is
  still a claim you are making.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
