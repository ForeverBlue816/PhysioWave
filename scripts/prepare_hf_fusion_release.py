#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assemble a Hugging Face release from a finished multimodal fusion run
(physiowave/fusion/train.py), the companion of prepare_hf_release.py.

    python scripts/prepare_hf_fusion_release.py \
        --run-dir $PW_CKPT_ROOT/fusion/ralf_frozen_szo \
        --out-dir ~/hf_release_fusion --repo-id ForeverBlue/RALF-SeizeIT2

    PRIVATE=1 bash scripts/upload_hf_release.sh ~/hf_release_fusion ForeverBlue/RALF-SeizeIT2

A fusion best.pth is {"model", "epoch", "score", "config"}: the whole model --
the three C1 encoders as they were loaded, the montage adapters, the fusion
core and the task heads -- with the run's config. That config names the
encoder checkpoints and the prepared data by absolute cluster path. The
release keeps the weights as they are and rewrites those paths: the encoders
become the file names they are published under in the three encoder repos,
and the data directory a placeholder, so `build_model` can rebuild the model
from the released config once the reader fills in where their data is.

Published beside the weights, all scrubbed and leak-checked like the encoder
releases: results.json (the test matrix), events_cross.json (the event-level
curves), history.json, encoders.json (which keys each encoder took),
config_resolved.yaml, and a model card written from those files.

NOTHING IS UPLOADED. This writes a directory and prints the command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from prepare_hf_release import (REDACTED, _scrub, find_paths,  # noqa: E402
                                human_bytes, text_paths)

MODALITIES = ("eeg", "ecg", "emg")
ENCODER_REPO = {"eeg": "ForeverBlue/EEG", "ecg": "ForeverBlue/ECG", "emg": "ForeverBlue/EMG"}
DATA_PLACEHOLDER = "<the task's prepared windows: train.h5, val.h5, test.h5>"
FA_BUDGETS = (2.0, 6.22, 11.0)
SUBSETS = [("complete", "EEG + ECG + EMG"), ("eeg_ecg", "EEG + ECG"), ("eeg_emg", "EEG + EMG"),
           ("ecg_emg", "ECG + EMG"), ("eeg_only", "EEG"), ("ecg_only", "ECG"),
           ("emg_only", "EMG")]
CORRUPTIONS = [("eeg_noise", "EEG noise"), ("eeg_bad_channel", "EEG bad channel"),
               ("eeg_drift", "EEG drift"), ("ecg_saturation", "ECG saturation"),
               ("ecg_noise", "ECG noise"), ("emg_dropout", "EMG dropout"),
               ("emg_noise", "EMG noise")]


def released_encoder_name(modality: str, path: str) -> str:
    """The file a pretraining checkpoint is published as by prepare_hf_release.py:
    <run>/best.pth -> pretrain_<m>_c1_moe_best.pth, <run>/epoch38.pth ->
    pretrain_<m>_c1_moe_epoch38.pth."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return f"pretrain_{modality}_c1_moe_{stem}.pth"


def release_config(cfg: dict) -> dict:
    """The run's config with the encoder paths pointing at the released files
    and the data directories replaced by a placeholder; any other absolute
    path is redacted by the shared scrub."""
    cfg = json.loads(json.dumps(cfg, default=str))
    enc = cfg.get("encoders") or {}
    for m in MODALITIES:
        if isinstance(enc.get(m), str) and enc[m]:
            enc[m] = released_encoder_name(m, enc[m])
    for spec in ((cfg.get("data") or {}).get("tasks") or {}).values():
        if isinstance(spec, dict) and spec.get("dir"):
            spec["dir"] = DATA_PLACEHOLDER
    return _scrub(cfg)


def load_json(path: str):
    return json.load(open(path)) if os.path.isfile(path) else None


def sens_at(curve, budget: float):
    ok = [r["official"]["sens_ovlp"] for r in curve
          if r["official"]["fa_ovlp_h"] == r["official"]["fa_ovlp_h"]
          and r["official"]["fa_ovlp_h"] <= budget]
    return 100 * max(ok) if ok else None


def f3(v, fmt="{:.3f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "n/a"


def model_card(repo_id: str, run: str, ck: dict, cfg: dict, results: dict, events: dict,
               n_params: int, files: list) -> str:
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
    enc = cfg.get("encoders") or {}
    fusion = mcfg.get("fusion", "?")
    mode = mcfg.get("encoder_mode", "?")
    tasks = (results or {}).get("tasks", {})
    task = "seizeit2" if "seizeit2" in tasks else next(iter(tasks), "?")
    test = (tasks.get(task) or {}).get("test", {})
    sel = ((cfg.get("data") or {}).get("tasks") or {}).get(task, {}) or {}
    select_by = sel.get("select_by", "?")

    ev_rows = ""
    if events and "complete" in (events.get("conditions") or {}):
        c = events["conditions"]["complete"]
        cur = c.get("curve") or []
        meta = (f"{c.get('n_recordings', '?')} recordings, "
                f"{c.get('hours', 0):,.0f} h, {c.get('n_seizures', '?')} seizures")
        ev_rows = "\n".join(
            [f"| sensitivity at <= {b:g} false alarms/h | {f3(sens_at(cur, b), '{:.1f} %')} |"
             for b in FA_BUDGETS]
            + [f"| AUROC (the dataset paper's epoch-based area) | {f3(c.get('auroc_paper'))} |",
               f"| AUPR (the dataset paper's epoch-based area) | {f3(c.get('aupr_paper'), '{:.4f}')} |"])
        ev_rows = (f"Continuous test recordings ({meta}), the SeizeIT2 paper's scoring "
                   f"(any-overlap, means over recordings):\n\n| metric | value |\n|---|---|\n"
                   + ev_rows)

    sub_rows = "\n".join(f"| {lab} | {f3((test.get(k) or {}).get('auprc'))} | "
                         f"{f3((test.get(k) or {}).get('auroc'))} |"
                         for k, lab in SUBSETS if k in test)
    rel_head = "| condition | AUPRC | reliability EEG | ECG | EMG |\n|---|---|---|---|---|"
    rel_rows = "\n".join(
        f"| {lab} | {f3((test.get(k) or {}).get('auprc'))} | "
        + " | ".join(f3((test.get(k) or {}).get(f"reliability_{m}"), "{:.2f}")
                     for m in MODALITIES) + " |"
        for k, lab in [("complete", "clean")] + CORRUPTIONS if k in test)
    sev = (results or {}).get("corruption_severity")

    enc_rows = "\n".join(
        f"| {m.upper()} | `{enc[m]}` | {ENCODER_REPO[m]} |" for m in MODALITIES if enc.get(m))
    file_rows = "\n".join(f"| `{n}` | {what} |" for n, what in files)
    classes = (tasks.get(task) or {}).get("classes")

    return f"""---
license: apache-2.0
tags:
  - multimodal
  - physiowave
  - physiowave-t
  - eeg
  - ecg
  - emg
  - seizure-detection
  - sensor-fusion
  - biosignal
library_name: pytorch
---

# PhysioWave-T-{fusion.upper() if fusion == 'ralf' else fusion}: multimodal seizure detection on SeizeIT2 (EEG + ECG + EMG)

The **PhysioWave-T** multimodal model, built on the three released encoders ([EEG](https://huggingface.co/ForeverBlue/EEG), [ECG](https://huggingface.co/ForeverBlue/ECG), [sEMG](https://huggingface.co/ForeverBlue/EMG)). Code: <https://github.com/ForeverBlue816/PhysioWave>.

Three independently pretrained C1 encoders -- EEG, ECG and sEMG -- fused by
**{'RALF (Reliability-Aware Latent Fusion)' if fusion == 'ralf' else fusion}** for
{'seizure detection on SeizeIT2 (behind-the-ear EEG, ECG and deltoid EMG)' if task == 'seizeit2' else task}.
Each encoder hands the fusion a few summary tokens; RALF predicts a
reliability per modality and sample, gates the cross-attention with it, masks
absent modalities out, and adds a residual of reliability-weighted unimodal
logits.

Encoders **{mode}** (the pretrained cores fixed; montage adapters, pools,
fusion core and heads trained), {n_params / 1e6:.1f} M parameters in all.
Run `{run}`, selected on validation {select_by} at epoch {ck.get('epoch', '?')}
counted from 0 (validation {select_by} {f3(ck.get('score'))}).
Classes: {classes if classes else '?'}.

## Test results

SeizeIT2's official split: patients sub-097 to sub-125 are the test set and
were never seen before the final evaluation; 12% of the training patients
were held out for checkpoint selection.

{ev_rows}

Window-level test set (4 s windows; background subsampled to 30 per hour):

| modalities present | AUPRC | AUROC |
|---|---|---|
{sub_rows}

Under corruption of one modality at severity {sev if sev is not None else '?'},
on top of the complete set, with the reliability the model assigned to each
modality:

{rel_head}
{rel_rows}

A corrupted modality's reliability drops and the other two stay where they
were; that is the mechanism the fusion is built around.

## Files

| file | what it is |
|---|---|
{file_rows}

## The encoders it was built on

| modality | file | repository |
|---|---|---|
{enc_rows}

The weights here already contain all three encoders as they were loaded, so
they are not needed to RUN the model. They are needed to BUILD it:
`build_model` constructs each encoder from its pretraining checkpoint before
the fusion weights are loaded on top. Download the three files into one
directory and point `encoders.<m>` at them (the released config already
names them).

## Loading

The model's montage adapters and heads are sized by the task's data, so it is
rebuilt from the prepared windows (`Fusion/seizeit2_prep.py` in the code
repository) and then loaded strictly:

```python
import torch
from physiowave.config import load_config
from physiowave.fusion.train import build_model, load_tasks

cfg = load_config("config_resolved.yaml")
cfg["data"]["tasks"]["{task}"]["dir"] = "path/to/{task}"     # train.h5 / val.h5 / test.h5
cfg["encoders"] = {{m: f"path/to/encoders/{{f}}" for m, f in cfg["encoders"].items()
                   if m in ("eeg", "ecg", "emg") and f}}

tasks = load_tasks(cfg, list(cfg["data"]["modalities"]))
model, _ = build_model(cfg, tasks, out_dir="encoders_built")
ck = torch.load("{files[0][0]}", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])     # strict: every key must match
model.eval()
```

Only `train.h5` is read to size the model; nothing is trained.

## Training

```
fusion      {fusion}, width {mcfg.get('embed_dim', '?')}, {mcfg.get('num_fusion_tokens', '?')} fusion tokens, depth {mcfg.get('depth', '?')}, heads {mcfg.get('num_heads', '?')}
summary     {mcfg.get('summary_tokens', '?')} tokens per modality
encoders    {mode}
lr          {tcfg.get('lr', '?')}, batch {tcfg.get('batch_size', '?')}, epochs {tcfg.get('epochs', '?')} (patience {tcfg.get('patience', '?')}), {tcfg.get('precision', '?')}
loss        task CE + {tcfg.get('uni_weight', '?')} x unimodal CE + {tcfg.get('reliability_weight', '?')} x reliability MSE + {tcfg.get('consistency_weight', '?')} x subset consistency
augment     each modality corrupted with p {(cfg.get('corruption') or {}).get('prob', '?')}; modalities dropped with p {tcfg.get('modality_dropout', '?')}
seed        {cfg.get('seed', '?')}
```

One run, one seed. The numbers above are that run's, not a mean.

Code: <https://github.com/ForeverBlue816/PhysioWave> (`docs/fusion.md`)

---
Prepared by `scripts/prepare_hf_fusion_release.py` from `{run}` on
{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo-id", default="ForeverBlue/RALF-SeizeIT2")
    p.add_argument("--checkpoint", default="best.pth")
    args = p.parse_args(argv)

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_dir = os.path.expanduser(args.out_dir)
    run = os.path.basename(run_dir)
    src = os.path.join(run_dir, args.checkpoint)
    if not os.path.isfile(src):
        print(f"ERROR: no checkpoint at {src}", file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    print(f"release for {args.repo_id}\n  from {run_dir}\n  into {out_dir}\n")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ck:
        print("ERROR: the checkpoint has no 'model' state dict", file=sys.stderr)
        return 1
    cfg = release_config(ck.get("config") or {})
    task = next(iter(((cfg.get("data") or {}).get("tasks") or {}).keys()), "task")
    results = load_json(os.path.join(run_dir, "results.json"))
    if results:
        task = next(iter(results.get("tasks", {})), task)
    fusion = (cfg.get("model") or {}).get("fusion", "fusion")
    name = f"{fusion}_{task}_best.pth"

    sd = ck["model"]
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    out = {"model": sd, "epoch": ck.get("epoch"), "score": ck.get("score"), "config": cfg,
           "run": run, "released_from": args.checkpoint,
           "released_utc": datetime.now(timezone.utc).isoformat(),
           "encoder_repositories": ENCODER_REPO}
    dst = os.path.join(out_dir, name)
    torch.save(out, dst)
    print(f"  {name}   {human_bytes(os.path.getsize(src))} -> "
          f"{human_bytes(os.path.getsize(dst))}   (epoch {ck.get('epoch')}, "
          f"{n_params / 1e6:.1f} M params)")
    files = [(name, f"the model: encoders, adapters, fusion core and heads "
                    f"(epoch {ck.get('epoch')})")]

    for fn, what in (("results.json", "the test matrix: every modality subset and corruption"),
                     ("events_cross.json", "event-level scoring on the continuous test recordings"),
                     ("history.json", "training and validation per epoch"),
                     ("encoders.json", "which pretrained keys each encoder took")):
        d = load_json(os.path.join(run_dir, fn))
        if d is None:
            print(f"  (no {fn})")
            continue
        with open(os.path.join(out_dir, fn), "w") as f:
            json.dump(_scrub(d), f, indent=1)
        files.append((fn, what))
        print(f"  {fn}   (paths scrubbed)")

    try:
        import yaml
        with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        files.append(("config_resolved.yaml", "the run's config; encoders point at the released files"))
        print("  config_resolved.yaml   (encoders renamed, data path replaced)")
    except Exception as exc:                                         # noqa: BLE001
        print(f"  SKIPPING config_resolved.yaml: {exc}", file=sys.stderr)

    events = load_json(os.path.join(run_dir, "events_cross.json"))
    card = os.path.join(out_dir, "README.md")
    with open(card, "w") as f:
        f.write(model_card(args.repo_id, run, ck, cfg, results, events, n_params, files))
    print("  README.md   (model card)")

    leaks = []
    for dp, _dn, fn in os.walk(out_dir):
        for f in fn:
            full = os.path.join(dp, f)
            if f.endswith(".pth"):
                leaks += [(f, w, v) for w, v in find_paths(full)]
            else:
                leaks += [(f, f"line {i}", m) for i, m in text_paths(full)
                          if m != REDACTED]
    if leaks:
        print("\n  REFUSING: absolute paths survive in what would be published:",
              file=sys.stderr)
        for f, where, v in leaks[:10]:
            print(f"    {f}  {where} = {v}", file=sys.stderr)
        return 1
    print("  no absolute paths survive in any published file")
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(out_dir) for f in fn)
    print(f"\n  total {human_bytes(total)}")
    print(f"""
  NOTHING HAS BEEN UPLOADED. To publish privately:

      PRIVATE=1 bash scripts/upload_hf_release.sh {out_dir} {args.repo_id}

  Read {card} first.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
