"""
Entry point for the EEG C1 route, dispatched from physiowave.train.pretrain_main.

Kept out of pretrain_main so that adding this path costs the WAST/TARE path one
`if`: the two share no model, no objective and no data builder, and threading
this one through the other's loop would have meant making every stage of it
conditional.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from .routes import PRETRAIN_DATASETS


def _merge_manifests(dataset_dirs: Dict[str, str], out_dir: str) -> Dict[str, str]:
    """One manifest per split across all datasets, with shard paths absolute."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for split in ("train", "val"):
        rows: List[str] = []
        for dataset_id, d in dataset_dirs.items():
            p = os.path.join(d, f"manifest_{split}.jsonl")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["path"] = os.path.abspath(
                        rec["path"] if os.path.isabs(rec["path"])
                        else os.path.join(d, os.path.relpath(rec["path"], d)))
                    rec.setdefault("dataset_id", dataset_id)
                    rows.append(json.dumps(rec))
        path = os.path.join(out_dir, f"manifest_{split}.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        written[split] = path
    return written


def build_smoke_corpus(root: str, datasets: Optional[List[str]] = None,
                       subjects: int = 4, recordings: int = 1,
                       windows: int = 4) -> Dict[str, str]:
    """Synthetic shards for every route, so the loop can be exercised offline.

    Reached only from --smoke-test. Every shard it writes carries
    ``"synthetic": true`` in its provenance, and the manifests live under a
    directory named ``smoke_corpus``; nothing in the real path can reach this.
    """
    datasets = datasets or list(PRETRAIN_DATASETS)
    script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "EEG",
        "preprocess_pretrain_corpus.py")
    dirs = {}
    for dataset_id in datasets:
        out = os.path.join(root, dataset_id)
        cmd = [sys.executable, script, "--dataset", "smoke", "--smoke-test",
               "--smoke-as", dataset_id, "--out-dir", out,
               "--smoke-subjects", str(subjects),
               "--smoke-recordings", str(recordings),
               "--smoke-windows", str(windows), "--mains-hz", "50"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"smoke corpus for {dataset_id} failed:\n{r.stderr}")
        dirs[dataset_id] = out
    return _merge_manifests(dirs, os.path.join(root, "merged"))


#: How long a non-zero rank waits for rank 0's smoke corpus. Seven datasets of
#: synthetic shards take a few seconds; five minutes is a bound, not an
#: estimate, and hitting it means rank 0 died rather than that it was slow.
SMOKE_CORPUS_TIMEOUT_S = 300


def _smoke_corpus_once(root: str, build=None) -> Dict[str, str]:
    """Build on rank 0, wait on the rest. Returns the merged manifest paths.

    ``build`` writes the corpus and returns its merged manifests; EEG's
    ``build_smoke_corpus`` by default, and physiowave.ecg_c1 passes its own.

    Called BEFORE ``setup_distributed``, so the rank comes from torchrun's
    environment rather than from a process group that does not exist yet, and
    the wait is a poll rather than a barrier for the same reason.
    """
    merged = os.path.join(root, "merged")
    paths = {split: os.path.join(merged, f"manifest_{split}.jsonl")
             for split in ("train", "val")}
    if int(os.environ.get("RANK", "0")) == 0:
        built = (build or build_smoke_corpus)(root)
        print(f"  smoke corpus (SYNTHETIC) written to {root}", flush=True)
        return built

    deadline = time.time() + SMOKE_CORPUS_TIMEOUT_S
    while time.time() < deadline:
        if all(os.path.isfile(p) and os.path.getsize(p) for p in paths.values()):
            return paths
        time.sleep(0.5)
    raise SystemExit(
        f"rank {os.environ.get('RANK')} waited {SMOKE_CORPUS_TIMEOUT_S}s for "
        f"rank 0 to write {merged} and it did not appear. Rank 0 builds the "
        f"smoke corpus; if it failed, its error is the one to read.")


#: A smoke run's per-rank batch. Two windows a route is enough to exercise
#: every shape; it has to fit on a CPU in seconds.
SMOKE_BATCH_BY_ROUTE = {"E19_256": 2, "E32_512": 2, "E64_256": 2,
                        "E128_512": 2}

#: What to say when the manifests are not set, per modality.
MISSING_MANIFEST_HINT = (
    "Preprocess the corpora first (EEG/preprocess_pretrain_corpus.py, one run "
    "per dataset) and merge their manifests, then point this at the result.")


def run(cfg: Dict, out_dir: str, args) -> int:
    """Build the trainer this config asks for and run it."""
    from .train import EEGC1Trainer
    return run_trainer(cfg, out_dir, args, EEGC1Trainer,
                       smoke_build=build_smoke_corpus,
                       smoke_batch_by_route=SMOKE_BATCH_BY_ROUTE,
                       missing_manifest_hint=MISSING_MANIFEST_HINT)


def run_trainer(cfg: Dict, out_dir: str, args, trainer_cls, smoke_build,
                smoke_batch_by_route: Dict[str, int],
                missing_manifest_hint: str) -> int:
    """The run itself, for any trainer built on EEGC1Trainer.

    Shared with physiowave.ecg_c1, which differs in the trainer class, the
    synthetic corpus a smoke run builds and the routes its batch names -- and
    in nothing about how a run is set up, resumed or initialised.
    """
    from ..train.utils import set_seed, setup_distributed, cleanup_distributed

    smoke = bool(getattr(args, "smoke_test", False))
    if smoke:
        # UNDER THE OUTPUT DIRECTORY, not a system temp dir that is deleted when
        # the process exits. Everything else about a run is reproducible from
        # its own output directory -- config_resolved.yaml, dataset_manifest.json,
        # train_command.txt -- and the corpus was the one exception: the figures
        # need real validation windows, and by the time you could run
        # scripts/visualize_eeg_pretraining.py on a smoke run, the windows it
        # trained on no longer existed, so fig_mask_reconstruction and
        # fig_raw_waveform_reconstruction silently did not appear.
        #
        # It stays unmistakable: the directory is named smoke_corpus, and every
        # shard inside it carries "synthetic": true in its provenance. Delete
        # it when you are done with the run; nothing in the real path reads it.
        #
        # RANK 0 BUILDS IT, the others wait for the manifests. One shared
        # directory and N writers is a race; N private directories -- what the
        # temp dir gave -- is worse still, because the ranks would then train
        # on different corpora and the schedule assumes they agree on every
        # dataset's length.
        corpus = _smoke_corpus_once(os.path.join(out_dir, "smoke_corpus"),
                                    build=smoke_build)
        cfg.setdefault("data", {})
        cfg["data"]["manifest_train"] = corpus["train"]
        cfg["data"]["manifest_val"] = corpus["val"]
        # A smoke run must fit on a CPU in seconds; the standard model does not.
        cfg["model"].update(embed_dim=96, depth=2, num_heads=4,
                            channel_embed_dim=32)
        cfg["train"].update(epochs=int(cfg["train"].get("epochs", 1)) and 1,
                            warmup_epochs=0, grad_accumulation_steps=1,
                            steps_per_epoch=int(getattr(args, "max_steps", 5) or 5),
                            precision="fp32",
                            batch_size_by_route=dict(smoke_batch_by_route))
    else:
        data = cfg.get("data", {})
        for key in ("manifest_train", "manifest_val"):
            if not data.get(key):
                raise SystemExit(
                    f"data.{key} is not set. {missing_manifest_hint} "
                    f"--smoke-test runs on synthetic data instead; nothing "
                    f"falls back to it silently.")

    info = setup_distributed()
    set_seed(int(cfg.get("seed", 42)) + getattr(info, "rank", 0), True)
    try:
        trainer = trainer_cls(cfg, out_dir, info,
                              max_steps=getattr(args, "max_steps", None))
        init_from = getattr(args, "init_from", None)
        resume = getattr(args, "resume", None)
        if init_from and resume:
            raise SystemExit(
                "--init-from and --resume are different operations and cannot "
                "both apply. --resume continues one run; --init-from starts a "
                "new one from another's weights.")
        if init_from:
            if not os.path.isfile(init_from):
                raise SystemExit(f"--init-from {init_from} does not exist")
            trainer.init_from(init_from)
        resumed = False
        if resume:
            path = (os.path.join(out_dir, "latest.pth") if resume == "auto"
                    else resume)
            if os.path.isfile(path):
                trainer.load(path)
                resumed = True
            elif resume != "auto":
                raise SystemExit(f"--resume {path} does not exist")
        # RESUME=auto with no checkpoint yet is a fresh start, not a resume, so
        # this keys off whether one was actually loaded.
        if not resumed:
            trainer.retire_previous_metrics()
        rc = trainer.fit()
        trainer.loader.close()
        return rc
    finally:
        cleanup_distributed()
