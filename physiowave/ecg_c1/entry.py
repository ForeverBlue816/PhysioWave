"""
Entry point for the ECG C1 route, dispatched from physiowave.train.pretrain_main.

The run itself -- smoke corpus on rank 0, distributed setup, resume versus
init-from -- is ``physiowave.eeg_c1.entry.run_trainer``. This names the ECG
trainer, the ECG synthetic corpus and the ECG routes' smoke batch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional

from ..eeg_c1.entry import _merge_manifests, run_trainer
from .routes import PRETRAIN_DATASETS

SMOKE_BATCH_BY_ROUTE = {"L12_500": 2, "L1_250": 4}

MISSING_MANIFEST_HINT = (
    "Download the corpora (scripts/download_ecg_pretrain_corpora.sh), "
    "preprocess each one (ECG/preprocess_ecg_corpus.py) and merge the "
    "manifests (scripts/build_eeg_c1_manifest.py --modality ecg), then point "
    "this at the result.")


def build_smoke_corpus(root: str, datasets: Optional[List[str]] = None,
                       records: int = 12) -> Dict[str, str]:
    """Synthetic ECG shards for every dataset, so the loop runs offline.

    Written by ECG/preprocess_ecg_corpus.py's own pipeline -- the filters, the
    resampling, the QC, the normalisation and the shard writer are the real
    ones; only the signal is made up. Every shard carries ``"synthetic": true``
    and lives under a directory named ``smoke_corpus``.
    """
    datasets = datasets or list(PRETRAIN_DATASETS)
    script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "ECG",
        "preprocess_ecg_corpus.py")
    dirs = {}
    for dataset_id in datasets:
        out = os.path.join(root, dataset_id)
        cmd = [sys.executable, script, "--dataset", dataset_id, "--smoke-test",
               "--out-dir", out, "--smoke-records", str(records),
               # Small shards, so the smoke corpus has several of them and the
               # multi-shard reader is exercised too.
               "--windows-per-shard", "8", "--val-fraction", "0.25"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"smoke corpus for {dataset_id} failed:\n"
                             f"{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        dirs[dataset_id] = out
    return _merge_manifests(dirs, os.path.join(root, "merged"))


def run(cfg: Dict, out_dir: str, args) -> int:
    from .train import ECGC1Trainer
    return run_trainer(cfg, out_dir, args, ECGC1Trainer,
                       smoke_build=build_smoke_corpus,
                       smoke_batch_by_route=SMOKE_BATCH_BY_ROUTE,
                       missing_manifest_hint=MISSING_MANIFEST_HINT)
