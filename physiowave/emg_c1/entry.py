"""
Entry point for the sEMG C1 route, dispatched from physiowave.train.pretrain_main.
The run itself is physiowave.eeg_c1.entry.run_trainer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional

from ..eeg_c1.entry import _merge_manifests, run_trainer
from .routes import PRETRAIN_DATASETS

SMOKE_BATCH_BY_ROUTE = {"W16_2000": 4, "A24_2000": 4, "G64_2000": 2}

MISSING_MANIFEST_HINT = (
    "Download the corpora (scripts/download_emg_pretrain_corpora.sh), "
    "preprocess each one (EMG/preprocess_emg_corpus.py) and merge the "
    "manifests (scripts/build_eeg_c1_manifest.py --modality emg), then point "
    "this at the result.")


def build_smoke_corpus(root: str, datasets: Optional[List[str]] = None,
                       records: int = 8) -> Dict[str, str]:
    """Synthetic sEMG shards for every dataset, written by the real pipeline."""
    datasets = datasets or list(PRETRAIN_DATASETS)
    script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "EMG",
        "preprocess_emg_corpus.py")
    dirs = {}
    for dataset_id in datasets:
        out = os.path.join(root, dataset_id)
        cmd = [sys.executable, script, "--dataset", dataset_id, "--smoke-test",
               "--out-dir", out, "--smoke-records", str(records),
               "--windows-per-shard", "4", "--val-fraction", "0.25"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"smoke corpus for {dataset_id} failed:\n"
                             f"{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        dirs[dataset_id] = out
    return _merge_manifests(dirs, os.path.join(root, "merged"))


def run(cfg: Dict, out_dir: str, args) -> int:
    from .train import EMGC1Trainer
    return run_trainer(cfg, out_dir, args, EMGC1Trainer,
                       smoke_build=build_smoke_corpus,
                       smoke_batch_by_route=SMOKE_BATCH_BY_ROUTE,
                       missing_manifest_hint=MISSING_MANIFEST_HINT)
