#!/usr/bin/env python
"""
Table of the sEMG C1 downstream runs, with the paper's PhysioWave (v1) row.

    python scripts/collect_emg_downstream.py --root $PW_CKPT_ROOT/emg_downstream \\
        [--json summary.json] [--markdown summary.md]

The ECG collector (scripts/collect_ecg_downstream.py) with the sEMG tasks: per
task every run -- fine-tuned, probe, scratch, each learning rate -- and, per
mode, the run with the best VALIDATION score marked and compared.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ecg_downstream import main  # noqa: E402

COLS = [("acc", "Acc"), ("balanced_acc", "BalAcc"), ("macro_f1", "F1-mac"),
        ("kappa", "Kappa")]
SPEC = {
    "heading": "sEMG C1 downstream",
    "tasks": ("epn612", "epn612_xuser", "grabmyo", "grabmyo_xsubj", "ninapro_db5",
              "ninapro_db2", "uci_emg", "uci_emg_xsess"),
    "columns": {t: COLS for t in ("epn612", "epn612_xuser", "grabmyo",
                                  "grabmyo_xsubj", "ninapro_db5", "ninapro_db2",
                                  "uci_emg", "uci_emg_xsess")},
    "titles": {
        "epn612": "EMG-EPN-612, 6 gestures, user-dependent (the paper's protocol; paper: accuracy)",
        "epn612_xuser": "EMG-EPN-612, 6 gestures, user-independent (test users never trained on)",
        "grabmyo": "GRABMyo, 17 classes, inter-session (sessions 1-2 train, session 3 tests)",
        "grabmyo_xsubj": "GRABMyo, 17 classes, subject-independent",
        "ninapro_db5": "NinaPro DB5, exercises A+B+C, 52 movements + rest, two Myo at 200 Hz (reps 1,3,4,6 / 2,5)",
        "ninapro_db2": "NinaPro DB2, exercises B+C+D, 49 movements + rest (reps 1,3,4,6 / 2,5)",
        "uci_emg": "UCI EMG for gestures (Lobov et al. 2018), 6 gestures, subject-independent (36 subjects, 60/20/20)",
        "uci_emg_xsess": "UCI EMG for gestures, 6 gestures, inter-session (series 1 trains, series 2 tests)",
    },
    # PhysioWave (v1), README "Benchmark Results".
    "paper": {"epn612": {"acc": 0.945}},
    "paper_note": ["v1 used max-abs normalisation, 1024-sample inputs and its own",
                   "training recipe on the same split. Indicative, not like-for-like."],
}

if __name__ == "__main__":
    sys.exit(main(spec=SPEC))
