#!/usr/bin/env python
"""
Table of the EEG C1 downstream runs on Sleep-EDF and PhysioNet P300.

    python scripts/collect_eeg_downstream.py --root $PW_CKPT_ROOT/eeg_downstream \\
        [--json summary.json] [--markdown summary.md]

The ECG collector (scripts/collect_ecg_downstream.py) with the EEG tasks: per task
every run -- fine-tuned, probe, scratch, each learning rate -- and, per mode, the
run with the best VALIDATION score marked and compared. The published rows are
EEGPT's table (Wang et al., NeurIPS 2024), on their own protocols.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_ecg_downstream import main  # noqa: E402

SPEC = {
    "heading": "EEG C1 downstream",
    "tasks": ("sleep_edf", "physio_p300"),
    "columns": {
        "sleep_edf": [("balanced_acc", "BalAcc"), ("kappa", "Kappa"),
                      ("weighted_f1", "F1-w"), ("acc", "Acc")],
        "physio_p300": [("balanced_acc", "BalAcc"), ("kappa", "Kappa"),
                        ("auroc", "AUROC"), ("weighted_f1", "F1-w")],
    },
    "titles": {
        "sleep_edf": "Sleep-EDF (Sleep Cassette), 5 stages, Fpz-Cz + Pz-Oz, subject-disjoint holdout",
        "physio_p300": "PhysioNet P300 (erpbci), target vs non-target, subject-disjoint split",
    },
    "paper": {},
    "references": {
        "sleep_edf": [("EEGPT (frozen + 4-layer head)", {"balanced_acc": 0.6917, "kappa": 0.6857})],
        "physio_p300": [("EEGPT (linear probe)", {"balanced_acc": 0.6502, "kappa": 0.2999,
                                                   "auroc": 0.7168}),
                        ("LaBraM (linear probe)", {"balanced_acc": 0.6477, "kappa": 0.2935,
                                                    "auroc": 0.7068})],
    },
    "paper_note": ["EEGPT's table: Sleep-EDF 10-fold, P300 leave-one-subject-out, each",
                   "scored on the fold it also validates on. Here one subject-disjoint",
                   "split with a held-out test set nothing selects on. Indicative only."],
}

if __name__ == "__main__":
    sys.exit(main(spec=SPEC))
