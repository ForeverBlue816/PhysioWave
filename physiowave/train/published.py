"""Published downstream numbers, and the protocol they were produced under.

One copy, because these are quoted in tables and a second copy is a copy that
can drift. Everything here is stdlib-only so a collection script can import it
in whatever environment happens to be active.

THE PROTOCOL MATTERS AS MUCH AS THE NUMBER. EEGPT's rows below come from a
setup that differs from ours in ways that move a score by more than the gap
between two of these rows, and every one of those differences is recorded in
``EEGPT_PROTOCOL`` so a comparison can say which of them it has matched:

* no test set. Their downstream scripts build ``train_dataset`` and
  ``valid_dataset``, run with ``enable_checkpointing=False``, and report the
  validation fold -- the set the run is monitored on. A held-out test set that
  nothing selected on is a strictly harder number.
* fold-averaged. PhysioP300 is a nine-fold LOSO mean, Sleep-EDFx a ten-fold
  subject-wise CV mean, each repeated three times. We report a single
  subject-disjoint split, which is a noisier estimate of the same quantity and
  not the same number. Their published +- is the spread of three repeats of a
  whole mean and is not an error bar a single split has an equivalent of.
* the probe is not only a linear layer. See ``EEGPT_PROTOCOL``.
"""

from __future__ import annotations

from typing import Dict, Optional

EEGPT = "EEGPT (NeurIPS'24)"

#: EEGPT Table 4. The third column is "Weighted F1 / AUROC": AUROC for the
#: binary tasks, weighted F1 for the multi-class ones.
PUBLISHED: Dict[str, Dict[str, Dict[str, float]]] = {
    "p300": {
        EEGPT:               {"balanced_acc": 0.6502, "kappa": 0.2999, "auroc": 0.7168},
        "LaBraM (ICLR'24)":  {"balanced_acc": 0.6477, "kappa": 0.2935, "auroc": 0.7068},
        "BENDR":             {"balanced_acc": 0.6114, "kappa": 0.2227, "auroc": 0.6588},
        "BIOT (NeurIPS'23)": {"balanced_acc": 0.5485, "kappa": 0.0968, "auroc": 0.5308},
    },
    "sleep": {
        EEGPT:               {"balanced_acc": 0.6917, "kappa": 0.6857, "weighted_f1": 0.7654},
        "LaBraM (ICLR'24)":  {"balanced_acc": 0.6771, "kappa": 0.6710, "weighted_f1": 0.7592},
        "BENDR":             {"balanced_acc": 0.6655, "kappa": 0.6659, "weighted_f1": 0.7507},
        "BIOT (NeurIPS'23)": {"balanced_acc": 0.6622, "kappa": 0.6461, "weighted_f1": 0.7415},
    },
}

#: Metrics to show for each task, in the order EEGPT reports them.
TASK_METRICS = {
    "p300": (("auroc", "AUROC"), ("balanced_acc", "BalAcc"), ("kappa", "Kappa")),
    "sleep": (("balanced_acc", "BalAcc"), ("kappa", "Kappa"),
              ("weighted_f1", "W-F1")),
}

#: What a metric reads at chance, so a number can be placed against the floor
#: rather than against zero. Plain accuracy has no entry: its floor is the
#: majority class, which is a property of the split (0.833 on P300) and not of
#: the metric, and quoting it as 0.5 is how a majority-class classifier looks
#: like a result.
CHANCE = {"balanced_acc": None, "kappa": 0.0, "auroc": 0.5, "weighted_f1": None}

#: Their downstream recipe, read off the paper (Appendix C.2.5, C.2.4, D) and
#: off ``downstream/linear_probe_EEGPT_PhysioP300.py`` and
#: ``downstream/finetune_EEGPT_SleepEDF.py`` in BINE022/EEGPT.
EEGPT_PROTOCOL = {
    "p300": {
        "source": "linear_probe_EEGPT_PhysioP300.py",
        "epochs": 100,
        "batch_size": 64,
        "optimizer": "AdamW, weight_decay=0.01",
        "schedule": "OneCycleLR max_lr=8e-4, pct_start=0.2",
        "loss": "CrossEntropyLoss, no label smoothing",
        "monitor": "AUROC",
        "trainable": ("chan_scale (58) + Linear(2048,16) + Linear(240,2) "
                      "= ~33k params, dropout 0.5"),
        "pooling": ("per-position projection then flatten over the 15 time "
                    "positions -- the time axis is NOT pooled away"),
        "split": "9-subject LOSO; the held-out subject is the validation set",
        "window": "-0.1 s to 2.0 s at 256 Hz (538 samples), patch 64 stride 32",
    },
    "sleep": {
        "source": "finetune_EEGPT_SleepEDF.py",
        "epochs": 40,
        "batch_size": 32,
        "optimizer": "AdamW, weight_decay=0.01",
        "schedule": "OneCycleLR max_lr=4e-4, pct_start=0.2",
        "loss": "CrossEntropyLoss, no label smoothing",
        "monitor": "Cohen's kappa",
        "trainable": ("Conv1d(2,13,1) + Linear(2048,64) + a 4-layer "
                      "TransformerDecoder(d=64) + cls token + Linear(64,5) "
                      "= ~330k params, dropout 0.5"),
        "pooling": ("a cls token attends over the 0.25 s positions -- this is "
                    "a sequence model on top of the encoder, not a linear head"),
        "split": "10-fold subject-wise CV over 64 subjects; no test set",
        "window": "30 s, resampled 100 Hz -> 256 Hz by interpolation",
    },
}


def task_of(name: str) -> Optional[str]:
    """``p300``/``sleep`` from a config name, run name or output directory."""
    low = str(name).lower()
    if "p300" in low or "erp" in low:
        return "p300"
    if "sleep" in low or "edf" in low:
        return "sleep"
    return None
