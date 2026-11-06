"""
The ECG lead vocabulary: one id per lead name, for the C1 channel embedding.

KEPT APART FROM channel_embedding.CHANNEL_VOCAB, deliberately. That list is
append-only and its SHA-256 is written into every EEG checkpoint; the trainer
refuses to load a checkpoint whose recorded hash differs from the current one.
Appending twelve ECG lead names to it would change that hash and make every
EEG checkpoint written so far unloadable -- to add a vocabulary no EEG model
ever uses. So ECG gets its own table, its own hash and its own
``channel_vocab.json`` beside its checkpoints, and the two modalities cannot
relabel each other.

The same rules hold here as there: APPEND ONLY. An id is a row of a trained
embedding table, so reordering, renaming or deleting an entry silently
relabels every lead a checkpoint learned. The list below is deliberately
generous for that reason -- the right-sided and posterior leads, the modified
Holter leads and the Frank leads are here now so that the dataset that needs
them later does not force a new vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Sequence, Tuple

PAD_ID = 0
UNK_ID = 1

#: APPEND ONLY. See the module docstring.
ECG_LEAD_VOCAB: List[str] = [
    "<pad>", "<unk>",
    # -- the standard twelve, in the conventional order ----------------------
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
    # -- single-lead ambulatory recordings -----------------------------------
    # Icentia11k's CardioSTAT is one bipolar lead from a patch on the upper
    # chest. It is NOT lead I, whatever it resembles: a different electrode
    # pair sees the heart along a different vector, and sharing an embedding
    # row would tell the model the two are the same measurement.
    "patch1",
    # -- leads other corpora record, reserved now ----------------------------
    "V3R", "V4R", "V5R", "V7", "V8", "V9",
    "MLI", "MLII", "MLIII", "MV1", "MV2", "MV5",
    "X", "Y", "Z",
]
ECG_LEAD_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(ECG_LEAD_VOCAB)}
assert len(ECG_LEAD_TO_ID) == len(ECG_LEAD_VOCAB), "duplicate ECG lead name"

#: The twelve standard leads, in the order every 12-lead route uses.
LEADS_12: Tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF",
                             "V1", "V2", "V3", "V4", "V5", "V6")

#: The six frontal-plane leads. Any two of them determine the other four
#: EXACTLY (Einthoven: III = II - I; Goldberger: aVR = -(I + II)/2, ...), so a
#: masked limb-lead patch whose time-aligned neighbours on two other limb leads
#: are visible is not inferred, it is computed. The ECG pretrainer masks these
#: six together for that reason -- see physiowave.ecg_c1.model.
LIMB_LEADS: Tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")

#: Spellings the source files use, mapped onto the canonical names above.
#: Keyed by the UPPER-CASE form with separators removed; see normalize_lead.
_ALIASES: Dict[str, str] = {
    "I": "I", "II": "II", "III": "III",
    "LI": "I", "LII": "II", "LIII": "III",
    "LEADI": "I", "LEADII": "II", "LEADIII": "III",
    # CODE (Brazil) writes the limb leads as DI, DII, DIII.
    "DI": "I", "DII": "II", "DIII": "III",
    "D1": "I", "D2": "II", "D3": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    **{f"V{i}": f"V{i}" for i in range(1, 10)},
    "V3R": "V3R", "V4R": "V4R", "V5R": "V5R",
    "MLI": "MLI", "MLII": "MLII", "MLIII": "MLIII",
    "MV1": "MV1", "MV2": "MV2", "MV5": "MV5",
    "X": "X", "Y": "Y", "Z": "Z",
    "PATCH1": "patch1",
}


def normalize_lead(name: str) -> str:
    """A source file's lead label -> the canonical vocabulary name.

    Case, whitespace, dashes and underscores are ignored, and CODE's DI/DII/DIII
    are the limb leads they are. A name this does not recognise is returned
    unchanged, so it resolves to <unk> and is REPORTED by the slot mapping
    rather than guessed onto a lead it might not be.
    """
    key = re.sub(r"[\s_\-]", "", str(name)).upper()
    if key.startswith("LEAD") and key[4:] in _ALIASES:
        key = key[4:]
    return _ALIASES.get(key, str(name).strip())


def lead_id(name: str) -> int:
    return ECG_LEAD_TO_ID.get(normalize_lead(name), UNK_ID)


def lead_ids_for(names: Sequence[str]) -> Tuple[List[int], List[str]]:
    """``(ids, names that resolved to <unk>)``."""
    ids, unknown = [], []
    for n in names:
        i = lead_id(n)
        ids.append(i)
        if i == UNK_ID:
            unknown.append(str(n))
    return ids, unknown


def ecg_vocab_sha256() -> str:
    return hashlib.sha256(
        json.dumps(ECG_LEAD_VOCAB, ensure_ascii=True).encode()).hexdigest()


def ecg_vocab_payload() -> Dict[str, object]:
    """What a checkpoint records about the vocabulary its embedding indexes.

    The same two keys the EEG checkpoints carry, so the shared resume check
    (``EEGC1Trainer._check_vocab``) compares like with like -- and an ECG
    checkpoint offered to an EEG run, or the reverse, fails on the hash rather
    than loading with every row meaning something else.
    """
    return {"channel_vocab_size": len(ECG_LEAD_VOCAB),
            "channel_vocab_sha256": ecg_vocab_sha256(),
            "channel_vocab_modality": "ecg"}


def save_ecg_vocab(path: str) -> None:
    with open(path, "w") as f:
        json.dump({"vocab": ECG_LEAD_VOCAB, **ecg_vocab_payload()}, f, indent=2)
