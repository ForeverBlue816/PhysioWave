"""
The sEMG electrode vocabulary: one id per electrode position, for the C1 channel
embedding.

Kept apart from the EEG and ECG vocabularies for the reason those two are kept
apart: each checkpoint records its vocabulary's hash, and a shared, growing
list would make every earlier checkpoint unloadable the day it grew.

APPEND ONLY -- an id is a row of a trained embedding table.

What a name means, per montage:

* ``band01`` .. ``band16`` -- the electrode pairs of Meta's sEMG-RD wristband,
  in the band's own channel order. emg2pose and emg2qwerty record with the same
  band, so the same index is the same electrode in both. Left and right wrists
  share these names: the band is the same object on either wrist, and a dataset
  carries one montage for all its shards, so a side-specific name would need
  the corpus split by side.
* ``putemg_r{1-3}e{1-8}`` -- putEMG's three rings of eight around the forearm.
* ``hyser_r{1-8}c{1-8}``, ``cemhsey_r{1-8}c{1-8}`` -- the position of an
  electrode within one 8x8 high-density grid. Each grid of a recording is its
  own sample on the 64-channel route, so these name a position IN a grid, not
  which muscle the grid lies on. The two datasets do not share names: nothing
  says their grids have the same orientation or inter-electrode distance.
* ``cemhsey13_r{1-5}c{1-13}`` -- a position in one of CEMHSEY's two 5x13
  grids (OT Bioelettronica GR08MM1305, 8 mm), whose corner r5c1 holds no
  electrode: 64 names. CEMHSEY's 8x8 grids are GR10MM0808 (10 mm). Where a
  channel of each sits is the map in the dataset's own PreProcess.m, copied
  below; it is what turns a file's row number into one of these names.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Sequence, Tuple

PAD_ID = 0
UNK_ID = 1

BAND_16: Tuple[str, ...] = tuple(f"band{i:02d}" for i in range(1, 17))
PUTEMG_24: Tuple[str, ...] = tuple(f"putemg_r{r}e{e}"
                                   for r in range(1, 4) for e in range(1, 9))


def grid_8x8(prefix: str) -> Tuple[str, ...]:
    """Row-major names of one 8x8 grid: ``<prefix>_r1c1`` .. ``<prefix>_r8c8``."""
    return tuple(f"{prefix}_r{r}c{c}" for r in range(1, 9) for c in range(1, 9))


HYSER_GRID: Tuple[str, ...] = grid_8x8("hyser")
CEMHSEY_GRID: Tuple[str, ...] = grid_8x8("cemhsey")

#: Channel number (1-64) at each position of the two OT Bioelettronica grids
#: CEMHSEY records with, as PreProcess.m (CEMHSEY's MyFunction.zip) has them
#: for ElectrodeType 2 (GR10MM0808) and the unrotated ElectrodeType 1/4
#: (GR08MM1305). 0 is the 5x13 grid's empty corner.
_OT_8X8 = [[8, 16, 24, 32, 40, 48, 56, 64],
           [7, 15, 23, 31, 39, 47, 55, 63],
           [6, 14, 22, 30, 38, 46, 54, 62],
           [5, 13, 21, 29, 37, 45, 53, 61],
           [4, 12, 20, 28, 36, 44, 52, 60],
           [3, 11, 19, 27, 35, 43, 51, 59],
           [2, 10, 18, 26, 34, 42, 50, 58],
           [1, 9, 17, 25, 33, 41, 49, 57]]
_OT_5X13 = [[4, 5, 11, 10, 24, 32, 34, 39, 40, 49, 50, 62, 61],
            [3, 6, 12, 9, 23, 31, 33, 38, 48, 41, 51, 63, 60],
            [2, 7, 13, 17, 22, 30, 27, 37, 47, 42, 52, 64, 59],
            [1, 8, 14, 18, 21, 29, 26, 36, 46, 43, 53, 56, 58],
            [0, 16, 15, 19, 20, 28, 25, 35, 45, 44, 54, 55, 57]]

CEMHSEY_GRID_5X13: Tuple[str, ...] = tuple(
    f"cemhsey13_r{r + 1}c{c + 1}" for r in range(5) for c in range(13)
    if _OT_5X13[r][c])


def _by_channel(table, prefix: str) -> Tuple[str, ...]:
    pos = {ch: (r + 1, c + 1) for r, row in enumerate(table)
           for c, ch in enumerate(row) if ch}
    assert sorted(pos) == list(range(1, 65)), "a grid map is not 1..64"
    return tuple(f"{prefix}_r{pos[k][0]}c{pos[k][1]}" for k in range(1, 65))


#: Names of a CEMHSEY grid's 64 rows, in the file's channel order.
CEMHSEY_8X8_BY_CHANNEL: Tuple[str, ...] = _by_channel(_OT_8X8, "cemhsey")
CEMHSEY_5X13_BY_CHANNEL: Tuple[str, ...] = _by_channel(_OT_5X13, "cemhsey13")

#: APPEND ONLY. See the module docstring.
EMG_ELECTRODE_VOCAB: List[str] = (
    ["<pad>", "<unk>"] + list(BAND_16) + list(PUTEMG_24)
    + list(HYSER_GRID) + list(CEMHSEY_GRID) + list(CEMHSEY_GRID_5X13)
)
EMG_ELECTRODE_TO_ID: Dict[str, int] = {n: i for i, n in
                                       enumerate(EMG_ELECTRODE_VOCAB)}
assert len(EMG_ELECTRODE_TO_ID) == len(EMG_ELECTRODE_VOCAB), "duplicate name"


def normalize_electrode(name: str) -> str:
    """A reader's channel label -> the vocabulary name. Unknown stays as given.

    Readers already label channels with these names; this only forgives case
    and the zero-padding of band indices (``band1`` -> ``band01``).
    """
    s = str(name).strip()
    m = re.fullmatch(r"band0*(\d+)", s, re.I)
    if m:
        return f"band{int(m.group(1)):02d}"
    low = s.lower()
    return low if low in EMG_ELECTRODE_TO_ID else s


def electrode_ids_for(names: Sequence[str]) -> Tuple[List[int], List[str]]:
    ids, unknown = [], []
    for n in names:
        i = EMG_ELECTRODE_TO_ID.get(normalize_electrode(n), UNK_ID)
        ids.append(i)
        if i == UNK_ID:
            unknown.append(str(n))
    return ids, unknown


def emg_vocab_sha256() -> str:
    return hashlib.sha256(
        json.dumps(EMG_ELECTRODE_VOCAB, ensure_ascii=True).encode()).hexdigest()


def emg_vocab_payload() -> Dict[str, object]:
    return {"channel_vocab_size": len(EMG_ELECTRODE_VOCAB),
            "channel_vocab_sha256": emg_vocab_sha256(),
            "channel_vocab_modality": "emg"}


def save_emg_vocab(path: str) -> None:
    with open(path, "w") as f:
        json.dump({"vocab": EMG_ELECTRODE_VOCAB, **emg_vocab_payload()}, f,
                  indent=2)
