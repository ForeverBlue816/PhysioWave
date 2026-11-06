#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What is actually on disk: the .npz caches, and the splits already built.

    python scripts/inspect_p300_inputs.py $PW_DATA_EEG

Answers the three questions that have each cost an allocation:

  * which cache directories hold subjects, and whether the channel names are
    inside the .npz -- a cache without them can still be adopted, by the list
    the directory name implies, but only for the set it was decoded with.
  * what montage and sampling rate each built split carries. A split with no
    `sampling_rate` attribute stops finetune_main before the first batch.
  * whether a split carries CHANNEL METADATA. finetune_main does not need it --
    it resolves ids from channel_names itself -- but the legacy finetune.py,
    which EEG/finetune_p300.sh drives, refuses to start without it as soon as
    --channel_encoding is not 'none'.

Standard library plus numpy and h5py, so it runs wherever the data does.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

#: What finetune.py reads when --channel_encoding is not 'none'. A split built
#: with --no-channel-metadata has channel_names and none of these.
METADATA_KEYS = ("channel_ids", "electrode_xyz", "positive_electrode_index",
                 "negative_electrode_index", "derivation_matrix")


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PW_DATA_EEG", ".")
    import h5py

    print(f"\n  under {base}\n")

    caches = sorted(glob.glob(os.path.join(base, "*", "cache", "c*")))
    print("  CACHES")
    if not caches:
        print("    none found")
    for d in caches:
        npz = sorted(glob.glob(os.path.join(d, "sub*.npz")))
        if not npz:
            print(f"    {d:<52} EMPTY")
            continue
        try:
            with np.load(npz[0]) as z:
                names = "names inside" if "channel_names" in z.files else "NO names (legacy)"
                chans = z["data"].shape[1]
        except Exception as exc:                       # truncated by a killed writer
            print(f"    {d:<52} UNREADABLE ({type(exc).__name__})")
            continue
        print(f"    {d:<52} {len(npz)} subjects, {chans} ch, {names}")

    print("\n  SPLITS")
    splits = sorted(glob.glob(os.path.join(base, "*", "train.h5")))
    if not splits:
        print("    none found")
    for h5 in splits:
        with h5py.File(h5, "r") as f:
            chans = f["data"].shape[1]
            rate = f.attrs.get("sampling_rate")
            first = ([x.decode() for x in f["channel_names"][:3]]
                     if "channel_names" in f else None)
            missing = [k for k in METADATA_KEYS if k not in f]
        d = os.path.basename(os.path.dirname(h5))
        rate_s = f"{rate:g} Hz" if rate is not None else "NO sampling_rate"
        meta_s = "metadata" if not missing else f"NO metadata ({len(missing)} keys)"
        print(f"    {d:<24} {chans:>3} ch  {rate_s:<17} {meta_s:<24} {first}")

    print("\n  A split with NO sampling_rate: pass "
          "--set model.eeg_c1.sampling_rate=<Hz>,")
    print("  or rebuild it -- the converters have written the attribute "
          "unconditionally\n  since f45cd37.")
    print("  A split with NO metadata works with finetune_main and is refused "
          "by the legacy\n  finetune.py unless CHANNEL_ENCODING=none.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
