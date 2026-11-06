#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch BCI Competition IV 2a and 2b from BNCI Horizon 2020, and verify them.

    python EEG/download_eegpt_benchmarks.py --dataset 2a --dest $PW_DATA_EEG/bcic_iv2a
    python EEG/download_eegpt_benchmarks.py --dataset 2b --dest $PW_DATA_EEG/bcic_iv2b

BNCI data sets 001-2014 and 004-2014 are the competition recordings, openly
downloadable, with the evaluation sessions' labels inside the files. 18 files
each (nine subjects x T/E), ~770 MB for 2a and ~420 MB for 2b.

Resumable: a file whose size matches the server's Content-Length is skipped, and
anything else is fetched again into a .part file that is renamed only once it
is complete. A truncated file is the failure this guards against -- scipy's
loadmat on a cut-off .mat raises somewhere in the converter, an hour later,
with an error that says nothing about the download.

KaggleERN is not here: it needs a Kaggle account. See EEG/download_kaggle_ern.sh.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# bnci-horizon-2020.eu redirects here; going direct saves two round trips per
# file and one hostname that can be down independently.
MIRROR = "https://lampx.tugraz.at/~bci/database"
SETS = {
    "2a": ("001-2014", [f"A{s:02d}{p}.mat" for s in range(1, 10) for p in "TE"]),
    "2b": ("004-2014", [f"B{s:02d}{p}.mat" for s in range(1, 10) for p in "TE"]),
}


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", -1))


def fetch(url: str, dest: str, retries: int = 4) -> str:
    size = remote_size(url)
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        return f"ok      {os.path.basename(dest)}  (already complete)"
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = os.path.getsize(part)
            if size >= 0 and got != size:
                raise IOError(f"{got} bytes, expected {size}")
            os.replace(part, dest)
            return f"fetched {os.path.basename(dest)}  {got / 2**20:.1f} MB"
        except Exception as exc:                              # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f"{os.path.basename(dest)}: {exc}") from exc
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SETS))
    p.add_argument("--dest", required=True)
    p.add_argument("--jobs", type=int, default=6)
    args = p.parse_args(argv)

    code, files = SETS[args.dataset]
    os.makedirs(args.dest, exist_ok=True)
    print(f"BCIC-IV-{args.dataset} (BNCI {code}) -> {args.dest}")
    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(fetch, f"{MIRROR}/{code}/{name}",
                          os.path.join(args.dest, name)): name for name in files}
        for fut in as_completed(futs):
            try:
                print("  " + fut.result(), flush=True)
            except Exception as exc:                          # noqa: BLE001
                failed.append(futs[fut])
                print(f"  FAILED  {exc}", file=sys.stderr, flush=True)
    if failed:
        print(f"{len(failed)} file(s) failed: {sorted(failed)}. Re-run to resume.",
              file=sys.stderr)
        return 1
    print(f"all {len(files)} files present and complete in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
