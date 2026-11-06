#!/usr/bin/env python3
"""
Mirror WFDB records from a PhysioNet HTTP directory, many at a time.

    python scripts/fetch_physionet_http.py \\
        --base https://physionet.org/files/challenge-2021/1.0.3/training/cpsc_2018/ \\
        --dest $ECG_ROOT/downstream/cpsc2018/raw/cpsc_2018 --jobs 16

For the sets that are not on PhysioNet's open S3 bucket (CPSC 2018 lives only
inside Challenge 2021). PhysioNet's HTTP server gives one connection
~20-40 KB/s, so records are fetched in parallel.

The record list comes from RECORDS files: <base>/RECORDS if there is one,
otherwise <base>/g1/RECORDS, g2/... until one is missing (Challenge 2021's
layout). For each record the header is fetched first and the signal files it
names after it. A file already present at a non-zero size is skipped, and a
download is written under .part and renamed when complete, so a rerun after an
interruption fetches only what is missing. Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

UA = {"User-Agent": "PhysioWave-fetch/1.0"}


def get(url: str, tries: int = 6) -> Optional[bytes]:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == tries - 1:
                raise
        except Exception:                                      # noqa: BLE001
            if attempt == tries - 1:
                raise
        time.sleep(3 * (attempt + 1))
    return None


def records(base: str) -> List[str]:
    """Record paths relative to base, from its RECORDS file(s)."""
    top = get(base + "RECORDS")
    if top is not None:
        out = []
        for ln in top.decode().split():
            if ln.endswith("/"):
                out += [ln + r for r in records(base + ln)]
            else:
                out.append(ln)
        return out
    out, g = [], 1
    while True:
        sub = get(f"{base}g{g}/RECORDS")
        if sub is None:
            break
        out += [f"g{g}/{r}" for r in sub.decode().split()]
        g += 1
    return out


def save(url: str, path: str) -> int:
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return 0
    data = get(url)
    if data is None:
        raise IOError(f"404 {url}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return len(data)


def fetch_record(base: str, dest: str, rec: str) -> Tuple[str, int]:
    folder = os.path.dirname(rec)
    hea = os.path.join(dest, rec + ".hea")
    n = save(base + rec + ".hea", hea)
    with open(hea, encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f.read().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    files = []
    for ln in lines[1:]:
        name = ln.split()[0]
        if name not in files:
            files.append(name)
    for name in files:
        rel = f"{folder}/{name}" if folder else name
        n += save(base + rel, os.path.join(dest, rel))
    return rec, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="directory URL, ending in /")
    p.add_argument("--dest", required=True)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args(argv)
    base = args.base if args.base.endswith("/") else args.base + "/"
    t0 = time.time()
    recs = records(base)
    if args.max_records:
        recs = recs[:args.max_records]
    if not recs:
        print(f"ERROR: no RECORDS under {base}", file=sys.stderr)
        return 1
    print(f"  {len(recs):,} record(s) under {base} (listed in {time.time() - t0:.0f}s)",
          flush=True)
    done = got = 0
    failed: List[str] = []
    last = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch_record, base, args.dest, r): r for r in recs}
        for fut in cf.as_completed(futs):
            done += 1
            try:
                got += fut.result()[1]
            except Exception as exc:                           # noqa: BLE001
                failed.append(f"{futs[fut]}: {exc}")
            if time.time() - last > 30 or done == len(recs):
                last = time.time()
                rate = got / max(1e-9, last - t0) / 1e6
                print(f"  {done:,}/{len(recs):,} records  {got / 1e6:.0f} MB new  "
                      f"{rate:.2f} MB/s  {len(failed)} failed", flush=True)
    if failed:
        print(f"ERROR: {len(failed)} record(s) failed, e.g. {failed[:3]}. Rerun "
              f"to fetch just those.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
