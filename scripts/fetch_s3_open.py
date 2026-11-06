#!/usr/bin/env python3
"""
Mirror part of a public S3 prefix, in parallel, verified by size.

    python scripts/fetch_s3_open.py --bucket physionet-open \\
        --prefix icentia11k-continuous-ecg/1.0/ --dest $ECG_ROOT/Icentia11k/raw \\
        --include '_s(00|05|10|15|20|25|30|35|40|45)\\.(hea|dat)$' --jobs 16

Written because PhysioNet's own download endpoints were measured at ~0.04
MB/s per connection (0.27 MB/s over sixteen) on 2026-09-29, while the same
files on its open S3 bucket arrive at full speed. Standard library only, so it
runs in any Python on a login node; no AWS CLI and no credentials -- the
bucket is anonymous.

The listing is the manifest: every key's size comes from it, a file whose size
already matches is skipped, and a download is written under a temporary name
and renamed only once it has the listed length. Rerunning after an
interruption therefore fetches exactly what is missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Tuple

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:                                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_prefix(base: str, prefix: str, delimiter: str = "") -> Tuple[List, List]:
    """``([(key, size)], [common prefixes])`` for one prefix, all pages."""
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(_get(f"{base}/?{urllib.parse.urlencode(q)}"))
        for c in root.findall(f"{NS}Contents"):
            keys.append((c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)))
        for p in root.findall(f"{NS}CommonPrefixes"):
            prefixes.append(p.find(f"{NS}Prefix").text)
        if root.findtext(f"{NS}IsTruncated") != "true":
            return keys, prefixes
        token = root.findtext(f"{NS}NextContinuationToken")


def fetch(base: str, key: str, size: int, dest: str) -> Tuple[str, int]:
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        return "skip", 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.part"
    url = f"{base}/{urllib.parse.quote(key)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, \
                    open(tmp, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if os.path.getsize(tmp) != size:
                raise IOError(f"got {os.path.getsize(tmp)} of {size} bytes")
            os.replace(tmp, dest)
            return "ok", size
        except Exception as exc:                               # noqa: BLE001
            if attempt == 4:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                return f"failed: {exc}", 0
            time.sleep(2 * (attempt + 1))
    return "failed", 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True, help="ends with /")
    p.add_argument("--dest", required=True)
    p.add_argument("--include", default=None,
                   help="regex a key must match (searched, not anchored)")
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args(argv)

    # A bucket with a dot in its name (openneuro.org) cannot be virtual-hosted
    # over TLS -- the wildcard certificate does not cover it -- so it is
    # addressed path-style.
    base = (f"https://s3.amazonaws.com/{args.bucket}" if "." in args.bucket
            else f"https://{args.bucket}.s3.amazonaws.com")
    inc = re.compile(args.include) if args.include else None
    t0 = time.time()
    # One level of sub-prefixes first, then those listed in parallel: a single
    # paginated listing of 1.6 M keys is ~1,600 requests in a row.
    top, subs = list_prefix(base, args.prefix, delimiter="/")
    keys = list(top)
    with cf.ThreadPoolExecutor(max_workers=min(16, max(1, len(subs)))) as ex:
        for ks, _ in ex.map(lambda s: list_prefix(base, s), subs):
            keys.extend(ks)
    if inc:
        keys = [(k, s) for k, s in keys if inc.search(k)]
    total = sum(s for _, s in keys)
    print(f"  {len(keys):,} file(s), {total / 1e9:.1f} GB under "
          f"s3://{args.bucket}/{args.prefix} (listed in "
          f"{time.time() - t0:.0f}s)", flush=True)
    if not keys:
        print("ERROR: nothing matched", file=sys.stderr)
        return 1

    done = fetched = skipped = 0
    failed: List[str] = []
    t1 = last = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch, base, k, s,
                          os.path.join(args.dest, k[len(args.prefix):])): k
                for k, s in keys}
        for fut in cf.as_completed(futs):
            status, n = fut.result()
            done += 1
            fetched += n
            if status == "skip":
                skipped += 1
            elif status != "ok":
                failed.append(f"{futs[fut]}: {status}")
            if time.time() - last > 30 or done == len(keys):
                last = time.time()
                rate = fetched / max(1e-9, last - t1) / 1e6
                print(f"  {done:,}/{len(keys):,} files  {fetched / 1e9:.1f} GB "
                      f"new  {rate:.1f} MB/s  {skipped:,} already here  "
                      f"{len(failed)} failed", flush=True)
    if failed:
        print(f"ERROR: {len(failed)} file(s) failed, e.g. {failed[:3]}. "
              f"Rerun to fetch just those.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
