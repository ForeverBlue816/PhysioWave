#!/usr/bin/env python3
"""
Check a re-hosted PhysioNet zip against PhysioNet's own checksums.

    python scripts/verify_zip_checksums.py MIMIC.zip \\
        --official https://physionet.org/files/mimic-iv-ecg/1.0/SHA256SUMS.txt

A mirror's copy (Kaggle, here) carries a SHA256SUMS.txt of its own, and a copy
that vouches for itself proves nothing. So three checks, each cheap enough for
a login node:

  1. the mirror's SHA256SUMS.txt starts with the same bytes as PhysioNet's
     (the first 2 MB -- PhysioNet serves ranges, and its full 176 MB file at
     PhysioNet's current speed would take an hour);
  2. every file that list names is present in the zip, at nothing more;
  3. a random sample of the files hashes to the value the list gives.
     --sample 0 hashes every one of them (all 90 GB inflated).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import urllib.request
import zipfile


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip")
    p.add_argument("--official", default=None,
                   help="URL of the publisher's SHA256SUMS.txt")
    p.add_argument("--head-bytes", type=int, default=2 * 1024 * 1024)
    p.add_argument("--sample", type=int, default=2000,
                   help="files to hash; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    zf = zipfile.ZipFile(args.zip)
    names = {i.filename for i in zf.infolist() if not i.is_dir()}
    sums_member = min((n for n in names if n.endswith("SHA256SUMS.txt")),
                      key=len, default=None)
    if sums_member is None:
        print("ERROR: no SHA256SUMS.txt in the zip", file=sys.stderr)
        return 1
    base = sums_member[: -len("SHA256SUMS.txt")]
    raw = zf.read(sums_member)
    print(f"  {len(names):,} files in the zip; checksum list {sums_member} "
          f"({len(raw) / 1e6:.0f} MB)")

    if args.official:
        req = urllib.request.Request(
            args.official, headers={"Range": f"bytes=0-{args.head_bytes - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            head = r.read()
        if raw[:len(head)] != head:
            print(f"ERROR: the zip's SHA256SUMS.txt differs from the official "
                  f"one within its first {len(head):,} bytes. This is not the "
                  f"published dataset.", file=sys.stderr)
            return 1
        print(f"  checksum list matches {args.official} "
              f"(first {len(head) / 1e6:.1f} MB)")

    expected = {}
    for line in raw.decode().splitlines():
        if line.strip():
            digest, path = line.split(None, 1)
            expected[base + path.strip()] = digest
    missing = [n for n in expected if n not in names]
    if missing:
        print(f"ERROR: {len(missing):,} of {len(expected):,} listed files are "
              f"not in the zip, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    print(f"  all {len(expected):,} listed files are present")

    pool = sorted(expected)
    pick = pool if args.sample == 0 else random.Random(args.seed).sample(
        pool, min(args.sample, len(pool)))
    bad = []
    for i, name in enumerate(pick, 1):
        h = hashlib.sha256()
        with zf.open(name) as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected[name]:
            bad.append(name)
        if i % 20000 == 0:
            print(f"    hashed {i:,}/{len(pick):,}", flush=True)
    if bad:
        print(f"ERROR: {len(bad)} of {len(pick):,} hashed files differ, e.g. "
              f"{bad[:3]}", file=sys.stderr)
        return 1
    print(f"  {len(pick):,} file(s) hashed, all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
