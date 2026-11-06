#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch PhysioNet ERP-BCI (erpbci 1.0.0) from the S3 mirror, in parallel.
----------------------------------------------------------------------
One command, no credentials, no extra tools:

    python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci

The whole corpus is 2.19 GiB across 245 EDF recordings -- twelve subjects,
roughly twenty runs each -- so unlike Sleep-EDF there is no reason to fetch a
subset. `--subjects` exists anyway for a quick pipeline test.

Unlike Sleep-EDF, this dataset ships its own checksums: SHA256SUMS.txt sits at
the top of the archive and covers every file. They are verified here, because a
recording truncated by a dropped connection is not an error downstream -- MNE
reads it as a short run, the preparation step writes fewer epochs, and the
result looks like data rather than like a failure.

Run it where there is internet. On Leonardo that means a login node.

Next:  python EEG/physio_p300_finetune.py --edf-dir <dest> --out-dir ...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

S3 = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "erpbci/1.0.0/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
MIRROR = "https://physionet.org/files/erpbci/1.0.0/"

# The nine subjects the EEGPT paper keeps: 8, 10 and 12 are dropped to match
# BENDR's configuration. Note that EEGPT's own prepare_PhysioNetP300.py loops
# over [2,3,4,5,6,7,9,11] and omits subject 1, while their LOSO loop in
# linear_probe_EEGPT_PhysioP300.py runs over [1,...,9,11] -- see docs.
PAPER_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 11]


def list_objects(prefix: str = PREFIX) -> list[tuple[str, int]]:
    """Every object under the prefix, as (key, size)."""
    out, token = [], None
    while True:
        url = f"{S3}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for c in root.findall(NS + "Contents"):
            out.append((c.find(NS + "Key").text, int(c.find(NS + "Size").text)))
        if (root.find(NS + "IsTruncated").text or "false") != "true":
            return out
        token = root.find(NS + "NextContinuationToken").text


def subject_of(key: str) -> int | None:
    """erpbci/1.0.0/s07/rc12.edf -> 7."""
    parts = key[len(PREFIX):].split("/")
    if len(parts) == 2 and parts[0].startswith("s") and parts[0][1:].isdigit():
        return int(parts[0][1:])
    return None


def fetch(key: str, size: int, dest: str, mirror: bool,
          timeout: float = 60.0, attempts: int = 4) -> tuple[str, str]:
    """Download one object unless a local file of the right size is already there.

    The timeout is per socket operation rather than per file, so it bounds a
    *stalled* connection and not a slow one: a transfer still delivering bytes
    is never interrupted, and one that has stopped is abandoned and retried
    rather than holding a worker while the rest of the batch queues behind it.
    """
    rel = key[len(PREFIX):]
    path = os.path.join(dest, rel)
    if os.path.exists(path) and os.path.getsize(path) == size:
        return rel, "skipped"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = (MIRROR + urllib.parse.quote(rel)) if mirror else (S3 + urllib.parse.quote(key))
    tmp = path + ".part"
    last = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            got = os.path.getsize(tmp)
            if got != size:
                last = f"short ({got} != {size})"
                continue
            # Rename only once the whole file is on disk, so an interrupted run
            # never leaves something the resume check reads as complete.
            os.replace(tmp, path)
            return rel, "ok" if attempt == 0 else f"ok (attempt {attempt + 1})"
        except Exception as exc:                               # noqa: BLE001
            last = repr(exc)
        finally:
            if os.path.exists(tmp) and not os.path.exists(path):
                os.remove(tmp)
        time.sleep(2 ** attempt)
    return rel, f"failed after {attempts}: {last}"


def verify(dest: str, rels: list[str]) -> int:
    """Check files against the archive's own SHA256SUMS.txt. Returns bad count."""
    sums = os.path.join(dest, "SHA256SUMS.txt")
    if not os.path.isfile(sums):
        print("  SHA256SUMS.txt was not downloaded; cannot verify.")
        return 0
    expected = {}
    with open(sums) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                expected[parts[1].lstrip("./")] = parts[0]

    def sha256(rel):
        h = hashlib.sha256()
        with open(os.path.join(dest, rel), "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return rel, h.hexdigest()

    todo = [r for r in rels
            if r in expected and os.path.exists(os.path.join(dest, r))]
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rel, digest in pool.map(sha256, todo):
            if digest == expected[rel]:
                ok += 1
            else:
                bad += 1
                print(f"  CORRUPT {rel}")
    print(f"  {ok} verified, {bad} corrupt, {len(rels) - len(todo)} not in the checksum list")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True,
                   help="directory to hold s01/ ... s12/; 2.19 GiB for all of it")
    p.add_argument("--subjects", default=None,
                   help="comma-separated ids (1-12), or 'paper' for the nine the "
                        "EEGPT paper keeps. Default is all twelve -- the whole "
                        "corpus is only 2.19 GiB.")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="seconds one socket read may stall before the transfer "
                        "is abandoned and retried")
    p.add_argument("--mirror", action="store_true",
                   help="fetch from physionet.org instead of its S3 mirror")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    # `--dest $PW_DATA_EEG/erpbci` with PW_DATA_EEG unset expands to /erpbci
    # before this process starts, and the failure is then a permission error on
    # the filesystem root rather than anything about the variable.
    if os.path.dirname(os.path.abspath(args.dest)) == os.sep:
        raise SystemExit(
            f"--dest is '{args.dest}' -- directly under the filesystem root.\n\n"
            f"  That is almost always an unset variable: the shell expands\n"
            f"  $PW_DATA_EEG before this script runs, and it is empty unless\n"
            f"  you have sourced the environment:\n\n"
            f"      source scripts/cineca_env.sh\n"
        )
    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    if args.subjects == "paper":
        want = set(PAPER_SUBJECTS)
    elif args.subjects:
        want = {int(s) for s in args.subjects.split(",")}
    else:
        want = None

    print(f"Destination: {dest}")
    print("Listing the mirror...")
    try:
        objects = list_objects()
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(
            f"Cannot reach the mirror: {exc!r}\n\n"
            f"  Compute nodes on Leonardo have no route to the internet. Run\n"
            f"  this on a login node, then train where the GPUs are."
        )
    if want is not None:
        # Top-level files (RECORDS, SHA256SUMS.txt) have no subject and are
        # always kept; the checksum file is what verification needs.
        objects = [(k, s) for k, s in objects
                   if subject_of(k) is None or subject_of(k) in want]
    rels = [k[len(PREFIX):] for k, _ in objects]
    total = sum(s for _, s in objects)
    have = sum(s for k, s in objects
               if os.path.exists(os.path.join(dest, k[len(PREFIX):]))
               and os.path.getsize(os.path.join(dest, k[len(PREFIX):])) == s)
    print(f"{len(objects)} files, {total / 2**30:.2f} GiB "
          f"({have / 2**30:.2f} GiB already present)")

    if not args.verify_only:
        done = 0
        problems = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(fetch, k, s, dest, args.mirror, args.timeout)
                       for k, s in objects]
            for fut in as_completed(futures):
                rel, status = fut.result()
                done += 1
                if status not in ("ok", "skipped"):
                    problems.append((rel, status))
                print(f"\r  {done}/{len(objects)}   {rel:24s}", end="", flush=True)
        print()
        if problems:
            print(f"\n{len(problems)} file(s) did not arrive:", file=sys.stderr)
            for rel, status in problems[:20]:
                print(f"  {rel}: {status}", file=sys.stderr)
            print("Rerun to retry them; complete files are skipped.", file=sys.stderr)

    print("\nVerifying against SHA256SUMS.txt...")
    bad = verify(dest, rels)
    if bad:
        raise SystemExit("Delete the corrupt files and rerun to refetch them.")
    print(f"\nNext:  python EEG/physio_p300_finetune.py --edf-dir {dest} \\\n"
          f"           --out-dir $PW_DATA_EEG/p300 --fold 0")


if __name__ == "__main__":
    main()
