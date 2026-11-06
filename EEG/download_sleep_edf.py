#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch Sleep-EDFx Sleep Cassette from PhysioNet's S3 mirror, in parallel.
------------------------------------------------------------------------
One command, no credentials, no extra tools:

    python EEG/download_sleep_edf.py

MNE would otherwise fetch one recording at a time as the preparation script
asks for it -- serial, no resume, and roughly 6 GB. This pulls the same files
concurrently into the directory MNE reads, so the preparation step downloads
nothing.

It also verifies them, which MNE will not. `mne.datasets.sleep_physionet`'s
`_fetch_one` returns a pre-existing file immediately:

    destination = op.join(path, fname)
    if op.isfile(destination) and not force_update:
        return destination, False

The SHA1 in its records is consulted only when pooch performs the download, so
a file truncated by a dropped connection is read as a short recording -- which
surfaces as a subject with fewer windows and looks like data rather than like
an error. Every file here is checked against MNE's own SHA1SUMS.

Run it where there is internet. On Leonardo that means a login node.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

S3 = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "sleep-edfx/1.0.0/sleep-cassette/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
MIRROR = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

# The subject list EEGPT's finetune script runs its folds over; downloading
# only these is 5.8 GiB rather than 7.1 GiB.
EEGPT_SUBJECTS = [
    0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24,
    25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 42, 44, 45, 46, 47, 48, 49,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74,
    75, 76, 77, 81, 82,
]


def list_objects() -> list[tuple[str, int]]:
    """Every object under the cassette prefix, as (filename, size)."""
    out, token = [], None
    while True:
        url = f"{S3}?list-type=2&prefix={urllib.parse.quote(PREFIX)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for c in root.findall(NS + "Contents"):
            key = c.find(NS + "Key").text
            if key.endswith(".edf"):
                out.append((key.rsplit("/", 1)[-1], int(c.find(NS + "Size").text)))
        if (root.find(NS + "IsTruncated").text or "false") != "true":
            return out
        token = root.find(NS + "NextContinuationToken").text


def subject_of(fname: str) -> int | None:
    """SC4ssNE0 -> ss. The letter after the night digit varies and is not a key."""
    m = re.match(r"SC4(\d{2})(\d)", fname)
    return int(m.group(1)) if m else None


def fetch(fname: str, size: int, dest: str, mirror: bool,
          timeout: float = 60.0, attempts: int = 4) -> tuple[str, str]:
    """Download one object unless a local file of the right size is already there.

    The timeout is per socket operation, not per file, so it bounds a *stalled*
    connection rather than a slow one: a transfer that keeps delivering bytes is
    never interrupted, and one that stops delivering them is abandoned and
    retried instead of holding a worker until the whole batch is stuck behind
    it. S3 drops long-lived connections often enough that this matters over
    250 files.
    """
    path = os.path.join(dest, fname)
    if os.path.exists(path) and os.path.getsize(path) == size:
        return fname, "skipped"
    url = (MIRROR if mirror else S3 + PREFIX) + urllib.parse.quote(fname)
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
            # never leaves something that looks complete to the resume check.
            os.replace(tmp, path)
            return fname, "ok" if attempt == 0 else f"ok (attempt {attempt + 1})"
        except Exception as exc:                               # noqa: BLE001
            last = repr(exc)
        finally:
            if os.path.exists(tmp) and not os.path.exists(path):
                os.remove(tmp)
        time.sleep(2 ** attempt)
    return fname, f"failed after {attempts}: {last}"


def verify(dest: str, names: list[str]) -> int:
    """Check every file against MNE's SHA1SUMS. Returns the number of bad ones."""
    try:
        import mne.datasets.sleep_physionet as sp
    except ImportError:
        print("  mne is not installed, so the files cannot be verified.\n"
              "  pip install braindecode, then rerun with --verify-only.")
        return 0
    sums = os.path.join(os.path.dirname(sp.__file__), "SHA1SUMS")
    if not os.path.isfile(sums):
        print(f"  MNE has no SHA1SUMS at {sums}; cannot verify.")
        return 0
    expected = {}
    with open(sums) as f:
        for line in f:
            parts = line.strip().split("  ")
            if len(parts) == 2 and parts[1].endswith(".edf"):
                expected[os.path.basename(parts[1])] = parts[0]

    def sha1(name):
        h = hashlib.sha1()
        with open(os.path.join(dest, name), "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return name, h.hexdigest()

    todo = [n for n in names if n in expected and os.path.exists(os.path.join(dest, n))]
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name, digest in pool.map(sha1, todo):
            if digest == expected[name]:
                ok += 1
            else:
                bad += 1
                print(f"  CORRUPT {name}")
    unknown = len(names) - len(todo)
    print(f"  {ok} verified, {bad} corrupt, {unknown} not in MNE's list")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=None,
                   help="default: $MNE_DATA/physionet-sleep-data, where MNE looks")
    p.add_argument("--subjects", default=None,
                   help="comma-separated ids, or 'all' for the whole cassette set. "
                        "Default is the 64 EEGPT uses (5.8 GiB rather than 7.1).")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="seconds a single socket read may stall before the "
                        "transfer is abandoned and retried")
    p.add_argument("--mirror", action="store_true",
                   help="fetch from physionet.org instead of its S3 mirror")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    root = os.environ.get("MNE_DATA") or os.path.expanduser("~/mne_data")
    dest = args.dest or os.path.join(root, "physionet-sleep-data")
    # MNE raises rather than creating MNE_DATA, so both levels are made here.
    os.makedirs(dest, exist_ok=True)

    if args.subjects == "all":
        want = None
    elif args.subjects:
        want = {int(s) for s in args.subjects.split(",")}
    else:
        want = set(EEGPT_SUBJECTS)

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
        objects = [(n, s) for n, s in objects if subject_of(n) in want]
    names = [n for n, _ in objects]
    total = sum(s for _, s in objects)
    have = sum(s for n, s in objects
               if os.path.exists(os.path.join(dest, n))
               and os.path.getsize(os.path.join(dest, n)) == s)
    print(f"{len(objects)} files, {total / 2**30:.2f} GiB "
          f"({have / 2**30:.2f} GiB already present)")

    if not args.verify_only:
        done = 0
        problems = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(fetch, n, s, dest, args.mirror, args.timeout)
                       for n, s in objects]
            for fut in as_completed(futures):
                name, status = fut.result()
                done += 1
                if status not in ("ok", "skipped"):
                    problems.append((name, status))
                print(f"\r  {done}/{len(objects)}   {name:28s}", end="", flush=True)
        print()
        if problems:
            print(f"\n{len(problems)} file(s) did not arrive:", file=sys.stderr)
            for name, status in problems[:20]:
                print(f"  {name}: {status}", file=sys.stderr)
            print("Rerun to retry them; files already complete are skipped.",
                  file=sys.stderr)

    print("\nVerifying against MNE's SHA1SUMS...")
    bad = verify(dest, names)
    if bad:
        raise SystemExit("Delete the corrupt files and rerun to refetch them.")
    print("\nNext:  python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf")


if __name__ == "__main__":
    main()
