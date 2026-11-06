#!/usr/bin/env python3
"""
Download large files over HTTP in parallel ranged pieces, resumably.

    python scripts/fetch_ranged.py --url URL --dest PATH [--jobs 16]
    python scripts/fetch_ranged.py --list files.tsv [--jobs 16]   # url<TAB>dest[<TAB>bytes]

Written for the sEMG corpora: emg2pose is one 431 GiB tar and emg2qwerty one
287 GiB tar.gz, which on a single connection is hours a login node may not
allow, and Zenodo serves CEMHSEY at ~0.6 MB/s a connection. Every piece is its
own ranged request written straight into place in the destination file, so
there is no second copy to concatenate, and the pieces already written are
recorded in ``<dest>.chunks`` -- a rerun after an interruption fetches only
the rest. The sidecar is removed when the file is whole, so a file at its full
size with no sidecar beside it is complete.

A partial file left by ``curl -C -`` (no sidecar, shorter than the full size)
keeps its prefix: curl writes in order, so every whole piece inside it is good.

Credentials in the URL (``https://user:pass@host/...``) are sent as basic
auth, which is how putEMG's public WebDAV share is reached. Standard library
only, so it runs in any Python on a login node.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

UA = "PhysioWave-fetch/1.0"


def _request(url: str, start: Optional[int] = None,
             end: Optional[int] = None) -> urllib.request.Request:
    p = urllib.parse.urlsplit(url)
    headers = {"User-Agent": UA}
    if p.username is not None:
        cred = f"{urllib.parse.unquote(p.username)}:" \
               f"{urllib.parse.unquote(p.password or '')}"
        headers["Authorization"] = "Basic " + base64.b64encode(
            cred.encode()).decode()
        netloc = p.hostname + (f":{p.port}" if p.port else "")
        url = urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, ""))
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.Request(url, headers=headers)


def remote_size(url: str) -> int:
    """The total from a one-byte ranged GET; servers that refuse HEAD answer it."""
    with urllib.request.urlopen(_request(url, 0, 0), timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if r.status != 206 or not cr:
            raise IOError(f"{url}: no ranged response (status {r.status}); "
                          f"this server cannot be fetched in pieces")
        return int(cr.rsplit("/", 1)[1])


class Target:
    def __init__(self, url: str, dest: str, size: Optional[int], chunk: int):
        self.url, self.dest, self.chunk = url, dest, chunk
        self.size = size if size is not None else remote_size(url)
        self.n = max(1, -(-self.size // chunk))
        self.side = dest + ".chunks"
        self.lock = threading.Lock()
        self.done: set = set()
        self.fd: Optional[int] = None

    def complete_on_disk(self) -> bool:
        return (os.path.isfile(self.dest) and not os.path.exists(self.side)
                and os.path.getsize(self.dest) == self.size)

    def open(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.dest)), exist_ok=True)
        have = os.path.getsize(self.dest) if os.path.isfile(self.dest) else 0
        if os.path.exists(self.side):
            with open(self.side) as f:
                self.done = {int(x) for x in f.read().split() if x.strip()}
        elif have:
            # A plain sequential download's prefix: every whole piece in it.
            self.done = {i for i in range(self.n)
                         if min((i + 1) * self.chunk, self.size) <= have}
            with open(self.side, "w") as f:
                f.write("".join(f"{i}\n" for i in sorted(self.done)))
        else:
            open(self.side, "w").close()
        self.fd = os.open(self.dest, os.O_RDWR | os.O_CREAT, 0o644)
        if have > self.size:
            raise IOError(f"{self.dest} is {have} bytes, larger than the "
                          f"{self.size} the server has: not this file")
        os.ftruncate(self.fd, self.size)

    def todo(self) -> List[int]:
        return [i for i in range(self.n) if i not in self.done]

    def mark(self, i: int) -> None:
        with self.lock:
            self.done.add(i)
            with open(self.side, "a") as f:
                f.write(f"{i}\n")

    def finish(self) -> bool:
        if len(self.done) < self.n:
            return False
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = None
        os.unlink(self.side)
        return True


def fetch_piece(t: Target, i: int, tries: int = 10) -> int:
    a = i * t.chunk
    b = min(a + t.chunk, t.size) - 1
    for attempt in range(tries):
        try:
            pos = a
            with urllib.request.urlopen(_request(t.url, a, b), timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"status {r.status} to a ranged request")
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    if pos + len(buf) > b + 1:
                        raise IOError("server sent more than the range")
                    os.pwrite(t.fd, buf, pos)
                    pos += len(buf)
            if pos != b + 1:
                raise IOError(f"piece {i}: {pos - a} of {b + 1 - a} bytes")
            os.fsync(t.fd)
            t.mark(i)
            return b + 1 - a
        except Exception as exc:                               # noqa: BLE001
            if attempt == tries - 1:
                raise IOError(f"{os.path.basename(t.dest)} piece {i}: {exc}")
            # 429 / 5xx from Zenodo want a real pause, not a hammering.
            wait = 30 if isinstance(exc, urllib.error.HTTPError) else 5
            time.sleep(wait * (attempt + 1))
    return 0


def run(targets: List[Target], jobs: int) -> int:
    pending: List[Tuple[Target, int]] = []
    total = 0
    for t in targets:
        if t.complete_on_disk():
            print(f"  {os.path.basename(t.dest)}: complete ({t.size:,} bytes)")
            continue
        t.open()
        todo = t.todo()
        print(f"  {os.path.basename(t.dest)}: {t.size / 1e9:.2f} GB, "
              f"{len(todo)}/{t.n} piece(s) to fetch", flush=True)
        pending += [(t, i) for i in todo]
        total += sum(min((i + 1) * t.chunk, t.size) - i * t.chunk for i in todo)
    failed: List[str] = []
    got = 0
    t0 = last = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch_piece, t, i): (t, i) for t, i in pending}
        for fut in as_completed(futs):
            try:
                got += fut.result()
            except Exception as exc:                           # noqa: BLE001
                failed.append(str(exc))
            now = time.time()
            if now - last > 30 or got == total:
                last = now
                rate = got / max(1e-9, now - t0) / 1e6
                eta = (total - got) / max(1.0, rate * 1e6)
                print(f"  {got / 1e9:8.2f} / {total / 1e9:.2f} GB  "
                      f"{rate:6.1f} MB/s  eta {eta / 3600:5.1f} h  "
                      f"{len(failed)} piece(s) failed", flush=True)
    incomplete = [t.dest for t in targets
                  if t.fd is not None and not t.finish()]
    if failed or incomplete:
        print(f"ERROR: {len(failed)} piece(s) failed (e.g. {failed[:2]}); "
              f"{len(incomplete)} file(s) incomplete. Rerun to fetch just "
              f"the missing pieces.", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url")
    p.add_argument("--dest")
    p.add_argument("--size", type=int, default=None,
                   help="expected bytes (default: ask the server)")
    p.add_argument("--list", help="TSV: url, dest[, bytes] per line")
    p.add_argument("--jobs", type=int, default=16,
                   help="connections at once, over all files")
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--check-only", action="store_true",
                   help="fetch nothing: exit 0 if every file is whole, else "
                        "say what is missing (needs --size / list sizes, so "
                        "it runs without a network)")
    args = p.parse_args(argv)
    chunk = args.chunk_mb << 20
    rows: List[Tuple[str, str, Optional[int]]] = []
    if args.list:
        with open(args.list) as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0]:
                    rows.append((parts[0], parts[1],
                                 int(parts[2]) if len(parts) > 2 and parts[2]
                                 else None))
    elif args.url and args.dest:
        rows.append((args.url, args.dest, args.size))
    else:
        p.error("--url and --dest, or --list")
    if args.check_only:
        missing = 0
        for url, dest, size in rows:
            if size is None:
                print(f"ERROR: --check-only needs the size of {dest}",
                      file=sys.stderr)
                return 2
            t = Target(url, dest, size, chunk)
            if t.complete_on_disk():
                continue
            missing += 1
            done = 0
            if os.path.exists(t.side):
                with open(t.side) as f:
                    done = len({x for x in f.read().split() if x.strip()})
            print(f"  {os.path.basename(dest)}: incomplete ({done}/{t.n} "
                  f"pieces)" if os.path.exists(dest) else
                  f"  {os.path.basename(dest)}: not downloaded")
        print(f"  {len(rows) - missing}/{len(rows)} file(s) whole")
        return 1 if missing else 0
    targets: List[Target] = []
    sizes: Dict[str, int] = {}
    for url, dest, size in rows:
        try:
            targets.append(Target(url, dest, size, chunk))
        except Exception as exc:                               # noqa: BLE001
            print(f"ERROR: {url}: {exc}", file=sys.stderr)
            return 1
        sizes[dest] = targets[-1].size
    print(f"  {len(targets)} file(s), {sum(sizes.values()) / 1e9:.1f} GB, "
          f"{args.jobs} connection(s), {args.chunk_mb} MB pieces", flush=True)
    return run(targets, args.jobs)


if __name__ == "__main__":
    sys.exit(main())
