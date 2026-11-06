#!/usr/bin/env python3
"""
Unpack a zip whose members are deflate64 (method 9), which Python's zipfile
cannot inflate and not every system unzip was built to.

    python scripts/unzip_deflate64.py GESTURE_S1.zip --dest GESTURE_S1

Written for CEMHSEY's GESTURE zips, after a Leonardo compute node had neither
a deflate64-capable unzip nor 7z. The archive directory is read with zipfile
(listing needs no decompression); each deflate64 member is inflated with the
``inflate64`` package (pip install inflate64 -- wheels for Linux and macOS),
streamed to disk, and checked against the CRC-32 and size the directory
records. Stored and deflate members go through zipfile as usual.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import zipfile
import zlib

CHUNK = 8 << 20


def _target(dest: str, name: str) -> str:
    path = os.path.normpath(os.path.join(dest, name))
    if not path.startswith(os.path.abspath(dest) + os.sep) and \
            path != os.path.abspath(dest):
        raise SystemExit(f"refusing a member outside the destination: {name}")
    return path


def _deflate64(fh, info: zipfile.ZipInfo, out_path: str) -> None:
    import inflate64
    fh.seek(info.header_offset)
    head = fh.read(30)
    if head[:4] != b"PK\x03\x04":
        raise IOError(f"{info.filename}: no local header at {info.header_offset}")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    fh.seek(info.header_offset + 30 + name_len + extra_len)
    inf = inflate64.Inflater()
    left, crc, size = info.compress_size, 0, 0
    tmp = out_path + ".part"
    with open(tmp, "wb") as out:
        while left > 0:
            buf = fh.read(min(CHUNK, left))
            if not buf:
                raise IOError(f"{info.filename}: archive ends early")
            left -= len(buf)
            data = inf.inflate(buf)
            crc = zlib.crc32(data, crc)
            size += len(data)
            out.write(data)
    if size != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
        os.unlink(tmp)
        raise IOError(f"{info.filename}: inflated to {size} bytes, CRC "
                      f"{crc & 0xFFFFFFFF:08x}; the directory says "
                      f"{info.file_size} bytes, CRC {info.CRC:08x}")
    os.replace(tmp, out_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip")
    p.add_argument("--dest", required=True)
    args = p.parse_args(argv)
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    n = 0
    with zipfile.ZipFile(args.zip) as zf, open(args.zip, "rb") as fh:
        for info in zf.infolist():
            path = _target(dest, info.filename)
            if info.is_dir():
                os.makedirs(path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if (os.path.isfile(path) and os.path.getsize(path) == info.file_size):
                n += 1
                continue                       # a rerun after an interruption
            if info.compress_type == 9:
                _deflate64(fh, info, path)
            else:
                with zf.open(info) as src, open(path + ".part", "wb") as out:
                    while True:
                        buf = src.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                os.replace(path + ".part", path)
            n += 1
            print(f"  {info.filename}", flush=True)
    print(f"{n} file(s) in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
