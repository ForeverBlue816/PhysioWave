#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove machine- and account-specific details from the tracked files.

    python scripts/scrub_private.py            # rewrite in place, print a summary
    python scripts/scrub_private.py --check    # exit 1 if anything is left

What goes: the cluster account and usernames, absolute paths on the cluster
and on the workstation, compute-node host names, and GPU model names. Paths
become the variables the scripts already use ($SCRATCH, $FAST, $PW_FAST_ROOT,
$HOME), so a shell command in a document still reads correctly.

The values the cluster jobs need are not lost, they move out of the
repository: scripts/cineca_env.sh sources ~/.physiowave_env, and sbatch takes
the account from $SBATCH_ACCOUNT (see docs/cineca_setup.md).

Run it before pushing to the public repository; --check is the gate.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Ordered: the longest, most specific path first, so a prefix never eats it.
REPLACEMENTS = [
    (r"/leonardo_scratch/fast/IscrB_WearUsFM/yanlchen", "$PW_FAST_ROOT"),
    (r"/leonardo_scratch/fast/IscrB_WearUsFM", "$FAST"),
    (r"/leonardo_scratch/large/userexternal/ychen003", "$SCRATCH"),
    (r"/leonardo/home/userexternal/ychen003", "$HOME"),
    (r"/Users/foreverblue/Desktop/PhysioWave-T/", ""),
    (r"/Users/foreverblue", "$HOME"),
    (r"\$FAST/yanlchen", "$PW_FAST_ROOT"),
    (r"\$\{PW_FAST\}/yanlchen", "${PW_FAST}/${PW_FAST_USER:-$USER}"),
    (r"(?m)^#SBATCH -A iscrb_wearusfm\n", ""),
    (r"-A iscrb_wearusfm", "-A <account>"),
    (r"(?i)iscrb_wearusfm", "<account>"),
    (r"\bychen003@login\d+\.leonardo\.local\b", "<user>@<login-node>"),
    (r"\bychen003\b", "<user>"),
    (r"\byanlchen@student\.ethz\.ch\b", "yanlong.chen@ntu.edu.sg"),
    (r"\byanlchen\b", "<user>"),
    (r"\blrdn\d{4}\b", "<node>"),
    (r"\blogin\d{2}\.leonardo\.local\b", "<login-node>"),
    # GPU models, in prose and in recorded device strings
    (r"NVIDIA (?:A100|H100|V100|L40S?|A40|A6000|RTX ?\w+)[-\w]*", "GPU"),
    (r"\b(?:four|4|one|1|16|sixteen) A100s?\b", "GPUs"),
    (r"Leonardo's A100s", "Leonardo's GPUs"),
    (r"\bA100s?\b(?![\w-])", "GPU"),
    (r"\b(?:H100|V100)s?\b", "GPU"),
]

#: What must not survive (checked after the rewrite, and by --check).
FORBIDDEN = re.compile(r"(?i)(yanlchen|ychen003|iscrb|wearusfm|userexternal|leonardo_scratch|"
                       r"/Users/foreverblue|\blrdn\d|\bA100\b|\bH100\b|\bV100\b|A100-SXM)")

#: Files that only document the scrub itself.
SKIP = {"scripts/scrub_private.py"}


def tracked_text_files():
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    for rel in (p.decode() for p in out if p):
        if rel in SKIP:
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            head = f.read(8192)
        if b"\0" in head:            # binary
            continue
        yield rel, path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="change nothing; fail if any remain")
    args = p.parse_args(argv)
    changed, left = [], []
    for rel, path in tracked_text_files():
        try:
            text = open(path, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        new = text
        if not args.check:
            for pat, rep in REPLACEMENTS:
                new = re.sub(pat, lambda _m, r=rep: r, new)
            if new != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
                changed.append(rel)
        for i, line in enumerate(new.splitlines(), 1):
            m = FORBIDDEN.search(line)
            if m and "base64" not in line and not line.lstrip().startswith(("<image", "xlink:href")):
                left.append((rel, i, m.group(0), line.strip()[:120]))
    if not args.check:
        print(f"rewrote {len(changed)} files")
    for rel, i, hit, line in left[:40]:
        print(f"  LEFT {rel}:{i} [{hit}] {line}")
    print(f"{len(left)} occurrence(s) left")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
