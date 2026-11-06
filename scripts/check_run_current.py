#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Is a finished run's result the one this sweep would produce?

    python scripts/check_run_current.py <test_results.json> NAME=VALUE ...

Exit 0  -- it matches, the runner may skip it
Exit 1  -- it is stale or unreadable, and says on stderr which field differs

A sweep runner has to decide whether a directory that already holds a result
may be skipped. "The file exists" is the wrong test, and quietly so: a result
written before a bug fix, or under a different EPOCHS, sits in the table looking
exactly like a current one, and the paired delta attributes the difference to
whatever the sweep was varying. It happened here twice -- an evaluation-sharding
fix landed between two runs of the same sweep, and a `rm -rf $PW_CKPT_ROOT/...`
with PW_CKPT_ROOT unset removed nothing and reported success, so the stale row
survived a deliberate attempt to delete it.

Every NAME is a shell variable the runner sets; they are resolved to their
places in the result's own `provenance` block. Floats compare with a relative
tolerance, since 3e-4 and 0.0003 are the same learning rate.

Only the standard library, so this runs in whichever environment is active.
"""

from __future__ import annotations

import json
import math
import os
import sys

#: A result written by an older evaluation path is not comparable to a current
#: one however well its hyper-parameters match, and that is exactly the case a
#: config comparison cannot see. Version 2 means val and test were scored on the
#: whole set rather than on one rank's shard.
MIN_RESULT_SCHEMA = 2

#: shell name -> where it lives in test_results.json's provenance.
#: "cfg" is provenance.resolved_model_config; "top" is provenance itself.
FIELDS = {
    "EPOCHS": ("cfg", "epochs"),
    "WARMUP_EPOCHS": ("cfg", "warmup_epochs"),
    "BATCH_SIZE": ("cfg", "batch_size"),
    "LR": ("cfg", "lr"),
    "WEIGHT_DECAY": ("cfg", "weight_decay"),
    "DROPOUT": ("cfg", "dropout"),
    "HEAD_DROPOUT": ("cfg", "head_dropout"),
    "LABEL_SMOOTHING": ("cfg", "label_smoothing"),
    "FOLD_KL": ("cfg", "fold_kl"),
    "SELECT_BY": ("cfg", "select_by"),
    "IN_CHANNELS": ("cfg", "in_channels"),
    "CHANNEL_ENCODING": ("top", "channel_encoding"),
    "CHANNEL_INJECTION": ("top", "channel_injection"),
    "SEED": ("top", "seed"),
    "GIT_COMMIT": ("top", "git_commit"),
    "METADATA_HASH": ("top", "metadata_hash"),
}
#: MIN_LR and the rest are not recorded in resolved_model_config, so they cannot
#: be checked here. Naming them keeps a silent no-op from looking like a check.
UNCHECKABLE = ("MIN_LR",)


def test_set_size(test_file):
    """How many windows the test file holds, or None if it cannot be read."""
    if not test_file or not os.path.isfile(test_file):
        return None
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(test_file, "r") as f:
            for key in ("label", "labels", "y"):
                if key in f:
                    return int(f[key].shape[0])
            return int(f["data"].shape[0]) if "data" in f else None
    except OSError:
        return None


def scored_on_whole_test_set(result, test_file):
    """True / False / None when it cannot be determined.

    `per_class_support` is the row sum of the confusion matrix, so it counts
    exactly the windows the reported metrics were computed from. Against the
    test file's own length it settles the question outright -- which is what
    made a quarter-sized result visible in the first place.
    """
    support = result.get("per_class_support")
    n = result.get("test_samples")
    if n is None and support:
        n = sum(support)
    total = test_set_size(test_file)
    if n is None or total is None:
        return None
    return n == total


def same(want: str, got) -> bool:
    if got is None:
        return False
    if isinstance(got, bool):
        return str(got).lower() == want.strip().lower()
    if isinstance(got, (int, float)):
        try:
            w = float(want)
        except ValueError:
            return False
        return math.isclose(w, float(got), rel_tol=1e-9, abs_tol=1e-12)
    return str(got) == want


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    argv = list(sys.argv[1:])
    test_file = None
    if "--test-file" in argv:
        i = argv.index("--test-file")
        test_file = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    path, pairs = argv[0], argv[1:]

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 1                       # nothing there; the runner runs it
    try:
        with open(path) as f:
            result = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  stale: {path} is unreadable ({exc})", file=sys.stderr)
        return 1

    schema = result.get("result_schema_version")
    if schema is None:
        # Written before the field existed. Rather than condemn every such
        # result -- most are perfectly good -- ask the question the version
        # stands in for: was this scored on the whole test set, or on one
        # rank's shard? The test file answers it directly.
        verdict = scored_on_whole_test_set(result, test_file)
        if verdict is False:
            n = sum(result.get("per_class_support") or [0])
            print(f"  stale: {path}", file=sys.stderr)
            print(f"    scored on {n} of {test_set_size(test_file)} test "
                  f"windows -- one rank's shard, from before val and test "
                  f"stopped being split across ranks.", file=sys.stderr)
            return 1
        if verdict is None:
            print(f"  stale: {path}", file=sys.stderr)
            print(f"    no result_schema_version, and the test set is not "
                  f"available to check what it was scored on. Re-running is "
                  f"the only way to know.", file=sys.stderr)
            return 1
    elif schema < MIN_RESULT_SCHEMA:
        print(f"  stale: {path}", file=sys.stderr)
        print(f"    result_schema_version {schema} < {MIN_RESULT_SCHEMA} -- it "
              f"was produced by an older evaluation path.", file=sys.stderr)
        return 1

    prov = result.get("provenance")
    if not prov:
        print(f"  stale: {path} records no provenance -- it predates it, so "
              f"there is no way to tell what produced it", file=sys.stderr)
        return 1
    cfg = prov.get("resolved_model_config", {})

    differences = []
    for pair in pairs:
        if "=" not in pair:
            continue
        name, want = pair.split("=", 1)
        if name in UNCHECKABLE or name not in FIELDS:
            continue
        where, key = FIELDS[name]
        got = (cfg if where == "cfg" else prov).get(key)
        if not same(want, got):
            differences.append(f"{name}: recorded {got!r}, this sweep wants {want!r}")

    if differences:
        print(f"  stale: {path}", file=sys.stderr)
        for d in differences:
            print(f"    {d}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
