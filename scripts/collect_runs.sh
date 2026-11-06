#!/bin/bash

# ============================================================================
# Collect finished run logs into the repository so they can be reviewed in a
# diff rather than over a terminal paste.
#
# Usage, from the repository root:
#   bash scripts/collect_runs.sh docs/runs/db5_fold_ablation ~/e40_*.log
#
# Each log is stripped of tqdm's carriage-return rewrites -- a progress bar
# writes one physical line per update and a 40-epoch run expands to megabytes
# of them, of which only the final state of each bar means anything. What
# survives is the hyper-parameter header, one line per epoch, and the test
# block, which is a couple of hundred lines and reads in a diff.
#
# Where a log records the output directory it was written from, that run's
# test_results.json is copied alongside it. That file, not the log, is the
# authoritative result.
# ============================================================================

set -euo pipefail

DEST="${1:-}"
shift || true

if [[ -z "${DEST}" || $# -eq 0 ]]; then
    echo "usage: bash scripts/collect_runs.sh <dest-dir> <log> [log ...]" >&2
    exit 1
fi

mkdir -p "${DEST}"

for log in "$@"; do
    [[ -f "${log}" ]] || { echo "skipping missing ${log}" >&2; continue; }
    name="$(basename "${log}")"; name="${name%.log}"

    # Collapse each progress bar to its final state. perl rather than sed
    # because BSD sed does not read \r in a pattern, and this script has to
    # work on a login node and on a laptop.
    perl -pe 's/.*\r//' "${log}" > "${DEST}/${name}.log"

    # "Fine-tuning completed. Results saved to /path/to/run"
    run_dir="$(grep -o 'Results saved to .*' "${DEST}/${name}.log" \
               | tail -1 | sed 's/^Results saved to //')"
    if [[ -n "${run_dir}" && -f "${run_dir}/test_results.json" ]]; then
        cp "${run_dir}/test_results.json" "${DEST}/${name}.json"
        extra=" + test_results.json"
    else
        # An unfinished or crashed run has no results file; that is worth
        # collecting anyway, and worth saying out loud rather than silently
        # producing a directory that looks complete.
        extra=" (no test_results.json -- run unfinished?)"
    fi

    # Size, not line count: stripping \r removes no newlines, so the whole
    # saving is invisible in a line count.
    size_in=$(du -h "${log}" | cut -f1)
    size_out=$(du -h "${DEST}/${name}.log" | cut -f1)
    printf '  %-22s %6s -> %6s%s\n' "${name}" "${size_in}" "${size_out}" "${extra}"
done

echo
echo "Collected into ${DEST}. Review, then:"
echo "  git add ${DEST} && git commit -m 'DB5 fold ablation logs' && git push"
