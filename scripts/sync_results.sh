#!/bin/bash
# ============================================================================
# Push a snapshot of the cluster's results to a separate git branch.
#
#   bash scripts/sync_results.sh                 # on a Leonardo login node
#   git fetch origin cineca-results              # anywhere else, to read them
#
# What goes on the branch -- small text, nothing that is code or data:
#   runs/<path under $PW_CKPT_ROOT>/   results.json, history.json, summary.*,
#       metrics_epoch.jsonl, config_resolved.yaml, train_command.txt,
#       environment.json, split.json, *.log -- for every run directory
#   jobs/                              the Slurm *.out / *.err in the repository
#   README.md                          when, from where, what was skipped
#
# What never does: checkpoints (*.pth), HDF5, arrays (*.npz), figures
# (docs/runs/ has those), tensorboard, caches, and any file over the size
# cap. A log over the cap keeps its first and last part, with a marker.
# Progress-bar carriage returns are collapsed, which is most of a log's size.
#
# The branch lives in its own worktree ($RESULTS_WORKTREE), so the repository
# you work in -- its branch, its uncommitted files -- is never touched. Run it
# as often as you like: each run commits only what changed.
#
# ENVIRONMENT
#   RESULTS_BRANCH    cineca-results
#   RESULTS_WORKTREE  $HOME/PhysioWave-results
#   MAX_MB            per-file cap, 5
#   PW_CKPT_ROOT      from scripts/cineca_env.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
if [[ -z "${PW_CKPT_ROOT:-}" ]]; then
    PW_VARS_ONLY=1 source "${REPO}/scripts/cineca_env.sh" >/dev/null 2>&1 || true
fi
: "${PW_CKPT_ROOT:?PW_CKPT_ROOT is not set (source scripts/cineca_env.sh)}"
BRANCH="${RESULTS_BRANCH:-cineca-results}"
WT="${RESULTS_WORKTREE:-${HOME}/PhysioWave-results}"
REMOTE="${RESULTS_REMOTE:-origin}"

# -- the worktree, on the results branch -----------------------------------
if [[ ! -d "${WT}/.git" && ! -f "${WT}/.git" ]]; then
    git fetch -q "${REMOTE}" "${BRANCH}" 2>/dev/null || true
    if git show-ref -q --verify "refs/remotes/${REMOTE}/${BRANCH}"; then
        git worktree add -q -B "${BRANCH}" "${WT}" "${REMOTE}/${BRANCH}"
    else
        # A branch with no history in common with the code: an orphan.
        git worktree add -q --detach "${WT}" HEAD
        (cd "${WT}" && git checkout -q --orphan "${BRANCH}" && git rm -rq --cached . \
            && git clean -fdq)
    fi
fi
cd "${WT}"
git pull -q --rebase "${REMOTE}" "${BRANCH}" 2>/dev/null || true

# -- the snapshot -----------------------------------------------------------
"${PYTHON:-$(command -v python || command -v python3)}" - "${PW_CKPT_ROOT}" "${REPO}" "${WT}" "${MAX_MB:-5}" <<'PY'
import os, re, shutil, sys, time
src, repo, wt, max_mb = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cap = int(max_mb * 2**20)
KEEP_NAMES = {"results.json", "history.json", "metrics_epoch.jsonl", "config_resolved.yaml",
              "train_command.txt", "environment.json", "split.json", "dataset_manifest.json",
              "error_histogram.json", "test_results.json", "training_metrics.json",
              "encoders.json", "events_cross.json", "events_both.json", "events_none.json"}
KEEP_EXT = (".log", ".txt", ".md", ".csv")
KEEP_PREFIX = ("summary",)
SKIP_DIRS = {"figure_data", "tensorboard", "smoke_corpus", "cache", "figures",
             "__pycache__", "shards", "parts"}
def collapse(raw):
    """Each line's last carriage-return segment: a progress bar's final state.
    Line by line, so a megabyte-long line costs a megabyte, not its square."""
    out = []
    for line in raw.split(b"\n"):
        crlf = line.endswith(b"\r")
        body = line[:-1] if crlf else line
        out.append(body.rsplit(b"\r", 1)[-1])
    return b"\n".join(out)

def text_copy(a, b):
    raw = open(a, "rb").read()
    raw = collapse(raw)
    if len(raw) > cap:
        head, tail = raw[: cap // 5], raw[-(cap - cap // 5):]
        raw = head + b"\n... [%d bytes cut by sync_results.sh] ...\n" % (len(raw) - len(head) - len(tail)) + tail
    os.makedirs(os.path.dirname(b), exist_ok=True)
    if not os.path.exists(b) or open(b, "rb").read() != raw:
        open(b, "wb").write(raw)

n, skipped = 0, []
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if not (f in KEEP_NAMES or f.endswith(KEEP_EXT) or f.startswith(KEEP_PREFIX)):
            continue
        a = os.path.join(root, f)
        if f == "metrics_step.jsonl" or os.path.getsize(a) > 20 * cap:
            skipped.append(os.path.relpath(a, src)); continue
        text_copy(a, os.path.join(wt, "runs", os.path.relpath(a, src)))
        n += 1
for f in sorted(os.listdir(repo)):
    if re.match(r"physiowave_.*_\d+(_\d+)?\.(out|err)$", f):
        text_copy(os.path.join(repo, f), os.path.join(wt, "jobs", f))
        n += 1
with open(os.path.join(wt, "README.md"), "w") as fh:
    fh.write(f"# Cluster results snapshot\n\nUpdated {time.strftime('%Y-%m-%d %H:%M %Z')} "
             f"from `{src}` and the Slurm logs in `{repo}` by `scripts/sync_results.sh`.\n\n"
             f"Text only: results, histories, summaries, configs, logs (capped at {max_mb:g} MB, "
             f"progress bars collapsed). No checkpoints, data or figures.\n\n"
             f"Skipped as too large: {len(skipped)} file(s)\n" +
             "".join(f"- `{s}`\n" for s in skipped[:50]))
print(f"  {n} file(s) in the snapshot, {len(skipped)} skipped as too large")
PY

git add -A
if git diff --cached --quiet; then
    echo "nothing new since the last snapshot"
    exit 0
fi
git commit -q -m "results snapshot $(date '+%Y-%m-%d %H:%M')"
git push -q -u "${REMOTE}" "${BRANCH}"
echo "pushed to ${REMOTE}/${BRANCH}: $(git log -1 --format='%h %s')"
