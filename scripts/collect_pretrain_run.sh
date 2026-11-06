#!/bin/bash
# ============================================================================
# Collect a C1 pretraining run's figures and curves into the repository.
#
#   bash scripts/collect_pretrain_run.sh <run-dir> <name>
#   bash scripts/collect_pretrain_run.sh $PW_CKPT_ROOT/pretrain_ecg_c1_moe ecg_c1_moe
#
# The figures are drawn first, into the run directory, by
#   python scripts/visualize_eeg_pretraining.py --run-dir <run-dir> --format svg,pdf
# (scripts/slurm/cineca_visualize_run.sbatch does both steps on a compute
# node). This copies what a reader of the result needs into docs/runs/<name>/:
#
#   figures/*.svg, *.pdf      every figure the visualiser drew, vector, with
#                             the text left editable
#   figure_metadata/*.json    which checkpoint, step, windows and seeds each used
#   metrics_epoch.jsonl       one line per epoch, train and val, per route/dataset
#   history.json, config_resolved.yaml, environment.json, train_command.txt,
#   dataset_manifest.json, channel_vocab.json, error_histogram.json
#   progress_by_route.txt, progress_by_dataset.txt   scripts/eeg_c1_progress.py
#   README.md                 the best epochs and every figure, inline
#
# NOT collected: checkpoints (*.pth, in .gitignore and hundreds of MB),
# figure_data/*.npz (the arrays behind each figure; large, and redrawable from
# the checkpoint) and metrics_step.jsonl (per-step, megabytes). Then, on a
# login node:
#
#   git add docs/runs/<name> && git commit -m "<name>: figures" && git push
# ============================================================================

set -euo pipefail

RUN_DIR="${1:-}"
NAME="${2:-}"
if [[ -z "${RUN_DIR}" || -z "${NAME}" ]]; then
    echo "usage: bash scripts/collect_pretrain_run.sh <run-dir> <name>" >&2
    exit 1
fi
[[ -f "${RUN_DIR}/metrics_epoch.jsonl" ]] || {
    echo "ERROR: ${RUN_DIR} has no metrics_epoch.jsonl -- not a C1 pretraining run" >&2
    exit 1
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
DEST="docs/runs/${NAME}"
mkdir -p "${DEST}"

if compgen -G "${RUN_DIR}/figures/*.svg" >/dev/null || \
   compgen -G "${RUN_DIR}/figures/*.pdf" >/dev/null; then
    rm -rf "${DEST}/figures"
    mkdir -p "${DEST}/figures"
    # Vector only. A PNG left in the run directory by an earlier draw is not
    # what this collects.
    cp "${RUN_DIR}"/figures/*.svg "${RUN_DIR}"/figures/*.pdf "${DEST}/figures/" 2>/dev/null || true
else
    echo "WARNING: no SVG/PDF figures in ${RUN_DIR}/figures. Draw them first:" >&2
    echo "  ${PYTHON} scripts/visualize_eeg_pretraining.py --run-dir ${RUN_DIR} --checkpoint best.pth --format svg,pdf" >&2
fi
if [[ -d "${RUN_DIR}/figure_metadata" ]]; then
    rm -rf "${DEST}/figure_metadata"
    cp -r "${RUN_DIR}/figure_metadata" "${DEST}/figure_metadata"
fi
for f in metrics_epoch.jsonl history.json config_resolved.yaml environment.json \
         train_command.txt dataset_manifest.json channel_vocab.json \
         error_histogram.json; do
    [[ -f "${RUN_DIR}/${f}" ]] && cp "${RUN_DIR}/${f}" "${DEST}/${f}"
done

"${PYTHON}" scripts/eeg_c1_progress.py "${RUN_DIR}" > "${DEST}/progress_by_route.txt" 2>&1 || true
"${PYTHON}" scripts/eeg_c1_progress.py "${RUN_DIR}" --by dataset > "${DEST}/progress_by_dataset.txt" 2>&1 || true

"${PYTHON}" - "${DEST}" "${NAME}" "${RUN_DIR}" <<'PYEOF'
import glob, json, os, sys
dest, name, run_dir = sys.argv[1:4]
rows = [json.loads(l) for l in open(os.path.join(dest, "metrics_epoch.jsonl"))
        if l.strip()]
rows = [r for r in rows if r.get("val/loss_total") == r.get("val/loss_total")]  # drop NaN
out = [f"# {name}", "",
       f"Collected from `{run_dir}` by `scripts/collect_pretrain_run.sh`.", ""]
if rows:
    last = rows[-1]
    out += [f"{len(rows)} epoch(s) with validation; last epoch {last['epoch']}, "
            f"step {last.get('global_step', '?')}.", "",
            "| selection | epoch | val total | val spec | val raw | spec r | raw r |",
            "|---|---|---|---|---|---|---|"]
    def line(label, r):
        return (f"| {label} | {r['epoch']} | {r['val/loss_total']:.5f} | "
                f"{r.get('val/loss_masked_spec_mse', float('nan')):.5f} | "
                f"{r.get('val/loss_masked_raw_smoothl1', float('nan')):.5f} | "
                f"{r.get('val/masked_spec_corr', float('nan')):.3f} | "
                f"{r.get('val/masked_raw_corr', float('nan')):.3f} |")
    for label, key in (("best total (best.pth)", "val/loss_total"),
                       ("best spec", "val/loss_masked_spec_mse"),
                       ("best raw", "val/loss_masked_raw_smoothl1")):
        have = [r for r in rows if key in r]
        if have:
            out.append(line(label, min(have, key=lambda r: r[key])))
    out.append(line("last", last))
    out.append("")
out += ["Per route and per dataset: `progress_by_route.txt`, "
        "`progress_by_dataset.txt`. What each figure was drawn from: "
        "`figure_metadata/`.", ""]
for svg in sorted(glob.glob(os.path.join(dest, "figures", "*.svg"))):
    base = os.path.basename(svg)
    stem = os.path.splitext(base)[0]
    pdf = f" ([PDF](figures/{stem}.pdf))" if os.path.isfile(
        os.path.join(dest, "figures", stem + ".pdf")) else ""
    out += [f"## {stem}{pdf}", "", f"![{base}](figures/{base})", ""]
open(os.path.join(dest, "README.md"), "w").write("\n".join(out))
PYEOF

echo "collected into ${DEST}:"
du -sh "${DEST}"
ls "${DEST}"
echo
echo "Review, then push from a login node:"
echo "  git add ${DEST} && git commit -m '${NAME}: pretraining figures and curves' && git push"
