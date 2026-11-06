#!/usr/bin/env bash
# =============================================================================
# Hyper-parameter sweep for the DB5 sEMG fine-tuning run.
#
#   bash scripts/sweep_db5.sh lr          # stage 1: learning rate
#   bash scripts/sweep_db5.sh reg         # stage 2: regularisation, at BEST_LR
#   bash scripts/sweep_db5.sh capacity    # stage 3: model size
#   bash scripts/sweep_db5.sh all
#
# Runs inside one interactive 4-GPU allocation; a 60-epoch run takes ~2 min.
#
# Configurations are selected on VALIDATION balanced accuracy and the test
# numbers are only reported. Balanced accuracy rather than accuracy because the
# 53 classes are not equally represented -- plain accuracy rewards a model that
# leans on the frequent ones.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/cineca_env.sh"

STAGE="${1:-lr}"
SWEEP_ROOT="${SWEEP_ROOT:-${PW_CKPT_ROOT}/sweep_db5}"
CSV="${SWEEP_ROOT}/results.csv"
mkdir -p "${SWEEP_ROOT}"

# Shared across every run in the sweep.
export TASK=db5 IN_CHANNELS=16 NUM_CLASSES=53
export NUM_GPUS="${NUM_GPUS:-4}" BATCH_SIZE="${BATCH_SIZE:-16}"
export EPOCHS="${EPOCHS:-60}" WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"

BEST_LR="${BEST_LR:-3e-4}"      # stage 2/3 build on the stage 1 winner

[[ -f "${CSV}" ]] || echo "name,lr,wd,dropout,embed_dim,depth,epochs,val_bal_acc,val_acc,test_bal_acc,test_acc,test_kappa,test_auroc,seconds" > "${CSV}"

run_one() {
    local name="$1"; shift
    local out="${SWEEP_ROOT}/${name}"
    if [[ -f "${out}/test_results.json" ]]; then
        echo "--- ${name}: already done, skipping"
        return 0
    fi
    echo "=== ${name}  ($*)"
    local t0 t1
    t0="$(date +%s)"
    # shellcheck disable=SC2086
    env "$@" OUTPUT_DIR="${out}" bash EMG/finetune_emg.sh > "${SWEEP_ROOT}/${name}.log" 2>&1
    local rc=$?
    t1="$(date +%s)"
    if [[ ${rc} -ne 0 ]]; then
        echo "    FAILED (rc=${rc}); tail of ${SWEEP_ROOT}/${name}.log:"
        tail -5 "${SWEEP_ROOT}/${name}.log" | sed 's/^/      /'
        return 0                      # keep the sweep going
    fi
    "${PYTHON:-python}" - "$name" "$out" "$((t1-t0))" "$CSV" "$@" <<'PYEOF'
import json, os, re, sys
name, out, secs, csv = sys.argv[1:5]
env = dict(kv.split("=", 1) for kv in sys.argv[5:] if "=" in kv)
test = json.load(open(os.path.join(out, "test_results.json")))
# Best validation epoch, from the log the trainer prints.
vb = va = float("nan")
log = os.path.join(os.path.dirname(out), name + ".log")
if os.path.exists(log):
    for line in open(log, errors="ignore"):
        if line.startswith("Best Validation:"):
            m = dict(re.findall(r"(\w+)=([\d.]+)", line))
            va, vb = float(m.get("Acc", "nan")), float(m.get("BalAcc", "nan"))
row = [name, env.get("LR", "3e-4"), env.get("WEIGHT_DECAY", "1e-3"),
       env.get("DROPOUT", "0.1"), env.get("EMBED_DIM", "256"),
       env.get("DEPTH", "6"), env.get("EPOCHS", os.environ.get("EPOCHS", "60")),
       f"{vb:.4f}", f"{va:.4f}",
       f"{test['test_balanced_acc']:.4f}", f"{test['test_acc']:.4f}",
       f"{test['test_kappa']:.4f}", f"{test['test_auroc']:.4f}", secs]
with open(csv, "a") as fh:
    fh.write(",".join(map(str, row)) + "\n")
print(f"    val_bal={vb:.4f}  test_bal={test['test_balanced_acc']:.4f}  "
      f"test_auroc={test['test_auroc']:.4f}  ({secs}s)")
PYEOF
}

case "${STAGE}" in
lr)
    for lr in 1e-4 3e-4 1e-3 3e-3; do
        run_one "lr_${lr}" "LR=${lr}"
    done
    ;;
reg)
    for wd in 1e-4 1e-2 5e-2; do
        run_one "reg_wd${wd}" "LR=${BEST_LR}" "WEIGHT_DECAY=${wd}"
    done
    for dp in 0.0 0.3; do
        run_one "reg_dp${dp}" "LR=${BEST_LR}" "DROPOUT=${dp}" "HEAD_DROPOUT=${dp}"
    done
    for ls in 0.0 0.2; do
        run_one "reg_ls${ls}" "LR=${BEST_LR}" "LABEL_SMOOTHING=${ls}"
    done
    ;;
capacity)
    run_one "cap_d4_e192"  "LR=${BEST_LR}" "DEPTH=4"  "EMBED_DIM=192" "NUM_HEADS=6"
    run_one "cap_d6_e256"  "LR=${BEST_LR}"
    run_one "cap_d8_e320"  "LR=${BEST_LR}" "DEPTH=8"  "EMBED_DIM=320"
    run_one "cap_d8_e384"  "LR=${BEST_LR}" "DEPTH=8"  "EMBED_DIM=384" "NUM_HEADS=12"
    ;;
all)
    for st in lr reg capacity; do bash "$0" "${st}"; done
    ;;
*)
    echo "unknown stage '${STAGE}'. Valid: lr reg capacity all" >&2; exit 1 ;;
esac

echo
echo "=== ranked by validation balanced accuracy ==="
"${PYTHON:-python}" - "$CSV" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def key(r):
    try: return -float(r["val_bal_acc"])
    except ValueError: return 0.0
if not rows:
    print("(no completed runs -- check the .log files in the sweep directory)")
    raise SystemExit(0)
rows.sort(key=key)
cols = ["name", "lr", "wd", "dropout", "embed_dim", "depth",
        "val_bal_acc", "test_bal_acc", "test_auroc", "seconds"]
w = {c: max([len(c)] + [len(r.get(c, "")) for r in rows]) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(r.get(c, "").ljust(w[c]) for c in cols))
PYEOF
echo
echo "CSV: ${CSV}"
