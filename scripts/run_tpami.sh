#!/usr/bin/env bash
# =============================================================================
# PhysioWave TPAMI extension -- single entry point.
#
#   bash scripts/run_tpami.sh smoke
#   bash scripts/run_tpami.sh pretrain --modality eeg  --config base
#   bash scripts/run_tpami.sh pretrain --modality ecg  --config base
#   bash scripts/run_tpami.sh pretrain --modality semg --config base
#   bash scripts/run_tpami.sh fusion   --config ralf
#   bash scripts/run_tpami.sh eval      --suite eeg
#   bash scripts/run_tpami.sh benchmark --suite tokens
#   bash scripts/run_tpami.sh benchmark --suite multimodal
#   bash scripts/run_tpami.sh experiments --tier 1
#   bash scripts/run_tpami.sh report
#   bash scripts/run_tpami.sh all --dry-run
#
# Nothing is hardcoded to a particular cluster.  Override anything through the
# environment:
#   PYTHON=/path/to/python  NUM_GPUS=4  OUTPUT_ROOT=/scratch/$USER/physiowave
#   EXTRA="--set train.epochs=50 data.roots.tueg=/data/tueg"
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results}"
EXTRA="${EXTRA:-}"

# GPU count: explicit env, else SLURM, else nvidia-smi, else 0 (CPU).
if [[ -n "${NUM_GPUS:-}" ]]; then
    :
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
else
    NUM_GPUS=0
fi

DRY_RUN=0
STAGE="${1:-}"
[[ -n "${STAGE}" ]] || { sed -n '2,25p' "$0"; exit 1; }
shift || true

MODALITY="eeg"
CONFIG=""
SUITE="eeg"
TIER="1 2 3"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modality)  MODALITY="$2"; shift 2 ;;
        --config)    CONFIG="$2";   shift 2 ;;
        --suite)     SUITE="$2";    shift 2 ;;
        --tier)      TIER="$2";     shift 2 ;;
        --dry-run)   DRY_RUN=1;     shift ;;
        --)          shift; break ;;
        *)           EXTRA="${EXTRA} $1"; shift ;;
    esac
done

fail() { echo "ERROR [stage=${STAGE}]: $*" >&2; exit 1; }

launch() {
    # Run a python module under torchrun when GPUs are available.
    local module="$1"; shift
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        echo "+ torchrun --standalone --nproc_per_node=${NUM_GPUS} -m ${module} $*"
        [[ "${DRY_RUN}" -eq 1 ]] && return 0
        torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m "${module}" "$@"
    else
        echo "+ ${PYTHON} -m ${module} $*"
        [[ "${DRY_RUN}" -eq 1 ]] && return 0
        "${PYTHON}" -m "${module}" "$@"
    fi
}

run_pretrain() {
    local modality="$1"
    local cfg="pretrain/${modality}"
    [[ -f "configs/${cfg}.yaml" ]] || fail "no config configs/${cfg}.yaml for modality ${modality}"
    local out="${OUTPUT_ROOT}/pretrain_${modality}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        "${PYTHON}" -m physiowave.train.pretrain_main --config "${cfg}" \
            --output-dir "${out}" --dry-run ${EXTRA}
    else
        launch physiowave.train.pretrain_main --config "${cfg}" --output-dir "${out}" \
            --resume auto ${EXTRA}
    fi
}

run_fusion() {
    local cfg="fusion/${CONFIG:-ralf}"
    [[ -f "configs/${cfg}.yaml" ]] || fail "no config configs/${cfg}.yaml"
    local out="${OUTPUT_ROOT}/fusion_${CONFIG:-ralf}"
    local pre=""
    for m in eeg ecg semg; do
        local ck="${OUTPUT_ROOT}/pretrain_${m}/best.pth"
        [[ -f "${ck}" ]] && pre="${pre} fusion.pretrained.${m}=${ck}"
    done
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        "${PYTHON}" -m physiowave.train.fusion_main --config "${cfg}" \
            --output-dir "${out}" --dry-run ${EXTRA}
    else
        # shellcheck disable=SC2086
        launch physiowave.train.fusion_main --config "${cfg}" --output-dir "${out}" \
            ${pre:+--set ${pre}} ${EXTRA}
    fi
}

case "${STAGE}" in
smoke)
    echo "=== SMOKE: synthetic data, CPU/single GPU, a few minutes ==============="
    mkdir -p "${OUTPUT_ROOT}/smoke" "${RESULT_ROOT}"
    SMOKE_SET="--set train.epochs=1 train.batch_size=2 train.num_workers=0 \
        data.synthetic.num_samples=8 data.synthetic.window_samples=512 \
        model.backbone.depth=2 train.log_every=2"

    for m in eeg ecg semg; do
        echo "--- smoke pretrain ${m} ---"
        # shellcheck disable=SC2086
        "${PYTHON}" -m physiowave.train.pretrain_main --config "pretrain/${m}" \
            --output-dir "${OUTPUT_ROOT}/smoke/pretrain_${m}" ${SMOKE_SET} \
            || fail "smoke pretrain ${m} failed"
    done

    echo "--- smoke resume (EEG) ---"
    # shellcheck disable=SC2086
    "${PYTHON}" -m physiowave.train.pretrain_main --config pretrain/eeg \
        --output-dir "${OUTPUT_ROOT}/smoke/pretrain_eeg" --resume auto \
        --set train.epochs=2 train.batch_size=2 train.num_workers=0 \
        data.synthetic.num_samples=8 data.synthetic.window_samples=512 \
        model.backbone.depth=2 train.log_every=100 || fail "smoke resume failed"

    echo "--- smoke fusion (RALF) ---"
    "${PYTHON}" -m physiowave.train.fusion_main --config fusion/ralf \
        --output-dir "${OUTPUT_ROOT}/smoke/fusion" \
        --set train.epochs=1 train.batch_size=2 data.synthetic.num_samples=8 \
        data.synthetic.window_samples=512 model.encoders.eeg.backbone.depth=2 \
        model.encoders.ecg.backbone.depth=2 model.encoders.semg.backbone.depth=2 \
        || fail "smoke fusion failed"

    echo "--- smoke eval + SSL cache build/hit ---"
    "${PYTHON}" -m physiowave.train.evaluate --suite eeg --quick \
        --output "${RESULT_ROOT}/smoke_eval_eeg.json" || fail "smoke eval failed"
    "${PYTHON}" -m physiowave.train.evaluate --suite multimodal --quick \
        --output "${RESULT_ROOT}/smoke_eval_multimodal.json" || fail "smoke multimodal eval failed"
    "${PYTHON}" -c "
import logging, shutil, tempfile, torch
logging.basicConfig(level=logging.INFO)
from physiowave.data.montages import montage
from physiowave.spatial.spline_laplacian import SSLOperatorCache, SSLConfig, verify_reference_invariance
names, xyz = montage('standard_1010_64')
# A throwaway cache dir: the check asserts one miss then one hit, so it must
# start cold. Reusing cache/ssl makes the second run of the smoke test fail.
cache_dir = tempfile.mkdtemp(prefix='physiowave_ssl_smoke_')
cache = SSLOperatorCache(cache_dir=cache_dir)
L1 = cache.get(names, xyz, None, SSLConfig())
L2 = cache.get(names, xyz, None, SSLConfig())
stats = cache.stats()
assert stats['hits'] >= 1 and stats['misses'] >= 1, f'SSL cache never hit: {stats}'
ok, resid = verify_reference_invariance(L1)
print(f'SSL cache {stats}; reference-invariance residual {resid:.3e} ok={ok}')
shutil.rmtree(cache_dir, ignore_errors=True)
assert ok, 'SSL operator is not reference invariant'
" || fail "SSL cache/invariance smoke failed"

    echo "--- smoke token benchmark ---"
    "${PYTHON}" -m physiowave.train.benchmark --suite tokens --quick \
        --output "${RESULT_ROOT}/smoke_benchmark_tokens.json" || fail "smoke benchmark failed"
    echo "=== SMOKE PASSED ======================================================="
    ;;

pretrain)
    run_pretrain "${MODALITY}"
    ;;

fusion)
    run_fusion
    ;;

eval)
    mkdir -p "${RESULT_ROOT}"
    CK="${OUTPUT_ROOT}/pretrain_eeg/best.pth"
    ARGS=(--suite "${SUITE}" --output "${RESULT_ROOT}/eval_${SUITE}.json")
    [[ -f "${CK}" && "${SUITE}" == "eeg" ]] && ARGS+=(--checkpoint "${CK}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON} -m physiowave.train.evaluate ${ARGS[*]} ${EXTRA}"
    else
        # shellcheck disable=SC2086
        "${PYTHON}" -m physiowave.train.evaluate "${ARGS[@]}" ${EXTRA}
    fi
    ;;

benchmark)
    mkdir -p "${RESULT_ROOT}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "+ ${PYTHON} -m physiowave.train.benchmark --suite ${SUITE} --output ${RESULT_ROOT}/benchmark_${SUITE}.json"
    else
        # shellcheck disable=SC2086
        "${PYTHON}" -m physiowave.train.benchmark --suite "${SUITE}" \
            --output "${RESULT_ROOT}/benchmark_${SUITE}.json" ${EXTRA}
    fi
    ;;

experiments)
    mkdir -p "${RESULT_ROOT}/experiments"
    for exp in channel_ablation spatial_branch_ablation token_efficiency multimodal_robustness; do
        [[ -f "configs/experiments/${exp}.yaml" ]] || continue
        ARGS=(--config "experiments/${exp}" --out-dir "${RESULT_ROOT}/experiments" --tier ${TIER})
        [[ "${DRY_RUN}" -eq 1 ]] && ARGS+=(--dry-run)
        # shellcheck disable=SC2086
        "${PYTHON}" -m physiowave.experiments.runner "${ARGS[@]}" ${EXTRA}
    done
    ;;

report)
    "${PYTHON}" -m physiowave.experiments.report \
        --inputs "${RESULT_ROOT}/*.json" "${RESULT_ROOT}/experiments/*.json" \
        --out-dir "${RESULT_ROOT}/tables" --name summary \
        --group-by variant --pareto samples_per_sec tokens
    ;;

all)
    echo "=== FULL PIPELINE (dry-run=${DRY_RUN}) ================================="
    for m in eeg ecg semg; do run_pretrain "${m}"; done
    run_fusion
    for s in eeg multimodal; do
        SUITE="${s}" DRY_RUN="${DRY_RUN}" bash "$0" eval --suite "${s}" ${DRY_RUN:+--dry-run}
    done
    for s in tokens multimodal; do
        bash "$0" benchmark --suite "${s}" ${DRY_RUN:+--dry-run}
    done
    bash "$0" experiments --tier "${TIER}" ${DRY_RUN:+--dry-run}
    [[ "${DRY_RUN}" -eq 1 ]] || bash "$0" report
    echo "=== PIPELINE COMPLETE =================================================="
    ;;

*)
    fail "unknown stage '${STAGE}'. Valid: smoke pretrain fusion eval benchmark experiments report all"
    ;;
esac
