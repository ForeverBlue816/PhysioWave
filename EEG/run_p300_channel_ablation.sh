#!/bin/bash

# ============================================================================
# PhysioP300 channel-embedding ablation
# ============================================================================
# The same six variants as the Sleep-EDF ablation, on the montage the question
# was actually about. Sleep-EDF has two bipolar derivations and the existing 2-D
# position embedding already tells them apart, so it is the least favourable
# place a channel code could be tested. erpbci has 62 electrodes in the
# converter's default montage (58 with --channels 58).
#
#   C0  none    none    the baseline this is measured against
#   C1  id      token   EEGPT-style learned channel-name embedding (their Eq. 11)
#   C2  signed  token   the electrode's geometry instead of its name
#   C3  signed  fold    the same code, biasing only the scale choice
#   C4  signed  dual    both injection sites
#   C5  hybrid  dual    whether a name and a position say different things
#
# WHAT `signed` MEANS HERE. This montage is monopolar: each channel is one
# electrode against a common reference, not a difference of a pair. Both endpoint indices of a
# channel point at the same electrode, so the encoder's direction term is
# exactly zero and what survives is the electrode's position on the sphere,
# marked as monopolar. C2 is therefore "position" on this dataset and "signed
# derivation" on Sleep-EDF -- the same encoder, degenerating honestly. No
# reference position is invented: erpbci's ear electrodes are dropped before
# preprocessing and standard_1020 has no scalp coordinate for them.
#
# THE CONFOUND TO KEEP IN MIND. CHANNELS_58 is in EEGPT's topographic order and
# the 2-D position embedding indexes rows by that order, so the model already
# has a one-dimensional walk over the scalp for free. What C2 adds is genuine
# 3-D position. A null result means "the topographic row order was enough", not
# "position does not matter".
#
# Plumbing check -- two variants, one fold:
#
#   VARIANTS=C0,C4 FOLDS=0 SEEDS=42 NUM_GPUS=4 EPOCHS=2 \
#       bash EEG/run_p300_channel_ablation.sh
#
# The real thing -- nine LOSO folds, since between-subject variance is the
# dominant term on an ERP task and a single fold says almost nothing:
#
#   VARIANTS=C0,C1,C2,C3,C4,C5 FOLDS=0,1,2,3,4,5,6,7,8 SEEDS=42 \
#       bash EEG/run_p300_channel_ablation.sh
#
# DISK. One fold's train/val/test is ~2.5 GB and every variant of that fold
# shares it. KEEP_SPLITS=1 leaves them all in place; the default removes a
# fold's HDF5 once every variant has been scored on it, since rebuilding one
# from the decode cache is a couple of minutes.
#
# WHAT IS HELD FIXED, and how:
#   * one HDF5 per fold, built once and shared -- the data cannot differ
#   * SEED goes to finetune.py, which seeds the sampler, so a given (fold, seed)
#     sees the same batch order in every variant
#   * every legacy module is constructed and initialised before any channel
#     module exists, so the same seed gives every variant the same backbone
#   * the hyper-parameters below are applied to every run from one place
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

VARIANTS="${VARIANTS:-C0,C1,C2,C3,C4,C5}"
SEEDS="${SEEDS:-42}"
FOLDS="${FOLDS:-0}"
NUM_GPUS="${NUM_GPUS:-4}"

# Sweep hyper-parameters, here rather than in the caller's shell. An ablation is
# only valid if every variant got the same ones, and a forgotten EPOCHS on one
# of six commands leaves no trace in any output -- the runs simply differ, and
# the paired delta attributes it to the channel flags.
EPOCHS="${EPOCHS:-15}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
DROPOUT="${DROPOUT:-0.1}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
FOLD_KL="${FOLD_KL:-1e-3}"
SELECT_BY="${SELECT_BY:-auroc}"
IN_CHANNELS="${IN_CHANNELS:-64}"

_SWEEP_VARS=(EPOCHS WARMUP_EPOCHS BATCH_SIZE LR MIN_LR WEIGHT_DECAY
             DROPOUT HEAD_DROPOUT LABEL_SMOOTHING FOLD_KL SELECT_BY IN_CHANNELS)

# The same values again, as NAME=VALUE, for the staleness check below. Derived
# from _SWEEP_VARS rather than written out, so a hyper-parameter cannot be
# applied to a run and then left out of the check that decides whether that run
# is current.
_check=()
for _v in "${_SWEEP_VARS[@]}"; do _check+=("${_v}=${!_v}"); done

EDF_DIR="${EDF_DIR:-${PW_DATA_EEG}/erpbci}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/p300_channel}"
SWEEP_ROOT="${SWEEP_ROOT:-${PW_CKPT_ROOT}/p300_channel_ablation}"
KEEP_SPLITS="${KEEP_SPLITS:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

variant_encoding() {
    case "$1" in
        C0) echo "none" ;;   C1) echo "id" ;;     C2) echo "signed" ;;
        C3) echo "signed" ;; C4) echo "signed" ;; C5) echo "hybrid" ;;
        *)  echo "" ;;
    esac
}
variant_injection() {
    case "$1" in
        C0) echo "none" ;;  C1) echo "token" ;; C2) echo "token" ;;
        C3) echo "fold" ;;  C4) echo "dual" ;;  C5) echo "dual" ;;
        *)  echo "" ;;
    esac
}

IFS=',' read -r -a _variants <<< "${VARIANTS}"
IFS=',' read -r -a _seeds    <<< "${SEEDS}"
IFS=',' read -r -a _folds    <<< "${FOLDS}"

for v in "${_variants[@]}"; do
    [[ -n "$(variant_encoding "${v}")" ]] || {
        echo "ERROR: unknown variant '${v}' (C0..C5)" >&2; exit 1; }
done

# The decode cache is the expensive part and this script does not build it.
# Say so once, rather than as nine identical split-stage failures.
CACHE_DIR="$(pw_p300_cache_dir "${EDF_DIR}" || true)"
if [[ -n "${CACHE_DIR}" && "$(basename "${CACHE_DIR}")" == "c58" \
      && "${IN_CHANNELS}" != "58" && "${DRY_RUN}" != "1" ]]; then
    echo "ERROR: ${CACHE_DIR} is the legacy cache and holds only EEGPT's 58" >&2
    echo "       electrodes, so no ${IN_CHANNELS}-channel split can come out" >&2
    echo "       of it. Re-decode once, or set IN_CHANNELS=58." >&2
    exit 1
fi
if [[ -z "${CACHE_DIR}" && "${DRY_RUN}" != "1" ]]; then
    echo "ERROR: no decode cache under ${EDF_DIR}/cache (looked for c64, c58)" >&2
    echo "       Build it first, where mne is available:" >&2
    echo "         source \$HOME/pwprep/bin/activate" >&2
    echo "         PW_VARS_ONLY=1 source scripts/cineca_env.sh" >&2
    echo "         python EEG/physio_p300_finetune.py --edf-dir ${EDF_DIR} \\" >&2
    echo "             --out-dir ${DATA_ROOT}/fold0 --stage cache --jobs 2" >&2
    exit 1
fi

echo "============================================================"
echo "  PhysioP300 channel-embedding ablation"
echo "  variants=${VARIANTS}  seeds=${SEEDS}  folds=${FOLDS}"
echo "  gpus=${NUM_GPUS}"
echo "  hyper-parameters, identical for every run:"
for _v in "${_SWEEP_VARS[@]}"; do printf '    %-16s %s\n' "${_v}" "${!_v}"; done
echo "  cache=${CACHE_DIR}"
echo "  data =${DATA_ROOT}"
echo "  out  =${SWEEP_ROOT}"
echo "============================================================"

total=0; ran=0; skipped=0; failed=()
for k in "${_folds[@]}"; do
  d="${DATA_ROOT}/fold${k}"

  # Is there anything to do for this fold, before spending two minutes on a
  # split? The same staleness test as below, minus --test-file: this fold's
  # test.h5 has been deleted by now, and a run that cannot be shown to be
  # current counts as work, which builds the split and lets the real check --
  # which does have the file -- decide.
  _todo=0
  for v in "${_variants[@]}"; do
    for s in "${_seeds[@]}"; do
      if [[ "${FORCE}" == "1" ]] || ! python scripts/check_run_current.py \
             "${SWEEP_ROOT}/fold${k}/${v}/seed${s}/test_results.json" \
             CHANNEL_ENCODING="$(variant_encoding "${v}")" \
             CHANNEL_INJECTION="$(variant_injection "${v}")" SEED="${s}" \
             "${_check[@]}" 2>/dev/null; then
          _todo=$((_todo + 1))
      fi
    done
  done
  if [[ "${_todo}" -eq 0 ]]; then
      echo ""
      echo "fold ${k}: every variant already finished, skipped"
      skipped=$((skipped + ${#_variants[@]} * ${#_seeds[@]}))
      total=$((total + ${#_variants[@]} * ${#_seeds[@]}))
      continue
  fi

  echo ""
  echo "===== fold ${k}: building the split shared by every variant ====="
  if [[ "${DRY_RUN}" != "1" ]]; then
      # Rebuilt rather than trusted: a directory left by an interrupted run can
      # hold a train.h5 from a DIFFERENT fold, and nothing downstream notices --
      # the trainer would happily test on subjects it trained on.
      if ! python EEG/physio_p300_finetune.py --edf-dir "${EDF_DIR}" \
               --out-dir "${d}" --stage split --fold "${k}" \
               --channels "${IN_CHANNELS}"; then
          echo "fold ${k}: split failed, every variant skipped" >&2
          for v in "${_variants[@]}"; do for s in "${_seeds[@]}"; do
              failed+=("fold${k}/${v}/seed${s}"); total=$((total + 1))
          done; done
          continue
      fi
  fi

  for v in "${_variants[@]}"; do
    for s in "${_seeds[@]}"; do
      total=$((total + 1))
      enc="$(variant_encoding "${v}")"
      inj="$(variant_injection "${v}")"
      out="${SWEEP_ROOT}/fold${k}/${v}/seed${s}"

      # Finished means it wrote a result file recording the configuration this
      # sweep is running. Both halves matter. A directory left by a job that
      # died mid-epoch has a checkpoint and no result, so skipping on the
      # directory alone would drop it from the mean without saying so -- and a
      # result written before a bug fix, or under a different EPOCHS, sits in
      # the table looking exactly like a current one while the paired delta
      # blames the difference on whatever the sweep was varying.
      if [[ "${FORCE}" != "1" ]] && python scripts/check_run_current.py \
             "${out}/test_results.json" --test-file "${d}/test.h5" \
             CHANNEL_ENCODING="${enc}" CHANNEL_INJECTION="${inj}" SEED="${s}" \
             "${_check[@]}"; then
          skipped=$((skipped + 1)); continue
      fi

      echo ""
      echo "----- fold ${k}  ${v} (${enc}/${inj})  seed ${s} -----"
      if [[ "${DRY_RUN}" == "1" ]]; then
          echo "  FOLD=${k} DATA_DIR=${d} OUTPUT_DIR=${out} \\"
          echo "  CHANNEL_ENCODING=${enc} CHANNEL_INJECTION=${inj} SEED=${s} \\"
          echo "  NUM_GPUS=${NUM_GPUS} \\"
          for _v in "${_SWEEP_VARS[@]}"; do echo "  ${_v}=${!_v} \\"; done
          echo "  bash EEG/finetune_p300.sh"
          continue
      fi
      mkdir -p "${out}"
      # An array handed to env(1), not a command prefix: bash decides which
      # words are assignments while parsing, so a conditional ${X:+X=$X} would
      # expand into the command NAME rather than into an assignment.
      _env=(FOLD="${k}" DATA_DIR="${d}" OUTPUT_DIR="${out}"
            CHANNEL_ENCODING="${enc}" CHANNEL_INJECTION="${inj}"
            SEED="${s}" NUM_GPUS="${NUM_GPUS}")
      for _v in "${_SWEEP_VARS[@]}"; do _env+=("${_v}=${!_v}"); done
      if env "${_env[@]}" \
         bash EEG/finetune_p300.sh 2>&1 | tee "${out}/run.log"; then
          ran=$((ran + 1))
      else
          echo "fold ${k} ${v} seed ${s}: FAILED" >&2
          failed+=("fold${k}/${v}/seed${s}")
      fi
    done
  done

  # 2.5 GB a fold and nine folds do not need to coexist. The decode cache stays,
  # so rebuilding any of them is a couple of minutes.
  if [[ "${KEEP_SPLITS}" != "1" && "${DRY_RUN}" != "1" ]]; then
      rm -f "${d}"/{train,val,test}.h5
      echo "fold ${k}: split removed (KEEP_SPLITS=1 to keep it)"
  fi
done

echo ""
echo "============================================================"
echo "  ${ran} run(s), ${skipped} skipped as already finished, of ${total}"
[[ ${#failed[@]} -gt 0 ]] && echo "  ${#failed[@]} failed: ${failed[*]}"
echo "  python scripts/collect_channel_ablation.py ${SWEEP_ROOT} \\"
echo "      --classes nontarget,target"
echo "============================================================"

[[ "${DRY_RUN}" == "1" ]] || \
    python scripts/collect_channel_ablation.py "${SWEEP_ROOT}" \
        --classes nontarget,target || true
