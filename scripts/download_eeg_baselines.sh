#!/bin/bash
# ============================================================================
# The released EEGPT and LaBraM weights, for the SeizeIT2 EEG-only comparison
# (physiowave/fusion/baselines.py). Run on the login node (compute nodes have
# no internet); each file is checked against the SHA-256 it was verified with.
#
#   bash scripts/download_eeg_baselines.sh
#
#   -> $PW_CKPT_ROOT/baselines/eegpt/eegpt_mcae_58chs_4s_large4E.safetensors
#   -> $PW_CKPT_ROOT/baselines/labram/labram-base.pth
#   -> $PW_CKPT_ROOT/baselines/luna/LUNA_{base,large,huge}.safetensors
#
# LaBraM-Base: the authors' file, from their repository (MIT) at a fixed
# commit. EEGPT large (eegpt_mcae_58chs_4s_large4E): the authors release it on
# figshare (figshare.com/s/e37df4f8a907a866df4b), whose download page does not
# serve scripts; this takes the same checkpoint converted to safetensors from
# huggingface.co/eeg-telecom-paris/eegpt-large-official at a fixed revision --
# a third-party mirror. Its target encoder loads strictly into the official
# EEGTransformer and gives bit-identical outputs; to use the authors' file
# instead, download the .ckpt in a browser and pass EEGPT_CKPT=<path> to the
# job (both formats load). LUNA (NeurIPS 2025): the authors' files from
# huggingface.co/PulpBio/LUNA at a fixed revision; the weights are CC BY-ND 4.0
# (use and cite; fine-tuned copies may not be redistributed). LUNA=0 skips them
# (the huge checkpoint is 1.2 GB).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PW_CKPT_ROOT:-}" ]]; then
    PW_CKPT_ROOT="$(set +u; source "${HERE}/cineca_env.sh" >/dev/null 2>&1; echo "${PW_CKPT_ROOT}")"
fi
[[ -n "${PW_CKPT_ROOT}" ]] || { echo "ERROR: PW_CKPT_ROOT is not set" >&2; exit 1; }
OUT="${PW_CKPT_ROOT}/baselines"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch() {   # url dest sha256
    local url="$1" dest="$2" sum="$3" got
    mkdir -p "$(dirname "${dest}")"
    if [[ -f "${dest}" && "$(sha256 "${dest}")" == "${sum}" ]]; then
        echo "==> ${dest}: present, checksum ok"; return 0
    fi
    echo "==> ${url}"
    curl -fL --retry 5 --retry-delay 10 -C - -sS -o "${dest}.part" "${url}"
    got="$(sha256 "${dest}.part")"
    if [[ "${got}" != "${sum}" ]]; then
        echo "ERROR: ${dest}: checksum mismatch (got ${got:0:16}..., want ${sum:0:16}...)" >&2
        rm -f "${dest}.part"; return 1
    fi
    mv "${dest}.part" "${dest}"
    echo "    ${dest}: $(du -h "${dest}" | cut -f1), checksum ok"
}

fetch "https://huggingface.co/eeg-telecom-paris/eegpt-large-official/resolve/11e0075c388e282a2762496f7f3775652c9e3db9/weights.safetensors" \
      "${OUT}/eegpt/eegpt_mcae_58chs_4s_large4E.safetensors" \
      c45c31cb42f68f8b9c630a7739a8084ae5dd6a6608a0d732a5165f180c755a96
fetch "https://raw.githubusercontent.com/935963004/LaBraM/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/labram-base.pth" \
      "${OUT}/labram/labram-base.pth" \
      7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c
if [[ "${LUNA:-1}" == "1" ]]; then
    LUNA_REV=999c1af0fd6fbfb43ed47169c61fe9faa49cfe9f
    fetch "https://huggingface.co/PulpBio/LUNA/resolve/${LUNA_REV}/LUNA_base.safetensors" \
          "${OUT}/luna/LUNA_base.safetensors" \
          482839ad9152b6948ff166d2a1638637c54b4d91f1cefadbce6312831ba9b11d
    fetch "https://huggingface.co/PulpBio/LUNA/resolve/${LUNA_REV}/LUNA_large.safetensors" \
          "${OUT}/luna/LUNA_large.safetensors" \
          03e0321a1e1a30dc074f7bfe512a901138200494f7694f31ab633e42b105c0e4
    fetch "https://huggingface.co/PulpBio/LUNA/resolve/${LUNA_REV}/LUNA_huge.safetensors" \
          "${OUT}/luna/LUNA_huge.safetensors" \
          7522e85c6403675bff3c1b3856e5e93b724ac0da4d7d0e7e11bfba9a57067346
fi
echo "done: ${OUT}"
