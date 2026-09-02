#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: remote_build_fa.sh fa1|fa2|fa3}"
case "${variant}" in
  fa1|fa2|fa3) ;;
  *)
    echo "unknown FlashAttention variant: ${variant}" >&2
    exit 2
    ;;
esac
workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
max_jobs="${INFRASWE_MAX_JOBS:-32}"
resume_build="${INFRASWE_RESUME_BUILD:-0}"
source_root="${workspace}/sources/${variant}"
target_root="${workspace}/envs/${variant}"

mkdir -p "${workspace}/sources" "${workspace}/envs"
if [[ "${resume_build}" == 1 && -d "${source_root}/.git" ]]; then
  rm -rf "${target_root}"
  echo "Resuming ${variant} build from existing source tree: ${source_root}"
else
  rm -rf "${source_root}" "${target_root}"
  git clone --filter=blob:none https://github.com/Dao-AILab/flash-attention.git "${source_root}"
fi

case "${variant}" in
  fa1)
    commit=6d48e14a6c2f551db96f0badc658a6279a929df3
    fa1_archs="${INFRASWE_FA1_ARCHS:-8.0}"
    cd "${source_root}"
    git checkout "${commit}"
    git submodule update --init --depth 1 csrc/flash_attn/cutlass
    MAX_JOBS="${max_jobs}" NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST="${fa1_archs}" \
      uv pip install --python "${python}" --target "${target_root}" \
      --no-deps --no-build-isolation .
    PYTHONPATH="${target_root}" "${python}" -c \
      "import torch, flash_attn, flash_attn_cuda; print('FA1_BUILD_OK', flash_attn.__version__, flash_attn_cuda.__file__)"
    ;;
  fa2)
    commit=ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
    cd "${source_root}"
    git checkout "${commit}"
    git submodule update --init --depth 1 csrc/cutlass
    fa2_archs="${INFRASWE_FA2_ARCHS:-80}"
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
      FLASH_ATTENTION_DISABLE_DROPOUT=TRUE \
      FLASH_ATTN_CUDA_ARCHS="${fa2_archs}" \
      MAX_JOBS="${max_jobs}" NVCC_THREADS=2 \
      uv pip install --python "${python}" --target "${target_root}" \
      --no-deps --no-build-isolation .
    PYTHONPATH="${target_root}" "${python}" -c \
      "import flash_attn, flash_attn_2_cuda; print('FA2_BUILD_OK', flash_attn.__version__, flash_attn_2_cuda.__file__)"
    ;;
  fa3)
    commit=ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
    cd "${source_root}"
    git checkout "${commit}"
    git submodule update --init --depth 1 csrc/cutlass
    cd hopper
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
      FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
      FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
      FLASH_ATTENTION_DISABLE_CLUSTER=TRUE \
      FLASH_ATTENTION_DISABLE_FP16=TRUE \
      FLASH_ATTENTION_DISABLE_FP8=TRUE \
      FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
      FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
      FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
      FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
      FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
      FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
      FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
      FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
      FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
      MAX_JOBS="${max_jobs}" NVCC_THREADS=2 \
      uv pip install --python "${python}" --target "${target_root}" \
      --no-deps --no-build-isolation .
    (
      cd "${workspace}/benchmarks/kernel_frontier"
      PYTHONPATH="${target_root}" "${python}" -c \
        "import torch; from flash_attn_3 import _C, flash_attn_interface; print('FA3_BUILD_OK', flash_attn_interface.__file__, _C.__file__)"
    )
    ;;
esac

git -C "${source_root}" rev-parse HEAD
