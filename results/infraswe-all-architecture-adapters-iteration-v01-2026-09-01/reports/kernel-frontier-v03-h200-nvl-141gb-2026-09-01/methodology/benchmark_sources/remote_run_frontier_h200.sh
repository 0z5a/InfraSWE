#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-frontier-v03-h200-nvl}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

attention_backends=(torch-sdpa-flash fa1 fa2 fa3 fa4)
attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)
classic_cases=(
  vector-add-bf16-16m
  softmax-bf16-4096x4096
  layernorm-bf16-4096x4096
  rmsnorm-bf16-4096x4096
  swiglu-bf16-8192x4096
  rope-bf16-b4-s2048-h16-d128
  gemm-bf16-4096-cube
)

mkdir -p "${run_root}/raw/calibration" "${run_root}/raw/classic"
for backend in "${attention_backends[@]}"; do
  mkdir -p "${run_root}/raw/attention/${backend}" "${run_root}/profiles/${backend}"
done
mkdir -p "${run_root}/profiles/triton-fixed-config"

cd "${benchmark_root}"
if [[ ! -s "${run_root}/nvidia-smi-before.txt" ]]; then
  nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-before.txt"
fi

for replay in 1 2 3; do
  output="${run_root}/raw/calibration/replay-${replay}.json"
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python}" calibrate_gpu.py \
      --output "${output}" \
      --replay-index "${replay}" \
      --samples 30
  fi
done

for backend in "${attention_backends[@]}"; do
  case "${backend}" in
    torch-sdpa-flash)
      python_path=""
      commit=pytorch-2.8.0-cu128
      ;;
    fa1)
      python_path="${workspace}/envs/fa1"
      commit=6d48e14a6c2f551db96f0badc658a6279a929df3
      ;;
    fa2)
      python_path="${workspace}/envs/fa2"
      commit=ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
      ;;
    fa3)
      python_path="${workspace}/envs/fa3"
      commit=ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
      ;;
    fa4)
      python_path="${workspace}/envs/fa4"
      commit=pypi-flash-attn-4-4.0.0b28
      ;;
  esac

  for replay in 1 2 3; do
    output="${run_root}/raw/attention/${backend}/replay-${replay}.json"
    if [[ ! -s "${output}" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
        --backend "${backend}" \
        --output "${output}" \
        --replay-index "${replay}" \
        --blocks 30 \
        --min-timed-span-ms 50 \
        --implementation-commit "${commit}"
    fi
  done

  for case_id in "${attention_cases[@]}"; do
    output="${run_root}/profiles/${backend}/${case_id}.json"
    if [[ ! -s "${output}" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
        --suite attention \
        --backend "${backend}" \
        --case-id "${case_id}" \
        --output "${output}"
    fi
  done
done

for replay in 1 2 3; do
  output="${run_root}/raw/classic/replay-${replay}.json"
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python}" classic_bench.py \
      --output "${output}" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50
  fi
done

for case_id in "${classic_cases[@]}"; do
  output="${run_root}/profiles/triton-fixed-config/${case_id}.json"
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python}" profile_case.py \
      --suite classic \
      --backend triton-fixed-config \
      --case-id "${case_id}" \
      --output "${output}"
  fi
done

PYTHONPATH="${workspace}/envs/fa4" "${python}" collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --implementations fa1 fa2 fa3 fa4
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "H200 NVL kernel-frontier formal run complete: ${run_root}"
