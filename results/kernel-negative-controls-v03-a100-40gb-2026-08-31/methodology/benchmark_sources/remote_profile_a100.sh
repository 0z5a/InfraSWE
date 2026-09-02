#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-frontier-v03-a100}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

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

cd "${benchmark_root}"
for backend in torch-sdpa-flash fa1 fa2 fa3; do
  case "${backend}" in
    fa1|fa2|fa3) python_path="${workspace}/envs/${backend}" ;;
    torch-sdpa-flash) python_path="" ;;
  esac
  mkdir -p "${run_root}/profiles/${backend}"
  for case_id in "${attention_cases[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
      --suite attention \
      --backend "${backend}" \
      --case-id "${case_id}" \
      --output "${run_root}/profiles/${backend}/${case_id}.json"
  done
done

mkdir -p "${run_root}/profiles/triton-fixed-config"
for case_id in "${classic_cases[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" profile_case.py \
    --suite classic \
    --backend triton-fixed-config \
    --case-id "${case_id}" \
    --output "${run_root}/profiles/triton-fixed-config/${case_id}.json"
done

echo "A100 per-case profiler evidence complete: ${run_root}/profiles"
