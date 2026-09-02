#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${workspace}/runs/kernel-sm120-fa4-probe-v03"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p "${run_root}/profiles" "${run_root}/raw"
nvidia-smi -q > "${run_root}/nvidia-smi-before.txt"
cd "${benchmark_root}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
  --suite attention \
  --backend fa4 \
  --case-id common-b2-s1024-h16-d64-causal \
  --output "${run_root}/profiles/fa4-common-b2-s1024-h16-d64-causal.json"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
  --backend fa4 \
  --output "${run_root}/raw/fa4-replay-1.json" \
  --replay-index 1 \
  --blocks 3 \
  --min-timed-span-ms 10 \
  --implementation-commit pypi-flash-attn-4-4.0.0b28

nvidia-smi -q > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "SM120 FA4 compatibility probe complete"

