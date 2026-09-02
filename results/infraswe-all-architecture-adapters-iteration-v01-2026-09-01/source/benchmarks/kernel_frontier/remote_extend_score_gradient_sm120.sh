#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${workspace}/runs/kernel-score-gradient-pilot-v03-sm120"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

cd "${benchmark_root}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" sweep_mediocre.py \
  --evidence-root "${run_root}" \
  --output "${run_root}/sweep-full.json" \
  --passes 0 8 16 32 48 64 96 128 160 192 256 384 512 768 1024 \
  --samples 7
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/extended-completed-at.txt"
echo "SM120 extended score-gradient sweep complete"

