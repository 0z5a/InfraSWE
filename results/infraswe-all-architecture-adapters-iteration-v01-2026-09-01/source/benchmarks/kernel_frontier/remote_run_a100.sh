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

mkdir -p \
  "${run_root}/raw/calibration" \
  "${run_root}/raw/attention/torch-sdpa-flash" \
  "${run_root}/raw/attention/fa1" \
  "${run_root}/raw/attention/fa2" \
  "${run_root}/raw/attention/fa3" \
  "${run_root}/raw/classic"

cd "${benchmark_root}"
nvidia-smi -q -i "${gpu}" >"${run_root}/nvidia-smi-before.txt"

for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" calibrate_gpu.py \
    --output "${run_root}/raw/calibration/replay-${replay}.json" \
    --replay-index "${replay}" \
    --samples 30
done

for backend in torch-sdpa-flash fa1 fa2 fa3; do
  case "${backend}" in
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
    torch-sdpa-flash)
      python_path=""
      commit=pytorch-2.8.0-cu128
      ;;
  esac
  for replay in 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
      --backend "${backend}" \
      --output "${run_root}/raw/attention/${backend}/replay-${replay}.json" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50 \
      --implementation-commit "${commit}"
  done
done

for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" classic_bench.py \
    --output "${run_root}/raw/classic/replay-${replay}.json" \
    --replay-index "${replay}" \
    --blocks 30 \
    --min-timed-span-ms 50
done

nvidia-smi -q -i "${gpu}" >"${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"${run_root}/completed-at.txt"
echo "A100 formal run complete: ${run_root}"
