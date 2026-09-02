#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-frontier-v03-h200-nvl}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
feature_root="${run_root}/features"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p "${feature_root}"
cd "${benchmark_root}"
for replay in 1 2 3; do
  output="${feature_root}/replay-${replay}.json"
  artifact_root="${feature_root}/artifacts/replay-${replay}"
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TRITON_CACHE_DIR="${feature_root}/triton-cache/replay-${replay}" \
      "${python}" h200_feature_bench.py \
        --output "${output}" \
        --artifact-root "${artifact_root}" \
        --replay-index "${replay}" \
        --blocks 30 \
        --min-timed-span-ms 50
  fi
done

"${python}" summarize_h200_features.py \
  --root "${feature_root}" \
  --json-output "${run_root}/h200-features.json" \
  --markdown-output "${run_root}/h200-features.md"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/h200-features-completed-at.txt"
echo "H200 TMA and multimem feature supplement complete: ${run_root}"
