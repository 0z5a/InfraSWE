#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu_list="${INFRASWE_GPUS:-0,1}"
run_root="${1:-${workspace}/runs/hopper-sm90-features}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
feature_root="${run_root}/features"

IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
if [[ "${#gpu_ids[@]}" -lt 2 ]]; then
  echo "multimem runtime proof requires at least two visible GPUs" >&2
  exit 2
fi
for gpu in "${gpu_ids[@]}"; do
  active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
  if [[ -n "${active_pids}" ]]; then
    echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
    exit 3
  fi
done

mkdir -p "${feature_root}"
cd "${benchmark_root}"
nvidia-smi -q > "${run_root}/nvidia-smi-before.txt"
nvidia-smi topo -m > "${run_root}/nvidia-smi-topology.txt"
for replay in 1 2 3; do
  output="${feature_root}/replay-${replay}.json"
  artifact_root="${feature_root}/artifacts/replay-${replay}"
  if [[ ! -s "${output}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_list}" \
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
  --json-output "${run_root}/hopper-features.json" \
  --markdown-output "${run_root}/hopper-features.md"
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
nvidia-smi -q > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "Hopper TMA, WGMMA, and multimem/NVLS feature supplement complete: ${run_root}"
