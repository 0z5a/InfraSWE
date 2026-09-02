#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
stage_root="${workspace}/runs/kernel-b200-cuda133-setup"
benchmark_root="${workspace}/benchmarks/kernel_frontier"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

for executable in nvcc ptxas cuobjdump nvdisasm nvidia-smi; do
  if ! command -v "${executable}" >/dev/null; then
    echo "required CUDA 13.3 tool is missing: ${executable}" >&2
    exit 4
  fi
done

mkdir -p "${stage_root}/target-probes"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/started-at.txt"
nvidia-smi -q -i "${gpu}" > "${stage_root}/nvidia-smi-before.txt"
nvidia-smi topo -m > "${stage_root}/nvidia-smi-topo.txt"

cd "${benchmark_root}"
PYTHONPATH="${workspace}/src" "${python}" b200_capability_probe.py \
  --device-index "${gpu}" \
  --artifact-root "${stage_root}/target-probes" \
  --output "${stage_root}/capability.json" \
  --require-b200 \
  --require-toolchain

PYTHONPATH="${workspace}/src" "${python}" -m infraswe lease preflight \
  --profile gpu-1x-sm100-b200-cuda133 \
  --profiles-dir "${workspace}/profiles" \
  --output "${stage_root}/hardware-manifest.json"

nvcc --version > "${stage_root}/nvcc-version.txt"
ptxas --version > "${stage_root}/ptxas-version.txt"
cuobjdump --version > "${stage_root}/cuobjdump-version.txt"
nvdisasm --version > "${stage_root}/nvdisasm-version.txt"
nvidia-smi -q -i "${gpu}" > "${stage_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/completed-at.txt"
echo "B200 CUDA 13.3 capability setup complete: ${stage_root}"
