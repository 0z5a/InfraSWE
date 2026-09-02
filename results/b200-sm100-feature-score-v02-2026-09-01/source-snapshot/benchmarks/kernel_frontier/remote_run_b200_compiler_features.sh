#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/b200-sm100-compiler-features-v01}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
candidate_root="${INFRASWE_B200_CANDIDATE_ROOT:-${workspace}/b200-candidates}"
dynamic_root="${INFRASWE_B200_DYNAMIC_ROOT:-${workspace}/b200-evaluator-evidence}"
strict="${INFRASWE_B200_REQUIRE_CERTIFIED:-0}"
include_optional="${INFRASWE_B200_INCLUDE_OPTIONAL:-0}"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p \
  "${run_root}/capability-targets" \
  "${run_root}/replays" \
  "${run_root}/native-evidence"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/started-at.txt"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-before.txt"

cd "${benchmark_root}"
PYTHONPATH="${workspace}/src" "${python}" b200_capability_probe.py \
  --device-index "${gpu}" \
  --artifact-root "${run_root}/capability-targets" \
  --output "${run_root}/capability.json" \
  --require-b200 \
  --require-toolchain

replay_options=()
summary_options=()
if [[ "${include_optional}" == "1" ]]; then
  replay_options+=(--include-optional)
fi
if [[ "${strict}" == "1" ]]; then
  replay_options+=(--require-certified)
  summary_options+=(--require-certified)
fi

for replay in 1 2 3; do
  PYTHONPATH="${workspace}/src" "${python}" b200_feature_replay.py \
    --replay-index "${replay}" \
    --capability "${run_root}/capability.json" \
    --artifact-root "${candidate_root}" \
    --dynamic-root "${dynamic_root}" \
    --evidence-root "${run_root}/native-evidence/replay-${replay}" \
    --output "${run_root}/replays/replay-${replay}.json" \
    "${replay_options[@]}"
done

PYTHONPATH="${workspace}/src" "${python}" summarize_b200_compiler_features.py \
  --root "${run_root}/replays" \
  --json-output "${run_root}/b200-compiler-features.json" \
  --markdown-output "${run_root}/b200-compiler-features.md" \
  "${summary_options[@]}"

nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
"${python}" -m zipfile -c "${run_root}.zip" "${run_root}"
echo "B200 compiler-feature initial report complete: ${run_root}.zip"
