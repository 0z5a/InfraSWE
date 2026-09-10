#!/usr/bin/env bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/infraswe
export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/workspace/tools:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export GH_CONFIG_DIR=/workspace/infraswe-secrets/gh

result_root=results/historical-pr-blind-20260901/inference-bulk-95pct
queue_lock="${result_root}/queue-lock-groups3000.json"
mkdir -p "${result_root}"

repository_pid=""
external_repository_prepare=false
cleanup() {
  if [[ -n "${repository_pid}" ]] && kill -0 "${repository_pid}" 2>/dev/null; then
    kill "${repository_pid}" 2>/dev/null || true
    wait "${repository_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

if [[ ! -f /workspace/inference-pr-corpus/READY ]]; then
  if pgrep -f '^bash benchmarks/historical_prs/prepare_inference_repositories.sh$' \
    >/dev/null; then
    external_repository_prepare=true
    echo "reusing in-flight inference repository preparation"
  else
    bash benchmarks/historical_prs/prepare_inference_repositories.sh &
    repository_pid="$!"
  fi
fi

if [[ ! -f "${queue_lock}" ]]; then
  prior_args=()
  while IFS= read -r path; do
    prior_args+=(--prior-lock "$path")
  done < <(
    find results/historical-pr-blind-20260901 -type f \
      \( -name selection-lock.json -o -name prediction-locks.json \) | sort
  )
  /venv/main/bin/python benchmarks/historical_prs/prepare_training_bulk_queue.py \
    --profile inference \
    --identity-source git-refs \
    --target-fraction 0.95 \
    --group-size 3000 \
    "${prior_args[@]}" \
    --output "${queue_lock}"
fi

if [[ ! -f "${result_root}/seed-policy.json" ]]; then
  /venv/main/bin/python \
    benchmarks/historical_prs/freeze_inference_bulk_seed_policy.py \
    --source-policy \
    results/historical-pr-blind-20260901/supplemental-r25/preselection-policy.json \
    --output "${result_root}/seed-policy.json"
fi

if [[ -n "${repository_pid}" ]]; then
  wait "${repository_pid}"
  touch /workspace/inference-pr-corpus/READY
elif [[ "${external_repository_prepare}" == true ]]; then
  while pgrep -f '^bash benchmarks/historical_prs/prepare_inference_repositories.sh$' \
    >/dev/null; do
    echo "waiting for in-flight inference repository preparation"
    sleep 30
  done
  bash benchmarks/historical_prs/prepare_inference_repositories.sh
  touch /workspace/inference-pr-corpus/READY
fi

jq '{profile,target_count,target_fraction,group_size,group_count,last_group_size,project_quotas,queue_lock_sha256}' \
  "${queue_lock}"
