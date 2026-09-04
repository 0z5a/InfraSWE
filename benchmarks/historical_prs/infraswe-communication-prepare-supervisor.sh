#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/infraswe
export PATH=/workspace/tools:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"
export GH_CONFIG_DIR=/workspace/infraswe-secrets/gh

result_root=results/historical-pr-blind-20260901/communication-bulk-95pct
queue_lock="${result_root}/queue-lock-groups3000.json"
mkdir -p "${result_root}"
touch "${result_root}/communication-95pct-requested"

repository_pid=""
cleanup() {
  if [[ -n "${repository_pid}" ]] && kill -0 "${repository_pid}" 2>/dev/null; then
    kill "${repository_pid}" 2>/dev/null || true
    wait "${repository_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

if [[ ! -f /workspace/communication-pr-corpus/READY ]]; then
  bash benchmarks/historical_prs/prepare_communication_repositories.sh &
  repository_pid="$!"
fi

if [[ ! -f "${queue_lock}" ]]; then
  /venv/main/bin/python benchmarks/historical_prs/prepare_training_bulk_queue.py \
    --profile communication \
    --identity-source git-refs \
    --target-fraction 0.95 \
    --group-size 3000 \
    --output "${queue_lock}"
fi

if [[ ! -f "${result_root}/seed-policy.json" ]]; then
  /venv/main/bin/python benchmarks/historical_prs/freeze_inference_bulk_seed_policy.py \
    --domain communication \
    --source-policy \
    results/historical-pr-blind-20260901/supplemental-r14/preselection-policy.json \
    --output "${result_root}/seed-policy.json"
fi

if [[ -n "${repository_pid}" ]]; then
  wait "${repository_pid}"
  touch /workspace/communication-pr-corpus/READY
fi

jq '{profile,target_count,target_fraction,group_size,group_count,last_group_size,project_quotas,queue_lock_sha256}' \
  "${queue_lock}"
