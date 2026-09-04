#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/infraswe
until [[ -f /workspace/training-pr-corpus/READY ]]; do
  echo "waiting for training repository preparation"
  sleep 30
done

common_env=(
  PATH=/workspace/tools:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  GH_CONFIG_DIR=/workspace/infraswe-secrets/gh \
  INFRASWE_EXECUTION_MODE=local \
  INFRASWE_LOCAL_PYTHON=/venv/main/bin/python \
  INFRASWE_TRAINING_RESULT_ROOT=results/historical-pr-blind-20260901/training-bulk-8000 \
  INFRASWE_TRAINING_WORKERS=8 \
  INFRASWE_TRAINING_LANES_PER_PROJECT=16 \
  INFRASWE_GITHUB_WORKERS=12 \
  INFRASWE_GITHUB_BATCH_SIZE=10 \
  INFRASWE_GITHUB_REVEAL_BATCH_SIZE=50 \
  INFRASWE_TEST_TIMEOUT=45
)

if [[ ! -f results/historical-pr-blind-20260901/training-bulk-8000/groups/group-0010/oracle-audit.json ]]; then
  env \
    "${common_env[@]}" \
    INFRASWE_GITHUB_CREDENTIAL_COPY= \
    INFRASWE_TRAINING_QUEUE_LOCK=results/historical-pr-blind-20260901/training-bulk-8000/queue-lock-groups100.json \
    INFRASWE_EXPECTED_CAMPAIGN_CASES=680 \
    INFRASWE_PUBLISH_ON_COMPLETE=0 \
    INFRASWE_STOP_INSTANCE_ON_COMPLETE=0 \
    stdbuf -oL -eL bash benchmarks/historical_prs/run_training_bulk_campaign.sh 6 10 2>&1
fi

result_root=results/historical-pr-blind-20260901/training-bulk-8000
queue_ready_marker="${result_root}/training-95pct-queue-ready"
queue_lock="${result_root}/queue-lock-95pct-groups3000.json"
if [[ ! -f "${queue_ready_marker}" || ! -f "${queue_lock}" ]]; then
  jq -n \
    --arg status awaiting_95pct_queue \
    --argjson processed_count 680 \
    --arg queue_lock "${queue_lock}" \
    --arg updated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{status:$status,processed_count:$processed_count,queue_lock:$queue_lock,updated_at:$updated_at}' \
    > "${result_root}/campaign-progress.json.tmp"
  mv "${result_root}/campaign-progress.json.tmp" "${result_root}/campaign-progress.json"
fi
until [[ -f "${queue_ready_marker}" && -f "${queue_lock}" ]]; do
  echo "waiting at group 10 boundary for frozen 95 percent training queue"
  sleep 30
done

target_count="$(jq -r '.target_count' "${queue_lock}")"
end_group="$(jq '[.cases[].group_index] | max' "${queue_lock}")"
if [[ "${target_count}" -le 680 || "${end_group}" -lt 11 ]]; then
  echo "invalid 95 percent training queue dimensions" >&2
  exit 2
fi

exec env \
  "${common_env[@]}" \
  INFRASWE_GITHUB_WORKERS=4 \
  INFRASWE_GITHUB_CREDENTIAL_COPY=/workspace/infraswe-secrets/gh/hosts.yml \
  INFRASWE_TRAINING_QUEUE_LOCK="${queue_lock}" \
  INFRASWE_EXPECTED_CAMPAIGN_CASES="${target_count}" \
  INFRASWE_PUBLISH_ON_COMPLETE=1 \
  INFRASWE_STOP_INSTANCE_ON_COMPLETE=1 \
  stdbuf -oL -eL bash benchmarks/historical_prs/run_training_bulk_campaign.sh 11 "${end_group}" 2>&1
