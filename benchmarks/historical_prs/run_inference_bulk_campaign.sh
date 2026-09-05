#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"

result_root="${INFRASWE_INFERENCE_RESULT_ROOT:-results/historical-pr-blind-20260901/inference-bulk-95pct}"
queue_lock="${INFRASWE_INFERENCE_QUEUE_LOCK:-${result_root}/queue-lock-groups3000.json}"
progress_file="${result_root}/campaign-progress.json"
v06_boundary_group="${INFRASWE_V06_BOUNDARY_GROUP:-0}"
v06_resume_marker="${INFRASWE_V06_RESUME_MARKER:-${result_root}/v06-rfc-implementation-complete}"
v06_wait_marker="${result_root}/v06-rfc-boundary-reached"

until [[ -f "${queue_lock}" && -f /workspace/inference-pr-corpus/READY ]]; do
  echo "waiting for frozen inference queue and repository preparation"
  sleep 30
done

target_count="$(jq -r '.target_count' "${queue_lock}")"
group_count="$(jq -r '.group_count' "${queue_lock}")"
if [[ "${target_count}" -le 0 || "${group_count}" -le 0 ]]; then
  echo "invalid inference queue dimensions" >&2
  exit 2
fi

wait_for_rate_budget() {
  local required="$1"
  local response remaining
  while true; do
    if response="$(gh api graphql -f 'query=query { rateLimit { remaining resetAt } }' 2>/dev/null)"; then
      remaining="$(jq -r '.data.rateLimit.remaining // 0' <<<"${response}")"
      if [[ "${remaining}" -ge "${required}" ]]; then
        echo "graphql remaining=${remaining} required=${required}"
        return 0
      fi
      echo "graphql budget low: remaining=${remaining}; waiting 60s"
    else
      echo "graphql budget probe failed; waiting 30s"
      sleep 30
      continue
    fi
    sleep 60
  done
}

group_is_complete() {
  local group_dir="$1"
  local artifact
  for artifact in \
    input-lock.json \
    exact-head-evidence.json \
    judgment-locks.json \
    outcome-reveal.json \
    oracle-audit.json \
    next-policy.json; do
    [[ -f "${group_dir}/${artifact}" ]] || return 1
  done
}

for ((group_index = 0; group_index < group_count; group_index++)); do
  group_dir="${result_root}/groups/$(printf 'group-%04d' "${group_index}")"
  if group_is_complete "${group_dir}"; then
    echo "campaign group=${group_index} already complete"
    continue
  fi
  wait_for_rate_budget 100
  jq -n \
    --arg status running \
    --argjson group_index "${group_index}" \
    --argjson group_count "${group_count}" \
    --argjson target_count "${target_count}" \
    --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{status:$status,group_index:$group_index,group_count:$group_count,target_count:$target_count,started_at:$started_at}' \
    > "${progress_file}.tmp"
  mv "${progress_file}.tmp" "${progress_file}"
  echo "campaign group=${group_index} start"
  bash benchmarks/historical_prs/run_inference_bulk_round.sh "${group_index}"
  echo "campaign group=${group_index} complete"
  if [[ "${group_index}" -eq "${v06_boundary_group}" && ! -f "${v06_resume_marker}" ]]; then
    jq -n \
      --arg status awaiting_v06_rfc \
      --argjson group_index "${group_index}" \
      --argjson group_count "${group_count}" \
      --argjson target_count "${target_count}" \
      --arg resume_marker "${v06_resume_marker}" \
      --arg reached_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{status:$status,group_index:$group_index,group_count:$group_count,target_count:$target_count,resume_marker:$resume_marker,reached_at:$reached_at}' \
      > "${progress_file}.tmp"
    mv "${progress_file}.tmp" "${progress_file}"
    touch "${v06_wait_marker}"
    echo "campaign waiting at group=${group_index} boundary for v0.6 implementation"
    until [[ -f "${v06_resume_marker}" ]]; do
      sleep 60
    done
    echo "v0.6 implementation marker found; resuming inference campaign"
  fi
done

"${INFRASWE_LOCAL_PYTHON:-/venv/main/bin/python}" \
  benchmarks/historical_prs/summarize_training_bulk_campaign.py \
  --result-root "${result_root}" \
  --expected-cases "${target_count}" \
  --coverage-queue "${queue_lock}" \
  --queue-lock "${queue_lock}" \
  --initial-policy "${result_root}/seed-policy.json" \
  --output "${result_root}/campaign-summary.json"

target_metric_improved="$(
  jq -r '.target_metric_improved' "${result_root}/campaign-summary.json"
)"
release_quality_gate_satisfied="$(
  jq -r '.release_quality_gate_satisfied' "${result_root}/campaign-summary.json"
)"

jq -n \
  --arg status complete \
  --argjson target_metric_improved "${target_metric_improved}" \
  --argjson release_quality_gate_satisfied "${release_quality_gate_satisfied}" \
  --argjson group_count "${group_count}" \
  --argjson target_count "${target_count}" \
  --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{status:$status,target_metric_improved:$target_metric_improved,release_quality_gate_satisfied:$release_quality_gate_satisfied,group_count:$group_count,target_count:$target_count,completed_at:$completed_at}' \
  > "${progress_file}.tmp"
mv "${progress_file}.tmp" "${progress_file}"

# Campaign completion is not release qualification or evidence synchronization.
# Deliberately fail closed: this worker has no independent publication attestor.
# A separately verified operator workflow must check all three campaigns, the
# 95/99/99 contract, full SHA-256 synchronization, tests and the exact release.
echo "campaign complete; finalization pending independent verification"
echo "automatic commit/push, credential deletion and instance stop are disabled"
