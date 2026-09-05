#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"

result_root="${INFRASWE_COMMUNICATION_RESULT_ROOT:-results/historical-pr-blind-20260901/communication-bulk-95pct}"
queue_lock="${INFRASWE_COMMUNICATION_QUEUE_LOCK:-${result_root}/queue-lock-groups3000.json}"
progress_file="${result_root}/campaign-progress.json"
inference_progress="${INFRASWE_INFERENCE_PROGRESS:-results/historical-pr-blind-20260901/inference-bulk-95pct/campaign-progress.json}"

until [[ -f "${queue_lock}" && -f /workspace/communication-pr-corpus/READY ]]; do
  echo "waiting for frozen communication queue and repository preparation"
  sleep 30
done

until [[ "$(jq -r '.status // "pending"' "${inference_progress}" 2>/dev/null || echo pending)" == complete ]]; do
  echo "waiting for inference campaign to release GitHub quota"
  sleep 60
done

target_count="$(jq -r '.target_count' "${queue_lock}")"
group_count="$(jq -r '.group_count' "${queue_lock}")"
if [[ "${target_count}" -le 0 || "${group_count}" -le 0 ]]; then
  echo "invalid communication queue dimensions" >&2
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
    shadow_dir="${result_root}/decision-v061-shadow/groups/$(printf 'group-%04d' "${group_index}")"
    if [[ "${INFRASWE_DECISION_V061_SHADOW:-0}" == 1 && -f "${shadow_dir}/activation.json" ]]; then
      # A crash after sealing the primary chain must not silently skip shadow audit.
      "${INFRASWE_LOCAL_PYTHON:-/venv/main/bin/python}" \
        benchmarks/historical_prs/decision_v061_shadow.py audit \
        --group-dir "${group_dir}" --queue-lock "${queue_lock}" --output-dir "${shadow_dir}"
    fi
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
  bash benchmarks/historical_prs/run_communication_bulk_round.sh "${group_index}"
  echo "campaign group=${group_index} complete"
done

"${INFRASWE_LOCAL_PYTHON:-/venv/main/bin/python}" \
  benchmarks/historical_prs/summarize_training_bulk_campaign.py \
  --result-root "${result_root}" \
  --expected-cases "${target_count}" \
  --coverage-queue "${queue_lock}" \
  --queue-lock "${queue_lock}" \
  --initial-policy "${result_root}/seed-policy.json" \
  --output "${result_root}/campaign-summary.json"

target_metric_improved="$(jq -r '.target_metric_improved' "${result_root}/campaign-summary.json")"
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
