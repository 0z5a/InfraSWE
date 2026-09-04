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

for ((group_index = 0; group_index < group_count; group_index++)); do
  group_dir="${result_root}/groups/$(printf 'group-%04d' "${group_index}")"
  if [[ -f "${group_dir}/oracle-audit.json" ]]; then
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
jq -n \
  --arg status complete \
  --argjson target_metric_improved "${target_metric_improved}" \
  --argjson group_count "${group_count}" \
  --argjson target_count "${target_count}" \
  --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{status:$status,target_metric_improved:$target_metric_improved,group_count:$group_count,target_count:$target_count,completed_at:$completed_at}' \
  > "${progress_file}.tmp"
mv "${progress_file}.tmp" "${progress_file}"

if [[ "${INFRASWE_PUBLISH_ON_COMPLETE:-0}" == 1 && "${target_metric_improved}" == true ]]; then
  if [[ "$(git branch --show-current)" != main ]]; then
    echo "completion publish requires the main branch" >&2
    exit 1
  fi
  git add \
    benchmarks/historical_prs/acquire_training_bulk_group.py \
    benchmarks/historical_prs/audit_training_bulk_group.py \
    benchmarks/historical_prs/derive_training_bulk_policy_iteration.py \
    benchmarks/historical_prs/freeze_inference_bulk_seed_policy.py \
    benchmarks/historical_prs/freeze_training_bulk_group.py \
    benchmarks/historical_prs/infraswe-communication-bulk-supervisor.conf \
    benchmarks/historical_prs/infraswe-communication-bulk-supervisor.sh \
    benchmarks/historical_prs/infraswe-communication-prepare-supervisor.conf \
    benchmarks/historical_prs/infraswe-communication-prepare-supervisor.sh \
    benchmarks/historical_prs/prepare_communication_repositories.sh \
    benchmarks/historical_prs/prepare_training_bulk_queue.py \
    benchmarks/historical_prs/reveal_training_bulk_group.py \
    benchmarks/historical_prs/run_communication_bulk_campaign.sh \
    benchmarks/historical_prs/run_communication_bulk_round.sh \
    benchmarks/historical_prs/run_inference_bulk_campaign.sh \
    benchmarks/historical_prs/run_training_bulk_group.py \
    benchmarks/historical_prs/summarize_training_bulk_campaign.py \
    tests/test_historical_bulk_gates.py \
    "${result_root}"
  if ! git diff --cached --quiet; then
    git_commit_name="${INFRASWE_GIT_USER_NAME:-$(git log -1 --format=%an)}"
    git_commit_email="${INFRASWE_GIT_USER_EMAIL:-$(git log -1 --format=%ae)}"
    git -c user.name="${git_commit_name}" -c user.email="${git_commit_email}" \
      commit -m "bench: complete 95 percent communication PR campaign"
  fi
  git fetch origin main
  git rebase origin/main
  git push origin HEAD:main
elif [[ "${INFRASWE_PUBLISH_ON_COMPLETE:-0}" == 1 ]]; then
  echo "aggregate target metric did not improve; skipping commit and push"
fi

training_progress=results/historical-pr-blind-20260901/training-bulk-8000/campaign-progress.json
inference_queue=results/historical-pr-blind-20260901/inference-bulk-95pct/queue-lock-groups3000.json
training_pending=false
inference_pending=false
if [[ "$(jq -r '.status // "pending"' "${training_progress}" 2>/dev/null || echo pending)" != complete ]]; then
  training_pending=true
fi
if [[ -f "${inference_queue}" ]] && \
   [[ "$(jq -r '.status // "pending"' "${inference_progress}" 2>/dev/null || echo pending)" != complete ]]; then
  inference_pending=true
fi

if [[ -n "${INFRASWE_GITHUB_CREDENTIAL_COPY:-}" && "${training_pending}" != true && "${inference_pending}" != true ]]; then
  rm -f -- "${INFRASWE_GITHUB_CREDENTIAL_COPY}"
fi

if [[ "${INFRASWE_STOP_INSTANCE_ON_COMPLETE:-0}" == 1 && "${training_pending}" != true && "${inference_pending}" != true ]]; then
  vastai stop instance "${CONTAINER_ID}" --api-key "${CONTAINER_API_KEY}"
fi
