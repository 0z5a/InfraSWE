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

if [[ "${INFRASWE_PUBLISH_ON_COMPLETE:-0}" == 1 && \
      "${target_metric_improved}" == true && \
      "${release_quality_gate_satisfied}" == true ]]; then
  if [[ "$(git branch --show-current)" != main ]]; then
    echo "completion publish requires the main branch" >&2
    exit 1
  fi
  git add \
    benchmarks/historical_prs/acquire_training_bulk_group.py \
    benchmarks/historical_prs/audit_training_bulk_group.py \
    benchmarks/historical_prs/compose_training_95pct_queue.py \
    benchmarks/historical_prs/derive_training_bulk_policy_iteration.py \
    benchmarks/historical_prs/freeze_inference_bulk_seed_policy.py \
    benchmarks/historical_prs/freeze_training_bulk_group.py \
    benchmarks/historical_prs/historical_bulk_quality_gates.py \
    benchmarks/historical_prs/infraswe-inference-bulk-supervisor.conf \
    benchmarks/historical_prs/infraswe-inference-bulk-supervisor.sh \
    benchmarks/historical_prs/infraswe-inference-prepare-supervisor.conf \
    benchmarks/historical_prs/infraswe-inference-prepare-supervisor.sh \
    benchmarks/historical_prs/prepare_inference_repositories.sh \
    benchmarks/historical_prs/prepare_training_bulk_queue.py \
    benchmarks/historical_prs/reveal_training_bulk_group.py \
    benchmarks/historical_prs/run_inference_bulk_campaign.sh \
    benchmarks/historical_prs/run_inference_bulk_round.sh \
    benchmarks/historical_prs/run_training_bulk_campaign.sh \
    benchmarks/historical_prs/run_training_bulk_group.py \
    benchmarks/historical_prs/summarize_training_bulk_campaign.py \
    tests/test_historical_bulk_gates.py \
    "${result_root}"
  if ! git diff --cached --quiet; then
    git_commit_name="${INFRASWE_GIT_USER_NAME:-$(git log -1 --format=%an)}"
    git_commit_email="${INFRASWE_GIT_USER_EMAIL:-$(git log -1 --format=%ae)}"
    git -c user.name="${git_commit_name}" -c user.email="${git_commit_email}" \
      commit -m "bench: complete 95 percent inference PR campaign"
  fi
  git fetch origin main
  git rebase origin/main
  git push origin HEAD:main
elif [[ "${INFRASWE_PUBLISH_ON_COMPLETE:-0}" == 1 && \
        "${release_quality_gate_satisfied}" != true ]]; then
  echo "hard release quality gate not satisfied; skipping commit and push"
elif [[ "${INFRASWE_PUBLISH_ON_COMPLETE:-0}" == 1 ]]; then
  echo "aggregate target metric did not improve; skipping commit and push"
fi

training_progress=results/historical-pr-blind-20260901/training-bulk-8000/campaign-progress.json
training_queue=results/historical-pr-blind-20260901/training-bulk-8000/queue-lock-95pct-groups3000.json
communication_root=results/historical-pr-blind-20260901/communication-bulk-95pct
communication_progress="${communication_root}/campaign-progress.json"
communication_queue="${communication_root}/queue-lock-groups3000.json"
communication_request="${communication_root}/communication-95pct-requested"
training_pending=false
communication_pending=false
if [[ -f "${training_queue}" ]] && \
   [[ "$(jq -r '.status // "pending"' "${training_progress}" 2>/dev/null || echo pending)" != complete ]]; then
  training_pending=true
  echo "training bulk campaign is pending; deferring credential cleanup and shutdown"
fi
if [[ -f "${communication_request}" || -f "${communication_queue}" ]] && \
   [[ "$(jq -r '.status // "pending"' "${communication_progress}" 2>/dev/null || echo pending)" != complete ]]; then
  communication_pending=true
  echo "communication bulk campaign is pending; deferring credential cleanup and shutdown"
fi

if [[ -n "${INFRASWE_GITHUB_CREDENTIAL_COPY:-}" && "${training_pending}" != true && "${communication_pending}" != true ]]; then
  rm -f -- "${INFRASWE_GITHUB_CREDENTIAL_COPY}"
fi

if [[ "${INFRASWE_STOP_INSTANCE_ON_COMPLETE:-0}" == 1 && "${training_pending}" != true && "${communication_pending}" != true ]]; then
  vastai stop instance "${CONTAINER_ID}" --api-key "${CONTAINER_API_KEY}"
fi
