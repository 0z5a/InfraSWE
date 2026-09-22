#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_GROUP END_GROUP" >&2
  exit 2
fi

start_group="$1"
end_group="$2"
result_root="${INFRASWE_TRAINING_RESULT_ROOT:-results/historical-pr-blind-20260901/training-bulk-8000}"
token_file="${INFRASWE_GITHUB_TOKEN_FILE:-}"
progress_file="$result_root/campaign-progress.json"
expected_cases="${INFRASWE_EXPECTED_CAMPAIGN_CASES:-8000}"
github_batch_size="${INFRASWE_GITHUB_BATCH_SIZE:-1}"
if [[ "${github_batch_size}" -le 0 ]]; then
  echo "INFRASWE_GITHUB_BATCH_SIZE must be positive" >&2
  exit 2
fi

if [[ -n "$token_file" ]]; then
  if [[ ! -r "$token_file" ]]; then
    echo "GitHub token file is unavailable: $token_file" >&2
    exit 1
  fi
  IFS= read -r GH_TOKEN < "$token_file"
else
  IFS= read -r GH_TOKEN < <(gh auth token)
fi
export GH_TOKEN

wait_for_rate_budget() {
  local required="$1"
  local response remaining
  while true; do
    if response="$(gh api graphql -f 'query=query { rateLimit { remaining resetAt } }' 2>/dev/null)"; then
      remaining="$(jq -r '.data.rateLimit.remaining // 0' <<<"$response")"
      if [[ "$remaining" -ge "$required" ]]; then
        echo "graphql remaining=$remaining required=$required"
        return 0
      fi
      echo "graphql budget low: remaining=$remaining; waiting 60s"
    else
      echo "graphql budget probe failed; waiting 30s"
      sleep 30
      continue
    fi
    sleep 60
  done
}

for ((group_index = start_group; group_index <= end_group; group_index++)); do
  group_case_count="$(
    jq --argjson group_index "$group_index" \
      '[.cases[] | select(.group_index == $group_index)] | length' \
      "$INFRASWE_TRAINING_QUEUE_LOCK"
  )"
  required_budget="$(((group_case_count + github_batch_size - 1) / github_batch_size + 40))"
  if [[ "${required_budget}" -gt 4500 ]]; then
    required_budget=4500
  fi
  wait_for_rate_budget "${required_budget}"
  jq -n \
    --arg status running \
    --argjson group_index "$group_index" \
    --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{status:$status,group_index:$group_index,started_at:$started_at}' \
    > "$progress_file.tmp"
  mv "$progress_file.tmp" "$progress_file"
  echo "campaign group=$group_index start"
  bash benchmarks/historical_prs/run_training_bulk_round.sh "$group_index"
  echo "campaign group=$group_index complete"
done

summary_queue_args=(--coverage-queue "$INFRASWE_TRAINING_QUEUE_LOCK")
for queue_path in \
  "$result_root/queue-lock.json" \
  "$result_root/queue-lock-groups100.json" \
  "$result_root/queue-lock-95pct-groups3000.json"; do
  [[ -f "$queue_path" ]] && summary_queue_args+=(--queue-lock "$queue_path")
done
"${INFRASWE_LOCAL_PYTHON:-/venv/main/bin/python}" \
  benchmarks/historical_prs/summarize_training_bulk_campaign.py \
  --result-root "$result_root" \
  --expected-cases "$expected_cases" \
  "${summary_queue_args[@]}" \
  --output "$result_root/campaign-summary.json"

target_metric_improved="$(
  jq -r '.target_metric_improved' "$result_root/campaign-summary.json"
)"
release_quality_gate_satisfied="$(
  jq -r '.release_quality_gate_satisfied' "$result_root/campaign-summary.json"
)"

jq -n \
  --arg status complete \
  --argjson target_metric_improved "$target_metric_improved" \
  --argjson release_quality_gate_satisfied "$release_quality_gate_satisfied" \
  --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{status:$status,target_metric_improved:$target_metric_improved,release_quality_gate_satisfied:$release_quality_gate_satisfied,completed_at:$completed_at}' \
  > "$progress_file.tmp"
mv "$progress_file.tmp" "$progress_file"

# Campaign completion is not release qualification or evidence synchronization.
# Deliberately fail closed: this worker has no independent publication attestor.
# A separately verified operator workflow must check all three campaigns, the
# 95/99/99 contract, full SHA-256 synchronization, tests and the exact release.
echo "campaign complete; finalization pending independent verification"
echo "automatic commit/push, credential deletion and instance stop are disabled"
