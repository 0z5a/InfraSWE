#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GROUP_INDEX" >&2
  exit 2
fi

group_index="$1"
printf -v group_name 'group-%04d' "${group_index}"
printf -v previous_group_name 'group-%04d' "$((group_index - 1))"

result_root="${INFRASWE_INFERENCE_RESULT_ROOT:-results/historical-pr-blind-20260901/inference-bulk-95pct}"
queue_lock="${INFRASWE_INFERENCE_QUEUE_LOCK:-${result_root}/queue-lock-groups3000.json}"
group_dir="${result_root}/groups/${group_name}"
if [[ "${group_index}" -eq 0 ]]; then
  previous_policy="${result_root}/seed-policy.json"
else
  previous_policy="${result_root}/groups/${previous_group_name}/next-policy.json"
fi
python="${INFRASWE_LOCAL_PYTHON:-.venv/bin/python}"
workers="${INFRASWE_INFERENCE_WORKERS:-8}"
lanes_per_project="${INFRASWE_INFERENCE_LANES_PER_PROJECT:-2}"
github_workers="${INFRASWE_GITHUB_WORKERS:-1}"
github_reveal_workers="${INFRASWE_GITHUB_REVEAL_WORKERS:-${github_workers}}"
github_batch_size="${INFRASWE_GITHUB_BATCH_SIZE:-1}"
github_reveal_batch_size="${INFRASWE_GITHUB_REVEAL_BATCH_SIZE:-${github_batch_size}}"
test_timeout="${INFRASWE_TEST_TIMEOUT:-60}"
execution_mode="${INFRASWE_EXECUTION_MODE:-local}"

runner_transport_args=()
if [[ "${execution_mode}" == local ]]; then
  runner_transport_args+=(--local)
elif [[ "${execution_mode}" == ssh ]]; then
  : "${INFRASWE_SSH_REMOTE:?INFRASWE_SSH_REMOTE is required}"
  runner_transport_args+=(--remote "${INFRASWE_SSH_REMOTE}")
else
  echo "unsupported INFRASWE_EXECUTION_MODE=${execution_mode}" >&2
  exit 2
fi

export PYTHONPATH="src:benchmarks/historical_prs${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "${group_dir}"

if [[ ! -f "${group_dir}/input-lock.json" ]]; then
  "${python}" benchmarks/historical_prs/acquire_training_bulk_group.py \
    --queue-lock "${queue_lock}" \
    --group-index "${group_index}" \
    --workers "${github_workers}" \
    --batch-size "${github_batch_size}" \
    --output "${group_dir}/input-lock.json"
fi

group_case_count="$(jq -r '.case_count' "${group_dir}/input-lock.json")"
if [[ "${group_case_count}" -ge 1000 && "${lanes_per_project}" -lt 16 ]]; then
  lanes_per_project=16
fi

if [[ ! -f "${group_dir}/exact-head-evidence.json" ]]; then
  "${python}" benchmarks/historical_prs/run_training_bulk_group.py \
    --input-lock "${group_dir}/input-lock.json" \
    --output "${group_dir}/exact-head-evidence.json" \
    --test-timeout "${test_timeout}" \
    --output-tail-bytes 2000 \
    --workers "${workers}" \
    --lanes-per-project "${lanes_per_project}" \
    "${runner_transport_args[@]}"
fi

rerun_args=()
while IFS= read -r case_id; do
  [[ -n "${case_id}" ]] && rerun_args+=(--only "${case_id}")
done < <(
  jq -r '.records[] | select(.returncode == 255 or (.status | IN("transport_timeout", "checkout_failed", "checkout_timeout"))) | .case_id' \
    "${group_dir}/exact-head-evidence.json"
)

if [[ ${#rerun_args[@]} -gt 0 && ! -f "${group_dir}/exact-head-infra-rerun.json" ]]; then
  "${python}" benchmarks/historical_prs/run_training_bulk_group.py \
    --input-lock "${group_dir}/input-lock.json" \
    --output "${group_dir}/exact-head-infra-rerun.json" \
    --test-timeout "${test_timeout}" \
    --output-tail-bytes 2000 \
    --workers "${workers}" \
    --lanes-per-project "${lanes_per_project}" \
    "${runner_transport_args[@]}" \
    "${rerun_args[@]}"
fi

evidence_args=(--evidence "${group_dir}/exact-head-evidence.json")
if [[ -f "${group_dir}/exact-head-infra-rerun.json" ]]; then
  evidence_args+=(--evidence "${group_dir}/exact-head-infra-rerun.json")
fi

if [[ ! -f "${group_dir}/judgment-locks.json" ]]; then
  "${python}" benchmarks/historical_prs/freeze_training_bulk_group.py \
    --input-lock "${group_dir}/input-lock.json" \
    "${evidence_args[@]}" \
    --policy "${previous_policy}" \
    --output "${group_dir}/judgment-locks.json"
fi

if [[ ! -f "${group_dir}/outcome-reveal.json" ]]; then
  "${python}" benchmarks/historical_prs/reveal_training_bulk_group.py \
    --input-lock "${group_dir}/input-lock.json" \
    --judgment-locks "${group_dir}/judgment-locks.json" \
    --workers "${github_reveal_workers}" \
    --batch-size "${github_reveal_batch_size}" \
    --output "${group_dir}/outcome-reveal.json"
fi

if [[ ! -f "${group_dir}/oracle-audit.json" ]]; then
  "${python}" benchmarks/historical_prs/audit_training_bulk_group.py \
    --judgment-locks "${group_dir}/judgment-locks.json" \
    --reveal "${group_dir}/outcome-reveal.json" \
    --output "${group_dir}/oracle-audit.json"
fi

if [[ ! -f "${group_dir}/next-policy.json" ]]; then
  "${python}" benchmarks/historical_prs/derive_training_bulk_policy_iteration.py \
    --input-lock "${group_dir}/input-lock.json" \
    --judgment-locks "${group_dir}/judgment-locks.json" \
    --reveal "${group_dir}/outcome-reveal.json" \
    --audit "${group_dir}/oracle-audit.json" \
    --output "${group_dir}/next-policy.json"
fi

jq '{group_index,summary}' "${group_dir}/oracle-audit.json"
