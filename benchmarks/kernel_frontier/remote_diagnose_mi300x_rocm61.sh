#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-mi300x-rocm61-torch240-diagnostic}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
attention_backend="torch-sdpa-aotriton"
classic_backend="triton-gfx942-initial"
export PATH="/opt/rocm/bin:${PATH}"

if [[ "${INFRASWE_DIAGNOSTIC_ONLY:-0}" != "1" ]]; then
  echo "set INFRASWE_DIAGNOSTIC_ONLY=1; this runner emits non-scoring evidence only" >&2
  exit 64
fi

mkdir -p "${run_root}/lease" "${run_root}/diagnostic"
cd "${benchmark_root}"

rocm-smi > "${run_root}/rocm-smi-before.txt"
pre_lease_status=0
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${run_root}/lease/exclusive-device.json" || pre_lease_status=$?

export ROCR_VISIBLE_DEVICES="${gpu}"
export HIP_VISIBLE_DEVICES="${gpu}"

PYTHONPATH="${workspace}/src" "${python}" -m infraswe lease preflight \
  --profile gpu-1x-gfx942-mi300x-rocm61 \
  --profiles-dir "${workspace}/profiles" \
  --output "${run_root}/hardware-manifest.json"

"${python}" attention_bench.py \
  --backend "${attention_backend}" \
  --output "${run_root}/diagnostic/attention.json" \
  --replay-index 1 \
  --blocks 1 \
  --min-timed-span-ms 1 \
  --implementation-commit pytorch-2.4.0-rocm6.1-embedded-aotriton

"${python}" classic_bench.py \
  --backend "${classic_backend}" \
  --implementation-commit infraswe-gfx942-portable-fixed-v1 \
  --output "${run_root}/diagnostic/classic.json" \
  --replay-index 1 \
  --blocks 1 \
  --min-timed-span-ms 1

"${python}" collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --platform-only
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
rocm-smi > "${run_root}/rocm-smi-after.txt"

post_lease_status=0
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${run_root}/lease/exclusive-device-after.json" || post_lease_status=$?

"${python}" - \
  "${run_root}" \
  "${pre_lease_status}" \
  "${post_lease_status}" <<'PY'
import json
import sys
from pathlib import Path

from bench_utils import atomic_write_json, utc_now

run_root = Path(sys.argv[1])
pre_lease_status = int(sys.argv[2])
post_lease_status = int(sys.argv[3])
attention = json.loads((run_root / "diagnostic/attention.json").read_text(encoding="utf-8"))
classic = json.loads((run_root / "diagnostic/classic.json").read_text(encoding="utf-8"))

native_tokens = ("aotriton", "attn_fwd", "fmha_fwd", "flash")
attention_native_cases = []
for case in attention.get("cases", []):
    names = [
        str(event.get("name", "")).lower()
        for event in case.get("profiler", {}).get("device_events", [])
    ]
    if any(
        not name.startswith("aten::")
        and any(token in name for token in native_tokens)
        for name in names
    ):
        attention_native_cases.append(case["case_id"])

classic_native_tokens = {
    "vector-add-bf16-16m": "_vector_add_kernel",
    "softmax-bf16-4096x4096": "_softmax_kernel",
    "layernorm-bf16-4096x4096": "_layernorm_kernel",
    "rmsnorm-bf16-4096x4096": "_rmsnorm_kernel",
    "swiglu-bf16-8192x4096": "_swiglu_kernel",
    "rope-bf16-b4-s2048-h16-d128": "_rope_kernel",
    "gemm-bf16-4096-cube": "_matmul_kernel",
}
classic_native_cases = []
for case in classic.get("cases", []):
    expected = classic_native_tokens.get(case["case_id"])
    names = [
        str(event.get("name", "")).lower()
        for event in case.get("profiler", {}).get("device_events", [])
    ]
    if expected and any(expected in name for name in names):
        classic_native_cases.append(case["case_id"])
functional_passed = bool(
    attention.get("status") == "passed"
    and attention.get("all_correct")
    and attention.get("case_count") == 5
    and len(attention_native_cases) == 5
    and classic.get("status") == "passed"
    and classic.get("all_correct")
    and classic.get("case_count") == 7
    and len(classic_native_cases) == 7
)
failure_codes = []
failure_codes.extend(
    sorted(
        {
            (
                "LIVENESS_DEVICE_NOT_EXCLUSIVE"
                if status == 3
                else "LIVENESS_EXCLUSIVE_LEASE_UNVERIFIABLE"
            )
            for status in (pre_lease_status, post_lease_status)
            if status != 0
        }
    )
)
if not functional_passed:
    failure_codes.append("MI300X_ROCM61_DIAGNOSTIC_INCOMPLETE")

atomic_write_json(
    run_root / "diagnostic-qualification.json",
    {
        "schema_version": "0.3",
        "evidence_kind": "mi300x-rocm61-full-adapter-diagnostic",
        "generated_at": utc_now(),
        "mode": "diagnostic-only",
        "status": "passed" if functional_passed else "failed",
        "hardware_profile_passed": True,
        "attention": {
            "case_count": attention.get("case_count"),
            "all_correct": attention.get("all_correct"),
            "native_case_ids": attention_native_cases,
        },
        "classic": {
            "case_count": classic.get("case_count"),
            "all_correct": classic.get("all_correct"),
            "native_case_ids": classic_native_cases,
        },
        "exclusive_device_precheck": pre_lease_status == 0,
        "exclusive_device_postcheck": post_lease_status == 0,
        "official_measurement_eligible": False,
        "timing_authority": "non-authoritative-diagnostic",
        "score_generated": False,
        "failure_codes": failure_codes,
    },
)
if not functional_passed:
    raise SystemExit(1)
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
"${python}" - "${run_root}" <<'PY'
import sys
from pathlib import Path

from score_results import write_manifest, write_zip

run_root = Path(sys.argv[1]).resolve()
write_manifest(run_root)
write_zip(run_root, run_root.with_suffix(".zip"))
PY
echo "MI300X ROCm 6.1 full adapter diagnostic complete (non-scoring): ${run_root}.zip"
