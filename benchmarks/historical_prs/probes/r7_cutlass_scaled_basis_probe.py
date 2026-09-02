#!/usr/bin/env python3
"""Compile the exact extracted ScaledBasis equality operator for CUTLASS R7."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _extract_operator(source: str) -> str:
    start = source.index("// Equality")
    end = source.index("// Not equal to anything else", start)
    block = source[start:end]
    if block.count("operator==") != 1:
        raise ValueError("unexpected ScaledBasis equality extraction")
    return block


def _translation_unit(operator_source: str) -> str:
    return f"""
#include <type_traits>

#define CUTE_HOST_DEVICE
#define CUTE_GCC_UNREACHABLE __builtin_unreachable()

namespace cute {{
using std::bool_constant;
using std::false_type;

template <class T, int... Ns>
struct ScaledBasis {{
  T stored;
  constexpr ScaledBasis(T const& value = {{}}) : stored(value) {{}}
  constexpr T const& value() const {{ return stored; }}
}};

{operator_source}
}}  // namespace cute

struct LeftValue {{ int value; }};
struct RightValue {{ int value; }};
constexpr bool operator==(LeftValue lhs, LeftValue rhs) {{ return lhs.value == rhs.value; }}
constexpr bool operator==(RightValue lhs, RightValue rhs) {{ return lhs.value == rhs.value; }}

using LeftBasis0 = cute::ScaledBasis<LeftValue, 0>;
using LeftBasis1 = cute::ScaledBasis<LeftValue, 1>;
using RightBasis1 = cute::ScaledBasis<RightValue, 1>;
constexpr LeftBasis0 left_one{{LeftValue{{1}}}};
constexpr LeftBasis0 left_one_again{{LeftValue{{1}}}};
constexpr LeftBasis0 left_two{{LeftValue{{2}}}};
constexpr LeftBasis1 other_basis_same_type{{LeftValue{{1}}}};
constexpr RightBasis1 other_basis_uncomparable_type{{RightValue{{1}}}};

static_assert(left_one == left_one_again);
static_assert(!(left_one == left_two));
static_assert(!(left_one == other_basis_same_type));
static_assert(!(left_one == other_basis_uncomparable_type));

int main() {{ return left_one == left_one_again ? 0 : 1; }}
"""


def _compile(compiler: str, source: str, root: Path, name: str) -> dict[str, Any]:
    source_path = root / f"{name}.cpp"
    binary_path = root / name
    source_path.write_text(source, encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(
        [compiler, "-std=c++17", str(source_path), "-o", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    run_code = None
    if completed.returncode == 0:
        run_code = subprocess.run([str(binary_path)], check=False).returncode
    return {
        "return_code": completed.returncode,
        "run_return_code": run_code,
        "duration_seconds": time.perf_counter() - started,
        "stdout_sha256": canonical_sha256(completed.stdout),
        "stderr_sha256": canonical_sha256(completed.stderr),
        "stderr_tail": completed.stderr[-4000:],
        "translation_unit_sha256": canonical_sha256(source),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-header", type=Path, required=True)
    parser.add_argument("--head-header", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    selection_sha = selection["selection_lock_sha256"]
    plan_sha = plan["test_plan_sha256"]
    if canonical_sha256(selection["selection_material"]) != selection_sha:
        raise ValueError("selection lock mismatch")
    plan_material = dict(plan)
    plan_material.pop("test_plan_sha256")
    if canonical_sha256(plan_material) != plan_sha:
        raise ValueError("test plan mismatch")

    case = next(
        item
        for item in selection["selection_material"]["cases"]
        if item["case_id"] == "cutlass-pr-3427"
    )
    base_source = args.base_header.read_text(encoding="utf-8")
    head_source = args.head_header.read_text(encoding="utf-8")
    base_operator = _extract_operator(base_source)
    head_operator = _extract_operator(head_source)
    with tempfile.TemporaryDirectory(prefix="infraswe-r7-cutlass-") as temp:
        root = Path(temp)
        base = _compile(args.compiler, _translation_unit(base_operator), root, "base")
        head = _compile(args.compiler, _translation_unit(head_operator), root, "head")

    facts = {
        "base": base,
        "head": head,
        "base_reproduces_uncomparable_value_compile_failure": (
            base["return_code"] != 0
            and any(
                token in base["stderr_tail"]
                for token in ("invalid operands", "invalid argument type", "operator==")
            )
        ),
        "head_compiles_and_executes": (head["return_code"] == 0 and head["run_return_code"] == 0),
        "equal_basis_behavior_retained": head["run_return_code"] == 0,
        "exact_operator_extraction": {
            "base_sha256": canonical_sha256(base_operator),
            "head_sha256": canonical_sha256(head_operator),
            "base_header_sha256": canonical_sha256(base_source),
            "head_header_sha256": canonical_sha256(head_source),
        },
        "sm120_metadata_aot": "reported-by-separate-exact-sm120-aot-probe",
        "steady_state_compile_seconds": 0.0,
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "probe": "r7-cutlass-scaled-basis-exact-extraction-v1",
        "case_id": "cutlass-pr-3427",
        "project": "cutlass-cute",
        "status": "pass" if facts["head_compiles_and_executes"] else "fail",
        "failure_codes": [] if facts["head_compiles_and_executes"] else ["HEAD_COMPILE_FAILED"],
        "facts": facts,
        "base_sha": case["base_sha"],
        "head_sha": case["head_sha"],
        "selection_lock_sha256": selection_sha,
        "test_plan_sha256": plan_sha,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
