from __future__ import annotations

import runpy
from pathlib import Path


def test_rocm_lease_guard_extracts_nested_process_ids() -> None:
    module = runpy.run_path(
        Path(__file__).parents[1] / "benchmarks/kernel_frontier/rocm_lease_guard.py"
    )

    pids = module["_pids_from_json"](
        {
            "card0": {
                "KFD Processes": [
                    {"PID": 1234, "name": "python"},
                    {"process_pid": "5678", "name": "worker"},
                ]
            }
        }
    )

    assert pids == {1234, 5678}


def test_rocm_lease_guard_extracts_pid_encoded_in_rocm_smi_key() -> None:
    module = runpy.run_path(
        Path(__file__).parents[1] / "benchmarks/kernel_frontier/rocm_lease_guard.py"
    )

    pids = module["_pids_from_json"](
        {
            "system": {
                "PID300235": "unknown, 0, 0, 0, 10757545984",
                "PID300236": "unknown, 0, 0, 0, 0",
            }
        }
    )

    assert pids == {300235, 300236}
