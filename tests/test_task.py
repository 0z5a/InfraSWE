from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infraswe.models.task import TaskPackage


def kernel_task_payload() -> dict:
    return {
        "schema_version": "0.3",
        "task": {
            "id": "fused-rmsnorm-sm80",
            "title": "Fused RMSNorm",
            "track": "kernel",
            "repository": "fixture",
            "base_commit": "abc",
            "level": "K1",
            "kind": "kernel-micro",
        },
        "environment": {
            "profile": "gpu-1x-sm80",
            "gpu_count": 1,
            "exclusive_gpu_lease": True,
            "mps": "disabled",
        },
        "replay": {"count": 3, "require_all": True},
        "scoring": {
            "kernel": {
                "benchmark_cell_id": "rmsnorm-sm80@test-v1/formula-v03/anchor-v1/env-v1",
                "leaderboard_season": "2026q3-kernel-v1",
                "formula_version": "kernel-artifact-v0.3-micro",
                "formula_parameters_sha256": "sha256:formula",
                "profile": "kernel-micro",
                "role_graph": {"path": "role-graph.json", "sha256": "sha256:graph"},
                "role_requirements": {
                    "certification_roles": ["correctness-public"],
                    "artifact_roles": ["anchor-scorer"],
                    "fresh_replays": 3,
                    "required_passes": 3,
                },
                "performance": {
                    "scoring_baseline_sha256": "sha256:baseline",
                    "anchor_manifest_sha256": "sha256:anchor",
                    "sampling_plan_sha256": "sha256:sampling",
                },
            }
        },
        "kernel_contract": {
            "entrypoint": "candidate.py:run",
            "reference_entrypoint": "reference.py:run",
            "target_arch": ["sm80"],
            "allowed_backends": ["cuda"],
            "artifact_surface": "device-kernel",
            "execution_scope": "single-device",
            "workload_semantics": "operator",
            "mechanism_policy": "strict-native",
            "measurement_domain": "device-time",
        },
    }


def test_kernel_v03_task_envelope_is_coherent() -> None:
    task = TaskPackage.model_validate(kernel_task_payload())
    assert task.task.level == "K1"
    assert task.scoring.kernel is not None
    assert task.scoring.kernel.profile == task.task.kind


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("environment", "exclusive_gpu_lease"), False, "exclusive GPU lease"),
        (("environment", "mps"), "enabled", "MPS disabled"),
        (("replay", "count"), 2, "fresh_replays"),
        (("scoring", "kernel", "profile"), "kernel-library", "match task.kind"),
    ],
)
def test_kernel_v03_task_rejects_identity_or_lease_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = kernel_task_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        TaskPackage.model_validate(payload)


def test_rollout_task_layout_is_complete(rollout_task: TaskPackage) -> None:
    assert rollout_task.validate_layout() == []
    assert rollout_task.replay.count == 3
    assert rollout_task.environment.verifier_mode == "separate"


def test_package_dir_is_not_part_of_public_schema() -> None:
    schema = TaskPackage.model_json_schema()
    assert "package_dir" not in schema.get("properties", {})


def test_nccl_oracle_is_conservative_for_missing_or_asymmetric_p2p(
    project_root: Path, tmp_path: Path
) -> None:
    task = TaskPackage.load(project_root / "tasks" / "nccl-topology-silent-fallback-2gpu")
    assert task.validate_layout() == []
    workspace = tmp_path / "repo"
    shutil.copytree(task.resolve(task.execution.repo), workspace)
    completed = subprocess.run(
        [sys.executable, str(task.resolve("solution/solve.py"))],
        cwd=workspace,
        check=False,
    )
    assert completed.returncode == 0
    spec = importlib.util.spec_from_file_location("nccl_oracle", workspace / "launch_policy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    direct = module.build_plan([[True, True], [True, True]])
    unavailable = module.build_plan([[True, False], [False, True]])
    asymmetric = module.build_plan([[True, True], [False, True]])
    malformed = module.build_plan([[True], [False, True]])

    assert direct["transport"] == "p2p" and not direct["fallback_reported"]
    for plan in (unavailable, asymmetric, malformed):
        assert plan["transport"] == "shm"
        assert plan["fallback_reported"]
        assert plan["nccl_env"]["NCCL_P2P_DISABLE"] == "1"


def test_kv_routing_oracle_is_stable_available_and_capacity_bounded(
    project_root: Path, tmp_path: Path
) -> None:
    task = TaskPackage.load(project_root / "tasks" / "kv-aware-routing-cache-collapse-2gpu")
    assert task.validate_layout() == []
    workspace = tmp_path / "repo"
    shutil.copytree(task.resolve(task.execution.repo), workspace)
    completed = subprocess.run(
        [sys.executable, str(task.resolve("solution/solve.py"))],
        cwd=workspace,
        check=False,
    )
    assert completed.returncode == 0
    spec = importlib.util.spec_from_file_location("kv_oracle", workspace / "router.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((workspace / "routing_config.json").read_text(encoding="utf-8"))
    workers = [
        {"cache": {}, "id": "gpu-0", "load": 0},
        {"cache": {}, "id": "gpu-1", "load": 0},
    ]
    request = {"prefix_id": "shared-prefix", "request_id": 1, "step": 20}

    owner = module.choose_worker(request, workers, [True, True], config)
    assert owner in {0, 1}
    assert module.choose_worker(request, workers, [True, True], config) == owner
    assert module.choose_worker(
        request, workers, [index != owner for index in range(2)], config
    ) == (1 - owner)
    assert config["strategy"] == "kv_affinity"
    assert 8 <= config["cache_ttl_steps"] <= 128
    assert config["cache_capacity_entries"] == 4

    with pytest.raises(ValueError):
        module.choose_worker({}, [], [], config)


def test_cuda_artifact_oracle_prefers_compatible_native_and_fails_closed(
    project_root: Path, tmp_path: Path
) -> None:
    task = TaskPackage.load(project_root / "tasks" / "cuda-artifact-capability-selection")
    assert task.validate_layout() == []
    workspace = tmp_path / "repo"
    shutil.copytree(task.resolve(task.execution.repo), workspace)
    completed = subprocess.run(
        [sys.executable, str(task.resolve("solution/solve.py"))],
        cwd=workspace,
        check=False,
    )
    assert completed.returncode == 0
    spec = importlib.util.spec_from_file_location("cuda_selector_oracle", workspace / "selector.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = json.loads((workspace / "build_policy.json").read_text(encoding="utf-8"))
    request = {"cxx11_abi": 1, "device_sm": 80, "driver_cuda": 126, "runtime_cuda": 126}
    artifacts = [
        {
            "built_cuda": 126,
            "cxx11_abi": 1,
            "id": "wrong-sm-first",
            "kind": "sass",
            "sms": [90],
        },
        {
            "built_cuda": 126,
            "cxx11_abi": 1,
            "id": "sm80-native",
            "kind": "sass",
            "sms": [80],
        },
    ]

    native = module.select_artifact(request, artifacts, policy)
    assert native["artifact_id"] == "sm80-native"
    assert native["mechanism"] == "native_sass"
    assert not native["fallback_reported"]

    incompatible = module.select_artifact({**request, "driver_cuda": 120}, artifacts, policy)
    assert incompatible["status"] == "blocked"
    assert incompatible["artifact_id"] is None
    assert incompatible["fallback_reported"]
    assert policy["allow_cpu_fallback"] is False


@pytest.mark.parametrize(
    "task_id",
    [
        "container-image-dependency-lock",
        "gpu-resource-request-contract",
        "health-probe-drain-regression",
        "telemetry-root-cause-correlation",
    ],
)
def test_cpu_batch_task_layouts_are_complete(project_root: Path, task_id: str) -> None:
    task = TaskPackage.load(project_root / "tasks" / task_id)
    assert task.validate_layout() == []
    assert task.environment.profile == "cpu-small"
    assert task.replay.count == 3


def _solved_workspace(project_root: Path, tmp_path: Path, task_id: str) -> tuple[TaskPackage, Path]:
    task = TaskPackage.load(project_root / "tasks" / task_id)
    workspace = tmp_path / task_id
    shutil.copytree(task.resolve(task.execution.repo), workspace)
    completed = subprocess.run(
        [sys.executable, str(task.resolve("solution/solve.py"))],
        cwd=workspace,
        check=False,
    )
    assert completed.returncode == 0
    return task, workspace


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_lock_oracle_rejects_mutable_first_candidate(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "container-image-dependency-lock")
    module = _module(workspace / "resolver.py", "image_lock_oracle")
    policy = json.loads((workspace / "lock_policy.json").read_text(encoding="utf-8"))
    request = {"architecture": "amd64", "name": "server", "version": "1.0"}
    candidates = [
        {
            "architecture": "amd64",
            "dependencies": [],
            "digest": "latest",
            "id": "bad-first",
            "name": "server",
            "version": "1.0",
        },
        {
            "architecture": "amd64",
            "dependencies": [],
            "digest": "sha256:" + "a" * 64,
            "id": "locked",
            "name": "server",
            "version": "1.0",
        },
    ]
    plan = module.resolve_image(request, candidates, policy)
    assert plan["status"] == "ready"
    assert plan["candidate_id"] == "locked"
    assert all(entry["digest"].startswith("sha256:") for entry in plan["lock"])


def test_gpu_admission_oracle_rejects_missing_request(project_root: Path, tmp_path: Path) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "gpu-resource-request-contract")
    module = _module(workspace / "admission.py", "gpu_admission_oracle")
    workload = json.loads((workspace / "workload.json").read_text(encoding="utf-8"))
    node = {
        "allocatable_gpus": 2,
        "compute_capability": "8.0",
        "runtime_classes": ["nvidia"],
    }
    policy = {
        "forbid_cpu_fallback": True,
        "require_capability_selector": True,
        "require_equal_requests_limits": True,
        "required_runtime_class": "nvidia",
    }
    assert module.admit_workload(workload, node, policy)["status"] == "admitted"
    missing = copy.deepcopy(workload)
    missing["resources"]["requests"] = {}
    rejected = module.admit_workload(missing, node, policy)
    assert rejected["status"] == "rejected" and rejected["fallback_reported"]


def test_health_drain_oracle_preserves_capacity_and_grace(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "health-probe-drain-regression")
    module = _module(workspace / "probe_policy.py", "health_drain_oracle")
    deployment = json.loads((workspace / "deployment.json").read_text(encoding="utf-8"))
    plan = module.build_rollout_plan(
        deployment, {"max_inflight_seconds": 3, "readiness_flap_seconds": 2}
    )
    assert plan["readiness_path"] == "/readyz"
    assert plan["drain_path"] == "/drainz"
    assert plan["max_unavailable"] == 0
    assert plan["termination_grace_seconds"] > plan["pre_stop_seconds"] >= 3


def test_telemetry_oracle_uses_three_modalities(project_root: Path, tmp_path: Path) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "telemetry-root-cause-correlation")
    module = _module(workspace / "diagnoser.py", "telemetry_oracle")
    policy = json.loads((workspace / "signal_policy.json").read_text(encoding="utf-8"))
    signals = {"logs": [], "metrics": [], "profiles": [], "traces": []}
    for offset, modality in enumerate(("logs", "metrics", "traces")):
        signals[modality].append(
            {
                "at_ms": 1000 + offset * 20,
                "cause": "queue_saturation",
                "correlation_id": "request-1",
                "id": f"evidence-{modality}",
            }
        )
    result = module.diagnose(signals, policy)
    assert result["status"] == "diagnosed"
    assert result["root_cause"] == "queue_saturation"
    assert result["confidence"] == 0.75


@pytest.mark.parametrize(
    "task_id",
    [
        "cuda-extension-arch-target",
        "dynamic-batching-slo-collapse",
        "numa-cpu-affinity-regression",
        "gpu-oom-worker-recovery",
    ],
)
def test_one_gpu_batch_task_layouts_are_complete(project_root: Path, task_id: str) -> None:
    task = TaskPackage.load(project_root / "tasks" / task_id)
    assert task.validate_layout() == []
    assert task.environment.profile == "gpu-1x-sm80"
    assert task.environment.gpu_count == 1
    assert task.replay.count == 3


def test_cuda_arch_oracle_targets_every_visible_supported_sm(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "cuda-extension-arch-target")
    module = _module(workspace / "arch_policy.py", "cuda_arch_oracle")
    config = json.loads((workspace / "build_config.json").read_text(encoding="utf-8"))
    toolkit = {"supported_sms": [75, 80, 86, 90]}
    ready = module.select_targets([90, 80, 80], toolkit, config)
    assert ready["status"] == "ready"
    assert ready["sass_targets"] == [80, 90]
    assert ready["ptx_target"] is None
    blocked = module.select_targets([80, 100], toolkit, config)
    assert blocked["status"] == "blocked" and blocked["fallback_reported"]


def test_dynamic_batching_oracle_is_order_independent_and_model_isolated(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "dynamic-batching-slo-collapse")
    module = _module(workspace / "batch_policy.py", "batching_oracle")
    config = json.loads((workspace / "serving_config.json").read_text(encoding="utf-8"))
    requests = [
        {
            "arrival_ms": index,
            "deadline_ms": 20 + index,
            "id": f"request-{index}",
            "model": "a" if index % 2 == 0 else "b",
            "tokens": 128,
        }
        for index in range(4)
    ]
    schedule = module.schedule_batches(requests, config)
    assert schedule == module.schedule_batches(list(reversed(requests)), config)
    by_id = {request["id"]: request for request in requests}
    assert {identifier for batch in schedule for identifier in batch} == set(by_id)
    assert all(len({by_id[identifier]["model"] for identifier in batch}) == 1 for batch in schedule)


def test_numa_affinity_oracle_intersects_local_allowed_and_non_reserved_cpus(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "numa-cpu-affinity-regression")
    module = _module(workspace / "affinity_policy.py", "numa_affinity_oracle")
    config = json.loads((workspace / "affinity_config.json").read_text(encoding="utf-8"))
    topology = {
        "allowed_cpus": [1, 2, 3, 4, 5, 8],
        "nodes": {"0": [8, 9], "1": [0, 1, 2, 3, 4, 5]},
        "reserved_cpus": [1],
    }
    ready = module.select_affinity({"numa_node": 1}, topology, config)
    assert ready["status"] == "ready"
    assert ready["cpu_ids"] == [2, 3, 4, 5]
    blocked = module.select_affinity({"numa_node": 0}, {**topology, "allowed_cpus": [8]}, config)
    assert blocked["status"] == "blocked" and blocked["fallback_reported"]


def test_gpu_oom_oracle_retries_smaller_then_quarantines(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "gpu-oom-worker-recovery")
    module = _module(workspace / "recovery.py", "gpu_oom_oracle")
    config = json.loads((workspace / "recovery_config.json").read_text(encoding="utf-8"))
    event = {
        "batch_size": 8,
        "failed_request_ids": ["request-7", "request-2"],
        "kind": "cuda_oom",
        "worker_id": "worker-0",
    }
    state = {
        "oom_count_by_worker": {"worker-0": 1},
        "workers": [
            {"id": "worker-0", "status": "failed"},
            {"id": "worker-2", "status": "healthy"},
            {"id": "worker-1", "status": "healthy"},
        ],
    }
    first = module.plan_recovery(event, state, config)
    assert first["status"] == "retry"
    assert first["retry_batch_size"] == 4
    assert first["restarted_worker_ids"] == ["worker-0"]
    assert first["preserve_worker_ids"] == ["worker-1", "worker-2"]
    repeated = module.plan_recovery(
        event,
        {**state, "oom_count_by_worker": {"worker-0": 2}},
        config,
    )
    assert repeated["status"] == "quarantine"
    assert repeated["retry_worker_id"] == "worker-1"
    assert repeated["quarantined_worker_ids"] == ["worker-0"]
    assert repeated["retry_request_ids"] == ["request-2", "request-7"]


@pytest.mark.parametrize(
    "task_id",
    [
        "tensor-parallel-shard-contract",
        "collective-order-rank-divergence",
        "collective-compute-overlap-regression",
        "rank-exit-collective-recovery",
    ],
)
def test_two_gpu_batch_task_layouts_are_complete(project_root: Path, task_id: str) -> None:
    task = TaskPackage.load(project_root / "tasks" / task_id)
    assert task.validate_layout() == []
    assert task.environment.profile == "gpu-2x-sm120-pcie"
    assert task.environment.gpu_count == 2
    assert task.replay.count == 3


def test_tp_shard_oracle_covers_row_column_and_replicated_parameters(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "tensor-parallel-shard-contract")
    module = _module(workspace / "shard_policy.py", "tp_shard_oracle")
    config = json.loads((workspace / "tp_config.json").read_text(encoding="utf-8"))
    parameters = [
        {"name": "attention.qkv.weight", "shape": [24, 8]},
        {"name": "attention.out.weight", "shape": [8, 8]},
        {"name": "mlp.gate_up.weight", "shape": [32, 8]},
        {"name": "norm.weight", "shape": [8]},
    ]
    plan = module.build_shard_plan(list(reversed(parameters)), 2, config)
    entries = {entry["name"]: entry for entry in plan["parameters"]}
    assert [entry["name"] for entry in plan["parameters"]] == sorted(entries)
    assert entries["attention.qkv.weight"]["axis"] == 0
    assert entries["attention.out.weight"]["axis"] == 1
    assert entries["norm.weight"]["replicated"] is True
    assert entries["norm.weight"]["shards"] == [
        {"end": 8, "rank": 0, "start": 0},
        {"end": 8, "rank": 1, "start": 0},
    ]
    odd = copy.deepcopy(parameters)
    odd[0]["shape"][0] = 25
    with pytest.raises(ValueError):
        module.build_shard_plan(odd, 2, config)


def test_collective_order_oracle_canonicalizes_and_blocks_metadata_mismatch(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "collective-order-rank-divergence")
    module = _module(workspace / "collective_policy.py", "collective_order_oracle")
    config = json.loads((workspace / "order_config.json").read_text(encoding="utf-8"))
    steps = [
        [
            {"elements": 1024, "id": "reduce-a", "kind": "all_reduce"},
            {"elements": 2048, "id": "reduce-b", "kind": "all_reduce"},
            {"elements": 4096, "id": "reduce-c", "kind": "all_reduce"},
        ],
        [
            {"elements": 4096, "id": "reduce-c", "kind": "all_reduce"},
            {"elements": 1024, "id": "reduce-a", "kind": "all_reduce"},
            {"elements": 2048, "id": "reduce-b", "kind": "all_reduce"},
        ],
    ]
    plan = module.build_collective_schedule(steps, 2, config)
    assert plan["divergence_detected"] is True
    assert plan["rank_schedules"] == {
        "0": config["canonical_order"],
        "1": config["canonical_order"],
    }
    mismatch = copy.deepcopy(steps)
    mismatch[1][2]["elements"] += 1
    blocked = module.build_collective_schedule(mismatch, 2, config)
    assert blocked["status"] == "blocked" and blocked["fallback_reported"]


def test_overlap_oracle_requires_async_event_fenced_comm_stream(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(
        project_root, tmp_path, "collective-compute-overlap-regression"
    )
    module = _module(workspace / "overlap_policy.py", "overlap_oracle")
    config = json.loads((workspace / "overlap_config.json").read_text(encoding="utf-8"))
    stages = [
        {
            "collective_elements": 1024,
            "compute_cycles": 1000,
            "id": f"stage-{index}",
            "sequence": index,
        }
        for index in range(3)
    ]
    topology = {"concurrent_kernels": True, "device_count": 2}
    plan = module.build_overlap_plan(list(reversed(stages)), topology, config)
    assert plan["comm_stream"] and plan["async_collectives"] and plan["event_fencing"]
    assert [stage["sequence"] for stage in plan["stages"]] == [0, 1, 2]
    assert all(stage["wait_with_event"] for stage in plan["stages"])
    blocked = module.build_overlap_plan(
        stages, {"concurrent_kernels": False, "device_count": 2}, config
    )
    assert blocked["status"] == "blocked" and blocked["fallback_reported"]


def test_rank_exit_oracle_restarts_full_group_then_exhausts_budget(
    project_root: Path, tmp_path: Path
) -> None:
    _, workspace = _solved_workspace(project_root, tmp_path, "rank-exit-collective-recovery")
    module = _module(workspace / "failure_policy.py", "rank_exit_oracle")
    config = json.loads((workspace / "recovery_config.json").read_text(encoding="utf-8"))
    event = {
        "failed_operation": "all_reduce",
        "failed_rank": 1,
        "kind": "rank_exit",
        "observed_step": 41,
    }
    state = {
        "last_committed_step": 40,
        "pending_request_ids": ["request-7", "request-2"],
        "ranks": [
            {"rank": 0, "status": "running"},
            {"rank": 1, "status": "exited"},
        ],
        "restart_count": 0,
        "world_size": 2,
    }
    first = module.plan_rank_failure(event, state, config)
    assert first["status"] == "restart_group"
    assert first["abort_ranks"] == first["restart_ranks"] == [0, 1]
    assert first["resume_step"] == 40
    assert first["replay_request_ids"] == ["request-2", "request-7"]
    exhausted = module.plan_rank_failure(event, {**state, "restart_count": 1}, config)
    assert exhausted["status"] == "abort"
    assert exhausted["restart_ranks"] == []
    assert exhausted["fallback_reported"] is True
