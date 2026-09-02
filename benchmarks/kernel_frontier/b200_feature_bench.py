# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any


def _early_option(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


# CuTe DSL reads these controls while importing cutlass.  Resolve --dump-root
# before that import so a standalone invocation retains the requested IR/PTX/
# cubin artifacts without relying on shell-side environment setup.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
os.environ.setdefault("CUTE_DSL_KEEP", "ir,ptx,cubin")
if _early_dump_root := _early_option("--dump-root"):
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", _early_dump_root)

import cutlass
import torch


DEFAULT_CUTE_ROOT = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/"
    "examples/python/CuTeDSL/blackwell"
)
CUDA_BINARY_ROOT = Path("/usr/local/cuda-13.3/bin")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(argv: list[str], *, timeout_seconds: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        error = None
    except (OSError, subprocess.TimeoutExpired) as exception:
        returncode = 127
        stdout = ""
        stderr = f"{type(exception).__name__}: {exception}\n"
        error = stderr.strip()
    return {
        "argv": argv,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
    }


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CuTe DSL example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hardware_manifest() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    smi = command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,"
            "clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    return {
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_smi_query": smi,
    }


def base_config() -> dict[str, Any]:
    return {
        "ab_dtype": cutlass.Float16,
        "c_dtype": cutlass.Float16,
        "acc_dtype": cutlass.Float32,
        "a_major": "k",
        "b_major": "k",
        "c_major": "n",
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "use_2cta_instrs": False,
        "use_tma_store": True,
    }


def tma2_config() -> dict[str, Any]:
    config = base_config()
    config.update(
        {
            "mma_tiler_mn": (256, 128),
            "cluster_shape_mn": (2, 1),
            "use_2cta_instrs": True,
        }
    )
    return config


def run_correctness_case(
    module: ModuleType,
    *,
    label: str,
    shape: tuple[int, int, int, int],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    try:
        return_value = module.run(
            mnkl=shape,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=False,
            benchmark=False,
            **config,
        )
        torch.cuda.synchronize()
        return {
            "label": label,
            "shape": list(shape),
            "layout": {
                "a_major": config["a_major"],
                "b_major": config["b_major"],
                "c_major": config["c_major"],
            },
            "seed": seed,
            "status": "passed",
            "passed": return_value == 0,
            "return_value": return_value,
            "wall_seconds": time.perf_counter() - started,
        }
    except Exception as exception:
        return {
            "label": label,
            "shape": list(shape),
            "layout": {
                "a_major": config["a_major"],
                "b_major": config["b_major"],
                "c_major": config["c_major"],
            },
            "seed": seed,
            "status": "failed",
            "passed": False,
            "exception": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
        }


def run_expected_rejection(
    module: ModuleType,
    *,
    shape: tuple[int, int, int, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module.run(
            mnkl=shape,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=False,
            benchmark=False,
            **config,
        )
    except Exception as exception:
        name = type(exception).__name__
        return {
            "shape": list(shape),
            "passed": name == "CantImplementError",
            "exception_type": name,
            "exception": str(exception),
            "wall_seconds": time.perf_counter() - started,
        }
    return {
        "shape": list(shape),
        "passed": False,
        "exception_type": None,
        "exception": "invalid alignment was not rejected",
        "wall_seconds": time.perf_counter() - started,
    }


def run_cute_benchmark(
    module: ModuleType,
    *,
    label: str,
    shape: tuple[int, int, int, int],
    config: dict[str, Any],
    warmups: int,
    iterations: int,
    samples: int,
    seed_base: int,
) -> dict[str, Any]:
    values = []
    walls = []
    failures = []
    for sample_index in range(samples):
        seed = seed_base + sample_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        try:
            latency = module.run(
                mnkl=shape,
                warmup_iterations=warmups,
                iterations=iterations,
                skip_ref_check=True,
                benchmark=True,
                **config,
            )
            torch.cuda.synchronize()
            values.append(float(latency))
        except Exception as exception:
            failures.append(
                {
                    "sample_index": sample_index,
                    "exception": f"{type(exception).__name__}: {exception}",
                    "traceback": traceback.format_exc(),
                }
            )
        walls.append(time.perf_counter() - started)
    ordered = sorted(values)
    return {
        "label": label,
        "shape": list(shape),
        "warmup_iterations": warmups,
        "iterations": iterations,
        "sample_count": samples,
        "latency_us_samples": values,
        "latency_us_median": ordered[len(ordered) // 2] if ordered else None,
        "latency_us_min": min(ordered) if ordered else None,
        "latency_us_max": max(ordered) if ordered else None,
        "wall_seconds_samples": walls,
        "failures": failures,
        "passed": len(values) == samples,
    }


def benchmark_torch_bmm(
    *,
    shape: tuple[int, int, int, int],
    warmups: int,
    iterations: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    m, n, k, batch = shape
    torch.manual_seed(seed)
    a = torch.empty((batch, m, k), dtype=torch.float16, device="cuda").normal_()
    b = torch.empty((batch, k, n), dtype=torch.float16, device="cuda").normal_()
    c = torch.empty((batch, m, n), dtype=torch.float16, device="cuda")

    def launch() -> None:
        torch.bmm(a, b, out=c)

    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            launch()
        stop.record()
        stop.synchronize()
        values.append(float(start.elapsed_time(stop) * 1000.0 / iterations))
    ordered = sorted(values)
    return {
        "label": "torch-bmm-portable-baseline",
        "shape": list(shape),
        "warmup_iterations": warmups,
        "iterations": iterations,
        "sample_count": samples,
        "latency_us_samples": values,
        "latency_us_median": ordered[len(ordered) // 2],
        "latency_us_min": min(ordered),
        "latency_us_max": max(ordered),
        "passed": all(math.isfinite(value) and value > 0.0 for value in values),
    }


def profile_cute_launch(
    module: ModuleType,
    *,
    shape: tuple[int, int, int, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            module.run(
                mnkl=shape,
                warmup_iterations=0,
                iterations=1,
                skip_ref_check=True,
                benchmark=False,
                **config,
            )
            torch.cuda.synchronize()
        names = sorted({str(event.key) for event in profile.key_averages()})
        candidate_names = [
            name for name in names if re.search(r"cutlass|cute|bmm|gemm", name, flags=re.IGNORECASE)
        ]
        return {
            "captured": bool(names),
            "event_count": len(names),
            "kernel_names": candidate_names[:128],
            "all_event_names": names[:256],
        }
    except Exception as exception:
        return {
            "captured": False,
            "exception": f"{type(exception).__name__}: {exception}",
            "traceback": traceback.format_exc(),
        }


def snapshot_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*")):
        if not path.is_file() or path.suffix not in {".ptx", ".cubin", ".mlir"}:
            continue
        shutil.copy2(path, destination / path.name)


def copy_tma_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".ptx", ".cubin"} and path.name not in {"compile.log"}:
            continue
        relative = path.relative_to(source).as_posix().replace("/", "-")
        shutil.copy2(path, destination / relative)


def collect_native_evidence(
    artifact_root: Path,
    *,
    ptx_patterns: dict[str, str],
    sass_patterns: dict[str, str],
) -> dict[str, Any]:
    ptx_paths = sorted(artifact_root.glob("*.ptx"))
    cubin_paths = sorted(artifact_root.glob("*.cubin"))
    ptx_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in ptx_paths)
    sass_parts = []
    disassembly = []
    cuobjdump = CUDA_BINARY_ROOT / "cuobjdump"
    for cubin in cubin_paths:
        result = command([str(cuobjdump), "--dump-sass", str(cubin)])
        sass_path = cubin.with_suffix(cubin.suffix + ".sass.txt")
        sass_path.write_text(result["stdout"] + result["stderr"], encoding="utf-8")
        disassembly.append(
            {
                "cubin": cubin.name,
                "sass": sass_path.name,
                "returncode": result["returncode"],
            }
        )
        if result["returncode"] == 0:
            sass_parts.append(result["stdout"])
    sass_text = "\n".join(sass_parts)
    ptx_matches = {
        name: len(re.findall(pattern, ptx_text, flags=re.IGNORECASE))
        for name, pattern in ptx_patterns.items()
    }
    sass_matches = {
        name: len(re.findall(pattern, sass_text, flags=re.IGNORECASE))
        for name, pattern in sass_patterns.items()
    }
    versions = sorted(set(re.findall(r"(?m)^\s*\.version\s+([0-9.]+)", ptx_text)))
    targets = sorted(set(re.findall(r"(?m)^\s*\.target\s+(sm_[0-9]+[af]?)", ptx_text)))
    forbidden = {
        "wgmma_mma_async": len(re.findall(r"\bwgmma\.mma_async\b", ptx_text, re.I)),
        "cublas_symbol": len(re.findall(r"\bcublas(?:lt)?\b", ptx_text + sass_text, re.I)),
    }
    required_passed = bool(
        ptx_paths
        and cubin_paths
        and all(value > 0 for value in ptx_matches.values())
        and all(value > 0 for value in sass_matches.values())
        and not any(forbidden.values())
        and all(item["returncode"] == 0 for item in disassembly)
    )
    return {
        "passed": required_passed,
        "artifact_root": str(artifact_root),
        "ptx_files": [path.name for path in ptx_paths],
        "cubin_files": [path.name for path in cubin_paths],
        "ptx_versions": versions,
        "targets": targets,
        "ptx_matches": ptx_matches,
        "sass_matches": sass_matches,
        "forbidden_matches": forbidden,
        "disassembly": disassembly,
        "binary_size_bytes": sum(path.stat().st_size for path in cubin_paths),
        "ptx_size_bytes": sum(path.stat().st_size for path in ptx_paths),
    }


def run_tma_irregular(binary: Path, *, iterations: int) -> dict[str, Any]:
    result = command([str(binary), str(iterations)], timeout_seconds=180)
    parsed = None
    for line in result["stdout"].splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("feature_id") == "BW-TMA-001":
            parsed = candidate
    return {
        "passed": bool(result["returncode"] == 0 and parsed and parsed.get("status") == "passed"),
        "process": result,
        "result": parsed,
    }


def feature_status(*conditions: bool) -> str:
    return "passed" if all(conditions) else "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one B200 SM100 feature-score replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--tma-binary", type=Path, required=True)
    parser.add_argument("--tma-artifact-root", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--cute-root", type=Path, default=DEFAULT_CUTE_ROOT)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lifecycle-iterations", type=int, default=3000)
    parser.add_argument("--tma-iterations", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B200 feature benchmark requires CUDA")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.dump_root.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"
    os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.dump_root)
    replay_seed = 100_000 + args.replay_index * 10_000
    started_at = utc_now()
    hardware = hardware_manifest()

    static = load_module(
        args.cute_root / "dense_gemm_persistent.py",
        f"infraswe_b200_static_replay_{args.replay_index}",
    )
    dynamic = load_module(
        args.cute_root / "dense_gemm_persistent_dynamic.py",
        f"infraswe_b200_dynamic_replay_{args.replay_index}",
    )

    standard = base_config()
    alternate = base_config()
    alternate.update({"a_major": "m", "b_major": "n", "c_major": "m"})
    two_cta = tma2_config()
    public_shape = (512, 768, 512, 1)
    tile_tail_shape = (513, 776, 520, 1)
    layout_tail_shape = (264, 264, 129, 1)
    perf_shape = (4096, 4096, 4096, 1)
    tail_perf_shape = (4224, 4096, 4096, 1)

    tmem_cases = [
        run_correctness_case(
            static,
            label="public-aligned",
            shape=public_shape,
            config=standard,
            seed=replay_seed + 1,
        ),
        run_correctness_case(
            static,
            label="hidden-mnk-tails",
            shape=tile_tail_shape,
            config=standard,
            seed=replay_seed + 2,
        ),
        run_correctness_case(
            static,
            label="hidden-alternate-leading-dim-k-tail",
            shape=layout_tail_shape,
            config=alternate,
            seed=replay_seed + 3,
        ),
    ]
    tmem_profile = profile_cute_launch(static, shape=public_shape, config=standard)
    tmem_candidate = run_cute_benchmark(
        static,
        label="cute-tcgen05-tmem-1cta",
        shape=perf_shape,
        config=standard,
        warmups=args.warmups,
        iterations=args.iterations,
        samples=args.samples,
        seed_base=replay_seed + 100,
    )
    tmem_portable = benchmark_torch_bmm(
        shape=perf_shape,
        warmups=args.warmups,
        iterations=args.iterations,
        samples=args.samples,
        seed=replay_seed + 200,
    )
    tmem_artifacts = args.artifact_root / "BW-TMEM-001"
    snapshot_artifacts(args.dump_root, tmem_artifacts)
    tmem_native = collect_native_evidence(
        tmem_artifacts,
        ptx_patterns={
            "tcgen05_alloc": r"\btcgen05\.alloc(?:\.|\s)",
            "tcgen05_mma": r"\btcgen05\.mma(?:\.|\s)",
            "tcgen05_dealloc": r"\btcgen05\.dealloc(?:\.|\s)",
            "relinquish": r"\btcgen05\.relinquish_alloc_permit(?:\.|\s|;)",
        },
        sass_patterns={"utchmma": r"\bUTCHMMA\b"},
    )
    tmem_correct = all(case["passed"] for case in tmem_cases)
    tmem_status = feature_status(
        tmem_correct,
        tmem_candidate["passed"],
        tmem_portable["passed"],
        tmem_native["passed"],
    )

    lifecycle_control = run_cute_benchmark(
        static,
        label="tmem-lifecycle-control",
        shape=public_shape,
        config=standard,
        warmups=args.warmups,
        iterations=max(30, args.iterations),
        samples=args.samples,
        seed_base=replay_seed + 300,
    )
    memory_before = torch.cuda.mem_get_info()
    lifecycle_stress = run_cute_benchmark(
        static,
        label="tmem-lifecycle-stress",
        shape=public_shape,
        config=standard,
        warmups=args.warmups,
        iterations=args.lifecycle_iterations,
        samples=1,
        seed_base=replay_seed + 400,
    )
    torch.cuda.synchronize()
    memory_after = torch.cuda.mem_get_info()
    rejection = run_expected_rejection(
        static,
        shape=(513, 769, 511, 1),
        config=standard,
    )
    lifecycle_artifacts = args.artifact_root / "BW-TMEM-003"
    snapshot_artifacts(args.dump_root, lifecycle_artifacts)
    lifecycle_native = collect_native_evidence(
        lifecycle_artifacts,
        ptx_patterns={
            "tcgen05_alloc": r"\btcgen05\.alloc(?:\.|\s)",
            "tcgen05_dealloc": r"\btcgen05\.dealloc(?:\.|\s)",
            "relinquish": r"\btcgen05\.relinquish_alloc_permit(?:\.|\s|;)",
        },
        sass_patterns={"utchmma": r"\bUTCHMMA\b"},
    )
    lifecycle_status = feature_status(
        tmem_correct,
        lifecycle_control["passed"],
        lifecycle_stress["passed"],
        rejection["passed"],
        lifecycle_native["passed"],
    )

    clc_cases = [
        run_correctness_case(
            dynamic,
            label="public-aligned",
            shape=public_shape,
            config=standard,
            seed=replay_seed + 501,
        ),
        run_correctness_case(
            dynamic,
            label="hidden-mnk-tails",
            shape=tile_tail_shape,
            config=standard,
            seed=replay_seed + 502,
        ),
        run_correctness_case(
            dynamic,
            label="hidden-alternate-leading-dim-k-tail",
            shape=layout_tail_shape,
            config=alternate,
            seed=replay_seed + 503,
        ),
    ]
    clc_profile = profile_cute_launch(dynamic, shape=public_shape, config=standard)
    clc_measurements = []
    for shape_index, shape in enumerate((perf_shape, tail_perf_shape)):
        dynamic_measurement = run_cute_benchmark(
            dynamic,
            label="clc-dynamic-persistent",
            shape=shape,
            config=standard,
            warmups=args.warmups,
            iterations=args.iterations,
            samples=args.samples,
            seed_base=replay_seed + 600 + shape_index * 20,
        )
        static_measurement = run_cute_benchmark(
            static,
            label="static-persistent-baseline",
            shape=shape,
            config=standard,
            warmups=args.warmups,
            iterations=args.iterations,
            samples=args.samples,
            seed_base=replay_seed + 610 + shape_index * 20,
        )
        clc_measurements.append(
            {
                "shape": list(shape),
                "candidate": dynamic_measurement,
                "baseline": static_measurement,
            }
        )
    clc_artifacts = args.artifact_root / "BW-CLC-001"
    snapshot_artifacts(args.dump_root, clc_artifacts)
    clc_native = collect_native_evidence(
        clc_artifacts,
        ptx_patterns={
            "try_cancel": r"\bclusterlaunchcontrol\.try_cancel(?:\.|\s)",
            "query_cancel": r"\bclusterlaunchcontrol\.query_cancel\.is_canceled(?:\.|\s)",
        },
        sass_patterns={"get_next_work_id": r"\bUGETNEXTWORKID\b"},
    )
    clc_correct = all(case["passed"] for case in clc_cases)
    clc_perf_passed = all(
        pair["candidate"]["passed"] and pair["baseline"]["passed"] for pair in clc_measurements
    )
    clc_status = feature_status(clc_correct, clc_perf_passed, clc_native["passed"])

    tma2_cases = [
        run_correctness_case(
            static,
            label="public-cta-pair",
            shape=public_shape,
            config=two_cta,
            seed=replay_seed + 701,
        ),
        run_correctness_case(
            static,
            label="hidden-cta-pair-tail",
            shape=tile_tail_shape,
            config=two_cta,
            seed=replay_seed + 702,
        ),
    ]
    tma2_profile = profile_cute_launch(static, shape=public_shape, config=two_cta)
    tma2_candidate = run_cute_benchmark(
        static,
        label="cta-pair-2cta-tma",
        shape=perf_shape,
        config=two_cta,
        warmups=args.warmups,
        iterations=args.iterations,
        samples=args.samples,
        seed_base=replay_seed + 720,
    )
    # Preserve the 2-CTA compile products before the single-CTA baseline uses
    # the same upstream module name and can replace its dump filenames.
    tma2_artifacts = args.artifact_root / "BW-TMA-002"
    snapshot_artifacts(args.dump_root, tma2_artifacts)
    tma2_baseline = run_cute_benchmark(
        static,
        label="single-cta-tma-baseline",
        shape=perf_shape,
        config=standard,
        warmups=args.warmups,
        iterations=args.iterations,
        samples=args.samples,
        seed_base=replay_seed + 730,
    )
    tma2_native = collect_native_evidence(
        tma2_artifacts,
        ptx_patterns={
            "bulk_tensor_copy": r"\bcp\.async\.bulk\.tensor(?:\.|\s)",
            "cta_group_2": r"\.cta_group::2\b",
        },
        sass_patterns={
            "utma_2cta": r"\bUTMALDG(?:\.[A-Z0-9]+)*\.2CTA\b",
            "utchmma_2cta": r"\bUTCHMMA\.2CTA\b",
        },
    )
    tma2_correct = all(case["passed"] for case in tma2_cases)
    tma2_status = feature_status(
        tma2_correct,
        tma2_candidate["passed"],
        tma2_baseline["passed"],
        tma2_native["passed"],
    )

    tma_irregular = run_tma_irregular(args.tma_binary, iterations=args.tma_iterations)
    tma1_artifacts = args.artifact_root / "BW-TMA-001"
    copy_tma_artifacts(args.tma_artifact_root, tma1_artifacts)
    tma1_native = collect_native_evidence(
        tma1_artifacts,
        ptx_patterns={
            "gather4": r"\.tile::gather4\b",
            "scatter4": r"\.tile::scatter4\b",
        },
        sass_patterns={
            "utma_gather4": r"\bUTMALDG\.2D\.GATHER4\b",
            "utma_scatter4": r"\bUTMASTG\.2D\.SCATTER4\b",
        },
    )
    tma1_status = feature_status(tma_irregular["passed"], tma1_native["passed"])

    features = {
        "BW-TMEM-001": {
            "feature_id": "BW-TMEM-001",
            "status": tmem_status,
            "correctness": {
                "passed": tmem_correct,
                "passed_cases": sum(case["passed"] for case in tmem_cases),
                "total_cases": len(tmem_cases),
                "cases": tmem_cases,
            },
            "performance": {"candidate": tmem_candidate, "portable": tmem_portable},
            "profiler": tmem_profile,
            "native": tmem_native,
        },
        "BW-TMEM-003": {
            "feature_id": "BW-TMEM-003",
            "status": lifecycle_status,
            "correctness": {
                "passed": tmem_correct and rejection["passed"],
                "passed_cases": sum(case["passed"] for case in tmem_cases)
                + int(rejection["passed"]),
                "total_cases": len(tmem_cases) + 1,
                "inherited_fast_path_cases": tmem_cases,
                "invalid_alignment_rejection": rejection,
            },
            "liveness": {
                "passed": lifecycle_stress["passed"],
                "launch_iterations": args.lifecycle_iterations,
                "memory_free_before_bytes": memory_before[0],
                "memory_free_after_bytes": memory_after[0],
                "memory_total_bytes": memory_after[1],
            },
            "performance": {"candidate_stress": lifecycle_stress, "control": lifecycle_control},
            "native": lifecycle_native,
        },
        "BW-CLC-001": {
            "feature_id": "BW-CLC-001",
            "status": clc_status,
            "correctness": {
                "passed": clc_correct,
                "passed_cases": sum(case["passed"] for case in clc_cases),
                "total_cases": len(clc_cases),
                "cases": clc_cases,
            },
            "performance": {"makespan_pairs": clc_measurements},
            "profiler": clc_profile,
            "native": clc_native,
        },
        "BW-TMA-001": {
            "feature_id": "BW-TMA-001",
            "status": tma1_status,
            "correctness": (
                tma_irregular["result"].get("correctness", {})
                if tma_irregular["result"]
                else {"passed": False, "case_count": 0}
            ),
            "performance": (
                tma_irregular["result"].get("performance", {}) if tma_irregular["result"] else {}
            ),
            "runtime": tma_irregular,
            "native": tma1_native,
        },
        "BW-TMA-002": {
            "feature_id": "BW-TMA-002",
            "status": tma2_status,
            "correctness": {
                "passed": tma2_correct,
                "passed_cases": sum(case["passed"] for case in tma2_cases),
                "total_cases": len(tma2_cases),
                "cases": tma2_cases,
            },
            "performance": {"candidate": tma2_candidate, "baseline": tma2_baseline},
            "profiler": tma2_profile,
            "native": tma2_native,
        },
    }
    payload = {
        "schema_version": "0.2",
        "suite_id": "b200-sm100-feature-score-v0.2",
        "replay_index": args.replay_index,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": (
            "passed" if all(item["status"] == "passed" for item in features.values()) else "failed"
        ),
        "hardware": hardware,
        "protocol": {
            "evaluator_owner": "infraswe",
            "fresh_process_replays": 3,
            "warmup_iterations": args.warmups,
            "timed_iterations": args.iterations,
            "samples_per_measurement": args.samples,
            "lifecycle_launch_iterations": args.lifecycle_iterations,
            "tma_launch_iterations": args.tma_iterations,
            "watchdog": "external timeout around each fresh-process replay",
            "cuda_toolkit": "13.3",
            "target": "sm_100a",
        },
        "features": features,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "replay_index": args.replay_index}))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
