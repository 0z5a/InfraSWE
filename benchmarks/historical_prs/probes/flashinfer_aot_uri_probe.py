#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


def install_torch_build_stub() -> None:
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.COMMON_NVCC_FLAGS = []
    cpp_extension.CUDAExtension = type("CUDAExtension", (), {})
    cpp_extension.BuildExtension = type("BuildExtension", (), {})
    cpp_extension._get_cuda_arch_flags = lambda: [  # type: ignore[attr-defined]
        "-gencode=arch=compute_89,code=sm_89"
    ]
    torch = types.ModuleType("torch")
    torch.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("torch.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    utils.cpp_extension = cpp_extension  # type: ignore[attr-defined]
    torch.utils = utils  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "torch": torch,
            "torch.utils": utils,
            "torch.utils.cpp_extension": cpp_extension,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    install_torch_build_stub()
    source = options.worktree / "python" / "aot_setup.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("blind_flashinfer_aot_setup", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory(prefix="flashinfer-aot-uri-probe-") as temporary:
        module.root = Path(temporary).resolve()
        fake_source = module.root / "python" / "aot_setup.py"
        fake_source.parent.mkdir(parents=True)
        fake_source.touch()
        module.__file__ = str(fake_source)
        (module.root / "python" / "csrc" / "generated").mkdir(parents=True)
        files_prefill, files_decode, uris = module.get_instantiation_cu()

    prefill_uris = [uri for uri in uris if "prefill" in uri]
    mask_encoded = [uri for uri in prefill_uris if "_mask_" in uri]
    duplicate_uris = len(uris) - len(set(uris))
    passed = bool(prefill_uris) and not mask_encoded and duplicate_uris == 0
    payload = {
        "schema_version": "0.5",
        "probe_id": "flashinfer-aot-uri-uniqueness-v0.5-r1",
        "status": "pass" if passed else "fail",
        "worktree_revision": options.worktree.name,
        "generated_prefill_files": len(files_prefill),
        "generated_decode_files": len(files_decode),
        "aot_uri_count": len(uris),
        "prefill_uri_count": len(prefill_uris),
        "mask_encoded_prefill_uri_count": len(mask_encoded),
        "duplicate_uri_count": duplicate_uris,
        "failure_codes": [] if passed else ["AOT_PREFILL_URI_MASK_DUPLICATION_RISK"],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
