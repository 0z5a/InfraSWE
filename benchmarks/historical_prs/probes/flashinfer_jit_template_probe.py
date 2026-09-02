#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from jinja2 import Environment, StrictUndefined, meta


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_templates(templates: list[str]) -> tuple[int, list[str]]:
    environment = Environment(undefined=StrictUndefined)
    rendered_count = 0
    failures: list[str] = []
    defaults: dict[str, object] = {
        "additional_func_params": "",
        "additional_params": "",
        "additional_params_data": "",
        "additional_params_decl": "",
        "additional_params_init": "",
        "dtype_idx": "int32_t",
        "dtype_kv": "half",
        "dtype_o": "half",
        "dtype_q": "half",
        "head_dim": 128,
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_fp16_qk_reduction": "false",
        "use_logits_soft_cap": "false",
        "use_sliding_window": "false",
        "variant_decl": "template <typename ParamsT> struct ProbeAttention;",
        "variant_name": "ProbeAttention",
    }
    for index, source in enumerate(templates):
        try:
            parsed = environment.parse(source)
            variables = meta.find_undeclared_variables(parsed)
            missing = sorted(variables - defaults.keys())
            if missing:
                failures.append(f"template-{index}:unknown-variables:{','.join(missing)}")
                continue
            rendered = environment.from_string(source).render(
                **{name: defaults[name] for name in variables}
            )
            if "{{" in rendered or "{%" in rendered:
                failures.append(f"template-{index}:unresolved-jinja")
                continue
            rendered_count += 1
        except Exception as error:  # pragma: no cover - evidence path
            failures.append(f"template-{index}:{type(error).__name__}:{error}")
    return rendered_count, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    jit = options.worktree / "python" / "flashinfer" / "jit"
    batch = load_module(jit / "batch_prefill_templ.py", "blind_batch_prefill_templ")
    single = load_module(jit / "single_prefill_templ.py", "blind_single_prefill_templ")

    batch_suffixes = list(batch.batch_prefill_suffix)
    single_suffixes = list(single.single_prefill_suffix)
    template_groups = {
        "batch_prefill": list(batch.batch_prefill_templ),
        "single_prefill": list(single.single_prefill_templ),
        "customizable_single_prefill": list(single.customizable_single_prefill_templ),
    }
    suffixes_by_group = {
        "batch_prefill": batch_suffixes,
        "single_prefill": single_suffixes,
        "customizable_single_prefill": single_suffixes,
    }

    length_mismatches = {
        name: {"suffixes": len(suffixes_by_group[name]), "templates": len(templates)}
        for name, templates in template_groups.items()
        if len(suffixes_by_group[name]) != len(templates)
    }
    duplicate_suffixes = {
        "batch_prefill": len(batch_suffixes) - len(set(batch_suffixes)),
        "single_prefill": len(single_suffixes) - len(set(single_suffixes)),
    }
    required_batch_splits = {
        f"_{kind}_kernel_mask_{mask_mode}.cu"
        for kind in ("ragged", "paged")
        for mask_mode in range(3)
    }
    required_single_splits = {f"_kernel_mask_{mask_mode}.cu" for mask_mode in range(3)}
    missing_splits = sorted(
        (required_batch_splits - set(batch_suffixes))
        | (required_single_splits - set(single_suffixes))
    )

    rendered_counts: dict[str, int] = {}
    render_failures: dict[str, list[str]] = {}
    for name, templates in template_groups.items():
        rendered_counts[name], failures = render_templates(templates)
        if failures:
            render_failures[name] = failures

    passed = not any(
        [
            length_mismatches,
            any(duplicate_suffixes.values()),
            missing_splits,
            render_failures,
        ]
    )
    payload = {
        "schema_version": "0.5",
        "probe_id": "flashinfer-jit-prefill-template-split-v0.5-r1",
        "status": "pass" if passed else "fail",
        "worktree_revision": options.worktree.name,
        "batch_suffix_count": len(batch_suffixes),
        "single_suffix_count": len(single_suffixes),
        "template_counts": {name: len(value) for name, value in template_groups.items()},
        "rendered_counts": rendered_counts,
        "length_mismatches": length_mismatches,
        "duplicate_suffixes": duplicate_suffixes,
        "missing_split_suffixes": missing_splits,
        "render_failures": render_failures,
        "failure_codes": [] if passed else ["JIT_PREFILL_TEMPLATE_SPLIT_CONTRACT_FAILED"],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
