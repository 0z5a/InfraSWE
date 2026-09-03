"""Fail-closed import shims for exact-source vLLM R15 tests.

The remote checkout is source-only, so the compiled vLLM and FlashAttention
extension modules are absent.  The focused R15 tests do not call those
extensions, but import-time platform detection and model inspection require
the module names to exist.  This shim supplies only those names.  Any
unexpected extension operation remains absent and therefore fails closed.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types

if os.environ.get("INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM") == "1":
    import torch

    for module_name in (
        "vllm._C",
        "vllm._C_stable_libtorch",
        "vllm.vllm_flash_attn._vllm_fa2_C",
    ):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    # These capability probes run at import time.  Returning False selects
    # the ordinary implementation; it does not pretend a compiled op exists.
    torch.ops._C.cutlass_scaled_mm_supports_fp8 = lambda _capability: False
    torch.ops._C.cutlass_scaled_mm_supports_block_fp8 = lambda _capability: False

    # Newer source trees import compilation matchers that take references to
    # extension operators at module import time.  Register schema-only unary
    # placeholders so those references can be formed.  No implementation is
    # supplied: an unexpected call therefore still fails closed at dispatch.
    _schema_only_ops = (
        "cutlass_scaled_mm",
        "awq_dequantize",
        "awq_gemm",
        "allspark_w8a16_gemm",
        "convert_weight_packed_scale_zp",
        "cutlass_encode_and_reorder_int4b",
        "cutlass_encode_and_reorder_int4b_grouped",
        "cutlass_mxfp8_grouped_mm",
        "cutlass_pack_scale_fp8",
        "cutlass_w4a8_mm",
        "dynamic_per_token_scaled_fp8_quant",
        "dynamic_scaled_fp8_quant",
        "fp32_router_gemm",
        "fp8_scaled_mm_cpu",
        "fused_experts_cpu",
        "fused_add_rms_norm",
        "fused_add_rms_norm_static_fp8_quant",
        "fused_qk_norm_rope",
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
        "gptq_gemm",
        "gptq_marlin_repack",
        "hadacore_transform",
        "int4_scaled_mm_cpu",
        "int8_scaled_mm_with_quant",
        "machete_mm",
        "machete_prepack_B",
        "marlin_gemm",
        "minimax_allreduce_rms",
        "minimax_allreduce_rms_qk",
        "mxfp8_experts_quant",
        "per_token_group_fp8_quant",
        "permute_cols",
        "rms_norm",
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "rms_norm_static_fp8_quant",
        "rotary_embedding",
        "scaled_fp4_quant",
        "scaled_fp4_quant.out",
        "silu_and_mul",
        "silu_and_mul_nvfp4_quant",
        "silu_and_mul_per_block_quant",
        "silu_and_mul_quant",
        "static_scaled_fp8_quant",
        "weight_packed_linear",
        "awq_marlin_repack",
    )
    _schema_library = torch.library.Library("_C", "FRAGMENT")
    for _op_name in _schema_only_ops:
        with contextlib.suppress(RuntimeError):
            _schema_library.define(f"{_op_name}(Tensor input) -> Tensor")
