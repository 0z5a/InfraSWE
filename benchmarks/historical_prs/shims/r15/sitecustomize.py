"""Fail-closed import shims for exact-source vLLM R15 tests.

The remote checkout is source-only, so the compiled vLLM and FlashAttention
extension modules are absent.  The focused R15 tests do not call those
extensions, but import-time platform detection and model inspection require
the module names to exist.  This shim supplies only those names.  Any
unexpected extension operation remains absent and therefore fails closed.
"""

from __future__ import annotations

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
