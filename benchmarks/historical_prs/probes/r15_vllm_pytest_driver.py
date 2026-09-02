#!/usr/bin/env python3
"""Run focused vLLM tests with an optional minimal tests.utils boundary."""

from __future__ import annotations

import contextlib
import sys
import types


def _install_tests_utils_boundary() -> None:
    module = types.ModuleType("tests.utils")

    @contextlib.contextmanager
    def ensure_current_vllm_config():
        from vllm.config import (
            VllmConfig,
            get_current_vllm_config_or_none,
            set_current_vllm_config,
        )

        if get_current_vllm_config_or_none() is not None:
            yield
        else:
            with set_current_vllm_config(VllmConfig()):
                yield

    module.ensure_current_vllm_config = ensure_current_vllm_config
    # Focused source-only tests do not consume the unrelated assets warmed by
    # broad multimodal conftests.  Keep that optional network boundary inert.
    module.prewarm_hf_cache = lambda _assets: None
    sys.modules[module.__name__] = module


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--stub-tests-utils":
        _install_tests_utils_boundary()
        arguments = arguments[1:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    import pytest

    return pytest.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
