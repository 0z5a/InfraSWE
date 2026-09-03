#!/usr/bin/env python3
"""Run candidate-owned R24 tests on exact frozen refs."""

from run_r15_upstream_tests import main

if __name__ == "__main__":
    raise SystemExit(main("R24"))
