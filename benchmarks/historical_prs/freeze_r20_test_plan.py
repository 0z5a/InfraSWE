#!/usr/bin/env python3
"""Freeze R20 contracts before body or diff access."""

from __future__ import annotations

from freeze_r17_test_plan import main

if __name__ == "__main__":
    raise SystemExit(main("R20", "--r19-iteration", "r19_policy_iteration_sha256"))
