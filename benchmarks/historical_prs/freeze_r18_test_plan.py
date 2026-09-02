#!/usr/bin/env python3
"""Freeze R18 contracts before body or diff access."""

from __future__ import annotations

from freeze_r17_test_plan import main

if __name__ == "__main__":
    raise SystemExit(main("R18", "--r17-iteration", "r17_policy_iteration_sha256"))
