#!/usr/bin/env python3
"""Freeze R21 contracts before body or diff access."""

from __future__ import annotations

from freeze_r17_test_plan import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "R21",
            "--r20-terminal",
            "r20_terminal_policy_sha256",
            "recommendation_sha256",
            "terminal_rules",
        )
    )
