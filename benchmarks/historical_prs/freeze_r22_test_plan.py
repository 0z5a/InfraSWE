#!/usr/bin/env python3
"""Freeze R22 contracts before body or diff access."""

from freeze_r17_test_plan import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "R22",
            "--previous-iteration",
            "previous_iteration_sha256",
            "iteration_sha256",
            "prospective_rules",
        )
    )
