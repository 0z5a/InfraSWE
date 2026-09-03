#!/usr/bin/env python3
"""Freeze R25 contracts before body or diff access."""

from freeze_r17_test_plan import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "R25",
            "--previous-iteration",
            "previous_iteration_sha256",
            "iteration_sha256",
            "prospective_rules",
        )
    )
