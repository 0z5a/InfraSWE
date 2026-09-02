#!/usr/bin/env python3
"""Freeze the metadata-only R20 selection."""

from __future__ import annotations

from freeze_r17_selection import main

if __name__ == "__main__":
    raise SystemExit(main("R20", "r19_policy_iteration_sha256"))
