#!/usr/bin/env python3
"""Freeze the metadata-only R21 selection."""

from __future__ import annotations

from freeze_r17_selection import main

if __name__ == "__main__":
    raise SystemExit(main("R21", "r20_terminal_policy_sha256"))
