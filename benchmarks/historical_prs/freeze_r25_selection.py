#!/usr/bin/env python3
"""Freeze the metadata-only R25 selection."""

from freeze_r17_selection import main

if __name__ == "__main__":
    raise SystemExit(main("R25", "previous_iteration_sha256"))
