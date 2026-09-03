#!/usr/bin/env python3
"""Freeze the metadata-only R24 selection."""

from freeze_r17_selection import main

if __name__ == "__main__":
    raise SystemExit(main("R24", "previous_iteration_sha256"))
