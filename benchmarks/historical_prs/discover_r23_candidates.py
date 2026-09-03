#!/usr/bin/env python3
"""Acquire an outcome-free metadata pool for R23."""

from discover_r17_candidates import main

if __name__ == "__main__":
    raise SystemExit(main("R23", mature_eligible_multiplier=16))
