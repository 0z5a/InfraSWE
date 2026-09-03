#!/usr/bin/env python3
"""Acquire an outcome-free metadata pool for R24."""

from discover_r17_candidates import main

if __name__ == "__main__":
    raise SystemExit(main("R24", mature_eligible_multiplier=20))
