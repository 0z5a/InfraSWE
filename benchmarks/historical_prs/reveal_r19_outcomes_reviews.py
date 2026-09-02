#!/usr/bin/env python3
"""Reveal R19 outcomes and review activity only after judgment lock."""

from __future__ import annotations

from reveal_r11_outcomes_reviews import main

if __name__ == "__main__":
    raise SystemExit(main(round_label="R19"))
