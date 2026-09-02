#!/usr/bin/env python3
"""Audit expanded R13 judgments with the shared disposition oracle."""

from __future__ import annotations

from audit_r11_oracle import main

if __name__ == "__main__":
    raise SystemExit(main(round_label="R13", selection_name="expanded-selection-lock.json"))
