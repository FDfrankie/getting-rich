#!/usr/bin/env python3
"""Download and cache a multi-stock US equity daily OHLCV panel for fast local PV Alpha pre-screening."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from worldquant_brain.local_cs_screener import LocalCrossSectionalScreener


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local US equity OHLCV panel cache for BRAIN PV pre-screening")
    parser.add_argument("--period", default="3y", help="yfinance history period (default: 3y)")
    args = parser.parse_args()

    screener = LocalCrossSectionalScreener()
    panel = screener.load_or_fetch_panel(allow_download=True, period=args.period)
    if panel is None:
        print("[ERROR] Failed to download US equity panel (check yfinance / network).")
        return 1
    close = panel["close"]
    print(f"[OK] Saved local US panel cache to {screener.cache_path} | shape={close.shape} ({close.index.min().date()} ~ {close.index.max().date()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
