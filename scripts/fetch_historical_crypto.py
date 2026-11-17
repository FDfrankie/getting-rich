#!/usr/bin/env python3
"""
Paginated Historical Crypto OHLCV & Funding Rate Crawler (scripts/fetch_historical_crypto.py).
Solves 33-day (800-bar) underfitting by fetching 6-12+ months of 1h OHLCV via CCXT pagination
and saving locally to crypto_trading/data_cache/ for high-sample ML/RL retraining.

Usage:
    python scripts/fetch_historical_crypto.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --days 180
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "crypto_trading" / "data_cache"


def fetch_paginated_ohlcv(symbol: str, timeframe: str = "1h", days: int = 180) -> pd.DataFrame:
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows: list[list] = []

    while since_ms < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Paginated Crypto Historical Data Crawler")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"[*] Fetching {args.days} days of {sym} ({args.timeframe})...")
        df = fetch_paginated_ohlcv(sym, args.timeframe, args.days)
        safe_name = sym.replace("/", "_")
        out_p = CACHE_DIR / f"{safe_name}_{args.timeframe}_{args.days}d.csv"
        df.to_csv(out_p)
        print(f"[✓] Saved {len(df)} bars to {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
