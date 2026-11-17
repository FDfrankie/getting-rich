#!/usr/bin/env python3
"""
Paginated Historical Crypto OHLCV Crawler (scripts/fetch_historical_crypto.py).
Fetches 180 days (~259,200 1m bars/symbol) of Binance Futures OHLCV via CCXT pagination,
enforces causal unclosed-bar exclusion, and saves to crypto_trading/data_cache/.

Usage:
    python scripts/fetch_historical_crypto.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1m --days 180
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from crypto_trading.dataset_manifest import build_dataset_manifest

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "crypto_trading" / "data_cache"

_TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_paginated_ohlcv(symbol: str, timeframe: str = "1m", days: int = 180) -> pd.DataFrame:
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    now_dt = datetime.now(timezone.utc)
    since_ms = int((now_dt - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(now_dt.timestamp() * 1000)
    all_rows: list[list] = []
    batch_count = 0

    while since_ms < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        batch_count += 1
        last_ts = int(batch[-1][0])
        if last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        if batch_count % 25 == 0:
            print(f"    ... {symbol} {timeframe}: {len(all_rows)} bars fetched")
        time.sleep(0.08)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()

    # Causal guard: drop unclosed current candle
    tf_sec = _TF_SECONDS.get(timeframe, 60)
    if len(df) > 0:
        last_open = df.index[-1].to_pydatetime()
        if last_open.tzinfo is None:
            last_open = last_open.replace(tzinfo=timezone.utc)
        if last_open + timedelta(seconds=tf_sec) > datetime.now(timezone.utc):
            df = df.iloc[:-1]

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Paginated Crypto Historical Data Crawler")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--days", type=int, default=180, help="Days of history to fetch (default: 180 = ~259,200 1m bars/symbol)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"[*] Fetching {args.days} days of {sym} ({args.timeframe})...")
        df = fetch_paginated_ohlcv(sym, args.timeframe, args.days)
        safe_name = sym.replace("/", "_")
        out_p = CACHE_DIR / f"{safe_name}_{args.timeframe}_{args.days}d.csv"
        df.to_csv(out_p)
        expected_rows = int(args.days * 86400 / _TF_SECONDS.get(args.timeframe, 60))
        manifest = build_dataset_manifest(
            out_p, symbol=sym, timeframe=args.timeframe, source="binance", expected_rows=expected_rows
        )
        span_info = f"{df.index[0]} -> {df.index[-1]}" if len(df) > 0 else "empty"
        print(
            f"[✓] Saved {len(df)} bars ({span_info}) to {out_p}; "
            f"missing={manifest['missing_intervals']} sha256={manifest['sha256']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
