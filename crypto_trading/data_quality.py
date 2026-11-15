"""
Data quality gates and validation rules for OHLCV data.
Validates monotonic time, uniqueness, OHLC sanity, exclusion of unclosed candles,
and freshness latency (<= 2 periods).
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Tuple


class DataQualityError(Exception):
    """Raised when OHLCV fails strict quality validation gates."""
    pass


def parse_timeframe_delta(timeframe: str) -> pd.Timedelta:
    tf = timeframe.lower().strip()
    if 's' in tf:
        secs = int(tf.replace('s', '')) if tf != 's' else 15
        return pd.Timedelta(seconds=secs)
    elif 'm' in tf:
        mins = int(tf.replace('m', '')) if tf != 'm' else 1
        return pd.Timedelta(minutes=mins)
    elif 'h' in tf:
        hours = int(tf.replace('h', '')) if tf != 'h' else 1
        return pd.Timedelta(hours=hours)
    elif 'd' in tf:
        days = int(tf.replace('d', '')) if tf != 'd' else 1
        return pd.Timedelta(days=days)
    else:
        return pd.Timedelta(hours=1)


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str = "1h",
    max_latency_periods: int = 2,
    now_utc: datetime | None = None
) -> Tuple[bool, str]:
    """
    Validates OHLCV DataFrame against strict data quality gates:
    1. Not empty and required columns exist: Open, High, Low, Close, Volume.
    2. Datetime index is strictly monotonic increasing.
    3. No duplicate timestamps in index.
    4. OHLC logical sanity: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close, all > 0, Volume >= 0.
    5. No unclosed candles (last candle close time must be <= now_utc).
    6. Latency guard: the latest candle must not be stale (now_utc - last_candle_close <= max_latency_periods * delta).

    Returns:
        (is_valid, reason_or_ok)
    """
    if df is None or len(df) == 0:
        return False, "OHLCV dataframe is empty"

    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(set(df.columns)):
        return False, f"Missing required OHLCV columns. Found: {list(df.columns)}"

    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "OHLCV index must be a DatetimeIndex"

    # 1. Monotonic increasing
    if not df.index.is_monotonic_increasing:
        return False, "OHLCV index is not strictly monotonic increasing"

    # 2. No duplicate timestamps
    if df.index.has_duplicates:
        return False, "OHLCV index contains duplicate timestamps"

    # 3. OHLC sanity
    opens = df['Open'].to_numpy()
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    closes = df['Close'].to_numpy()
    volumes = df['Volume'].to_numpy()

    if (highs < lows).any():
        return False, "Data quality violation: High < Low found"
    if (highs < opens).any():
        return False, "Data quality violation: High < Open found"
    if (highs < closes).any():
        return False, "Data quality violation: High < Close found"
    if (lows > opens).any():
        return False, "Data quality violation: Low > Open found"
    if (lows > closes).any():
        return False, "Data quality violation: Low > Close found"
    if (opens <= 0).any() or (highs <= 0).any() or (lows <= 0).any() or (closes <= 0).any():
        return False, "Data quality violation: Non-positive prices found"
    if (volumes < 0).any():
        return False, "Data quality violation: Negative volumes found"

    # Time calculations
    delta = parse_timeframe_delta(timeframe)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    # Ensure index is UTC-aware
    idx = df.index
    if idx.tz is None:
        idx_utc = idx.tz_localize(timezone.utc)
    else:
        idx_utc = idx.tz_convert(timezone.utc)

    last_candle_open = idx_utc[-1].to_pydatetime()
    last_candle_close = last_candle_open + delta.to_pytimedelta()

    # 4. Exclude unclosed candle: candle must have fully closed
    if last_candle_close > now_utc:
        return False, f"Unclosed candle detected: candle close {last_candle_close.isoformat()} > now {now_utc.isoformat()}"

    # 5. Latency guard: <= max_latency_periods
    latency = now_utc - last_candle_close
    max_allowed_latency = delta.to_pytimedelta() * max_latency_periods
    if latency > max_allowed_latency:
        return False, f"Data is stale: latency {latency} > allowed max {max_allowed_latency} (<= {max_latency_periods} periods)"

    return True, "OK"
