"""
Tests for DataQuality gates and validation rules.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from crypto_trading.data_quality import validate_ohlcv


def test_validate_ohlcv_healthy():
    now_utc = datetime.now(timezone.utc)
    delta = timedelta(hours=1)
    # Candle 0 closed 2 hours ago, candle 1 closed 1 hour ago
    timestamps = [now_utc - delta * 2, now_utc - delta]
    df = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [105.0, 106.0],
        "Low": [98.0, 101.0],
        "Close": [103.0, 104.0],
        "Volume": [1000.0, 1200.0]
    }, index=pd.DatetimeIndex(timestamps, tz="UTC"))

    ok, reason = validate_ohlcv(df, timeframe="1h", max_latency_periods=2, now_utc=now_utc)
    assert ok is True
    assert reason == "OK"


def test_validate_ohlcv_invalid_prices():
    now_utc = datetime.now(timezone.utc)
    delta = timedelta(hours=1)
    timestamps = [now_utc - delta * 2, now_utc - delta]
    # High < Low
    df = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [95.0, 106.0],  # Invalid High < Open and High < Low
        "Low": [98.0, 101.0],
        "Close": [103.0, 104.0],
        "Volume": [1000.0, 1200.0]
    }, index=pd.DatetimeIndex(timestamps, tz="UTC"))

    ok, reason = validate_ohlcv(df, timeframe="1h", now_utc=now_utc)
    assert ok is False
    assert "High < Low" in reason or "High < Open" in reason


def test_validate_ohlcv_stale_data():
    now_utc = datetime.now(timezone.utc)
    delta = timedelta(hours=1)
    # Latest candle opened 10 hours ago -> latency = 9 hours > 2 hours
    timestamps = [now_utc - delta * 11, now_utc - delta * 10]
    df = pd.DataFrame({
        "Open": [100.0, 102.0],
        "High": [105.0, 106.0],
        "Low": [98.0, 101.0],
        "Close": [103.0, 104.0],
        "Volume": [1000.0, 1200.0]
    }, index=pd.DatetimeIndex(timestamps, tz="UTC"))

    ok, reason = validate_ohlcv(df, timeframe="1h", max_latency_periods=2, now_utc=now_utc)
    assert ok is False
    assert "stale" in reason.lower()
