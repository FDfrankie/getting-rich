"""
Tests for closed candle policy: unclosed current candle must be excluded.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from crypto_trading.data_quality import validate_ohlcv


def test_unclosed_candle_is_rejected():
    now_utc = datetime.now(timezone.utc)
    delta = timedelta(hours=1)
    # Candle open is 10 minutes ago -> close is 50 minutes in the future
    candle_open = now_utc - timedelta(minutes=10)
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [98.0],
            "Close": [103.0],
            "Volume": [1000.0],
        },
        index=pd.DatetimeIndex([candle_open], tz="UTC"),
    )

    ok, reason = validate_ohlcv(df, timeframe="1h", now_utc=now_utc)
    assert ok is False
    assert "Unclosed candle detected" in reason
