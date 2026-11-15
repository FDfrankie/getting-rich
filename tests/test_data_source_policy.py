"""
Tests for DataSource policy and DataUnavailableError enforcement.
"""
import pytest
from unittest.mock import patch
from crypto_trading.data_loader import CryptoDataLoader, DataSource, DataUnavailableError


def test_fetch_ohlcv_disallow_synthetic_raises_when_no_network():
    loader = CryptoDataLoader()
    # Mock ccxt failure
    with patch("crypto_trading.data_loader.getattr", side_effect=Exception("Network unreachable")):
        with pytest.raises(DataUnavailableError, match="allow_synthetic=False"):
            loader.fetch_ohlcv_result(limit=20, allow_synthetic=False)


def test_fetch_ohlcv_allow_synthetic_returns_synthetic_source():
    loader = CryptoDataLoader()
    with patch("crypto_trading.data_loader.getattr", side_effect=Exception("Network unreachable")):
        res = loader.fetch_ohlcv_result(limit=20, allow_synthetic=True)
        assert res.source == DataSource.SYNTHETIC
        assert res.is_synthetic is True
        assert len(res.df) == 20
        assert res.df.index.tz is not None  # UTC aware
