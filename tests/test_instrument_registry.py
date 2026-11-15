"""
Tests for Instrument and InstrumentRegistry.
"""
import pytest
from crypto_trading.instrument_registry import Instrument, InstrumentRegistry, get_instrument_registry


def test_instrument_canonical_id():
    inst = Instrument("BTC/USDT", "BTC", "USDT", "USDT", venue="binance")
    assert inst.canonical_id == "BINANCE:BTC/USDT:USDT"


def test_registry_lookup_and_resolution():
    reg = InstrumentRegistry()
    btc = reg.get("BTC/USDT")
    assert btc is not None
    assert btc.base_asset == "BTC"
    assert btc.quote_asset == "USDT"

    # Base asset direct resolution
    assert reg.resolve_base_asset("BTC/USDT") == "BTC"
    assert reg.resolve_base_asset("BTCUSDT") == "BTC"
    assert reg.resolve_base_asset("ETH/USDC") == "ETH"

    # Custom registration
    custom = Instrument("DOGE/USDT", "DOGE", "USDT", "USDT")
    reg.register(custom)
    assert reg.has_instrument("DOGE/USDT")
    assert reg.resolve_base_asset("DOGE/USDT") == "DOGE"
