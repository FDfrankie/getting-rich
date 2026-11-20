"""Backward-compatibility alias for crypto_trading.instrument_registry."""
from __future__ import annotations

from crypto_trading.instrument_registry import (
    BNB_USDT_BINANCE_TESTNET,
    BTC_USDT_BINANCE_TESTNET,
    ETH_USDT_BINANCE_TESTNET,
    INSTRUMENT_SPECS,
    SOL_USDT_BINANCE_TESTNET,
    InstrumentSpec,
    get_instrument_spec,
)

__all__ = [
    "BNB_USDT_BINANCE_TESTNET",
    "BTC_USDT_BINANCE_TESTNET",
    "ETH_USDT_BINANCE_TESTNET",
    "INSTRUMENT_SPECS",
    "SOL_USDT_BINANCE_TESTNET",
    "InstrumentSpec",
    "get_instrument_spec",
]
