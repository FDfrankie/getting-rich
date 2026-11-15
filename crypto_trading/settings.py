"""
Trading settings and environment validation module.
Supports validated TradingMode: LOCAL_PAPER, BINANCE_TESTNET, BITUNIX_LIVE.
"""
from __future__ import annotations

import os
import warnings
from enum import Enum
from typing import Optional


class TradingMode(str, Enum):
    LOCAL_PAPER = "LOCAL_PAPER"
    BINANCE_TESTNET = "BINANCE_TESTNET"
    BITUNIX_LIVE = "BITUNIX_LIVE"


class Settings:
    """
    Validated application settings.
    Enforces strict risk management, credentials checking, and trading mode verification.
    """

    EXPECTED_OPERATOR_ACK_VERSION = "2026.1"

    def __init__(self, env: Optional[dict] = None):
        self._env = env if env is not None else os.environ

        raw_mode = self._env.get("TRADING_MODE", TradingMode.LOCAL_PAPER.value).strip().upper()
        try:
            self.trading_mode = TradingMode(raw_mode)
        except ValueError:
            raise ValueError(f"Invalid TRADING_MODE: '{raw_mode}'. Must be one of {[m.value for m in TradingMode]}")

        # Paper trading compatibility
        if "PAPER_TRADING" in self._env:
            warnings.warn(
                "PAPER_TRADING is deprecated; use TRADING_MODE instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Credentials (all default to empty string, no non-empty default)
        self.binance_api_key = self._env.get("BINANCE_API_KEY", "") or self._env.get("BINANCE_TESTNET_API_KEY", "")
        self.binance_api_secret = self._env.get("BINANCE_API_SECRET", "") or self._env.get("BINANCE_TESTNET_API_SECRET", "")
        self.bitunix_api_key = self._env.get("BITUNIX_API_KEY", "")
        self.bitunix_api_secret = self._env.get("BITUNIX_API_SECRET", "")

        # Risk parameters for live trading
        self.enable_live_trading = self._env.get("ENABLE_LIVE_TRADING", "")
        self.live_operator_ack_version = self._env.get("LIVE_OPERATOR_ACK_VERSION", "")
        self.allow_manual_testnet_order = self._env.get("ALLOW_MANUAL_TESTNET_ORDER", "").strip().upper()

        self.max_order_notional_usdt = self._float_or_zero(self._env.get("MAX_ORDER_NOTIONAL_USDT", "0"))
        self.max_total_exposure_usdt = self._float_or_zero(self._env.get("MAX_TOTAL_EXPOSURE_USDT", "0"))
        self.max_daily_loss_usdt = self._float_or_zero(self._env.get("MAX_DAILY_LOSS_USDT", "0"))
        self.max_leverage = self._float_or_zero(self._env.get("MAX_LEVERAGE", "0"))

        # Brain / auxiliary
        self.brain_simulation_enabled = self._env.get("BRAIN_SIMULATION_ENABLED", "false").lower() == "true"
        self.brain_auto_submit_enabled = self._env.get("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true"

        # Validate
        self._validate()

    @staticmethod
    def _float_or_zero(val: str) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.BITUNIX_LIVE

    def _validate(self) -> None:
        if self.trading_mode == TradingMode.BITUNIX_LIVE:
            if self.enable_live_trading != "I_UNDERSTAND_THE_RISK":
                raise ValueError(
                    "BITUNIX_LIVE requires ENABLE_LIVE_TRADING='I_UNDERSTAND_THE_RISK'"
                )
            if not self.bitunix_api_key or not self.bitunix_api_secret:
                raise ValueError("BITUNIX_LIVE requires non-empty BITUNIX_API_KEY and BITUNIX_API_SECRET")
            if self.live_operator_ack_version != self.EXPECTED_OPERATOR_ACK_VERSION:
                raise ValueError(
                    f"BITUNIX_LIVE requires LIVE_OPERATOR_ACK_VERSION='{self.EXPECTED_OPERATOR_ACK_VERSION}', got '{self.live_operator_ack_version}'"
                )
            if self.max_order_notional_usdt <= 0:
                raise ValueError("BITUNIX_LIVE requires MAX_ORDER_NOTIONAL_USDT > 0")
            if self.max_total_exposure_usdt <= 0:
                raise ValueError("BITUNIX_LIVE requires MAX_TOTAL_EXPOSURE_USDT > 0")
            if self.max_daily_loss_usdt <= 0:
                raise ValueError("BITUNIX_LIVE requires MAX_DAILY_LOSS_USDT > 0")
            if self.max_leverage <= 0:
                raise ValueError("BITUNIX_LIVE requires MAX_LEVERAGE > 0")


_DEFAULT_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings()
    return _DEFAULT_SETTINGS


def reset_settings(new_settings: Optional[Settings] = None) -> Settings:
    global _DEFAULT_SETTINGS
    _DEFAULT_SETTINGS = new_settings
    return get_settings()
