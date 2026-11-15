"""
Tests for TradingMode safety checks and Settings validation.
"""


import pytest

from crypto_trading.bitunix_client import BitunixClient
from crypto_trading.settings import Settings, TradingMode


def test_default_trading_mode_is_local_paper():
    s = Settings(env={})
    assert s.trading_mode == TradingMode.LOCAL_PAPER
    assert not s.is_live
    assert s.bitunix_api_key == ""
    assert s.bitunix_api_secret == ""


def test_paper_trading_deprecation_warning():
    with pytest.deprecated_call():
        Settings(env={"PAPER_TRADING": "true"})


def test_bitunix_live_requires_all_safety_conditions():
    # 1. Missing acknowledge string
    with pytest.raises(ValueError, match="ENABLE_LIVE_TRADING='I_UNDERSTAND_THE_RISK'"):
        Settings(env={"TRADING_MODE": "BITUNIX_LIVE"})

    # 2. Missing Key / Secret
    with pytest.raises(
        ValueError, match="non-empty BITUNIX_API_KEY and BITUNIX_API_SECRET"
    ):
        Settings(
            env={
                "TRADING_MODE": "BITUNIX_LIVE",
                "ENABLE_LIVE_TRADING": "I_UNDERSTAND_THE_RISK",
            }
        )

    # 3. Missing Ack Version
    with pytest.raises(ValueError, match="LIVE_OPERATOR_ACK_VERSION"):
        Settings(
            env={
                "TRADING_MODE": "BITUNIX_LIVE",
                "ENABLE_LIVE_TRADING": "I_UNDERSTAND_THE_RISK",
                "BITUNIX_API_KEY": "dummy_key",
                "BITUNIX_API_SECRET": "dummy_secret",
                "LIVE_OPERATOR_ACK_VERSION": "wrong_version",
            }
        )

    # 4. Missing Risk limits
    with pytest.raises(ValueError, match="MAX_ORDER_NOTIONAL_USDT > 0"):
        Settings(
            env={
                "TRADING_MODE": "BITUNIX_LIVE",
                "ENABLE_LIVE_TRADING": "I_UNDERSTAND_THE_RISK",
                "BITUNIX_API_KEY": "dummy_key",
                "BITUNIX_API_SECRET": "dummy_secret",
                "LIVE_OPERATOR_ACK_VERSION": "2026.1",
                "MAX_ORDER_NOTIONAL_USDT": "0",
            }
        )

    # 5. Full valid live settings
    valid_live_env = {
        "TRADING_MODE": "BITUNIX_LIVE",
        "ENABLE_LIVE_TRADING": "I_UNDERSTAND_THE_RISK",
        "BITUNIX_API_KEY": "dummy_key",
        "BITUNIX_API_SECRET": "dummy_secret",
        "LIVE_OPERATOR_ACK_VERSION": "2026.1",
        "MAX_ORDER_NOTIONAL_USDT": "100",
        "MAX_TOTAL_EXPOSURE_USDT": "1000",
        "MAX_DAILY_LOSS_USDT": "50",
        "MAX_LEVERAGE": "3",
    }
    s = Settings(env=valid_live_env)
    assert s.trading_mode == TradingMode.BITUNIX_LIVE
    assert s.is_live


def test_bitunix_client_injected_settings():
    # When trading mode is LOCAL_PAPER, client.is_live is False even if keys are provided
    paper_settings = Settings(
        env={
            "TRADING_MODE": "LOCAL_PAPER",
            "BITUNIX_API_KEY": "dummy_key",
            "BITUNIX_API_SECRET": "dummy_secret",
        }
    )
    client = BitunixClient(settings=paper_settings)
    assert client.is_live is False
