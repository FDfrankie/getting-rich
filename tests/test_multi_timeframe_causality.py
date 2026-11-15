"""
Tests for multi-timeframe causality and operational restrictions (15s status).
"""
import pytest
from crypto_trading.timeframe_policy import TimeframePolicy, TimeframeStatus


def test_15s_is_experimental_research_only():
    status = TimeframePolicy.get_status("15s")
    assert status == TimeframeStatus.EXPERIMENTAL_RESEARCH_ONLY

    # 1h is production ready
    assert TimeframePolicy.get_status("1h") == TimeframeStatus.PRODUCTION_READY


def test_15s_disallows_live_trading_without_aggtrade():
    can_trade, reason = TimeframePolicy.can_live_trade("15s", has_aggtrade_feed=False)
    assert can_trade is False
    assert "EXPERIMENTAL_RESEARCH_ONLY" in reason
    assert "aggTrade" in reason

    # With aggTrade feed, it is permitted
    can_trade_feed, _ = TimeframePolicy.can_live_trade("15s", has_aggtrade_feed=True)
    assert can_trade_feed is True

    # 1h does not require special aggtrade feed check
    can_trade_1h, _ = TimeframePolicy.can_live_trade("1h", has_aggtrade_feed=False)
    assert can_trade_1h is True
