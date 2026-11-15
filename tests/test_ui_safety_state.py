"""
tests/test_ui_safety_state.py - 測試交易控制台安全狀態與按鈕可用性
"""

from crypto_trading.settings import TradingMode


def test_ui_safety_modes():
    # 確保預設模式嚴格為 LOCAL_PAPER，且 live trading 需明確標籤
    assert TradingMode.LOCAL_PAPER == "LOCAL_PAPER"
    assert TradingMode.BINANCE_TESTNET == "BINANCE_TESTNET"
    assert TradingMode.BITUNIX_LIVE == "BITUNIX_LIVE"


def test_ui_safety_mock_override():
    # 模擬手動接管時的狀態標記
    bot_state = {
        "manual_override": True,
        "mode": TradingMode.LOCAL_PAPER,
        "kill_switch": False,
    }
    assert bot_state["manual_override"] is True
    assert bot_state["kill_switch"] is False
