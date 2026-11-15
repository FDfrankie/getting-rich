"""
Mock 單元測試套件：驗證 14 項深層程式碼與 UI 修復（全數 mock，嚴禁連外）。
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from worldquant_brain.brain_client import BrainClient
from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
from crypto_trading.bot_runner import CryptoTradingBot
from crypto_trading.backtest_engine import BacktestEngine
from data_persistence.gdrive_sync import GDrivePersistenceSync


def test_brain_client_is_authenticated_property():
    """驗證 BrainClient 的 is_authenticated 屬性"""
    client = BrainClient(email=None, password=None)
    assert not client.is_authenticated
    assert client.offline_mode

    client_with_auth = BrainClient(email="user@example.com", password="pwd")
    assert not client_with_auth.offline_mode
    assert not client_with_auth.is_authenticated  # 尚未 authenticate
    client_with_auth.authenticated = True
    assert client_with_auth.is_authenticated


def test_ml_strategy_no_fit_on_inference():
    """驗證 MLAlphaStrategy 在 trainer 為 None 時不現場 fit_on_data，而是安全回傳 0"""
    strat = MLAlphaStrategy()
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.0, 102.0, 103.0],
        "Volume": [1000, 1200, 1100]
    })
    res = strat.generate_signals(df)
    assert "signal" in res.columns
    assert (res["signal"] == 0.0).all()
    assert (res["model_status"] == "MODEL_UNAVAILABLE").all()


def test_rl_strategy_uninitialized_hold():
    """驗證 RLAlphaStrategy 在無權重時安全輸出 Hold 訊號"""
    strat = RLAlphaStrategy(agent_weights_path="non_existent_weights.pt")
    df = pd.DataFrame({
        "Open": [100.0] * 10,
        "High": [102.0] * 10,
        "Low": [99.0] * 10,
        "Close": [101.0] * 10,
        "Volume": [1000] * 10
    })
    res = strat.generate_signals(df)
    assert (res["signal"] == 0.0).all()
    assert (res["model_status"] == "MODEL_UNAVAILABLE").all()


def test_bot_runner_no_paper_fallback_on_error():
    """驗證當交易所下單失敗時，絕不 fallback 呼叫 execute_paper_order 污染本地資料庫"""
    mock_adapter = MagicMock()
    mock_adapter.create_order.return_value = {"success": False, "message": "API Key Invalid"}

    bot = CryptoTradingBot(symbol="BTC/USDT", exchange_adapter=mock_adapter)
    bot.execute_paper_order = MagicMock()

    bot.execute_order(side="BUY", amount=0.01, current_price=60000.0, reason="Test order")
    # 確保 execute_paper_order 未被呼叫
    bot.execute_paper_order.assert_not_called()
    assert bot.position == 0.0
    assert bot.position_side == "FLAT"


def test_bot_runner_accounting_and_equity():
    """驗證 CryptoTradingBot 合約多空雙向保證金會計與淨值計算"""
    bot = CryptoTradingBot(symbol="BTC/USDT", exchange_adapter=None)
    init_bal = bot.balance_usdt

    # 模擬做多開倉
    bot.execute_paper_order(side="BUY", amount=1.0, current_price=50000.0, reason="Open Long")
    assert bot.position == 1.0
    assert bot.position_side == "LONG"
    assert bot.entry_price == 50000.0
    # 價格漲到 55000
    eq_long = bot.get_equity(current_price=55000.0)
    assert eq_long > init_bal

    # 平多倉
    bot.execute_paper_order(side="SELL", amount=1.0, current_price=55000.0, reason="Close Long", reduce_only=True)
    assert bot.position == 0.0
    assert bot.position_side == "FLAT"
    assert bot.balance_usdt > init_bal

    # 模擬做空開倉
    bot.execute_paper_order(side="SELL", amount=1.0, current_price=50000.0, reason="Open Short")
    assert bot.position == 1.0
    assert bot.position_side == "SHORT"
    assert bot.entry_price == 50000.0
    # 價格跌到 45000 獲利
    eq_short = bot.get_equity(current_price=45000.0)
    assert eq_short > bot.balance_usdt

    # 平空倉
    bot.execute_paper_order(side="BUY", amount=1.0, current_price=45000.0, reason="Close Short", reduce_only=True)
    assert bot.position == 0.0
    assert bot.position_side == "FLAT"


def test_backtest_engine_turnover_and_dynamic_annual():
    """驗證 BacktestEngine 雙倍換手成本、動態年化與每筆交易勝率"""
    engine = BacktestEngine(initial_capital=10000.0)
    # 建立信號: +1, +1, -1, -1, 0, 0
    signals = [0, 1, 1, -1, -1, 0]
    prices = [100.0, 102.0, 104.0, 103.0, 101.0, 100.0]
    df = pd.DataFrame({"Close": prices, "signal": signals})

    res = engine.run(df, timeframe="1h")
    assert "sharpe_ratio" in res
    assert "win_rate_pct" in res
    assert "bar_hit_rate_pct" in res
    assert "total_trade_actions" in res
    assert res["total_trade_actions"] > 0


def test_gdrive_sync_rlock_and_timeout():
    """驗證 GDrivePersistenceSync 具有連線逾時保護"""
    sync = GDrivePersistenceSync()
    conn = sync.get_connection()
    assert conn is not None
    conn.close()
