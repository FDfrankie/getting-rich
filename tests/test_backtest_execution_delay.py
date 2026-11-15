import pytest
import pandas as pd
import numpy as np
from crypto_trading.backtest_engine import BacktestEngine, FeeSchedule

def test_fee_schedule_defaults_and_metadata():
    fee_sched = FeeSchedule(
        source="binance_futures_usdc",
        maker_fee_rate=0.0,
        taker_fee_rate=0.0002,
        spread_bps=0.5,
        slippage_bps=2.0,
        funding_rate_per_bar=0.00005,
        delay_bars=1,
        tick_size=0.01,
        step_size=0.001,
        min_notional=5.0
    )
    assert fee_sched.source == "binance_futures_usdc"
    assert fee_sched.delay_bars == 1
    assert fee_sched.min_notional == 5.0
    assert fee_sched.effective_time is not None

def test_backtest_execution_delay_t_plus_1():
    # 建立測試資料：5 根 K 棒
    prices = [100.0, 110.0, 120.0, 110.0, 130.0]
    # 在第 0 根產生信號 1.0，第 2 根產生信號 0.0
    signals = [1.0, 1.0, 0.0, 0.0, 0.0]
    df = pd.DataFrame({"Close": prices, "signal": signals})

    engine = BacktestEngine(fee_schedule=FeeSchedule(delay_bars=1, taker_fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0, funding_rate_per_bar=0.0))
    res = engine.run(df, timeframe="1h", data_source="unit_test")

    # t+1 執行：
    # Bar 0: signal 1.0, executed_pos = NaN -> 0. Return = 0
    # Bar 1: signal 1.0, executed_pos = 1.0. Price 100 -> 110 (+10%). Return = +10%
    # Bar 2: signal 0.0, executed_pos = 1.0. Price 110 -> 120 (+9.09%). Return = +9.09%
    # Bar 3: signal 0.0, executed_pos = 0.0. Price 120 -> 110. Return = 0 (已出場)
    assert res["signal_delay"] == 1
    assert res["data_source"] == "unit_test"
    assert res["net_return_pct"] > 15.0

def test_cost_breakdown_maker_taker_spread_slippage_funding():
    prices = [100.0, 105.0, 110.0, 115.0]
    signals = [1.0, 1.0, 1.0, 1.0]
    df = pd.DataFrame({"Close": prices, "signal": signals})

    fee_sched = FeeSchedule(
        source="mock_exchange",
        maker_fee_rate=0.0001,
        taker_fee_rate=0.0005,
        spread_bps=2.0,
        slippage_bps=4.0,
        funding_rate_per_bar=0.0002,
        delay_bars=1
    )
    engine = BacktestEngine(initial_capital=10000.0, fee_schedule=fee_sched)
    res = engine.run(df, timeframe="1h")

    costs = res["costs"]
    assert "taker_costs" in costs
    assert "maker_costs" in costs
    assert "spread_costs" in costs
    assert "slippage_costs" in costs
    assert "funding_costs" in costs
    assert costs["taker_costs"] > 0
    assert costs["slippage_costs"] > 0
    assert costs["funding_costs"] > 0
    assert costs["total_cost"] > 0

    # 指標包含 gross/net return, Sharpe, Sortino, Calmar, turnover, exposure, hit_rate
    assert "gross_return_pct" in res
    assert "net_return_pct" in res
    assert "sharpe_ratio" in res
    assert "sortino_ratio" in res
    assert "calmar_ratio" in res
    assert "turnover" in res
    assert "exposure" in res
    assert "hit_rate_pct" in res
    assert "profit_factor" in res
    assert res["gross_return_pct"] > res["net_return_pct"]
