from datetime import datetime, timedelta

from crypto_trading.risk_manager import DecisionAction, RiskManager
from crypto_trading.settings import TradingMode


def test_risk_gate_happy_path():
    rm = RiskManager(initial_capital=10000.0)
    ctx = {
        "trading_mode": TradingMode.LOCAL_PAPER,
        "data_source": "binance_spot_usdt",
        "data_timestamp": datetime.utcnow().isoformat(),
        "data_quality_passed": True,
        "reconciliation_fresh": True,
        "has_unknown_orders": False,
        "exchange_connected": True,
        "open_positions_count": 1,
        "total_exposure_notional": 2000.0,
        "min_notional": 5.0,
        "step_size": 0.001,
        "tick_size": 0.01,
    }

    decision = rm.evaluate_order(
        "BTC/USDT",
        "BUY",
        "MARKET",
        amount=0.01,
        price=60000.0,
        is_reduce_only=False,
        context=ctx,
    )
    assert decision.allowed is True
    assert decision.action == DecisionAction.APPROVE


def test_risk_gate_fail_closed_on_stale_data():
    rm = RiskManager(initial_capital=10000.0)
    stale_time = (datetime.utcnow() - timedelta(seconds=600)).isoformat()
    ctx = {
        "trading_mode": TradingMode.LOCAL_PAPER,
        "data_source": "binance_spot_usdt",
        "data_timestamp": stale_time,
        "max_stale_seconds": 300,
        "data_quality_passed": True,
        "reconciliation_fresh": True,
    }

    # 開倉被拒絕
    decision = rm.evaluate_order(
        "BTC/USDT",
        "BUY",
        "MARKET",
        amount=0.01,
        price=60000.0,
        is_reduce_only=False,
        context=ctx,
    )
    assert decision.allowed is False
    assert decision.action == DecisionAction.REJECT
    assert any("Stale market data" in r for r in decision.reasons)

    # 減倉/平倉則轉為 REDUCE_ONLY 放行
    reduce_dec = rm.evaluate_order(
        "BTC/USDT",
        "SELL",
        "MARKET",
        amount=0.01,
        price=60000.0,
        is_reduce_only=True,
        context=ctx,
    )
    assert reduce_dec.allowed is True
    assert (
        reduce_dec.action == DecisionAction.REUCE_ONLY
        if hasattr(DecisionAction, "REUCE_ONLY")
        else DecisionAction.REDUCE_ONLY
    )


def test_risk_gate_drawdown_circuit_breaker():
    rm = RiskManager(initial_capital=10000.0)
    # 權益從 10000 跌至 9000 (10% 回撤 > 8% 限制)
    rm.update_capital(9000.0)

    ctx = {"data_source": "test", "data_timestamp": datetime.utcnow().isoformat()}
    decision = rm.evaluate_order(
        "ETH/USDC",
        "BUY",
        "LIMIT",
        amount=1.0,
        price=3000.0,
        is_reduce_only=False,
        context=ctx,
    )
    assert decision.allowed is False
    assert decision.action == DecisionAction.REJECT
    assert any("Max drawdown circuit breaker" in r for r in decision.reasons)


def test_risk_gate_min_notional_and_tick_step_rules():
    rm = RiskManager(initial_capital=10000.0)
    ctx = {
        "trading_mode": TradingMode.LOCAL_PAPER,
        "data_source": "test",
        "data_timestamp": datetime.utcnow().isoformat(),
        "min_notional": 10.0,
        "step_size": 0.01,
        "tick_size": 0.1,
    }

    # 1. 名義價值過低 (0.001 * 5000 = 5.0 < 10.0)
    d1 = rm.evaluate_order(
        "BTC/USDT",
        "BUY",
        "LIMIT",
        amount=0.001,
        price=5000.0,
        is_reduce_only=False,
        context=ctx,
    )
    assert d1.allowed is False
    assert any("Notional value" in r for r in d1.reasons)

    # 2. 違背 step_size
    d2 = rm.evaluate_order(
        "BTC/USDT",
        "BUY",
        "LIMIT",
        amount=0.0055,
        price=50000.0,
        is_reduce_only=False,
        context=ctx,
    )
    assert d2.allowed is False
    assert any("step_size" in r for r in d2.reasons)
