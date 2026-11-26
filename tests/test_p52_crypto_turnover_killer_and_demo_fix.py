"""P52 Unit Tests: Turnover-Fee Killer, HTF Directional Gate, and CCXT Demo Trading Compatibility."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from scripts.backtest_crypto_usdt_walkforward import simulate_walkforward_oos


def _make_ohlcv(n: int = 800, seed: int = 101, drift: float = 0.00015, vol: float = 0.0035) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wave = (vol * 0.35) * np.sin(np.linspace(0, 10 * np.pi, n))
    close = 60000.0 * np.exp(np.cumsum(rng.normal(drift, vol, size=n) + wave))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    spread = np.abs(rng.normal(vol * 0.55, vol * 0.20, size=n)) * close
    high = np.maximum(open_p, close) + spread * rng.uniform(0.2, 1.0, size=n)
    low = np.minimum(open_p, close) - spread * rng.uniform(0.2, 1.0, size=n)
    volume = rng.uniform(80.0, 300.0, size=n)
    taker_buy = volume * np.clip(0.5 + 0.30 * np.sign(close - open_p), 0.1, 0.9)
    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume, "TakerBuyBaseVol": taker_buy}
    )


def test_p52_low_turnover_3m_slashed_trade_count() -> None:
    df = _make_ohlcv(800, seed=101)
    strat = ConditionalProbRLStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True)
    assert strat.rebalance_cooldown == 24
    assert strat.min_holding_bars == 12
    assert strat.sl_atr_mult >= 2.0
    assert strat.tp_atr_mult >= 4.0
    m = simulate_walkforward_oos(df, strat, label="P52_3m", symbol="BTCUSDT", timeframe="3m", fee_one_way=0.0002, warmup_bars=180)
    # Must NOT churn 50+ trades in 620 bars (previously churned ~90 trades per 620 bars!)
    assert m.trades <= 30, f"Too many trades ({m.trades}); turnover killer did not engage"


def test_p52_export_pnl_uses_enable_demo_trading() -> None:
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_crypto_pnl_to_drive.py").read_text(encoding="utf-8")
    assert "enable_demo_trading" in src
