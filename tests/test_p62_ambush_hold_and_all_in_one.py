"""P62 Regression Test: Ambush Limit Hold Discipline (Zero 1-Bar Flash-Exit) & All-in-One Verification."""
from __future__ import annotations

import numpy as np
import pandas as pd


class _MockAmbushStrategy:
    """Fires a single resting ambush Long on bar 105, then target=0 forever, with wide SL/TP to prove it is NOT flash-closed."""
    hold_until_sl_tp = True
    max_hold_bars = 50
    tp_atr_mult = 4.0
    partial_tp_atr_mult = 2.0
    sl_atr_mult = 1.6
    max_leverage_scale = 2.0
    enable_pyramiding = True
    pyramid_add_fraction = 0.5

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        out["ATR_PCT"] = 0.005
        out["signal"] = 0.0
        out["ambush_dir"] = 0
        out["ambush_limit_price"] = 0.0
        out["ambush_size"] = 0.0
        # Arm ambush limit on bar 105 that touches bar 106's low..high
        out.loc[105, "ambush_dir"] = 1
        out.loc[105, "ambush_limit_price"] = float(out.loc[106, "Close"])
        out.loc[105, "ambush_size"] = 1.0
        return out


def test_ambush_position_not_flash_closed_on_next_bar():
    from scripts.backtest_crypto_usdt_walkforward import simulate_walkforward_oos

    n = 260
    # Smooth upward trend after bar 106 so a held Long hits Partial TP + Full TP cleanly
    px = np.full(n, 100.0, dtype=np.float64)
    for i in range(106, n):
        px[i] = 100.0 + (i - 106) * 0.15
    df = pd.DataFrame({
        "Open": px,
        "High": px + 0.10,
        "Low": px - 0.10,
        "Close": px,
        "Volume": np.full(n, 100.0),
    })

    metrics = simulate_walkforward_oos(
        df,
        _MockAmbushStrategy(),
        label="MockAmbushHold",
        symbol="SOLUSDT",
        timeframe="3m",
        fee_one_way=0.0002,
        slippage_one_way=0.00003,
        warmup_bars=100,
    )
    # If 1-bar flash-exit bug existed, net_return would be ~0 or negative from fees!
    # With P62 fix, it holds cleanly to Partial TP + Full TP and earns > +1.0% net!
    assert metrics.net_return_pct > 1.0, f"Expected profitable held ambush trade, got net={metrics.net_return_pct}%"
    assert metrics.trades >= 1
    assert metrics.trades <= 3


def test_fib_strategy_has_hold_until_sl_tp_flag():
    from crypto_trading.strategies.fib_condprob_kline_strategy import FibCondProbKLineStrategy

    strat = FibCondProbKLineStrategy(timeframe="3m", maker_mode=True)
    assert getattr(strat, "hold_until_sl_tp", False) is True
    assert getattr(strat, "max_hold_bars", 0) == 80
