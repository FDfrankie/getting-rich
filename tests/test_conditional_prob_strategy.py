import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy


def test_conditional_prob_strategy_causal_and_holds():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # Trending + oscillating series so conditional patterns repeat
    drift = np.where((np.arange(n) % 40) < 25, 0.0012, -0.0008)
    rets = drift + rng.normal(0, 0.001, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": rng.uniform(100, 1000, size=n),
        },
        index=idx,
    )

    strat = ConditionalProbTechStrategy(
        horizon=3,
        entry_prob=0.52,
        exit_prob=0.46,
        min_ev=0.0,
        fee_roundtrip=0.0002,
        min_history_bars=80,
    )
    res = strat.generate_signals(df)
    assert "signal" in res.columns
    assert "cond_prob_long" in res.columns
    assert set(res["signal"].unique()).issubset({-1.0, 0.0, 1.0})
    # Should actually hold positions (> 5% of bars after warmup)
    hold_ratio = float((res["signal"].iloc[80:] != 0).mean())
    assert hold_ratio > 0.05, f"Expected active holding, got {hold_ratio}"
