"""P61 Full Edition Unit, Causality, Ambush Limit, Pyramiding, Vol-Leverage & Relative Strength Tests."""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd


def _make_synthetic_ohlcv(n: int = 800, seed: int = 42, slope: float = 18.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    drift = 60000.0 + slope * t + 950.0 * np.sin(t / 28.0) + np.cumsum(rng.normal(0, 65.0, n))
    close = np.maximum(1000.0, drift)
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0, 25.0, n)
    wick_up = np.abs(rng.normal(0, 95.0, n))
    wick_dn = np.abs(rng.normal(0, 95.0, n))
    for k in range(80, n - 20, 35):
        wick_dn[k] = 420.0
        open_[k] = close[k] - 40.0
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    vol = rng.uniform(50.0, 250.0, n)
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
        "TakerBuyBaseVol": vol * rng.uniform(0.42, 0.62, n),
    })


def test_fibonacci_grid_ambush_and_kline_patterns():
    from crypto_trading.strategies.fib_condprob_kline_strategy import (
        compute_causal_fibonacci_grid,
        detect_kline_patterns,
        FibCondProbKLineStrategy,
    )

    df = _make_synthetic_ohlcv(450)
    retr, zones, fib618, fib382 = compute_causal_fibonacci_grid(
        df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), swing_window=72
    )
    assert len(retr) == len(df)
    assert np.all((retr >= 0.0) & (retr <= 1.0))
    assert "GOLDEN_POCKET_SUPPORT" in zones
    assert np.all(fib382 >= fib618)

    strat = FibCondProbKLineStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True)
    res = strat.generate_signals(df)
    assert (res["ambush_dir"] != 0).sum() > 0
    assert (res["ambush_limit_price"] > 0).sum() > 0


def test_fib_condprob_strictly_causal_zero_lookahead():
    from crypto_trading.strategies.fib_condprob_kline_strategy import FibCondProbKLineStrategy

    df = _make_synthetic_ohlcv(350)
    strat = FibCondProbKLineStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True)
    res1 = strat.generate_signals(df)

    df2 = df.copy()
    df2.loc[251:, "Close"] *= 1.5
    df2.loc[251:, "High"] *= 1.5
    df2.loc[251:, "Low"] *= 1.5
    res2 = strat.generate_signals(df2)

    assert np.allclose(res1["signal"].iloc[:251].to_numpy(), res2["signal"].iloc[:251].to_numpy())
    assert np.allclose(res1["ambush_limit_price"].iloc[:251].to_numpy(), res2["ambush_limit_price"].iloc[:251].to_numpy())
    assert np.allclose(res1["cond_win_prob"].iloc[:251].to_numpy(), res2["cond_win_prob"].iloc[:251].to_numpy())


def test_relative_strength_rotation_multiplier():
    from crypto_trading.strategies.fib_condprob_kline_strategy import compute_relative_strength_multiplier

    btc = np.linspace(100.0, 102.0, 80)  # +2%
    leader = np.linspace(100.0, 108.0, 80)  # +8%
    laggard = np.linspace(100.0, 96.0, 80)  # -4%

    assert compute_relative_strength_multiplier(leader, btc, 1, 79, lookback=48) == 1.25
    assert compute_relative_strength_multiplier(laggard, btc, 1, 79, lookback=48) == 0.70
    assert compute_relative_strength_multiplier(laggard, btc, -1, 79, lookback=48) == 1.25
    assert compute_relative_strength_multiplier(leader, btc, -1, 79, lookback=48) == 0.70


def test_high_taker_fee_blocks_low_edge_trades():
    from crypto_trading.strategies.fib_condprob_kline_strategy import FibCondProbKLineStrategy

    df = _make_synthetic_ohlcv(500)
    maker_strat = FibCondProbKLineStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True)
    absurd_fee_strat = FibCondProbKLineStrategy(timeframe="3m", fee_roundtrip=0.02, maker_mode=False)

    r_maker = maker_strat.generate_signals(df)
    r_absurd = absurd_fee_strat.generate_signals(df)
    assert (r_maker["signal"] != 0).sum() >= (r_absurd["signal"] != 0).sum()
    assert (r_absurd["signal"] != 0).sum() == 0


def test_root_docs_exist_and_cover_core_sections():
    root = pathlib.Path(__file__).resolve().parents[1]
    gemini_md = (root / "GEMINI.md").read_text(encoding="utf-8")
    readme_md = (root / "README.md").read_text(encoding="utf-8")
    for kw in [
        "Resting Maker Limit Ambush",
        "Profit-Locked Pyramiding",
        "Relative Strength Rotation",
        "CondProb_MTF_KLine_3m_Maker",
        "CONSULTANT_RESERVE",
        "worst_first",
        "BoundedSemaphore",
        "Quant_BrainMiner",
        "Quant_Watchdog",
        "crypto_quant_kb_v13.json",
        "worldquant_brain_official_kb_v103_clean_quant.json.gz",
    ]:
        assert kw in gemini_md, f"Missing keyword in GEMINI.md: {kw}"
    assert "GEMINI.md" in readme_md
