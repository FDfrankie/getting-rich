"""P51 Unit & Integration Tests: 1m/3m MTF Candlestick Conditional Probability & Walk-Forward Gate."""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from scripts.backtest_crypto_usdt_walkforward import (
    simulate_walkforward_oos,
    write_profitability_report,
)


def _make_synthetic_ohlcv(n: int = 600, seed: int = 42, drift: float = 0.00015, vol: float = 0.004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    # Inject cyclical trend + mean-reversion waves so 3m setups & HTF regimes both appear
    wave = (vol * 0.30) * np.sin(np.linspace(0, 12 * np.pi, n))
    close = 50000.0 * np.exp(np.cumsum(rets + wave))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    spread = np.abs(rng.normal(vol * 0.55, vol * 0.20, size=n)) * close
    high = np.maximum(open_p, close) + spread * rng.uniform(0.2, 1.0, size=n)
    low = np.minimum(open_p, close) - spread * rng.uniform(0.2, 1.0, size=n)
    volume = rng.uniform(50.0, 250.0, size=n)
    taker_buy = volume * np.clip(0.5 + 0.35 * np.sign(close - open_p), 0.1, 0.9)
    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "TakerBuyBaseVol": taker_buy,
        }
    )


def test_candlestick_anatomy_and_ofi_detection() -> None:
    df = _make_synthetic_ohlcv(120, seed=7)
    # Craft an explicit bullish liquidity-sweep Pin Bar at index 90
    df.loc[90, "Open"] = 50000.0
    df.loc[90, "Close"] = 50030.0
    df.loc[90, "High"] = 50060.0
    df.loc[90, "Low"] = 49600.0
    df.loc[90, "Volume"] = 400.0
    df.loc[90, "TakerBuyBaseVol"] = 320.0

    # Craft an explicit bearish engulfing bar at index 95
    df.loc[94, "Open"] = 50000.0
    df.loc[94, "Close"] = 50100.0
    df.loc[94, "High"] = 50120.0
    df.loc[94, "Low"] = 49980.0
    df.loc[95, "Open"] = 50150.0
    df.loc[95, "Close"] = 49920.0
    df.loc[95, "High"] = 50160.0
    df.loc[95, "Low"] = 49900.0

    strat = ConditionalProbRLStrategy(timeframe="3m")
    out = strat.compute_regime_states(df)
    assert out.loc[90, "CANDLE_PATTERN"] == "BULL_SWEEP"
    assert out.loc[95, "CANDLE_PATTERN"] == "BEAR_ENGULF"
    assert -1.0 <= float(out.loc[90, "OFI"]) <= 1.0
    assert float(out.loc[90, "OFI"]) > 0.0


def test_causal_mtf_resampling_zero_lookahead() -> None:
    df_full = _make_synthetic_ohlcv(420, seed=11)
    cutoff = 300
    df_mutated = df_full.copy()
    # Radically alter all bars strictly after cutoff
    df_mutated.loc[cutoff + 1 :, "Close"] *= 2.5
    df_mutated.loc[cutoff + 1 :, "High"] *= 2.8
    df_mutated.loc[cutoff + 1 :, "Low"] *= 2.2
    df_mutated.loc[cutoff + 1 :, "Open"] *= 2.5

    strat_a = ConditionalProbRLStrategy(timeframe="3m", min_history_bars=120)
    strat_b = ConditionalProbRLStrategy(timeframe="3m", min_history_bars=120)
    out_a = strat_a.generate_signals(df_full)
    out_b = strat_b.generate_signals(df_mutated)

    # Every single causal output up to cutoff MUST be bit-for-bit identical (zero lookahead!)
    np.testing.assert_allclose(
        out_a["HTF_TREND_Z"].iloc[: cutoff + 1].to_numpy(),
        out_b["HTF_TREND_Z"].iloc[: cutoff + 1].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out_a["signal"].iloc[: cutoff + 1].to_numpy(),
        out_b["signal"].iloc[: cutoff + 1].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out_a["cond_prob_long"].iloc[: cutoff + 1].to_numpy(),
        out_b["cond_prob_long"].iloc[: cutoff + 1].to_numpy(),
        atol=1e-12,
    )


def test_triple_barrier_maker_vs_taker_cost_gate() -> None:
    # Low-vol 1m series where expected TP move is ~15 bps: clears Maker 4 bps * 2.5 = 10 bps, fails Taker 10 bps * 2.5 = 25 bps
    df = _make_synthetic_ohlcv(260, seed=19, drift=0.00002, vol=0.00075)
    strat_maker = ConditionalProbRLStrategy(timeframe="1m", fee_roundtrip=0.0004, maker_mode=True, min_history_bars=100)
    strat_taker = ConditionalProbRLStrategy(timeframe="1m", fee_roundtrip=0.0010, maker_mode=False, min_history_bars=100)

    out_maker = strat_maker.generate_signals(df)
    out_taker = strat_taker.generate_signals(df)

    maker_pass_rate = float(out_maker["cost_gate_pass"].iloc[100:].mean())
    taker_pass_rate = float(out_taker["cost_gate_pass"].iloc[100:].mean())
    assert maker_pass_rate > taker_pass_rate


def test_walkforward_backtest_engine_and_profitability_gate(tmp_path: pathlib.Path) -> None:
    df = _make_synthetic_ohlcv(550, seed=23, drift=0.0002, vol=0.0045)
    strat = ConditionalProbRLStrategy(timeframe="3m", fee_roundtrip=0.0004, maker_mode=True, min_history_bars=100)
    metrics = simulate_walkforward_oos(
        df,
        strat,
        label="CondProb_MTF_KLine_3m_Maker",
        symbol="BTCUSDT",
        timeframe="3m",
        fee_one_way=0.0002,
        warmup_bars=160,
    )
    assert metrics.bars_oos > 300
    assert metrics.fee_one_way_bps == 2.0
    jp, mp = write_profitability_report([metrics], tmp_path)
    assert jp.exists() and mp.exists()
    md_text = mp.read_text(encoding="utf-8")
    assert "CondProb_MTF_KLine_3m_Maker" in md_text
    assert "Gate Verdict" in md_text
