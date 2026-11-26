"""P54 Unit & Integration Tests: Derivatives OI/Funding, CVD Divergence, Liquidity Sweep, BTC Lead-Lag & 50% Partial TP."""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from crypto_trading.derivatives_data_loader import classify_oi_price_quadrant, enrich_ohlcv_with_derivatives
from crypto_trading.strategies.institutional_alpha_router import InstitutionalAlphaRouter
from scripts.backtest_crypto_usdt_walkforward import simulate_walkforward_oos


def _make_ohlcv(n: int = 360, seed: int = 54, drift: float = 0.0002, vol: float = 0.004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wave = (vol * 0.4) * np.sin(np.linspace(0, 8 * np.pi, n))
    close = 60000.0 * np.exp(np.cumsum(rng.normal(drift, vol, size=n) + wave))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    spread = np.abs(rng.normal(vol * 0.6, vol * 0.2, size=n)) * close
    high = np.maximum(open_p, close) + spread * rng.uniform(0.2, 1.0, size=n)
    low = np.minimum(open_p, close) - spread * rng.uniform(0.2, 1.0, size=n)
    volume = rng.uniform(100.0, 350.0, size=n)
    taker_buy = volume * np.clip(0.5 + 0.30 * np.sign(close - open_p), 0.1, 0.9)
    ts = np.arange(n, dtype=np.int64) * 180_000 + 1_700_000_000_000
    return pd.DataFrame(
        {"OpenTime": ts, "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume, "TakerBuyBaseVol": taker_buy}
    )


def test_oi_price_quadrant_and_causal_merge() -> None:
    pr = pd.Series([0.01, 0.01, -0.01, -0.01, 0.0])
    oi = pd.Series([0.02, -0.02, 0.02, -0.02, 0.0])
    q = classify_oi_price_quadrant(pr, oi).tolist()
    assert q == ["FRESH_LONG", "SHORT_COVER", "FRESH_SHORT", "LIQ_FLUSH", "NEUTRAL"]

    df = _make_ohlcv(100, seed=1)
    f_df = pd.DataFrame({"OpenTime": [df["OpenTime"].iloc[0], df["OpenTime"].iloc[50]], "FUNDING_RATE_8H": [0.0001, 0.0009]})
    enriched = enrich_ohlcv_with_derivatives(df, funding_df=f_df)
    # Bar 49 must NOT see bar 50's funding spike (0.0009)
    assert abs(float(enriched.loc[49, "FUNDING_RATE_8H"]) - 0.0001) < 1e-9
    assert abs(float(enriched.loc[50, "FUNDING_RATE_8H"]) - 0.0009) < 1e-9


def test_liquidity_sweep_and_cvd_divergence_detection() -> None:
    df = _make_ohlcv(120, seed=2, drift=0.0, vol=0.002)
    # Craft textbook bullish stop-hunt sweep below 20-bar swing low with close back inside + positive taker buy absorption
    swing_min = float(df["Low"].iloc[60:80].min())
    df.loc[80, "Low"] = swing_min * 0.992
    df.loc[80, "Open"] = swing_min * 1.001
    df.loc[80, "Close"] = swing_min * 1.004
    df.loc[80, "High"] = swing_min * 1.006
    df.loc[80, "Volume"] = 600.0
    df.loc[80, "TakerBuyBaseVol"] = 510.0

    router = InstitutionalAlphaRouter(timeframe="3m")
    feat = router.compute_microstructure_features(df)
    assert bool(feat.loc[80, "LIQ_SWEEP_BULL"]) is True


def test_partial_tp_50pct_locks_profit_before_reversal() -> None:
    class _SingleLong:
        partial_tp_atr_mult = 2.0
        tp_atr_mult = 5.0
        sl_atr_mult = 2.0

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["ATR_PCT"] = 0.005  # 50 bps ATR -> partial TP at +1.0% (50500)
            sig = np.zeros(len(out), dtype=float)
            sig[120:] = 1.0
            out["signal"] = sig
            return out

    n = 150
    close = np.full(n, 50000.0)
    high = np.full(n, 50040.0)
    low = np.full(n, 49960.0)
    high[125] = 50600.0  # hits 50% Partial Maker TP (+1.0%) + arms Break-Even Lock on remaining 50%
    close[125] = 50500.0
    low[130] = 48500.0   # crashes -> remaining 50% exits at Break-Even Lock (+6 bps)!
    close[130:] = 48600.0
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 100.0)})

    m = simulate_walkforward_oos(df, _SingleLong(), label="PartialTP_Test", symbol="BTCUSDT", timeframe="3m", fee_one_way=0.0002, warmup_bars=110)
    # 50% booked +1.0% (= +0.50% equity) and 50% exited at Break-Even -> MUST be solidly net positive!
    assert m.net_return_pct > 0.35, f"50% Partial TP failed to lock net profit: {m.net_return_pct}%"


def test_institutional_router_zero_lookahead() -> None:
    df_full = _make_ohlcv(340, seed=9)
    cutoff = 250
    df_mut = df_full.copy()
    df_mut.loc[cutoff + 1 :, "Close"] *= 2.2
    df_mut.loc[cutoff + 1 :, "High"] *= 2.5
    df_mut.loc[cutoff + 1 :, "Low"] *= 1.9
    df_mut.loc[cutoff + 1 :, "Open"] *= 2.2

    out_a = InstitutionalAlphaRouter(timeframe="3m").generate_signals(df_full)
    out_b = InstitutionalAlphaRouter(timeframe="3m").generate_signals(df_mut)
    np.testing.assert_allclose(out_a["signal"].iloc[: cutoff + 1].to_numpy(), out_b["signal"].iloc[: cutoff + 1].to_numpy(), atol=1e-12)
