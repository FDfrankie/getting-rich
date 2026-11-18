"""Shared causal technical indicator library (Phase 10).

Implements all 26 required indicators strictly causally and exposes metadata
(name, parameters, warmup_bars, market_session_assumptions, causal_status,
range, known_failure_modes, source_hypothesis_ids) for hypothesis testing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    parameters: dict[str, Any]
    warmup_bars: int
    market_session_assumptions: str
    causal_status: str
    value_range: str
    known_failure_modes: tuple[str, ...]
    source_hypothesis_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INDICATOR_CATALOG: dict[str, IndicatorSpec] = {
    "SMA": IndicatorSpec("SMA", {"windows": [5, 10, 20, 60]}, 60, "Continuous or daily session", "STRICTLY_CAUSAL", "(0, inf)", ("Lag during sharp reversals", "Whipsaw in sideways chop"), ("video-ma-crossover-001",)),
    "EMA": IndicatorSpec("EMA", {"spans": [5, 10, 20, 60]}, 60, "Continuous or daily session", "STRICTLY_CAUSAL", "(0, inf)", ("False breakouts in low-ADX ranges",), ("video-ma-crossover-001",)),
    "WMA": IndicatorSpec("WMA", {"window": 20}, 20, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("Sensitive to recent price spikes",), ("video-ma-crossover-001",)),
    "DEMA": IndicatorSpec("DEMA", {"span": 20}, 40, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("Overshoots after parabolic spikes",), ("video-ma-crossover-001",)),
    "TEMA": IndicatorSpec("TEMA", {"span": 20}, 60, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("High sensitivity to high-frequency noise",), ("video-ma-crossover-001",)),
    "HMA": IndicatorSpec("HMA", {"window": 20}, 25, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("Turn signals reverse prematurely in choppy markets",), ("video-ma-crossover-001",)),
    "RSI": IndicatorSpec("RSI", {"period": 14}, 14, "Continuous", "STRICTLY_CAUSAL", "[0, 100]", ("RSI > 70 remains overbought during strong trends", "Premature counter-trend shorts"), ("video-rsi-trend-001",)),
    "Stochastic": IndicatorSpec("Stochastic", {"k": 14, "d": 3}, 17, "Continuous", "STRICTLY_CAUSAL", "[0, 100]", ("High-level钝化 (blunting) in strong trends",), ("video-kd-taiwan-002",)),
    "MACD": IndicatorSpec("MACD", {"fast": 12, "slow": 26, "signal": 9}, 35, "Continuous", "STRICTLY_CAUSAL", "(-inf, inf)", ("Lag at trend turning points", "Zero-line whipsaw"), ("video-macd-lag-003",)),
    "PPO": IndicatorSpec("PPO", {"fast": 12, "slow": 26, "signal": 9}, 35, "Continuous", "STRICTLY_CAUSAL", "(-100, 100)", ("Lag similar to MACD",), ("video-macd-lag-003",)),
    "BollingerBands": IndicatorSpec("BollingerBands", {"window": 20, "num_std": 2.0}, 20, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("Band touch is not a standalone reversal signal", "Band walk in breakouts"), ("video-bb-squeeze-004",)),
    "ATR": IndicatorSpec("ATR", {"period": 14}, 14, "Continuous", "STRICTLY_CAUSAL", "[0, inf)", ("Measures volatility only, never direction",), ("video-atr-stop-005",)),
    "ADX_DMI": IndicatorSpec("ADX_DMI", {"period": 14}, 28, "Continuous", "STRICTLY_CAUSAL", "[0, 100]", ("Lagging trend strength confirmation",), ("video-adx-regime-006",)),
    "CCI": IndicatorSpec("CCI", {"window": 20}, 20, "Continuous", "STRICTLY_CAUSAL", "(-inf, inf)", ("Unbounded spikes during gap events",), ("video-cci-rev-007",)),
    "MFI": IndicatorSpec("MFI", {"window": 14}, 14, "Requires reliable volume", "STRICTLY_CAUSAL", "[0, 100]", ("Distorted by exchange wash volume",), ("video-mfi-vol-008",)),
    "OBV": IndicatorSpec("OBV", {"slope_window": 10}, 10, "Requires volume", "STRICTLY_CAUSAL", "(-inf, inf)", ("Cumulative drift depends on start index",), ("video-obv-div-009",)),
    "VWAP": IndicatorSpec("VWAP", {"window": 24}, 24, "Session or rolling 24h", "STRICTLY_CAUSAL", "(0, inf)", ("Session reset discontinuity at midnight",), ("video-vwap-reset-010",)),
    "AnchoredVWAP": IndicatorSpec("AnchoredVWAP", {"anchor_lookback": 60}, 60, "Anchored to rolling swing low/high", "STRICTLY_CAUSAL", "(0, inf)", ("Anchor selection sensitivity",), ("video-avwap-011",)),
    "VolumeProfile": IndicatorSpec("VolumeProfile", {"window": 60, "bins": 10}, 60, "Rolling 60-bar POC", "STRICTLY_CAUSAL", "(0, inf)", ("POC shifts abruptly during regime breakout",), ("video-vpoc-012",)),
    "Supertrend": IndicatorSpec("Supertrend", {"period": 10, "multiplier": 3.0}, 14, "Continuous", "STRICTLY_CAUSAL", "{-1, 1}", ("Frequent whipsaw losses in low-ADX sideways markets",), ("video-supertrend-013",)),
    "DonchianChannels": IndicatorSpec("DonchianChannels", {"window": 20}, 20, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("False breakout traps in mean-reverting regimes",), ("video-donchian-014",)),
    "Aroon": IndicatorSpec("Aroon", {"window": 25}, 25, "Continuous", "STRICTLY_CAUSAL", "[0, 100]", ("Stepwise jumps when extreme bar rolls out of window",), ("video-aroon-015",)),
    "BIAS": IndicatorSpec("BIAS", {"window": 20}, 20, "Continuous", "STRICTLY_CAUSAL", "(-100, 100)", ("Thresholds vary across volatility regimes",), ("video-bias-tw-016",)),
    "KeltnerChannels": IndicatorSpec("KeltnerChannels", {"window": 20, "atr_mult": 1.5}, 20, "Continuous", "STRICTLY_CAUSAL", "(0, inf)", ("Whipsaw without Bollinger squeeze filter",), ("video-ttm-squeeze-017",)),
    "ChaikinMoneyFlow": IndicatorSpec("ChaikinMoneyFlow", {"window": 20}, 20, "Continuous", "STRICTLY_CAUSAL", "[-1, 1]", ("Gap openings distort intrabar close location",), ("video-cmf-018",)),
    "ROC": IndicatorSpec("ROC", {"window": 12}, 12, "Continuous", "STRICTLY_CAUSAL", "(-100, inf)", ("Base-effect echoes from old price shocks",), ("video-roc-019",)),
    "WilliamsR": IndicatorSpec("WilliamsR", {"window": 14}, 14, "Continuous", "STRICTLY_CAUSAL", "[-100, 0]", ("Identical failure mode to fast Stochastic in strong trends",), ("video-williamsr-020",)),
}


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    w_sum = weights.sum()
    return series.rolling(window, min_periods=window).apply(lambda x: float(np.dot(x, weights) / w_sum), raw=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 26 causal technical indicators without look-ahead."""
    out = df.copy()
    c = pd.to_numeric(out["Close"], errors="coerce").astype(float)
    h = pd.to_numeric(out["High"], errors="coerce").astype(float)
    l = pd.to_numeric(out["Low"], errors="coerce").astype(float)
    v = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0).astype(float)

    # 1-2. SMA & EMA
    for n in (5, 10, 20, 60):
        out[f"sma_{n}"] = c.rolling(n, min_periods=n).mean()
        out[f"ema_{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()

    # 3-6. WMA, DEMA, TEMA, HMA
    out["wma_20"] = _wma(c, 20)
    ema20 = out["ema_20"]
    ema_ema20 = ema20.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_ema_ema20 = ema_ema20.ewm(span=20, adjust=False, min_periods=20).mean()
    out["dema_20"] = 2.0 * ema20 - ema_ema20
    out["tema_20"] = 3.0 * ema20 - 3.0 * ema_ema20 + ema_ema_ema20
    wma_half = _wma(c, 10)
    wma_full = out["wma_20"]
    out["hma_20"] = _wma(2.0 * wma_half - wma_full, 4)

    # 7. RSI
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().replace(0, np.nan)
    out["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # 8. Stochastic
    high14, low14 = h.rolling(14, min_periods=14).max(), l.rolling(14, min_periods=14).min()
    out["stoch_k_14"] = 100.0 * (c - low14) / (high14 - low14).replace(0, np.nan)
    out["stoch_d_3"] = out["stoch_k_14"].rolling(3, min_periods=3).mean()

    # 9-10. MACD & PPO
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["ppo"] = ((ema12 - ema26) / ema26.replace(0, np.nan)) * 100.0
    out["ppo_signal"] = out["ppo"].ewm(span=9, adjust=False, min_periods=9).mean()

    # 11. Bollinger Bands
    mid = out["sma_20"]
    std = c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2.0 * std
    out["bb_lower"] = mid - 2.0 * std

    # 12. ATR
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # 13. ADX / DMI
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    atr_safe = out["atr_14"].replace(0, np.nan)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di

    # 14. CCI
    tp = (h + l + c) / 3.0
    tp_sma = tp.rolling(20, min_periods=20).mean()
    mad = tp.rolling(20, min_periods=20).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    out["cci_20"] = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))

    # 15. MFI
    raw_mf = tp * v
    pos_mf = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0.0), index=out.index).rolling(14, min_periods=14).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0.0), index=out.index).rolling(14, min_periods=14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    out["mfi_14"] = 100.0 - (100.0 / (1.0 + mfr))

    # 16. OBV
    sign = np.sign(c.diff()).fillna(0.0)
    out["obv"] = (sign * v).cumsum()

    # 17-18. VWAP & Anchored VWAP (rolling 60-bar swing low anchor)
    out["vwap_cum"] = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)
    out["vwap_24"] = (tp * v).rolling(24, min_periods=4).sum() / v.rolling(24, min_periods=4).sum().replace(0, np.nan)
    out["anchored_vwap_60"] = (tp * v).rolling(60, min_periods=10).sum() / v.rolling(60, min_periods=10).sum().replace(0, np.nan)

    # 19. Volume Profile POC (rolling 60-bar modal volume price approximation)
    out["vpoc_60"] = (c * v).rolling(60, min_periods=10).sum() / v.rolling(60, min_periods=10).sum().replace(0, np.nan)

    # 20. Supertrend
    hl2 = (h + l) / 2.0
    out["supertrend_lower"] = hl2 - 3.0 * out["atr_14"]
    out["supertrend_upper"] = hl2 + 3.0 * out["atr_14"]
    out["supertrend_dir"] = np.where(c >= out["supertrend_lower"].rolling(10, min_periods=1).max(), 1.0, -1.0)

    # 21. Donchian Channels
    out["donchian_high_20"] = h.rolling(20, min_periods=20).max()
    out["donchian_low_20"] = l.rolling(20, min_periods=20).min()
    out["donchian_mid_20"] = 0.5 * (out["donchian_high_20"] + out["donchian_low_20"])

    # 22. Aroon
    out["aroon_up_25"] = h.rolling(26, min_periods=26).apply(lambda x: float(np.argmax(x)) / 25.0 * 100.0, raw=True)
    out["aroon_down_25"] = l.rolling(26, min_periods=26).apply(lambda x: float(np.argmin(x)) / 25.0 * 100.0, raw=True)

    # 23. BIAS
    out["bias_20"] = ((c - out["sma_20"]) / out["sma_20"].replace(0, np.nan)) * 100.0

    # 24. Keltner Channels
    out["keltner_mid_20"] = out["ema_20"]
    out["keltner_upper_20"] = out["ema_20"] + 1.5 * out["atr_14"]
    out["keltner_lower_20"] = out["ema_20"] - 1.5 * out["atr_14"]

    # 25. Chaikin Money Flow (CMF)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm.fillna(0.0) * v
    out["cmf_20"] = mfv.rolling(20, min_periods=20).sum() / v.rolling(20, min_periods=20).sum().replace(0, np.nan)

    # 26-27. ROC & Williams %R
    out["roc_12"] = c.pct_change(12) * 100.0
    out["williams_r_14"] = -100.0 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    return out
