"""Causal technical indicators used as hypotheses, never as untested trading rules."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = (out[x].astype(float) for x in ("Close", "High", "Low", "Volume"))
    for n in (5, 10, 20, 60):
        out[f"sma_{n}"] = c.rolling(n, min_periods=n).mean()
        out[f"ema_{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26; out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    mid = c.rolling(20, min_periods=20).mean(); std = c.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"] = mid; out["bb_upper"] = mid + 2*std; out["bb_lower"] = mid - 2*std
    prev = c.shift(1); tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    sign = np.sign(c.diff()).fillna(0); out["obv"] = (sign*v).cumsum()
    typical = (h+l+c)/3; out["vwap_cum"] = (typical*v).cumsum()/v.cumsum().replace(0,np.nan)
    high14, low14 = h.rolling(14).max(), l.rolling(14).min()
    out["stoch_k_14"] = 100*(c-low14)/(high14-low14).replace(0,np.nan)
    out["stoch_d_3"] = out["stoch_k_14"].rolling(3).mean()
    return out
